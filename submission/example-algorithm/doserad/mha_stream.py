"""Bounded, ordered streaming writer for 4D MetaImage stacks."""

from __future__ import annotations

import os
import queue
import struct
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# CompressedDataSize is only known once the last volume has been deflated, but
# the header is written first and MetaImage puts the payload immediately after
# it. Reserving a fixed-width, zero-padded field lets the real size be patched
# in afterwards without shifting a single payload byte. MetaIO parses the value
# with a plain integer read, so leading zeros are immaterial.
_SIZE_FIELD_WIDTH = 20
_ZLIB_HEADER_FASTEST = b"\x78\x01"
_ADLER32_BASE = 65521


@dataclass
class _QueuedROI:
    dose: np.ndarray
    roi_box: tuple[int, ...]
    volume_index: int
    release: object | None = None


def _adler32_combine(adler1: int, adler2: int, len2: int) -> int:
    """Return Adler-32 for ``data1 + data2`` from the component checksums."""
    if len2 < 0:
        raise ValueError("len2 must be non-negative")
    sum1_1 = int(adler1) & 0xFFFF
    sum2_1 = (int(adler1) >> 16) & 0xFFFF
    sum1_2 = int(adler2) & 0xFFFF
    sum2_2 = (int(adler2) >> 16) & 0xFFFF
    sum1 = (sum1_1 + sum1_2 - 1) % _ADLER32_BASE
    sum2 = (sum2_1 + sum2_2 + (int(len2) % _ADLER32_BASE) * (sum1_1 - 1)) % _ADLER32_BASE
    return int(sum1 | (sum2 << 16))


def _compression_module(name: str):
    """Resolve a zlib-compatible compressor, preferring Intel ISA-L."""
    backend = str(name).strip().lower()
    if backend not in {"auto", "isal", "zlib"}:
        raise ValueError("compressor_backend must be auto, isal, or zlib")
    if backend in {"auto", "isal"}:
        try:
            from isal import isal_zlib

            return "isal", isal_zlib
        except ImportError:
            if backend == "isal":
                raise RuntimeError(
                    "compressor_backend='isal' requires the 'isal' package"
                ) from None
    return "zlib", zlib


def _numbers(values: Sequence[float | int]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _direction_4d(direction_3d: Sequence[float]) -> tuple[float, ...]:
    direction = np.asarray(direction_3d, dtype=np.float64).reshape(3, 3)
    out = np.eye(4, dtype=np.float64)
    # MetaImage serializes TransformMatrix in the transpose convention relative
    # to SimpleITK's row-major GetDirection tuple.
    out[:3, :3] = direction.T
    return tuple(float(value) for value in out.reshape(-1))


def mha_header(
    *,
    size_xyz: Sequence[int],
    stack_size: int,
    spacing_xyz: Sequence[float],
    origin_xyz: Sequence[float],
    direction_xyz: Sequence[float],
    compressed: bool = False,
    element_dtype: np.dtype | type = np.float32,
) -> bytes:
    """Return a SimpleITK-compatible float32 or float64 4D MHA header.

    When ``compressed`` is set, CompressedDataSize is emitted as a zero-padded
    placeholder for :meth:`StreamingMHAWriter._patch_compressed_size` to fill in
    once the deflate stream is closed.
    """
    if len(size_xyz) != 3 or len(spacing_xyz) != 3 or len(origin_xyz) != 3:
        raise ValueError("size, spacing and origin must have three components")
    if stack_size <= 0:
        raise ValueError("stack_size must be positive")
    dtype = np.dtype(element_dtype)
    element_type = {
        np.dtype(np.float32): "MET_FLOAT",
        np.dtype(np.float64): "MET_DOUBLE",
    }.get(dtype)
    if element_type is None:
        raise ValueError(f"unsupported MHA element dtype {dtype}")
    compression_lines = (
        ("CompressedData = True", f"CompressedDataSize = {0:0{_SIZE_FIELD_WIDTH}d}")
        if compressed
        else ("CompressedData = False",)
    )
    lines = (
        "ObjectType = Image",
        "NDims = 4",
        "BinaryData = True",
        f"BinaryDataByteOrderMSB = {'True' if sys.byteorder == 'big' else 'False'}",
        *compression_lines,
        f"TransformMatrix = {_numbers(_direction_4d(direction_xyz))}",
        f"Offset = {_numbers((*origin_xyz, 0.0))}",
        "CenterOfRotation = 0 0 0 0",
        f"ElementSpacing = {_numbers((*spacing_xyz, 1.0))}",
        "DimSize = "
        + " ".join(str(int(value)) for value in (*size_xyz, stack_size)),
        "AnatomicalOrientation = ????",
        f"ElementType = {element_type}",
        "ElementDataFile = LOCAL",
    )
    return ("\n".join(lines) + "\n").encode("ascii")


class StreamingMHAWriter:
    """Write CP volumes in a background thread without retaining the 4D stack.

    With ``sparse=True``, :meth:`submit_roi` writes only the CT slabs spanning
    each populated ROI. Regions outside those slabs remain filesystem holes
    which read back as zeros, while the MHA retains its full logical dimensions.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        size_xyz: Sequence[int],
        stack_size: int,
        spacing_xyz: Sequence[float],
        origin_xyz: Sequence[float],
        direction_xyz: Sequence[float],
        queue_depth: int = 4,
        sparse: bool = False,
        compress_level: int = 0,
        element_dtype: np.dtype | type = np.float32,
        compressor_backend: str = "auto",
        compression_workers: int = 1,
        compression_slab_bytes: int = 32 * 1024 * 1024,
        roi_aware_compression: bool = False,
    ) -> None:
        # Sparse writing places each ROI at its own file offset and leaves the
        # gaps as filesystem holes; a deflate stream has no addressable offsets,
        # so the two are mutually exclusive by construction.
        if sparse and compress_level:
            raise ValueError(
                "sparse writing cannot be combined with compression: sparse "
                "output is positioned by offset, a deflate stream is sequential"
            )
        self.path = Path(path)
        self.tmp_path = self.path.with_name(self.path.name + ".tmp")
        self.size_xyz = tuple(int(value) for value in size_xyz)
        self.shape_zyx = tuple(reversed(self.size_xyz))
        self.stack_size = int(stack_size)
        self.sparse = bool(sparse)
        self.compress_level = int(compress_level)
        self.compression_workers = int(compression_workers)
        if self.compression_workers < 1:
            raise ValueError("compression_workers must be positive")
        self.compression_slab_bytes = int(compression_slab_bytes)
        self.roi_aware_compression = bool(roi_aware_compression)
        if self.compression_slab_bytes < 1:
            raise ValueError("compression_slab_bytes must be positive")
        self.compressor_backend, self._compression = _compression_module(
            compressor_backend
        )
        if self.compressor_backend == "isal" and self.compress_level > 3:
            raise ValueError(
                "ISA-L compression levels are 0 through 3; use level 1 for "
                "the production fast path"
            )
        self.element_dtype = np.dtype(element_dtype)
        if self.element_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(
                f"element_dtype must be float32 or float64, got {self.element_dtype}"
            )
        self.volume_bytes = int(np.prod(self.shape_zyx)) * self.element_dtype.itemsize
        self.header = mha_header(
            size_xyz=self.size_xyz,
            stack_size=self.stack_size,
            spacing_xyz=spacing_xyz,
            origin_xyz=origin_xyz,
            direction_xyz=direction_xyz,
            compressed=bool(self.compress_level),
            element_dtype=self.element_dtype,
        )
        self._queue: queue.Queue[
            np.ndarray | _QueuedROI | None
        ] = queue.Queue(
            maxsize=max(1, int(queue_depth))
        )
        self._thread: threading.Thread | None = None
        self._exception: BaseException | None = None
        self._submitted = 0
        self._zero_slab_cache: dict[tuple[int, bool], tuple[bytes, int, int]] = {}
        self._zero_slab_lock = threading.Lock()

    def __enter__(self) -> StreamingMHAWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="doserad-mha-writer", daemon=True
        )
        self._thread.start()
        return self

    def submit(self, dose_zyx: np.ndarray) -> None:
        if self.sparse:
            raise RuntimeError("use submit_roi() with a sparse MHA writer")
        self._check_submit_ready()
        dose = np.asarray(dose_zyx)
        if dose.shape != self.shape_zyx:
            raise ValueError(f"dose shape {dose.shape} != {self.shape_zyx}")
        # Preserve the producer's dtype here. The writer thread performs any
        # conversion to the element dtype, so inference overlaps that
        # conversion and compression rather than paying for them inline.
        if not dose.flags.c_contiguous:
            dose = np.ascontiguousarray(dose)
        self._enqueue(dose)
        self._submitted += 1

    def submit_roi(
        self,
        dose_roi_zyx: np.ndarray,
        roi_box_xyz: Sequence[int],
    ) -> None:
        """Queue one ROI, copying reusable input storage.

        Sparse output writes it by file offset. Sequential/compressed output
        expands it into a reusable final-dtype zero buffer in the worker, so
        inference never materialises or thresholds a full FP32 CT volume.
        """
        self._check_submit_ready()
        roi_box = tuple(int(value) for value in roi_box_xyz)
        if len(roi_box) != 6:
            raise ValueError("roi_box_xyz must contain xs, xe, ys, ye, zs, ze")
        xs, xe, ys, ye, zs, ze = roi_box
        x_size, y_size, z_size = self.size_xyz
        if not (
            0 <= xs < xe <= x_size
            and 0 <= ys < ye <= y_size
            and 0 <= zs < ze <= z_size
        ):
            raise ValueError(
                f"ROI box {roi_box} is outside CT size {self.size_xyz}"
            )
        expected_shape = (ze - zs, ye - ys, xe - xs)
        dose = np.asarray(dose_roi_zyx)
        if dose.shape != expected_shape:
            raise ValueError(
                f"dose ROI shape {dose.shape} != {expected_shape}"
            )
        # The predictor exposes a view of one of two reusable pinned buffers.
        # Own the queued ROI before advancing that iterator. Keep the producer's
        # dtype for compressed output: any conversion to the element dtype
        # belongs in the worker, alongside compression.
        #
        # Two producers hand over ownership by different protocols: the photon
        # predictor attaches a release_to_pool callback, while the proton one
        # yields a pinned view flagged _doserad_writer_owned whose slot is
        # released by the generator's own lease. Honour both, so neither path
        # falls back to copying every ROI.
        release = getattr(dose_roi_zyx, "release_to_pool", None)
        writer_owned = release is not None or bool(
            getattr(dose_roi_zyx, "_doserad_writer_owned", False)
        )
        if not writer_owned or self.sparse:
            dose = np.array(
                dose,
                dtype=self.element_dtype if self.sparse else None,
                order="C",
                copy=True,
            )
            release = None
        self._enqueue(_QueuedROI(dose, roi_box, self._submitted, release))
        self._submitted += 1

    def _check_submit_ready(self) -> None:
        if self._thread is None:
            raise RuntimeError("StreamingMHAWriter must be used as a context manager")
        if self._exception is not None:
            raise RuntimeError("MHA writer thread failed") from self._exception
        if self._submitted >= self.stack_size:
            raise ValueError("more volumes submitted than declared stack_size")

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._thread is not None:
            if self._thread.is_alive():
                self._enqueue(None)
            self._thread.join()
        if exc is None and self._exception is None:
            if self._submitted != self.stack_size:
                raise ValueError(
                    f"submitted {self._submitted} volumes, expected {self.stack_size}"
                )
            os.replace(self.tmp_path, self.path)
        else:
            try:
                self.tmp_path.unlink()
            except FileNotFoundError:
                pass
        if exc is None and self._exception is not None:
            raise RuntimeError("MHA writer thread failed") from self._exception
        return False

    def _enqueue(
        self,
        item: np.ndarray | _QueuedROI | None,
    ) -> None:
        while True:
            if self._exception is not None:
                raise RuntimeError("MHA writer thread failed") from self._exception
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if self._thread is None or not self._thread.is_alive():
                    raise RuntimeError("MHA writer thread stopped unexpectedly")

    def _run(self) -> None:
        try:
            if self.compress_level and self.compression_workers > 1:
                self._run_parallel_compressed()
                return
            with open(self.tmp_path, "wb", buffering=0) as output:
                output.write(self.header)
                if self.sparse:
                    output.truncate(
                        len(self.header) + self.stack_size * self.volume_bytes
                    )
                compressor = (
                    self._compression.compressobj(self.compress_level)
                    if self.compress_level
                    else None
                )
                conversion_buffer: np.ndarray | None = None
                compressed_bytes = 0
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    queued_release = (
                        item.release if isinstance(item, _QueuedROI) else None
                    )
                    if isinstance(item, _QueuedROI):
                        dose, roi_box, volume_index = (
                            item.dose,
                            item.roi_box,
                            item.volume_index,
                        )
                        if self.sparse:
                            self._write_sparse_roi(
                                output.fileno(), dose, roi_box, volume_index
                            )
                            if item.release is not None:
                                item.release()
                            continue
                        if conversion_buffer is None:
                            conversion_buffer = np.empty(
                                self.shape_zyx, dtype=self.element_dtype
                            )
                        conversion_buffer.fill(0.0)
                        xs, xe, ys, ye, zs, ze = roi_box
                        conversion_buffer[zs:ze, ys:ye, xs:xe] = dose
                        item = conversion_buffer

                    if compressor is not None:
                        if item.dtype != self.element_dtype:
                            if conversion_buffer is None:
                                conversion_buffer = np.empty(
                                    self.shape_zyx, dtype=self.element_dtype
                                )
                            np.copyto(
                                conversion_buffer, item, casting="unsafe"
                            )
                            item = conversion_buffer
                        # Deflated incrementally, so the stack is never held in
                        # memory in either raw or compressed form.
                        chunk = compressor.compress(memoryview(item).cast("B"))
                        compressed_bytes += len(chunk)
                        output.write(chunk)
                    else:
                        if item.dtype != self.element_dtype:
                            if conversion_buffer is None:
                                conversion_buffer = np.empty(
                                    self.shape_zyx, dtype=self.element_dtype
                                )
                            np.copyto(
                                conversion_buffer, item, casting="unsafe"
                            )
                            item = conversion_buffer
                        output.write(memoryview(item).cast("B"))
                    if queued_release is not None:
                        queued_release()

                if compressor is not None:
                    chunk = compressor.flush()
                    compressed_bytes += len(chunk)
                    output.write(chunk)
                    self._patch_compressed_size(output, compressed_bytes)
        except BaseException as exc:  # propagate thread errors to the producer
            self._exception = exc

    def _run_parallel_compressed(self) -> None:
        """Write one zlib stream from independently compressed ordered slabs.

        Every worker starts a fresh raw-DEFLATE stream. Intermediate slabs end
        with ``Z_SYNC_FLUSH`` (a non-final, byte-aligned block); the last slab
        ends with ``Z_FINISH``. Concatenating those blocks under one zlib header
        and a combined Adler-32 trailer produces a conventional single stream,
        while allowing slab construction, element-dtype conversion, and Deflate
        to use several CPU cores.
        """
        z_size, y_size, x_size = self.shape_zyx
        plane_bytes = y_size * x_size * self.element_dtype.itemsize
        slab_planes = max(1, self.compression_slab_bytes // plane_bytes)
        compressed_bytes = 0
        combined_adler = 1
        next_volume_index = 0

        with open(self.tmp_path, "wb", buffering=0) as output:
            output.write(self.header)
            output.write(_ZLIB_HEADER_FASTEST)
            compressed_bytes += len(_ZLIB_HEADER_FASTEST)

            with ThreadPoolExecutor(
                max_workers=self.compression_workers,
                thread_name_prefix="doserad-mha-deflate",
            ) as pool:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    volume_index = (
                        item.volume_index
                        if isinstance(item, _QueuedROI)
                        else next_volume_index
                    )
                    if volume_index != next_volume_index:
                        raise ValueError(
                            f"parallel compressor received volume {volume_index}, "
                            f"expected {next_volume_index}"
                        )
                    boundaries = set(range(0, z_size, slab_planes))
                    boundaries.add(z_size)
                    if self.roi_aware_compression and isinstance(item, _QueuedROI):
                        boundaries.add(item.roi_box[4])
                        boundaries.add(item.roi_box[5])
                    ordered = sorted(boundaries)
                    futures = []
                    for z_start, z_stop in zip(ordered, ordered[1:]):
                        final = (
                            volume_index == self.stack_size - 1
                            and z_stop == z_size
                        )
                        futures.append(pool.submit(
                            self._compress_slab,
                            item,
                            z_start,
                            z_stop,
                            final,
                        ))
                    for future in futures:
                        chunk, slab_adler, raw_bytes = future.result()
                        output.write(chunk)
                        compressed_bytes += len(chunk)
                        combined_adler = _adler32_combine(
                            combined_adler, slab_adler, raw_bytes
                        )
                    if isinstance(item, _QueuedROI) and item.release is not None:
                        item.release()
                    next_volume_index = volume_index + 1

            if next_volume_index != self.stack_size:
                raise ValueError(
                    f"parallel compressor received {next_volume_index} "
                    f"volumes, expected {self.stack_size}"
                )
            trailer = struct.pack(">I", combined_adler & 0xFFFFFFFF)
            output.write(trailer)
            compressed_bytes += len(trailer)
            self._patch_compressed_size(output, compressed_bytes)

    def _compress_slab(
        self,
        item: np.ndarray | _QueuedROI,
        z_start: int,
        z_stop: int,
        final: bool,
    ) -> tuple[bytes, int, int]:
        """Construct one dense final-dtype slab and return raw Deflate bytes."""
        if isinstance(item, _QueuedROI):
            dose, roi_box = item.dose, item.roi_box
            xs, xe, ys, ye, zs, ze = roi_box
            if self.roi_aware_compression and (z_stop <= zs or z_start >= ze):
                return self._compress_zero_slab(z_stop - z_start, final)
            slab = np.zeros(
                (z_stop - z_start, self.shape_zyx[1], self.shape_zyx[2]),
                dtype=self.element_dtype,
            )
            overlap_start = max(z_start, zs)
            overlap_stop = min(z_stop, ze)
            if overlap_start < overlap_stop:
                slab[
                    overlap_start - z_start : overlap_stop - z_start,
                    ys:ye,
                    xs:xe,
                ] = dose[overlap_start - zs : overlap_stop - zs]
        else:
            source = item[z_start:z_stop]
            slab = np.asarray(source, dtype=self.element_dtype, order="C")
            if not slab.flags.c_contiguous:
                slab = np.ascontiguousarray(slab)

        raw = memoryview(slab).cast("B")
        slab_adler = int(self._compression.adler32(raw)) & 0xFFFFFFFF
        compressor = self._compression.compressobj(
            self.compress_level,
            self._compression.DEFLATED,
            -self._compression.MAX_WBITS,
        )
        chunk = compressor.compress(raw)
        chunk += compressor.flush(
            self._compression.Z_FINISH
            if final
            else self._compression.Z_SYNC_FLUSH
        )
        return chunk, slab_adler, raw.nbytes

    def _compress_zero_slab(
        self, plane_count: int, final: bool
    ) -> tuple[bytes, int, int]:
        """Reuse Deflate output for identical all-zero CT-plane runs."""
        key = (int(plane_count), bool(final))
        with self._zero_slab_lock:
            cached = self._zero_slab_cache.get(key)
        if cached is not None:
            return cached
        raw_bytes = int(plane_count) * self.shape_zyx[1] * self.shape_zyx[2] * self.element_dtype.itemsize
        zeros = np.zeros(raw_bytes, dtype=np.uint8)
        raw = memoryview(zeros)
        checksum = int(self._compression.adler32(raw)) & 0xFFFFFFFF
        compressor = self._compression.compressobj(
            self.compress_level,
            self._compression.DEFLATED,
            -self._compression.MAX_WBITS,
        )
        chunk = compressor.compress(raw)
        chunk += compressor.flush(
            self._compression.Z_FINISH
            if final
            else self._compression.Z_SYNC_FLUSH
        )
        result = (chunk, checksum, raw_bytes)
        with self._zero_slab_lock:
            self._zero_slab_cache.setdefault(key, result)
            return self._zero_slab_cache[key]

    def _patch_compressed_size(self, output, compressed_bytes: int) -> None:
        """Overwrite the header's CompressedDataSize placeholder in place."""
        field = f"{compressed_bytes:0{_SIZE_FIELD_WIDTH}d}".encode("ascii")
        if len(field) != _SIZE_FIELD_WIDTH:
            raise ValueError(
                f"compressed size {compressed_bytes} exceeds the "
                f"{_SIZE_FIELD_WIDTH}-digit header field"
            )
        marker = b"CompressedDataSize = "
        offset = self.header.find(marker)
        if offset < 0:
            raise RuntimeError("compressed header is missing CompressedDataSize")
        output.seek(offset + len(marker))
        output.write(field)

    def _write_sparse_roi(
        self,
        file_descriptor: int,
        dose_roi_zyx: np.ndarray,
        roi_box: tuple[int, ...],
        volume_index: int,
    ) -> None:
        """Write one full-X slab per populated Z plane.

        Including the small zero X margins reduces millions of tiny row writes
        to roughly one large pwrite per ROI Z plane, while Y/Z margins and all
        entirely empty planes remain sparse holes.
        """
        xs, xe, ys, ye, zs, ze = roi_box
        x_size, y_size, _ = self.size_xyz
        y_span = ye - ys
        slab = np.zeros((y_span, x_size), dtype=self.element_dtype)
        payload_start = len(self.header) + volume_index * self.volume_bytes
        row_bytes = x_size * self.element_dtype.itemsize
        plane_bytes = y_size * row_bytes
        for local_z, z_index in enumerate(range(zs, ze)):
            slab.fill(0.0)
            slab[:, xs:xe] = dose_roi_zyx[local_z]
            offset = payload_start + z_index * plane_bytes + ys * row_bytes
            view = memoryview(slab).cast("B")
            written = os.pwrite(file_descriptor, view, offset)
            if written != view.nbytes:
                raise OSError(
                    f"short sparse MHA write: {written} != {view.nbytes}"
                )
