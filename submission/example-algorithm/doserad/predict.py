"""Segment dose prediction: CT + beam metadata -> dose in the CT grid.

Replaces the inference-relevant half of ``train/utils.OtfGpuBatchMaterializer``
(input materialisation) and of its ``loss()`` (the packed-coefficient sampler
that turns model output into CT-space dose). Everything training-only --
datasets, catalogs, losses, optimisers, metrics, checkpointing, augmentation --
is gone.

This class holds the shared machinery -- CT loading, BEV input construction,
back-projection and the packed-coefficient sampler. The architecture itself is
supplied by :class:`doserad.mamba_predict.PackedMambaDosePredictor`, the only
model this repository serves.

The checkpoint it targets was trained with ``--ct-space-loss
--ct-space-bev-pixel-shuffle --ct-space-direct-spline-coefficients
--ct-space-direct-packed-coefficients``, so the model emits packed bicubic
spline coefficients, not a dose map: dose
only exists after :func:`bicubic_iir_sample_affine_from_packed_coeff_triton`
evaluates them on the CT ROI.
"""

from __future__ import annotations

import contextlib
import json
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cupy as cp
import numpy as np
import torch
import torch.nn as nn

from . import bev_build, geometry
from .bev_grid import BevGridConfig, load_bev_grid_config
from .ct_sample import (
    bicubic_iir_sample_affine_from_packed_coeff_triton,
    bicubic_iir_sample_affine_from_packed_coeff_zyx_triton,
)
from .fast_preprocess import (
    build_plan_cache,
    preprocess_batch,
    preprocess_packed_encoder_batch,
    warmup_fast_preprocess,
)

MODEL_CAPACITY_SCALE = 1.0
UPSCALE_FACTOR = 2


# -- proton: writer-owned pinned output slots --------------------------------
#
# The proton predictors hand each ROI to the MHA writer as a lease on a
# pinned host buffer, so the device-to-host copy overlaps the next beamlet
# and the writer releases the slot when it has consumed the bytes.

@dataclass
class _PinnedOutputSlot:
    """A reusable double-buffered D2H workspace leased by one generator."""

    buffers: tuple[torch.Tensor, torch.Tensor]
    capacity: int
    in_use: bool = False

@dataclass
class _PinnedOutputLease:
    """Release a pool slot when a prediction generator ends or is closed."""

    slot: _PinnedOutputSlot | None

    def release(self) -> None:
        if self.slot is not None:
            self.slot.in_use = False
            self.slot = None

    def __del__(self) -> None:
        self.release()

class _WriterOwnedPinnedArray(np.ndarray):
    """Pinned NumPy view whose storage is transferred to the MHA writer."""

    _doserad_writer_owned = True


class BorrowedDoseROI(np.ndarray):
    """Pinned NumPy view returned to its predictor pool by the MHA writer."""

    def __new__(cls, array: np.ndarray, release):
        obj = np.asarray(array).view(cls)
        obj._release_to_pool = release
        obj._released = False
        return obj

    def release_to_pool(self) -> None:
        if not self._released:
            self._released = True
            self._release_to_pool()


def load_stats(path: str | Path) -> dict[str, float]:
    """Load and validate ``dl_segment_stats_*.json``."""
    stats = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("ct_min", "ct_max", "dose_scale"):
        if key not in stats:
            raise ValueError(f"{path}: stats JSON missing '{key}'")
    if float(stats["ct_max"]) <= float(stats["ct_min"]):
        raise ValueError(f"{path}: ct_max must exceed ct_min")
    if float(stats["dose_scale"]) <= 0:
        raise ValueError(f"{path}: dose_scale must be positive")
    return stats


@dataclass
class PatientCT:
    """A CT volume prepared once and reused by every control point on it."""

    coeff: cp.ndarray                      # (X, Y, Z) cubic spline coefficients
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    image: Any                             # SimpleITK image, for output geometry


@dataclass
class DosePredictor:
    """Loads the CNN-xLSTM checkpoint and predicts per-control-point dose."""

    weights: str | Path
    stats: str | Path
    bev_grid_config: str | Path | None = None
    device: str = "cuda:0"
    mode: str = "cubic"
    amp_dtype: torch.dtype | None = torch.bfloat16
    fast_preprocess: bool = True
    use_channels_last: bool = True
    fast_cudnn: bool = True
    allow_tf32: bool = True
    async_cpu_copy: bool = True
    direct_zyx_output: bool = False
    persistent_pinned_outputs: bool = True
    borrow_pinned_rois: bool = False
    prefetch_preprocessing: bool = False
    pipeline_cuda_streams: bool = False
    input_upscale_factor: int = 1
    # Checkpoints trained with --use-compact-physics take four extra encoder
    # channels (WED, Z_eff, air, bone) derived from the BEV CT inside the model,
    # so nothing changes in the input pipeline -- only the channel count.
    # Mutually exclusive with phase packing, as in training.
    use_compact_physics: bool = False
    # --model-capacity-scale from training. Widens every conv stage, so it must
    # match the checkpoint exactly or the state dict will not load.
    model_capacity_scale: float = MODEL_CAPACITY_SCALE

    grid: BevGridConfig = field(init=False)
    input_grid: BevGridConfig = field(init=False)
    model: nn.Module = field(init=False)
    _pinned_output_pool: list[_PinnedOutputSlot] = field(
        init=False, default_factory=list, repr=False
    )
    _stats: dict[str, float] = field(init=False)
    _g_lin: cp.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.torch_device = torch.device(self.device)
        if self.torch_device.type != "cuda":
            raise RuntimeError(
                "dose prediction requires CUDA: the BEV builder uses CuPy and the "
                "CT-space sampler is a Triton kernel"
            )
        self.device_index = self.torch_device.index or 0
        cp.cuda.Device(self.device_index).use()
        torch.cuda.set_device(self.device_index)
        if self.fast_cudnn:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.allow_tf32

        self._stats = load_stats(self.stats)
        self.grid = load_bev_grid_config(self.bev_grid_config)
        self.input_upscale_factor = int(self.input_upscale_factor)
        if self.input_upscale_factor not in (1, 2):
            raise ValueError("input_upscale_factor must be 1 or 2")
        if self.use_compact_physics and self.input_upscale_factor != 1:
            raise ValueError(
                "use_compact_physics requires input_upscale_factor=1; training "
                "rejects the same combination (build_bev_encoder_input cannot "
                "consume phase-packed inputs)"
            )
        self.input_grid = self._build_input_grid()
        self._validate_grid()
        self._g_lin = bev_build.build_bev_index_grid(self.input_grid)

        self.model = self._build_model()

    @torch.inference_mode()
    def warmup(self, *, include_preprocessing: bool = True) -> float:
        """Compile fixed-shape CUDA kernels before readiness is reported.

        Triton and CuPy both JIT on first use, and the mLSTM limit-chunk kernel
        is the expensive one. Doing it here moves that cost out of the first
        /invoke, which is what the platform times.

        No patient image or beam metadata is required, and none is invented:
        CT loading, spline filtering, affine construction and plan-cache
        building stay inside /invoke, because they are shaped by the patient
        and cannot be compiled ahead of one.

        ``include_preprocessing`` is skipped for predictors after the first --
        the preprocessing kernels are keyed on the grid, which every predictor
        shares, so recompiling them per region is wasted startup time.
        """
        started = time.perf_counter()
        torch.cuda.set_device(self.device_index)
        cp.cuda.Device(self.device_index).use()
        compute_dtype = self.amp_dtype or torch.float32
        direct_encoder_input = self._uses_packed_encoder_preprocess()

        if include_preprocessing:
            # CuPy's spline filter JITs on first call too, and /invoke reaches
            # it through load_ct() before any Triton kernel runs.
            dummy_ct = cp.zeros((8, 8, 8), dtype=cp.float32)
            dummy_coeff = bev_build.cpndi.spline_filter(
                dummy_ct, order=3, mode="mirror"
            )
            cp.cuda.Stream.null.synchronize()
            del dummy_coeff, dummy_ct
            warmup_fast_preprocess(
                output_shape=self.input_grid.shape_dhw,
                device=self.torch_device,
                output_dtype=compute_dtype,
                spacing_yz=(
                    float(self.input_grid.spacing_dhw[1]),
                    float(self.input_grid.spacing_dhw[2]),
                ),
                sad_mm=float(self.input_grid.sad_mm),
                packed_encoder_upscale_factor=(
                    self.input_upscale_factor if direct_encoder_input else None
                ),
            )

        if direct_encoder_input:
            phases = self.input_upscale_factor ** 2
            model_input = torch.empty(
                (
                    self.grid.nx,
                    2 * phases,
                    self.grid.ny,
                    self.grid.nz,
                ),
                device=self.torch_device,
                dtype=compute_dtype,
                memory_format=torch.channels_last,
            )
            model_input.zero_()
        else:
            # Built on the *fine* input grid and packed exactly as
            # predict_segments does, so the model sees the real shape.
            model_input = torch.zeros(
                (1, *self.input_grid.shape_dhw),
                device=self.torch_device,
                dtype=compute_dtype,
            )
            if self.input_upscale_factor > 1:
                model_input = self._pack_depth_height_phases(
                    model_input, self.input_upscale_factor
                )
        with torch.amp.autocast(
            "cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None
        ):
            if direct_encoder_input:
                packed = self.model(
                    model_input,
                    encoder_input=True,
                    batch_size=1,
                    depth=self.grid.nx,
                )
            else:
                packed = self.model(model_input, model_input)
        packed = self._pad_packed(packed.float())

        # The packed-coefficient sampler is not specialized on ROI extent, so a
        # token ROI compiles the same kernel a full back-projection uses.
        affine = torch.tensor(
            ((1.0, 0.0, 0.0, 0.0),
             (0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0)),
            device=self.torch_device,
            dtype=torch.float32,
        ).unsqueeze(0)
        sampled, valid = bicubic_iir_sample_affine_from_packed_coeff_triton(
            packed, affine, (8, 8, min(8, self.grid.nz))
        )
        torch.cuda.synchronize(self.torch_device)
        del sampled, valid, affine, packed, model_input
        if include_preprocessing:
            cp.get_default_memory_pool().free_all_blocks()
        return time.perf_counter() - started

    # -- setup ---------------------------------------------------------------

    def _uses_packed_encoder_preprocess(self) -> bool:
        """Whether fast preprocessing can feed a combined model input."""
        return False

    def _build_input_grid(self) -> BevGridConfig:
        """Return the fine sampling grid used before phase packing."""
        r = self.input_upscale_factor
        if r == 1:
            return self.grid
        return BevGridConfig(
            shape_dhw=(self.grid.nx * r, self.grid.ny * r, self.grid.nz),
            spacing_dhw=(
                float(self.grid.spacing_dhw[0]) / r,
                float(self.grid.spacing_dhw[1]) / r,
                float(self.grid.spacing_dhw[2]),
            ),
            sad_mm=self.grid.sad_mm,
            plane_origin_offset_mm=self.grid.plane_origin_offset_mm,
            segment_native_size=self.grid.segment_native_size,
            align_bev_z_to_ct_slices=self.grid.align_bev_z_to_ct_slices,
            align_orient_tol=self.grid.align_orient_tol,
            align_spacing_tol_mm=self.grid.align_spacing_tol_mm,
            aperture_sample_on_unaligned_grid=(
                self.grid.aperture_sample_on_unaligned_grid
            ),
            bicubic_z_align=self.grid.bicubic_z_align,
            bicubic_z_align_backend=self.grid.bicubic_z_align_backend,
            version=self.grid.version,
        )

    def _validate_grid(self) -> None:
        if not (self.grid.align_bev_z_to_ct_slices and self.grid.bicubic_z_align):
            raise ValueError(
                "this checkpoint requires align_bev_z_to_ct_slices=true and "
                "bicubic_z_align=true in the BEV grid config"
            )
        if self.grid.bicubic_z_align_backend != "triton":
            raise ValueError(
                "packed-coefficient sampling requires the Triton z-align backend"
            )
        if self.mode != "cubic":
            raise ValueError("this checkpoint was trained with --otf-bev-mode cubic")


    def _acquire_pinned_output_slot(
        self, capacity: int
    ) -> _PinnedOutputSlot:
        """Lease grow-only pinned buffers without sharing active generators."""
        required = int(capacity)
        for slot in self._pinned_output_pool:
            if slot.in_use:
                continue
            if slot.capacity < required:
                slot.buffers = (
                    torch.empty(required, dtype=torch.float32, pin_memory=True),
                    torch.empty(required, dtype=torch.float32, pin_memory=True),
                )
                slot.capacity = required
            slot.in_use = True
            return slot
        slot = _PinnedOutputSlot(
            buffers=(
                torch.empty(required, dtype=torch.float32, pin_memory=True),
                torch.empty(required, dtype=torch.float32, pin_memory=True),
            ),
            capacity=required,
            in_use=True,
        )
        self._pinned_output_pool.append(slot)
        return slot
    @staticmethod
    def _release_pinned_output_slot(slot: _PinnedOutputSlot | None) -> None:
        if slot is not None:
            slot.in_use = False
    def _warmup_pinned_output_slot(self, capacity: int) -> None:
        slot = self._acquire_pinned_output_slot(capacity)
        self._release_pinned_output_slot(slot)
    def _model_kwargs(self, segments) -> dict:
        """Extra forward kwargs for one batch, or for warmup when ``None``.

        Empty for every checkpoint whose forward is ``(ct, aperture,
        conditioning=...)``. The energy-token CNN-Mamba runs override this to
        supply ``energy_index``, which their prefix stage requires; see
        doserad/proton_mamba_predict.py.
        """
        return {}

    def _build_model(self) -> "nn.Module":
        """Subclasses build the architecture; this class no longer ships one.

        The xLSTM model this originally constructed is not part of the
        submitted photon-ct / photon-mri images, which serve the packed
        CNN-Mamba3 checkpoint through MambaDosePredictor.
        """
        raise NotImplementedError(
            "DosePredictor is an abstract base here; use "
            "doserad.mamba_predict.PackedMambaDosePredictor"
        )

    # -- CT ------------------------------------------------------------------

    def load_ct(self, ct_path: str | Path) -> PatientCT:
        coeff, shape, spacing, origin, image = bev_build.load_ct_volume_xy_z(
            str(ct_path), self.mode
        )
        return PatientCT(coeff=coeff, shape=shape, spacing=spacing, origin=origin, image=image)

    def prepare_ct(self, ct_image: Any) -> PatientCT:
        """Prepare a CT already in memory -- the synthetic CT of the MRI tasks."""
        coeff, shape, spacing, origin, image = bev_build.prepare_ct_volume_xy_z(
            ct_image, self.mode
        )
        return PatientCT(coeff=coeff, shape=shape, spacing=spacing, origin=origin, image=image)

    # -- per-segment ---------------------------------------------------------

    def _bev_inputs(
        self, segment: geometry.PhotonSegment, ct: PatientCT, mac_cache: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the (CT, aperture projection) BEV pair for one control point."""
        aperture = geometry.build_aperture(segment.mlc_left_mm, segment.mlc_right_mm)
        seg_resized = cp.asarray(
            geometry.resize_aperture(
                aperture, self.input_grid.ny, self.input_grid.nz
            ),
            cp.float32,
        )
        bev_ct, seg_proj, _ = bev_build.prepare_bev_input_volumes(
            segment.name,
            ct.coeff,
            ct.spacing,
            ct.origin,
            self.mode,
            mac_cache,
            self._g_lin,
            seg_resized,
            grid=self.input_grid,
            anatomy_cval=-1024.0,
            anatomy_clip_min=-1024.0,
        )
        bev_ct = bev_build.apply_ct_stats_to_bev(bev_ct, self._stats)
        ct_t = torch.from_dlpack(bev_ct.astype(cp.float32, copy=False))
        proj_t = torch.from_dlpack(seg_proj.astype(cp.float32, copy=False))
        return ct_t, proj_t

    def _back_projection(
        self, segment: geometry.PhotonSegment, ct: PatientCT, mac_cache: dict[str, Any]
    ) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, ...]]:
        """Inverse CT-to-BEV affine on the *fine* prediction grid, plus its CT ROI."""
        r = UPSCALE_FACTOR
        dx, dy, dz = (float(v) for v in self.grid.spacing_dhw)
        ctx = bev_build.build_back_projection_affine_ctx(
            segment.name,
            mac_cache,
            ct.shape,
            ct.spacing,
            ct.origin,
            NX=r * self.grid.nx,
            NY=r * self.grid.ny,
            NZ=self.grid.nz,
            spacing_dhw=(dx / r, dy / r, dz),
            align_bev_z_to_ct_slices=True,
            align_orient_tol=self.grid.align_orient_tol,
            align_spacing_tol_mm=self.grid.align_spacing_tol_mm,
        )
        affine = torch.from_dlpack(cp.ascontiguousarray(ctx.affine)).unsqueeze(0)
        w_row = affine[0, 2]
        z_error = torch.stack((
            w_row[0].abs(),
            w_row[1].abs(),
            (w_row[2].abs() - 1.0).abs(),
            (w_row[3] - w_row[3].round()).abs(),
        )).max()
        if float(z_error.item()) > 1e-3:
            raise RuntimeError(
                f"{segment.name}: inverse BEV-to-CT affine is not z aligned "
                f"(error={float(z_error.item()):.3e}); the packed-coefficient "
                "sampler assumes integer z planes"
            )
        return affine, ctx.roi_shape, ctx.roi_box

    # -- coefficient convention -----------------------------------------------
    #
    # Two checkpoint families reach the same CT-space sampler by different
    # routes, so the forward path is written against these two seams rather
    # than against either convention. See MambaDosePredictor for the other one.
    #
    #   this class   --ct-space-direct-packed-coefficients: the model emits the
    #                r**2 phases on the coarse lattice and the packed sampler
    #                interleaves them while reading.
    #   Mamba        --ct-space-direct-spline-coefficients alone: the model
    #                emits a materialised fine grid that *is* the coefficient
    #                array, read by the ordinary affine sampler.

    def _prepare_coefficients(self, raw: torch.Tensor) -> torch.Tensor:
        """Model output -> the coefficient array the sampler expects."""
        return self._pad_packed(raw.float())

    def _sample_ct_roi(self, coefficients, affine, roi_shape):
        """Evaluate one control point's coefficients on its CT ROI."""
        return bicubic_iir_sample_affine_from_packed_coeff_triton(
            coefficients, affine, roi_shape
        )

    @staticmethod
    def _minimum_cutoff_float32(minimum_cutoff: float) -> float:
        cutoff = np.float32(minimum_cutoff)
        if float(cutoff) < minimum_cutoff:
            cutoff = np.nextafter(cutoff, np.float32(np.inf))
        return float(cutoff)

    def _sample_ct_roi_zyx(
        self, coefficients, affine, roi_shape, dose_scale, minimum_cutoff
    ):
        return bicubic_iir_sample_affine_from_packed_coeff_zyx_triton(
            coefficients,
            affine,
            roi_shape,
            dose_scale=dose_scale,
            minimum_cutoff=minimum_cutoff,
        )

    def _pad_packed(self, packed: torch.Tensor) -> torch.Tensor:
        """Centre-pad packed coefficients to the coarse (T, r**2, H, W) lattice."""
        nx, ny, nz = self.grid.shape_dhw
        expected = (nx, UPSCALE_FACTOR ** 2, ny, nz)
        if packed.ndim != 5 or packed.shape[1] != expected[0] or packed.shape[2] != expected[1]:
            raise ValueError(
                f"expected packed coefficients (B,{expected[0]},{expected[1]},<={ny},<={nz}), "
                f"got {tuple(packed.shape)}"
            )
        pad_h = expected[2] - packed.shape[3]
        pad_w = expected[3] - packed.shape[4]
        if pad_h < 0 or pad_w < 0:
            raise ValueError(f"packed coefficients exceed the BEV lattice: {tuple(packed.shape)}")
        if pad_h or pad_w:
            packed = torch.nn.functional.pad(
                packed,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        return packed

    @staticmethod
    def _pack_depth_height_phases(
        volume: torch.Tensor, upscale_factor: int
    ) -> torch.Tensor:
        """Pack ``(B,D*r,H*r,W)`` into ``(B,D,r**2,H,W)``."""
        r = int(upscale_factor)
        if r == 1:
            return volume
        if volume.ndim != 4:
            raise ValueError(
                f"expected fine BEV batch (B,D,H,W), got {tuple(volume.shape)}"
            )
        batch, depth_fine, height_fine, width = volume.shape
        if depth_fine % r or height_fine % r:
            raise ValueError(
                f"fine BEV shape {tuple(volume.shape)} is not divisible by {r}"
            )
        depth = depth_fine // r
        height = height_fine // r
        return (
            volume.reshape(batch, depth, r, height, r, width)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, depth, r * r, height, width)
        )

    # -- public API ----------------------------------------------------------

    @torch.inference_mode()
    def predict_segment_rois(
        self,
        segments: Sequence[geometry.PhotonSegment],
        ct: PatientCT,
        *,
        batch_size: int = 1,
    ) -> Iterable[
        tuple[geometry.PhotonSegment, np.ndarray, tuple[int, ...]]
    ]:
        """Yield physical-dose ROIs as ``(segment, dose_zyx, roi_box_xyz)``.

        Dose is produced one control point at a time from the model's packed
        spline coefficients: each has its own CT ROI box and inverse affine, so
        batching only helps the forward pass, not the sampler.

        The returned array may view a reusable pinned transfer buffer and is
        therefore valid only until the iterator advances. Consumers retaining
        it must make a copy; ``StreamingMHAWriter.submit_roi`` does so.
        """
        cp.cuda.Device(self.device_index).use()
        torch.cuda.set_device(self.device_index)
        if not segments:
            return
        dose_scale = float(self._stats["dose_scale"])
        fast_cache = (
            build_plan_cache(
                segments,
                ct,
                self.input_grid,
                self.torch_device,
                output_upscale_factor=UPSCALE_FACTOR,
                output_grid=self.grid,
                collapsed_dtype=(
                    torch.float32
                    if self.input_upscale_factor > 1
                    else torch.bfloat16
                ),
            )
            if self.fast_preprocess
            else None
        )
        direct_encoder_input = (
            fast_cache is not None and self._uses_packed_encoder_preprocess()
        )
        copy_stream = (
            torch.cuda.Stream(device=self.torch_device)
            if self.async_cpu_copy and fast_cache is not None
            else None
        )
        pinned_buffers: tuple[torch.Tensor, ...] | None = None
        available_buffers: queue.Queue[int] | None = None
        pending: deque[
            tuple[
                geometry.PhotonSegment,
                torch.Tensor,
                torch.Tensor | None,
                torch.cuda.Event | None,
                tuple[int, ...],
                int | None,
            ]
        ] = deque()
        if fast_cache is not None:
            max_roi_elements = max(
                int(np.prod(context.roi_shape))
                for context in fast_cache.backprojection_contexts
            )
            if self.async_cpu_copy:
                buffer_count = 4 if self.borrow_pinned_rois else 2
                pinned_buffers = tuple(
                    torch.empty(max_roi_elements, dtype=torch.float32, pin_memory=True)
                    for _ in range(buffer_count)
                )
                if self.borrow_pinned_rois:
                    available_buffers = queue.Queue()
                    for buffer_index in range(buffer_count):
                        available_buffers.put(buffer_index)

        def finish_copy(
            item: tuple[
                geometry.PhotonSegment,
                torch.Tensor,
                torch.Tensor | None,
                torch.cuda.Event | None,
                tuple[int, ...],
                int | None,
            ],
        ) -> tuple[
            geometry.PhotonSegment, np.ndarray, tuple[int, ...]
        ]:
            segment, host_buffer, source, event, roi_box, buffer_index = item
            if event is not None:
                event.synchronize()
            xs, xe, ys, ye, zs, ze = roi_box
            roi_shape_zyx = (ze - zs, ye - ys, xe - xs)
            roi_elements = int(np.prod(roi_shape_zyx))
            dose_roi_zyx = (
                host_buffer[:roi_elements].view(roi_shape_zyx).numpy()
            )
            if buffer_index is not None:
                assert available_buffers is not None
                dose_roi_zyx = BorrowedDoseROI(
                    dose_roi_zyx,
                    lambda index=buffer_index: available_buffers.put(index),
                )
            # Retaining source until its event has completed prevents the CUDA
            # allocator from reusing storage still consumed by the copy stream.
            del source
            return segment, dose_roi_zyx, roi_box

        next_buffer = 0

        def prepare_fast_batch(start: int, chunk_size: int):
            assert fast_cache is not None
            indices = torch.arange(
                start,
                start + chunk_size,
                device=self.torch_device,
                dtype=torch.long,
            )
            if direct_encoder_input:
                encoder_input = preprocess_packed_encoder_batch(
                    fast_cache,
                    indices,
                    output_shape=self.input_grid.shape_dhw,
                    upscale_factor=self.input_upscale_factor,
                    ct_min=float(self._stats["ct_min"]),
                    ct_max=float(self._stats["ct_max"]),
                    output_dtype=(
                        self.amp_dtype
                        if self.amp_dtype is not None
                        else torch.float32
                    ),
                    spacing_yz=(
                        float(self.input_grid.spacing_dhw[1]),
                        float(self.input_grid.spacing_dhw[2]),
                    ),
                    sad_mm=float(self.input_grid.sad_mm),
                )
                return encoder_input, None
            ct_batch, proj_batch = preprocess_batch(
                fast_cache,
                indices,
                output_shape=self.input_grid.shape_dhw,
                ct_min=float(self._stats["ct_min"]),
                ct_max=float(self._stats["ct_max"]),
                output_dtype=(
                    self.amp_dtype
                    if self.amp_dtype is not None
                    else torch.float32
                ),
                spacing_yz=(
                    float(self.input_grid.spacing_dhw[1]),
                    float(self.input_grid.spacing_dhw[2]),
                ),
                sad_mm=float(self.input_grid.sad_mm),
            )
            if self.input_upscale_factor > 1:
                ct_batch = self._pack_depth_height_phases(
                    ct_batch, self.input_upscale_factor
                )
                proj_batch = self._pack_depth_height_phases(
                    proj_batch, self.input_upscale_factor
                )
            return ct_batch, proj_batch

        prefetch_stream = (
            torch.cuda.Stream(device=self.torch_device)
            if self.prefetch_preprocessing and fast_cache is not None
            else None
        )
        sampling_stream = (
            torch.cuda.Stream(device=self.torch_device)
            if self.pipeline_cuda_streams
            else None
        )
        prefetched = None
        if prefetch_stream is not None:
            first_size = min(max(1, batch_size), len(segments))
            with torch.cuda.stream(prefetch_stream):
                first_ct, first_proj = prepare_fast_batch(0, first_size)
                first_event = torch.cuda.Event()
                first_event.record(prefetch_stream)
            prefetched = (first_ct, first_proj, first_event)

        for start in range(0, len(segments), max(1, batch_size)):
            chunk = segments[start:start + max(1, batch_size)]
            if fast_cache is not None:
                if prefetched is None:
                    ct_batch, proj_batch = prepare_fast_batch(start, len(chunk))
                else:
                    ct_batch, proj_batch, ready = prefetched
                    torch.cuda.current_stream().wait_event(ready)
                    ct_batch.record_stream(torch.cuda.current_stream())
                    if proj_batch is not None:
                        proj_batch.record_stream(torch.cuda.current_stream())
                    next_start = start + len(chunk)
                    if next_start < len(segments):
                        next_size = min(max(1, batch_size), len(segments) - next_start)
                        assert prefetch_stream is not None
                        with torch.cuda.stream(prefetch_stream):
                            next_ct, next_proj = prepare_fast_batch(
                                next_start, next_size
                            )
                            next_event = torch.cuda.Event()
                            next_event.record(prefetch_stream)
                        prefetched = (next_ct, next_proj, next_event)
                    else:
                        prefetched = None
                mac_cache = None
            else:
                mac_cache = {
                    seg.name: geometry.mac_cache_entry(
                        seg.iso_center, seg.gantry_angle, self.input_grid
                    )
                    for seg in chunk
                }
                ct_list, proj_list = [], []
                for seg in chunk:
                    ct_t, proj_t = self._bev_inputs(seg, ct, mac_cache)
                    ct_list.append(ct_t)
                    proj_list.append(proj_t)
                ct_batch = torch.stack(ct_list, dim=0)
                proj_batch = torch.stack(proj_list, dim=0)

            if self.input_upscale_factor > 1 and fast_cache is None:
                ct_batch = self._pack_depth_height_phases(
                    ct_batch, self.input_upscale_factor
                )
                proj_batch = self._pack_depth_height_phases(
                    proj_batch, self.input_upscale_factor
                )

            autocast = torch.amp.autocast(
                "cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None
            )
            with autocast:
                if direct_encoder_input:
                    raw = self.model(
                        ct_batch,
                        encoder_input=True,
                        batch_size=len(chunk),
                        depth=self.grid.nx,
                    )
                else:
                    assert proj_batch is not None
                    raw = self.model(ct_batch, proj_batch)
            coefficients = self._prepare_coefficients(raw)

            for i, seg in enumerate(chunk):
                if fast_cache is not None:
                    context = fast_cache.backprojection_contexts[start + i]
                    affine = fast_cache.inverse_affines[start + i]
                    roi_shape = context.roi_shape
                    roi_box = context.roi_box
                else:
                    assert mac_cache is not None
                    affine, roi_shape, roi_box = self._back_projection(
                        seg, ct, mac_cache
                    )
                producer_stream = torch.cuda.current_stream()
                if sampling_stream is not None:
                    sampling_stream.wait_stream(producer_stream)
                    sample_context = torch.cuda.stream(sampling_stream)
                else:
                    sample_context = contextlib.nullcontext()
                with sample_context:
                    if self.direct_zyx_output:
                        pred_zyx = self._sample_ct_roi_zyx(
                            coefficients[i:i + 1],
                            affine,
                            roi_shape,
                            dose_scale,
                            self._minimum_cutoff_float32(seg.minimum_cutoff),
                        )[0]
                    else:
                        pred, valid = self._sample_ct_roi(
                            coefficients[i:i + 1], affine, roi_shape
                        )
                        pred = pred.masked_fill(~valid, 0.0)[0] * dose_scale
                        pred_zyx = pred.permute(2, 1, 0).contiguous()
                if sampling_stream is not None:
                    coefficients.record_stream(sampling_stream)
                    producer_stream = sampling_stream
                roi_elements = pred_zyx.numel()

                if copy_stream is not None:
                    assert pinned_buffers is not None
                    if self.borrow_pinned_rois:
                        assert available_buffers is not None
                        buffer_index = available_buffers.get()
                        host_buffer = pinned_buffers[buffer_index]
                    else:
                        buffer_index = None
                        host_buffer = pinned_buffers[next_buffer]
                        next_buffer = 1 - next_buffer
                    event = torch.cuda.Event()
                    copy_stream.wait_stream(producer_stream)
                    with torch.cuda.stream(copy_stream):
                        host_buffer[:roi_elements].copy_(
                            pred_zyx.reshape(-1), non_blocking=True
                        )
                        event.record(copy_stream)
                    pending.append(
                        (seg, host_buffer, pred_zyx, event, roi_box, buffer_index)
                    )
                    # Keep one transfer pending so it overlaps the next model
                    # forward; two pinned buffers bound host memory use.
                    if len(pending) >= 2:
                        yield finish_copy(pending.popleft())
                else:
                    host_buffer = pred_zyx.cpu().reshape(-1)
                    yield finish_copy(
                        (seg, host_buffer, None, None, roi_box, None)
                    )

        while pending:
            yield finish_copy(pending.popleft())

    @torch.inference_mode()
    def predict_segments(
        self,
        segments: Sequence[geometry.PhotonSegment],
        ct: PatientCT,
        *,
        batch_size: int = 1,
    ) -> Iterable[tuple[geometry.PhotonSegment, np.ndarray]]:
        """Yield full CT-grid ``(segment, dose_zyx)`` arrays.

        This preserves the original public API. The sparse MHA fast path calls
        :meth:`predict_segment_rois` directly and avoids materialising zeros
        outside each inverse-projection ROI.
        """
        for segment, dose_roi_zyx, roi_box in self.predict_segment_rois(
            segments, ct, batch_size=batch_size
        ):
            xs, xe, ys, ye, zs, ze = roi_box
            dose_zyx = np.zeros(
                (ct.shape[2], ct.shape[1], ct.shape[0]), dtype=np.float32
            )
            dose_zyx[zs:ze, ys:ye, xs:xe] = dose_roi_zyx
            yield segment, dose_zyx
