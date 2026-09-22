"""MRI -> synthetic CT for ``proton-mri``, via the SwinUNETR cGAN generators.

This is the ``feature/resolution_improvements`` photon pipeline's
``doserad/sct.py`` ported to the proton branch, with two changes:

  * ``ct_statistics.json`` is keyed by modality, so the same file can hold the
    photon and proton constants side by side; the modality is passed in.
  * :class:`SwinSCTManager` presents the ``begin_invoke`` / ``materialize``
    interface that ``inference.py`` already drives ``sct_predict.SCTManager``
    with, so the two backends are interchangeable.

Everything else is unchanged, including the validation the photon docstrings
record: the path reproduces DoseRAD2026_sCT's notebook to 0.001 HU MAE, and the
MR-derived body mask sits within 2.2 HU MAE of the CT-derived one.

The generators are per-anatomy even when a single dose checkpoint serves both
regions -- each was trained on its own anatomy's split, and crossing them shifts
the sCT by ~90 HU.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cupy as cp
import cupyx.scipy.ndimage as cpndi
import numpy as np
import SimpleITK as sitk
import torch

# doserad2026_sct.utils.preprocess's clamp bounds. The generator was trained
# against targets clipped to this window, so its output is only meaningful
# inside it.
CT_LOWEST_VALUE = -1024.0
CT_HIGHEST_VALUE = 3000.0

# The proton runs' config.json patch_size, and the overlap the notebook infers
# with. SwinUNETR is a windowed transformer, so the patch size is part of the
# trained configuration, not a free tiling choice.
PATCH_SIZE = (64, 160, 160)
OVERLAP = 0.5

# Fraction of the MR's 99th intensity percentile above which a voxel counts as
# patient. See body_mask_from_mr.
MR_BODY_THRESHOLD_FRACTION = 0.02


def load_population_stats(
    path: str | Path, modality: str, anatomy: str
) -> tuple[float, float]:
    """Read one modality/anatomy's CT population mean/std.

    These are the constants the generator's output is denormalised with: the
    mean of the per-patient means (and of the per-patient stds) over the
    generator's own train+test split, each computed on a body-masked, clipped
    CT. The container cannot recompute them -- it has no CTs -- so they ship as
    a constant.
    """
    stats = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        entry = stats[modality][anatomy]
    except KeyError as error:
        raise ValueError(
            f"{path}: no {modality}/{anatomy} entry; found "
            f"{ {m: sorted(v) for m, v in stats.items()} }"
        ) from error
    mean, std = float(entry["mean"]), float(entry["std"])
    if std <= 0:
        raise ValueError(f"{path}: {modality}/{anatomy} std must be positive")
    return mean, std


def _largest_component_filled(mask: cp.ndarray) -> cp.ndarray:
    """Close, keep the largest connected component, fill holes; returns uint8.

    The morphology of ``doserad2026_sct.utils.masks.get_body_mask``, on the GPU
    via CuPy rather than SciPy: the container has CuPy (the BEV resampler needs
    it) but deliberately ships no SciPy, and on a 300^3 volume the SciPy version
    of this costs tens of seconds per patient.
    """
    # brute_force is CuPy's only implemented iteration strategy. SciPy defaults
    # to False, but the flag selects how the iteration is evaluated, not what it
    # converges to, so the two agree voxel for voxel.
    mask = cpndi.binary_closing(mask, iterations=2, brute_force=True)
    mask = cpndi.binary_closing(mask, structure=cp.ones((5, 5, 5), dtype=bool))
    labels, count = cpndi.label(mask)
    if count == 0:
        return cp.zeros_like(mask, dtype=bool)
    # Component 0 is background; sizes are indexed from label 1.
    sizes = cp.bincount(labels.ravel())[1:]
    mask = labels == (int(cp.argmax(sizes)) + 1)
    # uint8, not bool: torch's dlpack import rejects bool.
    return cpndi.binary_fill_holes(mask).astype(cp.uint8)


def body_mask_from_mr(mr: np.ndarray, device_index: int = 0) -> cp.ndarray:
    """Body mask from the MR alone, standing in for the notebook's CT mask.

    MR intensity is not calibrated, so there is no absolute level to threshold
    at the way -850 HU works on a CT; the threshold is set from the volume's own
    99th percentile instead. Background is receiver noise near zero and the
    patient is orders of magnitude brighter, so the split is wide.

    Chosen by sweeping the fraction against the CT-derived mask on the 12
    held-out patients of both generators' splits. At 0.02 the finished sCT
    differs from the CT-masked one by 2.2 HU MAE on average (worst patient 9.0),
    Dice 0.990 against the CT mask. The response is flat from 0.005 to 0.04 --
    1.9 to 2.9 HU -- so this sits mid-plateau rather than on an edge.

    The alternatives were both far worse, and are recorded here because both
    look plausible: masking from the sCT itself with the same -850 HU rule
    scores 42 HU MAE (the generator does not render the air outside the patient
    at -1024, so the threshold leaks into background on some patients, up to
    73% of the body's volume), and not masking at all scores 55 HU.
    """
    with cp.cuda.Device(device_index):
        volume = cp.asarray(np.ascontiguousarray(mr, dtype=np.float32))
        positive = volume[volume > 0]
        if positive.size == 0:
            raise ValueError("MR volume has no positive intensities; cannot mask")
        threshold = MR_BODY_THRESHOLD_FRACTION * float(cp.percentile(positive, 99))
        return _largest_component_filled(volume > threshold)


@dataclass
class SCTGenerator:
    """SwinUNETR MR-to-CT generator for one anatomical region."""

    weights: str | Path
    population_stats: str | Path
    anatomy: str
    modality: str = "proton"
    device: str = "cuda:0"
    patch_size: tuple[int, int, int] = PATCH_SIZE
    overlap: float = OVERLAP
    # "fp32" runs the generator in full precision. "bf16" wraps the forward in
    # torch.autocast, which is what DoseRAD2026_sCT's own validation epoch used
    # (trainers/sct.py). The weights are fp32 either way; only the forward is
    # cast. Denormalisation always happens in fp32 -- see generate().
    precision: str = "fp32"
    # How many sliding windows are evaluated per forward. This is a pure
    # scheduling choice: MONAI batches the windows and blends them exactly as
    # before, so the output is bit-identical to sw_batch_size=1. It only trades
    # VRAM for GPU utilisation. Measured on a 152x520x547 volume, peak reserved
    # is ~4.9 GiB at 1 and ~11.9 GiB at 4 in bf16, against the A10G's 22.6 GB
    # shared with the resident dose model.
    sw_batch_size: int = 1

    model: Any = field(init=False)
    ct_mean: float = field(init=False)
    ct_std: float = field(init=False)

    def __post_init__(self) -> None:
        from monai.networks.nets import SwinUNETR

        self.sw_batch_size = int(self.sw_batch_size)
        if self.sw_batch_size < 1:
            raise ValueError(
                f"sw_batch_size must be >= 1, got {self.sw_batch_size}"
            )
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError(
                f"Unknown sCT precision {self.precision!r}; expected fp32 or bf16"
            )
        self.torch_device = torch.device(self.device)
        self.ct_mean, self.ct_std = load_population_stats(
            self.population_stats, self.modality, self.anatomy
        )

        # feature_size and use_v2 are the generator's trained architecture
        # (doserad2026_sct/trainers/sct.py); they are not tunable here.
        model = SwinUNETR(
            in_channels=1, out_channels=1, feature_size=48, use_v2=True
        )
        state = torch.load(str(self.weights), map_location="cpu", weights_only=True)
        # The training snapshots hold generator, discriminator and both
        # optimizer states in one 900 MB file. Either the extracted generator
        # state or a whole snapshot is accepted.
        if isinstance(state, dict) and "GENERATOR_MODEL_STATE" in state:
            state = state["GENERATOR_MODEL_STATE"]
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.torch_device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def generate(self, mr_image: sitk.Image) -> sitk.Image:
        """Return the sCT for one MR, on the MR's own voxel grid.

        Geometry is copied from the input rather than rebuilt: the dose stage
        places every beam from the image's spacing and origin, so an sCT that
        lost them would silently put the patient somewhere else.
        """
        from monai.inferers import sliding_window_inference

        mr = sitk.GetArrayFromImage(mr_image).astype(np.float32, copy=False)

        # Whole-volume z-score, background included -- preprocess_mr's
        # "z-score" method takes no mask.
        std = float(mr.std())
        if std == 0:
            raise ValueError("MR volume is constant; cannot z-score normalise")
        normalised = (mr - float(mr.mean())) / std

        with self._autocast():
            predicted = sliding_window_inference(
                inputs=torch.as_tensor(normalised)[None, None].to(self.torch_device),
                roi_size=list(self.patch_size),
                sw_batch_size=self.sw_batch_size,
                predictor=self.model,
                overlap=self.overlap,
            )
        # Back to fp32 before denormalising: bf16 carries 8 mantissa bits, so
        # scaling into a +-3000 HU range in that dtype would quantise the output
        # to steps of ~16 HU.
        predicted = predicted.float() * self.ct_std + self.ct_mean

        mask = body_mask_from_mr(mr, device_index=self.torch_device.index or 0)
        mask_t = torch.from_dlpack(mask).to(torch.bool)
        sct = predicted.squeeze()
        sct = torch.where(
            mask_t,
            sct,
            torch.as_tensor(CT_LOWEST_VALUE, device=sct.device, dtype=sct.dtype),
        )
        sct = torch.clamp(sct, CT_LOWEST_VALUE, CT_HIGHEST_VALUE)

        sct_image = sitk.GetImageFromArray(
            sct.cpu().numpy().astype(np.float32, copy=False)
        )
        sct_image.CopyInformation(mr_image)
        return sct_image

    def _autocast(self):
        """Autocast context for the forward pass, or a no-op under fp32."""
        if self.precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @torch.inference_mode()
    def warmup(self) -> float:
        """Run one patch through the generator before readiness is reported.

        Runs under the same autocast as generate(), so the kernels selected here
        are the ones the first /invoke will use.
        """
        started = time.perf_counter()
        dummy = torch.zeros(
            (1, 1, *self.patch_size), device=self.torch_device, dtype=torch.float32
        )
        with self._autocast():
            self.model(dummy)
        torch.cuda.synchronize(self.torch_device)
        del dummy
        return time.perf_counter() - started


class SwinSCTManager:
    """Own regional SwinUNETR generators and one per-invocation on-disk cache.

    Interface-compatible with ``sct_predict.SCTManager`` so ``inference.py``
    drives either backend unchanged.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | torch.device,
        generators: dict[str, str],
        population_stats: str = "sct_ct_population_statistics.json",
        modality: str = "proton",
        precision: str = "fp32",
        sw_batch_size: int = 1,
        lazy: bool = False,
        cache_dir: str | Path = "/tmp/doserad-sct",
    ) -> None:
        self.root = Path(model_dir)
        stats_path = self.root / population_stats
        if not stats_path.is_file():
            raise FileNotFoundError(f"missing sCT population statistics: {stats_path}")
        self._spec = {
            region: dict(
                weights=self.root / weights,
                population_stats=stats_path,
                anatomy=region,
                modality=modality,
                device=str(device),
                precision=precision,
                sw_batch_size=sw_batch_size,
            )
            for region, weights in generators.items()
        }
        # Even when loading is deferred, prove every checkpoint is present now:
        # a missing file should fail at startup, not part-way through an invoke.
        for region, spec in self._spec.items():
            if not Path(spec["weights"]).is_file():
                raise FileNotFoundError(
                    f"missing {region} sCT generator weights: {spec['weights']}"
                )
        self.lazy = bool(lazy)
        self.generators: dict[str, SCTGenerator] = {}
        if not self.lazy:
            for region in self._spec:
                self._generator(region)
        self.cache_dir = Path(cache_dir)
        self._paths: dict[tuple[str, int], Path] = {}

    def _generator(self, region: str) -> SCTGenerator:
        """Return the region's generator, constructing it on first use.

        Under lazy loading this is what pays the ~300 MB read and the SwinUNETR
        build, moving both off the startup path that the platform's health check
        is timing.
        """
        generator = self.generators.get(region)
        if generator is None:
            spec = self._spec.get(region)
            if spec is None:
                raise KeyError(
                    f"no sCT generator for {region!r}; available: "
                    f"{sorted(self._spec)}"
                )
            started = time.perf_counter()
            generator = SCTGenerator(**spec)
            self.generators[region] = generator
            print(
                f"Loaded {region} SwinUNETR sCT generator in "
                f"{time.perf_counter() - started:.2f}s",
                flush=True,
            )
        return generator

    def warmup(self) -> None:
        """Warm every already-constructed generator.

        Under lazy loading nothing is constructed yet, so there is deliberately
        nothing to warm -- the first use of each region pays it instead.
        """
        if self.lazy:
            print(
                "Deferred sCT generator loading; skipping warmup",
                flush=True,
            )
            return
        for region, generator in self.generators.items():
            elapsed = generator.warmup()
            print(f"Warmed {region} sCT generator in {elapsed:.2f}s", flush=True)

    def begin_invoke(self) -> None:
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._paths.clear()

    def materialize(
        self, mr_path: str | Path, region: str, image_index: int
    ) -> Path:
        key = (str(region), int(image_index))
        cached = self._paths.get(key)
        if cached is not None and cached.is_file():
            return cached
        generator = self._generator(region)
        started = time.perf_counter()
        sct = generator.generate(sitk.ReadImage(str(mr_path)))
        output = self.cache_dir / f"image_{int(image_index):02d}_{region}.mha"
        sitk.WriteImage(sct, str(output), useCompression=False)
        self._paths[key] = output
        print(
            f"Synthesized input image {int(image_index) + 1} with the "
            f"{region} SwinUNETR generator in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        return output
