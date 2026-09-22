"""CNN-Mamba variants of :class:`~doserad.predict.DosePredictor`.

Two classes live here, one per coefficient convention, because nothing in a
bare state dict says which a checkpoint was trained for:

:class:`MambaDosePredictor`
    ``--ct-space-direct-spline-coefficients`` alone. Documented below.

:class:`PackedMambaDosePredictor`
    that plus ``--ct-space-direct-packed-coefficients`` and
    ``--otf-input-upscale-factor 2``, which is the base class's own pipeline,
    so it overrides almost nothing.

For :class:`MambaDosePredictor`, the input pipeline, plan cache,
back-projection geometry and MHA writing are shared with the CNN-xLSTM path
unchanged -- only two things differ, and each is isolated to one override
here:

``_build_model``
    a ``CNN_Mamba2_L2_SpatialMix`` built from the run's ``mamba_arch`` block
    rather than a ``CNN_xLSTM_temporal``.

``_prepare_coefficients`` / ``_sample_ct_roi``
    this checkpoint trained with ``--ct-space-direct-spline-coefficients`` but
    *not* ``--ct-space-direct-packed-coefficients``, so the decoder's pixel
    shuffle materialises a fine (r*D, r*H, W) grid which is used directly as
    the spline coefficient array. Training does the same: with direct spline
    coefficients the model output is appended to ``coeffs`` with no prefilter
    (train/utils.py), then read by the non-packed affine sampler.

The back-projection geometry needs no change at all: ``build_plan_cache``
already builds its inverse affines against ``output_grid`` scaled by
``output_upscale_factor``, which is the same fine lattice training uses
(``NX=r*nx, NY=r*ny, NZ=nz`` at ``dx/r, dy/r, dz``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .ct_sample import bicubic_iir_sample_affine_from_coeff_triton
from .predict import UPSCALE_FACTOR, DosePredictor


def build_mamba_model(
    arch: str | Path | dict[str, Any] | None,
    weights: str | Path,
    torch_device: Any,
    use_channels_last: bool,
    *,
    expect_packed: bool,
):
    """Build ``CNN_Mamba2_L2_SpatialMix`` from a run's arch block and load it.

    Shared by both Mamba predictors below, which differ only in what they do
    with the model's output, not in how the model is built.
    """
    from .nets.cnn_mamba import CNN_Mamba2_L2_SpatialMix, MambaDoseArchConfig

    if arch is None:
        raise ValueError("a Mamba predictor requires arch=<config_hyper.json>")
    arch_json = dict(arch) if isinstance(arch, dict) else load_mamba_arch(arch)

    # Fields the dataclass does not know are a version skew between the
    # training tree and this vendored copy -- refuse rather than silently
    # building a differently shaped model than the weights expect.
    known = set(MambaDoseArchConfig.__dataclass_fields__)
    unknown = sorted(set(arch_json) - known)
    if unknown:
        raise ValueError(
            f"mamba_arch has fields unknown to the vendored model: {unknown}"
        )
    packed = bool(arch_json.get("return_packed_dh_coefficients"))
    if packed != expect_packed:
        raise ValueError(
            f"checkpoint has return_packed_dh_coefficients={packed}, but this "
            f"predictor implements the "
            f"{'packed' if expect_packed else 'direct-spline'}-coefficient path; "
            "use the other MambaDosePredictor class"
        )
    config = MambaDoseArchConfig(**arch_json)

    model = CNN_Mamba2_L2_SpatialMix(
        config, use_channels_last=use_channels_last
    ).to(torch_device)

    state = torch.load(str(weights), map_location=torch_device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    from .nets.mamba_inference_fusions import install_mamba_inference_fusions

    install_mamba_inference_fusions(model)
    return model, config


def load_mamba_arch(path: str | Path) -> dict[str, Any]:
    """Read the ``mamba_arch`` block out of a run's ``config_hyper.json``.

    Shipping the training config rather than restating its fields keeps the
    architecture and the weights from drifting apart: every field the model
    needs is already recorded there by train.py.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if "mamba_arch" not in config:
        raise ValueError(f"{path}: no 'mamba_arch' block; not a CNN-Mamba run")
    return dict(config["mamba_arch"])


@dataclass
class MambaDosePredictor(DosePredictor):
    """Predicts per-control-point dose with a CNN-Mamba checkpoint."""

    # Path to the run's config_hyper.json, or the mamba_arch dict itself.
    arch: str | Path | dict[str, Any] | None = None

    _arch: Any = field(init=False, default=None)

    def _build_model(self):
        model, self._arch = build_mamba_model(
            self.arch,
            self.weights,
            self.torch_device,
            self.use_channels_last,
            expect_packed=False,
        )
        return model

    def _prepare_coefficients(self, raw: torch.Tensor) -> torch.Tensor:
        """The fine dose grid is already the spline coefficient array."""
        if raw.ndim == 5 and raw.shape[1] == 1:
            raw = raw[:, 0]
        if raw.ndim != 4:
            raise ValueError(f"expected model output (B,D,H,W), got {tuple(raw.shape)}")

        nx, ny, nz = self.grid.shape_dhw
        expected = (UPSCALE_FACTOR * nx, UPSCALE_FACTOR * ny, nz)
        if (
            raw.shape[1] != expected[0]
            or raw.shape[2] > expected[1]
            or raw.shape[3] > expected[2]
        ):
            raise ValueError(
                f"model dose shape {tuple(raw.shape[1:])} is incompatible with the "
                f"fine prediction grid {expected}"
            )
        # Centre-pad a narrower head onto the lattice, exactly as training does
        # before sampling.
        pad_h = expected[1] - raw.shape[2]
        pad_w = expected[2] - raw.shape[3]
        raw = raw.float()
        if pad_h or pad_w:
            raw = torch.nn.functional.pad(
                raw,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        return raw

    def _sample_ct_roi(self, coefficients, affine, roi_shape):
        return bicubic_iir_sample_affine_from_coeff_triton(
            coefficients, affine, roi_shape
        )

    @torch.inference_mode()
    def warmup(self, *, include_preprocessing: bool = True) -> float:
        """Same contract as the base warmup, on this model's shapes.

        Mamba's scan is a Triton kernel keyed on the sequence length and head
        dimensions, so it JITs on first use just as the mLSTM one does.
        """
        import time

        import cupy as cp

        from . import bev_build
        from .fast_preprocess import warmup_fast_preprocess

        started = time.perf_counter()
        torch.cuda.set_device(self.device_index)
        cp.cuda.Device(self.device_index).use()
        compute_dtype = self.amp_dtype or torch.float32

        if include_preprocessing:
            dummy_ct = cp.zeros((8, 8, 8), dtype=cp.float32)
            dummy_coeff = bev_build.cpndi.spline_filter(dummy_ct, order=3, mode="mirror")
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
            )

        fine = torch.zeros(
            (1, *self.input_grid.shape_dhw),
            device=self.torch_device,
            dtype=compute_dtype,
        )
        with torch.amp.autocast(
            "cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None
        ):
            raw = self.model(fine, fine)
        coefficients = self._prepare_coefficients(raw)
        affine = torch.tensor(
            ((1.0, 0.0, 0.0, 0.0),
             (0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0)),
            device=self.torch_device,
            dtype=torch.float32,
        ).unsqueeze(0)
        sampled, valid = self._sample_ct_roi(
            coefficients, affine, (8, 8, min(8, self.grid.nz))
        )
        torch.cuda.synchronize(self.torch_device)
        del sampled, valid, affine, coefficients, raw, fine
        if include_preprocessing:
            cp.get_default_memory_pool().free_all_blocks()
        return time.perf_counter() - started


@dataclass
class PackedMambaDosePredictor(DosePredictor):
    """CNN-Mamba trained with ``--ct-space-direct-packed-coefficients``.

    Where :class:`MambaDosePredictor` steps *away* from the base class,
    this one only swaps the model in and leaves the base's pipeline alone.
    The checkpoint it serves was trained with the same three flags the
    CNN-xLSTM path assumes --

        --otf-input-upscale-factor 2
        --ct-space-direct-spline-coefficients
        --ct-space-direct-packed-coefficients

    -- so phase packing on the way in, the packed-coefficient sampler on the way
    out, and the base ``warmup`` all apply unchanged. Only ``_build_model``
    differs from :class:`DosePredictor`.

    ``model_capacity_scale`` and the phase-channel count are *not* taken from
    the constructor here as they are on the xLSTM path: both are recorded in the
    run's ``mamba_arch`` block, so they travel with the weights. The upscale
    factor is the one number that has to agree in two places -- the arch's
    ``input_phase_channels`` and this predictor's ``input_upscale_factor``,
    which sizes the input grid -- and that agreement is checked.
    """

    # Path to the run's config_hyper.json, or the mamba_arch dict itself.
    arch: str | Path | dict[str, Any] | None = None
    # Inference-only: have the affine Triton preprocessors write normalized,
    # phase-packed CT/aperture channels directly into the model's channels-last
    # encoder buffer. Disable for an exact old/new benchmark or diagnosis.
    fused_encoder_preprocess: bool = True

    _arch: Any = field(init=False, default=None)

    def _uses_packed_encoder_preprocess(self) -> bool:
        return bool(
            self.fused_encoder_preprocess
            and self.fast_preprocess
            and self.use_channels_last
            and self.input_upscale_factor > 1
            and self._arch is not None
            and int(self._arch.n_prefix) == 0
        )

    def _build_model(self):
        model, self._arch = build_mamba_model(
            self.arch,
            self.weights,
            self.torch_device,
            self.use_channels_last,
            expect_packed=True,
        )
        expected_phases = self.input_upscale_factor ** 2
        if int(self._arch.input_phase_channels) != expected_phases:
            raise ValueError(
                f"arch has input_phase_channels="
                f"{self._arch.input_phase_channels}, but "
                f"input_upscale_factor={self.input_upscale_factor} packs "
                f"{expected_phases} phases per modality"
            )
        if (int(self._arch.input_h), int(self._arch.input_w)) != (
            int(self.grid.ny), int(self.grid.nz)
        ):
            raise ValueError(
                f"arch input_h/input_w ({self._arch.input_h}, "
                f"{self._arch.input_w}) do not match the BEV grid "
                f"({self.grid.ny}, {self.grid.nz})"
            )
        return model
