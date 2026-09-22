"""CNN-Mamba variant of :class:`~doserad.proton_predict.ProtonDosePredictor`.

The proton pipeline -- plan cache, RSP material state, fixed range/WET
conditioning, packed-coefficient back-projection and MHA writing -- is entirely
shared with the CNN-xLSTM path. Only two things differ, and each is one override
here:

``_build_model``
    a ``CNN_Mamba2_L2_SpatialMix`` built from the run's ``mamba_arch`` block
    rather than a ``CNN_xLSTM_temporal`` assembled from module-level constants.
    When the checkpoint carries ``proton_output_refiner.*`` keys, the same
    zero-init Bragg residual head used on the xLSTM path is attached before
    ``load_state_dict``.

``_model_kwargs``
    supplies ``energy_index`` when the checkpoint carries an energy-token
    embedding. The base class returns no model-specific kwargs.

The inherited prediction loop has an optional proton-Mamba fast path which
fuses material-to-RSP conversion, cumulative WET, space-to-depth packing and
channels-last encoder-input construction. It supports energy tokens and
energy tokens; only architectures requiring an aperture-derived prefix or an
unknown conditioning layout keep the reference preprocessing path.

Unlike the xLSTM path, capacity, phase channels and the bottleneck width are
*not* passed in from constants: they are recorded in the run's ``mamba_arch``
block, so they travel with the weights. The two numbers that must agree in two
places -- the input upscale factor (arch ``input_phase_channels`` vs this
predictor's ``input_upscale_factor``) and the BEV grid (arch ``input_h``/
``input_w`` vs the loaded grid) -- are checked at build time rather than left to
surface as a state-dict shape error.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .proton_predict import ProtonDosePredictor


def _environment_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


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


def load_mamba_run_config(path: str | Path) -> dict[str, Any]:
    """Read architecture plus proton feature metadata from a training config."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if "mamba_arch" not in config:
        raise ValueError(f"{path}: no 'mamba_arch' block; not a CNN-Mamba run")
    return config


@dataclass
class ProtonMambaDosePredictor(ProtonDosePredictor):
    """Proton beamlet prediction with a packed-coefficient CNN-Mamba3 checkpoint."""

    # Path to the run's config_hyper.json, or the mamba_arch dict itself.
    arch: str | Path | dict[str, Any] | None = None
    # Combined density/class -> RSP/WET -> packed channels-last producer. The
    # reference path remains selectable for diagnosis.
    fused_proton_preprocess: bool = field(
        default_factory=lambda: _environment_flag(
            "DOSERAD_PROTON_FUSED_PREPROCESS", default=True
        )
    )
    fused_ct_material: bool = field(
        default_factory=lambda: _environment_flag(
            "DOSERAD_PROTON_FUSED_CT_MATERIAL", default=True
        )
    )
    # Available for A/B diagnosis, but the real 251-item plan benchmark was
    # 0.6% slower than the reference backprojection sequence on RTX A6000.
    fused_backprojection: bool = field(
        default_factory=lambda: _environment_flag(
            "DOSERAD_PROTON_FUSED_BACKPROJECTION", default=False
        )
    )
    # Apply only the evaluator's final threshold on the existing FP32 output
    # tensor before its asynchronous device-to-host copy. This keeps the
    # reference backprojection arithmetic while avoiding a full NumPy scan of
    # every ROI in the submission coordinator.
    gpu_output_cutoff: bool = field(
        default_factory=lambda: _environment_flag(
            "DOSERAD_PROTON_GPU_OUTPUT_CUTOFF", default=True
        )
    )

    _arch: Any = field(init=False, default=None)
    _mamba_run_config: dict[str, Any] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        levels = int(getattr(self.model, "energy_token_levels", 0))
        if levels and levels != len(self._energy_token_values):
            raise ValueError(
                f"checkpoint has an {levels}-level energy token embedding, but "
                f"{self.beam_parameters} yields "
                f"{len(self._energy_token_values)} unique energies"
            )
        enabled = self._uses_combined_proton_preprocess()
        print(
            "Proton combined preprocessing: "
            f"requested={self.fused_proton_preprocess}, enabled={enabled}, "
            f"fused_ct_material={self._uses_fused_ct_material()}, "
            f"fused_backprojection={self._uses_fused_backprojection()}, "
            f"gpu_output_cutoff={self._uses_gpu_output_cutoff()}",
            flush=True,
        )

    def _model_kwargs(self, segments) -> dict:
        """Return the energy row the token embedding needs, when present."""
        if int(getattr(self.model, "energy_token_levels", 0)) <= 0:
            return {}
        if segments is None:
            # Warmup: any valid row exercises the same embedding/IDD kernels.
            return {
                "energy_index": torch.zeros(
                    1, dtype=torch.long, device=self.torch_device
                )
            }
        energy_index = self._model_energy_indices(segments)
        if energy_index is None:
            raise RuntimeError("Mamba energy index was required but not built")
        return {"energy_index": energy_index}

    def _uses_combined_proton_preprocess(self) -> bool:
        phases = self.input_upscale_factor ** 2
        expected_conditioning = 2 * phases
        return bool(
            self.fused_proton_preprocess
            and self.use_channels_last
            and self.density_calibration == "g4dcm_rsp"
            and self.input_upscale_factor > 1
            and self._arch is not None
            and int(self._arch.input_phase_channels) == phases
            and int(self._arch.conditioning_channels)
            == expected_conditioning
            and (
                int(self._arch.n_prefix) == 0
                or str(self._arch.prefix_mode) == "constant"
            )
        )

    def _uses_fused_ct_material(self) -> bool:
        return bool(
            self.fused_ct_material
            and self._uses_combined_proton_preprocess()
        )

    def _uses_fused_backprojection(self) -> bool:
        return bool(self.fused_backprojection)

    def _uses_gpu_output_cutoff(self) -> bool:
        return bool(self.gpu_output_cutoff)

    @torch.inference_mode()
    def warmup(self, *, include_preprocessing: bool = True) -> float:
        elapsed = super().warmup(include_preprocessing=include_preprocessing)
        if not self._uses_combined_proton_preprocess():
            return elapsed
        from .proton_preprocess_fusions import (
            warmup_combined_proton_preprocess,
            warmup_sample_packed_ct_material,
        )
        from .ct_sample import (
            bicubic_iir_sample_affine_from_packed_coeff_scaled_zyx_triton,
        )

        started = time.perf_counter()
        compute_dtype = self.amp_dtype or torch.float32
        if self._uses_fused_ct_material():
            ct_state = warmup_sample_packed_ct_material(
                output_shape=self.input_grid.shape_dhw,
                upscale_factor=self.input_upscale_factor,
                unique_rays=4,
                ct_min=float(self._stats["ct_min"]),
                ct_max=float(self._stats["ct_max"]),
                ct_hu=self.material_conditioning.ct_hu,
                ct_rho=self.material_conditioning.ct_rho,
                class_dens_upper=self.material_conditioning.class_dens_upper,
                device=self.torch_device,
                output_dtype=compute_dtype,
            )
            del ct_state

        # cuDNN and the direct Mamba encoder specialize on B*T. Exercise all
        # full/partial production batch sizes before readiness, while U=B also
        # grows the fused preprocessing allocator to its maximum workspace.
        fused_backprojection_warmed = False
        for batch in range(1, 5):
            energy_index = (
                torch.zeros(
                    batch, device=self.torch_device, dtype=torch.long
                )
                if int(self._arch.energy_token_levels) > 0
                else None
            )
            preprocess_output = warmup_combined_proton_preprocess(
                output_shape=self.input_grid.shape_dhw,
                upscale_factor=self.input_upscale_factor,
                dz_mm=float(self.input_grid.spacing_dhw[0]),
                device=self.torch_device,
                output_dtype=compute_dtype,
                batch_size=batch,
                unique_rays=batch,
                return_encoder_input=True,
            )
            encoder_input = preprocess_output
            assert encoder_input is not None
            with torch.amp.autocast(
                "cuda",
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                packed = self.model(
                    encoder_input,
                    energy_index=energy_index,
                    encoder_input=True,
                    batch_size=batch,
                    depth=int(self.grid.nx),
                )
                refiner = getattr(self.model, "proton_output_refiner", None)
                if refiner is not None:
                    phases = int(self._arch.input_phase_channels)
                    encoder_5d = encoder_input.view(
                        batch,
                        int(self.grid.nx),
                        encoder_input.shape[1],
                        encoder_input.shape[2],
                        encoder_input.shape[3],
                    )
                    packed = refiner(
                        packed,
                        encoder_5d[:, :, 2 * phases :],
                        encoder_5d[:, :, phases : 2 * phases],
                    )
            packed = self._pad_packed(packed.float())
            if (
                self._uses_fused_backprojection()
                and not fused_backprojection_warmed
            ):
                affine = torch.tensor(
                    (
                        (1.0, 0.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                    ),
                    device=self.torch_device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                sampled = (
                    bicubic_iir_sample_affine_from_packed_coeff_scaled_zyx_triton(
                        packed[:1],
                        affine,
                        (8, 8, min(8, int(self.grid.nz))),
                        dose_scale=float(self._stats["dose_scale"]),
                        cutoff=0.0,
                    )
                )
                del sampled, affine
                fused_backprojection_warmed = True
            del packed, encoder_input

        pinned_mib = max(
            1, int(os.environ.get("DOSERAD_PINNED_ROI_WARMUP_MIB", "48"))
        )
        if self.persistent_pinned_outputs:
            self._warmup_pinned_output_slot(
                pinned_mib * 1024 * 1024 // 4
            )
        torch.cuda.synchronize(self.torch_device)
        return elapsed + (time.perf_counter() - started)

    def _build_model(self):
        from .nets.cnn_mamba import CNN_Mamba2_L2_SpatialMix, MambaDoseArchConfig
        from .nets.physics_conditioning import ProtonBraggResidualHead

        if self.arch is None:
            raise ValueError(
                "ProtonMambaDosePredictor requires arch=<config_hyper.json>"
            )
        if isinstance(self.arch, dict):
            supplied = dict(self.arch)
            if "mamba_arch" in supplied:
                run_config = supplied
                arch_json = dict(supplied["mamba_arch"])
            else:
                run_config = {}
                arch_json = supplied
        else:
            run_config = load_mamba_run_config(self.arch)
            arch_json = dict(run_config["mamba_arch"])
        self._mamba_run_config = run_config

        # Fields the dataclass does not know are a version skew between the
        # training tree and this vendored copy -- refuse rather than silently
        # building a differently shaped model than the weights expect.
        known = set(MambaDoseArchConfig.__dataclass_fields__)
        unknown = sorted(set(arch_json) - known)
        if unknown:
            raise ValueError(
                f"mamba_arch has fields unknown to the vendored model: {unknown}"
            )

        # The proton pipeline reads packed r**2 phase coefficients straight off
        # the coarse lattice; a direct-spline checkpoint would need the other
        # sampler and a different _pad_packed contract.
        if not bool(arch_json.get("return_packed_dh_coefficients")):
            raise ValueError(
                "the proton pipeline requires a checkpoint trained with "
                "--ct-space-direct-packed-coefficients "
                "(mamba_arch.return_packed_dh_coefficients is false)"
            )

        config = MambaDoseArchConfig(**arch_json)

        expected_phases = self.input_upscale_factor ** 2
        if int(config.input_phase_channels) != expected_phases:
            raise ValueError(
                f"arch has input_phase_channels={config.input_phase_channels}, "
                f"but input_upscale_factor={self.input_upscale_factor} packs "
                f"{expected_phases} phases per modality"
            )
        # Fixed range/WET conditioning contributes one packed WET plane and one
        # packed remaining-range plane per phase.
        expected_conditioning = 2 * expected_phases
        if int(config.conditioning_channels) != expected_conditioning:
            raise ValueError(
                f"arch has conditioning_channels={config.conditioning_channels}, "
                f"but proton A/B metadata requires {expected_conditioning}"
            )
        if (int(config.input_h), int(config.input_w)) != (
            int(self.grid.ny),
            int(self.grid.nz),
        ):
            raise ValueError(
                f"arch input_h/input_w ({config.input_h}, {config.input_w}) do "
                f"not match the BEV grid ({self.grid.ny}, {self.grid.nz})"
            )

        model = CNN_Mamba2_L2_SpatialMix(
            config, use_channels_last=self.use_channels_last
        ).to(self.torch_device)

        state = torch.load(
            str(self.weights), map_location=self.torch_device, weights_only=True
        )
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        has_bragg_refiner = any(
            key.startswith("proton_output_refiner.") for key in state
        )
        # Detect Bragg residual from the state dict so legacy CNN-Mamba
        # checkpoints without the head remain loadable.
        phase_channels = int(config.input_phase_channels)
        if has_bragg_refiner:
            model.proton_output_refiner = ProtonBraggResidualHead(
                phase_channels
            ).to(self.torch_device)

        model.load_state_dict(state, strict=True)
        if self.use_channels_last:
            model._apply(
                lambda tensor: tensor.contiguous(
                    memory_format=torch.channels_last
                )
                if tensor.ndim == 4
                else tensor
            )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        # Install only after strict checkpoint loading and freezing so the
        # checkpoint key space and training implementation stay unchanged.
        # Scratch-free c16 is the measured proton default; the fused-angle path
        # remains disabled unless explicitly requested because it was not part
        # of the validated benchmark winner.
        from .nets.mamba_inference_fusions import (
            install_mamba_inference_fusions,
        )

        scratch_free = _environment_flag(
            "DOSERAD_MAMBA3_SCRATCH_FREE", default=True
        )
        fused_angle = _environment_flag(
            "DOSERAD_MAMBA3_FUSED_ANGLE", default=False
        )
        norm_count, activation_count = install_mamba_inference_fusions(
            model,
            scratch_free_mamba3=scratch_free,
            fused_mamba3_angle=fused_angle,
        )
        print(
            "Proton Mamba inference fusions enabled: "
            f"chunk_size={config.mamba3_chunk_size}, "
            f"scratch_free={scratch_free}, fused_angle={fused_angle}, "
            f"layer_norms={norm_count}, decoder_activations={activation_count}",
            flush=True,
        )
        print(f"Proton Bragg residual head: {has_bragg_refiner}", flush=True)
        self._arch = model.arch
        return model
