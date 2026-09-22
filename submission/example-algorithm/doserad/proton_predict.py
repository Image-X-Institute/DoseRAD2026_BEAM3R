"""Production proton beamlet prediction with fixed range/WET conditioning."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cupy as cp
import numpy as np
import torch

from . import bev_build, geometry
from .ct_sample import (
    bicubic_iir_sample_affine_from_packed_coeff_scaled_zyx_triton,
    bicubic_iir_sample_affine_from_packed_coeff_triton,
)
from .fast_preprocess import (
    build_plan_cache,
    warmup_fast_preprocess,
    preprocess_aperture_batch,
    preprocess_ct_batch,
)
from .nets.physics_conditioning import (
    BEVMaterialConditioning,
    ProtonBraggResidualHead,
    proton_csda_range_mm,
)
from .predict import (
    UPSCALE_FACTOR,
    DosePredictor,
    PatientCT,
    _PinnedOutputLease,
    _PinnedOutputSlot,
    _WriterOwnedPinnedArray,
)


@dataclass
class ProtonDosePredictor(DosePredictor):
    """CNN-xLSTM predictor matching the v2 proton population checkpoints."""

    beam_parameters: str | Path = "beam_parameters.json"
    input_upscale_factor: int = 2
    density_calibration: str = "g4dcm_rsp"
    deduplicate_plan_geometry: bool = True
    # FP32, matching the challenge's own proton dose ground truth (every
    # proton/training/*/dose/*.mha is MET_FLOAT) and the photon predictor's
    # default. FP64 doubled every output stack -- 35 beamlets on a 490x491x93
    # grid is 5.8 GiB rather than 2.9 GiB -- and the organizers cap each stack
    # at ~790M voxels, which is just under 3 GiB only at FP32. It also buys no
    # precision: peak dose here is ~1.6e-3 Gy against cutoffs near 1e-6, well
    # inside FP32's ~7 significant digits.
    mha_dtype: np.dtype = field(
        init=False, default=np.dtype(np.float32), repr=False
    )
    _energy_table: tuple[tuple[float, float], ...] = field(init=False)
    _energy_token_values: tuple[float, ...] = field(init=False)
    _energy_sigma_values: tuple[float, ...] = field(init=False)
    # Advertises that predict_segment_rois accepts writer_owned_outputs, so
    # inference.py can hand the MHA writer a pinned-buffer lease instead of a
    # copied array. The photon predictor has no such parameter.
    writer_owned_roi_outputs: bool = field(init=False, default=True, repr=False)
    material_conditioning: BEVMaterialConditioning = field(init=False)

    def __post_init__(self) -> None:
        parameters = json.loads(
            Path(self.beam_parameters).read_text(encoding="utf-8")
        )
        rows = parameters.get("proton", {}).get("energy_table", [])
        if not rows:
            raise ValueError(
                f"{self.beam_parameters}: missing proton.energy_table"
            )
        sorted_rows = sorted(rows, key=lambda row: float(row["energy_mev"]))
        self._energy_table = tuple(
            (float(row["energy_mev"]), float(row["sigma_spot_mm"]))
            for row in sorted_rows
        )
        self._energy_token_values = tuple(row[0] for row in self._energy_table)
        if len(self._energy_token_values) != len(rows):
            raise ValueError(
                f"{self.beam_parameters}: proton.energy_table contains "
                "duplicate energies"
            )
        self._energy_sigma_values = tuple(
            float(row["sigma_energy_mev"]) for row in sorted_rows
        )
        # DosePredictor.__post_init__ builds the model. Parse the energy table
        # first so the energy-token table is available before the model is
        # built.
        super().__post_init__()
        self.material_conditioning = BEVMaterialConditioning(
            dz_mm=float(self.input_grid.spacing_dhw[0]),
            include_class=False,
            density_calibration=self.density_calibration,
        ).to(self.torch_device)

    def _build_model(self) -> "nn.Module":
        """Subclasses build the architecture; this class no longer ships one.

        The CNN-xLSTM model this originally constructed is not part of the
        submitted proton-ct / proton-mri images, which serve the packed
        CNN-Mamba3 checkpoint through ProtonMambaDosePredictor.
        """
        raise NotImplementedError(
            "ProtonDosePredictor is an abstract base here; use "
            "doserad.proton_mamba_predict.ProtonMambaDosePredictor"
        )

    # Overrides DosePredictor.warmup, which in this repo is the photon
    # variant and passes no conditioning: the proton model requires its
    # conditioning channels and Bragg refiner here. Body is the proton
    # reference's DosePredictor.warmup; photon keeps the base method.
    @torch.inference_mode()
    def warmup(self, *, include_preprocessing: bool = True) -> float:
        """Compile fixed-shape CUDA kernels before readiness is reported.

        No patient image or beam metadata is required. Patient-specific CT
        loading, spline filtering, affine construction, and cache building
        intentionally remain part of ``/invoke``.
        """
        started = time.perf_counter()
        torch.cuda.set_device(self.device_index)
        cp.cuda.Device(self.device_index).use()
        compute_dtype = self.amp_dtype or torch.float32

        if include_preprocessing:
            # Exercise CuPy's spline-filter JIT and the two fixed-shape Triton
            # preprocessing kernels. The generic affine sampler is configured
            # not to specialize on patient/ROI dimensions.
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
            )

        depth, height, width = self.grid.shape_dhw
        phases = int(self.model.input_phase_channels)
        model_input = torch.zeros(
            (1, depth, phases, height, width),
            device=self.torch_device,
            dtype=compute_dtype,
        )
        conditioning_channels = int(self.model.conditioning_channels)
        conditioning = (
            torch.zeros(
                (1, depth, conditioning_channels, height, width),
                device=self.torch_device,
                dtype=compute_dtype,
            )
            if conditioning_channels
            else None
        )
        with torch.amp.autocast(
            "cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None
        ):
            model_kwargs = self._model_kwargs(None)
            packed = self.model(
                model_input,
                model_input,
                conditioning=conditioning,
                **model_kwargs,
            )
            refiner = getattr(self.model, "proton_output_refiner", None)
            if refiner is not None:
                # Three arguments, matching this repo's ProtonBraggResidualHead
                # and every other refiner call site here. The proton reference's
                # head also accepts energy_index/analytic_gate; this one does not.
                packed = refiner(
                    packed,
                    conditioning.to(packed.dtype),
                    model_input.to(packed.dtype),
                )
        packed = self._pad_packed(packed.float())
        affine = torch.tensor(
            ((1.0, 0.0, 0.0, 0.0),
             (0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0)),
            device=self.torch_device,
            dtype=torch.float32,
        ).unsqueeze(0)
        sampled, valid = bicubic_iir_sample_affine_from_packed_coeff_triton(
            packed, affine, (8, 8, min(8, width))
        )
        torch.cuda.synchronize(self.torch_device)
        del sampled, valid, affine, packed, conditioning, model_input
        if include_preprocessing:
            cp.get_default_memory_pool().free_all_blocks()
        return time.perf_counter() - started

    def _spot_sigma_mm(self, segment: geometry.ProtonSegment) -> float:
        if segment.sigma_spot_mm is not None:
            return float(segment.sigma_spot_mm)
        energy, sigma = min(
            self._energy_table, key=lambda row: abs(row[0] - segment.energy_mev)
        )
        if abs(energy - segment.energy_mev) > 1.0e-3:
            raise ValueError(
                f"{segment.name}: energy {segment.energy_mev} MeV is not in "
                "the packaged beam-parameter table"
            )
        return sigma

    def _model_energy_indices(
        self, segments: Sequence[geometry.ProtonSegment]
    ) -> torch.Tensor | None:
        """Map beamlet energies to rows of the checkpoint's energy-token table."""
        levels = int(getattr(self.model, "energy_token_levels", 0))
        if levels == 0:
            return None
        expected_levels = levels
        if expected_levels != len(self._energy_token_values):
            raise RuntimeError(
                f"checkpoint expects {expected_levels} proton energy levels, but "
                f"{self.beam_parameters} defines {len(self._energy_token_values)}"
            )
        indices = []
        for segment in segments:
            index = min(
                range(expected_levels),
                key=lambda candidate: abs(
                    self._energy_token_values[candidate] - segment.energy_mev
                ),
            )
            if (
                abs(self._energy_token_values[index] - segment.energy_mev)
                > 1.0e-3
            ):
                raise ValueError(
                    f"{segment.name}: energy {segment.energy_mev} MeV is not "
                    f"in the checkpoint's token table"
                )
            indices.append(index)
        return torch.tensor(
            indices, device=self.torch_device, dtype=torch.long
        )

    def _range_conditioning(
        self,
        hu_fine: torch.Tensor,
        energies_mev: torch.Tensor,
        energy_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        density, class_id = self.material_conditioning.hu_to_material(
            hu_fine.float()
        )
        return self._range_conditioning_from_material(
            density, class_id, energies_mev, energy_index=energy_index
        )

    def _range_conditioning_from_material(
        self,
        density: torch.Tensor,
        class_id: torch.Tensor,
        energies_mev: torch.Tensor,
        energy_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        wet, remaining = self.material_conditioning.fixed_range_wet_from_material(
            density,
            class_id,
            energies_mev,
            normalisation_mm=300.0,
        )
        wet = wet.clamp(0.0, 2.0)
        remaining = remaining.clamp(-2.0, 1.0)
        r = self.input_upscale_factor
        packed_wet = self._pack_depth_height_phases(wet, r)
        planes = [packed_wet, self._pack_depth_height_phases(remaining, r)]
        return torch.cat(planes, dim=2)

    def _range_conditioning_from_rsp_state(
        self,
        density: torch.Tensor,
        electron_ratio: torch.Tensor,
        material_i_mev: torch.Tensor,
        energies_mev: torch.Tensor,
        energy_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        wet, remaining = (
            self.material_conditioning.fixed_range_wet_from_rsp_state(
                density,
                electron_ratio,
                material_i_mev,
                energies_mev,
                normalisation_mm=300.0,
            )
        )
        wet = wet.clamp(0.0, 2.0)
        remaining = remaining.clamp(-2.0, 1.0)
        r = self.input_upscale_factor
        packed_wet = self._pack_depth_height_phases(wet, r)
        planes = [packed_wet, self._pack_depth_height_phases(remaining, r)]
        return torch.cat(planes, dim=2)

    def _uses_combined_proton_preprocess(self) -> bool:
        """Whether inference can use the Mamba-only combined encoder kernel."""
        return False

    def _uses_fused_backprojection(self) -> bool:
        """Whether inference can emit scaled/cutoff ZYX in the sampler."""
        return False

    def _uses_gpu_output_cutoff(self) -> bool:
        """Whether to apply the final float32 cutoff before host transfer."""
        return False

    @property
    def applies_minimum_cutoff_on_device(self) -> bool:
        """Whether all yielded dose outputs already satisfy their cutoff."""
        return bool(
            self._uses_fused_backprojection()
            or self._uses_gpu_output_cutoff()
        )

    def _uses_fused_ct_material(self) -> bool:
        """Whether CT sampling can directly emit packed CT/material state."""
        return False

    @staticmethod
    def _float32_cutoff_threshold(minimum_cutoff: float) -> float:
        cutoff = float(minimum_cutoff)
        cutoff32 = np.float32(cutoff)
        if float(cutoff32) < cutoff:
            cutoff32 = np.nextafter(cutoff32, np.float32(np.inf))
        return float(cutoff32)

    @staticmethod
    def _apply_float32_minimum_cutoff_(
        dose: torch.Tensor, minimum_cutoff: float
    ) -> torch.Tensor:
        """Apply the evaluator-equivalent cutoff in the tensor's device."""
        threshold = ProtonDosePredictor._float32_cutoff_threshold(
            minimum_cutoff
        )
        return dose.masked_fill_(dose < threshold, 0.0)

    @torch.inference_mode()
    def predict_segment_rois(
        self,
        segments: Sequence[geometry.ProtonSegment],
        ct: PatientCT,
        *,
        batch_size: int = 1,
        writer_owned_outputs: bool = False,
    ) -> Iterable[tuple[geometry.ProtonSegment, np.ndarray, tuple[int, ...]]]:
        """Yield FP32 physical-dose proton ROIs on the CT lattice."""
        if not segments:
            return
        if not self.fast_preprocess:
            raise ValueError("proton inference requires cached affine preprocessing")

        torch.cuda.set_device(self.device_index)
        dose_scale = float(self._stats["dose_scale"])
        combined_preprocess = self._uses_combined_proton_preprocess()
        fused_ct_material = self._uses_fused_ct_material()
        fused_backprojection = self._uses_fused_backprojection()
        if combined_preprocess:
            from .proton_preprocess_fusions import (
                build_packed_encoder_input_fused_scan,
            )
        if fused_ct_material:
            from .proton_preprocess_fusions import (
                sample_packed_ct_material_from_plan_cache,
            )
        sigmas = [self._spot_sigma_mm(segment) for segment in segments]
        fast_cache = build_plan_cache(
            segments,
            ct,
            self.input_grid,
            self.torch_device,
            output_upscale_factor=UPSCALE_FACTOR,
            output_grid=self.grid,
            collapsed_dtype=torch.float32,
            proton_spot_sigmas_mm=sigmas,
            deduplicate_proton_geometry=self.deduplicate_plan_geometry,
        )
        copy_stream = (
            torch.cuda.Stream(device=self.torch_device)
            if self.async_cpu_copy
            else None
        )
        max_roi_elements = max(
            int(np.prod(context.roi_shape))
            for context in fast_cache.backprojection_contexts
        )
        pinned_slot: _PinnedOutputSlot | None = None
        pinned_lease: _PinnedOutputLease | None = None
        if (
            copy_stream is not None
            and self.persistent_pinned_outputs
            and not writer_owned_outputs
        ):
            pinned_slot = self._acquire_pinned_output_slot(max_roi_elements)
            pinned_lease = _PinnedOutputLease(pinned_slot)
            pinned_buffers = pinned_slot.buffers
        elif copy_stream is not None and not writer_owned_outputs:
            pinned_buffers = (
                torch.empty(
                    max_roi_elements, dtype=torch.float32, pin_memory=True
                ),
                torch.empty(
                    max_roi_elements, dtype=torch.float32, pin_memory=True
                ),
            )
        else:
            pinned_buffers = None
        pending: deque[tuple[
            geometry.ProtonSegment,
            torch.Tensor,
            torch.Tensor | None,
            torch.cuda.Event | None,
            tuple[int, ...],
        ]] = deque()

        def finish_copy(item):
            segment, host_buffer, source, event, roi_box = item
            if event is not None:
                event.synchronize()
            xs, xe, ys, ye, zs, ze = roi_box
            shape = (ze - zs, ye - ys, xe - xs)
            dose = host_buffer[: int(np.prod(shape))].view(shape).numpy()
            if writer_owned_outputs:
                dose = dose.view(_WriterOwnedPinnedArray)
            del source
            return segment, dose, roi_box

        next_buffer = 0
        step = max(1, int(batch_size))
        # Retain at most the final ray of the previous model batch. Proton
        # metadata orders energy layers within a ray, so this captures the
        # common cross-batch reuse without accumulating full BEV tensors for a
        # plan. The reference state is packed CT plus fully resolved RSP
        # values; the combined Mamba path retains only packed CT, density and a
        # uint8 material class map.
        cached_ray_key = None
        cached_ray_state = None

        def ray_key(segment: geometry.ProtonSegment):
            return (
                int(segment.beam_idx),
                int(segment.ray_idx),
                tuple(float(value) for value in segment.ray_source),
                tuple(float(value) for value in segment.ray_target),
            )

        for start in range(0, len(segments), step):
            chunk = segments[start : start + step]
            indices = torch.arange(
                start, start + len(chunk), device=self.torch_device
            )
            fluence_batch = preprocess_aperture_batch(
                fast_cache,
                indices,
                output_shape=self.input_grid.shape_dhw,
                output_dtype=(self.amp_dtype or torch.float32),
                spacing_yz=(
                    float(self.input_grid.spacing_dhw[1]),
                    float(self.input_grid.spacing_dhw[2]),
                ),
                sad_mm=float(self.input_grid.sad_mm),
            )

            keys = [ray_key(segment) for segment in chunk]
            states = {}
            if cached_ray_key in keys and cached_ray_state is not None:
                states[cached_ray_key] = cached_ray_state

            missing_keys = []
            representative_indices = []
            for offset, key in enumerate(keys):
                if key not in states and key not in missing_keys:
                    missing_keys.append(key)
                    representative_indices.append(start + offset)
            if missing_keys:
                representative_tensor = torch.tensor(
                    representative_indices,
                    device=self.torch_device,
                    dtype=torch.long,
                )
                if fused_ct_material:
                    unique_ct, unique_density, unique_class_id = (
                        sample_packed_ct_material_from_plan_cache(
                            fast_cache,
                            representative_tensor,
                            output_shape=self.input_grid.shape_dhw,
                            upscale_factor=self.input_upscale_factor,
                            ct_min=float(self._stats["ct_min"]),
                            ct_max=float(self._stats["ct_max"]),
                            ct_hu=self.material_conditioning.ct_hu,
                            ct_rho=self.material_conditioning.ct_rho,
                            class_dens_upper=(
                                self.material_conditioning.class_dens_upper
                            ),
                            output_dtype=(self.amp_dtype or torch.float32),
                        )
                    )
                    for unique_index, key in enumerate(missing_keys):
                        states[key] = (
                            unique_ct[unique_index : unique_index + 1],
                            unique_density[unique_index : unique_index + 1],
                            unique_class_id[unique_index : unique_index + 1],
                        )
                else:
                    unique_ct, unique_hu = preprocess_ct_batch(
                        fast_cache,
                        representative_tensor,
                        output_shape=self.input_grid.shape_dhw,
                        ct_min=float(self._stats["ct_min"]),
                        ct_max=float(self._stats["ct_max"]),
                        output_dtype=(self.amp_dtype or torch.float32),
                    )
                    unique_ct = self._pack_depth_height_phases(
                        unique_ct, self.input_upscale_factor
                    )
                if combined_preprocess and not fused_ct_material:
                    unique_density, unique_class_id = (
                        self.material_conditioning.hu_to_material(unique_hu)
                    )
                    for unique_index, key in enumerate(missing_keys):
                        states[key] = (
                            unique_ct[unique_index : unique_index + 1],
                            unique_density[unique_index : unique_index + 1],
                            unique_class_id[unique_index : unique_index + 1],
                        )
                elif not combined_preprocess:
                    unique_density, unique_electron_ratio, unique_material_i = (
                        self.material_conditioning.hu_to_rsp_state(unique_hu)
                    )
                    for unique_index, key in enumerate(missing_keys):
                        states[key] = (
                            unique_ct[unique_index : unique_index + 1],
                            unique_density[unique_index : unique_index + 1],
                            unique_electron_ratio[unique_index : unique_index + 1],
                            unique_material_i[unique_index : unique_index + 1],
                        )

            energies = torch.tensor(
                [segment.energy_mev for segment in chunk],
                device=self.torch_device,
                dtype=torch.float32,
            )
            model_kwargs = self._model_kwargs(chunk)
            energy_index = model_kwargs.get("energy_index")

            next_index = start + len(chunk)
            if (
                next_index < len(segments)
                and ray_key(segments[next_index]) == keys[-1]
            ):
                cached_ray_key = keys[-1]
                cached_ray_state = states[keys[-1]]
            else:
                cached_ray_key = None
                cached_ray_state = None
            if combined_preprocess:
                unique_batch_keys = []
                unique_index_by_key = {}
                ray_indices_host = []
                for key in keys:
                    unique_index = unique_index_by_key.get(key)
                    if unique_index is None:
                        unique_index = len(unique_batch_keys)
                        unique_index_by_key[key] = unique_index
                        unique_batch_keys.append(key)
                    ray_indices_host.append(unique_index)
                ray_indices = torch.tensor(
                    ray_indices_host,
                    device=self.torch_device,
                    dtype=torch.long,
                )
                ct_by_ray = torch.cat(
                    [states[key][0] for key in unique_batch_keys], dim=0
                )
                density_by_ray = torch.cat(
                    [states[key][1] for key in unique_batch_keys], dim=0
                )
                class_id_by_ray = torch.cat(
                    [states[key][2] for key in unique_batch_keys], dim=0
                )
                class_factors = self.material_conditioning.rsp_class_factors(
                    energies, device=self.torch_device
                )
                range_mm = proton_csda_range_mm(energies).to(
                    device=self.torch_device, dtype=torch.float32
                )
                preprocess_output = build_packed_encoder_input_fused_scan(
                    ct_by_ray,
                    fluence_batch,
                    density_by_ray,
                    class_id_by_ray,
                    ray_indices,
                    class_factors,
                    range_mm,
                    upscale_factor=self.input_upscale_factor,
                    dz_mm=float(self.input_grid.spacing_dhw[0]),
                )
                encoder_input = preprocess_output
                with torch.amp.autocast(
                    "cuda",
                    dtype=self.amp_dtype,
                    enabled=self.amp_dtype is not None,
                ):
                    packed = self.model(
                        encoder_input,
                        energy_index=energy_index,
                        encoder_input=True,
                        batch_size=len(chunk),
                        depth=int(self.grid.nx),
                    )
                    refiner = getattr(self.model, "proton_output_refiner", None)
                    if refiner is not None:
                        phases = self.input_upscale_factor ** 2
                        encoder_5d = encoder_input.view(
                            len(chunk),
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
            else:
                ct_batch = torch.cat([states[key][0] for key in keys], dim=0)
                density_batch = torch.cat(
                    [states[key][1] for key in keys], dim=0
                )
                electron_ratio_batch = torch.cat(
                    [states[key][2] for key in keys], dim=0
                )
                material_i_batch = torch.cat(
                    [states[key][3] for key in keys], dim=0
                )
                conditioning = self._range_conditioning_from_rsp_state(
                    density_batch,
                    electron_ratio_batch,
                    material_i_batch,
                    energies,
                    energy_index=energy_index,
                )
                fluence_batch = self._pack_depth_height_phases(
                    fluence_batch, self.input_upscale_factor
                )
                with torch.amp.autocast(
                    "cuda",
                    dtype=self.amp_dtype,
                    enabled=self.amp_dtype is not None,
                ):
                    packed = self.model(
                        ct_batch,
                        fluence_batch,
                        conditioning=conditioning,
                        **model_kwargs,
                    )
                    refiner = getattr(self.model, "proton_output_refiner", None)
                    if refiner is not None:
                        packed = refiner(
                            packed,
                            conditioning.to(packed.dtype),
                            fluence_batch.to(packed.dtype),
                        )
            packed = self._pad_packed(packed.float())

            for offset, segment in enumerate(chunk):
                cache_index = start + offset
                geometry_index = fast_cache.segment_geometry_index[cache_index]
                context = fast_cache.backprojection_contexts[geometry_index]
                if fused_backprojection:
                    pred_zyx = (
                        bicubic_iir_sample_affine_from_packed_coeff_scaled_zyx_triton(
                            packed[offset : offset + 1],
                            fast_cache.inverse_affines[geometry_index],
                            context.roi_shape,
                            dose_scale=dose_scale,
                            cutoff=self._float32_cutoff_threshold(
                                segment.minimum_cutoff
                            ),
                        )[0]
                    )
                else:
                    pred, valid = (
                        bicubic_iir_sample_affine_from_packed_coeff_triton(
                            packed[offset : offset + 1],
                            fast_cache.inverse_affines[geometry_index],
                            context.roi_shape,
                        )
                    )
                    # Keep inference and physical-dose scaling in FP32, which
                    # is also the MHA payload dtype.
                    pred_zyx = (
                        pred.masked_fill(~valid, 0.0)[0]
                        .mul(dose_scale)
                        .permute(2, 1, 0)
                        .contiguous()
                    )
                    if self._uses_gpu_output_cutoff():
                        self._apply_float32_minimum_cutoff_(
                            pred_zyx, segment.minimum_cutoff
                        )
                roi_elements = pred_zyx.numel()
                if copy_stream is None:
                    yield finish_copy(
                        (segment, pred_zyx.cpu().reshape(-1), None, None, context.roi_box)
                    )
                    continue
                if writer_owned_outputs:
                    host = torch.empty(
                        max_roi_elements,
                        dtype=torch.float32,
                        pin_memory=True,
                    )
                else:
                    assert pinned_buffers is not None
                    host = pinned_buffers[next_buffer]
                    next_buffer = 1 - next_buffer
                event = torch.cuda.Event()
                copy_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(copy_stream):
                    host[:roi_elements].copy_(
                        pred_zyx.reshape(-1), non_blocking=True
                    )
                    event.record(copy_stream)
                pending.append(
                    (segment, host, pred_zyx, event, context.roi_box)
                )
                if len(pending) >= 2:
                    yield finish_copy(pending.popleft())

        while pending:
            yield finish_copy(pending.popleft())
        if pinned_lease is not None:
            pinned_lease.release()

    @torch.inference_mode()
    def predict_segments(
        self,
        segments: Sequence[geometry.ProtonSegment],
        ct: PatientCT,
        *,
        batch_size: int = 1,
    ) -> Iterable[tuple[geometry.ProtonSegment, np.ndarray]]:
        """Yield full FP32 CT volumes, already in the MHA payload dtype."""
        for segment, roi, roi_box in self.predict_segment_rois(
            segments, ct, batch_size=batch_size
        ):
            xs, xe, ys, ye, zs, ze = roi_box
            dose = np.zeros(
                (ct.shape[2], ct.shape[1], ct.shape[0]), dtype=np.float32
            )
            dose[zs:ze, ys:ye, xs:xe] = roi
            yield segment, dose
