"""Shared BEV physics-conditioning utilities for dose model inputs."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProtonBraggResidualHead(nn.Module):
    """Identity-initialized packed-coefficient residual gated by range.

    This is the output head used by the shipped proton checkpoints. The
    correction is localised to where a beamlet actually deposits dose by a
    Gaussian peak plus sigmoid plateau on the remaining-range conditioning
    plane, and the last convolution is zero-initialised so the head is the
    identity at initialisation.
    """

    def __init__(
        self,
        phase_channels: int,
        *,
        peak_sigma_normalized: float = 0.025,
        plateau_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.phase_channels = int(phase_channels)
        if self.phase_channels <= 0:
            raise ValueError("phase_channels must be positive")
        self.peak_sigma_normalized = float(peak_sigma_normalized)
        if self.peak_sigma_normalized <= 0:
            raise ValueError("peak_sigma_normalized must be positive")
        self.plateau_weight = float(plateau_weight)
        hidden = max(16, 2 * self.phase_channels)
        self.residual = nn.Sequential(
            nn.Conv2d(2 * self.phase_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, self.phase_channels, 1),
        )
        # Preserve the primary model at initialization and match the training
        # checkpoint's state-dict layout.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        packed: torch.Tensor,
        conditioning: torch.Tensor,
        fluence: torch.Tensor,
    ) -> torch.Tensor:
        p = self.phase_channels
        if packed.ndim != 5 or packed.shape[2] != p:
            raise ValueError(f"expected packed dose channels={p}")
        if (
            conditioning.shape[:2] != packed.shape[:2]
            or conditioning.shape[3:] != packed.shape[3:]
        ):
            raise ValueError("conditioning must match packed dose geometry")
        if conditioning.shape[2] < 2 * p:
            raise ValueError("expected at least packed WET and remaining-range planes")
        if fluence.shape != packed.shape:
            raise ValueError("packed fluence must match packed dose geometry")

        batch, time, _, height, width = packed.shape
        features = torch.cat((packed, fluence), dim=2).reshape(
            batch * time, 2 * p, height, width
        )
        correction = self.residual(features).reshape(
            batch, time, p, height, width
        )
        remaining = conditioning[:, :, p : 2 * p]
        sigma = self.peak_sigma_normalized
        peak = torch.exp(-0.5 * (remaining / sigma).square())
        plateau = torch.sigmoid(remaining / sigma)
        gate = peak + self.plateau_weight * plateau
        return packed + correction * gate.to(correction.dtype)


def proton_csda_range_mm(energy_mev: float | torch.Tensor) -> torch.Tensor:
    """Fixed empirical proton CSDA range used by the population checkpoints."""
    energy = torch.as_tensor(energy_mev, dtype=torch.float32).clamp_min(1.0)
    return 0.022 * energy.pow(1.77)


class BEVMaterialConditioning(nn.Module):
    """
    Fast GPU material-conditioning from BEV CT HU.

    Input
    -----
    bev_ct_hu:
        Tensor of shape:
            (T, H, W)
            (B, T, H, W)

    Output
    ------
    conditioning:
        Tensor of shape:
            (B, T, C, H, W)

        Default channels:
            0: rho_rel
            1: wed_norm
            2: H mass fraction
            3: C mass fraction
            4: O mass fraction
            5: P mass fraction
            6: Ca mass fraction
            7: compact class_id / 6
    """

    def __init__(
        self,
        dz_mm: float = 2.0,
        wed_norm_mm: float = 400.0,
        include_class: bool = True,
        density_calibration: str = "legacy",
    ) -> None:
        super().__init__()
        self.dz_mm = float(dz_mm)
        self.wed_norm_mm = float(wed_norm_mm)
        self.include_class = bool(include_class)
        if density_calibration not in ("legacy", "g4dcm", "g4dcm_rsp"):
            raise ValueError(
                "density_calibration must be legacy, g4dcm, or g4dcm_rsp"
            )
        self.density_calibration = density_calibration

        ct_hu = torch.tensor(
            [
                -1024,
                -999 if density_calibration in ("g4dcm", "g4dcm_rsp") else -600,
                -200,
                -199,
                -10,
                -9,
                120,
                121,
                3000,
                4000,
            ],
            dtype=torch.float32,
        )
        ct_rho = torch.tensor(
            [
                1.20e-03,
                1.21e-03,
                8.043754e-01,
                8.183035e-01,
                1.006579e00,
                9.966749e-01,
                1.126553e00,
                1.095097e00,
                3.027294e00,
                3.698428e00,
            ],
            dtype=torch.float32,
        )

        class_dens_upper = torch.tensor(
            [
                0.05046545,
                0.92688567,
                0.99263267,
                1.02385869,
                1.12655300,
                1.57500000,
                float("inf"),
            ],
            dtype=torch.float32,
        )

        frac_by_class = torch.tensor(
            [
                [0.000, 0.000, 0.700, 0.300, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                [0.103, 0.105, 0.031, 0.749, 0.002, 0.000, 0.002, 0.003, 0.002, 0.003, 0.000, 0.000],
                [0.114, 0.598, 0.007, 0.278, 0.001, 0.000, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000],
                [0.112, 0.000, 0.000, 0.888, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                [0.102, 0.143, 0.034, 0.710, 0.001, 0.000, 0.002, 0.003, 0.001, 0.004, 0.000, 0.000],
                [0.085, 0.404, 0.058, 0.367, 0.001, 0.001, 0.034, 0.002, 0.002, 0.001, 0.044, 0.001],
                [0.056, 0.235, 0.050, 0.434, 0.001, 0.001, 0.072, 0.003, 0.001, 0.001, 0.146, 0.000],
            ],
            dtype=torch.float32,
        )

        atomic_number = torch.tensor(
            [1, 6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 26],
            dtype=torch.float32,
        )
        atomic_mass = torch.tensor(
            [
                1.00794, 12.0107, 14.0067, 15.9994, 22.98977, 24.305,
                30.97376, 32.065, 35.453, 39.0983, 40.078, 55.845,
            ],
            dtype=torch.float32,
        )
        excitation_energy_ev = torch.tensor(
            [
                19.2, 81.0, 82.0, 95.0, 149.0, 156.0,
                173.0, 180.0, 174.0, 190.0, 191.0, 286.0,
            ],
            dtype=torch.float32,
        )

        self.register_buffer("ct_hu", ct_hu, persistent=False)
        self.register_buffer("ct_rho", ct_rho, persistent=False)
        self.register_buffer("class_dens_upper", class_dens_upper, persistent=False)
        self.register_buffer("frac_by_class", frac_by_class, persistent=False)
        self.register_buffer(
            "element_z_over_a", atomic_number / atomic_mass, persistent=False
        )
        self.register_buffer(
            "excitation_energy_ev", excitation_energy_ev, persistent=False
        )

        # Collapse the composition vectors into the only two material values
        # used by Bethe stopping power.  The previous implementation gathered
        # a (..., 12) elemental-fraction tensor for every BEV voxel and every
        # beamlet.  These seven class constants are algebraically equivalent
        # and make the per-energy path both much smaller and cacheable.
        electron_components = frac_by_class * (atomic_number / atomic_mass)
        electron_per_mass = electron_components.sum(dim=-1).clamp_min(1e-8)
        electron_fraction = electron_components / electron_per_mass[:, None]
        material_i_mev = (
            electron_fraction * excitation_energy_ev.log()
        ).sum(dim=-1).exp() * 1e-6
        water_electron_per_mass = electron_per_mass[3]
        self.register_buffer(
            "class_electron_ratio",
            electron_per_mass / water_electron_per_mass,
            persistent=False,
        )
        self.register_buffer(
            "class_material_i_mev", material_i_mev, persistent=False
        )
        self.register_buffer(
            "water_i_mev", material_i_mev[3], persistent=False
        )

    def hu_to_density(self, hu: torch.Tensor) -> torch.Tensor:
        hu = hu.float().clamp(self.ct_hu[0], self.ct_hu[-1])
        idx = torch.searchsorted(self.ct_hu, hu, right=False)
        idx = idx.clamp(1, self.ct_hu.numel() - 1)

        x0 = self.ct_hu[idx - 1]
        x1 = self.ct_hu[idx]
        y0 = self.ct_rho[idx - 1]
        y1 = self.ct_rho[idx]

        t = (hu - x0) / (x1 - x0).clamp_min(1e-6)
        return y0 + t * (y1 - y0)

    def density_to_class_id(self, density: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(
            self.class_dens_upper, density, right=False
        ).clamp(0, self.frac_by_class.shape[0] - 1)

    def hu_to_rsp(
        self,
        hu: torch.Tensor,
        energy_mev: float | torch.Tensor,
    ) -> torch.Tensor:
        """Approximate proton stopping-power ratio from G4DCM materials."""
        density, class_id = self.hu_to_material(hu)
        return self.material_to_rsp(density, class_id, energy_mev)

    def hu_to_material(
        self, hu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reusable density and compact material-class tensors."""
        density = self.hu_to_density(hu)
        # Seven classes fit in uint8; retaining this for one ray is 8x smaller
        # than the default int64 searchsorted result.
        class_id = self.density_to_class_id(density).to(torch.uint8)
        return density, class_id

    def material_to_rsp(
        self,
        density: torch.Tensor,
        class_id: torch.Tensor,
        energy_mev: float | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate RSP from cached, energy-independent material state."""
        class_index = class_id.to(torch.long)
        material_i_mev = self.class_material_i_mev[class_index]
        electron_ratio = self.class_electron_ratio[class_index]

        energy = torch.as_tensor(
            energy_mev, device=density.device, dtype=torch.float32
        ).clamp_min(1.0)
        # Broadcast a per-batch energy across (T,H,W) when necessary.
        if energy.ndim == 1 and density.ndim == 4:
            energy = energy.view(-1, 1, 1, 1)
        gamma = 1.0 + energy / 938.2720813
        beta2 = (1.0 - gamma.reciprocal().square()).clamp_min(1e-8)
        kinetic_factor = 2.0 * 0.51099895 * beta2 * gamma.square()
        stopping_number = (kinetic_factor / material_i_mev).log() - beta2
        water_stopping_number = (
            kinetic_factor / self.water_i_mev
        ).log() - beta2
        mass_stopping_ratio = electron_ratio * (
            stopping_number / water_stopping_number
        )
        return density * mass_stopping_ratio

    def rsp_class_factors(
        self,
        energy_mev: float | torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return the seven energy-dependent material RSP multipliers.

        Material excitation energy and electron ratio are class constants, so
        evaluating the Bethe logarithms independently at every BEV voxel is
        redundant.  This compact table is algebraically identical to
        :meth:`material_to_rsp`; callers only need to multiply the selected
        class factor by the voxel density.
        """
        resolved_device = (
            self.class_material_i_mev.device if device is None else device
        )
        energy = torch.as_tensor(
            energy_mev, device=resolved_device, dtype=torch.float32
        ).clamp_min(1.0)
        scalar_energy = energy.ndim == 0
        energy = energy.reshape(-1, 1)
        gamma = 1.0 + energy / 938.2720813
        beta2 = (1.0 - gamma.reciprocal().square()).clamp_min(1e-8)
        kinetic_factor = 2.0 * 0.51099895 * beta2 * gamma.square()
        material_i = self.class_material_i_mev.to(resolved_device).reshape(
            1, -1
        )
        electron_ratio = self.class_electron_ratio.to(resolved_device).reshape(
            1, -1
        )
        stopping_number = (kinetic_factor / material_i).log() - beta2
        water_stopping_number = (
            kinetic_factor / self.water_i_mev.to(resolved_device)
        ).log() - beta2
        factors = electron_ratio * (stopping_number / water_stopping_number)
        return factors[0] if scalar_energy else factors

    def rsp_from_material_classes(
        self,
        density: torch.Tensor,
        class_id: torch.Tensor,
        energy_mev: float | torch.Tensor,
    ) -> torch.Tensor:
        """Reference implementation using one seven-value table per energy.

        The CUDA inference path uses the same table through a Triton uint8
        lookup, avoiding the temporary int64 class tensor created here.
        """
        if density.shape != class_id.shape:
            raise ValueError("density and class_id must have identical shapes")
        factors = self.rsp_class_factors(energy_mev, device=density.device)
        indices = class_id.to(torch.long)
        if factors.ndim == 1:
            return density * factors[indices]
        if density.ndim < 1 or factors.shape[0] != density.shape[0]:
            raise ValueError("expected one energy per material-state batch item")
        factor_shape = (factors.shape[0],) + (1,) * (density.ndim - 1) + (
            factors.shape[1],
        )
        expanded = factors.reshape(factor_shape).expand(*density.shape, -1)
        selected = torch.gather(
            expanded, -1, indices.unsqueeze(-1)
        ).squeeze(-1)
        return density * selected

    def hu_to_rsp_state(
        self, hu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve and cache all energy-independent per-voxel RSP values."""
        density, class_id = self.hu_to_material(hu)
        class_index = class_id.to(torch.long)
        return (
            density,
            self.class_electron_ratio[class_index],
            self.class_material_i_mev[class_index],
        )

    def rsp_from_state(
        self,
        density: torch.Tensor,
        electron_ratio: torch.Tensor,
        material_i_mev: torch.Tensor,
        energy_mev: float | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate RSP without repeating material classification/gathers."""
        energy = torch.as_tensor(
            energy_mev, device=density.device, dtype=torch.float32
        ).clamp_min(1.0)
        if energy.ndim == 1 and density.ndim == 4:
            energy = energy.view(-1, 1, 1, 1)
        gamma = 1.0 + energy / 938.2720813
        beta2 = (1.0 - gamma.reciprocal().square()).clamp_min(1e-8)
        kinetic_factor = 2.0 * 0.51099895 * beta2 * gamma.square()
        stopping_number = (kinetic_factor / material_i_mev).log() - beta2
        water_stopping_number = (
            kinetic_factor / self.water_i_mev
        ).log() - beta2
        mass_stopping_ratio = electron_ratio * (
            stopping_number / water_stopping_number
        )
        return density * mass_stopping_ratio

    def fixed_range_wet_from_rsp_state(
        self,
        density: torch.Tensor,
        electron_ratio: torch.Tensor,
        material_i_mev: torch.Tensor,
        energy_mev: float | torch.Tensor,
        *,
        normalisation_mm: float = 300.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Range/WET using a ray's fully resolved material cache."""
        if normalisation_mm <= 0.0:
            raise ValueError("normalisation_mm must be positive")
        if self.density_calibration == "g4dcm_rsp":
            stopping_power_ratio = self.rsp_from_state(
                density, electron_ratio, material_i_mev, energy_mev
            )
        else:
            stopping_power_ratio = density
        wet_mm = torch.cumsum(stopping_power_ratio * self.dz_mm, dim=-3)
        range_mm = proton_csda_range_mm(energy_mev).to(
            device=wet_mm.device, dtype=wet_mm.dtype
        )
        if range_mm.ndim == 1 and wet_mm.ndim == 4:
            range_mm = range_mm.view(-1, 1, 1, 1)
        return (
            wet_mm / normalisation_mm,
            (range_mm - wet_mm) / normalisation_mm,
        )

    def fixed_range_wet_from_material(
        self,
        density: torch.Tensor,
        class_id: torch.Tensor,
        energy_mev: float | torch.Tensor,
        *,
        normalisation_mm: float = 300.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Range/WET from reusable material state for one or more energies."""
        if normalisation_mm <= 0.0:
            raise ValueError("normalisation_mm must be positive")
        if self.density_calibration == "g4dcm_rsp":
            stopping_power_ratio = self.material_to_rsp(
                density, class_id, energy_mev
            )
        else:
            stopping_power_ratio = density
        wet_mm = torch.cumsum(stopping_power_ratio * self.dz_mm, dim=-3)
        range_mm = proton_csda_range_mm(energy_mev).to(
            device=wet_mm.device, dtype=wet_mm.dtype
        )
        if range_mm.ndim == 1 and wet_mm.ndim == 4:
            range_mm = range_mm.view(-1, 1, 1, 1)
        return (
            wet_mm / normalisation_mm,
            (range_mm - wet_mm) / normalisation_mm,
        )

    def fixed_range_wet(
        self,
        bev_ct_hu: torch.Tensor,
        energy_mev: float | torch.Tensor,
        *,
        normalisation_mm: float = 300.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized cumulative WET and remaining CSDA range."""
        density, class_id = self.hu_to_material(bev_ct_hu)
        return self.fixed_range_wet_from_material(
            density,
            class_id,
            energy_mev,
            normalisation_mm=normalisation_mm,
        )

    def forward(self, bev_ct_hu: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if bev_ct_hu.ndim == 3:
            bev_ct_hu = bev_ct_hu.unsqueeze(0)
            squeeze_batch = True
        elif bev_ct_hu.ndim != 4:
            raise ValueError(
                f"Expected (T,H,W) or (B,T,H,W), got {tuple(bev_ct_hu.shape)}"
            )

        rho = self.hu_to_density(bev_ct_hu)
        rho_rel = rho

        class_id = self.density_to_class_id(rho_rel)

        frac = self.frac_by_class[class_id]

        h_frac = frac[..., 0]
        c_frac = frac[..., 1]
        o_frac = frac[..., 3]
        p_frac = frac[..., 6]
        ca_frac = frac[..., 10]

        wed_mm = torch.cumsum(rho_rel * self.dz_mm, dim=1)
        wed_norm = (wed_mm / self.wed_norm_mm).clamp(0.0, 2.0)

        channels = [
            rho_rel,
            wed_norm,
            h_frac,
            c_frac,
            o_frac,
            p_frac,
            ca_frac,
        ]

        if self.include_class:
            channels.append(class_id.float() / 6.0)

        conditioning = torch.stack(channels, dim=2)

        if squeeze_batch:
            conditioning = conditioning.squeeze(0)

        return conditioning

class BEVCompactPhysicsConditioning(BEVMaterialConditioning):
    """
    Compact BEV physics conditioning from CT HU.

    Output channels:
        0: wed_norm
        1: z_eff_norm
        2: air mask
        3: bone mask
    """

    def __init__(
        self,
        dz_mm: float = 2.0,
        wed_norm_mm: float = 400.0,
        z_eff_norm: float = 20.0,
        air_density_threshold: float = 0.05,
        bone_density_threshold: float = 1.126553,
    ) -> None:
        super().__init__(dz_mm=dz_mm, wed_norm_mm=wed_norm_mm, include_class=False)
        self.z_eff_norm = float(z_eff_norm)
        self.air_density_threshold = float(air_density_threshold)
        self.bone_density_threshold = float(bone_density_threshold)
        z_eff_by_class = torch.tensor(
            [7.6, 7.4, 6.3, 7.4, 7.4, 11.0, 13.8],
            dtype=torch.float32,
        )
        self.register_buffer("z_eff_by_class", z_eff_by_class, persistent=False)

    def forward(self, bev_ct_hu: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if bev_ct_hu.ndim == 3:
            bev_ct_hu = bev_ct_hu.unsqueeze(0)
            squeeze_batch = True
        elif bev_ct_hu.ndim != 4:
            raise ValueError(
                f"Expected (T,H,W) or (B,T,H,W), got {tuple(bev_ct_hu.shape)}"
            )

        rho_rel = self.hu_to_density(bev_ct_hu)
        class_id = torch.searchsorted(
            self.class_dens_upper,
            rho_rel,
            right=False,
        ).clamp(0, 6)

        wed_mm = torch.cumsum(rho_rel * self.dz_mm, dim=1)
        wed_norm = (wed_mm / self.wed_norm_mm).clamp(0.0, 2.0)
        z_eff_norm = self.z_eff_by_class[class_id] / self.z_eff_norm
        air = (rho_rel < self.air_density_threshold).float()
        bone = (rho_rel > self.bone_density_threshold).float()

        conditioning = torch.stack([wed_norm, z_eff_norm, air, bone], dim=2)
        if squeeze_batch:
            conditioning = conditioning.squeeze(0)
        return conditioning



def build_bev_encoder_input(
    ct: torch.Tensor,
    proj: torch.Tensor,
    compact_physics: BEVCompactPhysicsConditioning | None,
    *,
    ct_is_normalized: bool,
    ct_min: torch.Tensor,
    ct_max: torch.Tensor,
) -> torch.Tensor:
    if ct.shape != proj.shape:
        raise ValueError(
            f"CT/projection shape mismatch: {tuple(ct.shape)} != "
            f"{tuple(proj.shape)}"
        )
    if ct.ndim == 4:
        base = torch.stack([ct, proj], dim=2)
    elif ct.ndim == 5:
        base = torch.cat([ct, proj], dim=2)
    else:
        raise ValueError(
            "CT/projection must be (B,T,H,W) or (B,T,C_phase,H,W), "
            f"got {tuple(ct.shape)}"
        )
    if compact_physics is None:
        return base
    if ct.ndim != 4:
        raise ValueError(
            "compact physics conditioning does not support phase-packed inputs"
        )
    ct_hu = ct * (ct_max - ct_min) + ct_min if ct_is_normalized else ct
    physics = compact_physics(ct_hu).to(dtype=ct.dtype)
    return torch.cat([base, physics], dim=2)


# -- CNN-Mamba encoder-stem helpers ---------------------------------------
#
# Copied verbatim from DL-segment-dose-calculation/model/physics_conditioning.py.
# The CNN-xLSTM path inlines the equivalent logic in cnn_xlstm.py, so these
# were previously unused here; doserad/nets/cnn_mamba.py imports all three.

def bev_input_shape(x: torch.Tensor) -> tuple[int, int, int, int, int]:
    """Return ``(B,T,C_phase,H,W)`` for scalar or phase-packed BEV inputs."""
    if x.ndim == 4:
        b, t, h, w = x.shape
        return b, t, 1, h, w
    if x.ndim == 5:
        b, t, channels, h, w = x.shape
        return b, t, channels, h, w
    raise ValueError(
        "BEV input must be (B,T,H,W) or (B,T,C_phase,H,W), got "
        f"{tuple(x.shape)}"
    )


def prefix_aperture(x: torch.Tensor) -> torch.Tensor:
    """Return one coarse aperture plane for the optional learned prefix."""
    if x.ndim == 4:
        return x[:, 0:1]
    if x.ndim == 5:
        return x[:, 0].mean(dim=1, keepdim=True)
    raise ValueError(f"unsupported aperture input shape {tuple(x.shape)}")


def append_broadcast_conditioning(
    encoder_input: torch.Tensor,
    conditioning: torch.Tensor | None,
    conditioning_channels: int,
) -> torch.Tensor:
    """Append per-sample/per-depth scalars as spatially broadcast input channels.

    ``encoder_input`` is ``(B,T,C,H,W)``. Conditioning may be ``(B,K)``,
    ``(B,T,K)``, or an already spatial ``(B,T,K,H,W)`` tensor. Keeping this
    optional preserves the exact two-channel photon stem when ``K == 0``.
    """
    expected = int(conditioning_channels)
    if expected == 0:
        return encoder_input
    if conditioning is None:
        raise ValueError(
            f"model requires {expected} conditioning channels, but none were provided"
        )

    b, t, _, h, w = encoder_input.shape
    if conditioning.ndim == 2:
        if conditioning.shape != (b, expected):
            raise ValueError(
                f"expected conditioning (B,K)=({b},{expected}), got "
                f"{tuple(conditioning.shape)}"
            )
        conditioning = conditioning[:, None, :, None, None]
    elif conditioning.ndim == 3:
        if conditioning.shape != (b, t, expected):
            raise ValueError(
                f"expected conditioning (B,T,K)=({b},{t},{expected}), got "
                f"{tuple(conditioning.shape)}"
            )
        conditioning = conditioning[:, :, :, None, None]
    elif conditioning.ndim == 5:
        if conditioning.shape[:3] != (b, t, expected):
            raise ValueError(
                f"expected conditioning prefix (B,T,K)=({b},{t},{expected}), got "
                f"{tuple(conditioning.shape)}"
            )
        if conditioning.shape[-2:] not in ((1, 1), (h, w)):
            raise ValueError(
                f"conditioning spatial shape must be (1,1) or ({h},{w}), got "
                f"{tuple(conditioning.shape[-2:])}"
            )
    else:
        raise ValueError(
            "conditioning must have shape (B,K), (B,T,K), or (B,T,K,H,W), "
            f"got {tuple(conditioning.shape)}"
        )
    conditioning = conditioning.to(
        device=encoder_input.device, dtype=encoder_input.dtype
    ).expand(b, t, expected, h, w)
    return torch.cat((encoder_input, conditioning), dim=2)
