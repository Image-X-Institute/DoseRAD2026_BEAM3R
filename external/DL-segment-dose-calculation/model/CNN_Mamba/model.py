from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba2 import Mamba2  # mamba-ssm>=2.x

_MODEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MODEL_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_ROOT)

from bev_pixel_shuffle import PixelShuffleDepthHeight, reshape_depth_height_shuffled

DecoderKind = Literal["conv_transpose", "bilinear_conv"]
MambaCoreKind = Literal["mamba2", "mamba3"]
PrefixMode = Literal["constant", "aperture"]
ScalerPool = Literal["avg", "max"]

# Mamba3 default expand=2 in mamba_ssm; SISO Triton kernels fail with nheads=1 (d=32, headdim=64).
_MAMBA3_EXPAND = 2
_MIN_MAMBA3_NHEADS = 2


def _bev_input_shape(x: torch.Tensor) -> tuple[int, int, int, int, int]:
    """Return ``(B,T,C_phase,H,W)`` for scalar or phase-packed BEV inputs.

    Mirrors the CNN_xLSTM helper of the same name; kept local rather than
    imported so the two model packages stay independent.
    """
    if x.ndim == 4:
        batch, depth, height, width = x.shape
        return batch, depth, 1, height, width
    if x.ndim == 5:
        return tuple(int(value) for value in x.shape)
    raise ValueError(
        "BEV input must be (B,T,H,W) or (B,T,C_phase,H,W), got "
        f"{tuple(x.shape)}"
    )


def _append_broadcast_conditioning(
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


def _prefix_aperture(x: torch.Tensor) -> torch.Tensor:
    """One coarse aperture plane for the learned prefix, phase-packed or not."""
    if x.ndim == 4:
        return x[:, 0:1]
    if x.ndim == 5:
        # collapse the phase axis; the prefix gate only needs a coarse aperture
        return x[:, 0].mean(dim=1, keepdim=True)
    raise ValueError(f"unsupported aperture input shape {tuple(x.shape)}")


def _scaled_capacity_width(width: int, scale: float) -> int:
    """Scale a channel width to the nearest positive multiple of eight."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"model_capacity_scale must be finite and positive, got {scale}")
    if scale == 1.0:
        return int(width)
    return max(8, int(math.floor(float(width) * scale / 8.0 + 0.5)) * 8)


def resolve_mamba_dose_arch_config(
    arch: MambaDoseArchConfig,
) -> MambaDoseArchConfig:
    """Return the effective architecture after whole-model capacity scaling."""
    scale = float(arch.model_capacity_scale)
    return replace(
        arch,
        d=_scaled_capacity_width(arch.d, scale),
        prefix_gate_hidden=_scaled_capacity_width(arch.prefix_gate_hidden, scale),
    )


def validate_mamba_dose_arch_config(arch: MambaDoseArchConfig) -> None:
    """Validate bottleneck width and Mamba3 layout before building the model."""
    if arch.d <= 0:
        raise ValueError(f"Mamba bottleneck width d must be positive, got d={arch.d}")

    if arch.input_phase_channels <= 0:
        raise ValueError(
            "input_phase_channels must be positive, got "
            f"{arch.input_phase_channels}"
        )
    if arch.conditioning_channels < 0:
        raise ValueError(
            "conditioning_channels must be non-negative, got "
            f"{arch.conditioning_channels}"
        )
    if arch.energy_token_levels < 0:
        raise ValueError(
            "energy_token_levels must be non-negative, got "
            f"{arch.energy_token_levels}"
        )
    if arch.prefix_mode not in ("constant", "aperture"):
        raise ValueError(
            f"prefix_mode must be 'constant' or 'aperture', got {arch.prefix_mode!r}"
        )
    if arch.scaler_pool not in ("avg", "max"):
        raise ValueError(f"scaler_pool must be 'avg' or 'max', got {arch.scaler_pool!r}")
    if not math.isfinite(arch.model_capacity_scale) or arch.model_capacity_scale <= 0:
        raise ValueError(
            "model_capacity_scale must be finite and positive, "
            f"got {arch.model_capacity_scale}"
        )

    if arch.mamba_core != "mamba3":
        return

    headdim = int(arch.mamba3_headdim)
    if headdim <= 0:
        raise ValueError(f"mamba3_headdim must be positive, got {headdim}")

    d_inner = arch.d * _MAMBA3_EXPAND
    if d_inner % headdim != 0:
        raise ValueError(
            f"Mamba3 requires (d * expand) divisible by mamba3_headdim: "
            f"d={arch.d}, expand={_MAMBA3_EXPAND} → d_inner={d_inner}, "
            f"mamba3_headdim={headdim} (remainder {d_inner % headdim}). "
            f"Try --mamba-d {headdim} or --mamba3-headdim {d_inner // 2}."
        )

    nheads = d_inner // headdim
    if nheads < _MIN_MAMBA3_NHEADS:
        raise ValueError(
            f"Mamba3 SISO Triton kernels require at least {_MIN_MAMBA3_NHEADS} heads "
            f"(d_inner // mamba3_headdim), but d={arch.d}, mamba3_headdim={headdim} "
            f"→ d_inner={d_inner}, nheads={nheads}. "
            f"Use --mamba-d {headdim} (nheads=2 with default headdim) or "
            f"--mamba3-headdim {max(d_inner // _MIN_MAMBA3_NHEADS, 1)} "
            f"(nheads={_MIN_MAMBA3_NHEADS} with current d)."
        )


@dataclass(frozen=True)
class MambaDoseArchConfig:
    """Architecture hyperparameters for ``CNN_Mamba2_L2_SpatialMix``."""

    decoder: DecoderKind = "conv_transpose"
    input_h: int = 200
    input_w: int = 200
    d: int = 32
    layers: int = 2
    dropout: float = 0.05
    use_window_attention: bool = False
    window_attn_heads: int = 4
    window_size: int = 5
    output_softplus: bool = False
    output_relu: bool = False
    decoder_final_relu: bool = False
    depth_height_pixel_shuffle: bool = False
    return_packed_dh_coefficients: bool = False
    spatial_mid: bool = False
    use_temporal_conv: bool = False
    temporal_kernel_size: int = 5
    mamba_core: MambaCoreKind = "mamba2"
    mamba3_headdim: int = 64
    mamba3_chunk_size: int = 64
    mamba3_is_mimo: bool = False
    mamba3_mimo_rank: int = 4
    n_prefix: int = 0
    prefix_mode: PrefixMode = "constant"
    prefix_gate_hidden: int = 16
    use_scaler: bool = False
    scaler_pool: ScalerPool = "max"
    scaler_layers: int = 3
    model_capacity_scale: float = 1.0
    # ReZero-gated encoder→decoder skips; the bottleneck is stride-8 laterally,
    # so without these the decoder rebuilds every lateral edge from a 25×25 code.
    unet_skips: bool = False
    # ReZero-gated FiLM on depth. The 2D blocks and the decoder are shared across
    # all depth slices and otherwise receive no depth coordinate at all.
    depth_film: bool = False
    film_n_freq: int = 8
    film_hidden: int = 32
    film_sad_mm: float = 1000.0
    film_depth_spacing_mm: float = 2.0
    # Phase-packed BEV inputs from --otf-input-upscale-factor r carry r**2 phases
    # per modality, so the encoder sees 2 * input_phase_channels planes. 1 is the
    # scalar (B,T,H,W) case.
    input_phase_channels: int = 1
    bidirectional_mamba: bool = False
    # Extra spatially broadcast conditioning planes (proton WET / remaining
    # range: ``2 * input_phase_channels``); 0 keeps the photon stem exact.
    conditioning_channels: int = 0
    # Optional discrete proton-energy embedding prepended as one additional
    # temporal token before every Mamba pass. Zero preserves photon models.
    energy_token_levels: int = 0

    def to_json_dict(self) -> dict:
        """JSON-serialisable snapshot (e.g. for ``config_hyper``)."""
        return asdict(self)


def _decoder_stage_act(*, decoder_final_relu: bool, final_stage: bool) -> nn.Module:
    """Activation after a decoder upsampling stage; optional ReLU on the last stage only."""
    if decoder_final_relu and final_stage:
        return nn.ReLU(inplace=True)
    return nn.LeakyReLU(0.2, inplace=True)


def _decoder_conv_transpose_stack(
    *,
    decoder_final_relu: bool = False,
    depth_height_pixel_shuffle: bool = False,
    encoder_channels: int = 64,
    capacity_scale: float = 1.0,
) -> nn.Sequential:
    """Original decoder: three ×2 upsamples + head (matches encoder geometry for 200² input)."""
    c1, c2, c3 = (
        _scaled_capacity_width(c, capacity_scale) for c in (32, 16, 16)
    )
    head: nn.Module
    if depth_height_pixel_shuffle:
        head = nn.Conv2d(c3, 4, kernel_size=1, stride=1, padding=0)
    else:
        head = nn.ConvTranspose2d(c3, 1, 1, 1, 0)
    return nn.Sequential(
        nn.ConvTranspose2d(encoder_channels, c1, 4, 2, 1),
        _decoder_stage_act(decoder_final_relu=decoder_final_relu, final_stage=False),
        nn.ConvTranspose2d(c1, c2, 4, 2, 1),
        _decoder_stage_act(decoder_final_relu=decoder_final_relu, final_stage=False),
        nn.ConvTranspose2d(c2, c3, 4, 2, 1),
        _decoder_stage_act(decoder_final_relu=decoder_final_relu, final_stage=True),
        head,
    )


def _decoder_bilinear_conv_stack(
    *,
    decoder_final_relu: bool = False,
    depth_height_pixel_shuffle: bool = False,
    encoder_channels: int = 64,
    capacity_scale: float = 1.0,
) -> nn.Sequential:
    """Three ×2 bilinear upsamples + 3×3 conv refinements; same channel schedule as transposed path."""
    c1, c2, c3 = (
        _scaled_capacity_width(c, capacity_scale) for c in (32, 16, 16)
    )
    head: nn.Module
    if depth_height_pixel_shuffle:
        head = nn.Conv2d(c3, 4, kernel_size=1, stride=1, padding=0)
    else:
        head = nn.Conv2d(c3, 1, kernel_size=1, stride=1, padding=0)
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(encoder_channels, c1, kernel_size=3, stride=1, padding=1),
        _decoder_stage_act(decoder_final_relu=decoder_final_relu, final_stage=False),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
        _decoder_stage_act(decoder_final_relu=decoder_final_relu, final_stage=False),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
        _decoder_stage_act(decoder_final_relu=decoder_final_relu, final_stage=True),
        head,
    )


# Encoder stage outputs tapped for --unet-skips, as indices into ``self.encoder``.
# The encoder is 200 →(0,1) 100 →(2,3) 100 →(4,5) 50 →(6,7) 25, so index 3 is the
# half-resolution tap and index 5 the quarter-resolution one.
_ENCODER_TAP_INDEX_HALF = 3
_ENCODER_TAP_INDEX_QUARTER = 5


def _decoder_injection_indices(kind: DecoderKind) -> tuple[int, int]:
    """Indices into the decoder ``Sequential`` after its first two upsample stages.

    Injection happens after the stage activation, so the returned indices point at
    the activation layers: the first output is quarter resolution (matching the
    encoder's index-5 tap) and the second is half resolution (matching index 3).
    """
    if kind == "conv_transpose":
        return 1, 3
    if kind == "bilinear_conv":
        return 2, 5
    raise ValueError(f"Unknown decoder kind: {kind!r}")


class DepthFiLM(nn.Module):
    """ReZero-gated FiLM conditioning on physical depth along the beam axis.

    ``spatial_pre``/``spatial_mid``/``spatial_post`` and the whole decoder are
    applied per depth slice with weights shared across all of them and no depth
    input, so they cannot express anything depth-dependent -- including the
    inverse-square falloff, which varies about 5x across the BEV slab, and the
    lateral shear between the parallel BEV lattice and the divergent ray fan.
    """

    def __init__(
        self,
        channels: int,
        *,
        n_freq: int = 8,
        hidden: int = 32,
        sad_mm: float = 1000.0,
        spacing_mm: float = 2.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.n_freq = int(n_freq)
        self.sad_mm = float(sad_mm)
        self.spacing_mm = float(spacing_mm)
        self.mlp = nn.Sequential(
            nn.Linear(2 + 2 * self.n_freq, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.channels),
        )
        # ReZero: the projection keeps its default init so it receives gradient
        # from step one, and the scalar gate starts at zero so the module is
        # exactly identity at initialisation. Zero-initialising both would leave
        # the projection with no gradient path and the module permanently dead.
        self.gate = nn.Parameter(torch.zeros(1))

    def _depth_features(self, depth: int, device: torch.device) -> torch.Tensor:
        """(depth, 2 + 2*n_freq) features from source distance and Fourier position."""
        t = torch.arange(depth, device=device, dtype=torch.float32)
        # Matches build_bev_index_grid: depth index 0 sits at the plane origin,
        # which inference/pipeline.py places at sad - 0.5 * depth * spacing.
        s_mm = (self.sad_mm - 0.5 * depth * self.spacing_mm) + t * self.spacing_mm
        s_norm = s_mm / self.sad_mm
        inv_square = (self.sad_mm / s_mm) ** 2
        u = (t + 0.5) / float(depth)
        freqs = torch.arange(self.n_freq, device=device, dtype=torch.float32)
        angles = u[:, None] * (2.0**freqs)[None, :] * math.pi
        return torch.cat(
            (s_norm[:, None], inv_square[:, None], angles.sin(), angles.cos()), dim=1
        )

    def forward(self, x: torch.Tensor, batch: int, depth: int) -> torch.Tensor:
        # x: (batch*depth, C, H, W), laid out with index batch_i * depth + depth_i
        gamma, beta = self.mlp(self._depth_features(depth, x.device)).chunk(2, dim=-1)
        gamma = gamma.repeat(batch, 1)[:, :, None, None].to(x.dtype)
        beta = beta.repeat(batch, 1)[:, :, None, None].to(x.dtype)
        return x + self.gate.to(x.dtype) * (x * gamma + beta)


def build_mamba_dose_decoder(
    kind: DecoderKind,
    *,
    decoder_final_relu: bool = False,
    depth_height_pixel_shuffle: bool = False,
    encoder_channels: int = 64,
    capacity_scale: float = 1.0,
) -> nn.Module:
    if kind == "conv_transpose":
        return _decoder_conv_transpose_stack(
            decoder_final_relu=decoder_final_relu,
            depth_height_pixel_shuffle=depth_height_pixel_shuffle,
            encoder_channels=encoder_channels,
            capacity_scale=capacity_scale,
        )
    if kind == "bilinear_conv":
        return _decoder_bilinear_conv_stack(
            decoder_final_relu=decoder_final_relu,
            depth_height_pixel_shuffle=depth_height_pixel_shuffle,
            encoder_channels=encoder_channels,
            capacity_scale=capacity_scale,
        )
    raise ValueError(f"Unknown decoder kind: {kind!r}")


class SimpleWindowAttention(nn.Module):
    """Spatial self-attention on slice-wise feature maps (after Mamba temporal mixing)."""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 5) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        b, c, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)  # (B*T, H*W, C)

        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = tokens + attn_out
        tokens = self.norm1(tokens)

        tokens = tokens + self.mlp(tokens)
        tokens = self.norm2(tokens)

        return tokens.transpose(1, 2).view(b, c, h, w)


def _build_mamba_core(arch: MambaDoseArchConfig) -> nn.Module:
    validate_mamba_dose_arch_config(arch)
    if arch.mamba_core == "mamba2":
        # Mamba2 normally applies the SSM to all ``2 * d`` inner channels,
        # which requires divisibility by its fixed 64-wide heads. A scaled
        # width such as d=48 has 96 inner channels: keep one 64-wide SSM head
        # and let Mamba2's supported gated-MLP branch carry the remaining 32.
        d_inner = _MAMBA3_EXPAND * arch.d
        d_ssm = (d_inner // 64) * 64
        if d_ssm <= 0:
            raise ValueError(
                f"Mamba2 requires 2 * d >= 64, got d={arch.d}"
            )
        return Mamba2(
            d_model=arch.d,
            d_ssm=None if d_ssm == d_inner else d_ssm,
        )
    if arch.mamba_core == "mamba3":
        from mamba_ssm.modules.mamba3 import Mamba3  # mamba-ssm>=2.3.2

        if arch.mamba3_is_mimo:
            raise ValueError(
                "Mamba3 MIMO (mamba3_is_mimo=True) requires optional TileLang kernels; "
                "use mamba3_is_mimo=False for training."
            )
        return Mamba3(
            d_model=arch.d,
            headdim=arch.mamba3_headdim,
            chunk_size=arch.mamba3_chunk_size,
            is_mimo=arch.mamba3_is_mimo,
            mimo_rank=arch.mamba3_mimo_rank,
        )
    raise ValueError(f"Unknown mamba_core: {arch.mamba_core!r}")


class MambaBlockDRes(nn.Module):
    """Pre-Norm + Mamba2/Mamba3 + Dropout + Res"""

    def __init__(
        self,
        arch: MambaDoseArchConfig,
        dropout: float = 0.05,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(arch.d)
        self.core = _build_mamba_core(arch)
        self.drop = nn.Dropout(dropout)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, T, d)
        y = self.core(self.norm(x))
        return x + self.res_scale * self.drop(y)


class Mamba2BlockDRes(MambaBlockDRes):
    """Backward-compatible alias: two-arg constructor with Mamba2 core only."""

    def __init__(self, d: int, dropout: float = 0.05, res_scale: float = 1.0) -> None:
        super().__init__(MambaDoseArchConfig(d=d, mamba_core="mamba2"), dropout=dropout, res_scale=res_scale)


def _make_spatial_mid_block(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels),
        nn.Conv2d(channels, channels, kernel_size=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class TemporalConv1d(nn.Module):
    """Non-causal depthwise 1D conv along depth ``T`` at each spatial location."""

    def __init__(self, channels: int, kernel_size: int = 5, padding_mode: str = "replicate") -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for centred (non-causal) padding")
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        seq = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, c, t)
        seq = self.conv(seq)
        return seq.view(b, h, w, c, t).permute(0, 4, 3, 1, 2)


class SliceScalePredictor(nn.Module):
    """Per-slice scalar gate in (0, 2) applied before the decoder.

    pool: 'max' or 'avg'.  n_layers: 1, 2, or 3.
    Final linear layer zero-initialised so the gate starts at 1.
    Input shape: (N, C, H, W). Output shape: (N, 1, 1, 1).
    """

    def __init__(self, channels: int, pool: str = "max", n_layers: int = 3) -> None:
        super().__init__()
        self.pool_layer = (nn.AdaptiveMaxPool2d(1) if pool == "max"
                           else nn.AdaptiveAvgPool2d(1))
        h1, h2 = max(channels // 4, 4), max(channels // 8, 4)
        sizes = {1: [channels, 1], 2: [channels, h2, 1], 3: [channels, h1, h2, 1]}[n_layers]
        layers: list[nn.Module] = [nn.Flatten(1)]
        for i in range(len(sizes) - 1):
            fc = nn.Linear(sizes[i], sizes[i + 1])
            if i == len(sizes) - 2:
                nn.init.zeros_(fc.weight)
                nn.init.zeros_(fc.bias)
            layers.append(fc)
            if i < len(sizes) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(self.mlp(self.pool_layer(x))) + 1.0).view(-1, 1, 1, 1)


class ApertureGateEncoder(nn.Module):
    """CNN encoder mapping the beam aperture (first BEV depth slice) to a
    per-channel multiplicative gate in (0, 2) for each prefix slot.

    Final conv zero-initialised so the gate starts at 1 everywhere.
    Input shape: (N, 1, H, W). Output shape: (N, n_prefix, channels, h_enc, w_enc).
    """

    def __init__(self, n_prefix: int, channels: int, h_enc: int, w_enc: int, *, hidden: int = 16) -> None:
        super().__init__()
        self.n_prefix = int(n_prefix)
        self.channels = int(channels)
        self.h_enc = int(h_enc)
        self.w_enc = int(w_enc)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden * 2, n_prefix * channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, aperture: torch.Tensor) -> torch.Tensor:
        logits = self.net(aperture)
        if logits.shape[-2:] != (self.h_enc, self.w_enc):
            logits = F.interpolate(logits, size=(self.h_enc, self.w_enc), mode="bilinear", align_corners=False)
        b = logits.shape[0]
        logits = logits.view(b, self.n_prefix, self.channels, self.h_enc, self.w_enc)
        return 2.0 * torch.sigmoid(logits)


class SmoothApertureGatedPrefix(nn.Module):
    """Learned low-res spatial prefix, bicubic-upsampled and multiplicatively
    gated by the beam aperture, prepended before each Mamba pass.

    Output shape: (B, n_prefix, channels, h_enc, w_enc).
    """

    def __init__(
        self, n_prefix: int, channels: int, h_enc: int, w_enc: int, *,
        lowres_hw: tuple[int, int] = (9, 9), gate_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.n_prefix = int(n_prefix)
        self.channels = int(channels)
        self.h_enc = int(h_enc)
        self.w_enc = int(w_enc)
        self.lowres_hw = tuple(lowres_hw)
        h0, w0 = self.lowres_hw
        self.prefix_lowres = nn.Parameter(torch.empty(1, n_prefix, channels, h0, w0))
        nn.init.normal_(self.prefix_lowres, mean=0.0, std=1e-3)
        self.gate = ApertureGateEncoder(
            n_prefix=n_prefix, channels=channels, h_enc=h_enc, w_enc=w_enc, hidden=gate_hidden,
        )

    def forward(self, aperture: torch.Tensor) -> torch.Tensor:
        b = aperture.shape[0]
        p = self.prefix_lowres.view(1, self.n_prefix * self.channels, self.lowres_hw[0], self.lowres_hw[1])
        p = F.interpolate(p, size=(self.h_enc, self.w_enc), mode="bicubic", align_corners=False)
        p = p.view(1, self.n_prefix, self.channels, self.h_enc, self.w_enc).expand(b, -1, -1, -1, -1)
        return p * self.gate(aperture)


class CNN_Mamba2_L2_SpatialMix(nn.Module):
    """Encoder–Mamba–decoder dose model; decoder variant chosen via ``MambaDoseArchConfig``."""

    def __init__(
        self,
        arch: MambaDoseArchConfig | None = None,
        *,
        use_channels_last: bool = False,
    ) -> None:
        super().__init__()
        self.base_arch = arch if arch is not None else MambaDoseArchConfig()
        self.arch = resolve_mamba_dose_arch_config(self.base_arch)
        validate_mamba_dose_arch_config(self.arch)
        self.use_channels_last = bool(use_channels_last)
        input_h, input_w = self.arch.input_h, self.arch.input_w
        d, layers, dropout = self.arch.d, self.arch.layers, self.arch.dropout
        capacity_scale = float(self.arch.model_capacity_scale)
        c1, c2, encoder_channels = (
            _scaled_capacity_width(c, capacity_scale) for c in (16, 32, 64)
        )

        self.input_h, self.input_w = input_h, input_w
        self.d = d
        self.encoder_channels = encoder_channels

        # CT and aperture, each carrying input_phase_channels phase planes
        self.input_phase_channels = int(self.arch.input_phase_channels)
        self.conditioning_channels = int(self.arch.conditioning_channels)
        # CT and aperture each contribute ``input_phase_channels`` planes;
        # proton range conditioning is appended as broadcast channels.
        encoder_in = 2 * self.input_phase_channels + self.conditioning_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_in, c1, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, encoder_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(encoder_channels, encoder_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # depthwise 7×7 + pointwise 1×1
        self.spatial_pre = nn.Sequential(
            nn.Conv2d(
                encoder_channels,
                encoder_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=encoder_channels,
            ),
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.in_proj = (
            nn.Linear(encoder_channels, d)
            if d != encoder_channels
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [MambaBlockDRes(self.arch, dropout=dropout) for _ in range(layers)]
        )
        if self.arch.bidirectional_mamba:
            self.blocks_bwd = nn.ModuleList(
                [MambaBlockDRes(self.arch, dropout=dropout) for _ in range(layers)]
            )
            # Zero init makes the bidirectional model *exactly* the unidirectional
            # one at step 0, so an existing checkpoint warm-starts cleanly and the
            # backward branch is admitted only as fast as it earns its way in.
            self.register_parameter("bwd_gate", nn.Parameter(torch.zeros(1)))
        else:
            self.blocks_bwd = None
            self.register_parameter("bwd_gate", None)
        self.out_proj = (
            nn.Linear(d, encoder_channels)
            if d != encoder_channels
            else nn.Identity()
        )

        if self.arch.spatial_mid:
            self.spatial_mid = _make_spatial_mid_block(encoder_channels)
            self.blocks2 = nn.ModuleList(
                [MambaBlockDRes(self.arch, dropout=dropout) for _ in range(layers)]
            )
            if self.arch.bidirectional_mamba:
                self.blocks2_bwd = nn.ModuleList(
                    [MambaBlockDRes(self.arch, dropout=dropout) for _ in range(layers)]
                )
                self.register_parameter("bwd_gate2", nn.Parameter(torch.zeros(1)))
            else:
                self.blocks2_bwd = None
                self.register_parameter("bwd_gate2", None)
        else:
            self.spatial_mid = None
            self.blocks2 = None
            self.blocks2_bwd = None
            self.register_parameter("bwd_gate2", None)

        self.temporal_conv = (
            TemporalConv1d(encoder_channels, kernel_size=self.arch.temporal_kernel_size)
            if self.arch.use_temporal_conv
            else None
        )

        self.window_attn: SimpleWindowAttention | None
        if self.arch.use_window_attention:
            self.window_attn = SimpleWindowAttention(
                dim=encoder_channels,
                num_heads=self.arch.window_attn_heads,
                window_size=self.arch.window_size,
            )
        else:
            self.window_attn = None

        # depthwise 7×7 + pointwise 1×1
        self.spatial_post = nn.Sequential(
            nn.Conv2d(
                encoder_channels,
                encoder_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=encoder_channels,
            ),
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = build_mamba_dose_decoder(
            self.arch.decoder,
            decoder_final_relu=self.arch.decoder_final_relu,
            depth_height_pixel_shuffle=self.arch.depth_height_pixel_shuffle,
            encoder_channels=encoder_channels,
            capacity_scale=capacity_scale,
        )
        dec_c1, dec_c2 = (
            _scaled_capacity_width(c, capacity_scale) for c in (32, 16)
        )
        self.unet_skips = bool(self.arch.unet_skips)
        self.decoder_inject = _decoder_injection_indices(self.arch.decoder)
        if self.unet_skips:
            # Quarter-resolution tap (encoder index 5) into the first decoder
            # stage, half-resolution tap (index 3) into the second. Additive and
            # ReZero-gated rather than concatenated, so no existing layer changes
            # its in_channels and the parent state dict still loads unmodified.
            self.skip_proj_quarter = nn.Conv2d(encoder_channels, dec_c1, kernel_size=1)
            self.skip_proj_half = nn.Conv2d(c2, dec_c2, kernel_size=1)
            self.skip_gate_quarter = nn.Parameter(torch.zeros(1))
            self.skip_gate_half = nn.Parameter(torch.zeros(1))
        else:
            self.skip_proj_quarter = None
            self.skip_proj_half = None
            self.register_parameter("skip_gate_quarter", None)
            self.register_parameter("skip_gate_half", None)

        film_kwargs = dict(
            n_freq=self.arch.film_n_freq,
            hidden=self.arch.film_hidden,
            sad_mm=self.arch.film_sad_mm,
            spacing_mm=self.arch.film_depth_spacing_mm,
        )
        if self.arch.depth_film:
            self.film_bottleneck = DepthFiLM(encoder_channels, **film_kwargs)
            self.film_decoder = DepthFiLM(dec_c1, **film_kwargs)
        else:
            self.film_bottleneck = None
            self.film_decoder = None

        self.pixel_shuffle_dh: PixelShuffleDepthHeight | None
        if self.arch.depth_height_pixel_shuffle:
            self.pixel_shuffle_dh = PixelShuffleDepthHeight(2)
        else:
            self.pixel_shuffle_dh = None
        if self.arch.return_packed_dh_coefficients and not self.arch.depth_height_pixel_shuffle:
            raise ValueError(
                "return_packed_dh_coefficients requires depth_height_pixel_shuffle"
            )
        if self.arch.output_softplus and self.arch.output_relu:
            raise ValueError("output_softplus and output_relu are mutually exclusive")
        if self.arch.output_softplus:
            self.output_activation = nn.Softplus()
        elif self.arch.output_relu:
            self.output_activation = nn.ReLU()
        else:
            self.output_activation = nn.Identity()

        with torch.no_grad():
            # must match the encoder's input width, which is 2 * phase channels
            _dummy = torch.zeros(1, encoder_in, input_h, input_w)
            _, _, _h_enc, _w_enc = self.encoder(_dummy).shape

        self.prefix_mode = self.arch.prefix_mode
        self.n_prefix = int(self.arch.n_prefix)
        self.prefix_enc = None
        self.prefix_module = None
        if self.n_prefix > 0:
            if self.prefix_mode == "constant":
                self.prefix_enc = nn.Parameter(
                    torch.zeros(1, self.n_prefix, encoder_channels, 1, 1)
                )
            else:
                self.prefix_module = SmoothApertureGatedPrefix(
                    n_prefix=self.n_prefix,
                    channels=encoder_channels,
                    h_enc=int(_h_enc),
                    w_enc=int(_w_enc),
                    gate_hidden=self.arch.prefix_gate_hidden,
                )

        self.energy_token_levels = int(self.arch.energy_token_levels)
        self.energy_token_embedding = (
            nn.Embedding(self.energy_token_levels, encoder_channels)
            if self.energy_token_levels > 0
            else None
        )
        if self.energy_token_embedding is not None:
            nn.init.normal_(self.energy_token_embedding.weight, mean=0.0, std=0.02)

        self.use_scaler = bool(self.arch.use_scaler)
        self.slice_scaler = (
            SliceScalePredictor(
                encoder_channels,
                pool=self.arch.scaler_pool,
                n_layers=self.arch.scaler_layers,
            )
            if self.use_scaler
            else None
        )

    def _compute_prefix(
        self,
        x2: torch.Tensor | None,
        batch: int,
        channels: int,
        h_enc: int,
        w_enc: int,
        energy_index: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Prefix frames prepended before each Mamba pass, or ``None`` when disabled.

        An energy-token model prepends one further token carrying the beamlet's
        discrete energy embedding, so this can return a prefix even when
        ``n_prefix`` is zero.
        """
        prefix = None
        if self.n_prefix > 0:
            if self.prefix_mode == "constant":
                prefix = self.prefix_enc.expand(batch, -1, channels, h_enc, w_enc)
            else:
                if x2 is None:
                    raise ValueError(
                        "an aperture-derived Mamba prefix requires the aperture input"
                    )
                prefix = self.prefix_module(_prefix_aperture(x2))
        if self.energy_token_embedding is None:
            return prefix
        if energy_index is None:
            raise ValueError("energy-token model requires an energy index")
        index = energy_index.reshape(-1).to(
            device=self.energy_token_embedding.weight.device, dtype=torch.long
        )
        if index.numel() != batch:
            raise ValueError(
                f"expected one energy index per batch item, got {index.numel()}"
            )
        if index.device.type == "cpu" and bool(
            ((index < 0) | (index >= self.energy_token_levels)).any()
        ):
            raise ValueError("energy index is outside the embedding table")
        token = self.energy_token_embedding(index).view(
            batch, 1, channels, 1, 1
        ).expand(-1, -1, -1, h_enc, w_enc)
        return token if prefix is None else torch.cat((token, prefix), dim=1)

    def _mamba_scan(
        self,
        x_5d: torch.Tensor,
        blocks: nn.ModuleList,
        *,
        prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run in_proj → Mamba blocks → out_proj along depth at each spatial location."""
        b, t, c, hc, wc = x_5d.shape
        n_prefix = 0
        if prefix is not None:
            n_prefix = prefix.shape[1]
            x_5d = torch.cat([prefix, x_5d], dim=1)
        t_full = t + n_prefix
        seq = x_5d.permute(0, 3, 4, 1, 2).reshape(b * hc * wc, t_full, c)
        z = self.in_proj(seq)
        for blk in blocks:
            z = blk(z)
        seq = self.out_proj(z)
        out = seq.reshape(b, hc, wc, t_full, c).permute(0, 3, 4, 1, 2)
        if n_prefix > 0:
            out = out[:, n_prefix:]
        return out

    def _run_mamba_temporal(
        self,
        x_5d: torch.Tensor,
        blocks: nn.ModuleList,
        *,
        prefix: torch.Tensor | None = None,
        blocks_bwd: nn.ModuleList | None = None,
        bwd_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply Mamba blocks along depth ``T`` at each encoded spatial location.

        When ``prefix`` is given it is prepended along depth before the scan and
        stripped from the output, matching the xLSTM prefix-token behaviour.

        With ``blocks_bwd``, a second scan runs over the depth-reversed sequence
        and is fused as ``fwd + g * (bwd - x)``:

        * ``g`` (``bwd_gate``) is zero-initialised, so at step 0 the output equals
          the unidirectional path exactly. That keeps warm-starting from a
          unidirectional checkpoint sound, and lets the optimiser co-adapt the
          downstream layers as the backward branch is phased in.
        * the ``- x`` removes a double-count: ``MambaBlockDRes`` is residual and
          both projections are ``Identity`` when ``d == encoder_channels``, so
          ``bwd`` already carries a copy of the input that ``fwd`` carries too.
          Adding raw ``bwd`` inflates the residual stream by ~1.5-1.8x at init.
        """
        fwd = self._mamba_scan(x_5d, blocks, prefix=prefix)
        if blocks_bwd is None:
            return fwd

        x_rev = torch.flip(x_5d, dims=[1])
        bwd = self._mamba_scan(x_rev, blocks_bwd, prefix=prefix)
        bwd = torch.flip(bwd, dims=[1])
        if bwd_gate is None:
            return fwd + bwd
        # `bwd - x_5d` is the clean backward delta only if the scan passes its
        # input through; with non-Identity projections it is not, so gate `bwd`.
        pass_through = isinstance(self.in_proj, nn.Identity) and isinstance(
            self.out_proj, nn.Identity
        )
        return fwd + bwd_gate * ((bwd - x_5d) if pass_through else bwd)

    def _run_spatial_mamba_pass(
        self, x_5d: torch.Tensor, *, prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply spatial_mid then a second Mamba stack (when ``--spatial-mid`` is enabled)."""
        b, t, c, hc, wc = x_5d.shape
        x_mid = x_5d.reshape(b * t, c, hc, wc)
        x_mid = x_mid + self.spatial_mid(x_mid)
        x_5d = x_mid.view(b, t, c, hc, wc)
        return self._run_mamba_temporal(
            x_5d, self.blocks2, prefix=prefix, blocks_bwd=self.blocks2_bwd,
            bwd_gate=self.bwd_gate2,
        )

    def _encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Run the encoder, capturing skip taps when ``--unet-skips`` is enabled.

        The ``Sequential`` is iterated rather than split into named submodules so
        every parameter keeps its original name and a parent checkpoint still
        loads without a key remap.
        """
        if not self.unet_skips:
            return self.encoder(x), None
        taps: dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index == _ENCODER_TAP_INDEX_HALF:
                taps["half"] = x
            elif index == _ENCODER_TAP_INDEX_QUARTER:
                taps["quarter"] = x
        return x, taps

    def _add_skip(
        self,
        x: torch.Tensor,
        tap: torch.Tensor,
        proj: nn.Module,
        gate: nn.Parameter,
    ) -> torch.Tensor:
        if tap.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"--unet-skips: encoder tap {tuple(tap.shape[-2:])} does not match "
                f"decoder stage {tuple(x.shape[-2:])}. The skip geometry assumes the "
                "stride-8 encoder and three ×2 decoder stages."
            )
        return x + gate.to(x.dtype) * proj(tap)

    def _decode(
        self,
        x: torch.Tensor,
        taps: dict[str, torch.Tensor] | None,
        batch: int,
        depth: int,
    ) -> torch.Tensor:
        if taps is None and self.film_decoder is None:
            return self.decoder(x)
        first, second = self.decoder_inject
        for index, layer in enumerate(self.decoder):
            x = layer(x)
            if index == first:
                if taps is not None:
                    x = self._add_skip(
                        x,
                        taps["quarter"],
                        self.skip_proj_quarter,
                        self.skip_gate_quarter,
                    )
                if self.film_decoder is not None:
                    x = self.film_decoder(x, batch, depth)
            elif index == second and taps is not None:
                x = self._add_skip(
                    x, taps["half"], self.skip_proj_half, self.skip_gate_half
                )
        return x

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        energy_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x1, x2: (B, T, H, W) scalar, or (B, T, C_phase, H, W) phase-packed
        B, T, phases, H, W = _bev_input_shape(x1)
        if x1.shape != x2.shape:
            raise ValueError(
                f"CT/projection shape mismatch: {tuple(x1.shape)} != "
                f"{tuple(x2.shape)}"
            )
        if phases != self.input_phase_channels:
            raise ValueError(
                f"expected {self.input_phase_channels} input phases, got {phases}"
            )

        # (B*T, 2 * phases, H, W); stack for scalar inputs, concat when the phase
        # axis already exists, matching build_bev_encoder_input
        if x1.ndim == 4:
            x = torch.stack((x1, x2), dim=2)
        else:
            x = torch.cat((x1, x2), dim=2)
        x = _append_broadcast_conditioning(x, conditioning, self.conditioning_channels)
        x = x.reshape(B * T, x.shape[2], H, W)
        if self.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        # (B*T, 64, 25, 25) for default 200×200 input
        x, skip_taps = self._encode(x)
        _, C, Hc, Wc = x.shape  # C=64

        x = x + self.spatial_pre(x)

        x_5d = x.view(B, T, C, Hc, Wc)
        prefix = self._compute_prefix(x2, B, C, Hc, Wc, energy_index)
        x_5d = self._run_mamba_temporal(
            x_5d, self.blocks, prefix=prefix, blocks_bwd=self.blocks_bwd,
            bwd_gate=self.bwd_gate,
        )
        if self.spatial_mid is not None:
            x_5d = self._run_spatial_mamba_pass(x_5d, prefix=prefix)
        if self.temporal_conv is not None:
            x_5d = x_5d + self.temporal_conv(x_5d)

        x = x_5d.reshape(B * T, C, Hc, Wc)
        if self.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        if self.window_attn is not None:
            x = self.window_attn(x)

        x = x + self.spatial_post(x)
        if self.film_bottleneck is not None:
            x = self.film_bottleneck(x, B, T)

        dec_feat = x
        x = self._decode(x, skip_taps, B, T)
        if self.slice_scaler is not None:
            x = x * self.slice_scaler(dec_feat)
        if self.pixel_shuffle_dh is not None:
            if self.arch.return_packed_dh_coefficients:
                # Keep the r**2 phase channels on the coarse lattice (B, T, r**2, H, W)
                # instead of interleaving into a materialised fine grid; the CT-space
                # sampler reads the packed phases directly.
                r = self.pixel_shuffle_dh.r
                x = x.reshape(B, T, r * r, self.input_h, self.input_w)
            else:
                x = self.pixel_shuffle_dh(x, B, T)
                x = reshape_depth_height_shuffled(
                    x, batch=B, depth=T, upscale_factor=self.pixel_shuffle_dh.r
                )
        else:
            x = x.reshape(B, T, self.input_h, self.input_w)
        return self.output_activation(x)
