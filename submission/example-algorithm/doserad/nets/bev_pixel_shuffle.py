"""Depth–height PixelShuffle for optional fine BEV dose heads (CT-space loss)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PixelShuffleDepthHeight(nn.Module):
    """Sub-pixel 2× upsample of the depth (T) and height (H) axes.

    Input ``(B*T, C*r*r, H, W)`` with ``C=1`` and ``r=2`` is ``(B*T, 4, H, W)``.
    Channel ``c = i*r + j`` selects sub-pixel ``(i in T, j in H)``.

    Output is ``(B*T*r, 1, H*r, W)`` so callers can reshape to
    ``(B, T*r, H*r, W)``.
    """

    def __init__(self, upscale_factor: int = 2) -> None:
        super().__init__()
        if int(upscale_factor) < 2:
            raise ValueError(f"upscale_factor must be >= 2, got {upscale_factor}")
        self.r = int(upscale_factor)

    def forward(self, x: torch.Tensor, batch: int, depth: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B*T, C, H, W), got {tuple(x.shape)}")
        bt, channels, h, w = x.shape
        r = self.r
        if bt != batch * depth:
            raise ValueError(
                f"first dim {bt} != batch*depth ({batch}*{depth}={batch * depth})"
            )
        if channels != r * r:
            raise ValueError(
                f"expected {r * r} packed channels for 1-channel PixelShuffle, got {channels}"
            )
        x = x.reshape(batch, depth, r, r, h, w)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, T, rT, H, rH, W)
        return x.reshape(batch * depth * r, 1, h * r, w)


def reshape_depth_height_shuffled(
    shuffled: torch.Tensor,
    *,
    batch: int,
    depth: int,
    upscale_factor: int = 2,
) -> torch.Tensor:
    """``(B*T*r, 1, H*r, W)`` -> ``(B, T*r, H*r, W)``."""
    r = int(upscale_factor)
    if shuffled.ndim != 4 or shuffled.shape[1] != 1:
        raise ValueError(f"expected (B*T*r, 1, H*r, W), got {tuple(shuffled.shape)}")
    _, _, h_fine, w = shuffled.shape
    return shuffled.reshape(batch, depth * r, h_fine, w)
