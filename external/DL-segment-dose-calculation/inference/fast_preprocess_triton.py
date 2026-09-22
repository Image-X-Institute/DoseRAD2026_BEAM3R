"""Batched affine inference preprocessing kernels and patient-plan caches.

The reference inference pipeline constructs a dense world-coordinate tensor and
then a dense CT-coordinate tensor for every control point.  The transforms are
affine, so this module evaluates them inside a Triton kernel and writes model
inputs directly into model-ready buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl

from patient_space_resample_triton import (
    _Bicubic2DAffineTriton,
)

@dataclass(frozen=True)
class FastPreprocessGeometry:
    """Compact per-control-point transforms used by the fused kernel."""

    ct_affine: np.ndarray
    world_affine: np.ndarray
    ray_origin: np.ndarray
    aperture_origin: np.ndarray
    basis: np.ndarray


@dataclass
class FastPlanInferenceCache:
    """GPU-resident preprocessing and compact CT backprojection state."""

    masks: torch.Tensor
    collapsed_ct: torch.Tensor
    collapsed_ct_index: torch.Tensor
    ct_affines: torch.Tensor
    world_affines: torch.Tensor
    ray_origins: torch.Tensor
    aperture_origins: torch.Tensor
    bases: torch.Tensor
    inverse_affines: tuple[torch.Tensor, ...]
    backprojection_contexts: tuple[object, ...]

    def preprocess_geometry(
        self, selection: slice | torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return (
            self.ct_affines[selection],
            self.world_affines[selection],
            self.ray_origins[selection],
            self.aperture_origins[selection],
            self.bases[selection],
        )


@triton.jit
def _mirror_index(index, size):
    period = 2 * (size - 1) + 2 * (size == 1)
    value = index % period
    return tl.where(size > 1, tl.where(value >= size, period - value, value), 0)


@triton.jit
def _cubic_weight(t, tap: tl.constexpr):
    if tap == 0:
        return (1.0 - t) * (1.0 - t) * (1.0 - t) * (1.0 / 6.0)
    if tap == 1:
        t2 = t * t
        return (4.0 - 6.0 * t2 + 3.0 * t2 * t) * (1.0 / 6.0)
    if tap == 2:
        t2 = t * t
        return (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) * (1.0 / 6.0)
    return t * t * t * (1.0 / 6.0)


@triton.jit
def _fast_preprocess_kernel(
    ct_ptr,
    mask_ptr,
    ct_affine_ptr,
    world_affine_ptr,
    ray_ptr,
    aperture_origin_ptr,
    basis_ptr,
    ct_out_ptr,
    aperture_out_ptr,
    ct_x,
    ct_y,
    ct_z,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    batch,
    ct_min,
    ct_inv_range,
    sad,
    aperture_inv_spacing_y,
    aperture_inv_spacing_z,
    normalize_ct: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate z-aligned bicubic CT and projected aperture volumes."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_voxels = nx * ny * nz
    total = batch * n_voxels
    valid_offset = offsets < total

    b = offsets // n_voxels
    local = offsets - b * n_voxels
    k = local % nz
    q = local // nz
    j = q % ny
    i = q // ny
    fi = i.to(tl.float32)
    fj = j.to(tl.float32)
    fk = k.to(tl.float32)

    affine_base = b * 12
    xq = (
        tl.load(ct_affine_ptr + affine_base + 0, mask=valid_offset) * fi
        + tl.load(ct_affine_ptr + affine_base + 1, mask=valid_offset) * fj
        + tl.load(ct_affine_ptr + affine_base + 2, mask=valid_offset) * fk
        + tl.load(ct_affine_ptr + affine_base + 3, mask=valid_offset)
    )
    yq = (
        tl.load(ct_affine_ptr + affine_base + 4, mask=valid_offset) * fi
        + tl.load(ct_affine_ptr + affine_base + 5, mask=valid_offset) * fj
        + tl.load(ct_affine_ptr + affine_base + 6, mask=valid_offset) * fk
        + tl.load(ct_affine_ptr + affine_base + 7, mask=valid_offset)
    )
    zq = (
        tl.load(ct_affine_ptr + affine_base + 8, mask=valid_offset) * fi
        + tl.load(ct_affine_ptr + affine_base + 9, mask=valid_offset) * fj
        + tl.load(ct_affine_ptr + affine_base + 10, mask=valid_offset) * fk
        + tl.load(ct_affine_ptr + affine_base + 11, mask=valid_offset)
    )
    zi = tl.floor(zq + 0.5).to(tl.int32)
    sample_valid = (
        valid_offset
        & (xq >= 0.0)
        & (xq <= ct_x - 1)
        & (yq >= 0.0)
        & (yq <= ct_y - 1)
        & (zi >= 0)
        & (zi < ct_z)
        & (tl.abs(zq - zi.to(tl.float32)) <= 1.0e-3)
    )

    x0 = tl.floor(xq).to(tl.int32)
    y0 = tl.floor(yq).to(tl.int32)
    fx = xq - x0.to(tl.float32)
    fy = yq - y0.to(tl.float32)
    ct_value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for xtap in range(4):
        xi = _mirror_index(x0 - 1 + xtap, ct_x)
        wx = _cubic_weight(fx, xtap)
        for ytap in range(4):
            yi = _mirror_index(y0 - 1 + ytap, ct_y)
            wy = _cubic_weight(fy, ytap)
            zm1 = tl.where(zi > 0, zi - 1, tl.minimum(1, ct_z - 1))
            zp1 = tl.where(zi < ct_z - 1, zi + 1, tl.maximum(ct_z - 2, 0))
            base = (xi * ct_y + yi) * ct_z
            collapsed = (
                tl.load(ct_ptr + base + zm1, mask=sample_valid, other=0.0)
                + 4.0 * tl.load(ct_ptr + base + zi, mask=sample_valid, other=0.0)
                + tl.load(ct_ptr + base + zp1, mask=sample_valid, other=0.0)
            ) * (1.0 / 6.0)
            ct_value += wx * wy * collapsed
    ct_value = tl.where(sample_valid, ct_value, -1024.0)
    ct_value = tl.maximum(ct_value, -1024.0)
    if normalize_ct:
        ct_value = tl.minimum(tl.maximum(ct_value, ct_min), ct_min + 1.0 / ct_inv_range)
        ct_value = (ct_value - ct_min) * ct_inv_range
    tl.store(ct_out_ptr + offsets, ct_value, mask=valid_offset)

    world_base = b * 12
    px = (
        tl.load(world_affine_ptr + world_base + 0, mask=valid_offset) * fi
        + tl.load(world_affine_ptr + world_base + 1, mask=valid_offset) * fj
        + tl.load(world_affine_ptr + world_base + 2, mask=valid_offset) * fk
        + tl.load(world_affine_ptr + world_base + 3, mask=valid_offset)
    )
    py = (
        tl.load(world_affine_ptr + world_base + 4, mask=valid_offset) * fi
        + tl.load(world_affine_ptr + world_base + 5, mask=valid_offset) * fj
        + tl.load(world_affine_ptr + world_base + 6, mask=valid_offset) * fk
        + tl.load(world_affine_ptr + world_base + 7, mask=valid_offset)
    )
    pz = (
        tl.load(world_affine_ptr + world_base + 8, mask=valid_offset) * fi
        + tl.load(world_affine_ptr + world_base + 9, mask=valid_offset) * fj
        + tl.load(world_affine_ptr + world_base + 10, mask=valid_offset) * fk
        + tl.load(world_affine_ptr + world_base + 11, mask=valid_offset)
    )

    geometry_base = b * 3
    sx = tl.load(ray_ptr + geometry_base + 0, mask=valid_offset)
    sy = tl.load(ray_ptr + geometry_base + 1, mask=valid_offset)
    sz = tl.load(ray_ptr + geometry_base + 2, mask=valid_offset)
    origin_x = tl.load(aperture_origin_ptr + geometry_base + 0, mask=valid_offset)
    origin_y = tl.load(aperture_origin_ptr + geometry_base + 1, mask=valid_offset)
    origin_z = tl.load(aperture_origin_ptr + geometry_base + 2, mask=valid_offset)
    basis_base = b * 9
    ux0 = tl.load(basis_ptr + basis_base + 0, mask=valid_offset)
    ux1 = tl.load(basis_ptr + basis_base + 1, mask=valid_offset)
    ux2 = tl.load(basis_ptr + basis_base + 2, mask=valid_offset)
    uy0 = tl.load(basis_ptr + basis_base + 3, mask=valid_offset)
    uy1 = tl.load(basis_ptr + basis_base + 4, mask=valid_offset)
    uy2 = tl.load(basis_ptr + basis_base + 5, mask=valid_offset)
    uz0 = tl.load(basis_ptr + basis_base + 6, mask=valid_offset)
    uz1 = tl.load(basis_ptr + basis_base + 7, mask=valid_offset)
    uz2 = tl.load(basis_ptr + basis_base + 8, mask=valid_offset)

    rx = px - sx
    ry = py - sy
    rz = pz - sz
    denominator = rx * ux0 + ry * ux1 + rz * ux2
    ray_valid = valid_offset & (tl.abs(denominator) >= 1.0e-9)
    scale = sad / denominator
    ray_valid &= scale > 0.0
    wx = sx + scale * rx - origin_x
    wy = sy + scale * ry - origin_y
    wz = sz + scale * rz - origin_z
    aperture_y = (
        (wx * uy0 + wy * uy1 + wz * uy2) * aperture_inv_spacing_y
        + (ny // 2 - 0.5)
    )
    aperture_z = (
        (wx * uz0 + wy * uz1 + wz * uz2) * aperture_inv_spacing_z
        + (nz // 2 - 0.5)
    )
    y0 = tl.floor(aperture_y).to(tl.int32)
    z0 = tl.floor(aperture_z).to(tl.int32)
    aperture_valid = (
        ray_valid
        & (aperture_y >= 0.0)
        & (aperture_y < ny - 1)
        & (aperture_z >= 0.0)
        & (aperture_z < nz - 1)
    )
    fy = aperture_y - y0.to(tl.float32)
    fz = aperture_z - z0.to(tl.float32)
    mask_base = b * ny * nz
    v00 = tl.load(mask_ptr + mask_base + y0 * nz + z0, mask=aperture_valid, other=0.0)
    v10 = tl.load(mask_ptr + mask_base + (y0 + 1) * nz + z0, mask=aperture_valid, other=0.0)
    v01 = tl.load(mask_ptr + mask_base + y0 * nz + z0 + 1, mask=aperture_valid, other=0.0)
    v11 = tl.load(mask_ptr + mask_base + (y0 + 1) * nz + z0 + 1, mask=aperture_valid, other=0.0)
    aperture = (
        (1.0 - fy) * ((1.0 - fz) * v00 + fz * v01)
        + fy * ((1.0 - fz) * v10 + fz * v11)
    )
    tl.store(aperture_out_ptr + offsets, aperture, mask=valid_offset)


@triton.jit
def _fast_ct_kernel(
    ct_ptr,
    affine_ptr,
    out_ptr,
    ct_x,
    ct_y,
    ct_z,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    batch,
    ct_min,
    ct_inv_range,
    normalize_ct: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_voxels = nx * ny * nz
    total = batch * n_voxels
    mask = offsets < total
    b = offsets // n_voxels
    local = offsets - b * n_voxels
    k = local % nz
    q = local // nz
    j = q % ny
    i = q // ny
    fi = i.to(tl.float32)
    fj = j.to(tl.float32)
    fk = k.to(tl.float32)
    base_affine = b * 12
    xq = (
        tl.load(affine_ptr + base_affine + 0, mask=mask) * fi
        + tl.load(affine_ptr + base_affine + 1, mask=mask) * fj
        + tl.load(affine_ptr + base_affine + 2, mask=mask) * fk
        + tl.load(affine_ptr + base_affine + 3, mask=mask)
    )
    yq = (
        tl.load(affine_ptr + base_affine + 4, mask=mask) * fi
        + tl.load(affine_ptr + base_affine + 5, mask=mask) * fj
        + tl.load(affine_ptr + base_affine + 6, mask=mask) * fk
        + tl.load(affine_ptr + base_affine + 7, mask=mask)
    )
    zq = (
        tl.load(affine_ptr + base_affine + 8, mask=mask) * fi
        + tl.load(affine_ptr + base_affine + 9, mask=mask) * fj
        + tl.load(affine_ptr + base_affine + 10, mask=mask) * fk
        + tl.load(affine_ptr + base_affine + 11, mask=mask)
    )
    zi = tl.floor(zq + 0.5).to(tl.int32)
    valid = (
        mask
        & (xq >= 0.0)
        & (xq <= ct_x - 1)
        & (yq >= 0.0)
        & (yq <= ct_y - 1)
        & (zi >= 0)
        & (zi < ct_z)
        & (tl.abs(zq - zi.to(tl.float32)) <= 1.0e-3)
    )
    x0 = tl.floor(xq).to(tl.int32)
    y0 = tl.floor(yq).to(tl.int32)
    fx = xq - x0.to(tl.float32)
    fy = yq - y0.to(tl.float32)
    zm1 = tl.where(zi > 0, zi - 1, tl.minimum(1, ct_z - 1))
    zp1 = tl.where(zi < ct_z - 1, zi + 1, tl.maximum(ct_z - 2, 0))
    value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for xtap in range(4):
        xi = _mirror_index(x0 - 1 + xtap, ct_x)
        xweight = _cubic_weight(fx, xtap)
        for ytap in range(4):
            yi = _mirror_index(y0 - 1 + ytap, ct_y)
            weight = xweight * _cubic_weight(fy, ytap)
            source = (xi * ct_y + yi) * ct_z
            collapsed = (
                tl.load(ct_ptr + source + zm1, mask=valid, other=0.0)
                + 4.0 * tl.load(ct_ptr + source + zi, mask=valid, other=0.0)
                + tl.load(ct_ptr + source + zp1, mask=valid, other=0.0)
            ) * (1.0 / 6.0)
            value += weight * collapsed
    value = tl.where(valid, tl.maximum(value, -1024.0), -1024.0)
    if normalize_ct:
        ct_max = ct_min + 1.0 / ct_inv_range
        value = tl.minimum(tl.maximum(value, ct_min), ct_max)
        value = (value - ct_min) * ct_inv_range
    tl.store(out_ptr + offsets, value, mask=mask)


@triton.jit
def _fast_aperture_kernel(
    masks_ptr,
    world_affine_ptr,
    ray_ptr,
    aperture_origin_ptr,
    basis_ptr,
    out_ptr,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    batch,
    sad,
    aperture_inv_spacing_y,
    aperture_inv_spacing_z,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_voxels = nx * ny * nz
    total = batch * n_voxels
    mask = offsets < total
    b = offsets // n_voxels
    local = offsets - b * n_voxels
    k = local % nz
    q = local // nz
    j = q % ny
    i = q // ny
    fi = i.to(tl.float32)
    fj = j.to(tl.float32)
    fk = k.to(tl.float32)
    affine_base = b * 12
    px = (
        tl.load(world_affine_ptr + affine_base + 0, mask=mask) * fi
        + tl.load(world_affine_ptr + affine_base + 1, mask=mask) * fj
        + tl.load(world_affine_ptr + affine_base + 2, mask=mask) * fk
        + tl.load(world_affine_ptr + affine_base + 3, mask=mask)
    )
    py = (
        tl.load(world_affine_ptr + affine_base + 4, mask=mask) * fi
        + tl.load(world_affine_ptr + affine_base + 5, mask=mask) * fj
        + tl.load(world_affine_ptr + affine_base + 6, mask=mask) * fk
        + tl.load(world_affine_ptr + affine_base + 7, mask=mask)
    )
    pz = (
        tl.load(world_affine_ptr + affine_base + 8, mask=mask) * fi
        + tl.load(world_affine_ptr + affine_base + 9, mask=mask) * fj
        + tl.load(world_affine_ptr + affine_base + 10, mask=mask) * fk
        + tl.load(world_affine_ptr + affine_base + 11, mask=mask)
    )
    vector_base = b * 3
    sx = tl.load(ray_ptr + vector_base + 0, mask=mask)
    sy = tl.load(ray_ptr + vector_base + 1, mask=mask)
    sz = tl.load(ray_ptr + vector_base + 2, mask=mask)
    ox = tl.load(aperture_origin_ptr + vector_base + 0, mask=mask)
    oy = tl.load(aperture_origin_ptr + vector_base + 1, mask=mask)
    oz = tl.load(aperture_origin_ptr + vector_base + 2, mask=mask)
    basis_base = b * 9
    ux0 = tl.load(basis_ptr + basis_base + 0, mask=mask)
    ux1 = tl.load(basis_ptr + basis_base + 1, mask=mask)
    ux2 = tl.load(basis_ptr + basis_base + 2, mask=mask)
    uy0 = tl.load(basis_ptr + basis_base + 3, mask=mask)
    uy1 = tl.load(basis_ptr + basis_base + 4, mask=mask)
    uy2 = tl.load(basis_ptr + basis_base + 5, mask=mask)
    uz0 = tl.load(basis_ptr + basis_base + 6, mask=mask)
    uz1 = tl.load(basis_ptr + basis_base + 7, mask=mask)
    uz2 = tl.load(basis_ptr + basis_base + 8, mask=mask)
    rx = px - sx
    ry = py - sy
    rz = pz - sz
    denominator = rx * ux0 + ry * ux1 + rz * ux2
    valid = mask & (tl.abs(denominator) >= 1.0e-9)
    ray_scale = sad / denominator
    valid &= ray_scale > 0.0
    wx = sx + ray_scale * rx - ox
    wy = sy + ray_scale * ry - oy
    wz = sz + ray_scale * rz - oz
    yq = (
        (wx * uy0 + wy * uy1 + wz * uy2) * aperture_inv_spacing_y
        + (ny // 2 - 0.5)
    )
    zq = (
        (wx * uz0 + wy * uz1 + wz * uz2) * aperture_inv_spacing_z
        + (nz // 2 - 0.5)
    )
    y0 = tl.floor(yq).to(tl.int32)
    z0 = tl.floor(zq).to(tl.int32)
    valid &= (
        (yq >= 0.0) & (yq < ny - 1) & (zq >= 0.0) & (zq < nz - 1)
    )
    fy = yq - y0.to(tl.float32)
    fz = zq - z0.to(tl.float32)
    source = b * ny * nz + y0 * nz + z0
    v00 = tl.load(masks_ptr + source, mask=valid, other=0.0)
    v10 = tl.load(masks_ptr + source + nz, mask=valid, other=0.0)
    v01 = tl.load(masks_ptr + source + 1, mask=valid, other=0.0)
    v11 = tl.load(masks_ptr + source + nz + 1, mask=valid, other=0.0)
    value = (
        (1.0 - fy) * ((1.0 - fz) * v00 + fz * v01)
        + fy * ((1.0 - fz) * v10 + fz * v11)
    )
    tl.store(out_ptr + offsets, value, mask=mask)


@triton.jit
def _fast_ct_from_collapsed_kernel(
    collapsed_ptr,
    affine_ptr,
    out_ptr,
    ct_x,
    ct_y,
    nx: tl.constexpr,
    ny: tl.constexpr,
    nz: tl.constexpr,
    batch,
    ct_min,
    ct_inv_range,
    normalize_ct: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Bicubic xy sampling from patient CT planes collapsed once per geometry."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_voxels = nx * ny * nz
    total = batch * n_voxels
    mask = offsets < total
    b = offsets // n_voxels
    local = offsets - b * n_voxels
    k = local % nz
    q = local // nz
    j = q % ny
    i = q // ny
    fi = i.to(tl.float32)
    fj = j.to(tl.float32)
    fk = k.to(tl.float32)
    affine_base = b * 12
    xq = (
        tl.load(affine_ptr + affine_base + 0, mask=mask) * fi
        + tl.load(affine_ptr + affine_base + 1, mask=mask) * fj
        + tl.load(affine_ptr + affine_base + 2, mask=mask) * fk
        + tl.load(affine_ptr + affine_base + 3, mask=mask)
    )
    yq = (
        tl.load(affine_ptr + affine_base + 4, mask=mask) * fi
        + tl.load(affine_ptr + affine_base + 5, mask=mask) * fj
        + tl.load(affine_ptr + affine_base + 6, mask=mask) * fk
        + tl.load(affine_ptr + affine_base + 7, mask=mask)
    )
    valid = (
        mask
        & (xq >= 0.0)
        & (xq <= ct_x - 1)
        & (yq >= 0.0)
        & (yq <= ct_y - 1)
    )
    x0 = tl.floor(xq).to(tl.int32)
    y0 = tl.floor(yq).to(tl.int32)
    fx = xq - x0.to(tl.float32)
    fy = yq - y0.to(tl.float32)
    plane_base = (b * nz + k) * ct_x * ct_y
    value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for xtap in range(4):
        xi = _mirror_index(x0 - 1 + xtap, ct_x)
        xweight = _cubic_weight(fx, xtap)
        for ytap in range(4):
            yi = _mirror_index(y0 - 1 + ytap, ct_y)
            weight = xweight * _cubic_weight(fy, ytap)
            sample = tl.load(
                collapsed_ptr + plane_base + xi * ct_y + yi,
                mask=valid,
                other=0.0,
            )
            value += weight * sample
    value = tl.where(valid, tl.maximum(value, -1024.0), -1024.0)
    if normalize_ct:
        ct_max = ct_min + 1.0 / ct_inv_range
        value = tl.minimum(tl.maximum(value, ct_min), ct_max)
        value = (value - ct_min) * ct_inv_range
    tl.store(out_ptr + offsets, value, mask=mask)


@triton.jit
def _normalize_or_copy_kernel(
    input_ptr,
    valid_ptr,
    output_ptr,
    total,
    ct_min,
    ct_max,
    inverse_range,
    fill_value,
    normalize: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    valid = tl.load(valid_ptr + offsets, mask=mask, other=0).to(tl.int1)
    value = tl.load(input_ptr + offsets, mask=mask, other=fill_value)
    value = tl.where(valid, tl.maximum(value, fill_value), fill_value)
    if normalize:
        value = tl.minimum(tl.maximum(value, ct_min), ct_max)
        value = (value - ct_min) * inverse_range
    tl.store(output_ptr + offsets, value, mask=mask)


def build_collapsed_ct_cache(
    ct_coeff: torch.Tensor,
    ct_affine: torch.Tensor,
    nz: int,
) -> torch.Tensor:
    """Collapse the three z taps once for every unique control-point geometry."""
    if ct_coeff.ndim != 3 or not ct_coeff.is_cuda:
        raise ValueError("ct_coeff must be a CUDA (X,Y,Z) tensor")
    batch = int(ct_affine.shape[0])
    k = torch.arange(nz, device=ct_coeff.device, dtype=torch.float32)
    rows = ct_affine[:, 2]
    if float(rows[:, :2].abs().max().item()) > 1.0e-4:
        raise ValueError("CT affine is not z aligned")
    zq = rows[:, 2:3] * k.unsqueeze(0) + rows[:, 3:4]
    zi = zq.round().to(torch.long)
    if float((zq - zi).abs().max().item()) > 1.0e-3:
        raise ValueError("CT affine does not map to integer z planes")
    z_size = int(ct_coeff.shape[2])
    in_range = (zi >= 0) & (zi < z_size)
    zc = zi.clamp(0, z_size - 1)
    zm1 = torch.where(zc > 0, zc - 1, torch.full_like(zc, min(1, z_size - 1)))
    zp1 = torch.where(
        zc < z_size - 1,
        zc + 1,
        torch.full_like(zc, max(z_size - 2, 0)),
    )
    planes = []
    for batch_index in range(batch):
        collapsed = (
            ct_coeff[:, :, zm1[batch_index]]
            + 4.0 * ct_coeff[:, :, zc[batch_index]]
            + ct_coeff[:, :, zp1[batch_index]]
        ) * (1.0 / 6.0)
        collapsed = collapsed.permute(2, 0, 1).contiguous()
        collapsed[~in_range[batch_index]] = -1024.0
        planes.append(collapsed)
    # Existing affine sampler expects (B,T,H,W).  Here W enumerates the
    # already-collapsed target z planes, while T/H are patient x/y.
    return torch.stack(planes).permute(0, 2, 3, 1).contiguous()


def build_unique_collapsed_ct_cache(
    ct_coeff: torch.Tensor,
    ct_affine: torch.Tensor,
    nz: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache one collapsed CT volume per distinct aligned z-plane sequence."""
    k = torch.arange(nz, device=ct_affine.device, dtype=torch.float32)
    rows = ct_affine[:, 2]
    zq = rows[:, 2:3] * k.unsqueeze(0) + rows[:, 3:4]
    keys = zq.round().to(torch.int32).cpu().numpy()
    unique_indices: list[int] = []
    key_to_unique: dict[bytes, int] = {}
    inverse: list[int] = []
    for index, key in enumerate(keys):
        encoded = np.ascontiguousarray(key).tobytes()
        unique_index = key_to_unique.get(encoded)
        if unique_index is None:
            unique_index = len(unique_indices)
            key_to_unique[encoded] = unique_index
            unique_indices.append(index)
        inverse.append(unique_index)
    representatives = torch.as_tensor(
        unique_indices, device=ct_affine.device, dtype=torch.long
    )
    unique_cache = build_collapsed_ct_cache(
        ct_coeff, ct_affine.index_select(0, representatives), nz
    )
    return unique_cache, torch.as_tensor(
        inverse, device=ct_affine.device, dtype=torch.long
    )


def build_fast_plan_inference_cache(
    tasks,
    mac_cache: dict,
    grid,
    ct_coeff: torch.Tensor,
    ct_shape: tuple[int, int, int],
    ct_spacing: tuple[float, float, float],
    ct_origin: tuple[float, float, float],
    device: torch.device,
    *,
    collapsed_dtype: torch.dtype = torch.bfloat16,
    output_upscale_factor: int = 2,
    include_backprojection: bool = True,
) -> FastPlanInferenceCache:
    """Build all reusable affine preprocessing state for one patient plan.

    The aperture is evaluated at the same z-aligned BEV voxel positions as the
    CT. Its physical plane origin remains the unshifted beam plane, matching
    :func:`pipeline.prepare_bev_input_volumes`.
    """
    import cupy as cp
    import cupyx.scipy.ndimage as cpndi

    from pipeline import (
        build_back_projection_affine_ctx,
        segment_geometry,
    )

    if not tasks:
        raise ValueError("cannot build a plan cache without control points")
    if not grid.align_bev_z_to_ct_slices or not grid.bicubic_z_align:
        raise ValueError(
            "triton-affine preprocessing requires z alignment and bicubic z alignment"
        )
    if not ct_coeff.is_cuda or ct_coeff.ndim != 3:
        raise ValueError("ct_coeff must be a CUDA (X,Y,Z) tensor")
    if include_backprojection and output_upscale_factor < 2:
        raise ValueError("output_upscale_factor must be at least 2")

    masks = []
    ct_affines = []
    world_affines = []
    ray_origins = []
    aperture_origins = []
    bases = []
    inverse_affines = []
    backprojection_contexts = []
    spacing = np.asarray(grid.spacing_dhw, dtype=np.float32)
    ct_spacing_array = np.asarray(ct_spacing, dtype=np.float32)
    ct_origin_array = np.asarray(ct_origin, dtype=np.float32)

    for task in tasks:
        native = cp.fromfile(task.segment_path, dtype=cp.int8).reshape(
            grid.segment_native_size, grid.segment_native_size
        )
        native = cp.ascontiguousarray(cp.flip(cp.rot90(native, 3), axis=0))
        resized = cpndi.zoom(
            native.astype(cp.float32),
            grid.segment_zoom_yz,
            order=1,
        ).astype(cp.float32, copy=False)
        resized = cp.ascontiguousarray(resized)
        masks.append(torch.from_dlpack(resized))

        basis_cp, aligned_origin_cp, offset_cp = segment_geometry(
            task.mac_file,
            mac_cache,
            grid.nx,
            grid.ny,
            grid.nz,
            plane_origin_offset_mm=grid.resolved_plane_origin_offset_mm,
            spacing_dhw=grid.spacing_dhw,
            ct_origin=ct_origin,
            ct_spacing=ct_spacing,
            align_bev_z_to_ct_slices=True,
            align_orient_tol=grid.align_orient_tol,
            align_spacing_tol_mm=grid.align_spacing_tol_mm,
        )
        _, physical_origin_cp, _ = segment_geometry(
            task.mac_file,
            mac_cache,
            grid.nx,
            grid.ny,
            grid.nz,
            plane_origin_offset_mm=grid.resolved_plane_origin_offset_mm,
            spacing_dhw=grid.spacing_dhw,
            align_bev_z_to_ct_slices=False,
        )
        basis = cp.asnumpy(basis_cp).astype(np.float32, copy=False)
        aligned_origin = cp.asnumpy(aligned_origin_cp).astype(
            np.float32, copy=False
        )
        physical_origin = cp.asnumpy(physical_origin_cp).astype(
            np.float32, copy=False
        )
        offset = cp.asnumpy(offset_cp).astype(np.float32, copy=False)
        linear = (spacing[:, None] * basis).T
        constant = aligned_origin + (offset * spacing) @ basis
        world_affine = np.concatenate((linear, constant[:, None]), axis=1)
        ct_affine = world_affine.copy()
        ct_affine[:, :3] /= ct_spacing_array[:, None]
        ct_affine[:, 3] = (constant - ct_origin_array) / ct_spacing_array
        ct_affines.append(ct_affine)
        world_affines.append(world_affine)
        aperture_origins.append(physical_origin)
        bases.append(basis)

        record = mac_cache.get(task.name)
        if not isinstance(record, dict) or not isinstance(record.get("s"), list):
            raise ValueError(f"missing cached ray origin for {task.name}")
        ray_origins.append(np.asarray(record["s"], dtype=np.float32))

        if include_backprojection:
            context = build_back_projection_affine_ctx(
                task.mac_file,
                ct_shape,
                ct_spacing,
                ct_origin,
                mac_cache=mac_cache,
                NX=grid.nx * output_upscale_factor,
                NY=grid.ny * output_upscale_factor,
                NZ=grid.nz,
                spacing_dhw=(
                    grid.spacing_dhw[0] / output_upscale_factor,
                    grid.spacing_dhw[1] / output_upscale_factor,
                    grid.spacing_dhw[2],
                ),
                plane_origin_offset_mm=grid.resolved_plane_origin_offset_mm,
                align_bev_z_to_ct_slices=True,
                align_orient_tol=grid.align_orient_tol,
                align_spacing_tol_mm=grid.align_spacing_tol_mm,
            )
            backprojection_contexts.append(context)
            inverse_affines.append(
                torch.from_dlpack(cp.ascontiguousarray(context.affine)).unsqueeze(0)
            )

    def tensor(values) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(np.stack(values)),
            device=device,
            dtype=torch.float32,
        )

    masks_tensor = torch.stack(masks).to(device=device)
    ct_affines_tensor = tensor(ct_affines)
    collapsed_ct, collapsed_ct_index = build_unique_collapsed_ct_cache(
        ct_coeff, ct_affines_tensor, grid.nz
    )
    if collapsed_ct.dtype != collapsed_dtype:
        collapsed_ct = collapsed_ct.to(collapsed_dtype)
    return FastPlanInferenceCache(
        masks=masks_tensor,
        collapsed_ct=collapsed_ct,
        collapsed_ct_index=collapsed_ct_index,
        ct_affines=ct_affines_tensor,
        world_affines=tensor(world_affines),
        ray_origins=tensor(ray_origins),
        aperture_origins=tensor(aperture_origins),
        bases=tensor(bases),
        inverse_affines=tuple(inverse_affines),
        backprojection_contexts=tuple(backprojection_contexts),
    )


def fast_preprocess_batch(
    collapsed_ct: torch.Tensor,
    masks: torch.Tensor,
    ct_affine: torch.Tensor,
    world_affine: torch.Tensor,
    ray_origin: torch.Tensor,
    aperture_origin: torch.Tensor,
    basis: torch.Tensor,
    *,
    output_shape: tuple[int, int, int],
    ct_min: float = -1024.0,
    ct_max: float = 3071.0,
    output_dtype: torch.dtype = torch.float32,
    normalize_ct: bool = False,
    fill_value: float = -1024.0,
    spacing_yz: tuple[float, float] = (2.0, 2.0),
    sad_mm: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch without materialising query-coordinate volumes.

    ``fill_value`` is the raw input-space value written outside the volume (and
    used as the lower clamp on interpolation undershoot). It defaults to the CT
    air value; non-HU inputs such as p01-p99 normalised MR pass 0.0.
    """
    if not collapsed_ct.is_cuda or collapsed_ct.ndim != 4:
        raise ValueError("collapsed_ct must be a CUDA (B,X,Y,Z) tensor")
    nx, ny, nz = (int(value) for value in output_shape)
    spacing_y, spacing_z = (float(value) for value in spacing_yz)
    if spacing_y <= 0.0 or spacing_z <= 0.0:
        raise ValueError("spacing_yz values must be positive")
    batch = int(ct_affine.shape[0])
    expected = {
        "masks": (batch, ny, nz),
        "ct_affine": (batch, 3, 4),
        "world_affine": (batch, 3, 4),
        "ray_origin": (batch, 3),
        "aperture_origin": (batch, 3),
        "basis": (batch, 3, 3),
    }
    tensors = {
        "masks": masks,
        "ct_affine": ct_affine,
        "world_affine": world_affine,
        "ray_origin": ray_origin,
        "aperture_origin": aperture_origin,
        "basis": basis,
    }
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensors[name].shape)} != {shape}")
    ct_out = torch.empty(
        (batch, nx, ny, nz), device=collapsed_ct.device, dtype=output_dtype
    )
    aperture_out = torch.empty_like(ct_out)
    total = ct_out.numel()
    sampling_affine = ct_affine.clone()
    sampling_affine[:, 2].zero_()
    sampling_affine[:, 2, 2] = 1.0
    ct_sample, valid = _Bicubic2DAffineTriton.apply(
        collapsed_ct, sampling_affine, (nx, ny, nz)
    )
    _normalize_or_copy_kernel[(triton.cdiv(total, 256),)](
        ct_sample,
        valid,
        ct_out,
        total,
        float(ct_min),
        float(ct_max),
        1.0 / (float(ct_max) - float(ct_min)),
        float(fill_value),
        normalize=normalize_ct,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    _fast_aperture_kernel[(triton.cdiv(total, 256),)](
        masks,
        world_affine,
        ray_origin,
        aperture_origin,
        basis,
        aperture_out,
        nx,
        ny,
        nz,
        batch,
        float(sad_mm),
        1.0 / spacing_y,
        1.0 / spacing_z,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return ct_out, aperture_out


def build_fast_proton_plan_cache(
    tasks,
    mac_cache: dict,
    input_grid,
    output_grid,
    ct_coeff: torch.Tensor,
    ct_shape: tuple[int, int, int],
    ct_spacing: tuple[float, float, float],
    ct_origin: tuple[float, float, float],
    device: torch.device,
    *,
    output_upscale_factor: int = 2,
    collapsed_dtype: torch.dtype = torch.float32,
) -> FastPlanInferenceCache:
    """Build reusable affine state directly from proton ray metadata.

    Proton catalog records deliberately have virtual MAC/aperture paths.  The
    generic photon cache builder therefore cannot be used: this variant builds
    Gaussian fluence masks and both transforms from the already-populated
    in-memory ``mac_cache``.
    """
    import cupy as cp

    from pipeline import build_back_projection_affine_ctx, segment_geometry

    if not tasks:
        raise ValueError("cannot build a proton plan cache without beamlets")
    masks = []
    ct_affines = []
    world_affines = []
    ray_origins = []
    aperture_origins = []
    bases = []
    inverse_affines = []
    contexts = []
    spacing = np.asarray(input_grid.spacing_dhw, dtype=np.float32)
    ct_spacing_array = np.asarray(ct_spacing, dtype=np.float32)
    ct_origin_array = np.asarray(ct_origin, dtype=np.float32)
    y = (
        np.arange(input_grid.ny, dtype=np.float32)
        - input_grid.ny // 2
        + 0.5
    ) * float(input_grid.spacing_dhw[1])
    z = (
        np.arange(input_grid.nz, dtype=np.float32)
        - input_grid.nz // 2
        + 0.5
    ) * float(input_grid.spacing_dhw[2])
    radius2 = y[:, None] ** 2 + z[None, :] ** 2

    for task in tasks:
        sigma = float(task.sigma_spot_mm or 0.0)
        if sigma <= 0.0:
            raise ValueError(f"{task.sample_id}: missing positive spot sigma")
        masks.append(np.exp(-0.5 * radius2 / (sigma * sigma)).astype(np.float32))

        basis_cp, aligned_origin_cp, offset_cp = segment_geometry(
            task.mac_file,
            mac_cache,
            input_grid.nx,
            input_grid.ny,
            input_grid.nz,
            plane_origin_offset_mm=input_grid.resolved_plane_origin_offset_mm,
            spacing_dhw=input_grid.spacing_dhw,
            ct_origin=ct_origin,
            ct_spacing=ct_spacing,
            align_bev_z_to_ct_slices=True,
            align_orient_tol=input_grid.align_orient_tol,
            align_spacing_tol_mm=input_grid.align_spacing_tol_mm,
        )
        _, physical_origin_cp, _ = segment_geometry(
            task.mac_file,
            mac_cache,
            input_grid.nx,
            input_grid.ny,
            input_grid.nz,
            plane_origin_offset_mm=input_grid.resolved_plane_origin_offset_mm,
            spacing_dhw=input_grid.spacing_dhw,
            align_bev_z_to_ct_slices=False,
        )
        basis = cp.asnumpy(basis_cp).astype(np.float32, copy=False)
        aligned_origin = cp.asnumpy(aligned_origin_cp).astype(np.float32, copy=False)
        physical_origin = cp.asnumpy(physical_origin_cp).astype(np.float32, copy=False)
        offset = cp.asnumpy(offset_cp).astype(np.float32, copy=False)
        linear = (spacing[:, None] * basis).T
        constant = aligned_origin + (offset * spacing) @ basis
        world_affine = np.concatenate((linear, constant[:, None]), axis=1)
        ct_affine = world_affine.copy()
        ct_affine[:, :3] /= ct_spacing_array[:, None]
        ct_affine[:, 3] = (constant - ct_origin_array) / ct_spacing_array
        ct_affines.append(ct_affine)
        world_affines.append(world_affine)
        aperture_origins.append(physical_origin)
        bases.append(basis)
        ray_origins.append(np.asarray(mac_cache[task.sample_id]["s"], np.float32))

        dx, dy, dz = (float(value) for value in output_grid.spacing_dhw)
        context = build_back_projection_affine_ctx(
            task.mac_file,
            ct_shape,
            ct_spacing,
            ct_origin,
            mac_cache=mac_cache,
            NX=output_grid.nx * output_upscale_factor,
            NY=output_grid.ny * output_upscale_factor,
            NZ=output_grid.nz,
            spacing_dhw=(dx / output_upscale_factor, dy / output_upscale_factor, dz),
            plane_origin_offset_mm=output_grid.resolved_plane_origin_offset_mm,
            align_bev_z_to_ct_slices=True,
            align_orient_tol=output_grid.align_orient_tol,
            align_spacing_tol_mm=output_grid.align_spacing_tol_mm,
        )
        contexts.append(context)
        inverse_affines.append(
            torch.from_dlpack(cp.ascontiguousarray(context.affine)).unsqueeze(0)
        )

    def tensor(values) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(np.stack(values)), device=device, dtype=torch.float32
        )

    ct_affines_tensor = tensor(ct_affines)
    collapsed_ct, collapsed_ct_index = build_unique_collapsed_ct_cache(
        ct_coeff, ct_affines_tensor, input_grid.nz
    )
    collapsed_ct = collapsed_ct.to(collapsed_dtype)
    return FastPlanInferenceCache(
        masks=tensor(masks),
        collapsed_ct=collapsed_ct,
        collapsed_ct_index=collapsed_ct_index,
        ct_affines=ct_affines_tensor,
        world_affines=tensor(world_affines),
        ray_origins=tensor(ray_origins),
        aperture_origins=tensor(aperture_origins),
        bases=tensor(bases),
        inverse_affines=tuple(inverse_affines),
        backprojection_contexts=tuple(contexts),
    )


def fast_preprocess_ct_batch(
    cache: FastPlanInferenceCache,
    indices: torch.Tensor,
    *,
    output_shape: tuple[int, int, int],
    ct_min: float,
    ct_max: float,
    normalize_ct: bool,
    anatomy_cval: float = -1024.0,
    anatomy_clip: tuple[float, float] | None = (-1024.0, 3071.0),
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the normalized network input plus the raw anatomy volume.

    ``anatomy_cval`` / ``anatomy_clip`` carry the CT HU conventions by default.
    A non-CT input (``--ct-name mr.mha``) arrives already normalized to [0, 1],
    so it needs ``anatomy_cval=0.0`` and a matching clip; filling out-of-field
    with -1024 there would inject a value 1000x the in-body range straight into
    the physics conditioning.
    """
    collapsed = cache.collapsed_ct.index_select(
        0, cache.collapsed_ct_index.index_select(0, indices)
    )
    affine = cache.ct_affines.index_select(0, indices).clone()
    affine[:, 2].zero_()
    affine[:, 2, 2] = 1.0
    sampled, valid = _Bicubic2DAffineTriton.apply(collapsed, affine, output_shape)
    anatomy = sampled.masked_fill(~valid, float(anatomy_cval))
    if anatomy_clip is not None:
        anatomy = anatomy.clamp(float(anatomy_clip[0]), float(anatomy_clip[1]))
    if normalize_ct:
        model_ct = anatomy.clamp(float(ct_min), float(ct_max))
        model_ct = model_ct.sub(float(ct_min)).div(float(ct_max) - float(ct_min))
    else:
        model_ct = anatomy
    return model_ct.to(output_dtype), anatomy


def fast_preprocess_aperture_batch(
    cache: FastPlanInferenceCache,
    indices: torch.Tensor,
    *,
    output_shape: tuple[int, int, int],
    spacing_yz: tuple[float, float],
    sad_mm: float,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate proton fluence for selected beamlets in one Triton launch."""
    nx, ny, nz = (int(value) for value in output_shape)
    spacing_y, spacing_z = (float(value) for value in spacing_yz)
    batch = int(indices.numel())
    output = torch.empty(
        (batch, nx, ny, nz), device=indices.device, dtype=output_dtype
    )
    total = output.numel()
    _fast_aperture_kernel[(triton.cdiv(total, 256),)](
        cache.masks.index_select(0, indices),
        cache.world_affines.index_select(0, indices),
        cache.ray_origins.index_select(0, indices),
        cache.aperture_origins.index_select(0, indices),
        cache.bases.index_select(0, indices),
        output,
        nx,
        ny,
        nz,
        batch,
        float(sad_mm),
        1.0 / spacing_y,
        1.0 / spacing_z,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output
