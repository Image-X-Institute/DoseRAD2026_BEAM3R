"""Compact, cached CUDA preprocessing for plan inference.

The reference path materialises world and CT query coordinates for every
control point.  Here the fixed transforms are stored as 3x4 affine matrices,
the aligned CT z planes are collapsed once, and Triton generates model inputs
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cupy as cp
import numpy as np
import torch
import triton
import triton.language as tl

from . import bev_build, geometry
from .bev_grid import cached_bev_offset
from .ct_sample import _Bicubic2DAffineTriton, _mirror


@dataclass
class FastPlanCache:
    masks: torch.Tensor
    # Maps each metadata/control-point row onto compact ray geometry. Proton
    # energy layers commonly share a ray, so affine/context state is stored
    # once per ray rather than once per energy.
    geometry_index: torch.Tensor
    segment_geometry_index: tuple[int, ...]
    collapsed_ct: torch.Tensor
    collapsed_ct_index: torch.Tensor
    ct_affines: torch.Tensor
    world_affines: torch.Tensor
    ray_origins: torch.Tensor
    aperture_origins: torch.Tensor
    bases: torch.Tensor
    inverse_affines: tuple[torch.Tensor, ...]
    backprojection_contexts: tuple[Any, ...]


def _segment_geometry_numpy(
    record: dict[str, Any],
    nx: int,
    ny: int,
    nz: int,
    *,
    spacing_dhw: tuple[float, float, float],
    ct_origin: tuple[float, float, float] | None = None,
    ct_spacing: tuple[float, float, float] | None = None,
    align_bev_z_to_ct_slices: bool,
    align_orient_tol: float,
    align_spacing_tol_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy equivalent of ``bev_build.segment_geometry`` for cache setup."""
    basis = np.asarray(record["U"], dtype=np.float32).reshape(3, 3)
    plane_origin = np.asarray(record["src"], dtype=np.float32)
    offset = np.asarray(
        cached_bev_offset(record, nx, ny, nz), dtype=np.float32
    )
    if (
        align_bev_z_to_ct_slices
        and ct_origin is not None
        and ct_spacing is not None
    ):
        uz = basis[2].astype(np.float64)
        dz = float(spacing_dhw[2])
        spacing_z = float(ct_spacing[2])
        if (
            abs(float(uz[2])) >= 1.0 - float(align_orient_tol)
            and abs(dz - spacing_z) <= float(align_spacing_tol_mm)
        ):
            reference_k = nz // 2
            local_z = (float(reference_k) + float(offset[2])) * dz
            world_z = float(plane_origin[2]) + local_z * float(uz[2])
            origin_z = float(ct_origin[2])
            target_z = (
                round((world_z - origin_z) / spacing_z) * spacing_z
                + origin_z
            )
            plane_origin = plane_origin.copy()
            plane_origin[2] += np.float32(target_z - world_z)
    return basis, plane_origin, offset


def _back_projection_context_numpy(
    record: dict[str, Any],
    ct_shape: tuple[int, int, int],
    ct_spacing: tuple[float, float, float],
    ct_origin: tuple[float, float, float],
    *,
    nx: int,
    ny: int,
    nz: int,
    spacing_dhw: tuple[float, float, float],
    align_orient_tol: float,
    align_spacing_tol_mm: float,
) -> Any:
    """NumPy equivalent of the CuPy inverse-affine/ROI cache builder."""
    basis, plane_origin, offset = _segment_geometry_numpy(
        record,
        nx,
        ny,
        nz,
        spacing_dhw=spacing_dhw,
        ct_origin=ct_origin,
        ct_spacing=ct_spacing,
        align_bev_z_to_ct_slices=True,
        align_orient_tol=align_orient_tol,
        align_spacing_tol_mm=align_spacing_tol_mm,
    )
    spacing = np.asarray(spacing_dhw, dtype=np.float32)
    ct_spacing_array = np.asarray(ct_spacing, dtype=np.float32)
    ct_origin_array = np.asarray(ct_origin, dtype=np.float32)
    corners = np.asarray(
        (
            (0, 0, 0),
            (nx - 1, 0, 0),
            (0, ny - 1, 0),
            (0, 0, nz - 1),
            (nx - 1, ny - 1, 0),
            (nx - 1, 0, nz - 1),
            (0, ny - 1, nz - 1),
            (nx - 1, ny - 1, nz - 1),
        ),
        dtype=np.float32,
    )
    world = plane_origin + ((corners + offset) * spacing) @ basis
    ct_indices = (world - ct_origin_array) / ct_spacing_array
    lo = np.floor(ct_indices.min(axis=0)) - np.float32(1.0)
    hi = np.ceil(ct_indices.max(axis=0)) + np.float32(1.0)
    shape_limit = np.asarray(ct_shape, dtype=np.int32) - 1
    lo_i = np.clip(lo, 0, shape_limit).astype(np.int32)
    hi_i = np.clip(hi, 0, shape_limit).astype(np.int32)
    starts = tuple(int(value) for value in lo_i)
    stops = tuple(int(value) + 1 for value in hi_i)

    inverse_scale = np.asarray(1.0 / spacing, dtype=np.float32)
    matrix = basis.T * inverse_scale[:, None]
    constant = ((ct_origin_array - plane_origin) @ matrix) - offset
    cx = ct_spacing_array[0] * matrix[0]
    cy = ct_spacing_array[1] * matrix[1]
    cz = ct_spacing_array[2] * matrix[2]
    local_constant = (
        constant
        + cx * starts[0]
        + cy * starts[1]
        + cz * starts[2]
    )
    affine = np.stack(
        (cx, cy, cz, local_constant), axis=1
    ).astype(np.float32, copy=False)
    roi_shape = tuple(stops[index] - starts[index] for index in range(3))
    roi_box = (
        starts[0],
        stops[0],
        starts[1],
        stops[1],
        starts[2],
        stops[2],
    )
    return bev_build.AffineBackCtx(
        affine=affine, roi_shape=roi_shape, roi_box=roi_box
    )


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


def _build_collapsed_ct(
    ct_coeff: torch.Tensor,
    ct_affines: torch.Tensor,
    nz: int,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one collapsed CT tensor for each distinct aligned z sequence."""
    k = torch.arange(nz, device=ct_affines.device, dtype=torch.float32)
    rows = ct_affines[:, 2]
    if float(rows[:, :2].abs().max().item()) > 1.0e-4:
        raise ValueError("CT affine is not z aligned")
    zq = rows[:, 2:3] * k.unsqueeze(0) + rows[:, 3:4]
    zi_all = zq.round().to(torch.int32)
    if float((zq - zi_all).abs().max().item()) > 1.0e-3:
        raise ValueError("CT affine does not map to integer z planes")

    keys = zi_all.cpu().numpy()
    representatives: list[int] = []
    inverse: list[int] = []
    seen: dict[bytes, int] = {}
    for index, key in enumerate(keys):
        encoded = np.ascontiguousarray(key).tobytes()
        unique_index = seen.get(encoded)
        if unique_index is None:
            unique_index = len(representatives)
            seen[encoded] = unique_index
            representatives.append(index)
        inverse.append(unique_index)

    z_size = int(ct_coeff.shape[2])
    planes = []
    for index in representatives:
        zi = zi_all[index].to(torch.long)
        in_range = (zi >= 0) & (zi < z_size)
        zc = zi.clamp(0, z_size - 1)
        zm1 = torch.where(
            zc > 0, zc - 1, torch.full_like(zc, min(1, z_size - 1))
        )
        zp1 = torch.where(
            zc < z_size - 1,
            zc + 1,
            torch.full_like(zc, max(z_size - 2, 0)),
        )
        collapsed = (
            ct_coeff[:, :, zm1]
            + 4.0 * ct_coeff[:, :, zc]
            + ct_coeff[:, :, zp1]
        ) * (1.0 / 6.0)
        collapsed = collapsed.permute(2, 0, 1).contiguous()
        collapsed[~in_range] = -1024.0
        planes.append(collapsed.permute(1, 2, 0).contiguous())
    return (
        torch.stack(planes).to(output_dtype),
        torch.as_tensor(inverse, device=ct_affines.device, dtype=torch.long),
    )


def build_plan_cache(
    segments: Sequence[geometry.PhotonSegment | geometry.ProtonSegment],
    ct: Any,
    grid: Any,
    device: torch.device,
    *,
    output_upscale_factor: int = 2,
    output_grid: Any | None = None,
    collapsed_dtype: torch.dtype = torch.bfloat16,
    proton_spot_sigmas_mm: Sequence[float] | None = None,
    deduplicate_proton_geometry: bool = True,
) -> FastPlanCache:
    """Prepare all reusable input and backprojection state for a plan."""
    if not segments:
        raise ValueError("cannot build a plan cache without control points")
    output_grid = grid if output_grid is None else output_grid
    spacing = np.asarray(grid.spacing_dhw, dtype=np.float32)
    ct_spacing = np.asarray(ct.spacing, dtype=np.float32)
    ct_origin = np.asarray(ct.origin, dtype=np.float32)

    masks = []
    segment_geometry_index: list[int] = []
    geometry_by_key: dict[tuple[Any, ...], int] = {}
    ct_affines = []
    world_affines = []
    ray_origins = []
    aperture_origins = []
    bases = []
    inverse_affines = []
    contexts = []
    proton_mask_cache: dict[float, np.ndarray] = {}
    proton_radius2: np.ndarray | None = None
    if any(isinstance(segment, geometry.ProtonSegment) for segment in segments):
        y = (
            np.arange(grid.ny, dtype=np.float32) - grid.ny // 2 + 0.5
        ) * float(grid.spacing_dhw[1])
        z = (
            np.arange(grid.nz, dtype=np.float32) - grid.nz // 2 + 0.5
        ) * float(grid.spacing_dhw[2])
        proton_radius2 = y[:, None] ** 2 + z[None, :] ** 2

    for segment_index, segment in enumerate(segments):
        if isinstance(segment, geometry.ProtonSegment):
            sigma = segment.sigma_spot_mm
            if proton_spot_sigmas_mm is not None:
                sigma = float(proton_spot_sigmas_mm[segment_index])
            if sigma is None or float(sigma) <= 0.0:
                raise ValueError(f"{segment.name}: missing positive proton spot width")
            sigma_value = float(sigma)
            resized = proton_mask_cache.get(sigma_value)
            if resized is None:
                assert proton_radius2 is not None
                resized = np.exp(
                    -0.5 * proton_radius2 / (sigma_value ** 2)
                ).astype(np.float32)
                proton_mask_cache[sigma_value] = resized
            geometry_key = (
                (
                    "proton",
                    int(segment.beam_idx),
                    int(segment.ray_idx),
                    tuple(float(value) for value in segment.ray_source),
                    tuple(float(value) for value in segment.ray_target),
                )
                if deduplicate_proton_geometry
                else ("proton-segment", segment_index)
            )
        else:
            aperture = geometry.build_aperture(
                segment.mlc_left_mm, segment.mlc_right_mm
            )
            resized = np.ascontiguousarray(
                geometry.resize_aperture(aperture, grid.ny, grid.nz),
                dtype=np.float32,
            )
            # Photon control points may carry aperture state not represented in
            # their gantry geometry. Retain one geometry row per segment here;
            # the proton path is the measured repeated-ray workload.
            geometry_key = ("photon", segment_index)
        masks.append(resized)

        geometry_index = geometry_by_key.get(geometry_key)
        if geometry_index is not None:
            segment_geometry_index.append(geometry_index)
            continue

        geometry_index = len(ct_affines)
        geometry_by_key[geometry_key] = geometry_index
        segment_geometry_index.append(geometry_index)
        if isinstance(segment, geometry.ProtonSegment):
            record = geometry.proton_mac_cache_entry(
                segment.ray_source, segment.ray_target, grid
            )
        else:
            record = geometry.mac_cache_entry(
                segment.iso_center, segment.gantry_angle, grid
            )
        basis, aligned_origin, offset = _segment_geometry_numpy(
            record,
            grid.nx,
            grid.ny,
            grid.nz,
            spacing_dhw=grid.spacing_dhw,
            ct_origin=ct.origin,
            ct_spacing=ct.spacing,
            align_bev_z_to_ct_slices=True,
            align_orient_tol=grid.align_orient_tol,
            align_spacing_tol_mm=grid.align_spacing_tol_mm,
        )
        _, physical_origin, _ = _segment_geometry_numpy(
            record,
            grid.nx,
            grid.ny,
            grid.nz,
            spacing_dhw=grid.spacing_dhw,
            align_bev_z_to_ct_slices=False,
            align_orient_tol=grid.align_orient_tol,
            align_spacing_tol_mm=grid.align_spacing_tol_mm,
        )
        linear = (spacing[:, None] * basis).T
        constant = aligned_origin + (offset * spacing) @ basis
        world_affine = np.concatenate((linear, constant[:, None]), axis=1)
        ct_affine = world_affine.copy()
        ct_affine[:, :3] /= ct_spacing[:, None]
        ct_affine[:, 3] = (constant - ct_origin) / ct_spacing
        ct_affines.append(ct_affine)
        world_affines.append(world_affine)
        ray_origins.append(np.asarray(record["s"], np.float32))
        aperture_origins.append(physical_origin)
        bases.append(basis)

        dx, dy, dz = (float(value) for value in output_grid.spacing_dhw)
        context = _back_projection_context_numpy(
            record,
            ct.shape,
            ct.spacing,
            ct.origin,
            nx=output_grid.nx * output_upscale_factor,
            ny=output_grid.ny * output_upscale_factor,
            nz=output_grid.nz,
            spacing_dhw=(
                dx / output_upscale_factor,
                dy / output_upscale_factor,
                dz,
            ),
            align_orient_tol=output_grid.align_orient_tol,
            align_spacing_tol_mm=output_grid.align_spacing_tol_mm,
        )
        contexts.append(context)
        inverse_affines.append(context.affine)

    def as_tensor(values: list[np.ndarray]) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(np.stack(values)),
            device=device,
            dtype=torch.float32,
        )

    ct_affines_tensor = as_tensor(ct_affines)
    ct_coeff = torch.from_dlpack(ct.coeff)
    collapsed_ct, collapsed_ct_index = _build_collapsed_ct(
        ct_coeff,
        ct_affines_tensor,
        grid.nz,
        output_dtype=collapsed_dtype,
    )
    inverse_affines_tensor = as_tensor(inverse_affines)
    return FastPlanCache(
        masks=as_tensor(masks),
        geometry_index=torch.as_tensor(
            segment_geometry_index, device=device, dtype=torch.long
        ),
        segment_geometry_index=tuple(segment_geometry_index),
        collapsed_ct=collapsed_ct,
        collapsed_ct_index=collapsed_ct_index,
        ct_affines=ct_affines_tensor,
        world_affines=as_tensor(world_affines),
        ray_origins=as_tensor(ray_origins),
        aperture_origins=as_tensor(aperture_origins),
        bases=as_tensor(bases),
        inverse_affines=tuple(
            affine.unsqueeze(0) for affine in inverse_affines_tensor
        ),
        backprojection_contexts=tuple(contexts),
    )


def preprocess_batch(
    cache: FastPlanCache,
    indices: torch.Tensor,
    *,
    output_shape: tuple[int, int, int],
    ct_min: float,
    ct_max: float,
    output_dtype: torch.dtype = torch.bfloat16,
    spacing_yz: tuple[float, float] = (2.0, 2.0),
    sad_mm: float = 1000.0,
    return_hu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate normalized CT and aperture inputs without dense coordinates."""
    ct_out, ct_hu = preprocess_ct_batch(
        cache,
        indices,
        output_shape=output_shape,
        ct_min=ct_min,
        ct_max=ct_max,
        output_dtype=output_dtype,
    )
    aperture_out = preprocess_aperture_batch(
        cache,
        indices,
        output_shape=output_shape,
        output_dtype=output_dtype,
        spacing_yz=spacing_yz,
        sad_mm=sad_mm,
    )
    if return_hu:
        return ct_out, aperture_out, ct_hu
    return ct_out, aperture_out


def preprocess_ct_batch(
    cache: FastPlanCache,
    indices: torch.Tensor,
    *,
    output_shape: tuple[int, int, int],
    ct_min: float,
    ct_max: float,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample normalized CT and HU, independently of beamlet fluence."""
    geometry_indices = cache.geometry_index.index_select(0, indices)
    collapsed = cache.collapsed_ct.index_select(
        0, cache.collapsed_ct_index.index_select(0, geometry_indices)
    )
    ct_affine = cache.ct_affines.index_select(0, geometry_indices)
    sampling_affine = ct_affine.clone()
    sampling_affine[:, 2].zero_()
    sampling_affine[:, 2, 2] = 1.0
    ct_sample, valid = _Bicubic2DAffineTriton.apply(
        collapsed, sampling_affine, output_shape
    )
    ct_hu = (
        ct_sample.masked_fill(~valid, -1024.0)
        .clamp_min(-1024.0)
        .clamp(float(ct_min), float(ct_max))
    )
    ct_out = (
        ct_hu
        .sub(float(ct_min))
        .div(float(ct_max) - float(ct_min))
        .to(output_dtype)
    )
    return ct_out, ct_hu


def preprocess_aperture_batch(
    cache: FastPlanCache,
    indices: torch.Tensor,
    *,
    output_shape: tuple[int, int, int],
    output_dtype: torch.dtype = torch.bfloat16,
    spacing_yz: tuple[float, float] = (2.0, 2.0),
    sad_mm: float = 1000.0,
) -> torch.Tensor:
    """Generate beamlet fluence without resampling the ray's CT again."""
    nx, ny, nz = output_shape
    spacing_y, spacing_z = (float(value) for value in spacing_yz)
    if spacing_y <= 0.0 or spacing_z <= 0.0:
        raise ValueError("spacing_yz values must be positive")
    batch = int(indices.numel())
    geometry_indices = cache.geometry_index.index_select(0, indices)
    aperture_out = torch.empty(
        (batch, nx, ny, nz), device=indices.device, dtype=output_dtype
    )
    total = aperture_out.numel()
    _fast_aperture_kernel[(triton.cdiv(total, 256),)](
        cache.masks.index_select(0, indices),
        cache.world_affines.index_select(0, geometry_indices),
        cache.ray_origins.index_select(0, geometry_indices),
        cache.aperture_origins.index_select(0, geometry_indices),
        cache.bases.index_select(0, geometry_indices),
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
    return aperture_out


@torch.inference_mode()
def warmup_fast_preprocess(
    *,
    output_shape: tuple[int, int, int],
    device: torch.device,
    output_dtype: torch.dtype,
    spacing_yz: tuple[float, float],
    sad_mm: float,
    packed_encoder_upscale_factor: int | None = None,
) -> None:
    """Compile fixed-shape preprocessing kernels without patient input.

    ``packed_encoder_upscale_factor`` selects the photon packed-encoder
    producer instead of the separate CT/aperture pair, so the kernel warmed
    here is the one the first ``/invoke`` will actually launch.
    """
    nx, ny, nz = (int(value) for value in output_shape)
    affine = torch.tensor(
        ((1.0, 0.0, 0.0, 0.0),
         (0.0, 1.0, 0.0, 0.0),
         (0.0, 0.0, 1.0, 0.0)),
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    cache = FastPlanCache(
        masks=torch.ones((1, ny, nz), device=device, dtype=torch.float32),
        geometry_index=torch.zeros(1, device=device, dtype=torch.long),
        segment_geometry_index=(0,),
        collapsed_ct=torch.zeros(
            (1, 8, 8, 8), device=device, dtype=torch.float32
        ),
        collapsed_ct_index=torch.zeros(1, device=device, dtype=torch.long),
        ct_affines=affine,
        world_affines=affine,
        ray_origins=torch.tensor(
            ((-float(sad_mm), 0.0, 0.0),),
            device=device,
            dtype=torch.float32,
        ),
        aperture_origins=torch.zeros((1, 3), device=device, dtype=torch.float32),
        bases=torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0),
        inverse_affines=(),
        backprojection_contexts=(),
    )
    indices = torch.zeros(1, device=device, dtype=torch.long)
    if packed_encoder_upscale_factor is None:
        ct_out, ct_hu = preprocess_ct_batch(
            cache,
            indices,
            output_shape=output_shape,
            ct_min=-1024.0,
            ct_max=3071.0,
            output_dtype=output_dtype,
        )
        aperture = preprocess_aperture_batch(
            cache,
            indices,
            output_shape=output_shape,
            output_dtype=output_dtype,
            spacing_yz=spacing_yz,
            sad_mm=sad_mm,
        )
        outputs = (ct_out, ct_hu, aperture)
    else:
        outputs = (
            preprocess_packed_encoder_batch(
                cache,
                indices,
                output_shape=output_shape,
                upscale_factor=int(packed_encoder_upscale_factor),
                ct_min=-1024.0,
                ct_max=3071.0,
                output_dtype=output_dtype,
                spacing_yz=spacing_yz,
                sad_mm=sad_mm,
            ),
        )
    # Force asynchronous launches to finish before /health can report ready.
    torch.cuda.synchronize(device)
    del outputs, indices, cache


# -- photon packed-encoder fast path ---------------------------------------
#
# Builds the combined channels-last encoder input in one pass for the photon
# CNN-Mamba path. The proton predictors use preprocess_ct_batch /
# preprocess_aperture_batch above instead, because they need the intermediate
# HU volume for the RSP material conversion.

@triton.jit
def _packed_channels_last_offset(
    b,
    i,
    j,
    k,
    coarse_depth,
    coarse_height,
    width,
    upscale_factor: tl.constexpr,
    output_channels: tl.constexpr,
    channel_offset: tl.constexpr,
):
    """Map one fine DH voxel into the combined packed NHWC encoder buffer."""
    coarse_i = i // upscale_factor
    coarse_j = j // upscale_factor
    phase = (i % upscale_factor) * upscale_factor + (j % upscale_factor)
    row = b * coarse_depth + coarse_i
    return (
        ((row * coarse_height + coarse_j) * width + k) * output_channels
        + channel_offset
        + phase
    )

@triton.jit
def _fast_aperture_packed_encoder_kernel(
    masks_ptr,
    world_affine_ptr,
    ray_ptr,
    aperture_origin_ptr,
    basis_ptr,
    out_ptr,
    fine_depth,
    fine_height,
    width,
    batch,
    sad,
    aperture_inv_spacing_y,
    aperture_inv_spacing_z,
    upscale_factor: tl.constexpr,
    output_channels: tl.constexpr,
    channel_offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Project aperture directly into the second half of packed encoder input."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    fine_voxels = fine_depth * fine_height * width
    total = batch * fine_voxels
    mask = offsets < total
    b = offsets // fine_voxels
    local = offsets - b * fine_voxels
    k = local % width
    q = local // width
    j = q % fine_height
    i = q // fine_height
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
        + (fine_height // 2 - 0.5)
    )
    zq = (
        (wx * uz0 + wy * uz1 + wz * uz2) * aperture_inv_spacing_z
        + (width // 2 - 0.5)
    )
    y0 = tl.floor(yq).to(tl.int32)
    z0 = tl.floor(zq).to(tl.int32)
    valid &= (
        (yq >= 0.0)
        & (yq < fine_height - 1)
        & (zq >= 0.0)
        & (zq < width - 1)
    )
    fy = yq - y0.to(tl.float32)
    fz = zq - z0.to(tl.float32)
    source = b * fine_height * width + y0 * width + z0
    v00 = tl.load(masks_ptr + source, mask=valid, other=0.0)
    v10 = tl.load(masks_ptr + source + width, mask=valid, other=0.0)
    v01 = tl.load(masks_ptr + source + 1, mask=valid, other=0.0)
    v11 = tl.load(masks_ptr + source + width + 1, mask=valid, other=0.0)
    value = (
        (1.0 - fy) * ((1.0 - fz) * v00 + fz * v01)
        + fy * ((1.0 - fz) * v10 + fz * v11)
    )
    destination = _packed_channels_last_offset(
        b,
        i,
        j,
        k,
        fine_depth // upscale_factor,
        fine_height // upscale_factor,
        width,
        upscale_factor,
        output_channels,
        channel_offset,
    )
    tl.store(out_ptr + destination, value, mask=mask)

@triton.jit
def _normalised_bicubic_packed_encoder_kernel(
    coeff_ptr,
    affine_ptr,
    out_ptr,
    nT,
    nH,
    nW,
    fine_depth,
    fine_height,
    width,
    N,
    ct_min,
    ct_max,
    inverse_ct_range,
    upscale_factor: tl.constexpr,
    output_channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample, clamp and normalize CT directly into packed channels-last input."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    fine_voxels = fine_depth * fine_height * width
    b = offsets // fine_voxels
    local = offsets - b * fine_voxels
    k = local % width
    q = local // width
    j = q % fine_height
    i = q // fine_height
    affine_base = b * 12
    x = i.to(tl.float32)
    y = j.to(tl.float32)
    z = k.to(tl.float32)
    tq = (
        tl.load(affine_ptr + affine_base + 0, mask=mask) * x
        + tl.load(affine_ptr + affine_base + 1, mask=mask) * y
        + tl.load(affine_ptr + affine_base + 2, mask=mask) * z
        + tl.load(affine_ptr + affine_base + 3, mask=mask)
    )
    hq = (
        tl.load(affine_ptr + affine_base + 4, mask=mask) * x
        + tl.load(affine_ptr + affine_base + 5, mask=mask) * y
        + tl.load(affine_ptr + affine_base + 6, mask=mask) * z
        + tl.load(affine_ptr + affine_base + 7, mask=mask)
    )
    wq = (
        tl.load(affine_ptr + affine_base + 8, mask=mask) * x
        + tl.load(affine_ptr + affine_base + 9, mask=mask) * y
        + tl.load(affine_ptr + affine_base + 10, mask=mask) * z
        + tl.load(affine_ptr + affine_base + 11, mask=mask)
    )
    wi = tl.floor(wq + 0.5).to(tl.int32)
    valid = (
        mask
        & (tq >= 0.0)
        & (tq <= nT - 1)
        & (hq >= 0.0)
        & (hq <= nH - 1)
        & (wi >= 0)
        & (wi < nW)
        & (tl.abs(wq - wi.to(tl.float32)) <= 1.0e-3)
    )
    t0 = tl.floor(tq).to(tl.int32)
    h0 = tl.floor(hq).to(tl.int32)
    ft = tq - t0.to(tq.dtype)
    fh = hq - h0.to(hq.dtype)
    period_t = 2 * (nT - 1) + 2 * (nT == 1)
    period_h = 2 * (nH - 1) + 2 * (nH == 1)
    acc = tl.zeros((BLOCK_SIZE,), tl.float32)
    wi_safe = tl.maximum(0, tl.minimum(wi, nW - 1))
    for a in range(4):
        t = ft
        if a == 0:
            wt = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
        elif a == 1:
            t2 = t * t
            wt = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
        elif a == 2:
            t2 = t * t
            wt = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) / 6.0
        else:
            wt = t * t * t / 6.0
        ti = tl.where(
            nT > 1, _mirror(t0 - 1 + a, period_t, nT - 1), 0
        )
        for htap in range(4):
            t = fh
            if htap == 0:
                wh = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
            elif htap == 1:
                t2 = t * t
                wh = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
            elif htap == 2:
                t2 = t * t
                wh = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) / 6.0
            else:
                wh = t * t * t / 6.0
            hi = tl.where(
                nH > 1, _mirror(h0 - 1 + htap, period_h, nH - 1), 0
            )
            source = ((b * nT + ti) * nH + hi) * nW + wi_safe
            acc += wt * wh * tl.load(
                coeff_ptr + source, mask=valid, other=0.0
            )

    # Match preprocess_batch exactly: invalid samples become air, values below
    # air are clipped, then the run's CT clipping and normalization are applied.
    value = tl.where(valid, acc, -1024.0)
    value = tl.maximum(value, -1024.0)
    value = tl.maximum(ct_min, tl.minimum(value, ct_max))
    value = (value - ct_min) * inverse_ct_range
    destination = _packed_channels_last_offset(
        b,
        i,
        j,
        k,
        fine_depth // upscale_factor,
        fine_height // upscale_factor,
        width,
        upscale_factor,
        output_channels,
        0,
    )
    tl.store(out_ptr + destination, value, mask=mask)

def preprocess_packed_encoder_batch(
    cache: FastPlanCache,
    indices: torch.Tensor,
    *,
    output_shape: tuple[int, int, int],
    upscale_factor: int,
    ct_min: float,
    ct_max: float,
    output_dtype: torch.dtype = torch.bfloat16,
    spacing_yz: tuple[float, float] = (2.0, 2.0),
    sad_mm: float = 1000.0,
) -> torch.Tensor:
    """Generate the complete phase-packed channels-last Mamba encoder input.

    Unlike :func:`preprocess_batch`, this inference-only path never
    materializes fine CT/aperture volumes, a validity tensor, normalized CT
    intermediates, two phase-packing copies, or the model's subsequent
    concatenate-and-layout copy. Both Triton producers write directly into one
    logical ``(B*D, 2*r**2, H, W)`` channels-last tensor.
    """
    fine_depth, fine_height, width = (int(value) for value in output_shape)
    r = int(upscale_factor)
    if r < 1 or fine_depth % r or fine_height % r:
        raise ValueError(
            f"output_shape={output_shape} is not divisible by upscale_factor={r}"
        )
    spacing_y, spacing_z = (float(value) for value in spacing_yz)
    if spacing_y <= 0.0 or spacing_z <= 0.0:
        raise ValueError("spacing_yz values must be positive")
    ct_min_value = float(ct_min)
    ct_max_value = float(ct_max)
    if ct_max_value <= ct_min_value:
        raise ValueError("ct_max must be greater than ct_min")
    batch = int(indices.numel())
    if batch <= 0:
        raise ValueError("indices cannot be empty")

    phases = r * r
    output_channels = 2 * phases
    coarse_depth = fine_depth // r
    coarse_height = fine_height // r
    encoder_input = torch.empty(
        (batch * coarse_depth, output_channels, coarse_height, width),
        device=cache.collapsed_ct.device,
        dtype=output_dtype,
        memory_format=torch.channels_last,
    )

    collapsed = cache.collapsed_ct.index_select(
        0, cache.collapsed_ct_index.index_select(0, indices)
    )
    ct_affine = cache.ct_affines.index_select(0, indices)
    sampling_affine = ct_affine.clone()
    sampling_affine[:, 2].zero_()
    sampling_affine[:, 2, 2] = 1.0
    _batch, nT, nH, nW = collapsed.shape
    total = batch * fine_depth * fine_height * width
    _normalised_bicubic_packed_encoder_kernel[
        (triton.cdiv(total, 256),)
    ](
        collapsed,
        sampling_affine,
        encoder_input,
        nT,
        nH,
        nW,
        fine_depth,
        fine_height,
        width,
        total,
        ct_min_value,
        ct_max_value,
        1.0 / (ct_max_value - ct_min_value),
        upscale_factor=r,
        output_channels=output_channels,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    _fast_aperture_packed_encoder_kernel[
        (triton.cdiv(total, 256),)
    ](
        cache.masks.index_select(0, indices),
        cache.world_affines.index_select(0, indices),
        cache.ray_origins.index_select(0, indices),
        cache.aperture_origins.index_select(0, indices),
        cache.bases.index_select(0, indices),
        encoder_input,
        fine_depth,
        fine_height,
        width,
        batch,
        float(sad_mm),
        1.0 / spacing_y,
        1.0 / spacing_z,
        upscale_factor=r,
        output_channels=output_channels,
        channel_offset=phases,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return encoder_input
