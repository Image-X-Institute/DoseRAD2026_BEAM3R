"""Inference-only Triton fusions for proton material preprocessing.

The production reference path remains available in :mod:`proton_predict`.
These functions expose independently benchmarkable steps which preserve the
same fixed range/WET representation while avoiding expanded per-beamlet
material tensors and phase-packing/layout intermediates.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_MATERIAL_CLASS_COUNT = 7


def _require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@triton.jit
def _mirror_index(index, period, dimension_minus_one):
    reflected = index % period
    reflected = tl.where(reflected < 0, reflected + period, reflected)
    return tl.where(
        reflected > dimension_minus_one, period - reflected, reflected
    )


@triton.jit
def _subtract_rn_f32(left, right):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _multiply_rn_f32(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _divide_rn_f32(left, right):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit(
    do_not_specialize=[11, 12, 13, 14, 15, 16, 17, 18, 19]
)
def _sample_pack_ct_material_kernel(
    collapsed_ptr,
    geometry_index_ptr,
    collapsed_index_ptr,
    ct_affine_ptr,
    representative_index_ptr,
    ct_hu_ptr,
    ct_rho_ptr,
    class_upper_ptr,
    packed_ct_ptr,
    density_ptr,
    class_ptr,
    nT,
    nH,
    nW,
    fine_depth,
    fine_height,
    width,
    ct_min,
    ct_max,
    total,
    UPSCALE_FACTOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample CT and directly emit packed normalized CT plus material state."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    fine_voxels = fine_depth * fine_height * width
    batch_index = offsets // fine_voxels
    local = offsets - batch_index * fine_voxels
    width_index = local % width
    quotient = local // width
    height_index = quotient % fine_height
    depth_index = quotient // fine_height

    segment_index = tl.load(
        representative_index_ptr + batch_index, mask=mask, other=0
    )
    geometry_index = tl.load(
        geometry_index_ptr + segment_index, mask=mask, other=0
    )
    collapsed_index = tl.load(
        collapsed_index_ptr + geometry_index, mask=mask, other=0
    )
    affine_base = geometry_index * 12
    x = depth_index.to(tl.float32)
    y = height_index.to(tl.float32)
    z = width_index.to(tl.float32)
    tq = (
        tl.load(ct_affine_ptr + affine_base + 0, mask=mask) * x
        + tl.load(ct_affine_ptr + affine_base + 1, mask=mask) * y
        + tl.load(ct_affine_ptr + affine_base + 2, mask=mask) * z
        + tl.load(ct_affine_ptr + affine_base + 3, mask=mask)
    )
    hq = (
        tl.load(ct_affine_ptr + affine_base + 4, mask=mask) * x
        + tl.load(ct_affine_ptr + affine_base + 5, mask=mask) * y
        + tl.load(ct_affine_ptr + affine_base + 6, mask=mask) * z
        + tl.load(ct_affine_ptr + affine_base + 7, mask=mask)
    )
    # preprocess_ct_batch replaces the aligned affine's third row by an
    # identity mapping, so the collapsed CT plane index is the output W index.
    wq = z
    wi = width_index
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
    accumulated = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for depth_tap in range(4):
        t = ft
        if depth_tap == 0:
            weight_t = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
        elif depth_tap == 1:
            t2 = t * t
            weight_t = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
        elif depth_tap == 2:
            t2 = t * t
            weight_t = (
                1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t
            ) / 6.0
        else:
            weight_t = t * t * t / 6.0
        ti = tl.where(
            nT > 1,
            _mirror_index(t0 - 1 + depth_tap, period_t, nT - 1),
            0,
        )
        for height_tap in range(4):
            t = fh
            if height_tap == 0:
                weight_h = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
            elif height_tap == 1:
                t2 = t * t
                weight_h = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
            elif height_tap == 2:
                t2 = t * t
                weight_h = (
                    1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t
                ) / 6.0
            else:
                weight_h = t * t * t / 6.0
            hi = tl.where(
                nH > 1,
                _mirror_index(h0 - 1 + height_tap, period_h, nH - 1),
                0,
            )
            source = (
                ((collapsed_index * nT + ti) * nH + hi) * nW + wi
            )
            accumulated += weight_t * weight_h * tl.load(
                collapsed_ptr + source, mask=valid, other=0.0
            )

    hu = tl.where(valid, accumulated, -1024.0)
    hu = tl.minimum(
        tl.maximum(tl.maximum(hu, -1024.0), ct_min), ct_max
    )
    density_index = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for table_index in range(10):
        breakpoint = tl.load(ct_hu_ptr + table_index)
        density_index += hu > breakpoint
    density_index = tl.minimum(tl.maximum(density_index, 1), 9)
    hu0 = tl.load(ct_hu_ptr + density_index - 1)
    hu1 = tl.load(ct_hu_ptr + density_index)
    rho0 = tl.load(ct_rho_ptr + density_index - 1)
    rho1 = tl.load(ct_rho_ptr + density_index)
    interpolation = _divide_rn_f32(
        _subtract_rn_f32(hu, hu0),
        tl.maximum(_subtract_rn_f32(hu1, hu0), 1.0e-6),
    )
    density = _add_rn_f32(
        rho0,
        _multiply_rn_f32(
            interpolation, _subtract_rn_f32(rho1, rho0)
        ),
    )
    class_id = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for class_index in range(6):
        class_id += density > tl.load(class_upper_ptr + class_index)

    coarse_depth = depth_index // UPSCALE_FACTOR
    coarse_height = height_index // UPSCALE_FACTOR
    phase = (
        (depth_index % UPSCALE_FACTOR) * UPSCALE_FACTOR
        + height_index % UPSCALE_FACTOR
    )
    phases = UPSCALE_FACTOR * UPSCALE_FACTOR
    output_depth = fine_depth // UPSCALE_FACTOR
    output_height = fine_height // UPSCALE_FACTOR
    packed_offset = (
        (
            (
                (batch_index * output_depth + coarse_depth) * phases
                + phase
            )
            * output_height
            + coarse_height
        )
        * width
        + width_index
    )
    normalized_ct = (hu - ct_min) / (ct_max - ct_min)
    tl.store(packed_ct_ptr + packed_offset, normalized_ct, mask=mask)
    tl.store(density_ptr + offsets, density, mask=mask)
    tl.store(class_ptr + offsets, class_id, mask=mask)


def sample_packed_ct_material_from_plan_cache(
    cache,
    representative_indices: torch.Tensor,
    *,
    output_shape: tuple[int, int, int],
    upscale_factor: int,
    ct_min: float,
    ct_max: float,
    ct_hu: torch.Tensor,
    ct_rho: torch.Tensor,
    class_dens_upper: torch.Tensor,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample unique rays directly into cached packed CT/material tensors."""
    for name, tensor in (
        ("representative_indices", representative_indices),
        ("cache.geometry_index", cache.geometry_index),
        ("cache.collapsed_ct", cache.collapsed_ct),
        ("cache.collapsed_ct_index", cache.collapsed_ct_index),
        ("cache.ct_affines", cache.ct_affines),
        ("ct_hu", ct_hu),
        ("ct_rho", ct_rho),
        ("class_dens_upper", class_dens_upper),
    ):
        _require_cuda_contiguous(name, tensor)
    if representative_indices.ndim != 1 or representative_indices.dtype != torch.long:
        raise ValueError("representative_indices must be one-dimensional int64")
    unique_rays = int(representative_indices.numel())
    if unique_rays < 1:
        raise ValueError("at least one representative ray is required")
    fine_depth, fine_height, width = (int(value) for value in output_shape)
    r = int(upscale_factor)
    if r < 1 or fine_depth % r or fine_height % r:
        raise ValueError("fine dimensions must be divisible by upscale_factor")
    if float(ct_max) <= float(ct_min):
        raise ValueError("ct_max must exceed ct_min")
    if ct_hu.shape != (10,) or ct_rho.shape != (10,):
        raise ValueError("CT calibration tables must each contain 10 values")
    if class_dens_upper.shape != (_MATERIAL_CLASS_COUNT,):
        raise ValueError("material class table must contain seven values")
    packed_ct = torch.empty(
        (
            unique_rays,
            fine_depth // r,
            r * r,
            fine_height // r,
            width,
        ),
        device=representative_indices.device,
        dtype=output_dtype,
    )
    density = torch.empty(
        (unique_rays, fine_depth, fine_height, width),
        device=representative_indices.device,
        dtype=torch.float32,
    )
    class_id = torch.empty_like(density, dtype=torch.uint8)
    total = density.numel()
    _, nT, nH, nW = cache.collapsed_ct.shape
    _sample_pack_ct_material_kernel[(triton.cdiv(total, 256),)](
        cache.collapsed_ct,
        cache.geometry_index,
        cache.collapsed_ct_index,
        cache.ct_affines,
        representative_indices,
        ct_hu,
        ct_rho,
        class_dens_upper,
        packed_ct,
        density,
        class_id,
        nT,
        nH,
        nW,
        fine_depth,
        fine_height,
        width,
        float(ct_min),
        float(ct_max),
        total,
        UPSCALE_FACTOR=r,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return packed_ct, density, class_id


@torch.inference_mode()
def warmup_sample_packed_ct_material(
    *,
    output_shape: tuple[int, int, int],
    upscale_factor: int,
    unique_rays: int,
    ct_min: float,
    ct_max: float,
    ct_hu: torch.Tensor,
    ct_rho: torch.Tensor,
    class_dens_upper: torch.Tensor,
    device: torch.device,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile and allocate the maximum production CT/material workspace."""
    from types import SimpleNamespace

    width = int(output_shape[2])
    count = int(unique_rays)
    affine = torch.zeros((count, 3, 4), device=device, dtype=torch.float32)
    affine[:, 2, 2] = 1.0
    cache = SimpleNamespace(
        geometry_index=torch.arange(count, device=device, dtype=torch.long),
        collapsed_ct=torch.zeros(
            (count, 8, 8, width), device=device, dtype=torch.float32
        ),
        collapsed_ct_index=torch.arange(
            count, device=device, dtype=torch.long
        ),
        ct_affines=affine,
    )
    representatives = torch.arange(count, device=device, dtype=torch.long)
    outputs = sample_packed_ct_material_from_plan_cache(
        cache,
        representatives,
        output_shape=output_shape,
        upscale_factor=upscale_factor,
        ct_min=ct_min,
        ct_max=ct_max,
        ct_hu=ct_hu,
        ct_rho=ct_rho,
        class_dens_upper=class_dens_upper,
        output_dtype=output_dtype,
    )
    torch.cuda.synchronize(device)
    return outputs


@triton.jit
def _rsp_from_class_factors_kernel(
    density_ptr,
    class_ptr,
    ray_index_ptr,
    factor_ptr,
    output_ptr,
    voxels_per_ray: tl.constexpr,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < total
    batch_index = offsets // voxels_per_ray
    local_index = offsets - batch_index * voxels_per_ray
    ray_index = tl.load(
        ray_index_ptr + batch_index, mask=valid, other=0
    )
    source_offset = ray_index * voxels_per_ray + local_index
    density = tl.load(density_ptr + source_offset, mask=valid, other=0.0).to(
        tl.float32
    )
    class_id = tl.load(class_ptr + source_offset, mask=valid, other=0).to(tl.int32)
    factor = tl.load(
        factor_ptr + batch_index * 7 + class_id,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(output_ptr + offsets, density * factor, mask=valid)


def rsp_from_class_factors(
    density_by_ray: torch.Tensor,
    class_id_by_ray: torch.Tensor,
    ray_indices: torch.Tensor,
    class_factors: torch.Tensor,
) -> torch.Tensor:
    """Expand compact unique-ray material state into per-energy FP32 RSP.

    ``density_by_ray`` and ``class_id_by_ray`` are ``(U,D,H,W)`` while
    ``ray_indices`` maps each of the ``B`` energies to one of those ``U`` rays.
    ``class_factors`` is the compact ``(B,7)`` table returned by
    ``BEVMaterialConditioning.rsp_class_factors``.
    """
    _require_cuda_contiguous("density_by_ray", density_by_ray)
    _require_cuda_contiguous("class_id_by_ray", class_id_by_ray)
    _require_cuda_contiguous("ray_indices", ray_indices)
    _require_cuda_contiguous("class_factors", class_factors)
    if density_by_ray.ndim != 4 or density_by_ray.shape != class_id_by_ray.shape:
        raise ValueError("density/class state must have matching (U,D,H,W) shapes")
    if class_id_by_ray.dtype != torch.uint8:
        raise ValueError("class_id_by_ray must use uint8 material classes")
    if ray_indices.ndim != 1 or ray_indices.dtype != torch.long:
        raise ValueError("ray_indices must be a one-dimensional int64 tensor")
    batch = int(ray_indices.numel())
    if class_factors.shape != (batch, _MATERIAL_CLASS_COUNT):
        raise ValueError(
            f"expected class_factors ({batch},{_MATERIAL_CLASS_COUNT}), got "
            f"{tuple(class_factors.shape)}"
        )
    if density_by_ray.shape[0] < 1:
        raise ValueError("at least one unique ray is required")
    output = torch.empty(
        (batch, *density_by_ray.shape[1:]),
        device=density_by_ray.device,
        dtype=torch.float32,
    )
    voxels_per_ray = density_by_ray[0].numel()
    total = output.numel()
    _rsp_from_class_factors_kernel[(triton.cdiv(total, 256),)](
        density_by_ray,
        class_id_by_ray,
        ray_indices,
        class_factors,
        output,
        voxels_per_ray=voxels_per_ray,
        total=total,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


@triton.jit
def _pack_wet_remaining_kernel(
    wet_ptr,
    range_ptr,
    output_ptr,
    fine_depth: tl.constexpr,
    fine_height: tl.constexpr,
    width: tl.constexpr,
    phases: tl.constexpr,
    upscale_factor: tl.constexpr,
    normalisation_mm: tl.constexpr,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < total
    fine_voxels = fine_depth * fine_height * width
    batch_index = offsets // fine_voxels
    local = offsets - batch_index * fine_voxels
    width_index = local % width
    quotient = local // width
    height_index = quotient % fine_height
    depth_index = quotient // fine_height

    coarse_depth = depth_index // upscale_factor
    coarse_height = height_index // upscale_factor
    phase = (
        (depth_index % upscale_factor) * upscale_factor
        + height_index % upscale_factor
    )
    output_depth = fine_depth // upscale_factor
    output_height = fine_height // upscale_factor
    output_base = (
        (
            (
                batch_index * output_depth + coarse_depth
            ) * (2 * phases)
        ) * output_height
        + coarse_height
    ) * width + width_index
    channel_stride = output_height * width

    wet_mm = tl.load(wet_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    range_mm = tl.load(range_ptr + batch_index, mask=valid, other=0.0).to(
        tl.float32
    )
    wet = tl.minimum(tl.maximum(wet_mm / normalisation_mm, 0.0), 2.0)
    remaining = tl.minimum(
        tl.maximum((range_mm - wet_mm) / normalisation_mm, -2.0), 1.0
    )
    tl.store(
        output_ptr + output_base + phase * channel_stride,
        wet,
        mask=valid,
    )
    tl.store(
        output_ptr + output_base + (phases + phase) * channel_stride,
        remaining,
        mask=valid,
    )


def pack_wet_remaining(
    wet_mm: torch.Tensor,
    range_mm: torch.Tensor,
    *,
    upscale_factor: int,
    normalisation_mm: float = 300.0,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Clamp and phase-pack WET/remaining range without FP32 intermediates."""
    _require_cuda_contiguous("wet_mm", wet_mm)
    _require_cuda_contiguous("range_mm", range_mm)
    if wet_mm.ndim != 4:
        raise ValueError("wet_mm must have shape (B,D,H,W)")
    batch, fine_depth, fine_height, width = wet_mm.shape
    if range_mm.shape != (batch,):
        raise ValueError(f"range_mm must have shape ({batch},)")
    r = int(upscale_factor)
    if r < 1 or fine_depth % r or fine_height % r:
        raise ValueError("fine WET dimensions must be divisible by upscale_factor")
    if normalisation_mm <= 0.0:
        raise ValueError("normalisation_mm must be positive")
    phases = r * r
    output = torch.empty(
        (batch, fine_depth // r, 2 * phases, fine_height // r, width),
        device=wet_mm.device,
        dtype=output_dtype,
    )
    total = wet_mm.numel()
    _pack_wet_remaining_kernel[(triton.cdiv(total, 256),)](
        wet_mm,
        range_mm,
        output,
        fine_depth=fine_depth,
        fine_height=fine_height,
        width=width,
        phases=phases,
        upscale_factor=r,
        normalisation_mm=float(normalisation_mm),
        total=total,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


@triton.jit
def _packed_encoder_input_kernel(
    ct_ptr,
    fluence_ptr,
    conditioning_ptr,
    ray_index_ptr,
    output_ptr,
    fine_depth: tl.constexpr,
    fine_height: tl.constexpr,
    width: tl.constexpr,
    phases: tl.constexpr,
    upscale_factor: tl.constexpr,
    output_channels: tl.constexpr,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < total
    fine_voxels = fine_depth * fine_height * width
    batch_index = offsets // fine_voxels
    local = offsets - batch_index * fine_voxels
    width_index = local % width
    quotient = local // width
    height_index = quotient % fine_height
    depth_index = quotient // fine_height
    coarse_depth = depth_index // upscale_factor
    coarse_height = height_index // upscale_factor
    phase = (
        (depth_index % upscale_factor) * upscale_factor
        + height_index % upscale_factor
    )
    output_depth = fine_depth // upscale_factor
    output_height = fine_height // upscale_factor
    packed_volume = output_depth * phases * output_height * width
    packed_offset = (
        ((coarse_depth * phases + phase) * output_height + coarse_height) * width
        + width_index
    )
    ray_index = tl.load(
        ray_index_ptr + batch_index, mask=valid, other=0
    )
    ct = tl.load(
        ct_ptr + ray_index * packed_volume + packed_offset,
        mask=valid,
        other=0.0,
    )
    fluence = tl.load(fluence_ptr + offsets, mask=valid, other=0.0)
    condition_offset = (
        (
            (coarse_depth * (2 * phases) + phase) * output_height
            + coarse_height
        ) * width
        + width_index
    )
    condition_base = batch_index * (2 * packed_volume) + condition_offset
    wet = tl.load(conditioning_ptr + condition_base, mask=valid, other=0.0)
    remaining = tl.load(
        conditioning_ptr + condition_base + phases * output_height * width,
        mask=valid,
        other=0.0,
    )
    output_n = batch_index * output_depth + coarse_depth
    output_base = (
        output_n * out_stride_n
        + coarse_height * out_stride_h
        + width_index * out_stride_w
    )
    tl.store(output_ptr + output_base + phase * out_stride_c, ct, mask=valid)
    tl.store(
        output_ptr + output_base + (phases + phase) * out_stride_c,
        fluence,
        mask=valid,
    )
    tl.store(
        output_ptr + output_base + (2 * phases + phase) * out_stride_c,
        wet,
        mask=valid,
    )
    tl.store(
        output_ptr + output_base + (3 * phases + phase) * out_stride_c,
        remaining,
        mask=valid,
    )


def build_packed_encoder_input(
    ct_by_ray: torch.Tensor,
    fluence_fine: torch.Tensor,
    conditioning: torch.Tensor,
    ray_indices: torch.Tensor,
    *,
    upscale_factor: int,
) -> torch.Tensor:
    """Write all four packed proton modalities into one channels-last buffer."""
    _require_cuda_contiguous("ct_by_ray", ct_by_ray)
    _require_cuda_contiguous("fluence_fine", fluence_fine)
    _require_cuda_contiguous("conditioning", conditioning)
    _require_cuda_contiguous("ray_indices", ray_indices)
    if ct_by_ray.ndim != 5 or fluence_fine.ndim != 4 or conditioning.ndim != 5:
        raise ValueError("unexpected CT, fluence, or conditioning rank")
    batch, fine_depth, fine_height, width = fluence_fine.shape
    r = int(upscale_factor)
    phases = r * r
    expected_ct = (
        ct_by_ray.shape[0], fine_depth // r, phases, fine_height // r, width
    )
    if ct_by_ray.shape != expected_ct:
        raise ValueError(f"expected packed CT {expected_ct}, got {tuple(ct_by_ray.shape)}")
    expected_conditioning = (
        batch, fine_depth // r, 2 * phases, fine_height // r, width
    )
    if conditioning.shape != expected_conditioning:
        raise ValueError(
            f"expected conditioning {expected_conditioning}, got "
            f"{tuple(conditioning.shape)}"
        )
    if ray_indices.shape != (batch,):
        raise ValueError(f"ray_indices must have shape ({batch},)")
    output = torch.empty(
        (
            batch * (fine_depth // r),
            4 * phases,
            fine_height // r,
            width,
        ),
        device=fluence_fine.device,
        dtype=fluence_fine.dtype,
        memory_format=torch.channels_last,
    )
    total = fluence_fine.numel()
    _packed_encoder_input_kernel[(triton.cdiv(total, 256),)](
        ct_by_ray,
        fluence_fine,
        conditioning,
        ray_indices,
        output,
        fine_depth=fine_depth,
        fine_height=fine_height,
        width=width,
        phases=phases,
        upscale_factor=r,
        output_channels=4 * phases,
        out_stride_n=output.stride(0),
        out_stride_c=output.stride(1),
        out_stride_h=output.stride(2),
        out_stride_w=output.stride(3),
        total=total,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


@triton.jit
def _fused_rsp_wet_encoder_kernel(
    ct_ptr,
    fluence_ptr,
    density_ptr,
    class_ptr,
    ray_index_ptr,
    factor_ptr,
    range_ptr,
    output_ptr,
    fine_depth: tl.constexpr,
    fine_height: tl.constexpr,
    width: tl.constexpr,
    phases: tl.constexpr,
    upscale_factor: tl.constexpr,
    dz_mm: tl.constexpr,
    normalisation_mm: tl.constexpr,
    output_is_bfloat16: tl.constexpr,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    SPATIAL_BLOCK: tl.constexpr,
):
    batch_index = tl.program_id(0)
    spatial_offsets = (
        tl.program_id(1) * SPATIAL_BLOCK + tl.arange(0, SPATIAL_BLOCK)
    )
    spatial_size = fine_height * width
    valid = spatial_offsets < spatial_size
    height_index = spatial_offsets // width
    width_index = spatial_offsets - height_index * width
    ray_index = tl.load(ray_index_ptr + batch_index)
    range_mm = tl.load(range_ptr + batch_index).to(tl.float32)
    fine_voxels = fine_depth * spatial_size
    output_depth = fine_depth // upscale_factor
    output_height = fine_height // upscale_factor
    packed_volume = output_depth * phases * output_height * width
    wet_mm = tl.zeros((SPATIAL_BLOCK,), dtype=tl.float32)

    for depth_index in tl.range(0, fine_depth):
        source_offset = (
            ray_index * fine_voxels
            + depth_index * spatial_size
            + spatial_offsets
        )
        density = tl.load(
            density_ptr + source_offset, mask=valid, other=0.0
        ).to(tl.float32)
        class_id = tl.load(
            class_ptr + source_offset, mask=valid, other=0
        ).to(tl.int32)
        factor = tl.load(
            factor_ptr + batch_index * 7 + class_id,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        wet_mm += density * factor * dz_mm

        coarse_depth = depth_index // upscale_factor
        coarse_height = height_index // upscale_factor
        phase = (
            (depth_index % upscale_factor) * upscale_factor
            + height_index % upscale_factor
        )
        packed_offset = (
            ((coarse_depth * phases + phase) * output_height + coarse_height)
            * width
            + width_index
        )
        ct = tl.load(
            ct_ptr + ray_index * packed_volume + packed_offset,
            mask=valid,
            other=0.0,
        )
        fluence = tl.load(
            fluence_ptr
            + batch_index * fine_voxels
            + depth_index * spatial_size
            + spatial_offsets,
            mask=valid,
            other=0.0,
        )
        wet = tl.minimum(tl.maximum(wet_mm / normalisation_mm, 0.0), 2.0)
        remaining = tl.minimum(
            tl.maximum((range_mm - wet_mm) / normalisation_mm, -2.0), 1.0
        )
        output_n = batch_index * output_depth + coarse_depth
        output_base = (
            output_n * out_stride_n
            + coarse_height * out_stride_h
            + width_index * out_stride_w
        )
        tl.store(
            output_ptr + output_base + phase * out_stride_c,
            ct,
            mask=valid,
        )
        tl.store(
            output_ptr + output_base + (phases + phase) * out_stride_c,
            fluence,
            mask=valid,
        )
        tl.store(
            output_ptr + output_base + (2 * phases + phase) * out_stride_c,
            wet,
            mask=valid,
        )
        tl.store(
            output_ptr + output_base + (3 * phases + phase) * out_stride_c,
            remaining,
            mask=valid,
        )
def build_packed_encoder_input_fused_scan(
    ct_by_ray: torch.Tensor,
    fluence_fine: torch.Tensor,
    density_by_ray: torch.Tensor,
    class_id_by_ray: torch.Tensor,
    ray_indices: torch.Tensor,
    class_factors: torch.Tensor,
    range_mm: torch.Tensor,
    *,
    upscale_factor: int,
    dz_mm: float,
    normalisation_mm: float = 300.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fuse RSP expansion, depth accumulation, packing and encoder layout."""
    for name, tensor in (
        ("ct_by_ray", ct_by_ray),
        ("fluence_fine", fluence_fine),
        ("density_by_ray", density_by_ray),
        ("class_id_by_ray", class_id_by_ray),
        ("ray_indices", ray_indices),
        ("class_factors", class_factors),
        ("range_mm", range_mm),
    ):
        _require_cuda_contiguous(name, tensor)
    if density_by_ray.shape != class_id_by_ray.shape or density_by_ray.ndim != 4:
        raise ValueError("density/class state must have matching (U,D,H,W) shapes")
    if class_id_by_ray.dtype != torch.uint8:
        raise ValueError("class_id_by_ray must use uint8 material classes")
    batch, fine_depth, fine_height, width = fluence_fine.shape
    unique_rays = density_by_ray.shape[0]
    if density_by_ray.shape[1:] != fluence_fine.shape[1:]:
        raise ValueError("material and fluence fine-grid shapes do not match")
    if ray_indices.shape != (batch,) or ray_indices.dtype != torch.long:
        raise ValueError(f"ray_indices must be int64 with shape ({batch},)")
    if class_factors.shape != (batch, _MATERIAL_CLASS_COUNT):
        raise ValueError(f"class_factors must have shape ({batch},7)")
    if range_mm.shape != (batch,):
        raise ValueError(f"range_mm must have shape ({batch},)")
    if unique_rays < 1:
        raise ValueError("at least one unique ray is required")
    r = int(upscale_factor)
    if r < 1 or fine_depth % r or fine_height % r:
        raise ValueError("fine dimensions must be divisible by upscale_factor")
    phases = r * r
    expected_ct = (
        unique_rays,
        fine_depth // r,
        phases,
        fine_height // r,
        width,
    )
    if ct_by_ray.shape != expected_ct:
        raise ValueError(
            f"expected packed CT {expected_ct}, got {tuple(ct_by_ray.shape)}"
        )
    if dz_mm <= 0.0 or normalisation_mm <= 0.0:
        raise ValueError("dz_mm and normalisation_mm must be positive")
    output_modalities = 4
    output = torch.empty(
        (
            batch * (fine_depth // r),
            output_modalities * phases,
            fine_height // r,
            width,
        ),
        device=fluence_fine.device,
        dtype=fluence_fine.dtype,
        memory_format=torch.channels_last,
    )
    spatial_block = 128
    _fused_rsp_wet_encoder_kernel[
        (batch, triton.cdiv(fine_height * width, spatial_block))
    ](
        ct_by_ray,
        fluence_fine,
        density_by_ray,
        class_id_by_ray,
        ray_indices,
        class_factors,
        range_mm,
        output,
        fine_depth=fine_depth,
        fine_height=fine_height,
        width=width,
        phases=phases,
        upscale_factor=r,
        dz_mm=float(dz_mm),
        normalisation_mm=float(normalisation_mm),
        output_is_bfloat16=(fluence_fine.dtype == torch.bfloat16),
        out_stride_n=output.stride(0),
        out_stride_c=output.stride(1),
        out_stride_h=output.stride(2),
        out_stride_w=output.stride(3),
        SPATIAL_BLOCK=spatial_block,
        num_warps=4,
    )
    return output


@torch.inference_mode()
def warmup_combined_proton_preprocess(
    *,
    output_shape: tuple[int, int, int],
    upscale_factor: int,
    dz_mm: float,
    device: torch.device,
    output_dtype: torch.dtype,
    batch_size: int = 1,
    unique_rays: int | None = None,
    return_encoder_input: bool = False,
) -> torch.Tensor | None:
    """Compile the production-shape combined kernel before server readiness."""
    fine_depth, fine_height, width = (int(value) for value in output_shape)
    r = int(upscale_factor)
    phases = r * r
    batch = int(batch_size)
    unique = batch if unique_rays is None else int(unique_rays)
    if not 1 <= unique <= batch:
        raise ValueError("unique_rays must be between one and batch_size")
    ct = torch.zeros(
        (unique, fine_depth // r, phases, fine_height // r, width),
        device=device,
        dtype=output_dtype,
    )
    fluence = torch.zeros(
        (batch, fine_depth, fine_height, width),
        device=device,
        dtype=output_dtype,
    )
    density = torch.ones(
        (unique, fine_depth, fine_height, width),
        device=device,
        dtype=torch.float32,
    )
    class_id = torch.zeros_like(density, dtype=torch.uint8)
    ray_indices = torch.arange(
        batch, device=device, dtype=torch.long
    ).remainder_(unique)
    class_factors = torch.ones(
        (batch, _MATERIAL_CLASS_COUNT), device=device, dtype=torch.float32
    )
    range_mm = torch.ones(batch, device=device, dtype=torch.float32)
    encoder_input = build_packed_encoder_input_fused_scan(
        ct,
        fluence,
        density,
        class_id,
        ray_indices,
        class_factors,
        range_mm,
        upscale_factor=r,
        dz_mm=float(dz_mm),
    )
    torch.cuda.synchronize(device)
    del range_mm, class_factors, ray_indices
    del class_id, density, fluence, ct
    if return_encoder_input:
        return encoder_input
    del encoder_input
    return None
