"""Triton implementation of z-aligned bicubic B-spline sampling.

The differentiable path applies an exact cubic IIR prefilter in-plane and then
selects integer z planes before evaluating a 16-tap 2D spline. The older
three-plane collapse remains available for prefiltered volumes and CuPy
compatibility.

Triton is the default z-aligned backend. CuPy remains available as a fallback and
for the general, non-aligned tricubic path. Query coordinates are fixed geometry;
gradients are computed only for the spline coefficients.
"""

from __future__ import annotations

import ast
import math

import torch
import triton
import triton.language as tl

from ._spline_torch import collapse_integer_z

if not hasattr(ast, "Num"):
    ast.Num = ast.Constant  # type: ignore[attr-defined,misc]

FWD_CONFIGS = [
    triton.Config({"BLOCK_SIZE": block}, num_warps=warps)
    for block in (128, 256, 512, 1024)
    for warps in (4, 8)
]
BWD_BLOCK_SIZE = 64
BWD_NUM_WARPS = 2

IIR_CONFIGS = [
    triton.Config({"BLOCK_LINES": 64}, num_warps=2),
    triton.Config({"BLOCK_LINES": 128}, num_warps=4),
    triton.Config({"BLOCK_LINES": 256}, num_warps=8),
]


def _gpu_class(device: torch.device) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


@triton.autotune(
    configs=IIR_CONFIGS,
    key=["GPU_CLASS", "N_LINES", "N_SAMPLES"],
)
@triton.jit
def _cubic_iir_prefilter_kernel(
    x_ptr,
    out_ptr,
    N_LINES,
    GPU_CLASS: tl.constexpr,
    N_SAMPLES: tl.constexpr,
    BLOCK_LINES: tl.constexpr,
):
    """Exact cubic B-spline coefficient filter for sample-major lines."""
    lines = tl.program_id(0) * BLOCK_LINES + tl.arange(0, BLOCK_LINES)
    mask = lines < N_LINES
    z = -0.2679491924311227
    gain = 6.0

    z_n_1 = z
    for _ in tl.range(1, N_SAMPLES - 1):
        z_n_1 *= z

    acc = gain * tl.load(x_ptr + lines, mask=mask, other=0.0)
    acc += z_n_1 * gain * tl.load(
        x_ptr + (N_SAMPLES - 1) * N_LINES + lines, mask=mask, other=0.0
    )
    z_i = z
    for i in tl.range(1, N_SAMPLES - 1):
        left = tl.load(x_ptr + i * N_LINES + lines, mask=mask, other=0.0)
        right = tl.load(
            x_ptr + (N_SAMPLES - 1 - i) * N_LINES + lines,
            mask=mask,
            other=0.0,
        )
        acc += z_i * gain * (left + z_n_1 * right)
        z_i *= z
    prev = acc / (1.0 - z_n_1 * z_n_1)
    tl.store(out_ptr + lines, prev, mask=mask)

    for i in tl.range(1, N_SAMPLES):
        value = gain * tl.load(
            x_ptr + i * N_LINES + lines, mask=mask, other=0.0
        ) + z * prev
        tl.store(out_ptr + i * N_LINES + lines, value, mask=mask)
        prev = value

    before_last = tl.load(
        out_ptr + (N_SAMPLES - 2) * N_LINES + lines, mask=mask, other=0.0
    )
    last = (z * before_last + prev) * z / (z * z - 1.0)
    tl.store(
        out_ptr + (N_SAMPLES - 1) * N_LINES + lines, last, mask=mask
    )
    next_value = last
    for reverse_i in tl.range(0, N_SAMPLES - 1):
        i = N_SAMPLES - 2 - reverse_i
        current = tl.load(out_ptr + i * N_LINES + lines, mask=mask, other=0.0)
        value = z * (next_value - current)
        tl.store(out_ptr + i * N_LINES + lines, value, mask=mask)
        next_value = value


@triton.autotune(
    configs=IIR_CONFIGS,
    key=["GPU_CLASS", "N_LINES", "N_SAMPLES"],
)
@triton.jit
def _cubic_iir_prefilter_bwd_kernel(
    grad_out_ptr,
    work_ptr,
    grad_x_ptr,
    N_LINES,
    GPU_CLASS: tl.constexpr,
    N_SAMPLES: tl.constexpr,
    BLOCK_LINES: tl.constexpr,
):
    """Adjoint of the mirror-boundary causal/anti-causal recurrence."""
    lines = tl.program_id(0) * BLOCK_LINES + tl.arange(0, BLOCK_LINES)
    mask = lines < N_LINES
    z = -0.2679491924311227
    gain = 6.0
    alpha = z / (z * z - 1.0)

    # Reverse the anti-causal pass, accumulating gradients for its next value.
    grad_d = tl.load(grad_out_ptr + lines, mask=mask, other=0.0)
    for i in tl.range(0, N_SAMPLES - 1):
        tl.store(work_ptr + i * N_LINES + lines, -z * grad_d, mask=mask)
        grad_d = tl.load(
            grad_out_ptr + (i + 1) * N_LINES + lines, mask=mask, other=0.0
        ) + z * grad_d

    before_last = tl.load(
        work_ptr + (N_SAMPLES - 2) * N_LINES + lines, mask=mask, other=0.0
    )
    tl.store(
        work_ptr + (N_SAMPLES - 2) * N_LINES + lines,
        before_last + alpha * z * grad_d,
        mask=mask,
    )
    tl.store(
        work_ptr + (N_SAMPLES - 1) * N_LINES + lines,
        alpha * grad_d,
        mask=mask,
    )

    # Reverse the causal recurrence. work holds gradients for causal values.
    for reverse_i in tl.range(0, N_SAMPLES - 1):
        i = N_SAMPLES - 1 - reverse_i
        grad_c = tl.load(work_ptr + i * N_LINES + lines, mask=mask, other=0.0)
        tl.store(grad_x_ptr + i * N_LINES + lines, gain * grad_c, mask=mask)
        grad_prev = tl.load(
            work_ptr + (i - 1) * N_LINES + lines, mask=mask, other=0.0
        ) + z * grad_c
        tl.store(work_ptr + (i - 1) * N_LINES + lines, grad_prev, mask=mask)

    # Distribute the causal initializer gradient to its mirrored input taps.
    grad_c0 = tl.load(work_ptr + lines, mask=mask, other=0.0)
    z_n_1 = z
    for _ in tl.range(1, N_SAMPLES - 1):
        z_n_1 *= z
    factor = gain * grad_c0 / (1.0 - z_n_1 * z_n_1)
    tl.store(grad_x_ptr + lines, factor, mask=mask)
    last_offset = (N_SAMPLES - 1) * N_LINES + lines
    last = tl.load(grad_x_ptr + last_offset, mask=mask, other=0.0)
    tl.store(grad_x_ptr + last_offset, last + z_n_1 * factor, mask=mask)

    z_i = z
    for i in tl.range(1, N_SAMPLES - 1):
        offset = i * N_LINES + lines
        value = tl.load(grad_x_ptr + offset, mask=mask, other=0.0)
        tl.store(grad_x_ptr + offset, value + z_i * factor, mask=mask)

        mirror_offset = (N_SAMPLES - 1 - i) * N_LINES + lines
        mirror = tl.load(grad_x_ptr + mirror_offset, mask=mask, other=0.0)
        tl.store(
            grad_x_ptr + mirror_offset,
            mirror + z_i * z_n_1 * factor,
            mask=mask,
        )
        z_i *= z


def _cubic_iir_prefilter_axis_forward(x: torch.Tensor, axis: int) -> torch.Tensor:
    n_samples = x.shape[axis]
    if n_samples == 1:
        return x.clone()
    sample_major = x.movedim(axis, 0).contiguous()
    lines = sample_major.reshape(n_samples, -1)
    out = torch.empty_like(lines)
    grid = lambda meta: (triton.cdiv(lines.shape[1], meta["BLOCK_LINES"]),)
    _cubic_iir_prefilter_kernel[grid](
        lines,
        out,
        lines.shape[1],
        _gpu_class(x.device),
        N_SAMPLES=n_samples,
    )
    return out.reshape(sample_major.shape).movedim(0, axis)


def _cubic_iir_prefilter_axis_backward(
    grad_output: torch.Tensor, axis: int
) -> torch.Tensor:
    n_samples = grad_output.shape[axis]
    if n_samples == 1:
        return grad_output.clone()
    sample_major = grad_output.movedim(axis, 0).contiguous()
    lines = sample_major.reshape(n_samples, -1)
    work = torch.empty_like(lines)
    grad_x = torch.empty_like(lines)
    grid = lambda meta: (triton.cdiv(lines.shape[1], meta["BLOCK_LINES"]),)
    _cubic_iir_prefilter_bwd_kernel[grid](
        lines,
        work,
        grad_x,
        lines.shape[1],
        _gpu_class(grad_output.device),
        N_SAMPLES=n_samples,
    )
    return grad_x.reshape(sample_major.shape).movedim(0, axis)


class _CubicIIRPrefilterAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, axis: int) -> torch.Tensor:
        ctx.axis = axis
        return _cubic_iir_prefilter_axis_forward(x, axis)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _cubic_iir_prefilter_axis_backward(grad_output, ctx.axis), None


def cubic_iir_prefilter_2d_triton(x: torch.Tensor) -> torch.Tensor:
    """Exact mirror-boundary cubic B-spline prefilter on the T/H axes."""
    if x.ndim != 4:
        raise ValueError(f"expected (B,T,H,W), got shape {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("cubic_iir_prefilter_2d_triton requires a CUDA tensor")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("IIR prefilter input must be float32 or float64")
    out = _CubicIIRPrefilterAxis.apply(x, 1)
    return _CubicIIRPrefilterAxis.apply(out, 2)


@triton.jit
def _mirror(idx, period, dim_minus_1):
    """Whole-sample mirror reflection of an out-of-range tap index into
    ``[0, dim_minus_1]``, matching ``patient_space_resample._mirror_reflect_index``."""
    m = idx % period
    m = tl.where(m < 0, m + period, m)
    return tl.where(m > dim_minus_1, period - m, m)


@triton.jit
def _collapse_integer_z_kernel(
    coeff_ptr, target_w_ptr, out_ptr,
    nT, nH, nW, nZ, N,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    h = offs % nH
    q = offs // nH
    t = q % nT
    q = q // nT
    z = q % nZ
    b = q // nZ

    w0 = tl.load(target_w_ptr + z, mask=mask, other=0).to(tl.int32)
    w0 = tl.maximum(0, tl.minimum(w0, nW - 1))
    wm1 = tl.where(w0 > 0, w0 - 1, tl.minimum(1, nW - 1))
    wp1 = tl.where(w0 < nW - 1, w0 + 1, tl.maximum(nW - 2, 0))

    base = ((b * nT + t) * nH + h) * nW
    value = (
        tl.load(coeff_ptr + base + wm1, mask=mask)
        + 4.0 * tl.load(coeff_ptr + base + w0, mask=mask)
        + tl.load(coeff_ptr + base + wp1, mask=mask)
    ) * (1.0 / 6.0)
    tl.store(out_ptr + offs, value, mask=mask)


def _collapse_integer_z_no_grad(
    coeff: torch.Tensor, target_w_index: torch.Tensor
) -> torch.Tensor:
    """Collapse three integer-z taps without building a dense blend matrix."""
    b, nT, nH, nW = coeff.shape
    target = target_w_index.to(device=coeff.device, dtype=torch.int32).contiguous()
    nZ = target.numel()
    out = torch.empty((b, nZ, nT, nH), device=coeff.device, dtype=coeff.dtype)
    total = out.numel()
    _collapse_integer_z_kernel[(triton.cdiv(total, 256),)](
        coeff, target, out, nT, nH, nW, nZ, total,
        BLOCK_SIZE=256, num_warps=4,
    )
    return out


@triton.autotune(
    configs=FWD_CONFIGS,
    key=[
        "GPU_CLASS", "nT", "nH", "nW", "nZ", "N",
        "SINGLE_BATCH", "ORDERED_Z", "CHECK_BOUNDS",
    ],
)
@triton.jit
def _bicubic2d_fwd_zmajor_kernel(
    collapsed_ptr, tq_ptr, hq_ptr, bq_ptr, zq_ptr, out_ptr,
    target_w_ptr, nT, nH, nW, nZ, cval,
    N,
    GPU_CLASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SINGLE_BATCH: tl.constexpr,
    ORDERED_Z: tl.constexpr,
    CHECK_BOUNDS: tl.constexpr,
):
    """Evaluate the z-collapsed spline in float32 or float64."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    tq = tl.load(tq_ptr + offs, mask=mask, other=0.0)
    hq = tl.load(hq_ptr + offs, mask=mask, other=0.0)
    if SINGLE_BATCH:
        bq = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        bq = tl.load(bq_ptr + offs, mask=mask, other=0).to(tl.int32)
    if ORDERED_Z:
        zq = offs % nZ
    else:
        zq = tl.load(zq_ptr + offs, mask=mask, other=0).to(tl.int32)

    t0 = tl.floor(tq).to(tl.int32)
    h0 = tl.floor(hq).to(tl.int32)
    ft = tq - t0.to(tq.dtype)
    fh = hq - h0.to(hq.dtype)

    ft2 = ft * ft; ft3 = ft2 * ft
    wt0 = (1.0 - ft); wt0 = wt0 * wt0 * wt0 / 6.0
    wt1 = (4.0 - 6.0 * ft2 + 3.0 * ft3) / 6.0
    wt2 = (1.0 + 3.0 * ft + 3.0 * ft2 - 3.0 * ft3) / 6.0
    wt3 = ft3 / 6.0

    fh2 = fh * fh; fh3 = fh2 * fh
    wh0 = (1.0 - fh); wh0 = wh0 * wh0 * wh0 / 6.0
    wh1 = (4.0 - 6.0 * fh2 + 3.0 * fh3) / 6.0
    wh2 = (1.0 + 3.0 * fh + 3.0 * fh2 - 3.0 * fh3) / 6.0
    wh3 = fh3 / 6.0

    # tl.where evaluates both branches, so the unused mirror period must be valid.
    period_t = 2 * (nT - 1) + 2 * (nT == 1)
    period_h = 2 * (nH - 1) + 2 * (nH == 1)

    ti0 = tl.where(nT > 1, _mirror(t0 - 1, period_t, nT - 1), 0)
    ti1 = tl.where(nT > 1, _mirror(t0 + 0, period_t, nT - 1), 0)
    ti2 = tl.where(nT > 1, _mirror(t0 + 1, period_t, nT - 1), 0)
    ti3 = tl.where(nT > 1, _mirror(t0 + 2, period_t, nT - 1), 0)

    hi0 = tl.where(nH > 1, _mirror(h0 - 1, period_h, nH - 1), 0)
    hi1 = tl.where(nH > 1, _mirror(h0 + 0, period_h, nH - 1), 0)
    hi2 = tl.where(nH > 1, _mirror(h0 + 1, period_h, nH - 1), 0)
    hi3 = tl.where(nH > 1, _mirror(h0 + 2, period_h, nH - 1), 0)

    # (B, Nz, T, H) layout: flat = ((b*nZ + z)*nT + t)*nH + h
    z_valid = (zq >= 0) & (zq < nZ)
    zq_safe = tl.where(z_valid, zq, 0)
    base_bz = (bq * nZ + zq_safe) * nT
    stride_t = nH

    acc = tl.zeros((BLOCK_SIZE,), dtype=tq.dtype)
    acc += wt0 * wh0 * tl.load(collapsed_ptr + (base_bz + ti0) * stride_t + hi0, mask=mask & z_valid, other=0.0)
    acc += wt0 * wh1 * tl.load(collapsed_ptr + (base_bz + ti0) * stride_t + hi1, mask=mask & z_valid, other=0.0)
    acc += wt0 * wh2 * tl.load(collapsed_ptr + (base_bz + ti0) * stride_t + hi2, mask=mask & z_valid, other=0.0)
    acc += wt0 * wh3 * tl.load(collapsed_ptr + (base_bz + ti0) * stride_t + hi3, mask=mask & z_valid, other=0.0)
    acc += wt1 * wh0 * tl.load(collapsed_ptr + (base_bz + ti1) * stride_t + hi0, mask=mask & z_valid, other=0.0)
    acc += wt1 * wh1 * tl.load(collapsed_ptr + (base_bz + ti1) * stride_t + hi1, mask=mask & z_valid, other=0.0)
    acc += wt1 * wh2 * tl.load(collapsed_ptr + (base_bz + ti1) * stride_t + hi2, mask=mask & z_valid, other=0.0)
    acc += wt1 * wh3 * tl.load(collapsed_ptr + (base_bz + ti1) * stride_t + hi3, mask=mask & z_valid, other=0.0)
    acc += wt2 * wh0 * tl.load(collapsed_ptr + (base_bz + ti2) * stride_t + hi0, mask=mask & z_valid, other=0.0)
    acc += wt2 * wh1 * tl.load(collapsed_ptr + (base_bz + ti2) * stride_t + hi1, mask=mask & z_valid, other=0.0)
    acc += wt2 * wh2 * tl.load(collapsed_ptr + (base_bz + ti2) * stride_t + hi2, mask=mask & z_valid, other=0.0)
    acc += wt2 * wh3 * tl.load(collapsed_ptr + (base_bz + ti2) * stride_t + hi3, mask=mask & z_valid, other=0.0)
    acc += wt3 * wh0 * tl.load(collapsed_ptr + (base_bz + ti3) * stride_t + hi0, mask=mask & z_valid, other=0.0)
    acc += wt3 * wh1 * tl.load(collapsed_ptr + (base_bz + ti3) * stride_t + hi1, mask=mask & z_valid, other=0.0)
    acc += wt3 * wh2 * tl.load(collapsed_ptr + (base_bz + ti3) * stride_t + hi2, mask=mask & z_valid, other=0.0)
    acc += wt3 * wh3 * tl.load(collapsed_ptr + (base_bz + ti3) * stride_t + hi3, mask=mask & z_valid, other=0.0)

    if CHECK_BOUNDS:
        wq = tl.load(target_w_ptr + zq_safe, mask=mask & z_valid, other=-1)
        query_valid = (
            z_valid
            & (tq >= 0.0) & (tq <= nT - 1)
            & (hq >= 0.0) & (hq <= nH - 1)
            & (wq >= 0) & (wq <= nW - 1)
        )
        acc = tl.where(query_valid, acc, cval)
    tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def _bicubic2d_bwd_zmajor_kernel(
    tq_ptr, hq_ptr, bq_ptr, zq_ptr, gout_ptr, grad_collapsed_ptr,
    nT, nH, nZ,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter coefficient gradients, recomputing weights to reduce register use."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    tq = tl.load(tq_ptr + offs, mask=mask, other=0.0)
    hq = tl.load(hq_ptr + offs, mask=mask, other=0.0)
    bq = tl.load(bq_ptr + offs, mask=mask, other=0).to(tl.int32)
    zq = tl.load(zq_ptr + offs, mask=mask, other=0).to(tl.int32)
    gout = tl.load(gout_ptr + offs, mask=mask, other=0.0)

    t0 = tl.floor(tq).to(tl.int32)
    h0 = tl.floor(hq).to(tl.int32)
    ft = tq - t0.to(tq.dtype)
    fh = hq - h0.to(hq.dtype)

    # tl.where evaluates both branches, so the unused mirror period must be valid.
    period_t = 2 * (nT - 1) + 2 * (nT == 1)
    period_h = 2 * (nH - 1) + 2 * (nH == 1)
    z_valid = (zq >= 0) & (zq < nZ)
    zq_safe = tl.where(z_valid, zq, 0)
    base_bz = (bq * nZ + zq_safe) * nT
    stride_t = nH

    for a in range(4):
        t = ft
        if a == 0:
            w_t = (1.0 - t); w_t = w_t * w_t * w_t / 6.0
        elif a == 1:
            t2 = t * t; t3 = t2 * t
            w_t = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0
        elif a == 2:
            t2 = t * t; t3 = t2 * t
            w_t = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0
        else:
            w_t = (t * t * t) / 6.0
        ti = tl.where(nT > 1, _mirror(t0 - 1 + a, period_t, nT - 1), 0)

        for b in range(4):
            t = fh
            if b == 0:
                w_h = (1.0 - t); w_h = w_h * w_h * w_h / 6.0
            elif b == 1:
                t2 = t * t; t3 = t2 * t
                w_h = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0
            elif b == 2:
                t2 = t * t; t3 = t2 * t
                w_h = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0
            else:
                w_h = (t * t * t) / 6.0
            hi = tl.where(nH > 1, _mirror(h0 - 1 + b, period_h, nH - 1), 0)

            flat = (base_bz + ti) * stride_t + hi
            contrib = w_t * w_h * gout
            tl.atomic_add(grad_collapsed_ptr + flat, contrib, mask=mask & z_valid)


def _flatten_points(points_th: torch.Tensor, dtype: torch.dtype):
    b, n, _ = points_th.shape
    tq = points_th[..., 0].reshape(-1).to(dtype).contiguous()
    hq = points_th[..., 1].reshape(-1).to(dtype).contiguous()
    bq = torch.arange(b, device=points_th.device, dtype=torch.int32).repeat_interleave(n)
    return tq, hq, bq, b, n


def _compute_dtype(storage_dtype: torch.dtype) -> torch.dtype:
    """Promote low-precision storage while preserving float32 and float64."""
    return torch.float32 if storage_dtype in (torch.float16, torch.bfloat16) else storage_dtype


@triton.jit
def _select_planes_fwd_kernel(
    coeff_ptr, target_ptr, out_ptr,
    nT, nH, nW, nZ, N,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    h = offs % nH
    q = offs // nH
    t = q % nT
    q = q // nT
    z = q % nZ
    b = q // nZ
    w = tl.load(target_ptr + z, mask=mask, other=0).to(tl.int32)
    src = ((b * nT + t) * nH + h) * nW + w
    tl.store(out_ptr + offs, tl.load(coeff_ptr + src, mask=mask), mask=mask)


@triton.jit
def _select_planes_bwd_kernel(
    grad_out_ptr, target_ptr, grad_coeff_ptr,
    nT, nH, nW, nZ, N,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    h = offs % nH
    q = offs // nH
    t = q % nT
    q = q // nT
    z = q % nZ
    b = q // nZ
    w = tl.load(target_ptr + z, mask=mask, other=0).to(tl.int32)
    dst = ((b * nT + t) * nH + h) * nW + w
    value = tl.load(grad_out_ptr + offs, mask=mask, other=0.0)
    tl.atomic_add(grad_coeff_ptr + dst, value, mask=mask)


class _SelectIntegerPlanesTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff: torch.Tensor, target_w_index: torch.Tensor) -> torch.Tensor:
        coeff = coeff.contiguous()
        b, nT, nH, nW = coeff.shape
        target = target_w_index.to(
            device=coeff.device, dtype=torch.int32
        ).clamp(0, nW - 1).contiguous()
        nZ = target.numel()
        out = torch.empty((b, nZ, nT, nH), device=coeff.device, dtype=coeff.dtype)
        total = out.numel()
        _select_planes_fwd_kernel[(triton.cdiv(total, 256),)](
            coeff, target, out, nT, nH, nW, nZ, total,
            BLOCK_SIZE=256, num_warps=4,
        )
        ctx.save_for_backward(target)
        ctx.coeff_shape = coeff.shape
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (target,) = ctx.saved_tensors
        b, nT, nH, nW = ctx.coeff_shape
        nZ = target.numel()
        grad_output = grad_output.contiguous()
        grad_coeff = torch.zeros(
            ctx.coeff_shape, device=grad_output.device, dtype=grad_output.dtype
        )
        total = grad_output.numel()
        _select_planes_bwd_kernel[(triton.cdiv(total, 256),)](
            grad_output, target, grad_coeff, nT, nH, nW, nZ, total,
            BLOCK_SIZE=256, num_warps=4,
        )
        return grad_coeff, None


def bicubic_iir_sample_from_coeff_triton(
    filtered: torch.Tensor,
    points_th: torch.Tensor,
    target_w_index: torch.Tensor,
    zq: torch.Tensor,
) -> torch.Tensor:
    """Sample integer W planes from exact T/H spline coefficients."""
    planes = _SelectIntegerPlanesTriton.apply(filtered, target_w_index)
    return _Bicubic2DZMajorTriton.apply(planes, points_th, zq)


@triton.jit
def _bicubic2d_affine_fwd_kernel(
    coeff_ptr, affine_ptr, out_ptr, valid_ptr,
    nT, nH, nW, roi_x, roi_y, roi_z, N,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample a CT ROI while generating its affine BEV coordinates in-kernel."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    n_roi = roi_x * roi_y * roi_z
    b = offs // n_roi
    local = offs - b * n_roi
    iz = local % roi_z
    q = local // roi_z
    iy = q % roi_y
    ix = q // roi_y
    abase = b * 12
    x = ix.to(tl.float32)
    y = iy.to(tl.float32)
    z = iz.to(tl.float32)
    tq = (
        tl.load(affine_ptr + abase + 0, mask=mask) * x
        + tl.load(affine_ptr + abase + 1, mask=mask) * y
        + tl.load(affine_ptr + abase + 2, mask=mask) * z
        + tl.load(affine_ptr + abase + 3, mask=mask)
    )
    hq = (
        tl.load(affine_ptr + abase + 4, mask=mask) * x
        + tl.load(affine_ptr + abase + 5, mask=mask) * y
        + tl.load(affine_ptr + abase + 6, mask=mask) * z
        + tl.load(affine_ptr + abase + 7, mask=mask)
    )
    wq = (
        tl.load(affine_ptr + abase + 8, mask=mask) * x
        + tl.load(affine_ptr + abase + 9, mask=mask) * y
        + tl.load(affine_ptr + abase + 10, mask=mask) * z
        + tl.load(affine_ptr + abase + 11, mask=mask)
    )
    wi = tl.floor(wq + 0.5).to(tl.int32)
    valid = (
        (tq >= 0.0) & (tq <= nT - 1)
        & (hq >= 0.0) & (hq <= nH - 1)
        & (wi >= 0) & (wi < nW)
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
            t2 = t * t; wt = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
        elif a == 2:
            t2 = t * t; wt = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) / 6.0
        else:
            wt = t * t * t / 6.0
        ti = tl.where(nT > 1, _mirror(t0 - 1 + a, period_t, nT - 1), 0)
        for htap in range(4):
            t = fh
            if htap == 0:
                wh = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
            elif htap == 1:
                t2 = t * t; wh = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
            elif htap == 2:
                t2 = t * t; wh = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) / 6.0
            else:
                wh = t * t * t / 6.0
            hi = tl.where(nH > 1, _mirror(h0 - 1 + htap, period_h, nH - 1), 0)
            src = ((b * nT + ti) * nH + hi) * nW + wi_safe
            acc += wt * wh * tl.load(coeff_ptr + src, mask=mask & valid, other=0.0)
    tl.store(out_ptr + offs, tl.where(valid, acc, 0.0), mask=mask)
    tl.store(valid_ptr + offs, valid.to(tl.int8), mask=mask)


@triton.jit
def _bicubic2d_affine_bwd_kernel(
    affine_ptr, gout_ptr, grad_coeff_ptr,
    nT, nH, nW, roi_x, roi_y, roi_z, N,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    n_roi = roi_x * roi_y * roi_z
    b = offs // n_roi
    local = offs - b * n_roi
    iz = local % roi_z
    q = local // roi_z
    iy = q % roi_y
    ix = q // roi_y
    abase = b * 12
    x = ix.to(tl.float32); y = iy.to(tl.float32); z = iz.to(tl.float32)
    tq = tl.load(affine_ptr + abase + 0, mask=mask) * x + tl.load(affine_ptr + abase + 1, mask=mask) * y + tl.load(affine_ptr + abase + 2, mask=mask) * z + tl.load(affine_ptr + abase + 3, mask=mask)
    hq = tl.load(affine_ptr + abase + 4, mask=mask) * x + tl.load(affine_ptr + abase + 5, mask=mask) * y + tl.load(affine_ptr + abase + 6, mask=mask) * z + tl.load(affine_ptr + abase + 7, mask=mask)
    wq = tl.load(affine_ptr + abase + 8, mask=mask) * x + tl.load(affine_ptr + abase + 9, mask=mask) * y + tl.load(affine_ptr + abase + 10, mask=mask) * z + tl.load(affine_ptr + abase + 11, mask=mask)
    wi = tl.floor(wq + 0.5).to(tl.int32)
    valid = (tq >= 0.0) & (tq <= nT - 1) & (hq >= 0.0) & (hq <= nH - 1) & (wi >= 0) & (wi < nW) & (tl.abs(wq - wi.to(tl.float32)) <= 1.0e-3)
    t0 = tl.floor(tq).to(tl.int32); h0 = tl.floor(hq).to(tl.int32)
    ft = tq - t0.to(tq.dtype); fh = hq - h0.to(hq.dtype)
    period_t = 2 * (nT - 1) + 2 * (nT == 1)
    period_h = 2 * (nH - 1) + 2 * (nH == 1)
    gout = tl.load(gout_ptr + offs, mask=mask, other=0.0)
    wi_safe = tl.maximum(0, tl.minimum(wi, nW - 1))
    for a in range(4):
        t = ft
        if a == 0: wt = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
        elif a == 1:
            t2 = t * t; wt = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
        elif a == 2:
            t2 = t * t; wt = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) / 6.0
        else: wt = t * t * t / 6.0
        ti = tl.where(nT > 1, _mirror(t0 - 1 + a, period_t, nT - 1), 0)
        for htap in range(4):
            t = fh
            if htap == 0: wh = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
            elif htap == 1:
                t2 = t * t; wh = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
            elif htap == 2:
                t2 = t * t; wh = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) / 6.0
            else: wh = t * t * t / 6.0
            hi = tl.where(nH > 1, _mirror(h0 - 1 + htap, period_h, nH - 1), 0)
            dst = ((b * nT + ti) * nH + hi) * nW + wi_safe
            tl.atomic_add(grad_coeff_ptr + dst, wt * wh * gout, mask=mask & valid)


class _Bicubic2DAffineTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff: torch.Tensor, affine: torch.Tensor, roi_shape: tuple[int, int, int]):
        coeff_c = coeff.contiguous()
        affine_c = affine.to(device=coeff.device, dtype=torch.float32).contiguous()
        b, nT, nH, nW = coeff_c.shape
        if affine_c.shape != (b, 3, 4):
            raise ValueError(f"expected affine ({b},3,4), got {tuple(affine_c.shape)}")
        roi_x, roi_y, roi_z = (int(v) for v in roi_shape)
        total = b * roi_x * roi_y * roi_z
        out = torch.empty(total, device=coeff.device, dtype=torch.float32)
        valid = torch.empty(total, device=coeff.device, dtype=torch.bool)
        _bicubic2d_affine_fwd_kernel[(triton.cdiv(total, 256),)](
            coeff_c, affine_c, out, valid, nT, nH, nW,
            roi_x, roi_y, roi_z, total, BLOCK_SIZE=256, num_warps=4,
        )
        ctx.save_for_backward(affine_c)
        ctx.coeff_shape = coeff_c.shape
        ctx.roi_shape = (roi_x, roi_y, roi_z)
        ctx.needs_grad = coeff.requires_grad
        ctx.mark_non_differentiable(valid)
        return out.reshape(b, roi_x, roi_y, roi_z), valid.reshape(b, roi_x, roi_y, roi_z)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _grad_valid: torch.Tensor | None):
        if not ctx.needs_grad:
            return None, None, None
        (affine,) = ctx.saved_tensors
        b, nT, nH, nW = ctx.coeff_shape
        roi_x, roi_y, roi_z = ctx.roi_shape
        gout = grad_output.contiguous().float()
        grad_coeff = torch.zeros(ctx.coeff_shape, device=gout.device, dtype=torch.float32)
        total = gout.numel()
        _bicubic2d_affine_bwd_kernel[(triton.cdiv(total, BWD_BLOCK_SIZE),)](
            affine, gout, grad_coeff, nT, nH, nW,
            roi_x, roi_y, roi_z, total,
            BLOCK_SIZE=BWD_BLOCK_SIZE, num_warps=BWD_NUM_WARPS,
        )
        return grad_coeff.to(grad_output.dtype), None, None


def bicubic_iir_sample_affine_from_coeff_triton(
    filtered: torch.Tensor,
    affine: torch.Tensor,
    roi_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a z-aligned CT ROI without materialising inverse coordinates."""
    if filtered.ndim != 4 or not filtered.is_cuda:
        raise ValueError("filtered must be a CUDA tensor with shape (B,T,H,W)")
    return _Bicubic2DAffineTriton.apply(filtered.float(), affine, roi_shape)


@triton.jit
def _bicubic2d_affine_packed_fwd_kernel(
    coeff_ptr, affine_ptr, out_ptr, valid_ptr,
    stride_b, stride_t, stride_c, stride_h, stride_w,
    nTc, nHc, nW, roi_x, roi_y, roi_z, N, dose_scale, minimum_cutoff,
    UPSCALE_FACTOR: tl.constexpr,
    OUTPUT_ZYX: tl.constexpr,
    APPLY_PHYSICAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample a fine depth/height spline directly from packed phase channels."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    n_roi = roi_x * roi_y * roi_z
    b = offs // n_roi
    local = offs - b * n_roi
    if OUTPUT_ZYX:
        ix = local % roi_x
        q = local // roi_x
        iy = q % roi_y
        iz = q // roi_y
    else:
        iz = local % roi_z
        q = local // roi_z
        iy = q % roi_y
        ix = q // roi_y
    abase = b * 12
    x = ix.to(tl.float32); y = iy.to(tl.float32); z = iz.to(tl.float32)
    tq = tl.load(affine_ptr + abase + 0, mask=mask) * x + tl.load(affine_ptr + abase + 1, mask=mask) * y + tl.load(affine_ptr + abase + 2, mask=mask) * z + tl.load(affine_ptr + abase + 3, mask=mask)
    hq = tl.load(affine_ptr + abase + 4, mask=mask) * x + tl.load(affine_ptr + abase + 5, mask=mask) * y + tl.load(affine_ptr + abase + 6, mask=mask) * z + tl.load(affine_ptr + abase + 7, mask=mask)
    wq = tl.load(affine_ptr + abase + 8, mask=mask) * x + tl.load(affine_ptr + abase + 9, mask=mask) * y + tl.load(affine_ptr + abase + 10, mask=mask) * z + tl.load(affine_ptr + abase + 11, mask=mask)
    nT = UPSCALE_FACTOR * nTc
    nH = UPSCALE_FACTOR * nHc
    wi = tl.floor(wq + 0.5).to(tl.int32)
    valid = (tq >= 0.0) & (tq <= nT - 1) & (hq >= 0.0) & (hq <= nH - 1) & (wi >= 0) & (wi < nW) & (tl.abs(wq - wi.to(tl.float32)) <= 1.0e-3)
    t0 = tl.floor(tq).to(tl.int32); h0 = tl.floor(hq).to(tl.int32)
    ft = tq - t0.to(tq.dtype); fh = hq - h0.to(hq.dtype)
    period_t = 2 * (nT - 1) + 2 * (nT == 1)
    period_h = 2 * (nH - 1) + 2 * (nH == 1)
    wi_safe = tl.maximum(0, tl.minimum(wi, nW - 1))
    acc = tl.zeros((BLOCK_SIZE,), tl.float32)
    for a in range(4):
        t = ft
        if a == 0: wt = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
        elif a == 1:
            t2 = t * t; wt = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
        elif a == 2:
            t2 = t * t; wt = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) / 6.0
        else: wt = t * t * t / 6.0
        ti = tl.where(nT > 1, _mirror(t0 - 1 + a, period_t, nT - 1), 0)
        tc = ti // UPSCALE_FACTOR
        ct = ti - UPSCALE_FACTOR * tc
        for htap in range(4):
            t = fh
            if htap == 0: wh = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
            elif htap == 1:
                t2 = t * t; wh = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
            elif htap == 2:
                t2 = t * t; wh = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) / 6.0
            else: wh = t * t * t / 6.0
            hi = tl.where(nH > 1, _mirror(h0 - 1 + htap, period_h, nH - 1), 0)
            hc = hi // UPSCALE_FACTOR
            ch = ct * UPSCALE_FACTOR + (hi - UPSCALE_FACTOR * hc)
            src = b * stride_b + tc * stride_t + ch * stride_c + hc * stride_h + wi_safe * stride_w
            acc += wt * wh * tl.load(coeff_ptr + src, mask=mask & valid, other=0.0)
    if APPLY_PHYSICAL:
        acc *= dose_scale
        valid = valid & (acc >= minimum_cutoff)
    tl.store(out_ptr + offs, tl.where(valid, acc, 0.0), mask=mask)
    tl.store(valid_ptr + offs, valid.to(tl.int8), mask=mask)


@triton.jit
def _bicubic2d_affine_packed_bwd_kernel(
    affine_ptr, gout_ptr, grad_coeff_ptr,
    nTc, nHc, nW, roi_x, roi_y, roi_z, N,
    UPSCALE_FACTOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    n_roi = roi_x * roi_y * roi_z
    b = offs // n_roi
    local = offs - b * n_roi
    iz = local % roi_z
    q = local // roi_z
    iy = q % roi_y
    ix = q // roi_y
    abase = b * 12
    x = ix.to(tl.float32); y = iy.to(tl.float32); z = iz.to(tl.float32)
    tq = tl.load(affine_ptr + abase + 0, mask=mask) * x + tl.load(affine_ptr + abase + 1, mask=mask) * y + tl.load(affine_ptr + abase + 2, mask=mask) * z + tl.load(affine_ptr + abase + 3, mask=mask)
    hq = tl.load(affine_ptr + abase + 4, mask=mask) * x + tl.load(affine_ptr + abase + 5, mask=mask) * y + tl.load(affine_ptr + abase + 6, mask=mask) * z + tl.load(affine_ptr + abase + 7, mask=mask)
    wq = tl.load(affine_ptr + abase + 8, mask=mask) * x + tl.load(affine_ptr + abase + 9, mask=mask) * y + tl.load(affine_ptr + abase + 10, mask=mask) * z + tl.load(affine_ptr + abase + 11, mask=mask)
    nT = UPSCALE_FACTOR * nTc
    nH = UPSCALE_FACTOR * nHc
    wi = tl.floor(wq + 0.5).to(tl.int32)
    valid = (tq >= 0.0) & (tq <= nT - 1) & (hq >= 0.0) & (hq <= nH - 1) & (wi >= 0) & (wi < nW) & (tl.abs(wq - wi.to(tl.float32)) <= 1.0e-3)
    t0 = tl.floor(tq).to(tl.int32); h0 = tl.floor(hq).to(tl.int32)
    ft = tq - t0.to(tq.dtype); fh = hq - h0.to(hq.dtype)
    period_t = 2 * (nT - 1) + 2 * (nT == 1)
    period_h = 2 * (nH - 1) + 2 * (nH == 1)
    gout = tl.load(gout_ptr + offs, mask=mask, other=0.0)
    wi_safe = tl.maximum(0, tl.minimum(wi, nW - 1))
    for a in range(4):
        t = ft
        if a == 0: wt = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
        elif a == 1:
            t2 = t * t; wt = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
        elif a == 2:
            t2 = t * t; wt = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) / 6.0
        else: wt = t * t * t / 6.0
        ti = tl.where(nT > 1, _mirror(t0 - 1 + a, period_t, nT - 1), 0)
        tc = ti // UPSCALE_FACTOR
        ct = ti - UPSCALE_FACTOR * tc
        for htap in range(4):
            t = fh
            if htap == 0: wh = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0
            elif htap == 1:
                t2 = t * t; wh = (4.0 - 6.0 * t2 + 3.0 * t2 * t) / 6.0
            elif htap == 2:
                t2 = t * t; wh = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t2 * t) / 6.0
            else: wh = t * t * t / 6.0
            hi = tl.where(nH > 1, _mirror(h0 - 1 + htap, period_h, nH - 1), 0)
            hc = hi // UPSCALE_FACTOR
            ch = ct * UPSCALE_FACTOR + (hi - UPSCALE_FACTOR * hc)
            dst = ((((b * nTc + tc) * (UPSCALE_FACTOR * UPSCALE_FACTOR) + ch) * nHc + hc) * nW + wi_safe)
            tl.atomic_add(grad_coeff_ptr + dst, wt * wh * gout, mask=mask & valid)


class _Bicubic2DAffinePackedTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff: torch.Tensor, affine: torch.Tensor, roi_shape: tuple[int, int, int]):
        if coeff.ndim != 5:
            raise ValueError(f"expected packed coefficients (B,T,C,H,W), got {tuple(coeff.shape)}")
        upscale_factor = math.isqrt(int(coeff.shape[2]))
        if upscale_factor < 2 or upscale_factor * upscale_factor != coeff.shape[2]:
            raise ValueError(
                "packed coefficient channels must be a square of an upscale "
                f"factor >= 2, got {coeff.shape[2]}"
            )
        affine_c = affine.to(device=coeff.device, dtype=torch.float32).contiguous()
        b, nTc, _, nHc, nW = coeff.shape
        if affine_c.shape != (b, 3, 4):
            raise ValueError(f"expected affine ({b},3,4), got {tuple(affine_c.shape)}")
        roi_x, roi_y, roi_z = (int(v) for v in roi_shape)
        total = b * roi_x * roi_y * roi_z
        out = torch.empty(total, device=coeff.device, dtype=torch.float32)
        valid = torch.empty(total, device=coeff.device, dtype=torch.bool)
        _bicubic2d_affine_packed_fwd_kernel[(triton.cdiv(total, 256),)](
            coeff, affine_c, out, valid,
            *coeff.stride(), nTc, nHc, nW,
            roi_x, roi_y, roi_z, total, 1.0, -math.inf,
            UPSCALE_FACTOR=upscale_factor,
            OUTPUT_ZYX=False,
            APPLY_PHYSICAL=False,
            BLOCK_SIZE=256, num_warps=4,
        )
        ctx.save_for_backward(affine_c)
        ctx.coeff_shape = coeff.shape
        ctx.coeff_dtype = coeff.dtype
        ctx.roi_shape = (roi_x, roi_y, roi_z)
        ctx.upscale_factor = upscale_factor
        ctx.needs_grad = coeff.requires_grad
        ctx.mark_non_differentiable(valid)
        return out.reshape(b, roi_x, roi_y, roi_z), valid.reshape(b, roi_x, roi_y, roi_z)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _grad_valid: torch.Tensor | None):
        if not ctx.needs_grad:
            return None, None, None
        (affine,) = ctx.saved_tensors
        b, nTc, _, nHc, nW = ctx.coeff_shape
        roi_x, roi_y, roi_z = ctx.roi_shape
        gout = grad_output.contiguous().float()
        grad_coeff = torch.zeros(ctx.coeff_shape, device=gout.device, dtype=torch.float32)
        total = gout.numel()
        _bicubic2d_affine_packed_bwd_kernel[(triton.cdiv(total, BWD_BLOCK_SIZE),)](
            affine, gout, grad_coeff, nTc, nHc, nW,
            roi_x, roi_y, roi_z, total,
            UPSCALE_FACTOR=ctx.upscale_factor,
            BLOCK_SIZE=BWD_BLOCK_SIZE, num_warps=BWD_NUM_WARPS,
        )
        return grad_coeff.to(ctx.coeff_dtype), None, None


def bicubic_iir_sample_affine_from_packed_coeff_triton(
    packed: torch.Tensor,
    affine: torch.Tensor,
    roi_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample fine-grid coefficients from packed ``(B,T,r**2,H,W)`` channels."""
    if not packed.is_cuda:
        raise ValueError("packed coefficients must be a CUDA tensor")
    return _Bicubic2DAffinePackedTriton.apply(packed, affine, roi_shape)


def bicubic_iir_sample_affine_from_packed_coeff_zyx_triton(
    packed: torch.Tensor,
    affine: torch.Tensor,
    roi_shape: tuple[int, int, int],
    *,
    dose_scale: float,
    minimum_cutoff: float,
) -> torch.Tensor:
    """Inference sampler emitting cutoff physical dose directly in ZYX order."""
    if packed.ndim != 5 or not packed.is_cuda:
        raise ValueError("packed must be CUDA (B,T,C,H,W)")
    upscale_factor = math.isqrt(int(packed.shape[2]))
    if upscale_factor < 2 or upscale_factor * upscale_factor != packed.shape[2]:
        raise ValueError("packed phase channels must be a square upscale factor")
    affine_c = affine.to(device=packed.device, dtype=torch.float32).contiguous()
    batch, nTc, _, nHc, nW = packed.shape
    if affine_c.shape != (batch, 3, 4):
        raise ValueError(
            f"expected affine ({batch},3,4), got {tuple(affine_c.shape)}"
        )
    roi_x, roi_y, roi_z = (int(value) for value in roi_shape)
    total = batch * roi_x * roi_y * roi_z
    out = torch.empty(total, device=packed.device, dtype=torch.float32)
    # Validity is not returned, but retaining the byte output avoids a second
    # inference-only kernel variant and is negligible beside the FP32 dose.
    valid = torch.empty(total, device=packed.device, dtype=torch.bool)
    _bicubic2d_affine_packed_fwd_kernel[(triton.cdiv(total, 256),)](
        packed,
        affine_c,
        out,
        valid,
        *packed.stride(),
        nTc,
        nHc,
        nW,
        roi_x,
        roi_y,
        roi_z,
        total,
        float(dose_scale),
        float(minimum_cutoff),
        UPSCALE_FACTOR=upscale_factor,
        OUTPUT_ZYX=True,
        APPLY_PHYSICAL=True,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return out.reshape(batch, roi_z, roi_y, roi_x)


def bicubic_iir_sample_triton(
    raw: torch.Tensor,
    points_th: torch.Tensor,
    target_w_index: torch.Tensor,
    zq: torch.Tensor,
    *,
    prefilter_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Prefilter T/H in FP32 by default, then sample in the output compute dtype."""
    if prefilter_dtype not in (torch.float32, torch.float64):
        raise ValueError("prefilter_dtype must be torch.float32 or torch.float64")
    output_dtype = _compute_dtype(raw.dtype)
    with torch.amp.autocast("cuda", enabled=False):
        filtered = cubic_iir_prefilter_2d_triton(raw.to(prefilter_dtype)).to(output_dtype)
        return bicubic_iir_sample_from_coeff_triton(
            filtered, points_th, target_w_index, zq
        )


class _Bicubic2DZMajorTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, collapsed: torch.Tensor, points_th: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        if collapsed.ndim != 4:
            raise ValueError(f"expected collapsed (B,Nz,T,H), got shape {tuple(collapsed.shape)}")
        if points_th.ndim != 3 or points_th.shape[-1] != 2:
            raise ValueError(f"expected points_th (B,N,2), got shape {tuple(points_th.shape)}")
        if zq.shape != points_th.shape[:2]:
            raise ValueError(f"zq shape {tuple(zq.shape)} must match points_th's (B,N)")
        if collapsed.shape[0] != points_th.shape[0]:
            raise ValueError(
                f"batch size mismatch: collapsed {collapsed.shape[0]} vs points_th {points_th.shape[0]}"
            )
        if not collapsed.is_cuda or not points_th.is_cuda:
            raise ValueError("bicubic_zcollapsed_sample_triton requires CUDA tensors")
        if collapsed.device != points_th.device:
            raise ValueError(
                f"device mismatch: collapsed is on {collapsed.device}, points_th is on {points_th.device}"
            )
        if collapsed.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            raise ValueError(f"unsupported dtype {collapsed.dtype}")

        storage_dtype = collapsed.dtype
        compute_dtype = _compute_dtype(storage_dtype)
        collapsed_c = collapsed.detach().to(compute_dtype).contiguous()
        tq, hq, bq, b, n = _flatten_points(points_th.detach(), compute_dtype)
        zq_flat = zq.detach().reshape(-1).to(device=collapsed.device, dtype=torch.int32).contiguous()

        _, nZ, nT, nH = collapsed_c.shape
        out = torch.empty(b * n, device=collapsed.device, dtype=compute_dtype)
        grid = lambda meta: (triton.cdiv(b * n, meta["BLOCK_SIZE"]),)
        _bicubic2d_fwd_zmajor_kernel[grid](
            collapsed_c, tq, hq, bq, zq_flat, out, zq_flat,
            nT, nH, 1, nZ, 0.0, b * n, _gpu_class(collapsed.device),
            SINGLE_BATCH=False, ORDERED_Z=False, CHECK_BOUNDS=False,
        )

        ctx.save_for_backward(tq, hq, bq, zq_flat)
        ctx.collapsed_shape = collapsed_c.shape
        ctx.storage_dtype = storage_dtype
        ctx.compute_dtype = compute_dtype
        ctx.needs_grad = collapsed.requires_grad
        return out.reshape(b, n).to(storage_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_grad:
            return None, None, None
        tq, hq, bq, zq_flat = ctx.saved_tensors
        b, nZ, nT, nH = ctx.collapsed_shape

        gout = grad_output.reshape(-1).detach().to(ctx.compute_dtype).contiguous()
        N = gout.numel()
        grad_accum = torch.zeros(ctx.collapsed_shape, device=grad_output.device, dtype=ctx.compute_dtype)
        grid = (triton.cdiv(N, BWD_BLOCK_SIZE),)
        _bicubic2d_bwd_zmajor_kernel[grid](
            tq, hq, bq, zq_flat, gout, grad_accum, nT, nH, nZ, N,
            BLOCK_SIZE=BWD_BLOCK_SIZE, num_warps=BWD_NUM_WARPS,
        )
        grad_collapsed = (
            grad_accum.to(ctx.storage_dtype) if ctx.storage_dtype != ctx.compute_dtype else grad_accum
        )
        return grad_collapsed, None, None




def bicubic_zcollapsed_sample_triton_single(
    coeff: torch.Tensor,
    tq: torch.Tensor,
    hq: torch.Tensor,
    target_w_index: torch.Tensor,
    zq: torch.Tensor | None = None,
    *,
    cval: float = 0.0,
) -> torch.Tensor:
    """Fast no-grad sampler for one volume with separate flat coordinates."""
    if coeff.ndim != 3:
        raise ValueError(f"expected coeff (T,H,W), got shape {tuple(coeff.shape)}")
    if tq.ndim != 1 or hq.shape != tq.shape:
        raise ValueError("tq and hq must be flat tensors with matching shapes")
    if zq is not None and zq.shape != tq.shape:
        raise ValueError("zq must match tq when provided")
    compute_dtype = _compute_dtype(coeff.dtype)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        coeff_compute = coeff.to(compute_dtype).contiguous().unsqueeze(0)
        target = target_w_index.to(
            device=coeff.device, dtype=torch.int32
        ).contiguous()
        collapsed = _collapse_integer_z_no_grad(coeff_compute, target)
        tq_c = tq.to(device=coeff.device, dtype=compute_dtype).contiguous()
        hq_c = hq.to(device=coeff.device, dtype=compute_dtype).contiguous()
        ordered_z = zq is None
        zq_c = (
            target
            if ordered_z
            else zq.to(device=coeff.device, dtype=torch.int32).contiguous()
        )
        out = torch.empty(tq.numel(), device=coeff.device, dtype=compute_dtype)
        _, nZ, nT, nH = collapsed.shape
        if ordered_z and out.numel() % nZ:
            raise ValueError("ordered z queries must contain a whole number of planes")
        grid = lambda meta: (triton.cdiv(out.numel(), meta["BLOCK_SIZE"]),)
        _bicubic2d_fwd_zmajor_kernel[grid](
            collapsed, tq_c, hq_c, zq_c, zq_c, out, target,
            nT, nH, coeff.shape[2], nZ, float(cval), out.numel(),
            _gpu_class(coeff.device),
            SINGLE_BATCH=True, ORDERED_Z=ordered_z, CHECK_BOUNDS=True,
        )
        return out


def bicubic_zcollapsed_sample_triton(
    coeff: torch.Tensor,
    points_th: torch.Tensor,
    target_w_index: torch.Tensor,
    zq: torch.Tensor,
) -> torch.Tensor:
    """Sample a z-aligned coefficient volume.

    `coeff` has shape `(B, T, H, W)`, `points_th` is `(B, N, 2)`,
    `target_w_index` maps target planes to integer W indices, and `zq` selects
    a target plane for each query. Callers must verify z alignment before using
    this fast path.

    Float16 and bfloat16 inputs are promoted before the collapse, including under
    autocast, and produce float32 output. Float32 and float64 keep their dtype.
    Autograd propagates through the input cast, so mixed-precision activations
    remain supported.
    """
    if coeff.ndim != 4:
        raise ValueError(f"expected coeff (B,T,H,W), got shape {tuple(coeff.shape)}")
    if coeff.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"unsupported dtype {coeff.dtype}")
    if not coeff.is_cuda or not points_th.is_cuda:
        raise ValueError("bicubic_zcollapsed_sample_triton requires CUDA tensors")
    if target_w_index.ndim != 1:
        raise ValueError(f"expected target_w_index (Nz,), got shape {tuple(target_w_index.shape)}")

    compute_dtype = _compute_dtype(coeff.dtype)
    with torch.amp.autocast("cuda", enabled=False):
        coeff_compute = coeff.to(compute_dtype)
        collapsed = (
            collapse_integer_z(coeff_compute, target_w_index)
            if coeff.requires_grad
            else _collapse_integer_z_no_grad(coeff_compute, target_w_index)
        )
        return _Bicubic2DZMajorTriton.apply(collapsed, points_th, zq)


def bicubic_zcollapsed_sample(
    coeff: torch.Tensor,
    points_th: torch.Tensor,
    target_w_index: torch.Tensor,
    zq: torch.Tensor,
    *,
    backend: str = "triton",
) -> torch.Tensor:
    """Use Triton by default.

    Upstream also offers a CuPy fallback via ``patient_space_resample_cuda``.
    That module is deliberately not vendored -- this checkpoint requires the
    Triton backend (``DosePredictor._validate_grid`` enforces it), and pulling
    it in would be the one import escaping this repository.
    """
    if backend == "triton":
        return bicubic_zcollapsed_sample_triton(coeff, points_th, target_w_index, zq)
    if backend == "cupy":
        raise NotImplementedError(
            "the CuPy z-collapse fallback is not vendored in the inference "
            "package; use bicubic_z_align_backend='triton'"
        )
    raise ValueError(f"unknown backend {backend!r}, expected 'triton' or 'cupy'")


# -- proton: packed-coefficient sampling with fused physical scaling and ZYX
# output. The photon path folds the same two steps into
# _bicubic2d_affine_packed_fwd_kernel via constexpr flags; this separate
# kernel is kept because the proton predictors call it directly and it is
# compiled with do_not_specialize, which matters for their varying ROI sizes.

@triton.jit(do_not_specialize=[8, 9, 10, 11, 12, 13, 14, 15, 16])
def _bicubic2d_affine_packed_scaled_zyx_kernel(
    coeff_ptr,
    affine_ptr,
    out_ptr,
    stride_b,
    stride_t,
    stride_c,
    stride_h,
    stride_w,
    nTc,
    nHc,
    nW,
    roi_x,
    roi_y,
    roi_z,
    total,
    dose_scale,
    cutoff,
    UPSCALE_FACTOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sample directly into physical-dose, cutoff-applied contiguous ZYX."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    voxels_per_batch = roi_x * roi_y * roi_z
    batch_index = offsets // voxels_per_batch
    local = offsets - batch_index * voxels_per_batch
    ix = local % roi_x
    quotient = local // roi_x
    iy = quotient % roi_y
    iz = quotient // roi_y
    affine_base = batch_index * 12
    x = ix.to(tl.float32)
    y = iy.to(tl.float32)
    z = iz.to(tl.float32)
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
    nT = UPSCALE_FACTOR * nTc
    nH = UPSCALE_FACTOR * nHc
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
    wi_safe = tl.maximum(0, tl.minimum(wi, nW - 1))
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
            _mirror(t0 - 1 + depth_tap, period_t, nT - 1),
            0,
        )
        tc = ti // UPSCALE_FACTOR
        depth_phase = ti - UPSCALE_FACTOR * tc
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
                _mirror(h0 - 1 + height_tap, period_h, nH - 1),
                0,
            )
            hc = hi // UPSCALE_FACTOR
            channel = (
                depth_phase * UPSCALE_FACTOR
                + hi
                - UPSCALE_FACTOR * hc
            )
            source = (
                batch_index * stride_b
                + tc * stride_t
                + channel * stride_c
                + hc * stride_h
                + wi_safe * stride_w
            )
            accumulated += weight_t * weight_h * tl.load(
                coeff_ptr + source, mask=valid, other=0.0
            )
    physical = accumulated * dose_scale
    physical = tl.where(valid & (physical >= cutoff), physical, 0.0)
    tl.store(out_ptr + offsets, physical, mask=mask)

@torch.inference_mode()
def bicubic_iir_sample_affine_from_packed_coeff_scaled_zyx_triton(
    packed: torch.Tensor,
    affine: torch.Tensor,
    roi_shape: tuple[int, int, int],
    *,
    dose_scale: float,
    cutoff: float,
) -> torch.Tensor:
    """Return a contiguous ``(B,Z,Y,X)`` physical-dose ROI in one kernel."""
    if packed.ndim != 5 or not packed.is_cuda:
        raise ValueError(
            "packed coefficients must be a CUDA tensor with shape (B,T,C,H,W)"
        )
    upscale_factor = math.isqrt(int(packed.shape[2]))
    if (
        upscale_factor < 2
        or upscale_factor * upscale_factor != packed.shape[2]
    ):
        raise ValueError("packed phase channels must form a square upscale factor")
    affine_c = affine.to(
        device=packed.device, dtype=torch.float32
    ).contiguous()
    batch, nTc, _, nHc, nW = packed.shape
    if affine_c.shape != (batch, 3, 4):
        raise ValueError(
            f"expected affine ({batch},3,4), got {tuple(affine_c.shape)}"
        )
    roi_x, roi_y, roi_z = (int(value) for value in roi_shape)
    output = torch.empty(
        (batch, roi_z, roi_y, roi_x),
        device=packed.device,
        dtype=torch.float32,
    )
    total = output.numel()
    _bicubic2d_affine_packed_scaled_zyx_kernel[
        (triton.cdiv(total, 256),)
    ](
        packed,
        affine_c,
        output,
        *packed.stride(),
        nTc,
        nHc,
        nW,
        roi_x,
        roi_y,
        roi_z,
        total,
        float(dose_scale),
        float(cutoff),
        UPSCALE_FACTOR=upscale_factor,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output
