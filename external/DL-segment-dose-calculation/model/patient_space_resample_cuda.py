"""CuPy kernels for differentiable cubic B-spline sampling.

`tricubic_sample_cuda` is the general 64-tap sampler. The z-aligned
`bicubic_zcollapsed_sample_cuda` path collapses the integer z axis first and
uses a 16-tap 2D gather. Triton is the default for that fast path; this module is
the fallback and the implementation used for non-aligned geometry.

Both samplers differentiate spline coefficients only. Query coordinates are
fixed geometry. Float16 accumulation uses float32, and the public z-aligned path
promotes float16 and bfloat16 inputs to float32 before the collapse.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import torch

from patient_space_resample import collapse_integer_z


def _weight_block(frac_var: str, out_prefix: str, vtype: str) -> str:
    return f"""
        {{
            {vtype} t = {frac_var}, t2 = t * t, t3 = t2 * t;
            {out_prefix}[0] = ({vtype})1.0 - t; {out_prefix}[0] = {out_prefix}[0] * {out_prefix}[0] * {out_prefix}[0] / ({vtype})6.0;
            {out_prefix}[1] = (({vtype})4.0 - ({vtype})6.0 * t2 + ({vtype})3.0 * t3) / ({vtype})6.0;
            {out_prefix}[2] = (({vtype})1.0 + ({vtype})3.0 * t + ({vtype})3.0 * t2 - ({vtype})3.0 * t3) / ({vtype})6.0;
            {out_prefix}[3] = t3 / ({vtype})6.0;
        }}
    """


# CuPy's ElementwiseKernel parameter declarations need a concrete dtype name (or the
# generic letter "T"); the operation *body* needs a real C++ type keyword. "float"/
# "double" are valid C++ keywords but not valid CuPy param dtype names, hence this map.
_CPP_TO_PARAM_DTYPE = {"T": "T", "float": "float32", "double": "float64"}


def _mirror_tap(offset_expr: str, dim: str, out_var: str) -> str:
    return f"""
            long long {out_var} = {offset_expr};
            if ({dim} > 1) {{
                long long period = 2 * ((long long){dim} - 1);
                long long m = {out_var} % period; if (m < 0) m += period;
                {out_var} = (m > {dim} - 1) ? (period - m) : m;
            }} else {{ {out_var} = 0; }}
    """


def _fracs(query_vars: list[str], base_vars: list[str], frac_vars: list[str], vtype: str) -> str:
    lines = []
    for q, base, frac in zip(query_vars, base_vars, frac_vars):
        lines.append(f"long long {base} = (long long)floor((double){q});")
        lines.append(f"{vtype} {frac} = {q} - ({vtype}){base};")
    return "\n".join(lines)


def _make_tricubic_kernels(storage_type: str, compute_type: str, suffix: str):
    """Build (forward, backward) ElementwiseKernel pair for the 3D (64-tap) gather.

    ``storage_type``: CuPy dtype spec for in/out arrays -- either the generic letter
    "T" (native-precision accumulate, used for float32/float64) or a concrete type
    like "float16" (explicit storage, always promoted to ``compute_type`` internally).
    """
    body = _fracs(["tq", "hq", "wq"], ["t0", "h0", "w0"], ["ft", "fh", "fw"], compute_type)
    body += f"{compute_type} wt[4], wh[4], ww[4];"
    body += _weight_block("ft", "wt", compute_type)
    body += _weight_block("fh", "wh", compute_type)
    body += _weight_block("fw", "ww", compute_type)

    taps = (
        "for (int a = 0; a < 4; a++) {"
        + _mirror_tap("t0 - 1 + a", "nT", "ti")
        + "for (int b = 0; b < 4; b++) {"
        + _mirror_tap("h0 - 1 + b", "nH", "hi")
        + "for (int c = 0; c < 4; c++) {"
        + _mirror_tap("w0 - 1 + c", "nW", "wi")
        + "long long flat = ((long long)bq * nT + ti) * nH * nW + hi * nW + wi;"
    )
    close = "}}}"

    cast_in = "" if compute_type == storage_type else f"({compute_type})"

    fwd = cp.ElementwiseKernel(
        in_params=f"raw {storage_type} coeff, {storage_type} tq, {storage_type} hq, {storage_type} wq, "
        "int32 bq, int32 nT, int32 nH, int32 nW",
        out_params=f"{storage_type} out",
        operation=body
        + f"{compute_type} acc = 0;"
        + taps
        + f"acc += wt[a] * wh[b] * ww[c] * {cast_in}coeff[flat];"
        + close
        + f"out = ({storage_type})acc;",
        name=f"tricubic_bspline_forward_{suffix}",
    )

    bwd = cp.ElementwiseKernel(
        in_params=f"{storage_type} tq, {storage_type} hq, {storage_type} wq, int32 bq, {storage_type} gout, "
        "int32 nT, int32 nH, int32 nW",
        out_params=f"raw {_CPP_TO_PARAM_DTYPE[compute_type]} grad_coeff, int32 dummy",
        operation=body
        + f"{compute_type} g = ({compute_type})gout;"
        + taps
        + "atomicAdd(&(grad_coeff[flat]), wt[a] * wh[b] * ww[c] * g);"
        + close
        + "dummy = 0;",
        name=f"tricubic_bspline_backward_{suffix}",
    )
    return fwd, bwd


def _make_bicubic2d_kernels(storage_type: str, compute_type: str, suffix: str):
    """Build the forward and backward kernels for `(B, Nz, T, H)` input.

    Keeping Nz outside the spatial axes isolates each plane's atomic updates.
    The earlier Nz-innermost layout caused severe cache-line contention.
    """
    body = _fracs(["tq", "hq"], ["t0", "h0"], ["ft", "fh"], compute_type)
    body += f"{compute_type} wt[4], wh[4];"
    body += _weight_block("ft", "wt", compute_type)
    body += _weight_block("fh", "wh", compute_type)

    taps = (
        "long long base_bz = ((long long)bq * nZ + zq) * nT;"
        + "for (int a = 0; a < 4; a++) {"
        + _mirror_tap("t0 - 1 + a", "nT", "ti")
        + "for (int b = 0; b < 4; b++) {"
        + _mirror_tap("h0 - 1 + b", "nH", "hi")
        + "long long flat = (base_bz + ti) * nH + hi;"
    )
    close = "}}"

    cast_in = "" if compute_type == storage_type else f"({compute_type})"

    fwd = cp.ElementwiseKernel(
        in_params=f"raw {storage_type} collapsed, {storage_type} tq, {storage_type} hq, "
        "int32 bq, int32 zq, int32 nT, int32 nH, int32 nZ",
        out_params=f"{storage_type} out",
        operation=body
        + f"{compute_type} acc = 0;"
        + taps
        + f"acc += wt[a] * wh[b] * {cast_in}collapsed[flat];"
        + close
        + f"out = ({storage_type})acc;",
        name=f"bicubic2d_bspline_forward_{suffix}",
    )

    bwd = cp.ElementwiseKernel(
        in_params=f"{storage_type} tq, {storage_type} hq, int32 bq, int32 zq, {storage_type} gout, "
        "int32 nT, int32 nH, int32 nZ",
        out_params=f"raw {_CPP_TO_PARAM_DTYPE[compute_type]} grad_collapsed, int32 dummy",
        operation=body
        + f"{compute_type} g = ({compute_type})gout;"
        + taps
        + "atomicAdd(&(grad_collapsed[flat]), wt[a] * wh[b] * g);"
        + close
        + "dummy = 0;",
        name=f"bicubic2d_bspline_backward_{suffix}",
    )
    return fwd, bwd


_TRICUBIC_FWD_GENERIC, _TRICUBIC_BWD_GENERIC = _make_tricubic_kernels("T", "T", "generic")
_TRICUBIC_FWD_FP16, _TRICUBIC_BWD_FP16 = _make_tricubic_kernels("float16", "float", "fp16")

_BICUBIC2D_FWD_GENERIC, _BICUBIC2D_BWD_GENERIC = _make_bicubic2d_kernels("T", "T", "generic")
_BICUBIC2D_FWD_FP16, _BICUBIC2D_BWD_FP16 = _make_bicubic2d_kernels("float16", "float", "fp16")


def _flatten_points(points: torch.Tensor, ndim: int, dtype: torch.dtype):
    b, n, _ = points.shape
    comps = [points[..., i].reshape(-1).to(dtype).contiguous() for i in range(ndim)]
    bq = torch.arange(b, device=points.device, dtype=torch.int32).repeat_interleave(n)
    return comps, bq, b, n


class _TricubicBSplineCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff: torch.Tensor, points_thw: torch.Tensor) -> torch.Tensor:
        if coeff.ndim != 4:
            raise ValueError(f"expected coeff (B,T,H,W), got shape {tuple(coeff.shape)}")
        if points_thw.ndim != 3 or points_thw.shape[-1] != 3:
            raise ValueError(f"expected points_thw (B,N,3), got shape {tuple(points_thw.shape)}")
        if coeff.shape[0] != points_thw.shape[0]:
            raise ValueError(
                f"batch size mismatch: coeff {coeff.shape[0]} vs points_thw {points_thw.shape[0]}"
            )
        if not coeff.is_cuda or not points_thw.is_cuda:
            raise ValueError("tricubic_sample_cuda requires CUDA tensors")
        if coeff.dtype == torch.float16:
            fwd_kernel = _TRICUBIC_FWD_FP16
        elif coeff.dtype in (torch.float32, torch.float64):
            fwd_kernel = _TRICUBIC_FWD_GENERIC
        else:
            raise ValueError(
                f"unsupported dtype {coeff.dtype}; bfloat16 goes through tricubic_sample_cuda's "
                "cast wrapper, not this internal Function directly"
            )

        coeff_c = coeff.detach().contiguous()
        (tq, hq, wq), bq, b, n = _flatten_points(points_thw.detach(), 3, coeff.dtype)

        _, nT, nH, nW = coeff_c.shape
        coeff_cp = cp.from_dlpack(coeff_c)
        out_cp = cp.empty(b * n, dtype=coeff_cp.dtype)
        fwd_kernel(
            coeff_cp,
            cp.from_dlpack(tq), cp.from_dlpack(hq), cp.from_dlpack(wq), cp.from_dlpack(bq),
            np.int32(nT), np.int32(nH), np.int32(nW),
            out_cp,
        )
        out = torch.from_dlpack(out_cp).reshape(b, n)

        ctx.save_for_backward(tq, hq, wq, bq)
        ctx.coeff_shape = coeff_c.shape
        ctx.coeff_dtype = coeff.dtype
        ctx.needs_coeff_grad = coeff.requires_grad
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_coeff_grad:
            return None, None
        tq, hq, wq, bq = ctx.saved_tensors
        b, nT, nH, nW = ctx.coeff_shape

        if ctx.coeff_dtype == torch.float16:
            bwd_kernel = _TRICUBIC_BWD_FP16
            accum_dtype = torch.float32
        else:
            bwd_kernel = _TRICUBIC_BWD_GENERIC
            accum_dtype = ctx.coeff_dtype

        gout = grad_output.reshape(-1).detach().to(tq.dtype).contiguous()
        grad_accum = torch.zeros(ctx.coeff_shape, device=grad_output.device, dtype=accum_dtype)
        dummy = torch.empty(gout.shape, device=grad_output.device, dtype=torch.int32)

        bwd_kernel(
            cp.from_dlpack(tq), cp.from_dlpack(hq), cp.from_dlpack(wq), cp.from_dlpack(bq),
            cp.from_dlpack(gout),
            np.int32(nT), np.int32(nH), np.int32(nW),
            cp.from_dlpack(grad_accum), cp.from_dlpack(dummy),
        )
        grad_coeff = grad_accum.to(ctx.coeff_dtype) if accum_dtype != ctx.coeff_dtype else grad_accum
        return grad_coeff, None


class _Bicubic2DZCollapsedCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, collapsed: torch.Tensor, points_th: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        if collapsed.ndim != 4:
            raise ValueError(f"expected collapsed (B,Nz,T,H), got shape {tuple(collapsed.shape)}")
        if points_th.ndim != 3 or points_th.shape[-1] != 2:
            raise ValueError(f"expected points_th (B,N,2), got shape {tuple(points_th.shape)}")
        if zq.shape != points_th.shape[:2]:
            raise ValueError(f"zq shape {tuple(zq.shape)} must match points_th's (B,N)")
        if not collapsed.is_cuda or not points_th.is_cuda:
            raise ValueError("bicubic_zcollapsed_sample_cuda requires CUDA tensors")
        if collapsed.dtype == torch.float16:
            fwd_kernel = _BICUBIC2D_FWD_FP16
        elif collapsed.dtype in (torch.float32, torch.float64):
            fwd_kernel = _BICUBIC2D_FWD_GENERIC
        else:
            raise ValueError(f"unsupported dtype {collapsed.dtype}")

        collapsed_c = collapsed.detach().contiguous()
        (tq, hq), bq, b, n = _flatten_points(points_th.detach(), 2, collapsed.dtype)
        zq_flat = zq.detach().reshape(-1).to(torch.int32).contiguous()

        _, nZ, nT, nH = collapsed_c.shape
        collapsed_cp = cp.from_dlpack(collapsed_c)
        out_cp = cp.empty(b * n, dtype=collapsed_cp.dtype)
        fwd_kernel(
            collapsed_cp,
            cp.from_dlpack(tq), cp.from_dlpack(hq), cp.from_dlpack(bq), cp.from_dlpack(zq_flat),
            np.int32(nT), np.int32(nH), np.int32(nZ),
            out_cp,
        )
        out = torch.from_dlpack(out_cp).reshape(b, n)

        ctx.save_for_backward(tq, hq, bq, zq_flat)
        ctx.collapsed_shape = collapsed_c.shape
        ctx.collapsed_dtype = collapsed.dtype
        ctx.needs_grad = collapsed.requires_grad
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_grad:
            return None, None, None
        tq, hq, bq, zq_flat = ctx.saved_tensors
        b, nZ, nT, nH = ctx.collapsed_shape

        if ctx.collapsed_dtype == torch.float16:
            bwd_kernel = _BICUBIC2D_BWD_FP16
            accum_dtype = torch.float32
        else:
            bwd_kernel = _BICUBIC2D_BWD_GENERIC
            accum_dtype = ctx.collapsed_dtype

        gout = grad_output.reshape(-1).detach().to(tq.dtype).contiguous()
        grad_accum = torch.zeros(ctx.collapsed_shape, device=grad_output.device, dtype=accum_dtype)
        dummy = torch.empty(gout.shape, device=grad_output.device, dtype=torch.int32)

        bwd_kernel(
            cp.from_dlpack(tq), cp.from_dlpack(hq), cp.from_dlpack(bq), cp.from_dlpack(zq_flat),
            cp.from_dlpack(gout),
            np.int32(nT), np.int32(nH), np.int32(nZ),
            cp.from_dlpack(grad_accum), cp.from_dlpack(dummy),
        )
        grad_collapsed = (
            grad_accum.to(ctx.collapsed_dtype) if accum_dtype != ctx.collapsed_dtype else grad_accum
        )
        return grad_collapsed, None, None


def tricubic_sample_cuda(coeff: torch.Tensor, points_thw: torch.Tensor) -> torch.Tensor:
    """CUDA-fused drop-in replacement for ``patient_space_resample.tricubic_sample``.

    ``coeff``: ``(B, T, H, W)`` cubic B-spline coefficients (e.g. from
    :func:`patient_space_resample.separable_cubic_prefilter`), CUDA tensor,
    float16/float32/float64 native, bfloat16 via an internal float32 cast (see module
    docstring).
    ``points_thw``: ``(B, N, 3)`` continuous index coordinates, CUDA tensor. Must lie
    within ``[0, T-1] x [0, H-1] x [0, W-1]``.

    Returns ``(B, N)``, differentiable w.r.t. ``coeff``. Not differentiable w.r.t.
    ``points_thw`` (returns ``None`` for that gradient) -- see module docstring.
    """
    if coeff.dtype == torch.bfloat16:
        out = _TricubicBSplineCuda.apply(coeff.to(torch.float32), points_thw)
        return out.to(torch.bfloat16)
    return _TricubicBSplineCuda.apply(coeff, points_thw)



def bicubic_zcollapsed_sample_cuda(
    coeff: torch.Tensor,
    points_th: torch.Tensor,
    target_w_index: torch.Tensor,
    zq: torch.Tensor,
) -> torch.Tensor:
    """Sample a z-aligned coefficient volume with the CuPy fallback.

    Shapes and geometry match `bicubic_zcollapsed_sample_triton`. Float16 and
    bfloat16 coefficients are promoted before the collapse and return float32;
    float32 and float64 preserve their dtype.
    """
    if coeff.ndim != 4:
        raise ValueError(f"expected coeff (B,T,H,W), got shape {tuple(coeff.shape)}")
    if coeff.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"unsupported dtype {coeff.dtype}")

    compute_dtype = (
        torch.float32 if coeff.dtype in (torch.float16, torch.bfloat16) else coeff.dtype
    )
    with torch.amp.autocast("cuda", enabled=False):
        collapsed = collapse_integer_z(coeff.to(compute_dtype), target_w_index)
        return _Bicubic2DZCollapsedCuda.apply(collapsed, points_th, zq)
