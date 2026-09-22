"""Pure-PyTorch differentiable cubic B-spline prefilter + tricubic sampling.

Approximates the CuPy/SciPy ``spline_filter(order=3, mode='mirror')`` +
``map_coordinates(order=3)`` pipeline (see
``docs/reference/bev-cubic-interpolation.md``) with autograd-differentiable
PyTorch ops, so a network's raw BEV-grid output can be spline-reconstructed
at arbitrary continuous points and backpropagated through. No repo-internal
imports (only ``torch``) so this is loadable standalone from either model or
train code.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Cubic B-spline prefilter pole: root of z^2 + 4z + 1 = 0 with |z| < 1.
_Z1 = math.sqrt(3.0) - 2.0


def cubic_bspline_fir_taps(radius: int = 8, *, dtype=torch.float32, device=None) -> torch.Tensor:
    """Return ``(2*radius+1,)`` symmetric FIR taps approximating the cubic
    B-spline prefilter IIR.

    The exact (doubly-infinite) prefilter impulse response is
    ``h[n] = sqrt(3) * z1**|n|`` with ``z1 = sqrt(3) - 2 ≈ -0.26795``, which
    has unit DC gain (``sum_n h[n] == 1``) analytically. Truncating to
    ``|n| <= radius`` and renormalizing to unit sum keeps that property
    exactly for the truncated filter, removing a small systematic bias on
    flat/high-value regions.

    Empirically (verified against ``cpndi.spline_filter`` + ``map_coordinates``
    end-to-end, see ``tests/test_patient_space_resample.py``), error from the
    3-axis compounded truncation decays roughly geometrically with radius
    (matching the |z1|~=0.268 pole magnitude): radius=4 (9 taps) gives ~5%
    mean relative error on a synthetic volume, radius=8 (17 taps) gives
    ~0.02%, radius=10 gives ~0.002%. Default is 8 -- the extra taps cost
    almost nothing since this is one global separable filter pass, not a
    per-point op, so there's little reason to run near the coarser end.
    """
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    n = torch.arange(-radius, radius + 1, dtype=torch.float64)
    taps = math.sqrt(3.0) * (_Z1 ** n.abs())
    taps = taps / taps.sum()
    return taps.to(dtype=dtype, device=device)


def _conv1d_along_axis(x: torch.Tensor, taps: torch.Tensor, axis: int) -> torch.Tensor:
    """Apply a 1D FIR filter along ``axis`` of an N-D tensor with reflect padding."""
    radius = (taps.numel() - 1) // 2
    x_moved = x.movedim(axis, -1)
    shape = x_moved.shape
    flat = x_moved.reshape(-1, 1, shape[-1])
    flat = F.pad(flat, (radius, radius), mode="reflect")
    kernel = taps.view(1, 1, -1).to(dtype=flat.dtype, device=flat.device)
    out = F.conv1d(flat, kernel)
    out = out.reshape(shape)
    return out.movedim(-1, axis)


def separable_cubic_prefilter(x: torch.Tensor, radius: int = 8) -> torch.Tensor:
    """Convert raw BEV-grid samples into cubic B-spline coefficients.

    ``x``: ``(B, T, H, W)`` raw model output (dose samples).
    Returns: ``(B, T, H, W)`` spline coefficients.

    Applies three sequential 1D FIR passes (T, then H, then W) with reflect
    boundary padding, approximating ``cpndi.spline_filter(x, order=3,
    mode='mirror')`` (the separable structure is exact for the true IIR
    filter; see ``docs/reference/bev-cubic-interpolation.md``).
    """
    if x.ndim != 4:
        raise ValueError(f"expected (B,T,H,W), got shape {tuple(x.shape)}")
    taps = cubic_bspline_fir_taps(radius, dtype=x.dtype, device=x.device)
    out = x
    for axis in (1, 2, 3):
        out = _conv1d_along_axis(out, taps, axis)
    return out


def _cubic_bspline_weights(frac: torch.Tensor) -> torch.Tensor:
    """``frac``: ``(...,)`` fractional offset in ``[0, 1)``.

    Returns ``(..., 4)`` weights for the neighbor taps at relative offsets
    ``[-1, 0, 1, 2]`` (standard cubic-convolution B-spline weights, polynomial
    in ``frac`` so this is trivially autograd-differentiable w.r.t. ``frac``
    if ever needed, though the callers here never backprop into coordinates).
    """
    t = frac
    t2 = t * t
    t3 = t2 * t
    w_m1 = (1.0 - t) ** 3 / 6.0
    w_0 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0
    w_1 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0
    w_2 = t3 / 6.0
    return torch.stack((w_m1, w_0, w_1, w_2), dim=-1)


def _catmull_rom_weights(frac: torch.Tensor) -> torch.Tensor:
    """``frac``: ``(...,)`` fractional offset in ``[0, 1)``.

    Returns ``(..., 4)`` weights for the neighbor taps at relative offsets
    ``[-1, 0, 1, 2]`` for the Catmull-Rom cubic convolution kernel (Keys 1981
    cubic convolution with a=-0.5). Unlike the B-spline kernel, Catmull-Rom is
    itself interpolating (w_0(0)=1, w_1(1)=1, all others 0 at integer offsets)
    -- it reproduces raw sample values exactly at grid nodes with no prefilter
    stage, at the cost of being only C1 (vs. the B-spline's C2) continuous.
    """
    t = frac
    t2 = t * t
    t3 = t2 * t
    w_m1 = -0.5 * t3 + t2 - 0.5 * t
    w_0 = 1.5 * t3 - 2.5 * t2 + 1.0
    w_1 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w_2 = 0.5 * t3 - 0.5 * t2
    return torch.stack((w_m1, w_0, w_1, w_2), dim=-1)


def _mirror_reflect_index(idx: torch.Tensor, dim: int) -> torch.Tensor:
    """Whole-sample symmetric ('mirror') reflection of an out-of-range index
    into ``[0, dim-1]``.

    ``cpndi.map_coordinates(mode="constant", prefilter=False)`` does *not*
    zero-pad individual out-of-range stencil taps for spline orders > 1 --
    verified empirically against ``scipy.ndimage.map_coordinates`` (the two
    APIs share this behavior): for a query point that itself lies within
    ``[0, dim-1]``, out-of-range taps needed by the cubic stencil are read via
    mirror (whole-sample symmetric) reflection, identical to ``mode="mirror"``.
    ``cval`` only takes effect when the *query point itself* falls outside
    ``[0, dim-1]`` (never the case here, since points are always sampled
    strictly inside the domain), so that case is intentionally not handled.
    ``mode="grid-constant"`` is the (different, newer) SciPy/CuPy mode that
    actually zero-pads taps -- not what this repo's pipeline uses.
    """
    if dim <= 1:
        return torch.zeros_like(idx)
    period = 2 * (dim - 1)
    folded = idx.remainder(period)
    return torch.where(folded > dim - 1, period - folded, folded)


def collapse_integer_z(coeff: torch.Tensor, target_w_index: torch.Tensor) -> torch.Tensor:
    """Collapse integer-aligned W planes into `(B, Nz, T, H)` spline images.

    The cubic B-spline weights at an integer coordinate are `(1, 4, 1) / 6`.
    Inference uses a direct three-plane gather. When gradients are required, a
    dense blend matrix keeps the backward pass in matmul instead of scatter-add.
    """
    if coeff.ndim != 4:
        raise ValueError(f"expected coeff (B,T,H,W), got shape {tuple(coeff.shape)}")
    w_dim = coeff.shape[3]
    w0 = target_w_index.to(device=coeff.device, dtype=torch.long).clamp(0, w_dim - 1)
    wm1 = torch.where(w0 > 0, w0 - 1, torch.full_like(w0, min(1, w_dim - 1)))
    wp1 = torch.where(w0 < w_dim - 1, w0 + 1, torch.full_like(w0, max(w_dim - 2, 0)))

    if not coeff.requires_grad:
        collapsed = (
            coeff[..., wm1] + 4.0 * coeff[..., w0] + coeff[..., wp1]
        ) * (1.0 / 6.0)
        return collapsed.permute(0, 3, 1, 2).contiguous()

    blend = coeff.new_zeros(target_w_index.shape[0], w_dim)
    rows = torch.arange(target_w_index.shape[0], device=coeff.device)
    for index, weight in ((wm1, 1.0 / 6), (w0, 4.0 / 6), (wp1, 1.0 / 6)):
        blend.index_put_((rows, index), blend.new_full(rows.shape, weight), accumulate=True)

    b, t_dim, h_dim, _ = coeff.shape
    collapsed = coeff.reshape(b * t_dim * h_dim, w_dim) @ blend.T
    return collapsed.reshape(b, t_dim, h_dim, -1).permute(0, 3, 1, 2).contiguous()


def _cubic_convolution_sample(
    values: torch.Tensor, points_thw: torch.Tensor, weight_fn,
) -> torch.Tensor:
    """Shared 4-tap-per-axis (64-tap total) cubic convolution gather, parametrized
    by ``weight_fn`` (e.g. :func:`_cubic_bspline_weights` or
    :func:`_catmull_rom_weights`). See :func:`tricubic_sample` for the full
    contract; this is the part that's identical across cubic-convolution
    schemes -- only the weight polynomial (and whether ``values`` needs a
    prefilter pass first) differs between them.
    """
    if values.ndim != 4:
        raise ValueError(f"expected values (B,T,H,W), got shape {tuple(values.shape)}")
    if points_thw.ndim != 3 or points_thw.shape[-1] != 3:
        raise ValueError(f"expected points_thw (B,N,3), got shape {tuple(points_thw.shape)}")
    if values.shape[0] != points_thw.shape[0]:
        raise ValueError(
            f"batch size mismatch: values {values.shape[0]} vs points_thw {points_thw.shape[0]}"
        )

    b, t_dim, h_dim, w_dim = values.shape
    _, n_points, _ = points_thw.shape

    pts = points_thw.to(values.dtype)
    floor_pts = torch.floor(pts)
    frac = pts - floor_pts  # (B, N, 3) in [0, 1)
    base = floor_pts.long()  # (B, N, 3)

    weights = weight_fn(frac)  # (B, N, 3, 4)
    w_t, w_h, w_w = weights.unbind(dim=2)  # each (B, N, 4)

    offsets = torch.arange(-1, 3, device=values.device, dtype=base.dtype)  # taps [-1,0,1,2]
    idx_t = _mirror_reflect_index(base[..., 0:1] + offsets, t_dim)  # (B, N, 4)
    idx_h = _mirror_reflect_index(base[..., 1:2] + offsets, h_dim)
    idx_w = _mirror_reflect_index(base[..., 2:3] + offsets, w_dim)

    flat_idx = (
        idx_t.view(b, n_points, 4, 1, 1) * (h_dim * w_dim)
        + idx_h.view(b, n_points, 1, 4, 1) * w_dim
        + idx_w.view(b, n_points, 1, 1, 4)
    ).reshape(b, n_points, 64)

    weight = (
        w_t.view(b, n_points, 4, 1, 1)
        * w_h.view(b, n_points, 1, 4, 1)
        * w_w.view(b, n_points, 1, 1, 4)
    ).reshape(b, n_points, 64)

    # Gather along dim=1 with matching batch size (no .expand()) -- gather's
    # backward on a broadcast/expand()-ed input materializes a full
    # (B, N, T*H*W) gradient buffer instead of an efficient scatter-add,
    # which is intractable at real grid sizes (confirmed empirically: at
    # T*H*W~51K and N=16384 this allocated ~7GB and took ~400ms per call
    # instead of ~2ms). Flattening N and the 64 taps into one dim keeps
    # input and index both exactly (B, ...), avoiding the broadcast path.
    values_flat = values.reshape(b, t_dim * h_dim * w_dim)
    gathered = torch.gather(values_flat, 1, flat_idx.reshape(b, n_points * 64))
    gathered = gathered.reshape(b, n_points, 64)

    return (gathered * weight).sum(dim=-1)


def tricubic_sample(coeff: torch.Tensor, points_thw: torch.Tensor) -> torch.Tensor:
    """Evaluate a tricubic B-spline at arbitrary continuous points.

    ``coeff``: ``(B, T, H, W)`` spline coefficients (e.g. from
    :func:`separable_cubic_prefilter`).
    ``points_thw``: ``(B, N, 3)`` continuous index coordinates, where
    component 0 -> T axis, 1 -> H axis, 2 -> W axis. Must lie within
    ``[0, T-1] x [0, H-1] x [0, W-1]``.

    Returns ``(B, N)``, differentiable w.r.t. ``coeff``. It is also technically
    differentiable w.r.t. ``points_thw`` (the B-spline weights are smooth
    polynomials in the fractional offset, so gradient flows through them even
    though the tap *indices* -- via ``floor()`` -- do not contribute); this
    package never uses that path since query points are fixed, non-learnable
    geometry in this project's training design.

    Out-of-range stencil taps near the boundary are handled via mirror
    reflection, matching ``cpndi.map_coordinates(mode="constant",
    prefilter=False)`` (see :func:`_mirror_reflect_index`).
    """
    return _cubic_convolution_sample(coeff, points_thw, _cubic_bspline_weights)


def catmull_rom_sample(raw: torch.Tensor, points_thw: torch.Tensor) -> torch.Tensor:
    """Evaluate a Catmull-Rom cubic convolution interpolant at arbitrary points.

    Same contract as :func:`tricubic_sample`, but takes ``raw`` (the model's
    unfiltered ``(B, T, H, W)`` output) directly -- no prefilter stage, since
    Catmull-Rom is itself interpolating (exactly reproduces sample values at
    integer coordinates). This removes the FIR-vs-exact-IIR approximation
    error entirely, at the cost of C1 (vs. B-spline's C2) smoothness -- worth
    comparing against ``separable_cubic_prefilter`` + ``tricubic_sample`` when
    evaluating interpolation choices.
    """
    return _cubic_convolution_sample(raw, points_thw, _catmull_rom_weights)


def trilinear_sample(raw: torch.Tensor, points_thw: torch.Tensor) -> torch.Tensor:
    """Evaluate a trilinear interpolant via PyTorch's native ``F.grid_sample``.

    Same contract as :func:`tricubic_sample`, but takes ``raw`` directly (no
    prefilter -- trilinear is interpolating) and uses the built-in CUDA kernel
    (8-tap, C0 continuous) instead of a custom 64-tap gather. Cheapest option
    and immune to cubic-spline ringing/overshoot near sharp gradients, at the
    cost of losing smoothness between nodes. Boundary handling is
    ``padding_mode='zeros'`` (native grid_sample has no mirror-reflect mode)
    -- a different convention from :func:`tricubic_sample`'s, but irrelevant
    for points sampled strictly inside the domain, as this package always does.
    """
    if raw.ndim != 4:
        raise ValueError(f"expected raw (B,T,H,W), got shape {tuple(raw.shape)}")
    if points_thw.ndim != 3 or points_thw.shape[-1] != 3:
        raise ValueError(f"expected points_thw (B,N,3), got shape {tuple(points_thw.shape)}")
    b, t_dim, h_dim, w_dim = raw.shape
    _, n_points, _ = points_thw.shape

    pts = points_thw.to(raw.dtype)
    dims = torch.tensor([t_dim - 1, h_dim - 1, w_dim - 1], dtype=raw.dtype, device=raw.device)
    norm = pts / dims * 2.0 - 1.0  # align_corners=True: 0 -> -1, dim-1 -> 1
    # grid_sample's grid channel order is (x, y, z) matching input dims (W, H, D)
    # reversed relative to this package's (T, H, W) convention.
    grid = norm.flip(-1).view(b, n_points, 1, 1, 3)
    vol = raw.unsqueeze(1)  # (B, 1, T, H, W)
    out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.view(b, n_points)
