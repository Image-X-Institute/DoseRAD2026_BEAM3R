"""Beam-direction IDD on GPU, numerically faithful to the challenge evaluator.

``doserad2026_evaluator.metrics_beam.idd_curve_distance`` collapses the dose over
z, then resamples the transverse plane onto a grid rotated by the beam's gantry
angle so that the output's first axis lies along the beam. Summing the other axis
gives the integrated depth-dose curve.

Two existing implementations in this repo do *not* do that:

``train/beam_level_metrics.py``
    integrates along a fixed ``beam_axis``. With ``beam_axis=0`` that is the
    patient's superior-inferior axis, which for a coplanar beam is perpendicular
    to the beam. It ranks patients at spearman +0.14 against the true metric.

``utils.idd_curve_distance_loss``
    sums the last two dims of its input. On BEV tensors ``(B, D, H, W)`` that is
    correct, because D is beam depth. On the CT-space ROI tensors the loss
    actually receives -- ``(1, X, Y, Z)`` -- it collapses Y and Z and leaves a
    curve along the CT left-right axis instead.

This module reproduces the evaluator's geometry in torch so it can run inside the
validation loop on the CT-space prediction that already exists there, for any
``--otf-input-upscale-factor`` and without the CT-to-BEV round trip.

``tests/test_idd_ct.py`` pins it against the SimpleITK original.
"""

from __future__ import annotations

import math

import torch


def compute_idd_curve_torch(
    dose_xyz: torch.Tensor,
    direction: torch.Tensor | tuple[float, float, float],
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    """Integrated depth-dose curve along ``direction``, from an (X, Y, Z) volume.

    Mirrors ``metrics_beam.compute_idd_curve``. Note the layout difference: the
    evaluator takes numpy ``(z, y, x)`` straight from SimpleITK, whereas the
    training pipeline works in ``(x, y, z)``; the transpose is done here so
    callers pass their native layout.

    Returns a 1-D curve whose length spans the in-plane diagonal, so it is
    independent of the gantry angle.
    """
    if dose_xyz.ndim != 3:
        raise ValueError(f"expected an (X, Y, Z) volume, got {tuple(dose_xyz.shape)}")
    d = torch.as_tensor(direction, dtype=torch.float64, device=dose_xyz.device)
    if abs(float(d[2])) > 1e-9:
        raise ValueError("beam leaves the transverse plane; z cannot be summed")

    sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
    # (X, Y, Z) -> sum over z -> (X, Y) -> transpose to the evaluator's (y, x)
    plane = dose_xyz.to(torch.float64).sum(dim=2).transpose(0, 1).contiguous()
    ny, nx = plane.shape

    step = max(sx, sy)
    n = int(math.ceil(math.hypot(nx * sx, ny * sy) / step)) + 1
    theta = math.atan2(float(d[1]), float(d[0]))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # sitk resamples by mapping each OUTPUT point back into the input, so the
    # rotation here is the forward Euler2D transform, not its inverse.
    cx, cy = (nx - 1) * sx / 2.0, (ny - 1) * sy / 2.0

    dev = dose_xyz.device
    u = (torch.arange(n, dtype=torch.float64, device=dev) - (n - 1) / 2.0) * step
    ox = u.view(1, n).expand(n, n)      # output x varies along the last axis
    oy = u.view(n, 1).expand(n, n)

    px = cos_t * ox - sin_t * oy + cx
    py = sin_t * ox + cos_t * oy + cy

    # physical mm -> input index -> grid_sample's [-1, 1], align_corners=True
    gx = (px / sx) * (2.0 / max(nx - 1, 1)) - 1.0
    gy = (py / sy) * (2.0 / max(ny - 1, 1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).to(torch.float32)

    sampled = torch.nn.functional.grid_sample(
        plane.to(torch.float32).unsqueeze(0).unsqueeze(0),
        grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )[0, 0].to(torch.float64)

    # sum over the output's y axis, leaving the curve along the beam
    return sampled.sum(dim=0)


def idd_curve_distance_torch(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    direction: torch.Tensor | tuple[float, float, float],
    spacing_xyz: tuple[float, float, float],
) -> float:
    """Normalised RMS difference of the two IDD curves; ``nan`` if GT is flat."""
    idd_gt = compute_idd_curve_torch(gt_xyz, direction, spacing_xyz)
    peak = float(idd_gt.max())
    if peak <= 0:
        return float("nan")
    idd_pred = compute_idd_curve_torch(pred_xyz, direction, spacing_xyz)
    return float(torch.sqrt(torch.mean(((idd_pred - idd_gt) / peak) ** 2)))


def _plane_from_roi(
    roi_xyz: torch.Tensor,
    roi_origin_xy: tuple[int, int] | None,
    full_shape_xy: tuple[int, int] | None,
) -> torch.Tensor:
    """Collapse an (X, Y, Z) ROI over z and place it in the full transverse plane.

    The evaluator's geometry is defined against the whole CT: the rotation centre
    is the full plane's centre and the output spans its diagonal. The loss only
    holds the ROI, and dose is zero outside it by construction, so embedding the
    summed 2-D plane reproduces the full-volume curve without ever allocating a
    full 3-D volume. Returns the evaluator's (y, x) orientation.
    """
    plane = roi_xyz.sum(dim=2).transpose(0, 1)          # (roi_y, roi_x)
    if roi_origin_xy is None or full_shape_xy is None:
        return plane
    nx, ny = int(full_shape_xy[0]), int(full_shape_xy[1])
    x0, y0 = int(roi_origin_xy[0]), int(roi_origin_xy[1])
    ry, rx = plane.shape
    full = plane.new_zeros((ny, nx))
    return full.index_put_(
        (torch.arange(y0, y0 + ry, device=plane.device).view(ry, 1),
         torch.arange(x0, x0 + rx, device=plane.device).view(1, rx)),
        plane,
    )


def idd_curve_distance_loss_ct(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    direction: torch.Tensor | tuple[float, float, float],
    spacing_xyz: tuple[float, float, float],
    *,
    roi_origin_xy: tuple[int, int] | None = None,
    full_shape_xy: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Differentiable form of :func:`idd_curve_distance_torch`, for use in a loss.

    Differences from the metric version, all deliberate:

    * returns a tensor rather than a float, so the graph survives;
    * stays in the input dtype (the metric uses float64 to match the evaluator
      bit-for-bit, which is wasteful and unsupported by some grid_sample builds);
    * takes the ROI plus its offset instead of a full volume, because that is
      what ``OtfGpuBatchMaterializer.loss`` has in hand;
    * normalises by the ground-truth peak as a *constant* -- the peak is a
      property of the reference, so letting gradient flow through it would let
      the model reduce the loss by rescaling the target it is being judged against.

    Returns a zero-valued tensor (still attached) when the reference is flat.
    """
    d = torch.as_tensor(direction, dtype=torch.float32)
    if abs(float(d[2])) > 1e-9:
        raise ValueError("beam leaves the transverse plane; z cannot be summed")

    sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
    plane_p = _plane_from_roi(pred_xyz, roi_origin_xy, full_shape_xy)
    with torch.no_grad():
        plane_g = _plane_from_roi(gt_xyz, roi_origin_xy, full_shape_xy)
    ny, nx = plane_g.shape

    step = max(sx, sy)
    n = int(math.ceil(math.hypot(nx * sx, ny * sy) / step)) + 1
    theta = math.atan2(float(d[1]), float(d[0]))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = (nx - 1) * sx / 2.0, (ny - 1) * sy / 2.0

    dev, dt = plane_p.device, plane_p.dtype
    u = (torch.arange(n, device=dev, dtype=dt) - (n - 1) / 2.0) * step
    ox = u.view(1, n).expand(n, n)
    oy = u.view(n, 1).expand(n, n)
    gx = ((cos_t * ox - sin_t * oy + cx) / sx) * (2.0 / max(nx - 1, 1)) - 1.0
    gy = ((sin_t * ox + cos_t * oy + cy) / sy) * (2.0 / max(ny - 1, 1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)

    def _curve(plane: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.grid_sample(
            plane.unsqueeze(0).unsqueeze(0), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )[0, 0].sum(dim=0)

    with torch.no_grad():
        curve_g = _curve(plane_g)
        peak = curve_g.max()
    if float(peak) <= 0:
        return plane_p.sum() * 0.0
    return torch.sqrt(torch.mean(((_curve(plane_p) - curve_g) / peak) ** 2))


def direction_from_gantry(gantry_deg: float) -> tuple[float, float, float]:
    """Beam propagation direction, matching ``metrics_beam.directions_of``."""
    g = math.radians(float(gantry_deg))
    return (-math.sin(g), math.cos(g), 0.0)
