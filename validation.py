"""
validation.py
=============
Validation metrics and visualisation utilities for BEV photon dose prediction.

Metrics
-------
1. Masked MAE          – MAE inside the high-dose region (≥10 % of max GT dose),
                         normalised by the GT beam maximum.
2. IDD Curve Distance  – RMS difference between predicted and GT integrated
                         depth-dose curves, normalised by the GT IDD peak.
3. MSE                 – Mean squared error over the full volume.
4. Gradient MAE        – MAE of the 3-D image gradient magnitude.
5. Gamma pass rate     – Fraction of valid reference voxels with γ ≤ 1 (optional;
                         ``pymedphys.gamma``, local 1 %/1 mm by default).

All metrics are computed per-beam and then summarised (mean ± std) across
the validation set.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Gamma (pymedphys) — defaults aligned with notebooks/pymedphys_gamma.ipynb
# ---------------------------------------------------------------------------

# BEV cuboid voxels are 2 mm along depth (axis 0) and transverse axes 1–2.
DEFAULT_BEV_SPACING_MM: tuple[float, float, float] = (2.0, 2.0, 2.0)

def patient_cohort(patient_id: str) -> str:
    """Anatomy cohort from DoseRAD patient id prefix."""
    if patient_id.startswith("1ABB"):
        return "abdomen"
    if patient_id.startswith("1THB"):
        return "thorax"
    return "other"


def _gantry_bin(gantry_angle_deg: float | None) -> float:
    if gantry_angle_deg is None or np.isnan(gantry_angle_deg):
        return float("nan")
    return float(int(round(float(gantry_angle_deg))))


DEFAULT_PHOTON_GAMMA_OPTIONS: dict[str, Any] = {
    "dose_percent_threshold": 1,
    "distance_mm_threshold": 1,
    "lower_percent_dose_cutoff": 20,
    "interp_fraction": 10,
    "max_gamma": 2,
    "random_subset": None,
    "local_gamma": True,
    "ram_available": 2**30*50,
}


def bev_dose_axes_zyx(
    shape_dhw: tuple[int, int, int],
    spacing_mm: tuple[float, float, float] = DEFAULT_BEV_SPACING_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinate axes (z, y, x) in mm for a BEV dose array shaped ``(D, H, W)``."""
    if len(shape_dhw) != 3:
        raise ValueError(f"expected 3-D shape (D, H, W), got {shape_dhw}")
    if len(spacing_mm) != 3:
        raise ValueError(f"expected 3 spacing values (mm), got {spacing_mm}")
    d, h, w = shape_dhw
    sz, sy, sx = spacing_mm
    return (
        np.arange(d, dtype=np.float64) * sz,
        np.arange(h, dtype=np.float64) * sy,
        np.arange(w, dtype=np.float64) * sx,
    )


def photon_beam_gamma_pass_rate(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] = DEFAULT_BEV_SPACING_MM,
    gamma_options: dict[str, Any] | None = None,
) -> float:
    """γ pass rate for one photon beam dose volume (prediction vs reference).

    Uses ``pymedphys.gamma`` with the reference = ground truth and evaluation =
    prediction, matching ``notebooks/pymedphys_gamma.ipynb``. Volumes are
    ``(D, H, W)`` BEV arrays (depth along axis 0); coordinates are regular
  grids with spacing ``spacing_mm`` (default 2 mm, from the 40×40 cm cuboid).

    Returns the fraction of non-NaN γ values with γ ≤ 1 in ``[0, 1]``.
    Returns NaN if there are no valid γ points (e.g. empty or below dose cutoff).
    """
    import pymedphys

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"expected 3-D volumes (D, H, W), got ndim={pred.ndim}")

    opts = dict(DEFAULT_PHOTON_GAMMA_OPTIONS)
    if gamma_options is not None:
        opts.update(gamma_options)

    # BEV (D, H, W): depth is axis 0, matching pymedphys zyx dose layout.
    axes = bev_dose_axes_zyx(pred.shape, spacing_mm)
    gamma = pymedphys.gamma(
        axes,
        target,
        axes,
        pred,
        **opts,
    )
    valid = gamma[~np.isnan(gamma)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid <= 1.0))


def bev_gamma_pass_rate(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] = DEFAULT_BEV_SPACING_MM,
    gamma_options: dict[str, Any] | None = None,
    crop_to_high_dose_bbox: bool = False,
    bbox_margin_vox: int = 2,
    random_subset: int | None = None,
    use_fast_local: bool = False,
    gamma_profile: Literal["training", "clinical"] = "training",
) -> float:
    """γ pass rate for BEV dose volumes ``(D, H, W)`` (depth = axis 0).

    * ``gamma_profile='training'`` — legacy notebook defaults (1 % / 1 mm, 20 % cutoff).
    * ``gamma_profile='clinical'`` — align with CT eval (2 % / 3 mm, 10 % cutoff).

    Speed options match ``ct_gamma_pass_rate`` (fast local, crop bbox, random subset).
    """
    if gamma_profile == "clinical":
        dose_percent, distance_mm, cutoff = 2.0, 3.0, 0.10
        default_opts = dict(DEFAULT_CT_GAMMA_2MM_3PCT)
    else:
        dose_percent, distance_mm, cutoff = 1.0, 1.0, 0.20
        default_opts = dict(DEFAULT_PHOTON_GAMMA_OPTIONS)

    if use_fast_local:
        pred_w = np.asarray(pred, dtype=np.float64)
        target_w = np.asarray(target, dtype=np.float64)
        if crop_to_high_dose_bbox:
            mask = high_dose_region_mask(target_w, cutoff)
            pred_w, target_w, _ = crop_volumes_to_mask_bbox(
                pred_w, target_w, mask, margin_vox=bbox_margin_vox
            )
        return ct_gamma_pass_rate_fast_local(
            pred_w,
            target_w,
            spacing_mm,
            dose_percent=dose_percent,
            distance_mm=distance_mm,
            dose_cutoff_fraction=cutoff,
        )

    if (
        gamma_profile == "training"
        and not crop_to_high_dose_bbox
        and random_subset is None
        and gamma_options is None
    ):
        return photon_beam_gamma_pass_rate(pred, target, spacing_mm=spacing_mm)

    opts = dict(default_opts)
    if gamma_options is not None:
        opts.update(gamma_options)
    return ct_gamma_pass_rate(
        pred,
        target,
        spacing_mm,
        gamma_options=opts,
        crop_to_high_dose_bbox=crop_to_high_dose_bbox,
        bbox_margin_vox=bbox_margin_vox,
        random_subset=random_subset,
        use_fast_local=False,
    )


def sitk_spacing_zyx_mm(ref_img: Any) -> tuple[float, float, float]:
    """Voxel spacing ``(dz, dy, dx)`` in mm for a SimpleITK image array ``(z, y, x)``."""
    import SimpleITK as sitk

    if not isinstance(ref_img, sitk.Image):
        raise TypeError("ref_img must be a SimpleITK.Image")
    sx, sy, sz = ref_img.GetSpacing()
    return float(sz), float(sy), float(sx)


def ct_spacing_axes_zyx(
    shape_zyx: tuple[int, int, int],
    spacing_zyx_mm: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular grid axes in mm for CT dose arrays with layout ``(z, y, x)``."""
    if len(shape_zyx) != 3 or len(spacing_zyx_mm) != 3:
        raise ValueError(f"expected 3-D shape and spacing, got {shape_zyx} {spacing_zyx_mm}")
    dz, dy, dx = spacing_zyx_mm
    nz, ny, nx = shape_zyx
    return (
        np.arange(nz, dtype=np.float64) * dz,
        np.arange(ny, dtype=np.float64) * dy,
        np.arange(nx, dtype=np.float64) * dx,
    )


DEFAULT_CT_GAMMA_2MM_3PCT: dict[str, Any] = {
    **DEFAULT_PHOTON_GAMMA_OPTIONS,
    "dose_percent_threshold": 2,
    "distance_mm_threshold": 3,
    "lower_percent_dose_cutoff": 10,
}


def high_dose_region_mask(target: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    """Boolean mask where ``target >= threshold * max(target)``."""
    max_gt = float(np.max(target))
    if max_gt <= 0:
        return np.zeros_like(target, dtype=bool)
    return target >= threshold * max_gt


def crop_volumes_to_mask_bbox(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    margin_vox: int = 2,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Crop ``(z, y, x)`` volumes to the bounding box of *mask* plus a voxel margin.

    Returns cropped arrays and the ``(z0, y0, x0)`` index origin for physical axes.
    """
    if not np.any(mask):
        return pred, target, (0, 0, 0)
    zz, yy, xx = np.where(mask)
    z0 = max(0, int(zz.min()) - margin_vox)
    y0 = max(0, int(yy.min()) - margin_vox)
    x0 = max(0, int(xx.min()) - margin_vox)
    z1 = min(pred.shape[0], int(zz.max()) + 1 + margin_vox)
    y1 = min(pred.shape[1], int(yy.max()) + 1 + margin_vox)
    x1 = min(pred.shape[2], int(xx.max()) + 1 + margin_vox)
    sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    return pred[sl], target[sl], (z0, y0, x0)


def ct_spacing_axes_zyx_with_origin(
    shape_zyx: tuple[int, int, int],
    spacing_zyx_mm: tuple[float, float, float],
    origin_index_zyx: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical mm axes for a cropped subvolume starting at ``origin_index_zyx``."""
    dz, dy, dx = spacing_zyx_mm
    oz, oy, ox = origin_index_zyx
    nz, ny, nx = shape_zyx
    return (
        (oz + np.arange(nz, dtype=np.float64)) * dz,
        (oy + np.arange(ny, dtype=np.float64)) * dy,
        (ox + np.arange(nx, dtype=np.float64)) * dx,
    )


def _gamma_offset_shifts(
    shape: tuple[int, ...],
    offset: tuple[int, int, int],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Slice pairs for evaluating γ at shifted grid positions (from eval_doserad_metrics)."""
    eval_slices: list[slice] = []
    ref_slices: list[slice] = []
    for size, shift in zip(shape, offset):
        start = max(0, -shift)
        stop = min(size, size - shift)
        eval_slices.append(slice(start, stop))
        ref_slices.append(slice(max(0, shift), max(0, shift) + (stop - start)))
    return tuple(eval_slices), tuple(ref_slices)


def _gamma_search_offsets_zyx(
    spacing_zyx_mm: tuple[float, float, float],
    distance_mm: float,
) -> list[tuple[tuple[int, int, int], float]]:
    """Integer voxel shifts within *distance_mm* (nearest-neighbour local γ)."""
    dz, dy, dx = spacing_zyx_mm
    limits = np.ceil(distance_mm / np.array([dz, dy, dx], dtype=float)).astype(int)
    offsets: list[tuple[tuple[int, int, int], float]] = []
    for iz in range(-limits[0], limits[0] + 1):
        for iy in range(-limits[1], limits[1] + 1):
            for ix in range(-limits[2], limits[2] + 1):
                dist = float(np.sqrt((iz * dz) ** 2 + (iy * dy) ** 2 + (ix * dx) ** 2))
                if dist <= distance_mm:
                    offsets.append(((iz, iy, ix), dist))
    return offsets


def ct_gamma_pass_rate_fast_local(
    pred: np.ndarray,
    target: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    *,
    dose_percent: float = 2.0,
    distance_mm: float = 3.0,
    dose_cutoff_fraction: float = 0.10,
) -> float:
    """Approximate local γ pass rate without pymedphys interpolation (much faster, coarser).

    Searches γ only at discrete voxel shifts within *distance_mm* (no sub-voxel interpolation).
    Pass rate is computed only over reference voxels ≥ ``dose_cutoff_fraction`` × max dose.
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = high_dose_region_mask(target, dose_cutoff_fraction)
    if not np.any(mask):
        return float("nan")

    dose_frac = dose_percent / 100.0
    denom = np.maximum(np.abs(target) * dose_frac, np.finfo(np.float32).eps)
    gamma2 = np.full(target.shape, np.inf, dtype=np.float32)
    for offset, dist in _gamma_search_offsets_zyx(spacing_zyx_mm, distance_mm):
        eval_sl, ref_sl = _gamma_offset_shifts(target.shape, offset)
        dose_term = (pred[eval_sl] - target[ref_sl]) / denom[eval_sl]
        values = dose_term * dose_term + (dist / distance_mm) ** 2
        gamma2[eval_sl] = np.minimum(gamma2[eval_sl], values)
    return float(np.mean((gamma2[mask] <= 1.0)))


def ct_gamma_pass_rate(
    pred: np.ndarray,
    target: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    *,
    gamma_options: dict[str, Any] | None = None,
    crop_to_high_dose_bbox: bool = False,
    bbox_margin_vox: int = 2,
    random_subset: int | None = None,
    use_fast_local: bool = False,
) -> float:
    """γ pass rate in patient CT coordinates (default 2 % / 3 mm, ref voxels ≥ 10 % max).

    Speed options (combine for large CT volumes):

    * ``crop_to_high_dose_bbox=True`` — restrict to the bounding box of the ≥10 % dose
      region (often 10–50× fewer voxels than the full CT grid).
    * ``random_subset=N`` — pymedphys only evaluates *N* random reference voxels
      (approximate pass rate; good for monitoring during training).
    * ``use_fast_local=True`` — discrete-shift local γ (no pymedphys; ~10–100× faster,
      no sub-voxel interpolation).
    * In ``gamma_options``, lower ``interp_fraction`` (e.g. 5 instead of 10) trades
      accuracy for speed when using pymedphys.
    """
    if use_fast_local:
        pred_w = np.asarray(pred, dtype=np.float64)
        target_w = np.asarray(target, dtype=np.float64)
        if crop_to_high_dose_bbox:
            mask = high_dose_region_mask(target_w, 0.10)
            pred_w, target_w, _ = crop_volumes_to_mask_bbox(
                pred_w, target_w, mask, margin_vox=bbox_margin_vox
            )
        return ct_gamma_pass_rate_fast_local(pred_w, target_w, spacing_zyx_mm)

    import pymedphys

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"expected 3-D volumes (z, y, x), got ndim={pred.ndim}")

    origin_zyx = (0, 0, 0)
    if crop_to_high_dose_bbox:
        mask = high_dose_region_mask(target, 0.10)
        pred, target, origin_zyx = crop_volumes_to_mask_bbox(
            pred, target, mask, margin_vox=bbox_margin_vox
        )

    opts = dict(DEFAULT_CT_GAMMA_2MM_3PCT)
    if gamma_options is not None:
        opts.update(gamma_options)
    if random_subset is not None:
        opts["random_subset"] = int(random_subset)

    axes = ct_spacing_axes_zyx_with_origin(pred.shape, spacing_zyx_mm, origin_zyx)
    gamma = pymedphys.gamma(axes, target, axes, pred, **opts)
    valid = gamma[~np.isnan(gamma)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid <= 1.0))


def high_dose_mse(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.10,
) -> float:
    """Mean squared error inside the high-dose region (≥ threshold × max reference)."""
    mask = high_dose_region_mask(target, threshold)
    if not np.any(mask):
        return float("nan")
    diff = pred[mask] - target[mask]
    return float(np.mean(diff * diff))


def high_dose_mae(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.10,
) -> float:
    """Mean absolute error inside the high-dose region (physical units, not normalised)."""
    mask = high_dose_region_mask(target, threshold)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - target[mask])))


# ---------------------------------------------------------------------------
# Individual metric functions  (operate on numpy arrays, shape (D, H, W))
# ---------------------------------------------------------------------------

def masked_mae(pred: np.ndarray, target: np.ndarray, threshold: float = 0.10) -> float:
    """Masked MAE inside the high-dose region, normalised by the GT maximum.

    The high-dose region is defined as voxels where target ≥ threshold * max(target).
    Returns NaN if the target is all-zero.
    """
    max_gt = float(target.max())
    if max_gt <= 0:
        return float("nan")
    mask = target >= threshold * max_gt
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - target[mask])) / max_gt)


def idd_curve_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Normalised RMS difference between predicted and GT IDD curves.

    The IDD curve is the sum of dose over the transverse plane at each
    depth slice (axis 0 = beam depth in BEV convention).

    Returns NaN if the GT IDD peak is zero.
    """
    idd_gt   = target.sum(axis=(1, 2))   # shape (D,)
    idd_pred = pred.sum(axis=(1, 2))

    peak_gt = float(idd_gt.max())
    if peak_gt <= 0:
        return float("nan")

    rms = float(np.sqrt(np.mean((idd_pred - idd_gt) ** 2)))
    return rms / peak_gt


def idd_curve_distance_floored(
    pred: np.ndarray, target: np.ndarray, cutoff_frac: float = 0.005
) -> float:
    """Like :func:`idd_curve_distance`, but floors both volumes first.

    Matches the submission-scoring convention for beams with no real
    ``minimum_cutoff`` to read (``scripts/postprocess_eval.py``): both
    prediction and target are zeroed below ``cutoff_frac`` of the target's
    own max, symmetrically, before the IDD curves are formed.
    """
    max_gt = float(target.max())
    if max_gt <= 0:
        return float("nan")
    thr = cutoff_frac * max_gt
    target_f = np.where(target < thr, 0.0, target)
    pred_f = np.where(pred < thr, 0.0, pred)
    return idd_curve_distance(pred_f, target_f)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over the full volume."""
    return float(np.mean((pred - target) ** 2))


def _squeeze_bev_batch(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """``(B, 1, D, H, W)`` or ``(B, D, H, W)`` -> ``(B, D, H, W)``."""
    if pred.ndim == 5 and target.ndim == 5 and pred.shape[1] == 1 and target.shape[1] == 1:
        return pred.squeeze(1), target.squeeze(1)
    return pred, target


def masked_mae_torch(
    pred: Tensor,
    target: Tensor,
    threshold: float = 0.10,
    *,
    reduction: Literal["mean", "none"] = "none",
) -> Tensor:
    """GPU masked MAE matching :func:`masked_mae` (per batch item when ``reduction='none'``)."""
    pred, target = _squeeze_bev_batch(pred, target)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if pred.ndim != 4:
        raise ValueError(f"expected (B, D, H, W), got {pred.shape}")

    max_gt = target.amax(dim=(1, 2, 3))
    expand = max_gt.view(-1, 1, 1, 1)
    mask = (target >= threshold * expand) & (max_gt > 0).view(-1, 1, 1, 1)

    abs_err = (pred - target).abs()
    sum_e = (abs_err * mask.to(dtype=abs_err.dtype)).flatten(1).sum(1)
    cnt = mask.flatten(1).sum(1).to(dtype=pred.dtype)
    valid = (max_gt > 0) & (cnt > 0)
    per = torch.full((pred.shape[0],), float("nan"), device=pred.device, dtype=pred.dtype)
    vals = (sum_e[valid] / cnt[valid]) / max_gt[valid]
    per[valid] = vals.to(dtype=per.dtype)

    if reduction == "mean":
        if not valid.any():
            return pred.sum() * 0.0
        return per[valid].mean()
    return per


def idd_curve_distance_torch(
    pred: Tensor,
    target: Tensor,
    *,
    reduction: Literal["mean", "none"] = "none",
    eps: float = 1e-12,
) -> Tensor:
    """GPU IDD distance matching :func:`idd_curve_distance` (RMS / GT peak)."""
    pred, target = _squeeze_bev_batch(pred, target)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if pred.ndim != 4:
        raise ValueError(f"expected (B, D, H, W), got {pred.shape}")

    idd_pred = pred.sum(dim=(2, 3))
    idd_gt = target.sum(dim=(2, 3))
    peak_gt = idd_gt.amax(dim=1)
    valid = peak_gt > 0
    denom = peak_gt.clamp_min(eps)
    mse_1d = (idd_pred - idd_gt).pow(2).mean(dim=1)
    per = torch.full((pred.shape[0],), float("nan"), device=pred.device, dtype=pred.dtype)
    per[valid] = (torch.sqrt(mse_1d[valid] + eps) / denom[valid]).to(dtype=per.dtype)

    if reduction == "mean":
        if not valid.any():
            return pred.sum() * 0.0
        return per[valid].mean()
    return per


def idd_curve_distance_floored_torch(
    pred: Tensor,
    target: Tensor,
    *,
    cutoff_frac: float = 0.005,
    reduction: Literal["mean", "none"] = "none",
    eps: float = 1e-12,
) -> Tensor:
    """GPU version of :func:`idd_curve_distance_floored`, per batch item."""
    pred, target = _squeeze_bev_batch(pred, target)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if pred.ndim != 4:
        raise ValueError(f"expected (B, D, H, W), got {pred.shape}")

    max_gt_vol = target.amax(dim=(1, 2, 3))
    thr = (cutoff_frac * max_gt_vol).view(-1, 1, 1, 1)
    target_f = torch.where(target < thr, torch.zeros_like(target), target)
    pred_f = torch.where(pred < thr, torch.zeros_like(pred), pred)
    return idd_curve_distance_torch(pred_f, target_f, reduction=reduction, eps=eps)


def mse_torch(
    pred: Tensor,
    target: Tensor,
    *,
    reduction: Literal["mean", "none"] = "none",
) -> Tensor:
    """GPU MSE matching :func:`mse` (per batch item when ``reduction='none'``)."""
    pred, target = _squeeze_bev_batch(pred, target)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    per = (pred - target).pow(2).mean(dim=(1, 2, 3))
    if reduction == "mean":
        return per.mean()
    return per


def _tensor_metric_value(value: Tensor) -> float:
    v = float(value.item())
    return v if v == v else float("nan")


def gradient_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """MAE of the 3-D gradient magnitude between prediction and target.

    Uses central differences along each spatial axis.
    """
    def grad_magnitude(vol: np.ndarray) -> np.ndarray:
        gd = np.gradient(vol, axis=0)
        gh = np.gradient(vol, axis=1)
        gw = np.gradient(vol, axis=2)
        return np.sqrt(gd ** 2 + gh ** 2 + gw ** 2)

    return float(np.mean(np.abs(grad_magnitude(pred) - grad_magnitude(target))))


# ---------------------------------------------------------------------------
# Per-batch metric accumulator
# ---------------------------------------------------------------------------

class ValidationMetrics:
    """Accumulate per-beam metric values across the entire validation set.

    Dose units
    ----------
    * ``dose_scale=None`` (default): metrics on **normalized** model I/O (training loss scale).
    * ``dose_scale=<float>``: multiply pred/target by ``dose_scale * dose_scale_extra`` before
      metrics — **physical** dose (matches historical ``run_model_eval`` catalog parquets when
      ``dose_scale`` is ``stats['dose_scale']``).
    """

    METRIC_NAMES = (
        "masked_mae",
        "idd_distance",
        "idd_distance_floored",
        "mse",
        "gradient_mae",
        "gamma_pass_rate",
        "high_dose_mse",
        "high_dose_mae",
    )

    def __init__(
        self,
        *,
        compute_gamma: bool = False,
        gamma_options: dict[str, Any] | None = None,
        spacing_mm: tuple[float, float, float] = DEFAULT_BEV_SPACING_MM,
        dose_scale: float | None = None,
        dose_scale_extra: float = 1.0,
        high_dose_threshold: float = 0.10,
        compute_high_dose_mse_mae: bool = False,
        gamma_use_fast_local: bool = False,
        gamma_crop_to_high_dose_bbox: bool = False,
        gamma_random_subset: int | None = None,
        gamma_profile: Literal["training", "clinical"] = "training",
        metrics_mode: Literal["fast", "full", "gamma-only"] = "fast",
        use_gpu_metrics: bool = True,
        compute_gradient_mae: bool = True,
        idd_floor_frac: float = 0.005,
    ) -> None:
        self._records: list[dict] = []
        self.metrics_mode = metrics_mode
        if metrics_mode == "full":
            compute_gamma = True
        elif metrics_mode == "gamma-only":
            compute_gamma = True
        else:
            compute_gamma = False
        self.compute_gamma = compute_gamma
        self._gamma_only = metrics_mode == "gamma-only"
        self._compute_fast_metrics = metrics_mode != "gamma-only"
        self.use_gpu_metrics = bool(use_gpu_metrics)
        self.compute_gradient_mae = bool(compute_gradient_mae)
        self.gamma_options = gamma_options
        self.spacing_mm = spacing_mm
        self.dose_scale = dose_scale
        self.dose_scale_extra = float(dose_scale_extra)
        self.high_dose_threshold = float(high_dose_threshold)
        self.compute_high_dose_mse_mae = compute_high_dose_mse_mae
        self.gamma_use_fast_local = gamma_use_fast_local
        self.gamma_crop_to_high_dose_bbox = gamma_crop_to_high_dose_bbox
        self.gamma_random_subset = gamma_random_subset
        self.gamma_profile = gamma_profile
        self.idd_floor_frac = float(idd_floor_frac)

    @property
    def dose_units(self) -> str:
        return "physical" if self.dose_scale is not None else "normalized"

    def _scale_dose_tensors(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if self.dose_scale is None:
            return pred, target
        scale = float(self.dose_scale) * self.dose_scale_extra
        return pred * scale, target * scale

    def _scale_dose_arrays(
        self, pred: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.dose_scale is None:
            return pred, target
        scale = float(self.dose_scale) * self.dose_scale_extra
        return pred * scale, target * scale

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        patient_id: str,
        beam_id: int,
        cp_id: int,
        *,
        gantry_angle_deg: float | None = None,
        cohort: str | None = None,
    ) -> None:
        """Compute all metrics for a single beam and store the result."""
        pred_m, target_m = self._scale_dose_arrays(
            np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
        )
        record: dict[str, Any] = {
            "patient_id": patient_id,
            "beam_id": int(beam_id),
            "cp_id": int(cp_id),
            "cohort": cohort if cohort is not None else patient_cohort(patient_id),
            "gantry_deg": _gantry_bin(gantry_angle_deg),
            "dose_units": self.dose_units,
        }
        if self._compute_fast_metrics:
            record["masked_mae"] = masked_mae(pred_m, target_m)
            record["idd_distance"] = idd_curve_distance(pred_m, target_m)
            record["idd_distance_floored"] = idd_curve_distance_floored(
                pred_m, target_m, self.idd_floor_frac
            )
            record["mse"] = mse(pred_m, target_m)
            record["gradient_mae"] = (
                gradient_mae(pred_m, target_m) if self.compute_gradient_mae else float("nan")
            )
        else:
            record["masked_mae"] = float("nan")
            record["idd_distance"] = float("nan")
            record["idd_distance_floored"] = float("nan")
            record["mse"] = float("nan")
            record["gradient_mae"] = float("nan")
        if self.compute_gamma:
            record["gamma_pass_rate"] = bev_gamma_pass_rate(
                pred_m,
                target_m,
                spacing_mm=self.spacing_mm,
                gamma_options=self.gamma_options,
                crop_to_high_dose_bbox=self.gamma_crop_to_high_dose_bbox,
                random_subset=self.gamma_random_subset,
                use_fast_local=self.gamma_use_fast_local,
                gamma_profile=self.gamma_profile,
            )
        else:
            record["gamma_pass_rate"] = float("nan")
        if self.compute_high_dose_mse_mae:
            record["high_dose_mse"] = high_dose_mse(
                pred_m, target_m, threshold=self.high_dose_threshold
            )
            record["high_dose_mae"] = high_dose_mae(
                pred_m, target_m, threshold=self.high_dose_threshold
            )
        else:
            record["high_dose_mse"] = float("nan")
            record["high_dose_mae"] = float("nan")
        self._records.append(record)

    def _gantry_for_batch_item(self, gantry_batch: Any, index: int) -> float | None:
        if gantry_batch is None:
            return None
        if torch.is_tensor(gantry_batch):
            return float(gantry_batch[index].item())
        return float(gantry_batch[index])

    def _update_batch_cpu(
        self,
        pred_batch: Tensor,
        target_batch: Tensor,
        batch: dict,
    ) -> None:
        pred_np = pred_batch.detach().cpu().float().numpy()
        target_np = target_batch.detach().cpu().float().numpy()
        gantry_batch = batch.get("gantry_angle_deg")
        b_count = pred_np.shape[0]
        for b in range(b_count):
            p = pred_np[b, 0]
            t = target_np[b, 0]
            self.update(
                p,
                t,
                patient_id=batch["patient_id"][b],
                beam_id=int(batch["beam_id"][b]),
                cp_id=int(batch["cp_id"][b]),
                gantry_angle_deg=self._gantry_for_batch_item(gantry_batch, b),
            )

    def _update_batch_gpu(
        self,
        pred_batch: Tensor,
        target_batch: Tensor,
        batch: dict,
    ) -> None:
        pred = pred_batch.detach().float()
        target = target_batch.detach().float()
        pred_m, target_m = self._scale_dose_tensors(pred, target)

        mmae_b = masked_mae_torch(pred_m, target_m, self.high_dose_threshold, reduction="none")
        idd_b = idd_curve_distance_torch(pred_m, target_m, reduction="none")
        idd_floored_b = idd_curve_distance_floored_torch(
            pred_m, target_m, cutoff_frac=self.idd_floor_frac, reduction="none"
        )
        mse_b = mse_torch(pred_m, target_m, reduction="none")

        need_cpu_volumes = (
            self.compute_gradient_mae
            or self.compute_gamma
            or self.compute_high_dose_mse_mae
        )
        pred_np = target_np = None
        if need_cpu_volumes:
            pred4, target4 = _squeeze_bev_batch(pred_m, target_m)
            pred_np = pred4.detach().cpu().float().numpy()
            target_np = target4.detach().cpu().float().numpy()

        mmae_cpu = mmae_b.detach().cpu()
        idd_cpu = idd_b.detach().cpu()
        idd_floored_cpu = idd_floored_b.detach().cpu()
        mse_cpu = mse_b.detach().cpu()
        gantry_batch = batch.get("gantry_angle_deg")
        b_count = int(pred_batch.shape[0])

        for b in range(b_count):
            patient_id = batch["patient_id"][b]
            record: dict[str, Any] = {
                "patient_id": patient_id,
                "beam_id": int(batch["beam_id"][b]),
                "cp_id": int(batch["cp_id"][b]),
                "cohort": patient_cohort(patient_id),
                "gantry_deg": _gantry_bin(self._gantry_for_batch_item(gantry_batch, b)),
                "dose_units": self.dose_units,
            }
            if self._compute_fast_metrics:
                record["masked_mae"] = _tensor_metric_value(mmae_cpu[b])
                record["idd_distance"] = _tensor_metric_value(idd_cpu[b])
                record["idd_distance_floored"] = _tensor_metric_value(idd_floored_cpu[b])
                record["mse"] = _tensor_metric_value(mse_cpu[b])
                if self.compute_gradient_mae and pred_np is not None:
                    record["gradient_mae"] = gradient_mae(pred_np[b], target_np[b])
                else:
                    record["gradient_mae"] = float("nan")
            else:
                record["masked_mae"] = float("nan")
                record["idd_distance"] = float("nan")
                record["idd_distance_floored"] = float("nan")
                record["mse"] = float("nan")
                record["gradient_mae"] = float("nan")
            if self.compute_gamma and pred_np is not None:
                record["gamma_pass_rate"] = bev_gamma_pass_rate(
                    pred_np[b],
                    target_np[b],
                    spacing_mm=self.spacing_mm,
                    gamma_options=self.gamma_options,
                    crop_to_high_dose_bbox=self.gamma_crop_to_high_dose_bbox,
                    random_subset=self.gamma_random_subset,
                    use_fast_local=self.gamma_use_fast_local,
                    gamma_profile=self.gamma_profile,
                )
            else:
                record["gamma_pass_rate"] = float("nan")
            if self.compute_high_dose_mse_mae and pred_np is not None:
                record["high_dose_mse"] = high_dose_mse(
                    pred_np[b], target_np[b], threshold=self.high_dose_threshold
                )
                record["high_dose_mae"] = high_dose_mae(
                    pred_np[b], target_np[b], threshold=self.high_dose_threshold
                )
            else:
                record["high_dose_mse"] = float("nan")
                record["high_dose_mae"] = float("nan")
            self._records.append(record)

    def update_batch(self, pred_batch: Tensor, target_batch: Tensor, batch: dict) -> None:
        """Process a whole DataLoader batch (tensors on any device).

        pred_batch / target_batch shape: (B, 1, D, H, W)
        batch dict must contain 'patient_id', 'beam_id', 'cp_id'.
        Optional: 'gantry_angle_deg' (tensor or list).

        When ``use_gpu_metrics`` is enabled and tensors are on CUDA, MSE / masked MAE /
        IDD are computed on GPU; only per-beam scalars are copied to CPU. NumPy is still
        used for gradient MAE, gamma, and high-dose metrics when those are enabled.
        """
        if (
            self.use_gpu_metrics
            and self._compute_fast_metrics
            and pred_batch.is_cuda
            and target_batch.is_cuda
        ):
            self._update_batch_gpu(pred_batch, target_batch, batch)
            return
        self._update_batch_cpu(pred_batch, target_batch, batch)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return mean ± std for each metric across all beams."""
        df = self.dataframe()
        result = {}
        for name in self.METRIC_NAMES:
            col = df[name].dropna()
            result[name] = {"mean": float(col.mean()), "std": float(col.std())}
        return result

    def reset(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _central_slice(vol: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the central 2-D slice along *axis*."""
    idx = vol.shape[axis] // 2
    return np.take(vol, idx, axis=axis)


def plot_beam_comparisons(
    pred_list:   Sequence[np.ndarray],
    target_list: Sequence[np.ndarray],
    meta_list:   Sequence[dict],
    save_path:   Path,
    n_cols:      int = 5,
    cmap:        str = "inferno",
) -> None:
    """2-row × n_cols subplot: top = GT central slice, bottom = prediction.

    Parameters
    ----------
    pred_list, target_list:
        Each element is a numpy array of shape (D, H, W) – the BEV dose volume.
    meta_list:
        List of dicts with keys 'patient_id', 'beam_id', 'cp_id'.
    save_path:
        File path (.png) to write.
    n_cols:
        Number of beams to display side-by-side (default 5).
    """
    n = min(n_cols, len(pred_list))
    fig = plt.figure(figsize=(3.5 * n, 7))
    fig.suptitle("Validation: GT vs Predicted Dose (central BEV slice, axis 0)",
                 fontsize=12, y=1.01)

    gs = gridspec.GridSpec(
        2, n,
        figure=fig,
        hspace=0.35,
        wspace=0.12,
        left=0.04, right=0.96,
        top=0.92, bottom=0.06,
    )

    for col in range(n):
        target = target_list[col]
        pred   = pred_list[col]
        meta   = meta_list[col]

        # Use a shared colour scale based on the GT maximum
        vmax = float(target.max()) if target.max() > 0 else 1.0
        vmin = 0.0

        slice_gt   = _central_slice(target, axis=0)
        slice_pred = _central_slice(pred,   axis=0)

        # ---- Ground truth row ----
        ax_gt = fig.add_subplot(gs[0, col])
        im = ax_gt.imshow(slice_gt, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect="auto", interpolation="nearest")
        ax_gt.axis("off")
        if col == 0:
            ax_gt.set_ylabel("Ground Truth", fontsize=9)
            ax_gt.yaxis.set_visible(True)
        title = (f"P{meta['patient_id']}\n"
                 f"B{meta['beam_id']} CP{meta['cp_id']}")
        ax_gt.set_title(title, fontsize=8)

        # ---- Prediction row ----
        ax_pr = fig.add_subplot(gs[1, col])
        ax_pr.imshow(slice_pred, cmap=cmap, vmin=vmin, vmax=vmax,
                     aspect="auto", interpolation="nearest")
        ax_pr.axis("off")
        if col == 0:
            ax_pr.set_ylabel("Prediction", fontsize=9)
            ax_pr.yaxis.set_visible(True)

        # Per-beam MAE annotation
        beam_mae = masked_mae(pred, target)
        label = f"MAE={beam_mae:.3f}" if not np.isnan(beam_mae) else "MAE=N/A"
        ax_pr.set_title(label, fontsize=8)

        # Shared colourbar per column
        cbar_ax = fig.add_axes([
            gs[1, col].get_position(fig).x0,
            gs[1, col].get_position(fig).y0 - 0.045,
            gs[1, col].get_position(fig).width,
            0.018,
        ])
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar_ax.tick_params(labelsize=7)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_idd_curves(
    pred_list:   Sequence[np.ndarray],
    target_list: Sequence[np.ndarray],
    meta_list:   Sequence[dict],
    save_path:   Path,
    n_cols:      int = 5,
) -> None:
    """Overlay predicted vs GT IDD curves for each selected beam."""
    n = min(n_cols, len(pred_list))
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Integrated Depth-Dose Curves (GT vs Predicted)", fontsize=11)

    for col, ax in enumerate(axes[:n]):
        target = target_list[col]
        pred   = pred_list[col]
        meta   = meta_list[col]

        idd_gt   = target.sum(axis=(1, 2))
        idd_pred = pred.sum(axis=(1, 2))
        depth    = np.arange(len(idd_gt))

        ax.plot(depth, idd_gt,   label="GT",   color="#2E86AB", lw=1.5)
        ax.plot(depth, idd_pred, label="Pred", color="#E84855", lw=1.5, ls="--")
        ax.set_xlabel("Depth slice (BEV axis 0)", fontsize=8)
        ax.set_ylabel("Summed dose", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

        dist = idd_curve_distance(pred, target)
        label = f"IDD dist={dist:.3f}" if not np.isnan(dist) else "IDD dist=N/A"
        ax.set_title(
            f"P{meta['patient_id']} B{meta['beam_id']} CP{meta['cp_id']}\n{label}",
            fontsize=8,
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level validation runner
# ---------------------------------------------------------------------------

def run_validation(
    model:       torch.nn.Module,
    val_loader,
    device:      torch.device,
    output_dir:  Path,
    epoch:       int,
    stats:       dict,
    n_vis:       int = 5,
    seed:        int = 0,
) -> dict[str, float]:
    """Run a full validation pass, compute metrics, and save outputs.

    Returns a flat dict of mean metric values for logging, e.g.
    {"masked_mae": 0.032, "idd_distance": 0.015, "mse": 0.0004, "gradient_mae": 0.002}.
    """
    if val_loader is None:
        return {}

    val_dir = output_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    dose_scale = float(stats["dose_scale"])
    metrics    = ValidationMetrics()
    model.eval()

    # Buffers for visualisation (collect the first n_vis beams deterministically)
    vis_preds:   list[np.ndarray] = []
    vis_targets: list[np.ndarray] = []
    vis_meta:    list[dict]       = []
    all_preds:   list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_metas:   list[dict]       = []

    with torch.no_grad():
        # add tqdm progress bar for validation batches
        for batch in tqdm(
            val_loader, desc="Validation", unit="batch", position=0, leave=True, dynamic_ncols=False
        ):
            ct     = batch["ct"].to(device)
            pb     = batch["pb_dose"].to(device)
            target = batch["target"].to(device)

            pred = model(ct, pb)
            metrics.update_batch(pred, target, batch)

            # Rescale to physical units for visualisation
            pred_np   = (pred.cpu().float().numpy()   * dose_scale)  # (B,1,D,H,W)
            target_np = (target.cpu().float().numpy() * dose_scale)

            B = pred_np.shape[0]
            for b in range(B):
                all_preds.append(pred_np[b, 0])
                all_targets.append(target_np[b, 0])
                all_metas.append({
                    "patient_id": batch["patient_id"][b],
                    "beam_id":    int(batch["beam_id"][b]),
                    "cp_id":      int(batch["cp_id"][b]),
                })

    # Randomly select n_vis beams for visualisation
    rng = random.Random(seed + epoch)
    indices = list(range(len(all_preds)))
    selected = rng.sample(indices, min(n_vis, len(indices)))
    for i in selected:
        vis_preds.append(all_preds[i])
        vis_targets.append(all_targets[i])
        vis_meta.append(all_metas[i])

    # ---- Dose slice comparison ----
    plot_beam_comparisons(
        vis_preds, vis_targets, vis_meta,
        save_path=val_dir / f"dose_slices_epoch{epoch:04d}.png",
        n_cols=n_vis,
    )

    # ---- IDD curves ----
    plot_idd_curves(
        vis_preds, vis_targets, vis_meta,
        save_path=val_dir / f"idd_curves_epoch{epoch:04d}.png",
        n_cols=n_vis,
    )

    # ---- Metrics table ----
    df = metrics.dataframe()
    df.to_csv(val_dir / f"metrics_epoch{epoch:04d}.csv", index=False)

    # Running summary CSV (one row per epoch)
    summary = metrics.summary()
    summary_row = {"epoch": epoch}
    for metric_name, vals in summary.items():
        summary_row[f"{metric_name}_mean"] = round(vals["mean"], 6)
        summary_row[f"{metric_name}_std"]  = round(vals["std"],  6)

    summary_path = val_dir / "metrics_summary.csv"
    summary_df   = pd.DataFrame([summary_row])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)

    # Print to console
    print(
        f"  [val epoch {epoch:03d}] "
        + "  ".join(
            f"{k}: {v['mean']:.4g} ± {v['std']:.4g}"
            for k, v in summary.items()
        )
    )

    # Return flat dict of means for the caller
    return {k: v["mean"] for k, v in summary.items()}
