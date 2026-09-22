"""
Shared inference pipeline for BEV-based segment dose prediction.

This module contains all geometry helpers, CUDA kernels, and the main run()
function. Each model-specific script imports from here and supplies only the
model class and output filename.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import cupy as cp
import cupyx.scipy.ndimage as cpndi
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn

from bev_grid_config import BevGridConfig, cached_bev_offset, load_bev_grid_config

# The sampler is imported lazily, but standalone inference still needs the model
# directory on its import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))


# Default BEV cuboid from bev_cuboid_cupy_40x40cm_doserad.py / prepare_bev_inputs.
_DEFAULT_BEV_GRID = BevGridConfig.default()
BEV_NX, BEV_NY, BEV_NZ = _DEFAULT_BEV_GRID.shape_dhw
BEV_SPACING_MM = _DEFAULT_BEV_GRID.spacing_dhw
# Legacy inference scaling when no training stats JSON is supplied.
LEGACY_CT_NORM = 1619.0
LEGACY_DOSE_SCALE = 453.2444152832031 * 10.0


# ------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------

def read_mha(path: str, option: str = "info"):
    img = sitk.ReadImage(path)
    if option == "info":
        return img.GetSpacing(), img.GetOrigin(), img.GetSize(), img
    if option == "array":
        return sitk.GetArrayFromImage(img)
    raise ValueError("option must be 'info' or 'array'")


def write_mha(path: str, arr: np.ndarray, ref_img: sitk.Image) -> None:
    im = sitk.GetImageFromArray(arr)
    im.SetOrigin(ref_img.GetOrigin())
    im.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(im, path)


def load_ct_volume_xy_z(
    ct_path: str,
    mode: str = "cubic",
) -> Tuple[cp.ndarray, Tuple[float, float, float], Tuple[float, float, float], sitk.Image]:
    """Load CT MHA and return volume in (X, Y, Z) CuPy layout used by the pipeline."""
    # Keep the image used for metadata and extract its array directly. Calling
    # read_mha(..., "info") followed by read_mha(..., "array") decoded a
    # compressed MHA twice.
    ref_img = sitk.ReadImage(ct_path)
    ct_spacing = ref_img.GetSpacing()
    ct_origin = ref_img.GetOrigin()
    ct_arr = sitk.GetArrayFromImage(ref_img)
    ct_arr = np.transpose(ct_arr, (2, 1, 0))
    ct_vol = cp.asarray(ct_arr, cp.float32)
    if mode == "cubic":
        ct_coeff = cpndi.spline_filter(ct_vol, order=3, mode="mirror").astype(cp.float32, copy=False)
    else:
        ct_coeff = ct_vol
    return ct_coeff, ct_spacing, ct_origin, ref_img


def load_bev_dose_array(path: str | Path) -> np.ndarray:
    """Load a single BEV dose volume ``(NX, NY, NZ)`` from ``.npy`` or raw float32 binary."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.fromfile(path, dtype=np.float32)
        expected = BEV_NX * BEV_NY * BEV_NZ
        if arr.size != expected:
            raise ValueError(
                f"Raw BEV file {path} has {arr.size} floats, expected {expected} "
                f"for shape ({BEV_NX}, {BEV_NY}, {BEV_NZ})"
            )
        arr = arr.reshape(BEV_NX, BEV_NY, BEV_NZ)
    if arr.shape != (BEV_NX, BEV_NY, BEV_NZ):
        raise ValueError(f"BEV dose shape {arr.shape} != ({BEV_NX}, {BEV_NY}, {BEV_NZ})")
    return np.asarray(arr, dtype=np.float32)


def apply_ct_stats_to_bev(bev_ct: cp.ndarray, stats: dict[str, Any]) -> cp.ndarray:
    """Match ``MultiModalHDF5Dataset`` CT normalisation on a BEV CT volume."""
    ct_min = cp.float32(stats["ct_min"])
    ct_max = cp.float32(stats["ct_max"])
    out = cp.clip(bev_ct, ct_min, ct_max)
    return ((out - ct_min) / (ct_max - ct_min)).astype(cp.float32, copy=False)


def model_bev_ct_input(bev_ct: cp.ndarray, stats: dict[str, Any] | None) -> cp.ndarray:
    """Convert resampled HU BEV CT to model input units."""
    if stats is not None:
        return apply_ct_stats_to_bev(bev_ct, stats)
    return (bev_ct * (cp.float32(1.0) / cp.float32(LEGACY_CT_NORM))).astype(cp.float32, copy=False)


def model_bev_dose_to_physical(pred_bev: cp.ndarray, stats: dict[str, Any] | None) -> cp.ndarray:
    """Convert model BEV dose output to physical dose before back-projection."""
    pred = cp.maximum(pred_bev.astype(cp.float32, copy=False), 0)
    if stats is not None:
        return pred * cp.float32(float(stats["dose_scale"]))
    return pred * cp.float32(LEGACY_DOSE_SCALE)


@dataclass
class SegmentTask:
    """One control-point segment for BEV inference / back-projection."""

    name: str
    mac_file: str
    segment_path: str
    beam_id: int
    cp_id: int
    mu_weight: float = 1.0


@dataclass
class BackCtx:
    interp: int
    map_idx: cp.ndarray  # shape (3, N) - pre-computed inverse mapping
    roi_shape: Tuple[int, int, int]
    roi_box: Tuple[int, int, int, int, int, int]


@dataclass
class AffineBackCtx:
    affine: cp.ndarray  # (3, 4), maps local CT ROI (x,y,z,1) to BEV indices
    roi_shape: Tuple[int, int, int]
    roi_box: Tuple[int, int, int, int, int, int]


def control_point_mu_weight(cp: dict[str, Any], default: float = 1.0) -> float:
    """Segment MU weight from plan JSON (falls back to 1.0)."""
    for key in ("mu_weight", "weight", "segment_weight", "mu"):
        if key in cp and cp[key] is not None:
            return float(cp[key])
    return default


def list_doserad_segment_tasks(
    plan_json_path: str | Path,
    mac_dir: str | Path,
    seg_dir: str | Path,
    *,
    mu_default: float = 1.0,
) -> List[SegmentTask]:
    """Enumerate segments from a DoseRAD plan JSON + ``calculate_segment_mac`` outputs."""
    plan_json_path = Path(plan_json_path)
    mac_dir = Path(mac_dir)
    seg_dir = Path(seg_dir)
    with plan_json_path.open() as fh:
        plan = json.load(fh)

    tasks: List[SegmentTask] = []
    patient_id = plan_json_path.stem
    for beam in plan.get("beams", []):
        beam_id = int(beam["beam_idx"])
        for cp in beam.get("control_points", []):
            cp_id = int(cp["cp_idx"])
            name = f"{patient_id}_{beam_id}_CP{cp_id:03d}"
            mac_file = mac_dir / f"{name}.mac"
            segment_path = seg_dir / f"{name}.bin"
            if not mac_file.is_file():
                continue
            if not segment_path.is_file():
                continue
            tasks.append(
                SegmentTask(
                    name=name,
                    mac_file=str(mac_file),
                    segment_path=str(segment_path),
                    beam_id=beam_id,
                    cp_id=cp_id,
                    mu_weight=control_point_mu_weight(cp, mu_default),
                )
            )
    if not tasks:
        raise RuntimeError(f"No segment tasks found for plan {plan_json_path}")
    return tasks


def segment_geometry(
    mac_file: str,
    mac_cache: Optional[Dict[str, Dict[str, List[float]]]],
    NX: int,
    NY: int,
    NZ: int,
    *,
    plane_origin_offset_mm: float | None = None,
    spacing_dhw: Tuple[float, float, float] | None = None,
    ct_origin: Tuple[float, float, float] | None = None,
    ct_spacing: Tuple[float, float, float] | None = None,
    align_bev_z_to_ct_slices: bool = False,
    align_orient_tol: float = 1e-4,
    align_spacing_tol_mm: float = 1e-6,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Return beam basis ``U`` (3×3), BEV plane origin, and index offset."""
    seg_name = os.path.basename(mac_file)[:-4]
    if (
        mac_cache
        and seg_name in mac_cache
        and isinstance(mac_cache[seg_name].get("U"), list)
        and isinstance(mac_cache[seg_name].get("src"), list)
    ):
        rec = mac_cache[seg_name]
        U = cp.asarray(np.asarray(rec["U"], dtype=np.float32).reshape(3, 3), cp.float32)
        plane_origin = cp.asarray(np.asarray(rec["src"], dtype=np.float32), cp.float32)
        off = cp.asarray(
            np.asarray(cached_bev_offset(rec, NX, NY, NZ), dtype=np.float32),
            cp.float32,
        )
        if align_bev_z_to_ct_slices and ct_origin is not None and ct_spacing is not None:
            dz_mm = float(spacing_dhw[2]) if spacing_dhw is not None else float(BEV_SPACING_MM[2])
            z_align_mm = _compute_bev_z_align_mm(
                U=U,
                plane_origin=plane_origin,
                off=off,
                nz=NZ,
                dz_mm=dz_mm,
                ct_origin=ct_origin,
                ct_spacing=ct_spacing,
                orient_tol=align_orient_tol,
                spacing_tol_mm=align_spacing_tol_mm,
                seg_name=seg_name,
            )
            plane_origin = plane_origin.copy()
            plane_origin[2] = plane_origin[2] + cp.float32(z_align_mm)
        return U, plane_origin, off

    s_np, dx_np, dy_np = extract_gps(mac_file)
    ray_origin = cp.asarray(s_np, cp.float32)
    dx = cp.asarray(dx_np, cp.float32)
    dy = cp.asarray(dy_np, cp.float32)
    ux, uy, uz = _basis(dx, dy)
    U = cp.stack((ux, uy, uz), 0)
    dx_mm = float(spacing_dhw[0]) if spacing_dhw is not None else float(BEV_SPACING_MM[0])
    offset_mm = (
        float(plane_origin_offset_mm)
        if plane_origin_offset_mm is not None
        else 1000.0 - 0.5 * float(NX) * dx_mm
    )
    plane_origin = ray_origin + cp.float32(offset_mm) * ux
    off = cp.asarray([0.0, -(NY // 2) + 0.5, -(NZ // 2) + 0.5], cp.float32)
    if align_bev_z_to_ct_slices and ct_origin is not None and ct_spacing is not None:
        dz_mm = float(spacing_dhw[2]) if spacing_dhw is not None else float(BEV_SPACING_MM[2])
        z_align_mm = _compute_bev_z_align_mm(
            U=U,
            plane_origin=plane_origin,
            off=off,
            nz=NZ,
            dz_mm=dz_mm,
            ct_origin=ct_origin,
            ct_spacing=ct_spacing,
            orient_tol=align_orient_tol,
            spacing_tol_mm=align_spacing_tol_mm,
            seg_name=seg_name,
        )
        plane_origin = plane_origin.copy()
        plane_origin[2] = plane_origin[2] + cp.float32(z_align_mm)
    return U, plane_origin, off


def _compute_bev_z_align_mm(
    *,
    U: cp.ndarray,
    plane_origin: cp.ndarray,
    off: cp.ndarray,
    nz: int,
    dz_mm: float,
    ct_origin: Tuple[float, float, float],
    ct_spacing: Tuple[float, float, float],
    orient_tol: float,
    spacing_tol_mm: float,
    seg_name: str,
) -> float:
    """Return patient-z translation that snaps a reference BEV z-plane to CT slice centers."""
    uz = cp.asnumpy(U[2]).astype(np.float64)
    z_dot = float(abs(uz[2]))
    if z_dot < (1.0 - float(orient_tol)):
        print(
            f"[warn] z-slice alignment skipped for {seg_name}: "
            f"|uz·z_hat|={z_dot:.6f} < {1.0 - float(orient_tol):.6f}"
        )
        return 0.0
    ct_spacing_z = float(ct_spacing[2])
    if abs(float(dz_mm) - ct_spacing_z) > float(spacing_tol_mm):
        print(
            f"[warn] z-slice alignment skipped for {seg_name}: "
            f"dz={dz_mm:.6f} != ct_spacing_z={ct_spacing_z:.6f} "
            f"(tol={float(spacing_tol_mm):.6f})"
        )
        return 0.0
    kz_ref = int(nz // 2)
    z_local = (float(kz_ref) + float(cp.asnumpy(off)[2])) * float(dz_mm)
    w0 = float(cp.asnumpy(plane_origin)[2]) + z_local * uz[2]
    c0 = float(ct_origin[2])
    target = round((w0 - c0) / ct_spacing_z) * ct_spacing_z + c0
    return float(target - w0)


def _bev_z_align_ok(
    *,
    U: cp.ndarray,
    dz_mm: float,
    ct_spacing_z: float,
    orient_tol: float,
    spacing_tol_mm: float,
) -> bool:
    """Check whether the basis and spacing can support integer-z alignment."""
    uz = cp.asnumpy(U[2]).astype(np.float64)
    if abs(float(uz[2])) < (1.0 - float(orient_tol)):
        return False
    return abs(float(dz_mm) - float(ct_spacing_z)) <= float(spacing_tol_mm)


def _integer_z_alignment_error(
    coords: cp.ndarray, nx: int, ny: int, nz: int
) -> tuple[float, float]:
    """Return within-plane and integer-slice errors for an affine BEV grid."""
    z = coords.reshape(3, nx, ny, nz)[2]
    corners = cp.stack((z[0, 0], z[0, -1], z[-1, 0], z[-1, -1]))
    plane_error = cp.max(cp.abs(corners - corners[0:1]))
    integer_error = cp.max(cp.abs(corners - cp.rint(corners)))
    return float(plane_error.item()), float(integer_error.item())


def _aperture_projection_points(
    bev_points: cp.ndarray,
    grid_linear: cp.ndarray,
    basis: cp.ndarray,
    unaligned_plane_origin: cp.ndarray,
    *,
    sample_on_unaligned_grid: bool,
) -> cp.ndarray:
    """Select the BEV points used to project the segment aperture.

    Unified geometry evaluates the fixed physical aperture at the same
    (possibly z-aligned) voxel centres as the CT channel. Legacy non-z-align
    checkpoints instead expect the aperture tensor generated on the original,
    unsnapped lattice while CT/dose continue to use the snapped lattice.
    """
    if sample_on_unaligned_grid:
        return grid_linear @ basis + unaligned_plane_origin
    return bev_points


# CuPy fallback for the z-aligned 16-tap gather. A single launch covers all
# output planes; the Triton backend below reuses the differentiable model sampler.
BICUBIC_ZPLANE_KERNEL = cp.ElementwiseKernel(
    in_params="raw float32 coeff, float32 xq, float32 yq, int32 kq, "
    "int32 H, int32 W, int32 NZc, float32 cval",
    out_params="float32 out",
    operation=r"""
        if (xq < 0.0f || xq > (float)(H-1) || yq < 0.0f || yq > (float)(W-1)) {
            out = cval; return;
        }
        int x0 = (int)floorf(xq), y0 = (int)floorf(yq);
        float fx = xq - (float)x0, fy = yq - (float)y0;

        float wx[4], wy[4];
        {
            float t = fx, t2 = t*t, t3 = t2*t;
            wx[0] = (1.0f - t)*(1.0f - t)*(1.0f - t) / 6.0f;
            wx[1] = (4.0f - 6.0f*t2 + 3.0f*t3) / 6.0f;
            wx[2] = (1.0f + 3.0f*t + 3.0f*t2 - 3.0f*t3) / 6.0f;
            wx[3] = t3 / 6.0f;
        }
        {
            float t = fy, t2 = t*t, t3 = t2*t;
            wy[0] = (1.0f - t)*(1.0f - t)*(1.0f - t) / 6.0f;
            wy[1] = (4.0f - 6.0f*t2 + 3.0f*t3) / 6.0f;
            wy[2] = (1.0f + 3.0f*t + 3.0f*t2 - 3.0f*t3) / 6.0f;
            wy[3] = t3 / 6.0f;
        }

        float acc = 0.0f;
        for (int a = 0; a < 4; a++) {
            int xi = x0 - 1 + a;
            if (H > 1) {
                int period = 2 * (H - 1);
                int m = xi % period; if (m < 0) m += period;
                xi = (m > H - 1) ? (period - m) : m;
            } else { xi = 0; }
            for (int b = 0; b < 4; b++) {
                int yi = y0 - 1 + b;
                if (W > 1) {
                    int period = 2 * (W - 1);
                    int m = yi % period; if (m < 0) m += period;
                    yi = (m > W - 1) ? (period - m) : m;
                } else { yi = 0; }
                acc += wx[a] * wy[b] * coeff[(xi * W + yi) * NZc + kq];
            }
        }
        out = acc;
    """,
    name="bicubic_zplane_gather",
)


def _zcollapsed_queries(
    ct_coeff: cp.ndarray, coords: cp.ndarray, nx: int, ny: int, nz: int
) -> tuple[int, int, int, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Prepare plane indices and flattened xy queries shared by both backends."""
    xdim, ydim, zdim = (int(size) for size in ct_coeff.shape)
    coords4 = coords.reshape(3, nx, ny, nz)
    z0 = cp.round(coords4[2, 0, 0]).astype(cp.int32)
    xq, yq = coords4[0].reshape(-1), coords4[1].reshape(-1)
    return (
        xdim, ydim, zdim, cp.clip(z0, 0, zdim - 1),
        (z0 >= 0) & (z0 < zdim), xq, yq,
        cp.tile(cp.arange(nz, dtype=cp.int32), nx * ny),
    )


def _bev_ct_bicubic_zcollapsed(
    ct_coeff: cp.ndarray,
    coords: cp.ndarray,
    nx: int,
    ny: int,
    nz: int,
    *,
    cval: float,
) -> cp.ndarray:
    """Sample CT or dose on a z-aligned BEV grid with the CuPy backend.

    Integer-aligned z coordinates reduce tricubic interpolation to a
    `(1, 4, 1) / 6` z blend followed by a 16-tap 2D spline. Out-of-range
    planes and query points return `cval`.
    """
    _, _, zdim, z0_clamped, in_range, xq, yq, kq = _zcollapsed_queries(
        ct_coeff, coords, nx, ny, nz
    )
    # Mirror the neighbouring taps at the first and last CT slices.
    zm1 = cp.where(z0_clamped > 0, z0_clamped - 1, min(1, zdim - 1))
    zp1 = cp.where(z0_clamped < zdim - 1, z0_clamped + 1, max(zdim - 2, 0))
    collapsed = (
        ct_coeff[:, :, zm1]
        + cp.float32(4.0) * ct_coeff[:, :, z0_clamped]
        + ct_coeff[:, :, zp1]
    ) * cp.float32(1.0 / 6.0)
    collapsed = cp.ascontiguousarray(collapsed)

    out = cp.empty(nx * ny * nz, dtype=cp.float32)
    BICUBIC_ZPLANE_KERNEL(
        collapsed, xq, yq, kq,
        np.int32(collapsed.shape[0]), np.int32(collapsed.shape[1]), np.int32(nz),
        np.float32(cval), out,
    )
    if not bool(cp.all(in_range)):
        out = cp.where(
            in_range[cp.newaxis, cp.newaxis, :],
            out.reshape(nx, ny, nz),
            cp.float32(cval),
        ).reshape(-1)
    return out


def _bev_ct_bicubic_zcollapsed_triton(
    ct_coeff: cp.ndarray,
    coords: cp.ndarray,
    nx: int,
    ny: int,
    nz: int,
    *,
    cval: float,
) -> cp.ndarray:
    """Sample CT or dose on a z-aligned BEV grid with the Triton backend.

    CuPy arrays are shared with PyTorch through DLPack. Bounds checking and
    out-of-volume fill are handled in the sampling kernel.
    """
    import torch as _torch
    from patient_space_resample_triton import bicubic_zcollapsed_sample_triton_single

    coords4 = coords.reshape(3, nx, ny, nz)
    z0 = cp.round(coords4[2, 0, 0]).astype(cp.int32)
    xq = coords4[0].reshape(-1)
    yq = coords4[1].reshape(-1)

    coeff_t = _torch.from_dlpack(
        cp.ascontiguousarray(ct_coeff.astype(cp.float32, copy=False))
    )
    with _torch.no_grad():
        out_t = bicubic_zcollapsed_sample_triton_single(
            coeff_t,
            _torch.from_dlpack(cp.ascontiguousarray(xq)),
            _torch.from_dlpack(cp.ascontiguousarray(yq)),
            _torch.from_dlpack(z0),
            cval=cval,
        )

    return cp.from_dlpack(out_t)


def build_back_projection_ctx(
    mac_file: str,
    ct_shape: Tuple[int, int, int],
    ct_spacing: Tuple[float, float, float],
    ct_origin: Tuple[float, float, float],
    *,
    mac_cache: Optional[Dict[str, Dict[str, List[float]]]] = None,
    NX: int = BEV_NX,
    NY: int = BEV_NY,
    NZ: int = BEV_NZ,
    interp: int = 3,
    spacing_dhw: Tuple[float, float, float] | None = None,
    plane_origin_offset_mm: float | None = None,
    align_bev_z_to_ct_slices: bool = False,
    align_orient_tol: float = 1e-4,
    align_spacing_tol_mm: float = 1e-6,
) -> BackCtx:
    """Pre-compute inverse map from patient CT ROI voxels to BEV dose indices."""
    ct_spacing_cp = cp.asarray(ct_spacing, cp.float32)
    ct_origin_cp = cp.asarray(ct_origin, cp.float32)
    U, plane_origin, off = segment_geometry(
        mac_file,
        mac_cache,
        NX,
        NY,
        NZ,
        plane_origin_offset_mm=plane_origin_offset_mm,
        spacing_dhw=spacing_dhw,
        ct_origin=ct_origin,
        ct_spacing=ct_spacing,
        align_bev_z_to_ct_slices=align_bev_z_to_ct_slices,
        align_orient_tol=align_orient_tol,
        align_spacing_tol_mm=align_spacing_tol_mm,
    )

    if spacing_dhw is None:
        spacing_dhw = BEV_SPACING_MM
    dxmm, dymm, dzmm = (cp.float32(spacing_dhw[0]), cp.float32(spacing_dhw[1]), cp.float32(spacing_dhw[2]))
    corners = cuboid_corners_world(NX, NY, NZ, dxmm, dymm, dzmm, U, plane_origin, off)
    idx = (corners - ct_origin_cp) / ct_spacing_cp
    lo = cp.floor(idx.min(0)) - 1.0
    hi = cp.ceil(idx.max(0)) + 1.0
    lo_i, hi_i = clamp_box(lo, hi, ct_shape)

    xi = cp.arange(int(lo_i[0].item()), int(hi_i[0].item()) + 1, dtype=cp.int32)
    yi = cp.arange(int(lo_i[1].item()), int(hi_i[1].item()) + 1, dtype=cp.int32)
    zi = cp.arange(int(lo_i[2].item()), int(hi_i[2].item()) + 1, dtype=cp.int32)
    nxr, nyr, nzr = xi.size, yi.size, zi.size

    inv_scale = cp.asarray([1.0 / dxmm, 1.0 / dymm, 1.0 / dzmm], cp.float32)
    A = U.T * inv_scale[:, None]
    C0 = ((ct_origin_cp - plane_origin) @ A) - off
    Cx = ct_spacing_cp[0] * A[0]
    Cy = ct_spacing_cp[1] * A[1]
    Cz = ct_spacing_cp[2] * A[2]

    xi_f, yi_f, zi_f = xi.astype(cp.float32), yi.astype(cp.float32), zi.astype(cp.float32)
    q = (
        C0[:, None, None, None]
        + Cx[:, None, None, None] * xi_f[None, :, None, None]
        + Cy[:, None, None, None] * yi_f[None, None, :, None]
        + Cz[:, None, None, None] * zi_f[None, None, None, :]
    )
    map_idx = q.reshape(3, -1)
    roi_shape = (nxr, nyr, nzr)
    roi_box = (
        int(lo_i[0]), int(hi_i[0]) + 1,
        int(lo_i[1]), int(hi_i[1]) + 1,
        int(lo_i[2]), int(hi_i[2]) + 1,
    )
    return BackCtx(interp=interp, map_idx=map_idx, roi_shape=roi_shape, roi_box=roi_box)


def build_back_projection_affine_ctx(
    mac_file: str,
    ct_shape: Tuple[int, int, int],
    ct_spacing: Tuple[float, float, float],
    ct_origin: Tuple[float, float, float],
    *,
    mac_cache: Optional[Dict[str, Dict[str, List[float]]]] = None,
    NX: int = BEV_NX,
    NY: int = BEV_NY,
    NZ: int = BEV_NZ,
    spacing_dhw: Tuple[float, float, float] | None = None,
    plane_origin_offset_mm: float | None = None,
    align_bev_z_to_ct_slices: bool = False,
    align_orient_tol: float = 1e-4,
    align_spacing_tol_mm: float = 1e-6,
) -> AffineBackCtx:
    """Build the compact affine form of the inverse CT-to-BEV transform."""
    ct_spacing_cp = cp.asarray(ct_spacing, cp.float32)
    ct_origin_cp = cp.asarray(ct_origin, cp.float32)
    U, plane_origin, off = segment_geometry(
        mac_file,
        mac_cache,
        NX,
        NY,
        NZ,
        plane_origin_offset_mm=plane_origin_offset_mm,
        spacing_dhw=spacing_dhw,
        ct_origin=ct_origin,
        ct_spacing=ct_spacing,
        align_bev_z_to_ct_slices=align_bev_z_to_ct_slices,
        align_orient_tol=align_orient_tol,
        align_spacing_tol_mm=align_spacing_tol_mm,
    )
    if spacing_dhw is None:
        spacing_dhw = BEV_SPACING_MM
    dxmm, dymm, dzmm = (cp.float32(v) for v in spacing_dhw)
    corners = cuboid_corners_world(NX, NY, NZ, dxmm, dymm, dzmm, U, plane_origin, off)
    idx = (corners - ct_origin_cp) / ct_spacing_cp
    lo = cp.floor(idx.min(0)) - 1.0
    hi = cp.ceil(idx.max(0)) + 1.0
    lo_i, hi_i = clamp_box(lo, hi, ct_shape)
    starts = tuple(int(lo_i[i].item()) for i in range(3))
    stops = tuple(int(hi_i[i].item()) + 1 for i in range(3))

    inv_scale = cp.asarray([1.0 / dxmm, 1.0 / dymm, 1.0 / dzmm], cp.float32)
    A = U.T * inv_scale[:, None]
    C0 = ((ct_origin_cp - plane_origin) @ A) - off
    Cx = ct_spacing_cp[0] * A[0]
    Cy = ct_spacing_cp[1] * A[1]
    Cz = ct_spacing_cp[2] * A[2]
    local_c0 = C0 + Cx * starts[0] + Cy * starts[1] + Cz * starts[2]
    affine = cp.stack((Cx, Cy, Cz, local_c0), axis=1).astype(cp.float32, copy=False)
    roi_shape = tuple(stops[i] - starts[i] for i in range(3))
    roi_box = (starts[0], stops[0], starts[1], stops[1], starts[2], stops[2])
    return AffineBackCtx(affine=affine, roi_shape=roi_shape, roi_box=roi_box)


def backproject_bev_dose_to_ct(
    pred_bev: np.ndarray | cp.ndarray,
    ctx: BackCtx,
    dose_sum: cp.ndarray,
    mu_weight: float = 1.0,
) -> float:
    """Accumulate one physical BEV dose cuboid into a patient CT-grid volume (in-place)."""
    if isinstance(pred_bev, np.ndarray):
        pred_bev = cp.asarray(pred_bev, dtype=cp.float32)
    else:
        pred_bev = pred_bev.astype(cp.float32, copy=False)
    return resample_to_ct(pred_bev, ctx, dose_sum, float(mu_weight))


def ct_volume_to_sitk_array(dose_sum_xy_z: cp.ndarray | np.ndarray) -> np.ndarray:
    """Convert internal (X, Y, Z) dose grid to SimpleITK (Z, Y, X) array."""
    arr = cp.asnumpy(dose_sum_xy_z) if isinstance(dose_sum_xy_z, cp.ndarray) else dose_sum_xy_z
    return np.transpose(arr, (2, 1, 0))


def extract_gps(
    mac_file: str,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """Parse a Geant4 MAC file and return (focus_point, direction_x, rot1)."""
    fp = r"/gps/ang/focuspoint\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+mm"
    dx = r"/gps/direction\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"
    r1 = r"/gps/pos/rot1\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"
    t = open(mac_file, "r").read()
    s = tuple(map(float, re.search(fp, t).groups()))
    x = tuple(map(float, re.search(dx, t).groups()))
    y = tuple(map(float, re.search(r1, t).groups()))
    return s, x, y


# ------------------------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------------------------

def _norm_np(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


def _basis_np(dx: Tuple[float, float, float], dy: Tuple[float, float, float]) -> np.ndarray:
    ux  = _norm_np(np.asarray(dx, dtype=np.float32))
    uy0 = _norm_np(np.asarray(dy, dtype=np.float32))
    uz  = _norm_np(np.cross(ux, uy0))
    uy  = _norm_np(np.cross(uz, ux))
    return np.stack([ux, uy, uz], axis=0).astype(np.float32)


def _norm(v: cp.ndarray) -> cp.ndarray:
    return v / cp.linalg.norm(v)


def _basis(dx: cp.ndarray, dy: cp.ndarray):
    ux  = _norm(dx)
    uy0 = _norm(dy)
    uz  = _norm(cp.cross(ux, uy0))
    uy  = _norm(cp.cross(uz, ux))
    return ux.astype(cp.float32), uy.astype(cp.float32), uz.astype(cp.float32)


# ------------------------------------------------------------------------------
# MAC geometry cache
# ------------------------------------------------------------------------------

def build_mac_cache_from_tasks(
    tasks: Sequence[SegmentTask],
    out_dir: str,
    *,
    patient: str = "patient",
    NX: int = BEV_NX,
    NY: int = BEV_NY,
    NZ: int = BEV_NZ,
    grid: BevGridConfig | None = None,
) -> str:
    """Build ``mac_cache.json`` from Geant4 MAC paths (DoseRAD / segment_mac layout)."""
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "mac_cache.json")
    g = grid or BevGridConfig(shape_dhw=(NX, NY, NZ), spacing_dhw=BEV_SPACING_MM)
    crop = float(g.resolved_plane_origin_offset_mm)
    off = [0.0, -(g.ny // 2) + 0.5, -(g.nz // 2) + 0.5]
    segments: Dict[str, Dict[str, List[float]]] = {}
    for task in tasks:
        seg_name = os.path.basename(task.mac_file)[:-4]
        s, dx, dy = extract_gps(task.mac_file)
        U = _basis_np(dx, dy)
        src = np.asarray(s, dtype=np.float32) + crop * U[0]
        segments[seg_name] = {
            "s": list(map(float, s)),
            "dx": list(map(float, dx)),
            "dy": list(map(float, dy)),
            "U": U.reshape(-1).astype(np.float32).tolist(),
            "src": src.astype(np.float32).tolist(),
            "off": [float(v) for v in off],
            "NX": int(g.nx),
            "NY": int(g.ny),
            "NZ": int(g.nz),
            "z_align_mm": 0.0,
        }
    with open(cache_path, "w") as f:
        json.dump(
            {
                "patient": patient,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "bev_grid": g.to_dict(),
                "segments": segments,
            },
            f,
        )
    return cache_path


def build_mac_cache(
    patient: str, sim_root: str, out_dir: str,
    NX: int = BEV_NX, NY: int = BEV_NY, NZ: int = BEV_NZ,
    grid: BevGridConfig | None = None,
) -> str:
    """Pre-compute and cache BEV geometry (rotation matrix, source position) for all segments."""
    dose_dir  = os.path.join(sim_root, patient, "dose_mha")
    setup_dir = os.path.join(sim_root, patient, "setup")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "mac_cache.json")
    if not os.path.isdir(dose_dir):
        raise FileNotFoundError(dose_dir)
    if not os.path.isdir(setup_dir):
        raise FileNotFoundError(setup_dir)

    g = grid or BevGridConfig(shape_dhw=(NX, NY, NZ), spacing_dhw=BEV_SPACING_MM)
    crop = float(g.resolved_plane_origin_offset_mm)
    off  = [0.0, -(g.ny // 2) + 0.5, -(g.nz // 2) + 0.5]

    segments: Dict[str, Dict[str, List[float]]] = {}
    for f in sorted(os.listdir(dose_dir)):
        m = re.search(r"MU(\d+\.?\d*)_G", f)
        if not m:
            continue
        seg      = f[5:-4]
        mac_file = os.path.join(setup_dir, f"{seg}.mac")
        if not os.path.exists(mac_file):
            continue
        s, dx, dy = extract_gps(mac_file)
        U   = _basis_np(dx, dy)
        src = np.asarray(s, dtype=np.float32) + crop * U[0]
        segments[seg] = {
            "s":   list(map(float, s)),
            "dx":  list(map(float, dx)),
            "dy":  list(map(float, dy)),
            "U":   U.reshape(-1).astype(np.float32).tolist(),
            "src": src.astype(np.float32).tolist(),
            "off": [float(v) for v in off],
            "NX":  int(g.nx), "NY": int(g.ny), "NZ": int(g.nz),
            "z_align_mm": 0.0,
        }
    if not segments:
        raise RuntimeError("no segments parsed")
    with open(cache_path, "w") as f:
        json.dump(
            {"patient": patient, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "bev_grid": g.to_dict(),
             "segments": segments},
            f,
        )
    return cache_path


def load_mac_cache(out_dir: str) -> Optional[Dict[str, Dict[str, List[float]]]]:
    p = os.path.join(out_dir, "mac_cache.json")
    if not os.path.exists(p):
        return None
    try:
        obj  = json.load(open(p, "r"))
        segs = obj.get("segments", None)
        return segs if isinstance(segs, dict) and segs else None
    except Exception:
        return None


# ------------------------------------------------------------------------------
# BEV index grid
# ------------------------------------------------------------------------------

def build_bev_index_grid(
    grid_or_nx: BevGridConfig | int | None = None,
    NY: int | None = None,
    NZ: int | None = None,
) -> cp.ndarray:
    """Build a flat (N, 3) grid of BEV voxel centres in mm."""
    if isinstance(grid_or_nx, BevGridConfig):
        nx, ny, nz = grid_or_nx.nx, grid_or_nx.ny, grid_or_nx.nz
        spacing = grid_or_nx.spacing_dhw
    elif isinstance(grid_or_nx, int):
        nx, ny, nz = int(grid_or_nx), int(NY), int(NZ)
        spacing = BEV_SPACING_MM
    else:
        default = BevGridConfig.default()
        nx, ny, nz = default.nx, default.ny, default.nz
        spacing = default.spacing_dhw
    x = cp.arange(nx, dtype=cp.float32)
    y = cp.arange(ny, dtype=cp.float32)
    z = cp.arange(nz, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
    off   = cp.asarray([0.0, -(ny // 2) + 0.5, -(nz // 2) + 0.5], cp.float32)
    scale = cp.asarray(spacing, cp.float32)
    G_lin = (cp.stack((X, Y, Z), -1) + off) * scale
    del X, Y, Z
    cp.cuda.Stream.null.synchronize()
    return G_lin.reshape(-1, 3)


def slice_bev_index_grid(
    g_lin: cp.ndarray,
    i0: int,
    i1: int,
    *,
    nx: int = BEV_NX,
    ny: int = BEV_NY,
    nz: int = BEV_NZ,
) -> cp.ndarray:
    """Contiguous depth subset of ``build_bev_index_grid`` output (no 4D reshape)."""
    if i0 < 0 or i1 >= nx or i0 > i1:
        raise ValueError(f"invalid depth slice [{i0}, {i1}] for nx={nx}")
    n_plane = ny * nz
    row0 = i0 * n_plane
    row1 = (i1 + 1) * n_plane
    return g_lin[row0:row1]


def bev_depth_axis_from_mac_cache(
    mac_cache: Dict[str, Dict[str, List[float]]],
    sample_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(ux, s0)`` unit depth axis and world origin of BEV depth index 0."""
    rec = mac_cache[sample_id]
    u = np.asarray(rec["U"], dtype=np.float32).reshape(3, 3)
    ux = u[0]
    s0 = np.asarray(rec["src"], dtype=np.float32)
    return ux, s0


def _bev_depth_range_from_body_mask(
    body_small: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    ux: np.ndarray,
    s0: np.ndarray,
    *,
    stride: int,
    margin_slices: int,
    nx: int,
    bev_dx: float,
) -> Tuple[int, int]:
    """Map coarse boolean body mask to BEV depth indices via 8-corner AABB projection."""
    xs, ys, zs = np.where(body_small)
    if xs.size == 0:
        return 0, nx - 1

    s = float(stride)
    s1 = float(stride - 1)
    x0, x1 = float(xs.min()) * s, float(xs.max()) * s + s1
    y0, y1 = float(ys.min()) * s, float(ys.max()) * s + s1
    z0, z1 = float(zs.min()) * s, float(zs.max()) * s + s1

    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y1, z0],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ],
        dtype=np.float64,
    ) + 0.5

    orig = np.asarray(origin, dtype=np.float64)
    sp = np.asarray(spacing, dtype=np.float64)
    ux_v = np.asarray(ux, dtype=np.float64).reshape(3)
    s0_v = np.asarray(s0, dtype=np.float64).reshape(3)

    pts = orig + corners * sp
    s_mm = (pts - s0_v) @ ux_v
    s_min = float(s_mm.min())
    s_max = float(s_mm.max())

    i0 = int(np.floor(s_min / bev_dx)) - int(margin_slices)
    i1 = int(np.ceil(s_max / bev_dx)) + int(margin_slices)
    return max(0, i0), min(nx - 1, i1)


def compute_bev_depth_range_from_ct_gpu(
    ct_vol: cp.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    ux: np.ndarray | cp.ndarray,
    s0: np.ndarray | cp.ndarray,
    *,
    hu_thresh: float = -500.0,
    margin_slices: int = 4,
    stride: int = 8,
    nx: int = BEV_NX,
    bev_dx: float = BEV_SPACING_MM[0],
) -> Tuple[int, int]:
    """HU body mask on GPU → project AABB corners on ``ux`` → BEV depth indices.

    Pass **raw HU** ``ct_vol`` (X, Y, Z) on GPU, not spline coefficients. Thresholds on
    a coarse grid (GPU), then ``np.where`` on a small CPU array (fast). Avoids
    full-volume ``body.any(axis=...)`` over millions of voxels.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if ct_vol.ndim != 3:
        raise ValueError(f"ct_vol must be 3D (X,Y,Z), got shape {ct_vol.shape}")

    body_small = cp.asnumpy(
        ct_vol[::stride, ::stride, ::stride] > cp.float32(hu_thresh)
    )
    return _bev_depth_range_from_body_mask(
        body_small,
        spacing,
        origin,
        np.asarray(ux, dtype=np.float32).reshape(3),
        np.asarray(s0, dtype=np.float32).reshape(3),
        stride=stride,
        margin_slices=margin_slices,
        nx=nx,
        bev_dx=bev_dx,
    )


def compute_bev_depth_range_from_ct_mac(
    ct_vol: cp.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    mac_cache: Dict[str, Dict[str, List[float]]],
    sample_id: str,
    **kwargs: Any,
) -> Tuple[int, int]:
    """``compute_bev_depth_range_from_ct_gpu`` with ``ux``/``s0`` from ``mac_cache``."""
    ux, s0 = bev_depth_axis_from_mac_cache(mac_cache, sample_id)
    return compute_bev_depth_range_from_ct_gpu(
        ct_vol, spacing, origin, ux, s0, **kwargs
    )


# ------------------------------------------------------------------------------
# ROI utilities
# ------------------------------------------------------------------------------

def cuboid_corners_world(
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    U: cp.ndarray, plane_origin: cp.ndarray, off: cp.ndarray,
) -> cp.ndarray:
    """Return the 8 world-space corners of the BEV cuboid."""
    c = cp.asarray(
        [[0, 0, 0], [nx-1, 0, 0], [0, ny-1, 0], [0, 0, nz-1],
         [nx-1, ny-1, 0], [nx-1, 0, nz-1], [0, ny-1, nz-1], [nx-1, ny-1, nz-1]],
        cp.float32,
    )
    sc = cp.asarray([dx, dy, dz], cp.float32)
    return plane_origin + ((c + off) * sc) @ U


def clamp_box(lo: cp.ndarray, hi: cp.ndarray, shape: Tuple[int, int, int]):
    lo = cp.clip(lo, 0, cp.asarray(shape) - 1)
    hi = cp.clip(hi, 0, cp.asarray(shape) - 1)
    return lo.astype(cp.int32), hi.astype(cp.int32)


# ------------------------------------------------------------------------------
# CUDA projection kernel (bilinear interpolation on the 2-D segment map)
# ------------------------------------------------------------------------------

PROJECTION_KERNEL = cp.ElementwiseKernel(
    in_params=(
        "raw float32 P, raw float32 seg, "
        "float32 sx, float32 sy, float32 sz, "
        "float32 iso, "
        "float32 nx, float32 ny, float32 nz, "
        "float32 ox, float32 oy, float32 oz, "
        "float32 pyx, float32 pyy, float32 pyz, "
        "float32 pzx, float32 pzy, float32 pzz, "
        "float32 dy, float32 dz, "
        "float32 offy, float32 offz, "
        "int32 Ny, int32 Nz, float32 eps"
    ),
    out_params="float32 out",
    operation=r'''
        float px=P[i*3+0], py=P[i*3+1], pz=P[i*3+2];
        float rx=px-sx, ry=py-sy, rz=pz-sz;
        float dp=rx*nx+ry*ny+rz*nz; if (abs(dp)<eps){out=0;return;}
        float t=iso/dp; if (t<=0){out=0;return;}
        float wx=sx+t*rx, wy=sy+t*ry, wz=sz+t*rz;
        float dx_=wx-ox, dy_=wy-oy, dz_=wz-oz;
        float yl=dx_*pyx+dy_*pyy+dz_*pyz;
        float zl=dx_*pzx+dy_*pzy+dz_*pzz;
        float yi=yl/dy - offy, zi=zl/dz - offz;
        if (yi<0||yi>=(float)(Ny-1)||zi<0||zi>=(float)(Nz-1)){out=0;return;}
        int y0=(int)floorf(yi), z0=(int)floorf(zi), y1=y0+1, z1=z0+1;
        float wyf=yi-(float)y0, wzf=zi-(float)z0;
        float v00=seg[y0*Nz+z0], v10=seg[y1*Nz+z0], v01=seg[y0*Nz+z1], v11=seg[y1*Nz+z1];
        float a=(1.0f-wzf)*v00 + wzf*v01;
        float b=(1.0f-wzf)*v10 + wzf*v11;
        out=(1.0f-wyf)*a + wyf*b;
    ''',
    name="proj2D",
)


# ------------------------------------------------------------------------------
# Per-segment BEV preparation + back-projection context
# ------------------------------------------------------------------------------


def prepare_bev_input_volumes(
    patient: str,
    mac_file: str,
    ct_coeff: cp.ndarray,
    ct_shape: Tuple[int, int, int],
    ct_spacing: Tuple[float, float, float],
    ct_origin: Tuple[float, float, float],
    mode: str,
    mac_cache: Optional[Dict[str, Dict[str, List[float]]]],
    NX: int,
    NY: int,
    NZ: int,
    G_lin: cp.ndarray,
    seg_dir: str | None = None,
    *,
    seg_resized: cp.ndarray | None = None,
    return_coords: bool = False,
    times: Dict[str, float] = {},
    grid: BevGridConfig | None = None,
    anatomy_cval: float = -1024.0,
    anatomy_clip_min: float | None = -1024.0,
) -> tuple[cp.ndarray, cp.ndarray, Dict[str, float]] | tuple[cp.ndarray, cp.ndarray, Dict[str, float], cp.ndarray]:
    """
    For one beam segment, build the two BEV input volumes (CT and segment projection).

    This helper intentionally does not build the back-projection context, so it can be
    reused by training/eval loaders that only need model inputs.

    ``anatomy_cval`` / ``anatomy_clip_min`` control out-of-bounds fill and optional floor
    clip after resampling (CT defaults; use 0.0 for already-normalized MRI).
    """
    _ = patient, ct_shape
    # times: Dict[str, float] = {}
    g = grid or BevGridConfig(
        shape_dhw=(NX, NY, NZ),
        spacing_dhw=BEV_SPACING_MM,
    )
    spacing_y, spacing_z = float(g.spacing_dhw[1]), float(g.spacing_dhw[2])
    sad_mm = float(g.sad_mm)
    plane_offset_mm = g.resolved_plane_origin_offset_mm
    seg_native = int(g.segment_native_size)
    seg_zoom_y, seg_zoom_z = g.segment_zoom_yz

    ct_spacing = cp.asarray(ct_spacing, cp.float32)
    ct_origin  = cp.asarray(ct_origin,  cp.float32)

    seg_name  = os.path.basename(mac_file)[:-4]
    has_cache = bool(
        mac_cache
        and seg_name in mac_cache
        and isinstance(mac_cache[seg_name].get("U"),   list)
        and isinstance(mac_cache[seg_name].get("src"), list)
    )

    if has_cache:
        rec          = mac_cache[seg_name]
        U            = cp.asarray(np.asarray(rec["U"], dtype=np.float32).reshape(3, 3), cp.float32)
        plane_origin = cp.asarray(np.asarray(rec["src"], dtype=np.float32), cp.float32)
        off          = cp.asarray(
            np.asarray(cached_bev_offset(rec, NX, NY, NZ), dtype=np.float32),
            cp.float32,
        )
        ray_origin  = cp.asarray(np.asarray(rec["s"], dtype=np.float32), cp.float32)
        ux, uy, uz  = U[0], U[1], U[2]
    else:
        s_np, dx_np, dy_np = extract_gps(mac_file)
        ray_origin   = cp.asarray(s_np,  cp.float32)
        dx           = cp.asarray(dx_np, cp.float32)
        dy           = cp.asarray(dy_np, cp.float32)
        ux, uy, uz   = _basis(dx, dy)
        U            = cp.stack((ux, uy, uz), 0)
        plane_origin = ray_origin + cp.float32(plane_offset_mm) * ux
        off          = cp.asarray([0.0, -(NY // 2) + 0.5, -(NZ // 2) + 0.5], cp.float32)

    # Physical (pre-z-align) plane origin, kept as the mask anchor for the
    # aperture ray-cast below: snapping the sampling grid to CT slices moves
    # the voxels, not the beam, so the 2-D segment mask must stay anchored at
    # the physical beam geometry while being evaluated at the snapped points.
    plane_origin_seg = plane_origin

    z_align_ok = False
    if g.align_bev_z_to_ct_slices:
        dz_mm_align = float(g.spacing_dhw[2])
        z_align_ok = _bev_z_align_ok(
            U=U,
            dz_mm=dz_mm_align,
            ct_spacing_z=float(ct_spacing[2]),
            orient_tol=float(g.align_orient_tol),
            spacing_tol_mm=float(g.align_spacing_tol_mm),
        )
        z_align_mm = _compute_bev_z_align_mm(
            U=U,
            plane_origin=plane_origin,
            off=off,
            nz=NZ,
            dz_mm=dz_mm_align,
            ct_origin=(float(ct_origin[0]), float(ct_origin[1]), float(ct_origin[2])),
            ct_spacing=(float(ct_spacing[0]), float(ct_spacing[1]), float(ct_spacing[2])),
            orient_tol=float(g.align_orient_tol),
            spacing_tol_mm=float(g.align_spacing_tol_mm),
            seg_name=seg_name,
        )
        plane_origin = plane_origin.copy()
        plane_origin[2] = plane_origin[2] + cp.float32(z_align_mm)

    # World-space coordinates for every BEV voxel centre
    t = time.perf_counter()
    bev_points = G_lin @ U + plane_origin
    cp.cuda.Stream.null.synchronize()
    times["prep.grid"] = time.perf_counter() - t

    # Convert world coordinates to CT voxel indices
    t = time.perf_counter()
    coords = cp.stack(
        (
            (bev_points[:, 0] - ct_origin[0]) / ct_spacing[0],
            (bev_points[:, 1] - ct_origin[1]) / ct_spacing[1],
            (bev_points[:, 2] - ct_origin[2]) / ct_spacing[2],
        ),
        0,
    )
    cp.cuda.Stream.null.synchronize()
    if z_align_ok:
        plane_error, integer_error = _integer_z_alignment_error(coords, NX, NY, NZ)
        z_align_ok = max(plane_error, integer_error) <= 1e-3
        times["prep.z_align_plane_error"] = plane_error
        times["prep.z_align_integer_error"] = integer_error
        if not z_align_ok:
            # Do not retain a physical grid translation when the integer-plane
            # fast path cannot represent the resulting coordinates exactly.
            plane_origin = plane_origin_seg
            bev_points = G_lin @ U + plane_origin
            coords = cp.stack(
                (
                    (bev_points[:, 0] - ct_origin[0]) / ct_spacing[0],
                    (bev_points[:, 1] - ct_origin[1]) / ct_spacing[1],
                    (bev_points[:, 2] - ct_origin[2]) / ct_spacing[2],
                ),
                0,
            )
            cp.cuda.Stream.null.synchronize()
    times["prep.coords"] = time.perf_counter() - t

    # Resample CT/MRI into BEV frame. Stash the z-collapse decision in `times`
    # too, so materialize() can reuse it for the dose target instead of
    # re-deriving z_align_ok itself.
    t = time.perf_counter()
    use_bicubic_zcollapse = (
        g.bicubic_z_align and mode == "cubic" and g.align_bev_z_to_ct_slices and z_align_ok
    )
    times["prep.use_bicubic_zcollapse"] = 1.0 if use_bicubic_zcollapse else 0.0
    bev_ct = resample_volume_to_bev(
        ct_coeff,
        coords,
        mode,
        NX,
        NY,
        NZ,
        cval=float(anatomy_cval),
        clip_min=anatomy_clip_min,
        clip_max=None,
        use_bicubic_zcollapse=use_bicubic_zcollapse,
        bicubic_z_align_backend=g.bicubic_z_align_backend,
    )
    cp.cuda.Stream.null.synchronize()
    times["prep.ct_map"] = time.perf_counter() - t

    # Load and resize the binary segment map unless the caller provides a cached 200x200 mask.
    if seg_resized is None:
        if seg_dir is None:
            raise ValueError("seg_dir is required when seg_resized is not provided")
        seg_path = os.path.join(seg_dir, f"{seg_name}.bin")
        if not os.path.exists(seg_path):
            raise FileNotFoundError(seg_path)
        t = time.perf_counter()
        seg = cp.fromfile(seg_path, dtype=cp.int8).reshape(seg_native, seg_native).astype(
            cp.float32, copy=False
        )
        seg_resized = cpndi.zoom(seg, (seg_zoom_y, seg_zoom_z), order=1).astype(cp.float32, copy=False)
        seg_resized = cp.flip(cp.rot90(seg_resized, 3), axis=0)
        cp.cuda.Stream.null.synchronize()
        times["prep.seg_io_zoom"] = time.perf_counter() - t
    else:
        t = time.perf_counter()
        seg_resized = seg_resized.astype(cp.float32, copy=False)
        if seg_resized.shape != (NY, NZ):
            raise ValueError(f"seg_resized shape {seg_resized.shape} != ({NY}, {NZ})")
        cp.cuda.Stream.null.synchronize()
        times["prep.seg_cache_to_gpu"] = time.perf_counter() - t

    # By default rays pass through the (possibly z-aligned) voxel positions so
    # projection and CT are co-located. The explicit legacy mode retains the
    # exact aperture lattice seen by checkpoints trained without z alignment,
    # while CT/dose and the inverse transform remain z aligned.
    aperture_points = _aperture_projection_points(
        bev_points,
        G_lin,
        U,
        plane_origin_seg,
        sample_on_unaligned_grid=g.aperture_sample_on_unaligned_grid,
    )
    bev_points32 = aperture_points.astype(cp.float32, copy=False)
    ray_h   = cp.asnumpy(ray_origin).astype(np.float32)
    plane_h = cp.asnumpy(plane_origin_seg).astype(np.float32)
    ux_h    = cp.asnumpy(ux).astype(np.float32)
    uy_h    = cp.asnumpy(uy).astype(np.float32)
    uz_h    = cp.asnumpy(uz).astype(np.float32)
    off_h   = cp.asnumpy(off).astype(np.float32)

    proj_flat = cp.zeros(bev_points32.shape[0], cp.float32)
    t = time.perf_counter()
    PROJECTION_KERNEL(
        bev_points32, seg_resized,
        *ray_h, np.float32(sad_mm),
        *ux_h, *plane_h,
        *uy_h, *uz_h,
        np.float32(spacing_y), np.float32(spacing_z),
        off_h[1], off_h[2],
        int(seg_resized.shape[0]), int(seg_resized.shape[1]),
        np.float32(1e-9),
        proj_flat,
    )
    cp.cuda.Stream.null.synchronize()
    times["prep.projection"] = time.perf_counter() - t

    seg_proj = proj_flat.reshape(NX, NY, NZ).astype(cp.float32, copy=False)
    if return_coords:
        return bev_ct, seg_proj, times, coords
    return bev_ct, seg_proj, times


def prepare_patient_volume_coeff_for_bev(
    volume: cp.ndarray,
    mode: str,
    *,
    cval: float = 0.0,
) -> cp.ndarray:
    """Prepare a patient-space volume for BEV ``map_coordinates`` resampling.

    For cubic mode, applies ``spline_filter`` with mirror boundaries. CuPy's
    ``spline_filter`` does not accept ``cval``; out-of-bounds fill is handled
    in ``resample_volume_to_bev`` via ``map_coordinates(..., cval=cval)``.
    """
    _ = cval
    vol = volume.astype(cp.float32, copy=False)
    if mode == "cubic":
        return cpndi.spline_filter(vol, order=3, mode="mirror").astype(
            cp.float32, copy=False
        )
    return vol


def resample_volume_to_bev(
    volume_coeff: cp.ndarray,
    coords: cp.ndarray,
    mode: str,
    nx: int,
    ny: int,
    nz: int,
    *,
    cval: float = 0.0,
    clip_min: float | None = 0.0,
    clip_max: float | None = None,
    use_bicubic_zcollapse: bool = False,
    bicubic_z_align_backend: str = "triton",
) -> cp.ndarray:
    """Resample a patient-space volume onto the BEV grid defined by ``coords``.

    ``use_bicubic_zcollapse=True`` uses the same z-collapse fast path as the CT
    channel. Only set it once the caller already knows z_align holds for this
    ``coords``, since it isn't re-checked here. Defaults to the plain generic path.
    """
    if use_bicubic_zcollapse and mode == "cubic":
        sampler = (
            _bev_ct_bicubic_zcollapsed_triton
            if bicubic_z_align_backend == "triton"
            else _bev_ct_bicubic_zcollapsed
        )
        try:
            flat = sampler(volume_coeff, coords, nx, ny, nz, cval=float(cval))
        except ImportError:
            if sampler is not _bev_ct_bicubic_zcollapsed_triton:
                raise
            flat = _bev_ct_bicubic_zcollapsed(volume_coeff, coords, nx, ny, nz, cval=float(cval))
    else:
        interp = 3 if mode == "cubic" else (1 if mode == "linear" else 0)
        flat = cpndi.map_coordinates(
            volume_coeff,
            coords,
            order=interp,
            mode="constant",
            cval=cval,
            prefilter=False,
        )
    bev = flat.reshape(nx, ny, nz).astype(cp.float32, copy=False)
    if clip_min is not None:
        bev = cp.maximum(bev, cp.float32(clip_min))
    if clip_max is not None:
        bev = cp.minimum(bev, cp.float32(clip_max))
    return bev


def prepare_bev_inputs(
    patient: str,
    mac_file: str,
    ct_coeff: cp.ndarray,
    ct_shape: Tuple[int, int, int],
    ct_spacing: Tuple[float, float, float],
    ct_origin: Tuple[float, float, float],
    mode: str,
    mac_cache: Optional[Dict[str, Dict[str, List[float]]]],
    NX: int,
    NY: int,
    NZ: int,
    G_lin: cp.ndarray,
    seg_dir: str,
    grid: BevGridConfig | None = None,
):
    """
    Build BEV CT/projection inputs and the inverse mapping needed to back-project
    predicted dose into patient coordinates.
    """
    bev_ct, seg_proj, times = prepare_bev_input_volumes(
        patient,
        mac_file,
        ct_coeff,
        ct_shape,
        ct_spacing,
        ct_origin,
        mode,
        mac_cache,
        NX,
        NY,
        NZ,
        G_lin,
        seg_dir=seg_dir,
        grid=grid,
    )

    ct_spacing = cp.asarray(ct_spacing, cp.float32)
    ct_origin = cp.asarray(ct_origin, cp.float32)
    interp = 3 if mode == "cubic" else (1 if mode == "linear" else 0)

    t = time.perf_counter()
    g = grid or BevGridConfig(shape_dhw=(NX, NY, NZ), spacing_dhw=BEV_SPACING_MM)
    ctx = build_back_projection_ctx(
        mac_file,
        ct_shape,
        (float(ct_spacing[0]), float(ct_spacing[1]), float(ct_spacing[2])),
        (float(ct_origin[0]), float(ct_origin[1]), float(ct_origin[2])),
        mac_cache=mac_cache,
        NX=NX,
        NY=NY,
        NZ=NZ,
        interp=interp,
        spacing_dhw=g.spacing_dhw,
        plane_origin_offset_mm=g.resolved_plane_origin_offset_mm,
        align_bev_z_to_ct_slices=g.align_bev_z_to_ct_slices,
        align_orient_tol=g.align_orient_tol,
        align_spacing_tol_mm=g.align_spacing_tol_mm,
    )
    cp.cuda.Stream.null.synchronize()
    times["prep.roi_q"] = time.perf_counter() - t
    return bev_ct, seg_proj, ctx, times


def resample_to_ct(pred_bev: cp.ndarray, ctx: BackCtx, dose_sum: cp.ndarray, MU: float) -> float:
    """Back-project a BEV dose prediction into patient space and accumulate MU-weighted dose."""
    t    = time.perf_counter()
    flat = cpndi.map_coordinates(
        pred_bev, ctx.map_idx,
        order=ctx.interp, mode="constant", cval=0.0,
        prefilter=True if ctx.interp > 1 else False,
    )
    roi = cp.maximum(flat.reshape(ctx.roi_shape), 0).astype(cp.float32, copy=False)
    xs, xe, ys, ye, zs, ze = ctx.roi_box
    dose_sum[xs:xe, ys:ye, zs:ze] += roi * cp.float32(MU)
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - t


# ------------------------------------------------------------------------------
# Shared argparser
# ------------------------------------------------------------------------------

def build_argparser(model_name: str) -> argparse.ArgumentParser:
    """Return a pre-configured ArgumentParser for segment-dose inference."""
    parser = argparse.ArgumentParser(
        description=f"Run {model_name} segment-dose inference on a single patient."
    )
    parser.add_argument(
        "--patient", required=True,
        help="Patient ID (e.g. P016_S).",
    )
    parser.add_argument(
        "--sim-root", required=True,
        help="Simulation data root; must contain {patient}/dose_mha/ and {patient}/setup/.",
    )
    parser.add_argument(
        "--seg-dir", required=True,
        help="Directory containing binary segment files ({seg_name}.bin, int8, shape 400x400).",
    )
    parser.add_argument(
        "--ct-root", required=True,
        help="CT root; must contain {patient}/maskedCT_3mm_shifted.mha.",
    )
    parser.add_argument(
        "--model-weights", required=True,
        help="Path to the model weights (.pth file).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for the predicted dose MHA (default: ./output/{patient}).",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="CUDA device string (default: cuda:0).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of segments per inference batch (default: 8).",
    )
    parser.add_argument(
        "--mode", choices=["linear", "cubic"], default="cubic",
        help="Spatial interpolation mode for CT resampling (default: cubic).",
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="Use FP32 instead of FP16 for inference.",
    )
    parser.add_argument(
        "--no-tf32", action="store_true",
        help="Disable TF32 acceleration (default: TF32 enabled).",
    )
    parser.add_argument(
        "--print-details", action="store_true",
        help="Print a per-step timing breakdown after inference.",
    )
    return parser


# ------------------------------------------------------------------------------
# Main inference function
# ------------------------------------------------------------------------------

def run(
    model_cls: Type[nn.Module],
    out_filename: str,
    patient: str,
    sim_root: str,
    seg_dir: str,
    ct_root: str,
    out_dir: str,
    model_weights: str,
    device: str = "cuda:0",
    batch_size: int = 8,
    mode: str = "cubic",
    allow_tf32: bool = True,
    use_fp16: bool = True,
    print_details: bool = False,  # set True to print per-step timing breakdown
    bev_grid: BevGridConfig | None = None,
) -> Tuple[float, float, float]:
    """
    Run segment-dose inference for one patient using the supplied model class.

    Args:
        model_cls:     PyTorch model class (must accept zero constructor arguments).
        out_filename:  Name of the output MHA file written to out_dir.
        patient:       Patient ID string.
        sim_root:      Root of Monte Carlo simulation data.
        seg_dir:       Directory of binary segment files.
        ct_root:       Root of masked CT volumes.
        out_dir:       Directory where the predicted dose MHA is saved.
        model_weights: Path to the .pth weights file.
        device:        CUDA device string.
        batch_size:    Segments processed per forward pass.
        mode:          Interpolation mode ('cubic' or 'linear').
        allow_tf32:    Enable TF32 for matmul and cuDNN.
        use_fp16:      Run inference in FP16 (AMP + static tensors).
        print_details: Print per-step wall-clock timings.

    Returns:
        (infer_time, preprocess_time, postprocess_time) in seconds.
    """
    g = bev_grid or BevGridConfig.default()
    NX, NY, NZ = g.nx, g.ny, g.nz
    os.makedirs(out_dir, exist_ok=True)

    dev = torch.device(device)
    torch.cuda.set_device(dev.index or 0)
    cp.cuda.Device(dev.index or 0).use()

    torch.backends.cudnn.benchmark         = True
    torch.backends.cuda.matmul.allow_tf32  = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32        = bool(allow_tf32)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    def sync_all():
        torch.cuda.synchronize()
        cp.cuda.Stream.null.synchronize()

    # ===== Initialization (excluded from the three main timing blocks) =====
    init_t0 = time.perf_counter()

    model     = model_cls().to(device).eval()
    amp_dtype = torch.float16 if use_fp16 else None
    state     = torch.load(model_weights, map_location=device)
    model.load_state_dict(state)

    ct_path = os.path.join(ct_root, patient, "maskedCT_3mm_shifted.mha")
    ct_coeff, ct_spacing, ct_origin, ref_img = load_ct_volume_xy_z(
        ct_path, mode=mode
    )

    dose_dir  = os.path.join(sim_root, patient, "dose_mha")
    setup_dir = os.path.join(sim_root, patient, "setup")
    tasks: List[Tuple[str, float]] = []
    for f in sorted(os.listdir(dose_dir)):
        m = re.search(r"MU(\d+\.?\d*)_G", f)
        if not m:
            continue
        mac      = f[5:-4]
        mac_file = os.path.join(setup_dir, f"{mac}.mac")
        if os.path.exists(mac_file):
            tasks.append((mac_file, float(m.group(1))))
    assert tasks, "no segments found"

    mac_cache = load_mac_cache(out_dir)

    def _ok(rec: dict) -> bool:
        return (
            isinstance(rec.get("U"),   list)
            and isinstance(rec.get("src"), list)
            and isinstance(rec.get("off"), list)
        )

    if (mac_cache is None) or (mac_cache and not all(_ok(rec) for rec in mac_cache.values())):
        print("[info] rebuilding mac_cache.json")
        build_mac_cache(patient, sim_root, out_dir, NX=NX, NY=NY, NZ=NZ, grid=g)
        mac_cache = load_mac_cache(out_dir) or {}

    G_lin = build_bev_index_grid(g)

    CT_NORM    = cp.float32(1619.0)
    DOSE_SCALE = cp.float32(453.2444152832031 * 10.0)

    dose_sum = cp.zeros_like(ct_vol, cp.float32)

    dtype_t     = torch.float16 if use_fp16 else torch.float32
    static_ct   = torch.empty((batch_size, NX, NY, NZ), device=device, dtype=dtype_t)
    static_proj = torch.empty((batch_size, NX, NY, NZ), device=device, dtype=dtype_t)
    static_out  = torch.empty((batch_size, NX, NY, NZ), device=device, dtype=dtype_t)

    graph = torch.cuda.CUDAGraph()

    def autocast_if(dtype):
        return torch.amp.autocast("cuda", dtype=dtype) if dtype else contextlib.nullcontext()

    # Warmup + CUDA Graph capture (counted as overhead, not inference time)
    WARMUP = 6
    sync_all()
    with torch.inference_mode():
        for _ in range(WARMUP):
            static_ct.zero_()
            static_proj.zero_()
            with autocast_if(amp_dtype):
                y = model(static_ct, static_proj)
            static_out.copy_(y)
    sync_all()

    with torch.inference_mode():
        sync_all()
        with torch.cuda.graph(graph):
            with autocast_if(amp_dtype):
                _y = model(static_ct, static_proj)
            static_out.copy_(_y)
    sync_all()

    overhead_time = time.perf_counter() - init_t0

    # ===== Accumulators for the three main timing blocks =====
    TOTAL  = defaultdict(float)  # pre / infer / post
    DETAIL = defaultdict(float)  # optional per-step details

    # ===== Main loop (accumulates the three main blocks only) =====
    loop_start = time.perf_counter()
    i = 0
    while i < len(tasks):
        batch = tasks[i : i + batch_size]

        # ---------- Preprocessing: BEV resampling, coord mapping, segment projection, copy to static tensors ----------
        t_pre0 = time.perf_counter()

        bev_list:     List[cp.ndarray] = []
        segproj_list: List[cp.ndarray] = []
        ctx_list:     List[BackCtx]    = []
        mu_list:      List[float]      = []

        t_prep_calls  = 0.0
        t_copy_static = 0.0

        # 1) Generate BEV inputs for each segment in the batch
        for mac_file, MU in batch:
            t1 = time.perf_counter()
            bev_ct, seg_proj, bctx, times = prepare_bev_inputs(
                patient, mac_file, ct_coeff, ct_vol.shape, ct_spacing, ct_origin,
                mode, mac_cache=mac_cache, NX=NX, NY=NY, NZ=NZ, G_lin=G_lin,
                seg_dir=seg_dir,
                grid=g,
            )
            t_prep_calls += time.perf_counter() - t1

            bev_norm = (bev_ct * (cp.float32(1.0) / CT_NORM)).astype(cp.float32, copy=False)
            if dtype_t == torch.float16:
                bev_norm = bev_norm.astype(cp.float16, copy=False)
                seg_proj = seg_proj.astype(cp.float16, copy=False)

            bev_list.append(bev_norm)
            segproj_list.append(seg_proj)
            ctx_list.append(bctx)
            mu_list.append(MU)

            if print_details and isinstance(times, dict):
                for k, v in times.items():
                    DETAIL[f"pre.{k}"] += v

        # 2) Copy to static tensors (still counted as preprocessing)
        t2 = time.perf_counter()
        with torch.inference_mode():
            for bi in range(len(batch)):
                static_ct[bi].copy_(torch.from_dlpack(bev_list[bi]),      non_blocking=True)
                static_proj[bi].copy_(torch.from_dlpack(segproj_list[bi]), non_blocking=True)
            for bi in range(len(batch), batch_size):
                static_ct[bi].zero_()
                static_proj[bi].zero_()
        sync_all()
        t_copy_static += time.perf_counter() - t2

        TOTAL["pre"] += time.perf_counter() - t_pre0
        if print_details:
            DETAIL["pre.prepare_inputs"] += t_prep_calls
            DETAIL["pre.copy_to_static"] += t_copy_static

        # ---------- Inference: CUDA graph replay ----------
        t_inf0 = time.perf_counter()
        with torch.inference_mode():
            graph.replay()
        sync_all()
        TOTAL["infer"] += time.perf_counter() - t_inf0

        # ---------- Postprocessing: scale conversion + inverse resampling to CT + MU-weighted accumulation ----------
        t_post0    = time.perf_counter()
        t_scale    = 0.0
        t_back_map = 0.0

        for bi in range(len(batch)):
            pred_bev = cp.from_dlpack(static_out[bi]).astype(cp.float32, copy=False)

            t_s0     = time.perf_counter()
            pred_bev = cp.maximum(pred_bev * DOSE_SCALE, 0)
            cp.cuda.Stream.null.synchronize()
            t_scale += time.perf_counter() - t_s0

            t_b0   = time.perf_counter()
            # MU-weighted accumulation into dose_sum is handled inside resample_to_ct
            t_back = resample_to_ct(pred_bev, ctx_list[bi], dose_sum, mu_list[bi])
            # use the returned wall-clock from resample_to_ct; fall back to outer measurement if needed
            t_back_map += t_back if isinstance(t_back, (float, int)) else (time.perf_counter() - t_b0)

        TOTAL["post"] += time.perf_counter() - t_post0
        if print_details:
            DETAIL["post.convert_scale"] += t_scale
            DETAIL["post.map_to_ct"]     += t_back_map

        i += batch_size

    sync_all()
    loop_time = time.perf_counter() - loop_start

    # ===== Print timing summary =====
    total_core = TOTAL["pre"] + TOTAL["infer"] + TOTAL["post"]

    def pct(x: float, tot: float) -> float:
        return 100.0 * (x / max(tot, 1e-9))

    print("\n=== Segment-dose pipeline timing (s / %) ===")
    print(f"{'Preprocess':12s}: {TOTAL['pre']:8.3f}  ({pct(TOTAL['pre'],   total_core):5.1f}%)")
    print(f"{'Inference':12s}: {TOTAL['infer']:8.3f}  ({pct(TOTAL['infer'], total_core):5.1f}%)")
    print(f"{'Postprocess':12s}: {TOTAL['post']:8.3f}  ({pct(TOTAL['post'],  total_core):5.1f}%)")
    print(f"{'TOTAL(core)':12s}: {total_core:8.3f}  (100.0%)")
    print(f"{'Overhead*':12s}: {overhead_time:8.3f}  (init + warmup + graph capture)")
    print(f"{'Loop only':12s}: {loop_time:8.3f}  (sum of all batches)\n")

    if print_details and DETAIL:
        print("---- details ----")
        for k in sorted(DETAIL.keys()):
            print(f"{k:24s}: {DETAIL[k]:8.3f}")

    # ===== Write accumulated dose volume to disk =====
    out      = np.transpose(cp.asnumpy(dose_sum), (2, 1, 0))
    out_path = os.path.join(out_dir, out_filename)
    write_mha(out_path, out, ref_img)
    print(f"Wrote: {out_path}")
    return TOTAL["infer"], TOTAL["pre"], TOTAL["post"]
