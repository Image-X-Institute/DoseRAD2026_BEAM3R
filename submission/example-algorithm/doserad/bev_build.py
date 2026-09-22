"""BEV cuboid construction and the inverse BEV-to-CT affine.

Extracted from ``DL-segment-dose-calculation/inference/pipeline.py``, keeping
only what the ``otf_gpu`` + ``cubic`` + z-aligned + Triton configuration
touches. Dropped from the original: the non-Triton back-projection context,
Geant4 ``.mac`` parsing and ``mac_cache`` file I/O, DoseRAD plan/dataset
enumeration, BEV depth-range trimming, forward dose accumulation, and the CLI.

Geometry always comes from an in-memory ``mac_cache`` record built by
``doserad.geometry.mac_cache_entry``, so the no-cache fallbacks are gone too.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cupy as cp
import cupyx.scipy.ndimage as cpndi
import numpy as np
import SimpleITK as sitk

from .bev_grid import BevGridConfig, cached_bev_offset


@dataclass
class AffineBackCtx:
    affine: cp.ndarray  # (3, 4), maps local CT ROI (x, y, z, 1) to BEV indices
    roi_shape: Tuple[int, int, int]
    roi_box: Tuple[int, int, int, int, int, int]


# ------------------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------------------

def load_ct_volume_xy_z(
    ct_path: str,
    mode: str = "cubic",
) -> Tuple[cp.ndarray, Tuple[int, int, int], Tuple[float, float, float], Tuple[float, float, float], sitk.Image]:
    """Load a CT ``.mha`` into the (X, Y, Z) CuPy layout the pipeline uses.

    Returns the cubic spline coefficients (or raw volume for linear mode), the
    volume shape, spacing, origin, and the SimpleITK image for output geometry.
    """
    return prepare_ct_volume_xy_z(sitk.ReadImage(str(ct_path)), mode)


def prepare_ct_volume_xy_z(
    img: sitk.Image,
    mode: str = "cubic",
) -> Tuple[cp.ndarray, Tuple[int, int, int], Tuple[float, float, float], Tuple[float, float, float], sitk.Image]:
    """Same as :func:`load_ct_volume_xy_z`, from an image already in memory.

    The ``photon-mri`` task's CT is generated from the MRI rather than read off
    disk, and writing it out only to read it back would cost a couple of
    hundred MB of I/O per patient inside a timed invoke.
    """
    arr = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0)).astype(np.float32, copy=False)
    ct_vol = cp.asarray(arr, cp.float32)
    if mode == "cubic":
        ct_coeff = cpndi.spline_filter(ct_vol, order=3, mode="mirror").astype(
            cp.float32, copy=False
        )
    else:
        ct_coeff = ct_vol
    shape = tuple(int(v) for v in ct_vol.shape)
    spacing = tuple(float(v) for v in img.GetSpacing())
    origin = tuple(float(v) for v in img.GetOrigin())
    return ct_coeff, shape, spacing, origin, img


def apply_ct_stats_to_bev(bev_ct: cp.ndarray, stats: Dict[str, Any]) -> cp.ndarray:
    """Match the training CT normalisation on a BEV CT volume."""
    ct_min = cp.float32(stats["ct_min"])
    ct_max = cp.float32(stats["ct_max"])
    out = cp.clip(bev_ct, ct_min, ct_max)
    return ((out - ct_min) / (ct_max - ct_min)).astype(cp.float32, copy=False)


# ------------------------------------------------------------------------------
# BEV index grid and cuboid bounds
# ------------------------------------------------------------------------------

def build_bev_index_grid(grid: BevGridConfig) -> cp.ndarray:
    """Flat (N, 3) grid of BEV voxel centres in mm, relative to the plane origin."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    x = cp.arange(nx, dtype=cp.float32)
    y = cp.arange(ny, dtype=cp.float32)
    z = cp.arange(nz, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
    off = cp.asarray([0.0, -(ny // 2) + 0.5, -(nz // 2) + 0.5], cp.float32)
    scale = cp.asarray(grid.spacing_dhw, cp.float32)
    g_lin = (cp.stack((X, Y, Z), -1) + off) * scale
    del X, Y, Z
    cp.cuda.Stream.null.synchronize()
    return g_lin.reshape(-1, 3)


def cuboid_corners_world(
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    U: cp.ndarray, plane_origin: cp.ndarray, off: cp.ndarray,
) -> cp.ndarray:
    """The 8 world-space corners of the BEV cuboid."""
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
# z alignment
# ------------------------------------------------------------------------------

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
    """Patient-z translation that snaps a reference BEV z-plane to CT slice centres."""
    uz = cp.asnumpy(U[2]).astype(np.float64)
    z_dot = float(abs(uz[2]))
    if z_dot < (1.0 - float(orient_tol)):
        print(
            f"[warn] z-slice alignment skipped for {seg_name}: "
            f"|uz.z_hat|={z_dot:.6f} < {1.0 - float(orient_tol):.6f}"
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
    """Whether the basis and spacing can support integer-z alignment."""
    uz = cp.asnumpy(U[2]).astype(np.float64)
    if abs(float(uz[2])) < (1.0 - float(orient_tol)):
        return False
    return abs(float(dz_mm) - float(ct_spacing_z)) <= float(spacing_tol_mm)


def _integer_z_alignment_error(
    coords: cp.ndarray, nx: int, ny: int, nz: int
) -> tuple[float, float]:
    """Within-plane and integer-slice errors for an affine BEV grid."""
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


# ------------------------------------------------------------------------------
# Segment geometry
# ------------------------------------------------------------------------------

def segment_geometry(
    seg_name: str,
    mac_cache: Dict[str, Dict[str, List[float]]],
    NX: int,
    NY: int,
    NZ: int,
    *,
    spacing_dhw: Tuple[float, float, float],
    ct_origin: Optional[Tuple[float, float, float]] = None,
    ct_spacing: Optional[Tuple[float, float, float]] = None,
    align_bev_z_to_ct_slices: bool = False,
    align_orient_tol: float = 1e-4,
    align_spacing_tol_mm: float = 1e-6,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Return beam basis ``U`` (3x3), BEV plane origin, and index offset."""
    rec = mac_cache[seg_name]
    U = cp.asarray(np.asarray(rec["U"], dtype=np.float32).reshape(3, 3), cp.float32)
    plane_origin = cp.asarray(np.asarray(rec["src"], dtype=np.float32), cp.float32)
    off = cp.asarray(
        np.asarray(cached_bev_offset(rec, NX, NY, NZ), dtype=np.float32), cp.float32
    )
    if align_bev_z_to_ct_slices and ct_origin is not None and ct_spacing is not None:
        z_align_mm = _compute_bev_z_align_mm(
            U=U,
            plane_origin=plane_origin,
            off=off,
            nz=NZ,
            dz_mm=float(spacing_dhw[2]),
            ct_origin=ct_origin,
            ct_spacing=ct_spacing,
            orient_tol=align_orient_tol,
            spacing_tol_mm=align_spacing_tol_mm,
            seg_name=seg_name,
        )
        plane_origin = plane_origin.copy()
        plane_origin[2] = plane_origin[2] + cp.float32(z_align_mm)
    return U, plane_origin, off


def build_back_projection_affine_ctx(
    seg_name: str,
    mac_cache: Dict[str, Dict[str, List[float]]],
    ct_shape: Tuple[int, int, int],
    ct_spacing: Tuple[float, float, float],
    ct_origin: Tuple[float, float, float],
    *,
    NX: int,
    NY: int,
    NZ: int,
    spacing_dhw: Tuple[float, float, float],
    align_bev_z_to_ct_slices: bool = True,
    align_orient_tol: float = 1e-4,
    align_spacing_tol_mm: float = 1e-6,
) -> AffineBackCtx:
    """Compact affine form of the inverse CT-to-BEV transform, plus the CT ROI box."""
    ct_spacing_cp = cp.asarray(ct_spacing, cp.float32)
    ct_origin_cp = cp.asarray(ct_origin, cp.float32)
    U, plane_origin, off = segment_geometry(
        seg_name,
        mac_cache,
        NX,
        NY,
        NZ,
        spacing_dhw=spacing_dhw,
        ct_origin=ct_origin,
        ct_spacing=ct_spacing,
        align_bev_z_to_ct_slices=align_bev_z_to_ct_slices,
        align_orient_tol=align_orient_tol,
        align_spacing_tol_mm=align_spacing_tol_mm,
    )
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


# ------------------------------------------------------------------------------
# CUDA kernels
# ------------------------------------------------------------------------------

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
# BEV resampling
# ------------------------------------------------------------------------------

def _zcollapsed_queries(
    ct_coeff: cp.ndarray, coords: cp.ndarray, nx: int, ny: int, nz: int
):
    """Plane indices and flattened xy queries shared by both backends."""
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
    ct_coeff: cp.ndarray, coords: cp.ndarray, nx: int, ny: int, nz: int, *, cval: float
) -> cp.ndarray:
    """Sample on a z-aligned BEV grid with the CuPy backend.

    Integer-aligned z coordinates reduce tricubic interpolation to a
    ``(1, 4, 1) / 6`` z blend followed by a 16-tap 2D spline. Out-of-range
    planes and query points return ``cval``.
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
    ct_coeff: cp.ndarray, coords: cp.ndarray, nx: int, ny: int, nz: int, *, cval: float
) -> cp.ndarray:
    """Sample on a z-aligned BEV grid with the Triton backend.

    CuPy arrays are shared with PyTorch through DLPack. Bounds checking and
    out-of-volume fill are handled in the sampling kernel.
    """
    import torch as _torch

    from .ct_sample import bicubic_zcollapsed_sample_triton_single

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

    ``use_bicubic_zcollapse=True`` uses the z-collapse fast path. Only set it
    once the caller already knows z_align holds for this ``coords``, since it
    isn't re-checked here.
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
            volume_coeff, coords, order=interp, mode="constant", cval=cval, prefilter=False,
        )
    bev = flat.reshape(nx, ny, nz).astype(cp.float32, copy=False)
    if clip_min is not None:
        bev = cp.maximum(bev, cp.float32(clip_min))
    if clip_max is not None:
        bev = cp.minimum(bev, cp.float32(clip_max))
    return bev


def prepare_bev_input_volumes(
    seg_name: str,
    ct_coeff: cp.ndarray,
    ct_spacing: Tuple[float, float, float],
    ct_origin: Tuple[float, float, float],
    mode: str,
    mac_cache: Dict[str, Dict[str, List[float]]],
    g_lin: cp.ndarray,
    seg_resized: cp.ndarray,
    *,
    grid: BevGridConfig,
    anatomy_cval: float = -1024.0,
    anatomy_clip_min: float | None = -1024.0,
) -> tuple[cp.ndarray, cp.ndarray, Dict[str, float]]:
    """Build the two BEV model inputs for one control point: CT and aperture projection."""
    NX, NY, NZ = grid.nx, grid.ny, grid.nz
    spacing_y, spacing_z = float(grid.spacing_dhw[1]), float(grid.spacing_dhw[2])
    sad_mm = float(grid.sad_mm)
    times: Dict[str, float] = {}

    ct_spacing_cp = cp.asarray(ct_spacing, cp.float32)
    ct_origin_cp = cp.asarray(ct_origin, cp.float32)

    rec = mac_cache[seg_name]
    U = cp.asarray(np.asarray(rec["U"], dtype=np.float32).reshape(3, 3), cp.float32)
    plane_origin = cp.asarray(np.asarray(rec["src"], dtype=np.float32), cp.float32)
    off = cp.asarray(
        np.asarray(cached_bev_offset(rec, NX, NY, NZ), dtype=np.float32), cp.float32
    )
    ray_origin = cp.asarray(np.asarray(rec["s"], dtype=np.float32), cp.float32)
    ux, uy, uz = U[0], U[1], U[2]

    # Physical (pre-z-align) plane origin, kept as the mask anchor for the
    # aperture ray-cast below: snapping the sampling grid to CT slices moves
    # the voxels, not the beam, so the 2-D segment mask must stay anchored at
    # the physical beam geometry while being evaluated at the snapped points.
    plane_origin_seg = plane_origin

    z_align_ok = False
    if grid.align_bev_z_to_ct_slices:
        dz_mm_align = float(grid.spacing_dhw[2])
        z_align_ok = _bev_z_align_ok(
            U=U,
            dz_mm=dz_mm_align,
            ct_spacing_z=float(ct_spacing[2]),
            orient_tol=float(grid.align_orient_tol),
            spacing_tol_mm=float(grid.align_spacing_tol_mm),
        )
        z_align_mm = _compute_bev_z_align_mm(
            U=U,
            plane_origin=plane_origin,
            off=off,
            nz=NZ,
            dz_mm=dz_mm_align,
            ct_origin=ct_origin,
            ct_spacing=ct_spacing,
            orient_tol=float(grid.align_orient_tol),
            spacing_tol_mm=float(grid.align_spacing_tol_mm),
            seg_name=seg_name,
        )
        plane_origin = plane_origin.copy()
        plane_origin[2] = plane_origin[2] + cp.float32(z_align_mm)

    def _coords_for(origin: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        points = g_lin @ U + origin
        return points, cp.stack(
            (
                (points[:, 0] - ct_origin_cp[0]) / ct_spacing_cp[0],
                (points[:, 1] - ct_origin_cp[1]) / ct_spacing_cp[1],
                (points[:, 2] - ct_origin_cp[2]) / ct_spacing_cp[2],
            ),
            0,
        )

    t = time.perf_counter()
    bev_points, coords = _coords_for(plane_origin)
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
            bev_points, coords = _coords_for(plane_origin)
            cp.cuda.Stream.null.synchronize()
    times["prep.coords"] = time.perf_counter() - t

    t = time.perf_counter()
    use_bicubic_zcollapse = (
        grid.bicubic_z_align and mode == "cubic"
        and grid.align_bev_z_to_ct_slices and z_align_ok
    )
    times["prep.use_bicubic_zcollapse"] = 1.0 if use_bicubic_zcollapse else 0.0
    bev_ct = resample_volume_to_bev(
        ct_coeff, coords, mode, NX, NY, NZ,
        cval=float(anatomy_cval),
        clip_min=anatomy_clip_min,
        clip_max=None,
        use_bicubic_zcollapse=use_bicubic_zcollapse,
        bicubic_z_align_backend=grid.bicubic_z_align_backend,
    )
    cp.cuda.Stream.null.synchronize()
    times["prep.ct_map"] = time.perf_counter() - t

    seg_resized = seg_resized.astype(cp.float32, copy=False)
    if seg_resized.shape != (NY, NZ):
        raise ValueError(f"seg_resized shape {seg_resized.shape} != ({NY}, {NZ})")

    aperture_points = _aperture_projection_points(
        bev_points, g_lin, U, plane_origin_seg,
        sample_on_unaligned_grid=grid.aperture_sample_on_unaligned_grid,
    )
    bev_points32 = aperture_points.astype(cp.float32, copy=False)
    ray_h = cp.asnumpy(ray_origin).astype(np.float32)
    plane_h = cp.asnumpy(plane_origin_seg).astype(np.float32)
    ux_h = cp.asnumpy(ux).astype(np.float32)
    uy_h = cp.asnumpy(uy).astype(np.float32)
    uz_h = cp.asnumpy(uz).astype(np.float32)
    off_h = cp.asnumpy(off).astype(np.float32)

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
    return bev_ct, seg_proj, times
