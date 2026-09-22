"""Beam geometry and MLC aperture, derived directly from plan metadata.

Training generated a Geant4 ``.mac`` macro and a 400x400 ``.bin`` aperture per
control point (``preprocessing/calculate_segment_mac.py``), then parsed the
macro back with a regex (``pipeline.extract_gps``) to recover the beam basis.
Inference skips that round-trip and computes both quantities in memory, but
reproduces the macro's numerics exactly -- including its ``%.2f`` formatting,
which is what training actually consumed.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np

# Fixed machine constants for the generic photon machine (calculate_segment_mac.py).
MLC_LEAF_WIDTH_MM = 5.0
JAW_HALF_MM = 200.0
LEAF_PAIRS = 80
APERTURE_NATIVE = 400
SAD_MM = 1000.0


# ---------------------------------------------------------------------------
# MLC aperture
# ---------------------------------------------------------------------------

def build_aperture(
    mlc_left_mm: Sequence[float],
    mlc_right_mm: Sequence[float],
) -> np.ndarray:
    """Binary 400x400 fluence map from MLC leaf positions, in BEV (ny, nz) layout.

    Combines ``calculate_segment_mac.get_aperture`` (the fluence map), the
    ``flip(aperture.T, axis=0)`` applied when writing the ``.bin``, and the
    ``flip(rot90(seg, 3), axis=0)`` applied when reading it back in the training
    materializer. The two flips compose to a single transpose, but both are
    written out here so each step stays traceable to its origin.
    """
    left = np.asarray(mlc_left_mm, dtype=np.float32)
    right = np.asarray(mlc_right_mm, dtype=np.float32)
    if left.shape != (LEAF_PAIRS,) or right.shape != (LEAF_PAIRS,):
        raise ValueError(
            f"expected {LEAF_PAIRS} leaf positions per bank, got "
            f"{left.shape} / {right.shape}"
        )

    mask_jaw = np.zeros((APERTURE_NATIVE, APERTURE_NATIVE), dtype=np.float32)
    mask_mlc = np.zeros((APERTURE_NATIVE, APERTURE_NATIVE), dtype=np.float32)

    x0 = int(round(-JAW_HALF_MM + 200))
    x1 = int(round(JAW_HALF_MM + 200))
    mask_jaw[max(0, x0):min(400, x1), max(0, x0):min(400, x1)] = 1.0

    for i in range(LEAF_PAIRS):
        col0 = max(0, min(400, int(round(float(left[i]))) + 200))
        col1 = max(0, min(400, int(round(float(right[i]))) + 200))
        if col1 > col0:
            mask_mlc[
                col0:col1,
                int(i * MLC_LEAF_WIDTH_MM):int((i + 1) * MLC_LEAF_WIDTH_MM),
            ] = 1.0

    aperture = mask_mlc * mask_jaw
    on_disk = np.flip(aperture.T, axis=0)                  # generate_segment()
    return np.flip(np.rot90(on_disk, 3), axis=0).astype(np.float32).copy()


def resize_aperture(aperture: np.ndarray, ny: int, nz: int) -> np.ndarray:
    """Bilinear resize of the native aperture onto the BEV (ny, nz) lattice.

    Reproduces ``scipy.ndimage.zoom(..., order=1)`` as used in training, which
    with the default ``grid_mode=False`` maps output index ``i`` to input
    coordinate ``i * (n_src - 1) / (n_dst - 1)`` -- endpoint-aligned, not
    pixel-area. ``tests/test_aperture_resize.py`` pins this against scipy.
    """
    src_y, src_z = aperture.shape
    if (ny, nz) == (src_y, src_z):
        return aperture.astype(np.float32, copy=False)

    def _axis(n_src: int, n_dst: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_dst == 1:
            pos = np.zeros(1, dtype=np.float64)
        else:
            pos = np.arange(n_dst, dtype=np.float64) * ((n_src - 1) / (n_dst - 1))
        pos = np.clip(pos, 0.0, n_src - 1.0)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_src - 1)
        return i0, i1, (pos - i0).astype(np.float32)

    y0, y1, wy = _axis(src_y, ny)
    z0, z1, wz = _axis(src_z, nz)
    top = aperture[y0][:, z0] * (1 - wz) + aperture[y0][:, z1] * wz
    bot = aperture[y1][:, z0] * (1 - wz) + aperture[y1][:, z1] * wz
    return (top * (1 - wy)[:, None] + bot * wy[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Beam basis
# ---------------------------------------------------------------------------

def _round2(values: Sequence[float]) -> list[float]:
    """Reproduce the ``%.2f`` rounding the Geant4 macro applied on write."""
    return [float(f"{float(v):.2f}") for v in values]


def gps_vectors(
    iso_center_mm: Sequence[float],
    gantry_angle_deg: float,
) -> tuple[list[float], list[float], list[float]]:
    """Return ``(focuspoint, direction, rot1)`` as written by ``generate_mac``.

    These are exactly the three vectors ``pipeline.extract_gps`` used to parse
    back out of the macro file, and they feed ``mac_cache`` entries ``s``,
    ``dx`` and ``dy``.
    """
    iso_x, iso_y, iso_z = (float(v) for v in iso_center_mm)
    theta = np.radians((270.0 + float(gantry_angle_deg)) % 360.0)
    gps_dist = SAD_MM * 2.0
    cx = gps_dist * np.cos(theta)
    cy = gps_dist * np.sin(theta)

    centre = np.array([cx, cy, iso_z])
    rot1 = np.cross(centre, [0.0, 0.0, 1.0])
    direction = np.array([-cx, -cy, 0.0])
    focus = centre / 2.0

    focuspoint = [focus[0] + iso_x, focus[1] + iso_y, iso_z]
    return _round2(focuspoint), _round2(direction), _round2(rot1)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


def beam_basis(
    direction: Sequence[float],
    rot1: Sequence[float],
) -> np.ndarray:
    """Orthonormal beam basis ``U = [ux, uy, uz]`` (``pipeline._basis_np``)."""
    ux = _normalize(np.asarray(direction, dtype=np.float32))
    uy0 = _normalize(np.asarray(rot1, dtype=np.float32))
    uz = _normalize(np.cross(ux, uy0))
    uy = _normalize(np.cross(uz, ux))
    return np.stack([ux, uy, uz], axis=0).astype(np.float32)


def mac_cache_entry(
    iso_center_mm: Sequence[float],
    gantry_angle_deg: float,
    grid: Any,
) -> dict[str, list[float] | int | float]:
    """Build one ``mac_cache`` record without touching the filesystem.

    Field-for-field identical to ``train/utils._build_mac_cache_entries``, which
    is what the BEV builder and the inverse-affine builder both consume.
    """
    s, dx, dy = gps_vectors(iso_center_mm, gantry_angle_deg)
    u = beam_basis(dx, dy)
    crop = float(grid.resolved_plane_origin_offset_mm)
    src = np.asarray(s, dtype=np.float32) + crop * u[0]
    return {
        "s": [float(v) for v in s],
        "dx": [float(v) for v in dx],
        "dy": [float(v) for v in dy],
        "U": u.reshape(-1).astype(np.float32).tolist(),
        "src": src.astype(np.float32).tolist(),
        "off": [0.0, -(grid.ny // 2) + 0.5, -(grid.nz // 2) + 0.5],
        "NX": int(grid.nx),
        "NY": int(grid.ny),
        "NZ": int(grid.nz),
        "z_align_mm": 0.0,
    }


# ---------------------------------------------------------------------------
# Grand Challenge metadata
# ---------------------------------------------------------------------------

class PhotonSegment:
    """One photon control point, resolved against its beam."""

    __slots__ = (
        "name", "image_file_idx", "beam_idx", "cp_idx", "iso_center",
        "gantry_angle", "mlc_left_mm", "mlc_right_mm",
        "output_file_idx", "idx_in_output", "minimum_cutoff",
    )

    def __init__(self, **kwargs: Any) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs[key])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PhotonSegment({self.name}, gantry={self.gantry_angle}, "
            f"out={self.output_file_idx}[{self.idx_in_output}])"
        )


def iter_photon_segments(metadata: Any) -> Iterator[PhotonSegment]:
    """Flatten ``stacked-photon-beam-level-metadata.json`` into control points.

    Nesting is ``image -> beams -> control_points -> output_info``. ``SAD``,
    ``iso_center`` and ``num_mlc_leaf_pairs`` sit at beam level for photons;
    ``iso_center`` may also appear at image level (the proton schema), which is
    used as a fallback.
    """
    for image in metadata:
        image_idx = int(image["image_file_idx"])
        image_iso = image.get("iso_center")
        for beam_idx, beam in enumerate(image.get("beams", [])):
            iso = beam.get("iso_center", image_iso)
            if iso is None:
                raise ValueError(
                    f"image {image_idx} beam {beam_idx}: no iso_center at beam "
                    "or image level"
                )
            sad = float(beam.get("SAD", SAD_MM))
            if abs(sad - SAD_MM) > 1e-6:
                raise ValueError(
                    f"image {image_idx} beam {beam_idx}: SAD={sad} but the "
                    f"trained geometry assumes SAD={SAD_MM}"
                )
            n_leaves = int(beam.get("num_mlc_leaf_pairs", LEAF_PAIRS))
            if n_leaves != LEAF_PAIRS:
                raise ValueError(
                    f"image {image_idx} beam {beam_idx}: num_mlc_leaf_pairs="
                    f"{n_leaves} but the trained aperture model assumes {LEAF_PAIRS}"
                )
            beam_id = int(beam.get("beam_idx", beam_idx))
            for cp_idx, control_point in enumerate(beam.get("control_points", [])):
                cp_id = int(control_point.get("cp_idx", cp_idx))
                info = control_point["output_info"]
                yield PhotonSegment(
                    name=f"img{image_idx}_B{beam_id}_CP{cp_id:03d}",
                    image_file_idx=image_idx,
                    beam_idx=beam_id,
                    cp_idx=cp_id,
                    iso_center=[float(v) for v in iso],
                    gantry_angle=float(control_point["gantry_angle"]),
                    mlc_left_mm=control_point["mlc_left_int_mm"],
                    mlc_right_mm=control_point["mlc_right_int_mm"],
                    output_file_idx=int(info["output_file_idx"]),
                    idx_in_output=int(info["idx_in_output"]),
                    minimum_cutoff=float(info["minimum_cutoff"]),
                )


def proton_mac_cache_entry(
    ray_source_mm: Sequence[float],
    ray_target_mm: Sequence[float],
    grid: Any,
) -> dict[str, list[float] | int | float]:
    """Build the ray-centred geometry used by proton population training."""
    source = np.asarray(ray_source_mm, dtype=np.float64)
    target = np.asarray(ray_target_mm, dtype=np.float64)
    depth = target - source
    depth_norm = float(np.linalg.norm(depth))
    if depth_norm <= 0.0:
        raise ValueError("proton ray source and target coincide")
    depth /= depth_norm

    patient_axial = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    horizontal = np.cross(patient_axial, depth)
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < 1.0e-6:
        raise ValueError("proton ray is parallel to the CT axial direction")
    horizontal /= horizontal_norm
    axial = np.cross(depth, horizontal)
    axial /= np.linalg.norm(axial)
    basis = np.stack((depth, horizontal, axial)).astype(np.float32)

    nx = int(grid.nx)
    dx = float(grid.spacing_dhw[0])
    plane_origin = target - (0.5 * nx * dx) * depth
    return {
        "s": source.astype(np.float32).tolist(),
        "dx": basis[0].tolist(),
        "dy": basis[1].tolist(),
        "U": basis.reshape(-1).tolist(),
        "src": plane_origin.astype(np.float32).tolist(),
        "off": [0.0, -(int(grid.ny) // 2) + 0.5, -(int(grid.nz) // 2) + 0.5],
        "NX": nx,
        "NY": int(grid.ny),
        "NZ": int(grid.nz),
        "z_align_mm": 0.0,
    }

class ProtonSegment:
    """One proton beamlet, resolved against its explicit ray geometry."""

    __slots__ = (
        "name", "image_file_idx", "beam_idx", "ray_idx", "beamlet_idx",
        "ray_source", "ray_target", "energy_mev", "sigma_spot_mm", "output_file_idx",
        "idx_in_output", "minimum_cutoff",
    )

    def __init__(self, **kwargs: Any) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs[key])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ProtonSegment({self.name}, energy={self.energy_mev}, "
            f"out={self.output_file_idx}[{self.idx_in_output}])"
        )

def iter_proton_segments(metadata: Any) -> Iterator[ProtonSegment]:
    """Flatten ``stacked-proton-beam-level-metadata.json`` into beamlets."""
    for image in metadata:
        image_idx = int(image["image_file_idx"])
        for beam_position, beam in enumerate(image.get("beams", [])):
            beam_idx = int(beam.get("beam_idx", beam_position))
            for ray_position, ray in enumerate(beam.get("rays", [])):
                ray_idx = int(ray.get("ray_idx", ray_position))
                source = ray.get("ray_source", ray.get("source"))
                target = ray.get("ray_target", ray.get("target"))
                if source is None or target is None:
                    raise ValueError(
                        f"image {image_idx} beam {beam_idx} ray {ray_idx}: "
                        "missing ray_source/ray_target"
                    )
                for beamlet_position, beamlet in enumerate(
                    ray.get("beamlets", [])
                ):
                    beamlet_idx = int(
                        beamlet.get("beamlet_idx", beamlet_position)
                    )
                    energy = beamlet.get("energy_mev", beamlet.get("energy"))
                    if energy is None:
                        raise ValueError(
                            f"image {image_idx} beam {beam_idx} ray {ray_idx} "
                            f"beamlet {beamlet_idx}: missing energy"
                        )
                    info = beamlet.get("output_info")
                    if info is None:
                        raise ValueError(
                            f"image {image_idx} beam {beam_idx} ray {ray_idx} "
                            f"beamlet {beamlet_idx}: missing output_info"
                        )
                    yield ProtonSegment(
                        name=(
                            f"img{image_idx}_B{beam_idx}_R{ray_idx:02d}_"
                            f"L{beamlet_idx}"
                        ),
                        image_file_idx=image_idx,
                        beam_idx=beam_idx,
                        ray_idx=ray_idx,
                        beamlet_idx=beamlet_idx,
                        ray_source=[float(value) for value in source],
                        ray_target=[float(value) for value in target],
                        energy_mev=float(energy),
                        sigma_spot_mm=(
                            None
                            if beamlet.get("sigma_spot_mm") is None
                            else float(beamlet["sigma_spot_mm"])
                        ),
                        output_file_idx=int(info["output_file_idx"]),
                        idx_in_output=int(info["idx_in_output"]),
                        minimum_cutoff=float(info["minimum_cutoff"]),
                    )
