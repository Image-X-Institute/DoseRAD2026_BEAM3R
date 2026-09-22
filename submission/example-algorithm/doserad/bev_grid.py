"""BEV cuboid grid geometry for OTF training and inference pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_spacing_mm(value: Any) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        v = float(value)
        if v <= 0:
            raise ValueError(f"spacing_mm must be positive, got {value!r}")
        return (v, v, v)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        out = tuple(float(x) for x in value)
        if any(x <= 0 for x in out):
            raise ValueError(f"spacing_mm components must be positive, got {value!r}")
        return out
    raise ValueError(
        f"spacing_mm must be a positive number or length-3 list, got {value!r}"
    )


def cached_bev_offset(
    rec: dict[str, Any], nx: int, ny: int, nz: int
) -> list[float]:
    """Return an offset centred for the requested lattice shape.

    Cache offsets are generated for the normal coarse grid. A prediction grid
    with a different shape must be re-centred instead of inheriting a coarse
    lateral offset. Legacy caches without shape metadata retain their offset.
    """
    default = [0.0, -(ny // 2) + 0.5, -(nz // 2) + 0.5]
    cached_shape = (rec.get("NX"), rec.get("NY"), rec.get("NZ"))
    if all(value is not None for value in cached_shape):
        if tuple(int(value) for value in cached_shape) != (nx, ny, nz):
            return default
    return list(rec.get("off", default))


@dataclass(frozen=True)
class BevGridConfig:
    """BEV cuboid shape, voxel spacing, and segment projection geometry."""

    shape_dhw: tuple[int, int, int]
    spacing_dhw: tuple[float, float, float]
    sad_mm: float = 1000.0
    plane_origin_offset_mm: float | None = None
    segment_native_size: int = 400
    align_bev_z_to_ct_slices: bool = False
    align_orient_tol: float = 1e-4
    align_spacing_tol_mm: float = 1e-6
    aperture_sample_on_unaligned_grid: bool = False
    bicubic_z_align: bool = True
    bicubic_z_align_backend: str = "triton"
    version: int = 1

    @property
    def nx(self) -> int:
        return self.shape_dhw[0]

    @property
    def ny(self) -> int:
        return self.shape_dhw[1]

    @property
    def nz(self) -> int:
        return self.shape_dhw[2]

    @property
    def resolved_plane_origin_offset_mm(self) -> float:
        if self.plane_origin_offset_mm is not None:
            return float(self.plane_origin_offset_mm)
        dx = float(self.spacing_dhw[0])
        return float(self.sad_mm) - 0.5 * float(self.nx) * dx

    @property
    def segment_zoom(self) -> float:
        """Backward-compatible isotropic zoom accessor (uses Y zoom)."""
        return float(self.ny) / float(self.segment_native_size)

    @property
    def segment_zoom_yz(self) -> tuple[float, float]:
        """``scipy.ndimage.zoom`` factors from native segment grid to BEV ``(ny, nz)``."""
        native = float(self.segment_native_size)
        return (float(self.ny) / native, float(self.nz) / native)

    @property
    def physical_extent_dhw_mm(self) -> tuple[float, float, float]:
        """Physical BEV cuboid extent in mm as ``(nx*dx, ny*dy, nz*dz)``."""
        dx, dy, dz = self.spacing_dhw
        return (float(self.nx) * dx, float(self.ny) * dy, float(self.nz) * dz)

    @classmethod
    def default(cls) -> BevGridConfig:
        return cls(
            shape_dhw=(256, 200, 200),
            spacing_dhw=(2.0, 2.0, 2.0),
            sad_mm=1000.0,
            plane_origin_offset_mm=None,
            segment_native_size=400,
            version=1,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BevGridConfig:
        version = int(data.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported bev grid config version: {version}")

        raw_shape = data.get("shape_dhw")
        if raw_shape is None:
            raise ValueError("bev grid config missing required field 'shape_dhw'")
        shape = tuple(int(x) for x in raw_shape)
        if len(shape) != 3:
            raise ValueError(f"shape_dhw must have length 3, got {raw_shape!r}")
        if any(x <= 0 for x in shape):
            raise ValueError(f"shape_dhw entries must be positive, got {shape}")

        _nx, ny, nz = shape

        spacing = _parse_spacing_mm(data.get("spacing_mm", 2.0))
        sad_mm = float(data.get("sad_mm", 1000.0))
        if sad_mm <= 0:
            raise ValueError(f"sad_mm must be positive, got {sad_mm}")

        plane_offset = data.get("plane_origin_offset_mm")
        if plane_offset is not None:
            plane_offset = float(plane_offset)

        segment_native_size = int(data.get("segment_native_size", 400))
        if segment_native_size <= 0:
            raise ValueError(
                f"segment_native_size must be positive, got {segment_native_size}"
            )
        if ny <= 0 or nz <= 0:
            raise ValueError(f"shape_dhw entries must be positive, got {shape}")

        align_bev_z_to_ct_slices = bool(data.get("align_bev_z_to_ct_slices", False))
        align_orient_tol = float(data.get("align_orient_tol", 1e-4))
        align_spacing_tol_mm = float(data.get("align_spacing_tol_mm", 1e-6))
        aperture_sample_on_unaligned_grid = bool(
            data.get("aperture_sample_on_unaligned_grid", False)
        )
        bicubic_z_align = bool(data.get("bicubic_z_align", True))
        bicubic_z_align_backend = str(data.get("bicubic_z_align_backend", "triton"))
        if bicubic_z_align_backend not in ("triton", "cupy"):
            raise ValueError(
                f"bicubic_z_align_backend must be 'triton' or 'cupy', got {bicubic_z_align_backend!r}"
            )
        if align_orient_tol < 0:
            raise ValueError(f"align_orient_tol must be >= 0, got {align_orient_tol}")
        if align_spacing_tol_mm < 0:
            raise ValueError(
                f"align_spacing_tol_mm must be >= 0, got {align_spacing_tol_mm}"
            )

        return cls(
            shape_dhw=shape,
            spacing_dhw=spacing,
            sad_mm=sad_mm,
            plane_origin_offset_mm=plane_offset,
            segment_native_size=segment_native_size,
            align_bev_z_to_ct_slices=align_bev_z_to_ct_slices,
            align_orient_tol=align_orient_tol,
            align_spacing_tol_mm=align_spacing_tol_mm,
            aperture_sample_on_unaligned_grid=aperture_sample_on_unaligned_grid,
            bicubic_z_align=bicubic_z_align,
            bicubic_z_align_backend=bicubic_z_align_backend,
            version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "shape_dhw": list(self.shape_dhw),
            "spacing_mm": list(self.spacing_dhw),
            "sad_mm": self.sad_mm,
            "plane_origin_offset_mm": self.plane_origin_offset_mm,
            "segment_native_size": self.segment_native_size,
            "align_bev_z_to_ct_slices": self.align_bev_z_to_ct_slices,
            "align_orient_tol": self.align_orient_tol,
            "align_spacing_tol_mm": self.align_spacing_tol_mm,
            "aperture_sample_on_unaligned_grid": (
                self.aperture_sample_on_unaligned_grid
            ),
            "bicubic_z_align": self.bicubic_z_align,
            "bicubic_z_align_backend": self.bicubic_z_align_backend,
        }

    def validate_c3d_crop(self, crop_hw: int) -> None:
        if self.ny < crop_hw or self.nz < crop_hw:
            raise ValueError(
                f"BEV ny/nz ({self.ny}, {self.nz}) must be >= C3D crop_hw ({crop_hw})"
            )
        if (self.ny - crop_hw) % 2 != 0:
            raise ValueError(
                f"(ny - crop_hw) must be even for center crop, got ny={self.ny}, crop_hw={crop_hw}"
            )


def load_bev_grid_config(path: str | Path | None) -> BevGridConfig:
    if path is None or str(path) == "":
        return BevGridConfig.default()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"BEV grid config not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return BevGridConfig.from_dict(data)
