"""SimpleITK geometry comparison helpers.

SimpleITK exposes image sizes and spacing in x-y-z order. NumPy arrays
returned by GetArrayFromImage use z-y-x order. This module deliberately
keeps the geometry representation in SimpleITK's x-y-z order to avoid
accidental axis swaps during QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImageGeometry:
    size_xyz: tuple[int, ...]
    spacing_xyz: tuple[float, ...]
    origin_xyz: tuple[float, ...]
    direction: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "size_xyz": list(self.size_xyz),
            "spacing_xyz": list(self.spacing_xyz),
            "origin_xyz": list(self.origin_xyz),
            "direction": list(self.direction),
        }


def geometry_of(image: Any) -> ImageGeometry:
    return ImageGeometry(
        size_xyz=tuple(int(v) for v in image.GetSize()),
        spacing_xyz=tuple(float(v) for v in image.GetSpacing()),
        origin_xyz=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
    )


def _close(left: tuple[float, ...], right: tuple[float, ...], tolerance: float) -> bool:
    if len(left) != len(right):
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def geometries_match(
    image: Any,
    other: Any,
    *,
    tolerance: float = 1e-4,
) -> bool:
    """Compare shape, spacing, origin, and direction."""

    a = geometry_of(image)
    b = geometry_of(other)
    return (
        a.size_xyz == b.size_xyz
        and _close(a.spacing_xyz, b.spacing_xyz, tolerance)
        and _close(a.origin_xyz, b.origin_xyz, tolerance)
        and _close(a.direction, b.direction, tolerance)
    )


def geometry_differences(
    image: Any,
    other: Any,
    *,
    tolerance: float = 1e-4,
) -> list[str]:
    """Return human-readable geometry differences."""

    a = geometry_of(image)
    b = geometry_of(other)
    differences: list[str] = []
    if a.size_xyz != b.size_xyz:
        differences.append(f"size_xyz {a.size_xyz} != {b.size_xyz}")
    if not _close(a.spacing_xyz, b.spacing_xyz, tolerance):
        differences.append(f"spacing_xyz {a.spacing_xyz} != {b.spacing_xyz}")
    if not _close(a.origin_xyz, b.origin_xyz, tolerance):
        differences.append(f"origin_xyz {a.origin_xyz} != {b.origin_xyz}")
    if not _close(a.direction, b.direction, tolerance):
        differences.append("direction matrices differ")
    return differences


def scrub_metadata(image: Any) -> Any:
    """Remove DICOM/NIfTI metadata while keeping voxel data and geometry."""

    for key in list(image.GetMetaDataKeys()):
        image.EraseMetaData(key)
    return image
