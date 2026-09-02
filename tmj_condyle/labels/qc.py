"""Strict binary condyle-mask quality control."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import BACKGROUND_LABEL, CONDYLE_LABEL
from ..utils.geometry import geometry_differences


def _connected_components(mask: np.ndarray) -> tuple[int | None, int | None]:
    try:
        from scipy import ndimage
    except ImportError:
        return None, None
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    components, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return 0, 0
    sizes = np.bincount(components.ravel())[1:]
    return int(count), int((sizes <= 10).sum())


def validate_label_array(
    array: np.ndarray,
    *,
    allow_empty: bool = False,
    many_component_threshold: int = 10,
) -> dict[str, Any]:
    """Validate a label array without modifying it."""

    errors: list[str] = []
    warnings: list[str] = []
    values = np.asarray(array)
    if not np.issubdtype(values.dtype, np.number):
        errors.append(f"label dtype is not numeric: {values.dtype}")
    elif not np.isfinite(values).all():
        errors.append("label contains NaN or Inf")
    elif not np.equal(values, np.floor(values)).all():
        errors.append("label contains non-integer values")

    unique_values: list[int] = []
    if not errors:
        unique_values = sorted(int(value) for value in np.unique(values))
        invalid = [value for value in unique_values if value not in (BACKGROUND_LABEL, CONDYLE_LABEL)]
        if invalid:
            errors.append(f"label values must be {{0, 1}}; found {invalid}")

    foreground_voxels = int(np.count_nonzero(values == CONDYLE_LABEL))
    if foreground_voxels == 0 and not allow_empty:
        errors.append("foreground is empty; an annotated condyle is required")

    component_count, small_component_count = _connected_components(values == CONDYLE_LABEL)
    if component_count is not None and component_count > many_component_threshold:
        warnings.append(
            f"mask has {component_count} connected components; inspect small isolated regions"
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "unique_values": unique_values,
            "foreground_voxels": foreground_voxels,
            "component_count": component_count,
            "small_component_count": small_component_count,
        },
    }


def validate_pair(
    image: Any,
    label: Any,
    *,
    allow_empty: bool = False,
    geometry_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Validate image finiteness, image/label geometry, and binary labels."""

    errors: list[str] = []
    warnings: list[str] = []
    try:
        import SimpleITK as sitk

        image_array = sitk.GetArrayFromImage(image)
        label_array = sitk.GetArrayFromImage(label)
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("SimpleITK is required for image/label validation") from exc

    if not np.isfinite(image_array).all():
        errors.append("image contains NaN or Inf")
    differences = geometry_differences(image, label, tolerance=geometry_tolerance)
    errors.extend(f"image/label geometry mismatch: {difference}" for difference in differences)
    label_qc = validate_label_array(label_array, allow_empty=allow_empty)
    errors.extend(label_qc["errors"])
    warnings.extend(label_qc["warnings"])
    spacing_xyz = tuple(float(value) for value in image.GetSpacing())
    stats = {
        **label_qc["stats"],
        "physical_volume_mm3": float(
            label_qc["stats"]["foreground_voxels"] * np.prod(spacing_xyz)
        ),
        "spacing_xyz": spacing_xyz,
    }
    return {"errors": errors, "warnings": warnings, "stats": stats}
