"""Deterministic synthetic TMJ MRI volumes for isolated test-only runs.

The generator is intentionally small and explicit. It creates MRI-like float
volumes plus a binary ellipsoid or condyle-like foreground mask. The generated
files are for pipeline validation only; they are never used by the formal
manifest or the formal nnU-Net output directories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, radians, sin
from pathlib import Path
from typing import Literal

import numpy as np


ShapeMode = Literal["ellipsoid", "condyle"]


@dataclass(frozen=True)
class SyntheticCaseSpec:
    """All parameters needed to regenerate one synthetic case."""

    case_id: str
    seed: int
    shape_zyx: tuple[int, int, int] = (32, 48, 48)
    spacing_xyz: tuple[float, float, float] = (0.8, 0.8, 1.0)
    center_xyz_mm: tuple[float, float, float] = (19.0, 19.0, 13.5)
    axes_xyz_mm: tuple[float, float, float] = (5.5, 8.0, 5.5)
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    shape_mode: ShapeMode = "condyle"
    neck_scale_xyz: tuple[float, float, float] = (0.62, 0.66, 0.88)
    neck_offset_local_z_mm: float = 4.0
    intensity_scale: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _rotation_matrix_xyz(rotation_xyz_deg: tuple[float, float, float]) -> np.ndarray:
    """Return a local-to-world rotation matrix for intrinsic XYZ angles."""

    rx, ry, rz = (radians(float(value)) for value in rotation_xyz_deg)
    cx, sx = cos(rx), sin(rx)
    cy, sy = cos(ry), sin(ry)
    cz, sz = cos(rz), sin(rz)
    rot_x = np.asarray(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    rot_y = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rot_z = np.asarray(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    return rot_z @ rot_y @ rot_x


def _coordinate_grid(
    shape_zyx: tuple[int, int, int], spacing_xyz: tuple[float, float, float]
) -> np.ndarray:
    nz, ny, nx = (int(value) for value in shape_zyx)
    sx, sy, sz = (float(value) for value in spacing_xyz)
    x = np.arange(nx, dtype=np.float32) * sx
    y = np.arange(ny, dtype=np.float32) * sy
    z = np.arange(nz, dtype=np.float32) * sz
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.stack((xx, yy, zz), axis=-1)


def _ellipsoid(
    coordinates_xyz: np.ndarray,
    center_xyz_mm: np.ndarray,
    axes_xyz_mm: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    delta = coordinates_xyz - center_xyz_mm
    # rotation maps local coordinates to world coordinates, so R.T maps the
    # sampled world-space displacement back to the ellipsoid frame.
    local = np.einsum("ij,zyxj->zyxi", rotation.T, delta, optimize=True)
    scaled = local / np.maximum(axes_xyz_mm, 1e-6)
    return np.sum(scaled * scaled, axis=-1) <= 1.0


def _make_mask(spec: SyntheticCaseSpec, coordinates_xyz: np.ndarray) -> np.ndarray:
    center = np.asarray(spec.center_xyz_mm, dtype=np.float32)
    axes = np.asarray(spec.axes_xyz_mm, dtype=np.float32)
    rotation = _rotation_matrix_xyz(spec.rotation_xyz_deg)
    cap = _ellipsoid(coordinates_xyz, center, axes, rotation)
    if spec.shape_mode == "ellipsoid":
        return cap.astype(np.uint8)

    neck_center = center + rotation @ np.asarray(
        (0.0, 0.0, float(spec.neck_offset_local_z_mm)), dtype=np.float32
    )
    neck_axes = axes * np.asarray(spec.neck_scale_xyz, dtype=np.float32)
    neck = _ellipsoid(coordinates_xyz, neck_center, neck_axes, rotation)
    return np.logical_or(cap, neck).astype(np.uint8)


def generate_synthetic_case(spec: SyntheticCaseSpec) -> tuple[np.ndarray, np.ndarray]:
    """Generate one MRI-like volume and its binary mask as ``(image, mask)``.

    The MRI contrast is tied to the mask but also contains smooth anatomy-like
    distractors and noise. The random seed is part of the spec, so the output
    is deterministic across runs on the same dependency versions.
    """

    shape = tuple(int(value) for value in spec.shape_zyx)
    coordinates = _coordinate_grid(shape, spec.spacing_xyz)
    mask = _make_mask(spec, coordinates)
    rng = np.random.default_rng(int(spec.seed))

    finite_coordinates = coordinates.reshape(-1, 3)
    extent = np.maximum(finite_coordinates.max(axis=0), 1.0)
    normalized = coordinates / extent
    image = 0.18 + 0.035 * normalized[..., 0] + 0.025 * normalized[..., 1]

    # Smooth low-frequency bias field and a few non-label anatomical structures
    # provide a non-trivial image for the real nnU-Net preprocessing/trainer.
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:  # pragma: no cover - dependency is project base
        raise RuntimeError("scipy is required for synthetic MRI generation") from exc

    low_frequency = gaussian_filter(rng.normal(size=shape), sigma=4.0)
    low_frequency = low_frequency / max(float(np.std(low_frequency)), 1e-6)
    image += 0.025 * low_frequency
    for _ in range(3):
        blob_center = rng.uniform(extent * 0.18, extent * 0.82)
        blob_axes = rng.uniform(extent * 0.08, extent * 0.22)
        blob = np.sum(((coordinates - blob_center) / blob_axes) ** 2, axis=-1)
        image += rng.uniform(0.025, 0.075) * np.exp(-0.5 * blob)

    mask_float = mask.astype(np.float32)
    soft_mask = gaussian_filter(mask_float, sigma=0.8)
    image += float(spec.intensity_scale) * (0.24 * mask_float + 0.24 * soft_mask)
    image += rng.normal(0.0, 0.018, size=shape)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = image - float(np.percentile(image, 1.0))
    image /= max(float(np.percentile(image, 99.0)), 1e-6)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image, mask.astype(np.uint8)


def write_synthetic_case(
    spec: SyntheticCaseSpec,
    image_path: str | Path,
    label_path: str | Path,
) -> tuple[Path, Path]:
    """Generate and write a metadata-scrubbed SimpleITK image/label pair."""

    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - dependency is project base
        raise RuntimeError("SimpleITK is required for synthetic NIfTI generation") from exc

    image_array, mask_array = generate_synthetic_case(spec)
    image = sitk.GetImageFromArray(image_array)
    label = sitk.GetImageFromArray(mask_array)
    image.SetSpacing(tuple(float(value) for value in spec.spacing_xyz))
    label.CopyInformation(image)
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    label.CopyInformation(image)

    image_destination = Path(image_path)
    label_destination = Path(label_path)
    image_destination.parent.mkdir(parents=True, exist_ok=True)
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(image_destination), useCompression=True)
    sitk.WriteImage(label, str(label_destination), useCompression=True)
    return image_destination, label_destination
