"""Medical image I/O kept intentionally small and explicit."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def require_simpleitk() -> Any:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "SimpleITK is required. Install requirements/base.txt in the project venv."
        ) from exc
    return sitk


def read_image(path: str | Path) -> Any:
    sitk = require_simpleitk()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return sitk.ReadImage(str(path))


def write_clean_image(image: Any, path: str | Path) -> Path:
    """Write a metadata-scrubbed image without altering voxel values/geometry."""

    sitk = require_simpleitk()
    from .geometry import scrub_metadata

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean = scrub_metadata(image)
    sitk.WriteImage(clean, str(destination), useCompression=True)
    return destination
