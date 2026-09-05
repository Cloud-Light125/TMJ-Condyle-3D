"""Project-wide paths and immutable task definition."""

from __future__ import annotations

import re
from pathlib import Path

from .runtime import application_root, user_data_dir

PROJECT_ROOT = application_root()
APP_ROOT = PROJECT_ROOT
USER_DATA_DIR = user_data_dir(app_root=PROJECT_ROOT)
WORKSPACE_DIR = USER_DATA_DIR
RAW_DIR = WORKSPACE_DIR / "raw"
NIFTI_DIR = WORKSPACE_DIR / "nifti"
LABELS_DIR = WORKSPACE_DIR / "labels"
PREDICTIONS_DIR = WORKSPACE_DIR / "predictions"
REPORTS_DIR = WORKSPACE_DIR / "reports"
NNUNET_RAW_DIR = WORKSPACE_DIR / "nnUNet_raw"
NNUNET_PREPROCESSED_DIR = WORKSPACE_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS_DIR = WORKSPACE_DIR / "nnUNet_results"
SLICER_MODELS_DIR = WORKSPACE_DIR / "slicer_models"
MANIFEST_PATH = WORKSPACE_DIR / "dataset_manifest.csv"

DATASET_ID = 501
DATASET_NAME = "Dataset501_CondyleMRI"
CONFIGURATION = "3d_fullres"
CHANNEL_NAME = "MRI"
CONDYLE_LABEL = 1
BACKGROUND_LABEL = 0
LABELS = {"background": BACKGROUND_LABEL, "mandibular_condyle": CONDYLE_LABEL}
FILE_ENDING = ".nii.gz"
N_FOLDS = 5
DEFAULT_SPLIT_SEED = 20260902
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def ensure_workspace_dirs() -> None:
    """Create only project-owned working directories.

    This never copies, deletes, or modifies any input medical files.
    """

    for path in (
        RAW_DIR,
        NIFTI_DIR,
        LABELS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
        NNUNET_RAW_DIR,
        NNUNET_PREPROCESSED_DIR,
        NNUNET_RESULTS_DIR,
        SLICER_MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_project_path(value: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a manifest path without interpreting it as a shell command.

    Manifest paths are relative to the writable user-data directory. Absolute
    paths remain supported for importing a file selected by the user.
    """

    path = Path(value)
    return path.resolve() if path.is_absolute() else ((base or USER_DATA_DIR) / path).resolve()


def manifest_path_for(path: str | Path) -> str:
    """Return a stable project-relative path for the private manifest."""

    resolved = Path(path).resolve()
    for root in (USER_DATA_DIR, PROJECT_ROOT):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def validate_case_id(case_id: str) -> str:
    """Validate an anonymous case identifier."""

    value = str(case_id).strip()
    if not CASE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "case_id must use only ASCII letters, digits, '_' or '-' and "
            "must not contain a patient name or other identifying text"
        )
    return value


def nifti_case_id(path: str | Path) -> str:
    """Get a case id from a .nii.gz filename."""

    name = Path(path).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return validate_case_id(name)
