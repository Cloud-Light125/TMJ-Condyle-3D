"""Relocatable application-runtime and subprocess environment helpers.

The installed application has two roots:

* ``APP_ROOT`` contains immutable program files, the bundled Slicer runtime,
  the bundled CPython runtime and project scripts.
* ``USER_DATA_DIR`` contains all medical data and generated experiment output.

This module intentionally has no third-party imports so it can be used by the
launcher boundary and by small diagnostics before the scientific packages are
loaded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping


APP_ROOT_ENV = "TMJ_APP_ROOT"
USER_DATA_ENV = "TMJ_USER_DATA_DIR"
RUNTIME_MODE_ENV = "TMJ_RUNTIME_MODE"

_SOURCE_ROOT = Path(__file__).resolve().parents[1]


def application_root(value: str | Path | None = None) -> Path:
    """Resolve the immutable application root without consulting PATH."""

    candidate = value or os.environ.get(APP_ROOT_ENV) or _SOURCE_ROOT
    return Path(candidate).expanduser().resolve()


def default_user_data_dir() -> Path:
    """Return the per-user Documents directory used by the release build."""

    documents = os.environ.get("TMJ_DOCUMENTS_DIR")
    if documents:
        base = Path(documents).expanduser().resolve() / "TMJ-Condyle-3D"
    else:
        base = (Path.home() / "Documents" / "TMJ-Condyle-3D").resolve()
    return base / "workspace"


def user_data_dir(
    value: str | Path | None = None,
    *,
    app_root: str | Path | None = None,
) -> Path:
    """Resolve the writable user-data root.

    ``TMJ_USER_DATA_DIR`` is set by the self-contained Windows launcher.  The
    fallback keeps source-mode command-line tests usable while still keeping
    medical data outside the source tree by default.
    """

    candidate = value or os.environ.get(USER_DATA_ENV) or default_user_data_dir()
    resolved = Path(candidate).expanduser().resolve()
    if app_root is not None and resolved == application_root(app_root):
        raise ValueError("User data must not be stored in the immutable application root")
    return resolved


def is_packaged_application(app_root: str | Path | None = None) -> bool:
    """Whether the expected bundled Python and Slicer files are present."""

    root = application_root(app_root)
    return (
        (root / "runtime" / "python" / "python.exe").is_file()
        and (root / "runtime" / "slicer" / "Slicer.exe").is_file()
    )


def runtime_python_executable(
    app_root: str | Path | None = None,
    *,
    allow_development_fallback: bool = True,
) -> Path:
    """Return an absolute Python executable for this application.

    A packaged build is strict: if its runtime is missing, this raises instead
    of silently falling back to a global interpreter.  The source-mode
    fallback exists only for repository tests and developer scripts.
    """

    root = application_root(app_root)
    bundled = root / "runtime" / "python" / "python.exe"
    if bundled.is_file():
        return bundled.resolve()
    packaged_marker = (
        os.environ.get(RUNTIME_MODE_ENV, "").casefold() == "packaged"
        or (root / "runtime").is_dir()
    )
    if packaged_marker or not allow_development_fallback:
        raise FileNotFoundError(f"Bundled Python runtime was not found: {bundled}")
    development = root / ".venv" / "Scripts" / "python.exe"
    if development.is_file():
        return development.resolve()
    development_unix = root / ".venv" / "bin" / "python"
    if development_unix.is_file():
        return development_unix.resolve()
    # This is still an absolute path and is only reachable in source mode.
    return Path(sys.executable).resolve()


def bundled_slicer_executable(app_root: str | Path | None = None) -> Path:
    """Return the bundled Slicer executable, never a system installation."""

    root = application_root(app_root)
    executable = root / "runtime" / "slicer" / "Slicer.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"Bundled Slicer runtime was not found: {executable}")
    return executable.resolve()


def nnunet_module(executable: str) -> str:
    """Map an nnU-Net v2 console name to its importable module."""

    modules = {
        "nnUNetv2_plan_and_preprocess": "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
        "nnUNetv2_train": "nnunetv2.run.run_training",
        "nnUNetv2_predict": "nnunetv2.inference.predict_from_raw_data",
    }
    try:
        return modules[executable]
    except KeyError as exc:
        raise ValueError(f"No module entry point is defined for {executable}") from exc


def nnunet_command(
    executable: str,
    *,
    python_executable: str | Path | None = None,
    app_root: str | Path | None = None,
) -> list[str]:
    """Build an nnU-Net command using the bundled interpreter and no PATH.

    nnU-Net 2.8.1's prediction module contains a legacy demonstration block
    under ``if __name__ == '__main__'``.  Calling it with ``python -m`` runs
    that block after the real CLI and can launch an unrelated Hippocampus
    example.  The installed console entry point calls ``predict_entry_point``
    directly, so reproduce that behavior with ``python -c`` for prediction.
    """

    interpreter = Path(python_executable).expanduser().resolve() if python_executable else runtime_python_executable(app_root)
    if executable == "nnUNetv2_predict":
        return [
            str(interpreter),
            "-c",
            "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()",
        ]
    return [str(interpreter), "-m", nnunet_module(executable)]


def runtime_environment(
    *,
    app_root: str | Path | None = None,
    data_root: str | Path | None = None,
    base: Mapping[str, str] | None = None,
    include_pythonpath: bool = True,
) -> dict[str, str]:
    """Create an isolated child-process environment without mutating globals.

    Slicer's embedded Python exports ``PYTHONHOME``/``PYTHONPATH`` values that
    must not reach the separate CPython runtime.  The only Python path we add
    is the application root, which contains this project's importable package.
    ``PATH`` is augmented for native DLL discovery, but all application entry
    points still use absolute executable paths.
    """

    root = application_root(app_root)
    writable = user_data_dir(data_root, app_root=root)
    environment = dict(os.environ if base is None else base)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
        environment.pop(name, None)
    python_root = root / "runtime" / "python"
    slicer_root = root / "runtime" / "slicer"
    path_parts = [
        str(python_root),
        str(python_root / "DLLs"),
        str(python_root / "Scripts"),
        str(slicer_root),
    ]
    existing_path = environment.get("PATH", "")
    if existing_path:
        path_parts.append(existing_path)
    environment["PATH"] = os.pathsep.join(path_parts)
    environment[APP_ROOT_ENV] = str(root)
    environment[USER_DATA_ENV] = str(writable)
    environment["nnUNet_raw"] = str(writable / "nnUNet_raw")
    environment["nnUNet_preprocessed"] = str(writable / "nnUNet_preprocessed")
    environment["nnUNet_results"] = str(writable / "nnUNet_results")
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    if include_pythonpath:
        environment["PYTHONPATH"] = str(root)
    return environment


def ensure_user_data_dirs(root: str | Path | None = None) -> Path:
    """Create the empty, user-writable workspace template."""

    writable = user_data_dir(root, app_root=application_root())
    for name in (
        "raw",
        "nifti",
        "labels",
        "predictions",
        "reports",
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results",
        "slicer_models",
        "experiments",
        "exports",
        "logs",
    ):
        (writable / name).mkdir(parents=True, exist_ok=True)
    return writable


__all__ = [
    "APP_ROOT_ENV",
    "USER_DATA_ENV",
    "application_root",
    "bundled_slicer_executable",
    "default_user_data_dir",
    "ensure_user_data_dirs",
    "is_packaged_application",
    "nnunet_command",
    "nnunet_module",
    "runtime_environment",
    "runtime_python_executable",
    "user_data_dir",
]
