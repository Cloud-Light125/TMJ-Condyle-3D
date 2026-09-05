"""Slicer discovery helpers shared by source mode and the packaged GUI.

The release launcher never searches the host for Slicer: it starts the copy at
``<app-root>\\runtime\\slicer\\Slicer.exe``.  The broader discovery functions
remain available for source-mode onboarding and for backwards-compatible tests.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .runtime import application_root, user_data_dir


PROJECT_ROOT = application_root()
SLICER_CONFIG_NAME = ".tmj_platform_config.json"
# Tests and source-mode callers may monkeypatch this value.  A real release
# never supplies a host-specific default here.
DEFAULT_SLICER_PATH: Path | None = None


@dataclass(frozen=True)
class SlicerCandidate:
    """One executable discovered by the launcher."""

    path: Path
    source: str


def slicer_config_path(
    project_root: str | Path = PROJECT_ROOT,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Return the writable user configuration path.

    The project-root fallback is retained only when source-mode tests do not
    provide ``TMJ_USER_DATA_DIR``.  Installed builds never write into Program
    Files.
    """

    root = Path(project_root).expanduser().resolve()
    if data_root is not None:
        return Path(data_root).expanduser().resolve() / SLICER_CONFIG_NAME
    configured_data = os.environ.get("TMJ_USER_DATA_DIR")
    if configured_data and root == application_root(project_root):
        return user_data_dir(configured_data, app_root=root) / SLICER_CONFIG_NAME
    return root / "workspace" / SLICER_CONFIG_NAME


def read_slicer_config(path: str | Path) -> dict[str, object]:
    """Read a launcher configuration, tolerating a missing/corrupt file."""

    destination = Path(path)
    if not destination.is_file():
        return {}
    try:
        value = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def configured_slicer_path(project_root: str | Path = PROJECT_ROOT) -> Path | None:
    """Return a valid configured Slicer executable, if one exists."""

    root = Path(project_root)
    value = read_slicer_config(slicer_config_path(root)).get("slicer_path")
    if not value:
        return None
    candidate = Path(str(value).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    return candidate if candidate.is_file() and candidate.name.lower() == "slicer.exe" else None


def write_slicer_config(
    slicer_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Persist the selected executable for future one-click launches."""

    executable = Path(slicer_path).expanduser().resolve()
    if not executable.is_file() or executable.name.lower() != "slicer.exe":
        raise ValueError(f"Slicer.exe was not found: {executable}")
    destination = slicer_config_path(project_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "slicer_path": str(executable),
        "updated_by": "TMJ-Condyle-3D",
    }
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _env_value(environment: Mapping[str, str] | None, name: str) -> str:
    if environment is None:
        return os.environ.get(name, "")
    return str(environment.get(name, "") or "")


def _existing_executable(path: Path) -> Path | None:
    try:
        candidate = path.expanduser().resolve()
    except OSError:
        return None
    if candidate.is_file() and candidate.name.lower() == "slicer.exe":
        return candidate
    return None


def _glob_executables(parent: Path, pattern: str | Path) -> list[Path]:
    if not parent.is_dir():
        return []
    try:
        return [
            path.resolve()
            for path in parent.glob(str(pattern))
            if path.is_file() and path.name.lower() == "slicer.exe"
        ]
    except OSError:
        return []


def discover_slicer_candidates(
    *,
    project_root: str | Path = PROJECT_ROOT,
    environment: Mapping[str, str] | None = None,
) -> list[SlicerCandidate]:
    """Find Slicer installations in the documented preference order.

    The bundled runtime is first, followed by the optional test/source default,
    the user's saved choice and then common Windows locations. Duplicate paths
    are removed case-insensitively.
    """

    root = Path(project_root)
    program_files = Path(_env_value(environment, "ProgramFiles") or r"C:\Program Files")
    program_files_x86 = Path(
        _env_value(environment, "ProgramFiles(x86)") or r"C:\Program Files (x86)"
    )
    local_app_data = Path(
        _env_value(environment, "LOCALAPPDATA")
        or (Path.home() / "AppData" / "Local")
    )
    user_profile = Path(_env_value(environment, "USERPROFILE") or Path.home())

    ordered: list[tuple[Path, str]] = []
    bundled = root / "runtime" / "slicer" / "Slicer.exe"
    if bundled.is_file():
        ordered.append((bundled, "包内 Slicer"))
    if DEFAULT_SLICER_PATH is not None:
        ordered.append((DEFAULT_SLICER_PATH, "项目默认位置"))
    configured = configured_slicer_path(root)
    if configured is not None:
        ordered.append((configured, "项目设置"))

    common_patterns = [
        (program_files, Path("3D Slicer*") / "Slicer.exe", "Program Files"),
        (program_files, Path("Slicer*") / "Slicer.exe", "Program Files"),
        (program_files_x86, Path("3D Slicer*") / "Slicer.exe", "Program Files (x86)"),
        (program_files_x86, Path("Slicer*") / "Slicer.exe", "Program Files (x86)"),
        (local_app_data, Path("slicer.org") / "*" / "Slicer.exe", "用户本地安装"),
        (local_app_data, Path("NA-MIC") / "*" / "Slicer.exe", "用户本地安装"),
        (local_app_data, Path("Programs") / "3D Slicer*" / "Slicer.exe", "用户本地安装"),
        (local_app_data, Path("Programs") / "Slicer*" / "Slicer.exe", "用户本地安装"),
        (user_profile, Path("Apps") / "3D Slicer*" / "Slicer.exe", "用户目录"),
        (user_profile, Path("Apps") / "Slicer*" / "Slicer.exe", "用户目录"),
        (user_profile, Path("3D Slicer*") / "Slicer.exe", "用户目录"),
        (user_profile, Path("Slicer*") / "Slicer.exe", "用户目录"),
    ]
    for parent, pattern, source in common_patterns:
        for path in _glob_executables(parent, pattern):
            ordered.append((path, source))

    output: list[SlicerCandidate] = []
    seen: set[str] = set()
    for path, source in ordered:
        executable = _existing_executable(path)
        if executable is None:
            continue
        key = str(executable).casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(SlicerCandidate(executable, source))
    return output


def discover_slicer_paths(
    *,
    project_root: str | Path = PROJECT_ROOT,
    environment: Mapping[str, str] | None = None,
) -> list[Path]:
    """Convenience API for callers that only need executable paths."""

    return [item.path for item in discover_slicer_candidates(project_root=project_root, environment=environment)]


__all__ = [
    "DEFAULT_SLICER_PATH",
    "SLICER_CONFIG_NAME",
    "SlicerCandidate",
    "configured_slicer_path",
    "discover_slicer_candidates",
    "discover_slicer_paths",
    "read_slicer_config",
    "slicer_config_path",
    "write_slicer_config",
]
