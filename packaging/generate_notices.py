"""Generate a reproducible third-party notice bundle from staged metadata."""

from __future__ import annotations

import argparse
import email
import re
from pathlib import Path
from email.message import Message


def _metadata(path: Path) -> Message:
    message = email.message_from_bytes(path.read_bytes())
    return message


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "package"


def _compact_license(value: str) -> str:
    """Keep the inventory line readable when METADATA embeds full license text."""

    first_line = " ".join(str(value).split())
    return first_line[:240] + ("…" if len(first_line) > 240 else "")


def _license_files(dist_info: Path) -> list[Path]:
    candidates: list[Path] = []
    for directory in (dist_info / "licenses", dist_info):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.stat().st_size <= 2 * 1024 * 1024:
                candidates.append(path)
    return sorted(set(candidates), key=lambda path: str(path).casefold())


def _copy_license_files(files: list[Path], destination: Path, prefix: str) -> list[str]:
    copied: list[str] = []
    for index, source in enumerate(files, start=1):
        suffix = source.suffix or ".txt"
        target = destination / f"{_safe_name(prefix)}-{index}{suffix}"
        target.write_bytes(source.read_bytes())
        copied.append(target.name)
    return copied


def _distribution_rows(site_packages: Path, source_label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not site_packages.is_dir():
        return rows
    for dist_info in sorted(site_packages.glob("*.dist-info"), key=lambda path: path.name.casefold()):
        metadata_path = dist_info / "METADATA"
        if not metadata_path.is_file():
            continue
        try:
            metadata = _metadata(metadata_path)
        except (OSError, UnicodeError, ValueError):
            continue
        name = metadata.get("Name")
        version = metadata.get("Version")
        if not name or not version:
            continue
        project_urls = metadata.get_all("Project-URL") or []
        homepage = metadata.get("Home-page") or ""
        if not homepage and project_urls:
            homepage = project_urls[0].split(",", 1)[-1].strip()
        rows.append(
            {
                "name": str(name),
                "version": str(version),
                "license": _compact_license(str(metadata.get("License") or "metadata not declared")),
                "homepage": homepage,
                "source": source_label,
                "license_files": _license_files(dist_info),
            }
        )
    return rows


def _copy_named_file(source: Path, destination: Path, name: str) -> None:
    if source.is_file():
        destination.joinpath(name).write_bytes(source.read_bytes())


def generate(stage: Path) -> tuple[Path, Path]:
    licenses = stage / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    rows = _distribution_rows(
        stage / "runtime" / "python" / "Lib" / "site-packages",
        "portable CPython runtime",
    )
    rows.extend(
        _distribution_rows(
            stage / "runtime" / "slicer" / "lib" / "Python" / "Lib" / "site-packages",
            "3D Slicer embedded Python runtime",
        )
    )

    for row in rows:
        prefix = f"{row['source']}-{row['name']}-{row['version']}"
        copied = _copy_license_files(row["license_files"], licenses, prefix)  # type: ignore[arg-type]
        row["copied"] = copied

    _copy_named_file(stage / "LICENSE", licenses, "TMJ-Condyle-3D-LICENSE.txt")
    _copy_named_file(
        stage / "runtime" / "python" / "LICENSE.txt",
        licenses,
        "Python-LICENSE.txt",
    )
    _copy_named_file(
        stage / "runtime" / "slicer" / "lib" / "Python" / "Lib" / "LICENSE.txt",
        licenses,
        "Slicer-Python-LICENSE.txt",
    )

    rows.sort(key=lambda row: (str(row["name"]).casefold(), str(row["source"]).casefold()))
    lines = [
        "TMJ-Condyle-3D v0.1.0 — THIRD-PARTY NOTICES",
        "",
        "This offline Windows x64 release redistributes the components listed below.",
        "License files found in staged distributions are preserved under licenses/.",
        "",
        "Project: TMJ-Condyle-3D (MIT), see licenses/TMJ-Condyle-3D-LICENSE.txt.",
        "Python: CPython 3.10.21, PSF License, see licenses/Python-LICENSE.txt.",
        "3D Slicer: bundled runtime, see its preserved runtime license files and the Slicer website.",
        "This release redistributes the official Windows CPU-only PyTorch wheels; no CUDA or NVIDIA runtime is bundled.",
        "The optional GPU mode requires a compatible NVIDIA driver and a GPU-enabled runtime supplied separately by the user.",
        "",
        "Package inventory",
        "-----------------",
    ]
    for row in rows:
        lines.append(
            f"{row['name']}=={row['version']} | {row['license']} | "
            f"{row['source']} | {row['homepage'] or 'no homepage in metadata'}"
        )
        copied = row.get("copied") or []
        if copied:
            lines.append("  license files: " + ", ".join(f"licenses/{name}" for name in copied))
    lines.extend(
        [
            "",
            "Redistribution notes",
            "--------------------",
            "* Runtime binaries were taken only from the configured official/reference runtimes and wheels.",
            "* No patient scans, masks, checkpoints, test outputs, credentials, or development environments are included.",
            "* Review the license files in this directory together with each component's terms before external redistribution.",
            "",
        ]
    )
    notices = stage / "THIRD_PARTY_NOTICES.txt"
    notices.write_text("\n".join(lines), encoding="utf-8")
    return notices, licenses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    notices, licenses = generate(args.stage.resolve())
    print(f"Third-party notices: {notices}")
    print(f"License directory: {licenses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
