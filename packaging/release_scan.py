"""Scan a release staging directory for data, secrets, and development residue."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FORBIDDEN_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".dcm",
    ".nrrd",
    ".mha",
    ".mhd",
    ".h5",
    ".hdf5",
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".log",
)
FORBIDDEN_COMPONENTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "workspace",
    "test_only_tmj_synthetic",
    "tests",
}
RUNTIME_SUPPORT_ALLOWLIST = {
    "runtime/python/lib/site-packages/numpy/_core/tests",
    "runtime/python/lib/site-packages/numpy/_core/tests/_natype.py",
}
FORBIDDEN_TEXT = (
    re.compile(r"PatientName|PatientID|BirthDate|AccessionNumber|Institution", re.I),
    re.compile(r"case_001", re.I),
    re.compile(r"C:\\Users\\[^\\/\r\n]*", re.I),
    re.compile(r"C:\\code\\", re.I),
    re.compile(r"(?:^|[\\/])\.venv(?:[\\/]|$)", re.I),
    re.compile(r"(?:password|secret|access[_ -]?token|api[_ -]?key)\s*[:=]", re.I),
)
PATH_AND_SECRET_TEXT = FORBIDDEN_TEXT[2:]
TEXT_SUFFIXES = {
    ".bat",
    ".cmd",
    ".cs",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".pyi",
    ".rst",
    ".toml",
    ".txt",
    ".vbs",
    ".xml",
    ".yaml",
    ".yml",
}


def _is_dependency_runtime(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root).parts
    except ValueError:
        return False
    return bool(relative) and relative[0] == "runtime"


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    if not root.is_dir():
        return [f"staging directory does not exist: {root}"]
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        lowered_parts = {part.casefold() for part in relative.parts}
        if lowered_parts & {item.casefold() for item in FORBIDDEN_COMPONENTS}:
            normalized = relative.as_posix().casefold()
            if normalized not in {item.casefold() for item in RUNTIME_SUPPORT_ALLOWLIST}:
                findings.append(f"forbidden directory/component: {relative}")
                continue
        if path.is_file() and path.name.casefold().endswith(FORBIDDEN_SUFFIXES):
            findings.append(f"forbidden medical/model artifact: {relative}")
            continue
        # Vendor runtimes can legitimately mention generic DICOM field names or
        # build paths in their own documentation.  They are still checked for
        # forbidden file types/dirs above; application text is checked strictly.
        if not path.is_file() or _is_dependency_runtime(path, root):
            continue
        if path.suffix.casefold() not in TEXT_SUFFIXES or path.stat().st_size > 4 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            findings.append(f"unreadable release text file: {relative}: {exc}")
            continue
        relative_first = relative.parts[0].casefold() if relative.parts else ""
        # Source code may legitimately contain the names of DICOM fields or a
        # synthetic case template because it implements scrubbing/tests.  The
        # same strings in user-facing data, reports, or package notices remain
        # forbidden.  Paths and credential patterns are always checked.
        patterns = (
            PATH_AND_SECRET_TEXT
            if path.suffix.casefold() in {".py", ".cs", ".ps1", ".bat", ".vbs"}
            or relative_first in {"licenses", "third_party_notices.txt"}
            else FORBIDDEN_TEXT
        )
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                findings.append(f"forbidden text {pattern.pattern!r} in {relative} (offset {match.start()})")
                break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    findings = scan(args.root.resolve())
    report = [f"Release safety scan: {'PASS' if not findings else 'FAIL'}", f"Root: {args.root.resolve()}"]
    report.extend(f"- {finding}" for finding in findings)
    output = "\n".join(report) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
