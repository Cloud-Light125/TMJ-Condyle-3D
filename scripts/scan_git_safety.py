from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from _bootstrap import PROJECT_ROOT

FORBIDDEN_SUFFIXES = (
    ".dcm", ".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".raw",
    ".pth", ".pt", ".ckpt", ".h5", ".hdf5", ".npz",
)
SENSITIVE_VALUE_PATTERN = re.compile(
    r'''(?i)(?:PatientName|PatientID|StudyInstanceUID|AccessionNumber|'''
    r'''BirthDate|InstitutionName)\s*["']?\s*[:=]\s*["']?[^,}\n]+'''
)


def _git_names(staged: bool) -> list[str]:
    command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan tracked/staged files for medical data and identifiers.")
    parser.add_argument("--staged", action="store_true", help="Scan staged files; otherwise scan tracked files.")
    args = parser.parse_args()
    names = _git_names(args.staged)
    issues: list[str] = []
    for name in names:
        lower = name.lower()
        if lower.endswith(FORBIDDEN_SUFFIXES):
            issues.append(f"medical/model artifact suffix: {name}")
            continue
        path = PROJECT_ROOT / name
        if path.is_file() and path.stat().st_size <= 5_000_000:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if SENSITIVE_VALUE_PATTERN.search(text):
                issues.append(f"possible patient metadata key/value in {name}")
    if issues:
        print("MEDICAL DATA SCAN: FAIL")
        print("\n".join(issues))
        return 2
    print(f"MEDICAL DATA SCAN: PASS ({len(names)} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
