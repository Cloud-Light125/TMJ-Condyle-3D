"""Dataset and image/label validation shared by CLI scripts."""

from __future__ import annotations

import csv
from pathlib import Path

from ..config import LABELS_DIR, NIFTI_DIR, REPORTS_DIR, resolve_project_path
from ..labels.qc import validate_pair
from ..utils.io import read_image
from .manifest import read_manifest


def nifti_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and (path.name.endswith(".nii.gz") or path.name.endswith(".nii"))
    )


def _case_id_from_filename(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def _index_nifti_files(directory: str | Path) -> tuple[dict[str, Path], set[str]]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in nifti_files(directory):
        case_id = _case_id_from_filename(path)
        if case_id in index:
            duplicates.add(case_id)
        else:
            index[case_id] = path
    return index, duplicates


def validate_manifest_dataset(
    *,
    manifest_path: str | Path,
    images_dir: str | Path = NIFTI_DIR,
    labels_dir: str | Path = LABELS_DIR,
    report_dir: str | Path = REPORTS_DIR,
    statuses: set[str] | None = None,
) -> tuple[list[dict[str, object]], bool]:
    """Validate manifest rows and report orphan files.

    When ``statuses`` is supplied, only rows with one of those annotation
    statuses are considered.  The training workflow uses ``{"VERIFIED"}`` so
    a newly imported MRI or an unconfirmed mask cannot block or enter formal
    training data.
    """

    all_manifest_rows = read_manifest(manifest_path)
    allowed_statuses = {str(value).upper() for value in statuses} if statuses else None
    manifest_rows = (
        [
            row
            for row in all_manifest_rows
            if str(row.get("annotation_status", "")).upper() in allowed_statuses
        ]
        if allowed_statuses is not None
        else all_manifest_rows
    )
    image_files, duplicate_image_ids = _index_nifti_files(images_dir)
    label_files, duplicate_label_ids = _index_nifti_files(labels_dir)
    duplicate_manifest_ids = {
        case_id
        for case_id in [row.get("case_id", "") for row in manifest_rows]
        if case_id and sum(r.get("case_id") == case_id for r in manifest_rows) > 1
    }
    by_case: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        case_id = row.get("case_id", "")
        by_case.setdefault(case_id, row)

    all_case_ids = (
        sorted(set(by_case))
        if allowed_statuses is not None
        else sorted(set(by_case) | set(image_files) | set(label_files))
    )
    result_rows: list[dict[str, object]] = []
    for case_id in all_case_ids:
        row = by_case.get(case_id, {})
        image_value = row.get("image_path") or str(image_files.get(case_id, ""))
        label_value = row.get("label_path") or str(label_files.get(case_id, ""))
        image_path = resolve_project_path(image_value) if image_value else None
        label_path = resolve_project_path(label_value) if label_value else None
        errors: list[str] = []
        warnings: list[str] = []
        if case_id in duplicate_manifest_ids:
            errors.append("duplicate case_id in manifest")
        if case_id in duplicate_image_ids:
            errors.append("duplicate case_id in image directory")
        if case_id in duplicate_label_ids:
            errors.append("duplicate case_id in label directory")
        if not row:
            errors.append("case is not present in manifest")
        if image_path is None or not image_path.is_file():
            errors.append("missing image")
        if label_path is None or not label_path.is_file():
            errors.append("missing label")

        stats: dict[str, object] = {}
        if image_path is not None and label_path is not None and image_path.is_file() and label_path.is_file():
            try:
                image = read_image(image_path)
                label = read_image(label_path)
                qc = validate_pair(image, label, allow_empty=False)
                errors.extend(qc["errors"])
                warnings.extend(qc["warnings"])
                stats.update(qc["stats"])
            except Exception as exc:  # noqa: BLE001 - report per-case failures
                errors.append(f"read/validation error: {type(exc).__name__}: {exc}")

        result_rows.append(
            {
                "case_id": case_id,
                "group_id": row.get("group_id", ""),
                "side": row.get("side", ""),
                "image_path": str(image_path) if image_path is not None else "",
                "label_path": str(label_path) if label_path is not None else "",
                "annotation_status": row.get("annotation_status", ""),
                "errors": "; ".join(errors),
                "warnings": "; ".join(warnings),
                "status": "PASS" if not errors else "FAIL",
                **stats,
            }
        )

    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "dataset_validation.csv"
    fieldnames = [
        "case_id",
        "group_id",
        "side",
        "image_path",
        "label_path",
        "annotation_status",
        "status",
        "errors",
        "warnings",
        "foreground_voxels",
        "physical_volume_mm3",
        "component_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result_rows)

    passed = sum(row["status"] == "PASS" for row in result_rows)
    failed = len(result_rows) - passed
    md_path = report_root / "dataset_validation.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Dataset validation\n\n")
        handle.write(f"- Cases checked: {len(result_rows)}\n")
        handle.write(f"- Passing cases: {passed}\n")
        handle.write(f"- Failing cases: {failed}\n")
        handle.write("- Empty masks are rejected for training.\n")
        handle.write("- Geometry means size, spacing, origin, and direction.\n\n")
        handle.write("| Case | Group | Status | Errors | Warnings |\n")
        handle.write("|---|---|---|---|---|\n")
        for result in result_rows:
            handle.write(
                f"| {result['case_id']} | {result['group_id']} | {result['status']} | "
                f"{result['errors']} | {result['warnings']} |\n"
            )
    return result_rows, failed == 0 and bool(result_rows)
