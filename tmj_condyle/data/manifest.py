"""Private anonymous dataset manifest helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from ..config import MANIFEST_PATH, manifest_path_for, validate_case_id

MANIFEST_FIELDS = [
    "case_id",
    "group_id",
    "side",
    "image_path",
    "label_path",
    "annotation_status",
    "geometry_valid",
    "label_valid",
    "notes",
]
ANNOTATION_STATUSES = {"NEW", "ANNOTATING", "ANNOTATED", "VERIFIED"}


def read_manifest(path: str | Path = MANIFEST_PATH) -> list[dict[str, str]]:
    manifest = Path(path)
    if not manifest.exists():
        return []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in MANIFEST_FIELDS if field not in fieldnames]
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
        return [{field: (row.get(field) or "").strip() for field in MANIFEST_FIELDS} for row in reader]


def write_manifest(
    rows: Iterable[dict[str, str]],
    path: str | Path = MANIFEST_PATH,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized: list[dict[str, str]] = []
    for raw in rows:
        row = {field: str(raw.get(field, "") or "").strip() for field in MANIFEST_FIELDS}
        row["case_id"] = validate_case_id(row["case_id"])
        row["group_id"] = row["group_id"] or row["case_id"]
        if row["annotation_status"] not in ANNOTATION_STATUSES:
            raise ValueError(
                f"annotation_status for {row['case_id']} must be one of {sorted(ANNOTATION_STATUSES)}"
            )
        normalized.append(row)
    normalized.sort(key=lambda item: item["case_id"])
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
    return destination


def upsert_case(
    *,
    case_id: str,
    image_path: str | Path,
    label_path: str | Path = "",
    group_id: str = "",
    side: str = "",
    annotation_status: str = "NEW",
    geometry_valid: bool | str = "",
    label_valid: bool | str = "",
    notes: str = "",
    path: str | Path = MANIFEST_PATH,
) -> Path:
    """Insert or update one anonymous case without patient identifiers."""

    case_id = validate_case_id(case_id)
    rows = read_manifest(path)
    updated = {
        "case_id": case_id,
        "group_id": group_id or case_id,
        "side": side,
        "image_path": manifest_path_for(image_path),
        "label_path": manifest_path_for(label_path) if label_path else "",
        "annotation_status": annotation_status,
        "geometry_valid": str(geometry_valid).lower() if geometry_valid != "" else "",
        "label_valid": str(label_valid).lower() if label_valid != "" else "",
        "notes": notes,
    }
    rows = [row for row in rows if row.get("case_id") != case_id]
    rows.append(updated)
    return write_manifest(rows, path)


def find_case(rows: Iterable[dict[str, str]], case_id: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("case_id") == case_id), None)
