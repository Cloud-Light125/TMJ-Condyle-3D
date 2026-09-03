"""nnU-Net v2 DatasetXXX_Name builder for the one-class task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..config import (
    CHANNEL_NAME,
    CONDYLE_LABEL,
    DATASET_ID,
    DATASET_NAME,
    FILE_ENDING,
    LABELS,
    NNUNET_PREPROCESSED_DIR,
    NNUNET_RAW_DIR,
    REPORTS_DIR,
    resolve_project_path,
)
from ..labels.qc import validate_pair
from ..utils.io import read_image, write_clean_image
from .splits import build_grouped_splits, write_fold_assignments, write_splits


def dataset_folder(
    nnunet_raw: str | Path = NNUNET_RAW_DIR,
    *,
    dataset_name: str = DATASET_NAME,
) -> Path:
    return Path(nnunet_raw) / dataset_name


def preprocessed_dataset_folder(
    nnunet_preprocessed: str | Path = NNUNET_PREPROCESSED_DIR,
    *,
    dataset_name: str = DATASET_NAME,
) -> Path:
    return Path(nnunet_preprocessed) / dataset_name


def _require_training_rows(
    rows: Iterable[dict[str, str]],
    *,
    allowed_statuses: set[str] | None = None,
) -> list[dict[str, str]]:
    """Select training rows, defaulting to the safety-critical VERIFIED set."""

    allowed = {str(value).upper() for value in (allowed_statuses or {"VERIFIED"})}
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("case_id", "")
        if not case_id or case_id in seen:
            raise ValueError(f"Duplicate or empty case_id in manifest: {case_id!r}")
        seen.add(case_id)
        status = row.get("annotation_status", "").upper()
        if status not in allowed:
            continue
        if not row.get("image_path") or not row.get("label_path"):
            raise ValueError(f"{case_id} is marked {status} but has no image_path/label_path")
        selected.append(row)
    if not selected:
        raise ValueError(
            "No VERIFIED mandibular condyle masks are available. "
            "Confirm the annotation in the TMJ workbench before training."
        )
    return selected


def build_dataset(
    rows: Iterable[dict[str, str]],
    *,
    nnunet_raw: str | Path = NNUNET_RAW_DIR,
    nnunet_preprocessed: str | Path = NNUNET_PREPROCESSED_DIR,
    reports_dir: str | Path = REPORTS_DIR,
    dataset_name: str = DATASET_NAME,
    dataset_id: int = DATASET_ID,
    n_splits: int = 5,
    seed: int = 20260902,
    allowed_statuses: set[str] | None = None,
) -> tuple[Path, Path, list[dict[str, list[str]]]]:
    """Build nnU-Net raw data from VERIFIED manifest rows only."""

    selected = _require_training_rows(rows, allowed_statuses=allowed_statuses)
    target = dataset_folder(nnunet_raw, dataset_name=dataset_name)
    images_dir = target / "imagesTr"
    labels_dir = target / "labelsTr"
    images_ts_dir = target / "imagesTs"
    for path in (images_dir, labels_dir, images_ts_dir):
        path.mkdir(parents=True, exist_ok=True)

    for row in selected:
        case_id = row["case_id"]
        image_path = resolve_project_path(row["image_path"])
        label_path = resolve_project_path(row["label_path"])
        image = read_image(image_path)
        label = read_image(label_path)
        qc = validate_pair(image, label, allow_empty=False)
        if qc["errors"]:
            raise ValueError(f"{case_id} failed QC: {'; '.join(qc['errors'])}")
        write_clean_image(image, images_dir / f"{case_id}_0000{FILE_ENDING}")
        write_clean_image(label, labels_dir / f"{case_id}{FILE_ENDING}")

    dataset_json = {
        "channel_names": {"0": CHANNEL_NAME},
        "labels": LABELS,
        "numTraining": len(selected),
        "file_ending": FILE_ENDING,
        "overwrite_image_reader_writer": "SimpleITKIO",
        "name": dataset_name,
        "description": "Single-class mandibular condyle segmentation from TMJ MRI",
        "reference": "TMJ-Condyle-3D; no patient data is included in this repository",
    }
    dataset_json_path = target / "dataset.json"
    with dataset_json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    splits = build_grouped_splits(selected, n_splits=n_splits, seed=seed)
    preprocessed_target = preprocessed_dataset_folder(
        nnunet_preprocessed, dataset_name=dataset_name
    )
    split_path = write_splits(splits, preprocessed_target / "splits_final.json")
    write_splits(splits, Path(reports_dir) / "splits_final.json")
    write_fold_assignments(splits, selected, Path(reports_dir) / "fold_assignments.csv")
    return target, split_path, splits
