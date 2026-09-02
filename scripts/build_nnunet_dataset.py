from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from tmj_condyle.config import (
    DATASET_ID,
    DATASET_NAME,
    DEFAULT_SPLIT_SEED,
    LABELS_DIR,
    MANIFEST_PATH,
    NIFTI_DIR,
    NNUNET_PREPROCESSED_DIR,
    NNUNET_RAW_DIR,
    REPORTS_DIR,
)
from tmj_condyle.data.manifest import read_manifest
from tmj_condyle.data.nnunet import build_dataset
from tmj_condyle.data.validation import validate_manifest_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate real annotated cases and build an nnU-Net v2 raw dataset."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--images-dir", type=Path, default=NIFTI_DIR)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--nnunet-raw", type=Path, default=NNUNET_RAW_DIR)
    parser.add_argument("--nnunet-preprocessed", type=Path, default=NNUNET_PREPROCESSED_DIR)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-id", type=int, default=DATASET_ID)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    rows, valid = validate_manifest_dataset(
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        report_dir=args.reports_dir,
    )
    if not valid:
        print("Dataset build blocked: validation failed. See dataset_validation.md.")
        return 2
    manifest_rows = read_manifest(args.manifest)
    selected = [
        row
        for row in manifest_rows
        if row.get("annotation_status") in {"ANNOTATED", "VERIFIED"}
    ]
    if not selected:
        print("Dataset build blocked: no verified mandibular condyle masks.")
        return 2
    try:
        target, split_path, splits = build_dataset(
            selected,
            nnunet_raw=args.nnunet_raw,
            nnunet_preprocessed=args.nnunet_preprocessed,
            reports_dir=args.reports_dir,
            dataset_name=args.dataset_name,
            dataset_id=args.dataset_id,
            n_splits=args.n_splits,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports the actionable cause
        print(f"Dataset build failed: {type(exc).__name__}: {exc}")
        return 2
    print(f"nnU-Net raw dataset: {target.resolve()}")
    print(f"Grouped split file: {split_path.resolve()}")
    print(f"Training cases: {sum(len(split['val']) for split in splits)}")
    print("Patient/group leakage: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
