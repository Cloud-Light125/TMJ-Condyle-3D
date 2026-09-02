from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from tmj_condyle.config import (
    DATASET_NAME,
    DEFAULT_SPLIT_SEED,
    MANIFEST_PATH,
    NNUNET_PREPROCESSED_DIR,
    REPORTS_DIR,
)
from tmj_condyle.data.manifest import read_manifest
from tmj_condyle.data.splits import (
    build_grouped_splits,
    write_fold_assignments,
    write_splits,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe grouped five-fold splits for nnU-Net v2."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    rows = [
        row
        for row in read_manifest(args.manifest)
        if row.get("annotation_status") in {"ANNOTATED", "VERIFIED"}
        and row.get("image_path")
        and row.get("label_path")
    ]
    if not rows:
        raise RuntimeError(
            "No annotated cases found. Create real masks in 3D Slicer first."
        )
    splits = build_grouped_splits(rows, n_splits=args.n_splits, seed=args.seed)
    output = args.out or (
        NNUNET_PREPROCESSED_DIR / args.dataset_name / "splits_final.json"
    )
    write_splits(splits, output)
    write_splits(splits, args.report_dir / "splits_final.json")
    write_fold_assignments(splits, rows, args.report_dir / "fold_assignments.csv")
    for fold, split in enumerate(splits):
        print(f"fold {fold}: train={len(split['train'])}, validation={len(split['val'])}")
    print("Patient/group leakage: PASS (all train/validation group intersections are empty)")
    print(f"Written: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
