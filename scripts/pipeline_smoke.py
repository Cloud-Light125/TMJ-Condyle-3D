"""Read-only CPU smoke check for the GUI workflow.

This command intentionally does not train, predict, or write Dice/IoU/HD95.
It only verifies that the real dataset/QC/split artifacts needed by the
formal CUDA experiment can be found and loaded.
"""

from __future__ import annotations

import json

from _bootstrap import PROJECT_ROOT  # noqa: F401

from tmj_condyle.config import (
    DATASET_NAME,
    MANIFEST_PATH,
    NNUNET_PREPROCESSED_DIR,
    NNUNET_RAW_DIR,
    REPORTS_DIR,
)
from tmj_condyle.data.manifest import read_manifest
from tmj_condyle.experiment import read_splits, read_validation_csv


def main() -> int:
    validation = read_validation_csv(REPORTS_DIR / "dataset_validation.csv")
    passed = [row for row in validation if row.get("status") == "PASS"]
    dataset_json = NNUNET_RAW_DIR / DATASET_NAME / "dataset.json"
    split_path = NNUNET_PREPROCESSED_DIR / DATASET_NAME / "splits_final.json"
    result: dict[str, object] = {
        "formal_training": False,
        "metrics_generated": False,
        "manifest": MANIFEST_PATH.is_file(),
        "validated_cases": len(validation),
        "passing_cases": len(passed),
        "dataset_json": dataset_json.is_file(),
        "splits": 0,
        "message": "CPU smoke only; no model training, prediction, or metrics were run.",
    }
    if not MANIFEST_PATH.is_file() or not validation or not dataset_json.is_file() or not split_path.is_file():
        print(json.dumps(result, ensure_ascii=False))
        return 2
    try:
        result["splits"] = len(read_splits(split_path))
        # Loading the manifest is a final cheap check that the GUI and the
        # command-line workflow see the same anonymous case records.
        result["manifest_cases"] = len(read_manifest(MANIFEST_PATH))
    except Exception as exc:  # noqa: BLE001 - smoke output is user-facing
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
