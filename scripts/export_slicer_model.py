from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from tmj_condyle.config import (
    CONFIGURATION,
    DATASET_NAME,
    NNUNET_PREPROCESSED_DIR,
    NNUNET_RAW_DIR,
    NNUNET_RESULTS_DIR,
    SLICER_MODELS_DIR,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export only the official nnU-Net model artifacts for SlicerNNUnet."
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--results-root", type=Path, default=NNUNET_RESULTS_DIR)
    parser.add_argument("--raw-root", type=Path, default=NNUNET_RAW_DIR)
    parser.add_argument("--preprocessed-root", type=Path, default=NNUNET_PREPROCESSED_DIR)
    parser.add_argument("--output-root", type=Path, default=SLICER_MODELS_DIR)
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--configuration", default=CONFIGURATION, choices=["3d_fullres"])
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    args = parser.parse_args()

    config_name = f"{args.trainer}__{args.plans}__{args.configuration}"
    source_config = args.results_root / args.dataset / config_name
    source_dataset = args.raw_root / args.dataset
    output_dataset = args.output_root / args.dataset
    output_config = output_dataset / config_name
    folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    dataset_json = source_config / "dataset.json"
    if not dataset_json.exists():
        dataset_json = source_dataset / "dataset.json"
    plans_json = source_config / "plans.json"
    if not plans_json.exists():
        plans_json = source_config / "nnUNetPlans.json"
    if not plans_json.exists():
        plans_json = args.preprocessed_root / args.dataset / "nnUNetPlans.json"
    missing = []
    if not dataset_json.exists():
        missing.append(str(source_dataset / "dataset.json"))
    if not plans_json.exists():
        missing.append("plans.json (from the trained result or preprocessed dataset)")
    checkpoint_paths = [source_config / f"fold_{fold}" / args.checkpoint for fold in folds]
    missing.extend(str(path) for path in checkpoint_paths if not path.exists())
    if missing:
        print("Slicer model export blocked; missing:")
        print("\n".join(missing))
        return 2

    output_config.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_json, output_config / "dataset.json")
    shutil.copy2(plans_json, output_config / "plans.json")
    for fold, checkpoint in zip(folds, checkpoint_paths):
        target_fold = output_config / f"fold_{fold}"
        target_fold.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target_fold / args.checkpoint)

    report = {
        "dataset": args.dataset,
        "configuration": config_name,
        "model_path_for_slicer": str(output_config.resolve()),
        "folds": folds,
        "checkpoint": args.checkpoint,
        "dataset_json": "dataset.json",
        "plans_json": "plans.json",
        "note": "Select the configuration folder in SlicerNNUnet; no medical images or labels are exported.",
    }
    report_path = output_dataset / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
