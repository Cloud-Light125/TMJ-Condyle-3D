from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from tmj_condyle.config import (
    CONFIGURATION,
    DATASET_NAME,
    NNUNET_PREPROCESSED_DIR,
    NNUNET_RAW_DIR,
    NNUNET_RESULTS_DIR,
    PREDICTIONS_DIR,
)


def _predict_command() -> list[str]:
    local_dir = Path(sys.executable).resolve().parent
    for candidate in (
        local_dir / "nnUNetv2_predict.exe",
        local_dir / "nnUNetv2_predict",
        local_dir / "nnUNetv2_predict.bat",
    ):
        if candidate.exists():
            return [str(candidate)]
    found = shutil.which("nnUNetv2_predict")
    return [found] if found else [sys.executable, "-m", "nnunetv2.inference.predict_from_raw_data"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate true out-of-fold predictions: each case uses only its validation fold."
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--configuration", default=CONFIGURATION, choices=["3d_fullres"])
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--predictions-root", type=Path, default=PREDICTIONS_DIR / "oof")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_path = NNUNET_PREPROCESSED_DIR / args.dataset / "splits_final.json"
    raw_images = NNUNET_RAW_DIR / args.dataset / "imagesTr"
    if not split_path.exists() or not raw_images.exists():
        print("OOF prediction blocked: build the nnU-Net dataset and run preprocessing first.")
        return 2
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env.update(
        {
            "nnUNet_raw": str(NNUNET_RAW_DIR.resolve()),
            "nnUNet_preprocessed": str(NNUNET_PREPROCESSED_DIR.resolve()),
            "nnUNet_results": str(NNUNET_RESULTS_DIR.resolve()),
        }
    )
    command_base = _predict_command()
    for fold, split in enumerate(splits):
        input_dir = args.predictions_root / "_inputs" / f"fold_{fold}"
        output_dir = args.predictions_root / f"fold_{fold}"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        for case_id in split["val"]:
            source = raw_images / f"{case_id}_0000.nii.gz"
            destination = input_dir / source.name
            if not source.exists():
                print(f"Missing raw image for {case_id}: {source}")
                return 2
            if not destination.exists():
                shutil.copy2(source, destination)
        expected = [output_dir / f"{case_id}.nii.gz" for case_id in split["val"]]
        if expected and all(path.exists() for path in expected) and not args.force:
            print(f"fold {fold}: OOF predictions already present, skipped")
            continue
        command = [
            *command_base,
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-d",
            args.dataset,
            "-c",
            args.configuration,
            "-f",
            str(fold),
            "-tr",
            args.trainer,
            "-p",
            args.plans,
            "-device",
            args.device,
        ]
        print("Running:", " ".join(command))
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        if completed.returncode != 0:
            return completed.returncode or 2
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            print(f"fold {fold} completed but expected predictions are missing: {missing}")
            return 2
    print("OOF prediction generation complete: every case was predicted by its validation fold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
