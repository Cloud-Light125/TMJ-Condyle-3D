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
from tmj_condyle.runtime import nnunet_command, runtime_environment


def _predict_command() -> list[str]:
    return nnunet_command("nnUNetv2_predict", python_executable=sys.executable, app_root=PROJECT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate true out-of-fold predictions: each case uses only its validation fold."
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--configuration", default=CONFIGURATION, choices=["3d_fullres"])
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--predictions-root", type=Path, default=PREDICTIONS_DIR / "oof")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_path = NNUNET_PREPROCESSED_DIR / args.dataset / "splits_final.json"
    raw_images = NNUNET_RAW_DIR / args.dataset / "imagesTr"
    if not split_path.exists() or not raw_images.exists():
        print("OOF prediction blocked: build the nnU-Net dataset and run preprocessing first.")
        return 2
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    env = runtime_environment(
        app_root=PROJECT_ROOT,
        data_root=NNUNET_RAW_DIR.parent,
        base=os.environ,
        include_pythonpath=True,
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
