from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from tmj_condyle.config import (
    CONFIGURATION,
    DATASET_NAME,
    FILE_ENDING,
    NNUNET_PREPROCESSED_DIR,
    NNUNET_RAW_DIR,
    NNUNET_RESULTS_DIR,
    PREDICTIONS_DIR,
    nifti_case_id,
)
from tmj_condyle.labels.qc import validate_label_array
from tmj_condyle.utils.geometry import geometry_differences
from tmj_condyle.utils.io import read_image, require_simpleitk, write_clean_image
from tmj_condyle.runtime import nnunet_command, runtime_environment


def _predict_command() -> list[str]:
    return nnunet_command("nnUNetv2_predict", python_executable=sys.executable, app_root=PROJECT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict one new MRI with the official nnU-Net v2 five-fold ensemble."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--configuration", default=CONFIGURATION, choices=["3d_fullres"])
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()

    image = read_image(args.input)
    case_id = nifti_case_id(args.input)
    destination = (args.output or (PREDICTIONS_DIR / f"{case_id}_condyle.nii.gz")).resolve()
    work_root = PREDICTIONS_DIR / "_single_case" / case_id
    input_dir = work_root / "input"
    output_dir = work_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_input = input_dir / f"{case_id}_0000{FILE_ENDING}"
    write_clean_image(image, clean_input)

    env = runtime_environment(
        app_root=PROJECT_ROOT,
        data_root=NNUNET_RAW_DIR.parent,
        base=os.environ,
        include_pythonpath=True,
    )
    command = [
        *_predict_command(),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        args.dataset,
        "-c",
        args.configuration,
        "-f",
        "0",
        "1",
        "2",
        "3",
        "4",
        "-tr",
        args.trainer,
        "-p",
        args.plans,
        "-device",
        args.device,
    ]
    print("Using full five-fold ensemble.")
    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    if completed.returncode != 0:
        return completed.returncode or 2
    predicted = output_dir / f"{case_id}.nii.gz"
    if not predicted.exists():
        print(f"Prediction finished but output is missing: {predicted}")
        return 2
    sitk = require_simpleitk()
    prediction = sitk.ReadImage(str(predicted))
    differences = geometry_differences(image, prediction)
    if differences:
        print(f"Prediction geometry mismatch: {differences}")
        return 2
    qc = validate_label_array(sitk.GetArrayFromImage(prediction), allow_empty=True)
    if qc["errors"]:
        print(f"Prediction label QC failed: {qc['errors']}")
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_clean_image(prediction, destination)
    print(f"Saved binary condyle mask: {destination}")
    if not args.keep_workdir:
        shutil.rmtree(work_root)
        print(f"Removed generated intermediate directory: {work_root}")
    else:
        print(f"Intermediate files retained for reproducibility: {work_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
