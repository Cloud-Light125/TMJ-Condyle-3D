"""Run an isolated synthetic TMJ MRI nnU-Net v2 smoke experiment.

This script deliberately owns every generated path below
``workspace/test_only_*``. It never reads the formal manifest and it never
sets the formal nnU-Net environment variables for another process. The
training, prediction and evaluation commands are the real nnU-Net v2 CLI
commands; only the trainer length is reduced for a test-only run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT

from tmj_condyle.data.manifest import write_manifest
from tmj_condyle.data.nnunet import build_dataset
from tmj_condyle.evaluation.figures import generate_case_figures
from tmj_condyle.labels.qc import validate_label_array
from tmj_condyle.synthetic import SyntheticCaseSpec, write_synthetic_case
from tmj_condyle.utils.geometry import geometry_differences
from tmj_condyle.utils.io import read_image, require_simpleitk, write_clean_image
from tmj_condyle.runtime import nnunet_command, runtime_environment


DATASET_ID = 999
DATASET_NAME = "Dataset999_TMJTestOnly"
CONFIGURATION = "3d_fullres"
TRAINER = "nnUNetTrainer_TMJTestOnly_1epoch"
PLANS = "nnUNetPlans"
DEFAULT_ROOT = Path(os.environ.get("TMJ_USER_DATA_DIR") or (Path.home() / "Documents" / "TMJ-Condyle-3D")) / "test_only_tmj_synthetic"


def _parse_tuple(value: str, *, size: int, cast: Any) -> tuple[Any, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != size:
        raise argparse.ArgumentTypeError(
            f"expected {size} comma-separated values, got {len(parts)}"
        )
    try:
        return tuple(cast(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid tuple: {value}") from exc


def _safe_test_root(value: Path) -> Path:
    root = value.resolve()
    workspace = Path(os.environ.get("TMJ_USER_DATA_DIR") or DEFAULT_ROOT.parent).resolve()
    try:
        root.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(
            f"test-only output must be below {workspace}; received {root}"
        ) from exc
    if root == workspace or not root.name.lower().startswith("test_only_"):
        raise ValueError(
            "test-only output directory must have a name beginning with 'test_only_'"
        )
    formal_names = {
        "raw",
        "nifti",
        "labels",
        "predictions",
        "reports",
        "nnunet_raw",
        "nnunet_preprocessed",
        "nnunet_results",
        "slicer_models",
    }
    if root.name.lower() in formal_names:
        raise ValueError(f"refusing to use a formal project directory: {root}")
    return root


def _command(executable: str) -> list[str]:
    return nnunet_command(executable, python_executable=sys.executable, app_root=PROJECT_ROOT)


def _command_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def _run_streaming(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    label: str,
) -> float:
    """Run one real CLI command while preserving a test-only log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[{label}] {_command_text(command)}", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {_command_text(command)}\n")
        log.write(f"CWD: {PROJECT_ROOT}\n")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    elapsed = time.perf_counter() - start
    if return_code != 0:
        raise RuntimeError(f"{label} failed with exit code {return_code}; see {log_path}")
    print(f"[{label}] PASS ({elapsed:.1f}s)", flush=True)
    return elapsed


def _runtime_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA was requested but is unavailable")
            except ImportError as exc:
                raise RuntimeError("CUDA was requested but PyTorch is unavailable") from exc
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _environment(root: Path, *, device: str, trainer_path: Path) -> dict[str, str]:
    env = runtime_environment(
        app_root=PROJECT_ROOT,
        data_root=root,
        base=os.environ,
        include_pythonpath=True,
    )
    env.update(
        {
            "nnUNet_raw": str((root / "nnUNet_raw").resolve()),
            "nnUNet_preprocessed": str((root / "nnUNet_preprocessed").resolve()),
            "nnUNet_results": str((root / "nnUNet_results").resolve()),
            "nnUNet_extTrainer": str(trainer_path.resolve()),
            "TMJ_TEST_ONLY": "1",
            # nnU-Net's planner also feeds this value to torch.set_num_threads,
            # which requires a positive integer. One worker keeps this run
            # small while remaining compatible with the official planner.
            "nnUNet_n_proc_DA": "1",
            "nnUNet_def_n_proc": "1",
            "nnUNet_compile": "0",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _make_specs(
    *,
    n_cases: int,
    seed: int,
    shape_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    shape_mode: str,
) -> tuple[list[SyntheticCaseSpec], SyntheticCaseSpec]:
    nz, ny, nx = shape_zyx
    sx, sy, sz = spacing_xyz
    extent = ((nx - 1) * sx, (ny - 1) * sy, (nz - 1) * sz)
    base_center = (extent[0] * 0.50, extent[1] * 0.50, extent[2] * 0.38)
    specs: list[SyntheticCaseSpec] = []
    for index in range(n_cases):
        case_rng = np.random.default_rng(seed + 1009 * (index + 1))
        center = (
            base_center[0] + float(case_rng.uniform(-2.0, 2.0)),
            base_center[1] + float(case_rng.uniform(-2.0, 2.0)),
            base_center[2] + float(case_rng.uniform(-1.0, 1.0)),
        )
        axes = (
            float(case_rng.uniform(5.0, 6.3)),
            float(case_rng.uniform(7.0, 9.0)),
            float(case_rng.uniform(5.0, 6.4)),
        )
        rotation = (
            float(case_rng.uniform(-8.0, 8.0)),
            float(case_rng.uniform(-8.0, 8.0)),
            float(case_rng.uniform(-20.0, 20.0)),
        )
        specs.append(
            SyntheticCaseSpec(
                case_id=f"syn_{index + 1:03d}",
                seed=seed + 1009 * (index + 1),
                shape_zyx=shape_zyx,
                spacing_xyz=spacing_xyz,
                center_xyz_mm=center,
                axes_xyz_mm=axes,
                rotation_xyz_deg=rotation,
                shape_mode=shape_mode,
                neck_scale_xyz=(
                    float(case_rng.uniform(0.56, 0.68)),
                    float(case_rng.uniform(0.60, 0.72)),
                    float(case_rng.uniform(0.80, 0.95)),
                ),
                neck_offset_local_z_mm=float(case_rng.uniform(3.2, 4.6)),
                intensity_scale=float(case_rng.uniform(0.85, 1.15)),
            )
        )

    new_rng = np.random.default_rng(seed + 100_003)
    new_spec = SyntheticCaseSpec(
        case_id="new_case_001",
        seed=seed + 100_003,
        shape_zyx=shape_zyx,
        spacing_xyz=spacing_xyz,
        center_xyz_mm=(
            base_center[0] + float(new_rng.uniform(-1.5, 1.5)),
            base_center[1] + float(new_rng.uniform(-1.5, 1.5)),
            base_center[2] + float(new_rng.uniform(-0.8, 0.8)),
        ),
        axes_xyz_mm=(5.8, 8.5, 5.7),
        rotation_xyz_deg=(5.0, -4.0, 13.0),
        shape_mode=shape_mode,
        neck_scale_xyz=(0.62, 0.66, 0.90),
        neck_offset_local_z_mm=4.1,
        intensity_scale=1.0,
    )
    return specs, new_spec


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


def _prepare_synthetic_dataset(
    root: Path,
    *,
    n_cases: int,
    seed: int,
    shape_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    shape_mode: str,
) -> dict[str, Any]:
    source_images = root / "synthetic_source" / "images"
    source_labels = root / "synthetic_source" / "labels"
    metadata_dir = root / "metadata"
    specs, new_spec = _make_specs(
        n_cases=n_cases,
        seed=seed,
        shape_zyx=shape_zyx,
        spacing_xyz=spacing_xyz,
        shape_mode=shape_mode,
    )
    all_specs = [*specs, new_spec]
    for spec in all_specs:
        write_synthetic_case(
            spec,
            source_images / f"{spec.case_id}.nii.gz",
            source_labels / f"{spec.case_id}.nii.gz",
        )

    rows = [
        {
            "case_id": spec.case_id,
            "group_id": spec.case_id,
            "side": "synthetic",
            "image_path": str((source_images / f"{spec.case_id}.nii.gz").resolve()),
            "label_path": str((source_labels / f"{spec.case_id}.nii.gz").resolve()),
            "annotation_status": "VERIFIED",
            "geometry_valid": "true",
            "label_valid": "true",
            "notes": "SYNTHETIC TEST-ONLY; forbidden for formal training/paper results",
        }
        for spec in specs
    ]
    manifest_path = metadata_dir / "synthetic_manifest.csv"
    write_manifest(rows, manifest_path)
    _write_json(
        metadata_dir / "case_parameters.json",
        {
            "purpose": "TEST_ONLY_SYNTHETIC_ONLY",
            "warning": "Not a clinical dataset and not a formal experiment result.",
            "shape_mode": shape_mode,
            "train_cases": [spec.as_dict() for spec in specs],
            "new_case": new_spec.as_dict(),
        },
    )

    raw_root = root / "nnUNet_raw"
    preprocessed_root = root / "nnUNet_preprocessed"
    reports_root = root / "reports"
    raw_dataset, split_path, splits = build_dataset(
        rows,
        nnunet_raw=raw_root,
        nnunet_preprocessed=preprocessed_root,
        reports_dir=reports_root,
        dataset_name=DATASET_NAME,
        dataset_id=DATASET_ID,
        n_splits=5,
        seed=seed,
        allowed_statuses={"VERIFIED"},
    )
    new_image = source_images / f"{new_spec.case_id}.nii.gz"
    new_label = source_labels / f"{new_spec.case_id}.nii.gz"
    images_ts = raw_dataset / "imagesTs"
    images_ts.mkdir(parents=True, exist_ok=True)
    write_clean_image(read_image(new_image), images_ts / f"{new_spec.case_id}_0000.nii.gz")

    _write_json(
        root / "run_config.json",
        {
            "purpose": "TEST_ONLY_SYNTHETIC_ONLY",
            "warning": "All data, checkpoints, predictions, metrics and figures in this root are isolated test artifacts. Do not use them as paper or formal experiment results.",
            "dataset_id": DATASET_ID,
            "dataset_name": DATASET_NAME,
            "configuration": CONFIGURATION,
            "trainer": TRAINER,
            "plans": PLANS,
            "n_train_cases": n_cases,
            "n_folds": 5,
            "shape_zyx": shape_zyx,
            "spacing_xyz": spacing_xyz,
            "shape_mode": shape_mode,
            "split_path": str(split_path.resolve()),
            "formal_project_paths_used": [],
        },
    )
    return {
        "specs": specs,
        "new_spec": new_spec,
        "manifest": manifest_path,
        "raw_root": raw_root,
        "raw_dataset": raw_dataset,
        "preprocessed_root": preprocessed_root,
        "reports_root": reports_root,
        "split_path": split_path,
        "splits": splits,
        "new_image": new_image,
        "new_label": new_label,
    }


def _train_folds(root: Path, *, env: dict[str, str], device: str, logs_dir: Path) -> list[dict[str, Any]]:
    executable = _command("nnUNetv2_train")
    config_dir = root / "nnUNet_results" / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIGURATION}"
    records: list[dict[str, Any]] = []
    for fold in range(5):
        fold_dir = config_dir / f"fold_{fold}"
        final_checkpoint = fold_dir / "checkpoint_final.pth"
        if final_checkpoint.exists():
            log_path = logs_dir / f"train_fold_{fold}.log"
            prior_run_passed = False
            if log_path.exists():
                prior_log = log_path.read_text(encoding="utf-8", errors="replace")
                prior_run_passed = "Training done." in prior_log and "Validation complete" in prior_log
            status = (
                "PASS_REUSED_AFTER_REAL_RUN"
                if prior_run_passed
                else "REUSED_EXISTING_TEST_ONLY_CHECKPOINT"
            )
            print(f"[train fold {fold}] existing test-only checkpoint found; reusing it")
            records.append(
                {
                    "fold": fold,
                    "status": status,
                    "checkpoint": str(final_checkpoint.resolve()),
                    "training_log": str(log_path.resolve()),
                    "validation_summary": str((fold_dir / "validation" / "summary.json").resolve()),
                }
            )
            continue
        command = [
            *executable,
            DATASET_NAME,
            CONFIGURATION,
            str(fold),
            "-tr",
            TRAINER,
            "-p",
            PLANS,
            "--npz",
            "-device",
            device,
        ]
        elapsed = _run_streaming(
            command,
            env=env,
            log_path=logs_dir / f"train_fold_{fold}.log",
            label=f"train fold {fold}",
        )
        if not final_checkpoint.exists():
            raise RuntimeError(f"fold {fold} finished without {final_checkpoint}")
        records.append(
            {
                "fold": fold,
                "status": "PASS",
                "seconds": elapsed,
                "checkpoint": str(final_checkpoint.resolve()),
                "validation_summary": str((fold_dir / "validation" / "summary.json").resolve()),
            }
        )
    return records


def _run_oof_predictions(
    root: Path,
    *,
    splits: list[dict[str, list[str]]],
    env: dict[str, str],
    device: str,
    logs_dir: Path,
) -> Path:
    executable = _command("nnUNetv2_predict")
    raw_images = root / "nnUNet_raw" / DATASET_NAME / "imagesTr"
    predictions_root = root / "predictions" / "oof"
    for fold, split in enumerate(splits):
        input_dir = root / "predictions" / "oof_inputs" / f"fold_{fold}"
        output_dir = predictions_root / f"fold_{fold}"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        expected = []
        for case_id in split["val"]:
            source = raw_images / f"{case_id}_0000.nii.gz"
            destination = input_dir / source.name
            if not source.exists():
                raise RuntimeError(f"missing OOF source image: {source}")
            shutil.copy2(source, destination)
            expected.append(output_dir / f"{case_id}.nii.gz")
        if expected and all(path.exists() for path in expected):
            print(f"[OOF fold {fold}] existing predictions found; reusing them")
            continue
        command = [
            *executable,
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-d",
            str(DATASET_ID),
            "-c",
            CONFIGURATION,
            "-f",
            str(fold),
            "-tr",
            TRAINER,
            "-p",
            PLANS,
            "-device",
            device,
            "-step_size",
            "1.0",
            "--disable_tta",
            "-npp",
            "0",
            "-nps",
            "0",
            "--disable_progress_bar",
        ]
        _run_streaming(
            command,
            env=env,
            log_path=logs_dir / f"oof_fold_{fold}.log",
            label=f"OOF prediction fold {fold}",
        )
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise RuntimeError(f"OOF fold {fold} did not produce: {missing}")
    return predictions_root


def _evaluate_oof(
    root: Path,
    *,
    manifest: Path,
    split_path: Path,
    predictions_root: Path,
    reports_root: Path,
    env: dict[str, str],
    logs_dir: Path,
) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_cv.py"),
        "--manifest",
        str(manifest),
        "--splits",
        str(split_path),
        "--predictions-root",
        str(predictions_root),
        "--report-dir",
        str(reports_root),
        "--dataset-name",
        DATASET_NAME,
        "--make-figures",
    ]
    _run_streaming(
        command,
        env=env,
        log_path=logs_dir / "evaluate_oof.log",
        label="OOF evaluation + figures",
    )


def _run_new_case_inference(
    root: Path,
    *,
    new_image: Path,
    new_label: Path,
    env: dict[str, str],
    device: str,
    reports_root: Path,
    logs_dir: Path,
) -> Path:
    executable = _command("nnUNetv2_predict")
    input_dir = root / "predictions" / "new_case" / "input"
    output_dir = root / "predictions" / "new_case" / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "new_case_001_0000.nii.gz"
    shutil.copy2(new_image, input_path)
    predicted_path = output_dir / "new_case_001.nii.gz"
    if not predicted_path.exists():
        command = [
            *executable,
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-d",
            str(DATASET_ID),
            "-c",
            CONFIGURATION,
            "-f",
            "0",
            "1",
            "2",
            "3",
            "4",
            "-tr",
            TRAINER,
            "-p",
            PLANS,
            "-device",
            device,
            "-step_size",
            "1.0",
            "--disable_tta",
            "-npp",
            "0",
            "-nps",
            "0",
            "--disable_progress_bar",
        ]
        _run_streaming(
            command,
            env=env,
            log_path=logs_dir / "new_case_inference.log",
            label="new case five-fold inference",
        )
    else:
        print("[new case inference] existing test-only prediction found; reusing it")

    sitk = require_simpleitk()
    image = read_image(new_image)
    prediction = sitk.ReadImage(str(predicted_path))
    differences = geometry_differences(image, prediction)
    if differences:
        raise RuntimeError(f"new case prediction geometry mismatch: {differences}")
    qc = validate_label_array(sitk.GetArrayFromImage(prediction), allow_empty=True)
    if qc["errors"]:
        raise RuntimeError(f"new case prediction label QC failed: {qc['errors']}")
    final_path = root / "predictions" / "new_case_001_condyle.nii.gz"
    write_clean_image(prediction, final_path)
    figure_dir = reports_root / "new_case_inference"
    generate_case_figures(
        image=image,
        ground_truth=read_image(new_label),
        prediction=prediction,
        output_dir=figure_dir,
    )
    _write_json(
        reports_root / "new_case_inference.json",
        {
            "purpose": "TEST_ONLY_SYNTHETIC_ONLY",
            "case_id": "new_case_001",
            "prediction": str(final_path.resolve()),
            "geometry_differences": differences,
            "prediction_qc": qc,
            "prediction_foreground_voxels": int((sitk.GetArrayFromImage(prediction) == 1).sum()),
            "ground_truth_available_for_visual_qc_only": True,
        },
    )
    return final_path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_final_summary(
    root: Path,
    *,
    device: str,
    training_records: list[dict[str, Any]],
    manifest: Path,
    split_path: Path,
    predictions_root: Path,
    reports_root: Path,
    new_prediction: Path,
) -> Path:
    per_case_path = reports_root / "metrics_per_case.csv"
    summary_path = reports_root / "metrics_summary.csv"
    per_case = _read_csv_rows(per_case_path)
    summary = _read_csv_rows(summary_path)
    payload = {
        "status": "TEST_ONLY_COMPLETE",
        "warning": "Synthetic data, checkpoints, metrics and figures are isolated debug artifacts. They must not be used as paper or formal experiment results.",
        "device": device,
        "dataset": DATASET_NAME,
        "configuration": CONFIGURATION,
        "trainer": TRAINER,
        "training": {
            "folds": training_records,
            "epochs": 1,
            "train_iterations_per_epoch": 2,
            "validation_iterations_per_epoch": 1,
        },
        "oof": {
            "cases": len(per_case),
            "metrics_summary": summary,
            "manifest": str(manifest.resolve()),
            "splits": str(split_path.resolve()),
            "predictions_root": str(predictions_root.resolve()),
            "report_root": str(reports_root.resolve()),
        },
        "new_case_inference": {
            "case_id": "new_case_001",
            "prediction": str(new_prediction.resolve()),
            "visual_qc_dir": str((reports_root / "new_case_inference").resolve()),
        },
        "formal_project_paths_used": [],
    }
    destination = root / "TEST_ONLY_RUN_SUMMARY.json"
    _write_json(destination, payload)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic TMJ MRI and run isolated real nnU-Net v2 five-fold smoke validation."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--shape", default="32,48,48", help="z,y,x voxel shape")
    parser.add_argument("--spacing", default="0.8,0.8,1.0", help="x,y,z spacing in mm")
    parser.add_argument("--shape-mode", choices=("condyle", "ellipsoid"), default="condyle")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    try:
        root = _safe_test_root(args.root)
        if args.cases < 5:
            raise ValueError("at least 5 synthetic training cases are required for five folds")
        shape_zyx = _parse_tuple(args.shape, size=3, cast=int)
        spacing_xyz = _parse_tuple(args.spacing, size=3, cast=float)
        if any(value < 16 for value in shape_zyx):
            raise ValueError("each synthetic volume dimension must be at least 16 voxels")
        if any(value <= 0 for value in spacing_xyz):
            raise ValueError("synthetic spacing values must be positive")
        device = _runtime_device(args.device)
        root.mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "reports").mkdir(parents=True, exist_ok=True)
        (root / "TEST_ONLY_README.md").write_text(
            "# TEST-ONLY synthetic TMJ MRI run\n\n"
            "This directory is intentionally isolated from the formal project.\n\n"
            "- The MRI volumes and masks are synthetic.\n"
            "- The trainer is one epoch with two training iterations per epoch.\n"
            "- Checkpoints, OOF predictions, new-case inference, metrics and figures are debug artifacts only.\n"
            "- Do not use any file here as a paper result or formal experiment result.\n",
            encoding="utf-8",
        )
        trainer_path = PROJECT_ROOT / "tmj_condyle" / "test_only_trainers"
        env = _environment(root, device=device, trainer_path=trainer_path)

        print(f"TEST-ONLY root: {root}")
        print(f"Device: {device}")
        print(f"Dataset: {DATASET_NAME} ({args.cases} CV cases, 5 folds)")
        print(f"Trainer: {TRAINER} (1 epoch x 2 train iterations)")
        print("Formal project paths used: []")

        dataset = _prepare_synthetic_dataset(
            root,
            n_cases=args.cases,
            seed=args.seed,
            shape_zyx=shape_zyx,
            spacing_xyz=spacing_xyz,
            shape_mode=args.shape_mode,
        )
        plan_command = [
            *_command("nnUNetv2_plan_and_preprocess"),
            "-d",
            str(DATASET_ID),
            "--verify_dataset_integrity",
            "--clean",
            "-c",
            CONFIGURATION,
            "-npfp",
            "1",
            "-np",
            "1",
            "--no_pbar",
        ]
        _run_streaming(
            plan_command,
            env=env,
            log_path=root / "logs" / "plan_and_preprocess.log",
            label="nnU-Net plan + preprocess",
        )
        plans_path = root / "nnUNet_preprocessed" / DATASET_NAME / f"{PLANS}.json"
        if not plans_path.exists():
            raise RuntimeError(f"nnU-Net planner did not create {plans_path}")

        training_records = _train_folds(
            root,
            env=env,
            device=device,
            logs_dir=root / "logs",
        )
        predictions_root = _run_oof_predictions(
            root,
            splits=dataset["splits"],
            env=env,
            device=device,
            logs_dir=root / "logs",
        )
        _evaluate_oof(
            root,
            manifest=dataset["manifest"],
            split_path=dataset["split_path"],
            predictions_root=predictions_root,
            reports_root=dataset["reports_root"],
            env=env,
            logs_dir=root / "logs",
        )
        new_prediction = _run_new_case_inference(
            root,
            new_image=dataset["new_image"],
            new_label=dataset["new_label"],
            env=env,
            device=device,
            reports_root=dataset["reports_root"],
            logs_dir=root / "logs",
        )
        summary_path = _write_final_summary(
            root,
            device=device,
            training_records=training_records,
            manifest=dataset["manifest"],
            split_path=dataset["split_path"],
            predictions_root=predictions_root,
            reports_root=dataset["reports_root"],
            new_prediction=new_prediction,
        )
        print("\nTEST_ONLY_PIPELINE_COMPLETE")
        print(f"Summary: {summary_path.resolve()}")
        print(f"Metrics: {(dataset['reports_root'] / 'metrics_summary.csv').resolve()}")
        print(f"New inference: {new_prediction.resolve()}")
        print(f"Figures: {(dataset['reports_root'] / 'figures').resolve()}")
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"TEST_ONLY_PIPELINE_FAILED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
