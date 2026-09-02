from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from tmj_condyle.config import (
    CONFIGURATION,
    DATASET_NAME,
    NNUNET_PREPROCESSED_DIR,
    NNUNET_RAW_DIR,
    NNUNET_RESULTS_DIR,
    REPORTS_DIR,
)


def _command(executable: str) -> list[str]:
    local_candidates = [
        Path(sys.executable).resolve().parent / f"{executable}.exe",
        Path(sys.executable).resolve().parent / executable,
        Path(sys.executable).resolve().parent / f"{executable}.bat",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return [str(candidate)]
    found = shutil.which(executable)
    if found:
        return [found]
    modules = {
        "nnUNetv2_train": "nnunetv2.run.run_training",
    }
    module = modules.get(executable)
    if module is None:
        raise RuntimeError(f"No Python-module fallback is defined for {executable}")
    return [sys.executable, "-m", module]


def _metric_from_json(value):
    if isinstance(value, dict):
        for key in ("foreground_mean", "Dice", "dice", "mean"):
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
        for child in value.values():
            found = _metric_from_json(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _metric_from_json(child)
            if found is not None:
                return found
    return None


def _fold_metric(path: Path) -> float | None:
    for candidate in (path / "validation" / "summary.json", path / "summary.json"):
        if candidate.exists():
            try:
                return _metric_from_json(json.loads(candidate.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def _runtime_device_info(device: str) -> str:
    if device != "cuda":
        return device
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda unavailable"
        index = torch.cuda.current_device()
        return f"cuda:{index} {torch.cuda.get_device_name(index)}"
    except Exception as exc:  # noqa: BLE001 - summary should remain best effort
        return f"cuda check error: {type(exc).__name__}"


def _checkpoint_state(path: Path) -> str:
    states = []
    for name in ("checkpoint_best.pth", "checkpoint_final.pth"):
        if (path / name).exists():
            states.append(name.removesuffix(".pth"))
    return "+".join(states) if states else "none"


def _write_summary(records: list[dict[str, object]], path: Path, device_info: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training summary",
        "",
        "Only the official nnU-Net v2 3d_fullres configuration is used.",
        "",
        f"- Runtime device: {device_info}",
        "",
        "| Fold | Status | Checkpoints | Seconds | Validation Dice (if available) |",
        "|---:|---|---|---:|---:|",
    ]
    for record in records:
        metric = record.get("validation_dice")
        metric_text = "" if metric is None else f"{float(metric):.6f}"
        lines.append(
            f"| {record['fold']} | {record['status']} | {record.get('checkpoints', 'none')} | "
            f"{record.get('seconds', 0):.1f} | {metric_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official nnU-Net v2 3d_fullres folds 0-4.")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--configuration", default=CONFIGURATION, choices=["3d_fullres"])
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--resume", action="store_true", help="Continue incomplete folds with --c.")
    parser.add_argument("--no-skip-completed", action="store_true")
    args = parser.parse_args()

    dataset_dir = NNUNET_RAW_DIR / args.dataset
    split_file = NNUNET_PREPROCESSED_DIR / args.dataset / "splits_final.json"
    if not (dataset_dir / "dataset.json").exists():
        print(f"Training blocked: missing {dataset_dir / 'dataset.json'}")
        return 2
    if not split_file.exists():
        print(f"Training blocked: missing grouped split file {split_file}")
        return 2
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    config_dir = NNUNET_RESULTS_DIR / args.dataset / f"{args.trainer}__{args.plans}__{args.configuration}"
    device_info = _runtime_device_info(args.device)
    if args.device == "cuda" and device_info == "cuda unavailable":
        print("FULL TRAINING BLOCKED BY GPU: CUDA is unavailable in the active environment.")
        print("Run check_environment.py and use a verified compatible CUDA/PyTorch environment.")
        return 3
    env = os.environ.copy()
    env.update(
        {
            "nnUNet_raw": str(NNUNET_RAW_DIR.resolve()),
            "nnUNet_preprocessed": str(NNUNET_PREPROCESSED_DIR.resolve()),
            "nnUNet_results": str(NNUNET_RESULTS_DIR.resolve()),
        }
    )
    executable = _command("nnUNetv2_train")
    records: list[dict[str, object]] = []
    for fold in folds:
        fold_dir = config_dir / f"fold_{fold}"
        final_checkpoint = fold_dir / "checkpoint_final.pth"
        if final_checkpoint.exists() and not args.no_skip_completed:
            record = {
                "fold": fold,
                "status": "SKIPPED_COMPLETED",
                "seconds": 0.0,
                "validation_dice": _fold_metric(fold_dir),
                "checkpoints": _checkpoint_state(fold_dir),
            }
            records.append(record)
            print(f"fold {fold}: already complete, skipped")
            continue
        command = [
            *executable,
            args.dataset,
            args.configuration,
            str(fold),
            "-tr",
            args.trainer,
            "-p",
            args.plans,
            "--npz",
            "-device",
            args.device,
        ]
        if args.resume:
            command.append("--c")
        print("Running:", " ".join(command))
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        elapsed = time.perf_counter() - start
        status = "PASS" if completed.returncode == 0 else f"FAIL({completed.returncode})"
        record = {
            "fold": fold,
            "status": status,
            "seconds": elapsed,
            "validation_dice": _fold_metric(fold_dir),
        }
        records.append(record)
        record["checkpoints"] = _checkpoint_state(fold_dir)
        _write_summary(records, REPORTS_DIR / "training_summary.md", device_info)
        if completed.returncode != 0:
            print(f"fold {fold} failed; later folds were not started")
            return completed.returncode or 2
    _write_summary(records, REPORTS_DIR / "training_summary.md", device_info)
    return 0 if all(str(record["status"]).startswith(("PASS", "SKIPPED")) for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
