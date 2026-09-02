from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from tmj_condyle.config import (
    LABELS_DIR,
    MANIFEST_PATH,
    NIFTI_DIR,
    NNUNET_PREPROCESSED_DIR,
    NNUNET_RAW_DIR,
    NNUNET_RESULTS_DIR,
    REPORTS_DIR,
)
from tmj_condyle.data.manifest import read_manifest
from tmj_condyle.data.validation import nifti_files, validate_manifest_dataset


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def _entrypoint(name: str) -> str | None:
    local_dir = Path(sys.executable).resolve().parent
    for candidate in (local_dir / f"{name}.exe", local_dir / name, local_dir / f"{name}.bat"):
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def _import_version(module_name: str, package_name: str | None = None) -> tuple[str, object | None]:
    try:
        module = importlib.import_module(module_name)
        return _version(package_name or module_name), module
    except Exception as exc:  # noqa: BLE001 - environment report
        return f"ERROR: {type(exc).__name__}: {exc}", None


def _cuda_report(torch_module: object | None) -> dict[str, object]:
    if torch_module is None:
        return {"available": False, "status": "torch not installed"}
    torch = torch_module
    report: dict[str, object] = {"available": False}
    try:
        report["available"] = bool(torch.cuda.is_available())
        report["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        report["arch_list"] = list(torch.cuda.get_arch_list())
        if not report["available"]:
            report["status"] = "CUDA unavailable"
            return report
        device = torch.cuda.current_device()
        report["device_index"] = int(device)
        report["device_name"] = torch.cuda.get_device_name(device)
        report["capability"] = list(torch.cuda.get_device_capability(device))
        left = torch.ones((4, 4), device="cuda")
        right = torch.eye(4, device="cuda")
        result = left @ right
        torch.cuda.synchronize()
        report["kernel_smoke"] = bool(float(result.sum().item()) == 16.0)
        report["status"] = "PASS" if report["kernel_smoke"] else "FAIL"
    except Exception as exc:  # noqa: BLE001 - catch incompatible kernel/runtime
        report["status"] = f"FAIL: {type(exc).__name__}: {exc}"
    return report


def _write_lock(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    path.write_text(
        "# Generated from the active project environment by check_environment.py\n"
        "# This file records the environment that was actually inspected.\n"
        + result.stdout,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command TMJ-Condyle-3D environment/QC check.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()

    print("TMJ-Condyle-3D environment check")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Project: {PROJECT_ROOT}")

    numpy_version, _ = _import_version("numpy")
    sitk_version, _ = _import_version("SimpleITK", "SimpleITK")
    nibabel_version, _ = _import_version("nibabel")
    scipy_version, _ = _import_version("scipy")
    torch_version, torch = _import_version("torch")
    nnunet_version, _ = _import_version("nnunetv2", "nnunetv2")
    print(f"NumPy: {numpy_version}")
    print(f"SimpleITK: {sitk_version}")
    print(f"nibabel: {nibabel_version}")
    print(f"SciPy: {scipy_version}")
    print(f"PyTorch: {torch_version}")
    print(f"nnUNet v2: {nnunet_version}")
    print(f"nnUNetv2_plan_and_preprocess: {_entrypoint('nnUNetv2_plan_and_preprocess') or 'NOT FOUND'}")
    print(f"nnUNetv2_train: {_entrypoint('nnUNetv2_train') or 'NOT FOUND'}")
    print(f"nnUNetv2_predict: {_entrypoint('nnUNetv2_predict') or 'NOT FOUND'}")
    cuda = _cuda_report(torch)
    print(f"CUDA: {json.dumps(cuda, ensure_ascii=False)}")

    for name, path in (
        ("images", NIFTI_DIR),
        ("labels", LABELS_DIR),
        ("nnUNet_raw", NNUNET_RAW_DIR),
        ("nnUNet_preprocessed", NNUNET_PREPROCESSED_DIR),
        ("nnUNet_results", NNUNET_RESULTS_DIR),
        ("reports", REPORTS_DIR),
    ):
        print(f"{name} directory: {path} ({'EXISTS' if path.exists() else 'MISSING'})")
    image_count = len(nifti_files(NIFTI_DIR))
    label_count = len(nifti_files(LABELS_DIR))
    print(f"Number of images: {image_count}")
    print(f"Number of labels: {label_count}")

    validation_rows, validation_ok = ([], False)
    if sitk_version != "NOT INSTALLED" and args.manifest.exists():
        validation_rows, validation_ok = validate_manifest_dataset(
            manifest_path=args.manifest,
            images_dir=NIFTI_DIR,
            labels_dir=LABELS_DIR,
            report_dir=REPORTS_DIR,
        )
    ready_cases = [
        row for row in validation_rows
        if row["status"] == "PASS" and row.get("annotation_status") in {"ANNOTATED", "VERIFIED"}
    ]
    groups = {str(row.get("group_id") or row["case_id"]) for row in ready_cases}
    missing_labels = [
        row.get("case_id")
        for row in validation_rows
        if "missing label" in str(row.get("errors", ""))
    ]
    geometry_errors = [
        row.get("case_id")
        for row in validation_rows
        if "geometry" in str(row.get("errors", "")).lower()
    ]
    print(f"Ready cases: {len(ready_cases)}")
    print(f"Missing labels: {missing_labels or 'none'}")
    print(f"Geometry errors: {geometry_errors or 'none'}")
    annotation_ready = image_count > 0 and sitk_version != "NOT INSTALLED"
    env_ready = (
        sitk_version != "NOT INSTALLED"
        and scipy_version != "NOT INSTALLED"
        and torch_version != "NOT INSTALLED"
        and nnunet_version != "NOT INSTALLED"
    )
    gpu_ready = cuda.get("status") == "PASS"
    training_ready = validation_ok and len(ready_cases) >= 5 and len(groups) >= 5 and env_ready and gpu_ready
    print("READY FOR ANNOTATION" if annotation_ready else "NOT READY FOR ANNOTATION")
    if training_ready:
        print("READY FOR TRAINING")
    elif env_ready and not gpu_ready:
        print("FULL TRAINING BLOCKED BY GPU")
        print("NOT READY")
    else:
        print("NOT READY")

    if args.write_lock:
        lock = PROJECT_ROOT / "requirements-lock.txt"
        _write_lock(lock)
        print(f"requirements-lock.txt written: {lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
