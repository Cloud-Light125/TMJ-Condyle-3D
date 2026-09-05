from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
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
from tmj_condyle.runtime import nnunet_command, nnunet_module


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def _entrypoint(name: str) -> str | None:
    try:
        command = nnunet_command(name, python_executable=sys.executable, app_root=PROJECT_ROOT)
        # Prediction is intentionally emitted as ``python -c`` by the
        # runtime resolver because nnU-Net 2.8.1's module has a legacy demo
        # block under ``__main__``.  Validate the importable module rather
        # than assuming command[2] is always a module name.
        importlib.import_module(nnunet_module(name))
    except ValueError:
        return None
    except Exception:
        return None
    return " ".join(command)


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
    distributions = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name and version:
            distributions.append(f"{name}=={version}")
    path.write_text(
        "# Generated from the active project environment by check_environment.py\n"
        "# This file records the environment that was actually inspected.\n"
        + "\n".join(sorted(set(distributions), key=str.casefold))
        + "\n",
        encoding="utf-8",
    )


def collect_environment_report(manifest: Path = MANIFEST_PATH) -> dict[str, object]:
    """Collect one machine-readable report for both CLI and the Slicer GUI."""

    numpy_version, _ = _import_version("numpy")
    sitk_version, _ = _import_version("SimpleITK", "SimpleITK")
    nibabel_version, _ = _import_version("nibabel")
    scipy_version, _ = _import_version("scipy")
    skimage_version, _ = _import_version("skimage", "scikit-image")
    pandas_version, _ = _import_version("pandas")
    torch_version, torch = _import_version("torch")
    nnunet_version, _ = _import_version("nnunetv2", "nnunetv2")
    entrypoints = {
        name: _entrypoint(name)
        for name in (
            "nnUNetv2_plan_and_preprocess",
            "nnUNetv2_train",
            "nnUNetv2_predict",
        )
    }
    cuda = _cuda_report(torch)
    image_count = len(nifti_files(NIFTI_DIR))
    label_count = len(nifti_files(LABELS_DIR))

    validation_rows, validation_ok = ([], False)
    if sitk_version != "NOT INSTALLED" and manifest.exists():
        validation_rows, validation_ok = validate_manifest_dataset(
            manifest_path=manifest,
            images_dir=NIFTI_DIR,
            labels_dir=LABELS_DIR,
            report_dir=REPORTS_DIR,
            statuses={"VERIFIED"},
        )
    ready_cases = [
        row
        for row in validation_rows
        if row["status"] == "PASS"
        and row.get("annotation_status", "").upper() == "VERIFIED"
    ]
    groups = {str(row.get("group_id") or row["case_id"]) for row in ready_cases}
    manifest_rows = read_manifest(manifest)
    manual_case_count = sum(
        str(row.get("annotation_status", "")).upper() in {"ANNOTATED", "VERIFIED"}
        for row in manifest_rows
    )
    verified_case_count = sum(
        str(row.get("annotation_status", "")).upper() == "VERIFIED"
        for row in manifest_rows
    )
    env_ready = (
        sitk_version != "NOT INSTALLED"
        and scipy_version != "NOT INSTALLED"
        and torch_version != "NOT INSTALLED"
        and nnunet_version != "NOT INSTALLED"
        and entrypoints["nnUNetv2_plan_and_preprocess"] is not None
        and entrypoints["nnUNetv2_train"] is not None
        and entrypoints["nnUNetv2_predict"] is not None
    )
    gpu_ready = cuda.get("status") == "PASS"
    data_ready = validation_ok and len(ready_cases) >= 5 and len(groups) >= 5
    return {
        "platform": platform.platform(),
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "python_ready": True,
        "packages": {
            "numpy": numpy_version,
            "SimpleITK": sitk_version,
            "nibabel": nibabel_version,
            "scipy": scipy_version,
            "scikit-image": skimage_version,
            "pandas": pandas_version,
            "torch": torch_version,
            "nnunetv2": nnunet_version,
        },
        "entrypoints": entrypoints,
        "cuda": cuda,
        "directories": {
            "images": str(NIFTI_DIR),
            "labels": str(LABELS_DIR),
            "nnUNet_raw": str(NNUNET_RAW_DIR),
            "nnUNet_preprocessed": str(NNUNET_PREPROCESSED_DIR),
            "nnUNet_results": str(NNUNET_RESULTS_DIR),
            "reports": str(REPORTS_DIR),
        },
        "runtime": {
            "mode": "packaged" if (PROJECT_ROOT / "runtime").is_dir() else "source",
            "app_root": str(PROJECT_ROOT),
            "python": str(Path(sys.executable).resolve()),
            "environment": {
                "PYTHONHOME": "isolated",
                "PYTHONPATH": "application root only",
            },
        },
        "image_count": image_count,
        "label_count": label_count,
        "validated_case_count": len(validation_rows),
        "annotated_case_count": manual_case_count,
        "verified_case_count": verified_case_count,
        "trainable_case_count": len(ready_cases),
        "group_count": len(groups),
        "validation_ok": validation_ok,
        "data_ready": data_ready,
        "nnunet_ready": env_ready,
        "gpu_ready": gpu_ready,
        # Formal training defaults to CPU.  GPU is optional and is checked
        # only when the user explicitly selects it in the training page.
        "formal_training_ready": data_ready and env_ready,
        "missing_labels": [
            row.get("case_id")
            for row in validation_rows
            if "missing label" in str(row.get("errors", ""))
        ],
        "geometry_errors": [
            row.get("case_id")
            for row in validation_rows
            if "geometry" in str(row.get("errors", "")).lower()
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command TMJ-Condyle-3D environment/QC check.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--write-lock", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report for the GUI.")
    args = parser.parse_args()
    report = collect_environment_report(args.manifest)

    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        print("TMJ-Condyle-3D environment check")
        print(f"OS: {report['platform']}")
        print(f"Python: {report['python']['version']} ({report['python']['executable']})")
        for name, version in report["packages"].items():
            print(f"{name}: {version}")
        for name, path in report["entrypoints"].items():
            print(f"{name}: {path or 'NOT FOUND'}")
        print(f"CUDA: {json.dumps(report['cuda'], ensure_ascii=False)}")
        for name, path in report["directories"].items():
            print(f"{name} directory: {path} ({'EXISTS' if Path(path).exists() else 'MISSING'})")
        print(f"Number of images: {report['image_count']}")
        print(f"Number of labels: {report['label_count']}")
        print(f"Manual annotation complete: {report['annotated_case_count']}")
        print(f"Verified cases: {report['verified_case_count']}")
        print(f"Trainable cases: {report['trainable_case_count']}")
        print(f"Missing labels: {report['missing_labels'] or 'none'}")
        print(f"Geometry errors: {report['geometry_errors'] or 'none'}")
        print("READY FOR ANNOTATION" if report["image_count"] > 0 and report["packages"]["SimpleITK"] != "NOT INSTALLED" else "NOT READY FOR ANNOTATION")
        if report["formal_training_ready"]:
            print("READY FOR TRAINING")
        else:
            print("NOT READY")

    if args.write_lock:
        lock = PROJECT_ROOT / "requirements-lock.txt"
        _write_lock(lock)
        print(f"requirements-lock.txt written: {lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
