"""Verify that a staged runtime imports without the source environment."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


IMPORTS = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scipy.ndimage", "scipy"),
    ("SimpleITK", "SimpleITK"),
    ("matplotlib", "matplotlib"),
    ("skimage", "scikit-image"),
    ("pandas", "pandas"),
    ("nibabel", "nibabel"),
    ("nnunetv2", "nnunetv2"),
)


def _version(package: str) -> str:
    try:
        from importlib import metadata

        return metadata.version(package)
    except Exception as exc:  # noqa: BLE001 - diagnostic output
        return f"ERROR: {type(exc).__name__}: {exc}"


def verify(app_root: Path) -> dict[str, object]:
    expected_python = (app_root / "runtime" / "python" / "python.exe").resolve()
    actual_python = Path(sys.executable).resolve()
    result: dict[str, object] = {
        "app_root": str(app_root),
        "expected_python": str(expected_python),
        "actual_python": str(actual_python),
        "python_matches_bundle": actual_python == expected_python,
        "runtime_mode": os.environ.get("TMJ_RUNTIME_MODE", ""),
        "python_home_absent": "PYTHONHOME" not in os.environ,
        "imports": {},
        "cuda": {},
        "cpu_only_expected": True,
    }
    imports: dict[str, object] = {}
    for module_name, package_name in IMPORTS:
        try:
            module = importlib.import_module(module_name)
            imports[module_name] = {"status": "PASS", "version": _version(package_name)}
        except Exception as exc:  # noqa: BLE001 - report every missing package
            imports[module_name] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
    result["imports"] = imports

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        result["cuda"] = {
            "available": cuda_available,
            "torch_version": getattr(torch, "__version__", ""),
            "compiled_cuda": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic output
        result["cuda"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    compiled_cuda = result.get("cuda", {}).get("compiled_cuda") if isinstance(result.get("cuda"), dict) else None
    result["cpu_only"] = compiled_cuda in (None, "")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.app_root.resolve())
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    required = [item[0] for item in IMPORTS]
    imports_ok = all(report["imports"].get(name, {}).get("status") == "PASS" for name in required)  # type: ignore[union-attr]
    ok = bool(report["python_matches_bundle"] and imports_ok and report["cpu_only"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
