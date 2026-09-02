from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT

from tmj_condyle.config import (
    CONFIGURATION,
    DATASET_NAME,
    NNUNET_PREPROCESSED_DIR,
    REPORTS_DIR,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _display(value: Any) -> str:
    if value is None:
        return "not present"
    if isinstance(value, (dict, list, tuple)):
        return f"`{json.dumps(value, ensure_ascii=False, separators=(',', ':'))}`"
    return f"`{value}`"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _median_anisotropy(spacings: Any) -> float | None:
    if not isinstance(spacings, list) or not spacings:
        return None
    ratios: list[float] = []
    for spacing in spacings:
        if not isinstance(spacing, list) or not spacing:
            continue
        values = [float(value) for value in spacing if float(value) > 0]
        if values:
            ratios.append(max(values) / min(values))
    if not ratios:
        return None
    ratios.sort()
    return ratios[len(ratios) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the actual nnU-Net v2 fingerprint and 3d_fullres planner output."
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--configuration", default=CONFIGURATION, choices=["3d_fullres"])
    parser.add_argument("--preprocessed-root", type=Path, default=NNUNET_PREPROCESSED_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    dataset_dir = args.preprocessed_root / args.dataset
    fingerprint_path = dataset_dir / "dataset_fingerprint.json"
    plans_path = dataset_dir / "nnUNetPlans.json"
    missing = [str(path) for path in (fingerprint_path, plans_path) if not path.exists()]
    if missing:
        print("Planner report blocked; run official nnUNetv2_plan_and_preprocess first.")
        print("Missing:")
        print("\n".join(missing))
        return 2

    fingerprint = _read_json(fingerprint_path)
    plans = _read_json(plans_path)
    configurations = plans.get("configurations", {})
    configuration = configurations.get(args.configuration)
    if not isinstance(configuration, dict):
        print(f"Planner report blocked; configuration is absent: {args.configuration}")
        return 2

    fingerprint_spacings = fingerprint.get("spacings")
    fingerprint_cases = fingerprint.get("num_cases", fingerprint.get("numTraining"))
    if fingerprint_cases is None and isinstance(fingerprint_spacings, list):
        fingerprint_cases = len(fingerprint_spacings)
    rows = [
        ("Fingerprint cases", fingerprint_cases),
        ("Original median spacing", plans.get("original_median_spacing_after_transp")),
        ("Original median image size", plans.get("original_median_shape_after_transp")),
        ("Fingerprint spacings", fingerprint_spacings),
        ("Fingerprint cropped shapes", fingerprint.get("shapes_after_crop")),
        ("Median anisotropy ratio", _median_anisotropy(fingerprint_spacings)),
        ("Target spacing", configuration.get("spacing")),
        ("Patch size", configuration.get("patch_size")),
        ("Batch size", configuration.get("batch_size")),
        ("Preprocessor", configuration.get("preprocessor_name")),
        ("Normalization schemes", configuration.get("normalization_schemes")),
        ("Use mask for normalization", configuration.get("use_mask_for_norm")),
        ("Data resampling", configuration.get("resampling_fn_data")),
        ("Segmentation resampling", configuration.get("resampling_fn_seg")),
        ("Transpose forward", plans.get("transpose_forward")),
        ("Transpose backward", plans.get("transpose_backward")),
        (
            "Foreground intensity properties",
            plans.get("foreground_intensity_properties_per_channel"),
        ),
    ]
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "planner_summary.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# nnU-Net v2 planner summary\n\n")
        handle.write(
            "This report is generated from the actual official nnU-Net v2 JSON outputs; "
            "no spacing, patch size, padding, or normalization is manually substituted.\n\n"
        )
        handle.write(f"- Fingerprint: `{_relative(fingerprint_path)}`\n")
        handle.write(f"- Plans: `{_relative(plans_path)}`\n")
        handle.write(f"- Configuration: `{args.configuration}`\n\n")
        handle.write("| Planner field | Actual value |\n|---|---|\n")
        for name, value in rows:
            handle.write(f"| {name} | {_display(value)} |\n")
        handle.write(
            "\nThe anisotropy ratio is a descriptive QC value. The nnU-Net planner's "
            "target spacing and resampling fields above are authoritative for this run.\n"
        )
    print(f"Planner summary: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
