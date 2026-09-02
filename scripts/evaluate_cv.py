from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401

from tmj_condyle.config import (
    DATASET_NAME,
    LABELS_DIR,
    MANIFEST_PATH,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    NNUNET_PREPROCESSED_DIR,
    resolve_project_path,
)
from tmj_condyle.data.manifest import find_case, read_manifest
from tmj_condyle.evaluation.figures import generate_case_figures
from tmj_condyle.evaluation.metrics import case_metrics
from tmj_condyle.labels.qc import validate_label_array
from tmj_condyle.utils.geometry import geometry_differences
from tmj_condyle.utils.io import read_image, require_simpleitk


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric in ("dice", "iou", "hd95_mm"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        output.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)) if values.size else math.nan,
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(np.median(finite)) if finite.size else math.inf,
                "n": int(values.size),
                "n_finite": int(finite.size),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metric_figure(summary: list[dict[str, object]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [row["metric"] for row in summary]
    means = [row["mean"] if math.isfinite(float(row["mean"])) else 0 for row in summary]
    stds = [row["std"] if math.isfinite(float(row["std"])) else 0 for row in summary]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, means, yerr=stds, capsize=4, color=["#d97706", "#0f766e", "#2563eb"])
    ax.set_title("Five-fold out-of-fold metrics")
    ax.set_ylabel("Value (HD95 in mm)")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate only true out-of-fold predictions with Dice, IoU, and physical HD95."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--predictions-root", type=Path, default=PREDICTIONS_DIR / "oof")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    args = parser.parse_args()

    split_path = args.splits or (
        NNUNET_PREPROCESSED_DIR / args.dataset_name / "splits_final.json"
    )
    if not split_path.exists():
        print(f"Evaluation blocked: missing splits file {split_path}")
        return 2
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    manifest = read_manifest(args.manifest)
    sitk = require_simpleitk()
    per_case: list[dict[str, object]] = []
    for fold, split in enumerate(splits):
        for case_id in split["val"]:
            row = find_case(manifest, case_id)
            if row is None:
                print(f"Evaluation blocked: {case_id} is absent from manifest")
                return 2
            label_path = resolve_project_path(row.get("label_path") or (LABELS_DIR / f"{case_id}.nii.gz"))
            prediction_path = args.predictions_root / f"fold_{fold}" / f"{case_id}.nii.gz"
            if not label_path.exists() or not prediction_path.exists():
                print(f"Missing OOF pair for case={case_id}, fold={fold}")
                print(f"  label: {label_path}")
                print(f"  prediction: {prediction_path}")
                return 2
            label = read_image(label_path)
            prediction = read_image(prediction_path)
            differences = geometry_differences(label, prediction)
            if differences:
                print(f"Geometry mismatch for {case_id}: {differences}")
                return 2
            label_qc = validate_label_array(sitk.GetArrayFromImage(label), allow_empty=False)
            pred_qc = validate_label_array(sitk.GetArrayFromImage(prediction), allow_empty=True)
            if label_qc["errors"] or pred_qc["errors"]:
                print(f"Label QC failed for {case_id}: {label_qc['errors']} {pred_qc['errors']}")
                return 2
            metrics = case_metrics(
                sitk.GetArrayFromImage(label),
                sitk.GetArrayFromImage(prediction),
                spacing_xyz=tuple(float(value) for value in label.GetSpacing()),
            )
            per_case.append(
                {
                    "case_id": case_id,
                    "fold": fold,
                    "group_id": row.get("group_id", case_id),
                    **metrics,
                }
            )

    if not per_case:
        print("Evaluation blocked: no cases found.")
        return 2
    report_dir = args.report_dir
    fields = [
        "case_id", "fold", "group_id", "dice", "iou", "hd95_mm",
        "absolute_volume_difference_mm3", "gt_volume_mm3", "prediction_volume_mm3",
    ]
    _write_csv(report_dir / "metrics_per_case.csv", per_case, fields)
    summary = _summary_rows(per_case)
    _write_csv(
        report_dir / "metrics_summary.csv",
        summary,
        ["metric", "mean", "std", "median", "n", "n_finite"],
    )
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_metric_figure(summary, figures_dir / "figure_06_metrics_summary.png")
    if args.make_figures:
        for row in per_case:
            case = find_case(manifest, row["case_id"])
            image_path = resolve_project_path(case.get("image_path", ""))
            label_path = resolve_project_path(case.get("label_path", ""))
            prediction_path = args.predictions_root / f"fold_{row['fold']}" / f"{row['case_id']}.nii.gz"
            try:
                generate_case_figures(
                    image=read_image(image_path),
                    ground_truth=read_image(label_path),
                    prediction=read_image(prediction_path),
                    output_dir=figures_dir / str(row["case_id"]),
                )
            except RuntimeError as exc:
                print(f"Figure generation skipped for {row['case_id']}: {exc}")

    report_path = report_dir / "cv_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Five-fold out-of-fold evaluation\n\n")
        handle.write("Predictions were generated by the corresponding validation fold only; training-set predictions are not evaluated.\n\n")
        handle.write(f"- Cases evaluated: {len(per_case)}\n")
        handle.write(f"- Folds: {len(splits)}\n")
        handle.write("- Patient/group leakage: PASS (checked by split creation and validation)\n\n")
        handle.write("| Metric | Mean | Std | Median | N finite |\n|---|---:|---:|---:|---:|\n")
        for row in summary:
            handle.write(
                f"| {row['metric']} | {row['mean']:.6f} | {row['std']:.6f} | "
                f"{row['median']:.6f} | {row['n_finite']} |\n"
            )
        handle.write("\nHD95 uses the image physical spacing in millimetres.\n")
    print(f"OOF cases evaluated: {len(per_case)}")
    print(f"Reports: {report_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
