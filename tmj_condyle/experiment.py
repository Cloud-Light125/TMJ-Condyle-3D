"""GUI-facing helpers for the reproducible TMJ segmentation experiment.

The Slicer module is deliberately kept thin around this module.  Everything
here is pure Python (apart from the existing dataset readers) so the workflow
state can be tested without starting 3D Slicer, loading a patient image, or
pretending that a model/metric exists.
"""

from __future__ import annotations

import csv
import datetime as _datetime
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import (
    CONFIGURATION,
    DATASET_NAME,
    DEFAULT_SPLIT_SEED,
    LABELS_DIR,
    MANIFEST_PATH,
    N_FOLDS,
    NIFTI_DIR,
    NNUNET_PREPROCESSED_DIR,
    NNUNET_RAW_DIR,
    NNUNET_RESULTS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    WORKSPACE_DIR,
    resolve_project_path,
)
from .data.manifest import read_manifest


FOLDS = tuple(range(N_FOLDS))
FORMAL_TRAINING_DEVICE = "cuda"
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetPlans"
CASE_COMPLETE_STATUSES = {"ANNOTATED", "VERIFIED"}

_NIFTI_SUFFIXES = (".nii.gz", ".nii", ".nrrd")
_FOLD_MARKER = re.compile(
    r"TMJ_(?:FOLD_)?(?P<event>START|COMPLETE|FAILED)\s+(?P<fold>[0-9]+)(?:\s+(?P<code>-?[0-9]+))?",
    re.IGNORECASE,
)
_EPOCH_PATTERNS = (
    re.compile(r"\bEpoch\s*[:=]?\s*(?P<epoch>[0-9]+)(?:\s*/\s*(?P<total>[0-9]+))?", re.I),
    re.compile(r"\bepoch\s+(?P<epoch>[0-9]+)\b", re.I),
)


@dataclass(frozen=True)
class Readiness:
    """A user-facing readiness decision with an explicit blocking reason."""

    formal_ready: bool
    pipeline_ready: bool
    level: str
    message: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "formal_ready": self.formal_ready,
            "pipeline_ready": self.pipeline_ready,
            "level": self.level,
            "message": self.message,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FoldState:
    fold: int
    status: str
    checkpoint_final: str = ""
    checkpoint_latest: str = ""
    has_progress: bool = False

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def resumable(self) -> bool:
        return self.status == "incomplete"

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "status": self.status,
            "checkpoint_final": self.checkpoint_final,
            "checkpoint_latest": self.checkpoint_latest,
            "has_progress": self.has_progress,
        }


def _case_id_from_filename(path: Path) -> str:
    for suffix in _NIFTI_SUFFIXES:
        if path.name.lower().endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _image_files(directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    if not root.exists():
        return {}
    output: dict[str, Path] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.name.lower().endswith(_NIFTI_SUFFIXES):
            output.setdefault(_case_id_from_filename(path), path.resolve())
    return output


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "通过"}


def load_case_inventory(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    images_dir: str | Path = NIFTI_DIR,
    labels_dir: str | Path = LABELS_DIR,
) -> list[dict[str, object]]:
    """Return anonymous case rows for the GUI case list.

    A row is considered trainable only when its manifest status is annotated
    or verified and a label file is present.  Pixel/geometry QC remains the
    responsibility of ``validate_dataset.py``; this function never upgrades a
    case to ready based on a filename alone.
    """

    manifest_rows = read_manifest(manifest_path)
    images = _image_files(images_dir)
    labels = _image_files(labels_dir)
    by_id: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        case_id = str(row.get("case_id") or "").strip()
        if case_id and case_id not in by_id:
            by_id[case_id] = row

    case_ids = sorted(set(by_id) | set(images) | set(labels))
    inventory: list[dict[str, object]] = []
    for case_id in case_ids:
        row = by_id.get(case_id, {})
        image_value = row.get("image_path") or images.get(case_id) or ""
        label_value = row.get("label_path") or labels.get(case_id) or ""
        image_path = resolve_project_path(str(image_value)) if image_value else Path("")
        label_path = resolve_project_path(str(label_value)) if label_value else Path("")
        image_exists = bool(str(image_path)) and image_path.is_file()
        label_exists = bool(str(label_path)) and label_path.is_file()
        annotation_status = str(row.get("annotation_status") or "").strip()
        trainable = annotation_status in CASE_COMPLETE_STATUSES and label_exists and image_exists
        if trainable:
            status = "已标注"
        elif annotation_status == "ANNOTATING" or label_exists:
            status = "标注中"
        elif image_exists:
            status = "未标注"
        else:
            status = "数据问题"
        problems: list[str] = []
        if not image_exists:
            problems.append("没有 MRI")
        if annotation_status in CASE_COMPLETE_STATUSES and not label_exists:
            problems.append("没有标注文件")
        if label_exists and annotation_status not in CASE_COMPLETE_STATUSES:
            problems.append("标注状态未完成")
        inventory.append(
            {
                "case_id": case_id,
                "group_id": str(row.get("group_id") or case_id),
                "side": str(row.get("side") or ""),
                "image_path": str(image_path) if image_exists else str(row.get("image_path") or ""),
                "label_path": str(label_path) if label_exists else str(row.get("label_path") or ""),
                "annotation_status": annotation_status,
                "status": status,
                "trainable": trainable,
                "problems": problems,
            }
        )
    return inventory


def case_counts(
    inventory: Iterable[Mapping[str, object]],
    *,
    validation_rows: Iterable[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Calculate homepage/dataset counters without inventing validation."""

    rows = list(inventory)
    ready = [row for row in rows if bool(row.get("trainable"))]
    validation = list(validation_rows or [])
    validation_passed = {
        str(row.get("case_id"))
        for row in validation
        if str(row.get("status")) == "PASS"
    }
    validation_failed = [row for row in validation if str(row.get("status")) == "FAIL"]
    groups = {str(row.get("group_id") or row.get("case_id")) for row in ready}
    return {
        "total": len(rows),
        "annotated": len(ready),
        "trainable": (
            sum(str(row.get("case_id")) in validation_passed for row in ready)
            if validation
            else len(ready)
        ),
        "group_count": len(groups),
        "issues": len(validation_failed),
        "validation_checked": bool(validation),
        "validation_passed": bool(validation) and not validation_failed,
        "cases": rows,
    }


def count_guidance(group_count: int) -> str:
    """Return the plain-language sample-size hint requested by the GUI."""

    count = int(group_count)
    if count < 5:
        return "病例数量不足，暂时不能做 5 折实验。"
    if count < 15:
        return "可以测试流程，但数据较少，结果只适合作为初步实验。"
    if count < 50:
        return "可以进行小样本实验。"
    return "病例数量充足，可以正常进行 5 折实验。"


def home_next_step(
    *,
    total_cases: int,
    annotated_cases: int,
    validation_passed: bool,
    dataset_prepared: bool,
    training_active: bool = False,
    model_ready: bool = False,
    results_ready: bool = False,
) -> str:
    """Return the homepage's next action for a novice user."""

    annotated = int(annotated_cases)
    total = int(total_cases)
    if annotated == 0:
        return "下一步：先导入核磁病例，并标注下颌髁突。"
    if not validation_passed or not dataset_prepared:
        if annotated < len(FOLDS):
            return (
                f"下一步：继续标注；当前已有 {annotated} 个病例，至少需要 "
                f"{len(FOLDS)} 个不同患者组才能做 5 折实验。"
            )
        return f"下一步：已有 {annotated} 个病例可以训练，进入“训练数据”检查并准备数据。"
    if training_active:
        return "下一步：等待后台实验完成；你仍然可以查看病例标注。"
    if model_ready:
        if results_ready:
            return "实验结果已完成。下一步：查看指标，或使用新 MRI 自动分割。"
        return "模型训练完成，下一步：等待真实 OOF 评价结果。"
    return f"当前已有 {total} 个病例。下一步：进入“模型训练”开始实验。"


def assess_training_readiness(
    *,
    annotated_cases: int,
    group_count: int,
    validation_passed: bool,
    environment_ready: bool,
    gpu_ready: bool,
    dataset_prepared: bool = False,
) -> Readiness:
    """Decide whether formal training may start.

    GPU absence is represented as a block for formal training, never silently
    converted into a CPU result.  ``pipeline_ready`` means that the user can
    inspect or prepare the pipeline; it does not mean that a model exists.
    """

    reasons: list[str] = []
    if annotated_cases < N_FOLDS:
        reasons.append(f"还需要至少 {N_FOLDS} 个已标注病例才能进行 5 折实验。")
    if group_count < N_FOLDS:
        reasons.append(f"需要至少 {N_FOLDS} 个不同患者组，当前只有 {group_count} 个。")
    if not validation_passed:
        reasons.append("训练数据还没有通过完整检查。")
    if not environment_ready:
        reasons.append("Python、nnU-Net 或训练依赖还没有准备好。")
    if not dataset_prepared:
        reasons.append("请先点击“准备训练数据”。")
    if not gpu_ready:
        reasons.append("当前电脑没有检测到可用于训练的 NVIDIA 显卡。")

    pipeline_ready = annotated_cases >= N_FOLDS and group_count >= N_FOLDS and validation_passed
    if not pipeline_ready:
        level = "blocked_data"
        message = reasons[0] if reasons else "当前还不能开始实验。"
    elif not environment_ready:
        level = "blocked_environment"
        message = "训练环境还没有准备好，请先完成系统检查。"
    elif not gpu_ready:
        level = "blocked_gpu"
        message = "当前机器暂不适合正式训练。"
    elif not dataset_prepared:
        level = "needs_dataset"
        message = "数据检查通过，下一步请准备训练数据。"
    else:
        level = "ready"
        message = "可以开始正式 5 折训练。"
    return Readiness(
        formal_ready=not reasons,
        pipeline_ready=pipeline_ready,
        level=level,
        message=message,
        reasons=tuple(reasons),
    )


def read_validation_csv(path: str | Path = REPORTS_DIR / "dataset_validation.csv") -> list[dict[str, str]]:
    destination = Path(path)
    if not destination.exists():
        return []
    with destination.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_splits(path: str | Path) -> list[dict[str, list[str]]]:
    """Read and minimally validate the grouped split artifact."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("splits_final.json must contain a list")
    splits: list[dict[str, list[str]]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"split {index} is not an object")
        train = raw.get("train")
        val = raw.get("val")
        if not isinstance(train, list) or not isinstance(val, list):
            raise ValueError(f"split {index} must contain train and val lists")
        splits.append({"train": [str(item) for item in train], "val": [str(item) for item in val]})
    return splits


def fold_results_directory(
    *,
    results_root: str | Path = NNUNET_RESULTS_DIR,
    dataset: str = DATASET_NAME,
    trainer: str = TRAINER,
    plans: str = PLANS,
    configuration: str = CONFIGURATION,
) -> Path:
    return Path(results_root) / dataset / f"{trainer}__{plans}__{configuration}"


def detect_fold_states(
    results_directory: str | Path | None = None,
    *,
    folds: Sequence[int] = FOLDS,
) -> list[FoldState]:
    """Detect completed and resumable folds from actual nnU-Net artifacts."""

    root = Path(results_directory) if results_directory is not None else fold_results_directory()
    output: list[FoldState] = []
    for fold in folds:
        fold_dir = root / f"fold_{int(fold)}"
        final = fold_dir / "checkpoint_final.pth"
        latest_candidates = (
            fold_dir / "checkpoint_latest.pth",
            fold_dir / "checkpoint_best.pth",
        )
        latest = next((path for path in latest_candidates if path.exists()), None)
        progress = any(
            (fold_dir / name).exists()
            for name in ("progress.png", "training_log.json", "debug.json", "plans.json")
        )
        if final.exists():
            status = "completed"
        elif latest is not None or progress or fold_dir.exists():
            status = "incomplete"
        else:
            status = "waiting"
        output.append(
            FoldState(
                fold=int(fold),
                status=status,
                checkpoint_final=str(final) if final.exists() else "",
                checkpoint_latest=str(latest) if latest else "",
                has_progress=progress,
            )
        )
    return output


def completed_folds(states: Iterable[FoldState]) -> tuple[int, ...]:
    return tuple(state.fold for state in states if state.completed)


def folds_needing_training(states: Iterable[FoldState]) -> tuple[int, ...]:
    return tuple(state.fold for state in states if not state.completed)


def parse_training_line(line: str) -> dict[str, object]:
    """Parse only reliable wrapper markers and explicit epoch text."""

    text = str(line).strip()
    event: dict[str, object] = {"raw": text}
    marker = _FOLD_MARKER.search(text)
    if marker:
        event["event"] = marker.group("event").lower()
        event["fold"] = int(marker.group("fold"))
        if marker.group("code") is not None:
            event["code"] = int(marker.group("code"))
    for pattern in _EPOCH_PATTERNS:
        match = pattern.search(text)
        if match:
            event["epoch"] = int(match.group("epoch"))
            if match.groupdict().get("total"):
                event["epoch_total"] = int(match.group("total"))
            break
    return event


def user_training_message(event: Mapping[str, object]) -> str:
    event_name = str(event.get("event") or "")
    fold = int(event.get("fold", 0)) + 1 if str(event.get("fold", "")).isdigit() else None
    if event_name == "start" and fold is not None:
        return f"正在训练第 {fold} 组"
    if event_name == "complete" and fold is not None:
        return f"第 {fold} 组训练完成"
    if event_name == "failed" and fold is not None:
        return f"第 {fold} 组训练未完成"
    if "epoch" in event:
        total = event.get("epoch_total")
        prefix = f"第 {fold} 组：" if fold is not None else ""
        return f"{prefix}Epoch {event['epoch']} / {total}" if total else f"{prefix}Epoch {event['epoch']}"
    return "正在处理训练日志"


def project_python_executable(project_root: str | Path = Path(__file__).resolve().parents[1]) -> str:
    """Prefer the project's venv; avoid using Slicer's embedded interpreter."""

    root = Path(project_root)
    candidates = (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    path = shutil.which("python")
    if path:
        return path
    return sys.executable


def script_command(
    script_name: str,
    args: Iterable[object] = (),
    *,
    project_root: str | Path = Path(__file__).resolve().parents[1],
    python_executable: str | Path | None = None,
) -> list[str]:
    root = Path(project_root)
    return [
        str(python_executable or project_python_executable(root)),
        str(root / "scripts" / script_name),
        *(str(value) for value in args),
    ]


def environment_command(*, project_root: str | Path = Path(__file__).resolve().parents[1], python_executable: str | Path | None = None) -> list[str]:
    return script_command("check_environment.py", ["--json"], project_root=project_root, python_executable=python_executable)


def dataset_validation_command(*, project_root: str | Path = Path(__file__).resolve().parents[1], python_executable: str | Path | None = None) -> list[str]:
    return script_command("validate_dataset.py", project_root=project_root, python_executable=python_executable)


def dataset_build_command(*, project_root: str | Path = Path(__file__).resolve().parents[1], python_executable: str | Path | None = None) -> list[str]:
    return script_command("build_nnunet_dataset.py", project_root=project_root, python_executable=python_executable)


def training_command(
    *,
    device: str = FORMAL_TRAINING_DEVICE,
    resume: bool = False,
    plan: bool = True,
    project_root: str | Path = Path(__file__).resolve().parents[1],
    python_executable: str | Path | None = None,
) -> list[str]:
    args: list[object] = ["--device", device]
    if plan:
        args.append("--plan")
    if resume:
        args.append("--resume")
    return script_command("train_all_folds.py", args, project_root=project_root, python_executable=python_executable)


def oof_command(*, device: str = FORMAL_TRAINING_DEVICE, project_root: str | Path = Path(__file__).resolve().parents[1], python_executable: str | Path | None = None) -> list[str]:
    return script_command("run_oof_predictions.py", ["--device", device], project_root=project_root, python_executable=python_executable)


def evaluation_command(*, project_root: str | Path = Path(__file__).resolve().parents[1], python_executable: str | Path | None = None) -> list[str]:
    return script_command("evaluate_cv.py", ["--make-figures"], project_root=project_root, python_executable=python_executable)


def prediction_command(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    device: str = FORMAL_TRAINING_DEVICE,
    project_root: str | Path = Path(__file__).resolve().parents[1],
    python_executable: str | Path | None = None,
) -> list[str]:
    args: list[object] = [str(input_path), "--device", device]
    if output_path is not None:
        args.extend(["--output", str(output_path)])
    return script_command("predict.py", args, project_root=project_root, python_executable=python_executable)


def parse_environment_json(output: str) -> dict[str, object]:
    """Parse --json output while tolerating a launcher warning before JSON."""

    text = str(output).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
        for line in reversed(text.splitlines()):
            candidate = line.strip()
            if not candidate.startswith("{"):
                continue
            try:
                value = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if value is None:
            raise ValueError("Environment check did not return JSON")
    if not isinstance(value, dict):
        raise ValueError("Environment check JSON must be an object")
    return value


def environment_display(report: Mapping[str, object] | None) -> dict[str, str]:
    """Map the technical environment report to Chinese UI labels."""

    data = report or {}
    cuda = data.get("cuda") if isinstance(data.get("cuda"), Mapping) else {}
    gpu_ready = bool(cuda.get("status") == "PASS")
    return {
        "python": "✓" if data.get("python_ready", True) else "不可用",
        "nnunet": "✓" if data.get("nnunet_ready") else "不可用",
        "data": "✓" if data.get("data_ready") else "未准备",
        "gpu": "✓" if gpu_ready else "不可用",
        "gpu_message": "已检测到可用 NVIDIA GPU" if gpu_ready else "当前电脑没有检测到可用于训练的 NVIDIA 显卡。",
    }


def read_metrics_csv(path: str | Path) -> list[dict[str, object]]:
    destination = Path(path)
    if not destination.exists():
        return []
    rows: list[dict[str, object]] = []
    with destination.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key in ("fold", "n", "n_finite"):
                if row.get(key) not in (None, ""):
                    try:
                        row[key] = int(float(str(row[key])))
                    except ValueError:
                        pass
            for key in (
                "dice",
                "iou",
                "hd95_mm",
                "mean",
                "std",
                "median",
                "absolute_volume_difference_mm3",
                "gt_volume_mm3",
                "prediction_volume_mm3",
            ):
                if row.get(key) not in (None, ""):
                    try:
                        row[key] = float(str(row[key]))
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def read_metrics_summary(path: str | Path = REPORTS_DIR / "metrics_summary.csv") -> dict[str, dict[str, object]]:
    return {
        str(row.get("metric")): row
        for row in read_metrics_csv(path)
        if row.get("metric")
    }


def summarize_metrics_by_fold(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        try:
            fold = int(row.get("fold"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(fold, []).append(row)
    output: list[dict[str, object]] = []
    for fold in sorted(grouped):
        cases = grouped[fold]
        result: dict[str, object] = {"fold": fold, "case_count": len(cases)}
        for metric in ("dice", "iou", "hd95_mm"):
            values = []
            for case in cases:
                try:
                    value = float(case[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            result[f"{metric}_mean"] = sum(values) / len(values) if values else math.nan
            result[f"{metric}_std"] = _sample_std(values)
            result[f"{metric}_n_finite"] = len(values)
        output.append(result)
    return output


def _sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def format_metric(value: object, std: object | None = None, *, unit: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "暂无真实结果"
    if not math.isfinite(number):
        return "∞" + unit
    if std is None:
        return f"{number:.2f}{unit}"
    try:
        deviation = float(std)
    except (TypeError, ValueError):
        deviation = 0.0
    if not math.isfinite(deviation):
        return f"{number:.2f}{unit}"
    return f"{number:.2f} ± {deviation:.2f}{unit}"


def has_evaluation_results(report_dir: str | Path = REPORTS_DIR) -> bool:
    root = Path(report_dir)
    return (root / "metrics_summary.csv").is_file() and (root / "metrics_per_case.csv").is_file()


def prediction_result_ready(path: str | Path | None) -> bool:
    return bool(path) and Path(path).is_file() and Path(path).stat().st_size > 0


def _timestamp(value: _datetime.datetime | None = None) -> str:
    current = value or _datetime.datetime.now()
    return current.strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def create_experiment_run(
    *,
    workspace_dir: str | Path = WORKSPACE_DIR,
    config: Mapping[str, object] | None = None,
    now: _datetime.datetime | None = None,
) -> Path:
    """Create an anonymous run directory before any model process starts."""

    root = Path(workspace_dir) / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"run_{_timestamp(now)}"
    destination = root / stem
    suffix = 2
    while destination.exists():
        destination = root / f"{stem}_{suffix}"
        suffix += 1
    (destination / "logs").mkdir(parents=True, exist_ok=True)
    normalized = {
        "created_at": (now or _datetime.datetime.now()).isoformat(timespec="seconds"),
        "dataset": DATASET_NAME,
        "configuration": CONFIGURATION,
        "folds": list(FOLDS),
        "split_seed": DEFAULT_SPLIT_SEED,
        "task": "binary mandibular condyle segmentation",
    }
    if config:
        normalized.update({str(key): value for key, value in config.items()})
    _write_json(destination / "config.json", normalized)
    _write_json(destination / "summary.json", {"status": "running", "folds": []})
    return destination


def finalize_experiment_run(
    run_dir: str | Path,
    *,
    summary: Mapping[str, object],
    report_dir: str | Path = REPORTS_DIR,
) -> Path:
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "summary.json", dict(summary))
    report_root = Path(report_dir)
    for name in ("metrics_summary.csv", "metrics_per_case.csv", "cv_report.md", "training_summary.md", "planner_summary.md"):
        source = report_root / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    per_case = destination / "metrics_per_case.csv"
    metrics = destination / "metrics.csv"
    if per_case.is_file() and not metrics.exists():
        shutil.copy2(per_case, metrics)
    figures = report_root / "figures"
    if figures.is_dir():
        shutil.copytree(figures, destination / "figures", dirs_exist_ok=True)
    return destination


def list_experiment_runs(workspace_dir: str | Path = WORKSPACE_DIR) -> list[Path]:
    root = Path(workspace_dir) / "experiments"
    if not root.is_dir():
        return []
    return sorted((path for path in root.glob("run_*") if path.is_dir()), reverse=True)


def read_experiment_record(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir)
    value: dict[str, object] = {}
    for name in ("config.json", "summary.json"):
        path = root / name
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                value[name.removesuffix(".json")] = raw
    return value


def export_experiment_results(
    run_dir: str | Path,
    *,
    destination: str | Path,
) -> Path:
    """Copy report artifacts only; never copy MRI, labels, or checkpoints."""

    source = Path(run_dir)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "summary.json", "metrics.csv", "metrics_summary.csv", "metrics_per_case.csv", "cv_report.md"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
    logs = source / "logs"
    if logs.is_dir():
        (output / "logs").mkdir(parents=True, exist_ok=True)
        for path in logs.iterdir():
            if path.is_file():
                shutil.copy2(path, output / "logs" / path.name)
    figures = source / "figures"
    if figures.is_dir():
        (output / "figures").mkdir(parents=True, exist_ok=True)
        for path in figures.rglob("*"):
            if path.is_file():
                relative = path.relative_to(figures)
                target = output / "figures" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        for source_name, target_name in (
            ("figure_dice.png", "Dice.png"),
            ("figure_iou.png", "IoU.png"),
            ("figure_hd95_mm.png", "HD95.png"),
        ):
            source_chart = figures / source_name
            if source_chart.is_file():
                shutil.copy2(source_chart, output / target_name)
    screenshots = source / "screenshots"
    if screenshots.is_dir():
        (output / "screenshots").mkdir(parents=True, exist_ok=True)
        for path in screenshots.rglob("*"):
            if path.is_file():
                relative = path.relative_to(screenshots)
                target = output / "screenshots" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    return output


__all__ = [
    "CASE_COMPLETE_STATUSES",
    "FOLDS",
    "FoldState",
    "Readiness",
    "assess_training_readiness",
    "case_counts",
    "completed_folds",
    "count_guidance",
    "create_experiment_run",
    "dataset_build_command",
    "dataset_validation_command",
    "detect_fold_states",
    "environment_command",
    "environment_display",
    "evaluation_command",
    "export_experiment_results",
    "finalize_experiment_run",
    "fold_results_directory",
    "folds_needing_training",
    "format_metric",
    "has_evaluation_results",
    "home_next_step",
    "list_experiment_runs",
    "load_case_inventory",
    "oof_command",
    "parse_environment_json",
    "parse_training_line",
    "prediction_command",
    "prediction_result_ready",
    "project_python_executable",
    "read_experiment_record",
    "read_metrics_csv",
    "read_metrics_summary",
    "read_splits",
    "read_validation_csv",
    "script_command",
    "summarize_metrics_by_fold",
    "training_command",
    "user_training_message",
]
