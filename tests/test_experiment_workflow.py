from __future__ import annotations

import datetime
import json

from tmj_condyle.experiment import (
    assess_training_readiness,
    case_counts,
    count_guidance,
    create_experiment_run,
    detect_fold_states,
    export_experiment_results,
    finalize_experiment_run,
    folds_needing_training,
    format_metric,
    home_next_step,
    load_case_inventory,
    parse_training_line,
    prediction_result_ready,
    read_metrics_csv,
    summarize_metrics_by_fold,
    training_command,
    prediction_command,
    user_training_message,
)
from tmj_condyle.data.manifest import write_manifest


def test_training_conditions_and_sample_guidance_are_explicit():
    blocked = assess_training_readiness(
        annotated_cases=4,
        group_count=4,
        validation_passed=True,
        environment_ready=True,
        gpu_ready=True,
        dataset_prepared=True,
    )
    assert not blocked.formal_ready
    assert "至少 5" in blocked.message
    assert "病例数量不足" in count_guidance(4)
    assert "初步实验" in count_guidance(8)
    assert "小样本" in count_guidance(20)
    assert "充足" in count_guidance(50)


def test_home_state_machine_and_real_script_commands_are_explicit(tmp_path):
    assert "先导入" in home_next_step(
        total_cases=0,
        annotated_cases=0,
        validation_passed=False,
        dataset_prepared=False,
    )
    assert "继续标注" in home_next_step(
        total_cases=8,
        annotated_cases=4,
        validation_passed=False,
        dataset_prepared=False,
    )
    assert "训练数据" in home_next_step(
        total_cases=8,
        annotated_cases=8,
        validation_passed=True,
        dataset_prepared=False,
    )
    assert "自动分割" in home_next_step(
        total_cases=8,
        annotated_cases=8,
        validation_passed=True,
        dataset_prepared=True,
        model_ready=True,
        results_ready=True,
    )
    train = training_command(
        project_root=tmp_path,
        python_executable="python",
        plan=True,
        resume=True,
    )
    assert train[1].endswith("train_all_folds.py")
    assert {"--plan", "--resume", "--device", "cuda"}.issubset(train)
    predict = prediction_command(
        tmp_path / "new_case.nii.gz",
        tmp_path / "prediction.nii.gz",
        project_root=tmp_path,
        python_executable="python",
    )
    assert predict[1].endswith("predict.py")
    assert "--device" in predict and "cuda" in predict


def test_gpu_block_is_not_misreported_as_formal_ready():
    readiness = assess_training_readiness(
        annotated_cases=5,
        group_count=5,
        validation_passed=True,
        environment_ready=True,
        gpu_ready=False,
        dataset_prepared=True,
    )
    assert readiness.pipeline_ready
    assert not readiness.formal_ready
    assert readiness.level == "blocked_gpu"
    assert "NVIDIA" in " ".join(readiness.reasons)


def test_case_inventory_and_counts_use_real_annotation_status(tmp_path):
    image = tmp_path / "nifti" / "case_001.nii.gz"
    label = tmp_path / "labels" / "case_001.nii.gz"
    image.parent.mkdir()
    label.parent.mkdir()
    image.touch()
    label.touch()
    manifest = tmp_path / "dataset_manifest.csv"
    write_manifest(
        [
            {
                "case_id": "case_001",
                "group_id": "group_001",
                "image_path": str(image),
                "label_path": str(label),
                "annotation_status": "ANNOTATED",
                "geometry_valid": "true",
                "label_valid": "true",
            },
            {
                "case_id": "case_002",
                "group_id": "group_002",
                "image_path": str(tmp_path / "nifti" / "case_002.nii.gz"),
                "label_path": "",
                "annotation_status": "NEW",
            },
        ],
        manifest,
    )
    inventory = load_case_inventory(
        manifest_path=manifest,
        images_dir=tmp_path / "nifti",
        labels_dir=tmp_path / "labels",
    )
    counts = case_counts(inventory)
    assert counts["total"] == 2
    assert counts["annotated"] == 1
    assert counts["trainable"] == 1
    assert inventory[0]["status"] == "已标注"
    assert inventory[1]["status"] == "数据问题"


def test_fold_state_detection_and_resume_selection(tmp_path):
    root = tmp_path / "nnUNet_results" / "config"
    (root / "fold_0").mkdir(parents=True)
    (root / "fold_0" / "checkpoint_final.pth").touch()
    (root / "fold_1").mkdir(parents=True)
    (root / "fold_1" / "checkpoint_latest.pth").touch()
    states = detect_fold_states(root)
    assert states[0].status == "completed"
    assert states[1].status == "incomplete"
    assert states[2].status == "waiting"
    assert folds_needing_training(states) == (1, 2, 3, 4)


def test_training_markers_and_metrics_are_read_without_fake_values(tmp_path):
    event = parse_training_line("TMJ_FOLD_START 2")
    assert event["event"] == "start"
    assert event["fold"] == 2
    assert parse_training_line("Epoch 13 / 100")["epoch_total"] == 100
    assert user_training_message(parse_training_line("Epoch 13 / 100")) == "Epoch 13 / 100"

    metrics = tmp_path / "metrics_per_case.csv"
    metrics.write_text(
        "case_id,fold,dice,iou,hd95_mm\n"
        "case_001,0,0.9,0.8,2.0\n"
        "case_002,0,0.7,0.6,4.0\n",
        encoding="utf-8",
    )
    rows = read_metrics_csv(metrics)
    folds = summarize_metrics_by_fold(rows)
    assert folds[0]["case_count"] == 2
    assert folds[0]["dice_mean"] == 0.8
    assert format_metric(None) == "暂无真实结果"
    assert format_metric(0.8, 0.1) == "0.80 ± 0.10"


def test_experiment_record_and_export_only_copy_report_artifacts(tmp_path):
    workspace = tmp_path / "workspace"
    run = create_experiment_run(
        workspace_dir=workspace,
        now=datetime.datetime(2026, 9, 3, 12, 34, 56),
        config={"case_count": 5, "gpu": "unavailable"},
    )
    report = tmp_path / "reports"
    report.mkdir()
    (report / "metrics_summary.csv").write_text("metric,mean,std\ndice,0.8,0.1\n", encoding="utf-8")
    (report / "metrics_per_case.csv").write_text("case_id,fold,dice\ncase_001,0,0.8\n", encoding="utf-8")
    (report / "training_summary.md").write_text("summary", encoding="utf-8")
    finalize_experiment_run(run, summary={"status": "complete"}, report_dir=report)
    (run / "screenshots").mkdir()
    (run / "screenshots" / "gt_prediction_3d.png").write_bytes(b"anonymous-view")
    assert json.loads((run / "config.json").read_text(encoding="utf-8"))["case_count"] == 5
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["status"] == "complete"
    destination = export_experiment_results(run, destination=tmp_path / "experiment_export")
    assert (destination / "metrics_summary.csv").exists()
    assert (destination / "metrics_per_case.csv").exists()
    assert (destination / "screenshots" / "gt_prediction_3d.png").exists()
    assert not (destination / "checkpoint_final.pth").exists()


def test_prediction_result_detection(tmp_path):
    missing = tmp_path / "missing.nii.gz"
    assert not prediction_result_ready(missing)
    missing.write_bytes(b"mask")
    assert prediction_result_ready(missing)
