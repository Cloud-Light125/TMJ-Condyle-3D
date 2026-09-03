from __future__ import annotations

import json

import pytest

from tmj_condyle import launcher
from tmj_condyle.data.manifest import write_manifest
from tmj_condyle.data.nnunet import build_dataset
from tmj_condyle.experiment import (
    case_counts,
    home_next_action,
    load_case_inventory,
    should_show_first_run_wizard,
    training_prerequisite_summary,
)


def _write_placeholder_pair(root, case_id="case_001"):
    image = root / "nifti" / f"{case_id}.nii.gz"
    label = root / "labels" / f"{case_id}.nii.gz"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    label.write_bytes(b"label")
    return image, label


def test_launcher_discovers_default_configured_and_common_paths_in_order(tmp_path, monkeypatch):
    default = tmp_path / "default" / "Slicer.exe"
    configured = tmp_path / "configured" / "Slicer.exe"
    common_root = tmp_path / "localappdata"
    common = common_root / "slicer.org" / "3D Slicer 5.12.3" / "Slicer.exe"
    for path in (default, configured, common):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"exe")

    monkeypatch.setattr(launcher, "DEFAULT_SLICER_PATH", default)
    launcher.write_slicer_config(configured, project_root=tmp_path / "project")
    candidates = launcher.discover_slicer_candidates(
        project_root=tmp_path / "project",
        environment={
            "ProgramFiles": str(tmp_path / "program-files"),
            "ProgramFiles(x86)": str(tmp_path / "program-files-x86"),
            "LOCALAPPDATA": str(common_root),
            "USERPROFILE": str(tmp_path / "user"),
        },
    )
    assert [item.path for item in candidates] == [default, configured, common]
    assert [item.source for item in candidates] == ["项目默认位置", "项目设置", "用户本地安装"]


def test_launcher_config_round_trip_and_corrupt_config_are_safe(tmp_path):
    executable = tmp_path / "Slicer.exe"
    executable.write_bytes(b"exe")
    project = tmp_path / "project"
    destination = launcher.write_slicer_config(executable, project_root=project)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["slicer_path"] == str(executable.resolve())
    assert launcher.configured_slicer_path(project) == executable.resolve()
    destination.write_text("not json", encoding="utf-8")
    assert launcher.configured_slicer_path(project) is None


def test_project_state_separates_saved_annotation_from_verified_training(tmp_path):
    image_a, label_a = _write_placeholder_pair(tmp_path, "case_001")
    image_b, label_b = _write_placeholder_pair(tmp_path, "case_002")
    manifest = tmp_path / "dataset_manifest.csv"
    write_manifest(
        [
            {
                "case_id": "case_001",
                "group_id": "group_001",
                "image_path": str(image_a),
                "label_path": str(label_a),
                "annotation_status": "ANNOTATED",
            },
            {
                "case_id": "case_002",
                "group_id": "group_002",
                "image_path": str(image_b),
                "label_path": str(label_b),
                "annotation_status": "VERIFIED",
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
    assert counts["annotated"] == 2
    assert counts["verified"] == 1
    assert counts["trainable"] == 1
    assert {row["status"] for row in inventory} == {"已标注", "已确认"}


def test_only_verified_cases_can_build_training_dataset(tmp_path):
    image, label = _write_placeholder_pair(tmp_path, "case_001")
    with pytest.raises(ValueError, match="VERIFIED"):
        build_dataset(
            [
                {
                    "case_id": "case_001",
                    "group_id": "group_001",
                    "image_path": str(image),
                    "label_path": str(label),
                    "annotation_status": "ANNOTATED",
                }
            ],
            nnunet_raw=tmp_path / "nnUNet_raw",
            nnunet_preprocessed=tmp_path / "nnUNet_preprocessed",
            reports_dir=tmp_path / "reports",
        )


def test_first_run_state_next_action_and_training_summary_are_explicit():
    assert should_show_first_run_wizard("")
    assert should_show_first_run_wizard("false")
    assert not should_show_first_run_wizard("true")

    action = home_next_action(total_cases=5, annotated_cases=5, verified_cases=0)
    assert action.key == "cases"
    assert "确认" in action.message
    assert action.button == "确认标注"

    summary = training_prerequisite_summary(
        available_cases=4,
        patient_groups=3,
        validation_passed=False,
        environment_ready=False,
        gpu_ready=False,
        dataset_prepared=False,
    )
    assert not summary["formal_ready"]
    assert "可用病例：4" in summary["text"]
    assert any("不同患者组不足 5 组" in reason for reason in summary["reasons"])
    assert any(
        "当前电脑没有检测到可用于正式训练的 NVIDIA 显卡" in reason
        for reason in summary["reasons"]
    )
