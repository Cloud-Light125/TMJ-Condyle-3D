from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from tmj_condyle.data.manifest import write_manifest
from tmj_condyle.data.validation import validate_manifest_dataset


def _write_image(path):
    image = sitk.GetImageFromArray(np.ones((2, 3, 4), dtype=np.float32))
    image.SetSpacing((0.25, 0.25, 5.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def test_missing_label_is_reported_without_dot_path(tmp_path):
    image_path = tmp_path / "nifti" / "case_001.nii.gz"
    _write_image(image_path)
    manifest = tmp_path / "dataset_manifest.csv"
    write_manifest(
        [
            {
                "case_id": "case_001",
                "group_id": "group_001",
                "side": "L",
                "image_path": str(image_path),
                "label_path": "",
                "annotation_status": "NEW",
                "geometry_valid": "true",
                "label_valid": "",
                "notes": "",
            }
        ],
        manifest,
    )
    rows, valid = validate_manifest_dataset(
        manifest_path=manifest,
        images_dir=tmp_path / "nifti",
        labels_dir=tmp_path / "labels",
        report_dir=tmp_path / "reports",
    )
    assert not valid
    assert rows[0]["label_path"] == ""
    assert "missing label" in rows[0]["errors"]


def test_duplicate_image_stems_are_reported(tmp_path):
    _write_image(tmp_path / "nifti" / "case_001.nii.gz")
    _write_image(tmp_path / "nifti" / "case_001.nii")
    manifest = tmp_path / "dataset_manifest.csv"
    write_manifest(
        [
            {
                "case_id": "case_001",
                "group_id": "group_001",
                "side": "",
                "image_path": "",
                "label_path": "",
                "annotation_status": "NEW",
                "geometry_valid": "",
                "label_valid": "",
                "notes": "",
            }
        ],
        manifest,
    )
    rows, valid = validate_manifest_dataset(
        manifest_path=manifest,
        images_dir=tmp_path / "nifti",
        labels_dir=tmp_path / "labels",
        report_dir=tmp_path / "reports",
    )
    assert not valid
    assert "duplicate case_id in image directory" in rows[0]["errors"]
