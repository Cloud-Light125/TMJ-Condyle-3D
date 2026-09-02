from __future__ import annotations

import json

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from tmj_condyle.data.nnunet import build_dataset
from tmj_condyle.data.manifest import write_manifest


def _write_pair(root, case_id):
    image_path = root / "nifti" / f"{case_id}.nii.gz"
    label_path = root / "labels" / f"{case_id}.nii.gz"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.ones((4, 5, 6), dtype=np.float32))
    image.SetSpacing((0.25, 0.25, 5.0))
    label_array = np.zeros((4, 5, 6), dtype=np.uint8)
    label_array[1:3, 2:4, 2:4] = 1
    label = sitk.GetImageFromArray(label_array)
    label.CopyInformation(image)
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(label, str(label_path))
    return image_path, label_path


def test_dataset_json_nnunet_names_and_splits(tmp_path):
    rows = []
    for i in range(1, 6):
        case_id = f"case_{i:03d}"
        image, label = _write_pair(tmp_path, case_id)
        rows.append(
            {
                "case_id": case_id,
                "group_id": f"group_{i:03d}",
                "side": "L",
                "image_path": str(image),
                "label_path": str(label),
                "annotation_status": "VERIFIED",
                "geometry_valid": "true",
                "label_valid": "true",
                "notes": "",
            }
        )
    manifest = tmp_path / "dataset_manifest.csv"
    write_manifest(rows, manifest)
    raw, split_path, splits = build_dataset(
        rows,
        nnunet_raw=tmp_path / "nnUNet_raw",
        nnunet_preprocessed=tmp_path / "nnUNet_preprocessed",
        reports_dir=tmp_path / "reports",
        seed=1,
    )
    dataset = json.loads((raw / "dataset.json").read_text(encoding="utf-8"))
    assert dataset["channel_names"] == {"0": "MRI"}
    assert dataset["labels"] == {"background": 0, "mandibular_condyle": 1}
    assert dataset["numTraining"] == 5
    assert sorted(path.name for path in (raw / "imagesTr").glob("*.nii.gz")) == [
        f"case_{i:03d}_0000.nii.gz" for i in range(1, 6)
    ]
    assert sorted(path.name for path in (raw / "labelsTr").glob("*.nii.gz")) == [
        f"case_{i:03d}.nii.gz" for i in range(1, 6)
    ]
    assert split_path.exists()
    assert len(splits) == 5
