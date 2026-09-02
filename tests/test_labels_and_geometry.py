from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from tmj_condyle.labels.qc import validate_label_array, validate_pair
from tmj_condyle.utils.geometry import geometries_match


def _image_and_label(label_array):
    image = sitk.GetImageFromArray(np.zeros_like(label_array, dtype=np.float32))
    image.SetSpacing((0.25, 0.25, 5.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    label = sitk.GetImageFromArray(label_array.astype(np.uint8))
    label.CopyInformation(image)
    return image, label


def test_binary_mask_and_geometry_match():
    data = np.zeros((4, 5, 6), dtype=np.uint8)
    data[1:3, 2:4, 2:4] = 1
    image, label = _image_and_label(data)
    assert geometries_match(image, label)
    report = validate_pair(image, label)
    assert report["errors"] == []
    assert report["stats"]["foreground_voxels"] == 8


def test_empty_mask_is_rejected():
    image, label = _image_and_label(np.zeros((4, 5, 6), dtype=np.uint8))
    report = validate_pair(image, label)
    assert any("empty" in error for error in report["errors"])
    assert validate_pair(image, label, allow_empty=True)["errors"] == []


def test_non_binary_mask_is_rejected():
    data = np.zeros((4, 5, 6), dtype=np.uint8)
    data[1, 1, 1] = 2
    report = validate_label_array(data, allow_empty=True)
    assert any("0, 1" in error for error in report["errors"])


def test_geometry_mismatch_is_rejected():
    data = np.zeros((4, 5, 6), dtype=np.uint8)
    data[1, 1, 1] = 1
    image, label = _image_and_label(data)
    label.SetSpacing((0.25, 0.25, 6.0))
    report = validate_pair(image, label)
    assert any("geometry mismatch" in error for error in report["errors"])
