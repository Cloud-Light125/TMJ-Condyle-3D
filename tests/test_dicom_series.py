from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from tmj_condyle.dicom.series import discover_series, read_series


def _write_dicom_slice(path, value, instance, z, series_uid):
    image = sitk.GetImageFromArray(np.full((4, 4), value, dtype=np.uint16))
    image.SetSpacing((0.25, 0.25))
    image.SetMetaData("0008|0060", "MR")
    image.SetMetaData("0020|000e", series_uid)
    image.SetMetaData("0020|0013", str(instance))
    image.SetMetaData("0020|0032", f"0\\0\\{z}")
    image.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")
    image.SetMetaData("0008|0018", f"1.2.826.0.1.3680043.8.498.{instance}")
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(str(path))
    writer.Execute(image)


def test_gdcm_series_reader_orders_by_dicom_geometry_not_filename(tmp_path):
    series_uid = "1.2.826.0.1.3680043.8.498.100"
    _write_dicom_slice(tmp_path / "slice_z20.dcm", 20, 2, 20, series_uid)
    _write_dicom_slice(tmp_path / "slice_z00.dcm", 0, 0, 0, series_uid)
    _write_dicom_slice(tmp_path / "slice_z10.dcm", 10, 1, 10, series_uid)
    found = discover_series(tmp_path)
    assert len(found) == 1
    image, _ = read_series(tmp_path)
    array = sitk.GetArrayFromImage(image)
    assert array[:, 0, 0].tolist() == [0, 10, 20]


def test_gdcm_series_reader_finds_series_in_nested_directory(tmp_path):
    nested = tmp_path / "anonymous_case_folder"
    nested.mkdir()
    series_uid = "1.2.826.0.1.3680043.8.498.101"
    _write_dicom_slice(nested / "a.dcm", 7, 0, 0, series_uid)
    _write_dicom_slice(nested / "b.dcm", 9, 1, 10, series_uid)
    found = discover_series(tmp_path)
    assert len(found) == 1
    image, _ = read_series(tmp_path)
    assert sitk.GetArrayFromImage(image)[:, 0, 0].tolist() == [7, 9]
