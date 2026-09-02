from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT

from tmj_condyle.config import (
    NIFTI_DIR,
    REPORTS_DIR,
    ensure_workspace_dirs,
    manifest_path_for,
    validate_case_id,
)
from tmj_condyle.data.manifest import find_case, read_manifest, upsert_case
from tmj_condyle.dicom.series import read_series
from tmj_condyle.utils.geometry import geometries_match, geometry_of, scrub_metadata
from tmj_condyle.utils.io import require_simpleitk


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert one DICOM series to a scrubbed, geometry-preserving NIfTI."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--case-id", required=True, help="Anonymous id such as case_001")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--series-uid", help="Private local selector when multiple series exist")
    parser.add_argument("--series-index", type=int, help="Redacted series index from inspect_dicom.py")
    parser.add_argument("--group-id", default="", help="Anonymous grouping id, never a PatientID")
    parser.add_argument("--side", default="", choices=("", "L", "R"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_workspace_dirs()
    case_id = validate_case_id(args.case_id)
    output = (args.output or (NIFTI_DIR / f"{case_id}.nii.gz")).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite for this generated file: {output}")

    sitk = require_simpleitk()
    original, series = read_series(
        args.input_dir,
        series_uid=args.series_uid,
        series_index=args.series_index,
    )
    original_array = sitk.GetArrayFromImage(original)
    clean = scrub_metadata(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(clean, str(output), useCompression=True)
    reread = sitk.ReadImage(str(output))
    if not geometries_match(original, reread):
        raise RuntimeError("DICOM to NIfTI geometry verification failed")
    if not np.array_equal(original_array, sitk.GetArrayFromImage(reread)):
        raise RuntimeError("DICOM to NIfTI voxel value verification failed")

    manifest = (args.manifest or (PROJECT_ROOT / "workspace" / "dataset_manifest.csv")).resolve()
    existing = find_case(read_manifest(manifest), case_id)
    existing_group_id = existing.get("group_id", "") if existing else ""
    existing_side = existing.get("side", "") if existing else ""
    existing_label_path = existing.get("label_path", "") if existing else ""
    reconversion_note = (
        " Existing label path preserved but must be re-validated after image reconversion."
        if existing_label_path
        else ""
    )
    upsert_case(
        case_id=case_id,
        image_path=output,
        label_path=existing_label_path,
        group_id=args.group_id or existing_group_id or case_id,
        side=args.side or existing_side,
        annotation_status="NEW",
        geometry_valid=True,
        label_valid="",
        notes="DICOM series converted locally; condyle mask still required." + reconversion_note,
        path=manifest,
    )
    report = {
        "case_id": case_id,
        "input_dir": "<private DICOM input redacted>",
        "output": manifest_path_for(output),
        "series_index": series.index,
        "file_count": series.file_count,
        "geometry": geometry_of(reread).to_dict(),
        "voxel_values_preserved": True,
        "dicom_metadata_scrubbed": True,
        "note": "Series UID and DICOM patient metadata are not written to the report.",
    }
    report_path = REPORTS_DIR / f"{case_id}_dicom_to_nifti.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
