from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from tmj_condyle.config import LABELS_DIR, MANIFEST_PATH, NIFTI_DIR, REPORTS_DIR
from tmj_condyle.data.manifest import ANNOTATION_STATUSES
from tmj_condyle.data.validation import validate_manifest_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate VERIFIED manifest image/label pairs before nnU-Net."
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--images-dir", type=Path, default=NIFTI_DIR)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--include-status",
        action="append",
        choices=sorted(ANNOTATION_STATUSES),
        dest="include_statuses",
        help="高级覆盖：额外允许指定状态；默认只检查 VERIFIED。可重复传入。",
    )
    args = parser.parse_args()

    allowed_statuses = {
        value.upper() for value in (args.include_statuses or ["VERIFIED"])
    }

    rows, passed = validate_manifest_dataset(
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        report_dir=args.report_dir,
        statuses=allowed_statuses,
    )
    print(f"Cases checked: {len(rows)}")
    print(f"PASS: {sum(row['status'] == 'PASS' for row in rows)}")
    print(f"FAIL: {sum(row['status'] == 'FAIL' for row in rows)}")
    print(f"Reports: {args.report_dir.resolve()}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
