from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from tmj_condyle.dicom.series import discover_series


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List DICOM series using GDCM without printing patient identifiers."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    series = discover_series(args.input_dir)
    payload = {
        "input_dir": "<private DICOM input redacted>",
        "series": [item.safe_dict() for item in series],
        "note": "SeriesInstanceUID and patient metadata are intentionally redacted.",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
