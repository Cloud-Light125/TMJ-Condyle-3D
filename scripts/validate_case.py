from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from tmj_condyle.labels.qc import validate_pair
from tmj_condyle.utils.io import read_image


def _json_safe(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one MRI and one binary mandibular-condyle label."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--label", required=True, type=Path)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    image = read_image(args.image)
    label = read_image(args.label)
    report = validate_pair(image, label, allow_empty=args.allow_empty)
    payload = {
        "image": str(args.image.resolve()),
        "label": str(args.label.resolve()),
        "status": "PASS" if not report["errors"] else "FAIL",
        **report,
    }
    print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
