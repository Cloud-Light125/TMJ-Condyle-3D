from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from tmj_condyle.evaluation.figures import generate_case_figures
from tmj_condyle.utils.io import read_image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate anonymous MRI/GT/prediction figures from real supplied files."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--prediction", type=Path)
    args = parser.parse_args()
    generated = generate_case_figures(
        image=read_image(args.image),
        ground_truth=read_image(args.ground_truth) if args.ground_truth else None,
        prediction=read_image(args.prediction) if args.prediction else None,
        output_dir=args.output_dir,
    )
    for path in generated:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
