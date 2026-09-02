"""Grouped deterministic cross-validation splits for nnU-Net v2."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..config import N_FOLDS, validate_case_id


def build_grouped_splits(
    cases: Iterable[dict[str, str]],
    *,
    n_splits: int = N_FOLDS,
    seed: int = 20260902,
) -> list[dict[str, list[str]]]:
    rows = list(cases)
    case_ids = [validate_case_id(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case_id in split input")
    if len(case_ids) < n_splits:
        raise ValueError(f"At least {n_splits} cases are required for {n_splits}-fold CV")

    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get("group_id") or row["case_id"]).strip()
        if not group_id:
            raise ValueError(f"Empty group_id for {row['case_id']}")
        groups[group_id].append(row["case_id"])
    if len(groups) < n_splits:
        raise ValueError(
            f"At least {n_splits} distinct group_id values are required; found {len(groups)}"
        )

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    fold_cases: list[list[str]] = [[] for _ in range(n_splits)]
    for group_id, group_case_ids in group_items:
        target = min(range(n_splits), key=lambda index: (len(fold_cases[index]), index))
        fold_cases[target].extend(sorted(group_case_ids))

    all_cases = set(case_ids)
    splits: list[dict[str, list[str]]] = []
    for fold_index in range(n_splits):
        val = sorted(fold_cases[fold_index])
        train = sorted(all_cases.difference(val))
        splits.append({"train": train, "val": val})

    validate_splits(splits, rows, n_splits=n_splits)
    return splits


def validate_splits(
    splits: list[dict[str, list[str]]],
    cases: Iterable[dict[str, str]],
    *,
    n_splits: int = N_FOLDS,
) -> None:
    rows = list(cases)
    expected_cases = {row["case_id"] for row in rows}
    case_to_group = {row["case_id"]: str(row.get("group_id") or row["case_id"]) for row in rows}
    if len(splits) != n_splits:
        raise ValueError(f"Expected {n_splits} folds, found {len(splits)}")
    seen_validation: list[str] = []
    for index, split in enumerate(splits):
        train = set(split.get("train", []))
        val = set(split.get("val", []))
        if not train or not val:
            raise ValueError(f"Fold {index} has an empty train or validation set")
        if train & val:
            raise ValueError(f"Fold {index} contains train/validation case overlap")
        if not train | val == expected_cases:
            raise ValueError(f"Fold {index} does not cover the complete case set")
        train_groups = {case_to_group[case_id] for case_id in train}
        val_groups = {case_to_group[case_id] for case_id in val}
        if train_groups & val_groups:
            raise ValueError(f"Fold {index} has patient/group leakage: {train_groups & val_groups}")
        seen_validation.extend(val)
    if sorted(seen_validation) != sorted(expected_cases):
        raise ValueError("Validation coverage is not exactly once per case")


def write_splits(
    splits: list[dict[str, list[str]]],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return destination


def write_fold_assignments(
    splits: list[dict[str, list[str]]],
    cases: Iterable[dict[str, str]],
    path: str | Path,
) -> Path:
    import csv

    case_to_group = {row["case_id"]: str(row.get("group_id") or row["case_id"]) for row in cases}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "group_id", "fold"])
        writer.writeheader()
        for fold, split in enumerate(splits):
            for case_id in split["val"]:
                writer.writerow(
                    {"case_id": case_id, "group_id": case_to_group[case_id], "fold": fold}
                )
    return destination
