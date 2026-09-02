from __future__ import annotations

import pytest

from tmj_condyle.data.splits import build_grouped_splits, validate_splits


def _cases():
    return [
        {"case_id": "case_001_L", "group_id": "group_001"},
        {"case_id": "case_001_R", "group_id": "group_001"},
        {"case_id": "case_002_L", "group_id": "group_002"},
        {"case_id": "case_003_L", "group_id": "group_003"},
        {"case_id": "case_004_L", "group_id": "group_004"},
        {"case_id": "case_005_L", "group_id": "group_005"},
        {"case_id": "case_006_L", "group_id": "group_006"},
    ]


def test_grouped_five_fold_has_no_leakage_and_full_coverage():
    cases = _cases()
    splits = build_grouped_splits(cases, n_splits=5, seed=123)
    validate_splits(splits, cases, n_splits=5)
    validation = [case_id for split in splits for case_id in split["val"]]
    assert sorted(validation) == sorted(case["case_id"] for case in cases)
    for split in splits:
        train_groups = {
            next(case["group_id"] for case in cases if case["case_id"] == case_id)
            for case_id in split["train"]
        }
        val_groups = {
            next(case["group_id"] for case in cases if case["case_id"] == case_id)
            for case_id in split["val"]
        }
        assert train_groups.isdisjoint(val_groups)


def test_too_few_groups_is_rejected():
    cases = [{"case_id": f"case_{i:03d}", "group_id": "same"} for i in range(5)]
    with pytest.raises(ValueError, match="distinct group_id"):
        build_grouped_splits(cases, n_splits=5)
