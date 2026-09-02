from __future__ import annotations

import numpy as np
import pytest

from tmj_condyle.evaluation.metrics import case_metrics, dice_coefficient, hd95_mm, iou_score


def test_perfect_metrics():
    mask = np.zeros((3, 5, 5), dtype=np.uint8)
    mask[1, 2, 2] = 1
    assert dice_coefficient(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0
    assert hd95_mm(mask, mask, spacing_xyz=(1.0, 1.0, 5.0)) == 0.0


def test_physical_hd95_uses_spacing():
    gt = np.zeros((3, 5, 5), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[1, 2, 2] = 1
    pred[1, 2, 3] = 1
    metrics = case_metrics(gt, pred, spacing_xyz=(0.25, 0.25, 5.0))
    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["hd95_mm"] == pytest.approx(0.25)


def test_empty_both_metrics_are_defined():
    empty = np.zeros((2, 3, 4), dtype=np.uint8)
    assert dice_coefficient(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0
    assert hd95_mm(empty, empty, spacing_xyz=(1.0, 1.0, 1.0)) == 0.0


def test_empty_one_side_has_infinite_hd95():
    empty = np.zeros((2, 3, 4), dtype=np.uint8)
    nonempty = empty.copy()
    nonempty[0, 0, 0] = 1
    assert dice_coefficient(empty, nonempty) == 0.0
    assert iou_score(empty, nonempty) == 0.0
    assert np.isinf(hd95_mm(empty, nonempty, spacing_xyz=(1.0, 1.0, 1.0)))
