"""Physical-space binary segmentation metrics."""

from __future__ import annotations

import numpy as np


def dice_coefficient(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    gt_count = int(gt.sum())
    pred_count = int(pred.sum())
    if gt_count == 0 and pred_count == 0:
        return 1.0
    denominator = gt_count + pred_count
    return float(2 * np.logical_and(gt, pred).sum() / denominator) if denominator else 0.0


def iou_score(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(gt, pred).sum() / union)


def hd95_mm(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
) -> float:
    """95th percentile symmetric Hausdorff distance in millimetres.

    SimpleITK spacing is x-y-z while NumPy data is z-y-x, hence the reversed
    sampling tuple passed to scipy's distance transform.
    """

    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if not gt.any() and not pred.any():
        return 0.0
    if not gt.any() or not pred.any():
        return float("inf")
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("scipy is required to calculate HD95") from exc

    connectivity = ndimage.generate_binary_structure(3, 1)
    gt_surface = np.logical_xor(gt, ndimage.binary_erosion(gt, structure=connectivity))
    pred_surface = np.logical_xor(
        pred, ndimage.binary_erosion(pred, structure=connectivity)
    )
    sampling_zyx = tuple(float(value) for value in spacing_xyz[::-1])
    distance_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=sampling_zyx)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=sampling_zyx)
    distances = np.concatenate(
        [distance_to_gt[pred_surface], distance_to_pred[gt_surface]]
    )
    return float(np.percentile(distances, 95))


def case_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, float]:
    gt = np.asarray(ground_truth) == 1
    pred = np.asarray(prediction) == 1
    voxel_volume = float(np.prod(spacing_xyz))
    gt_volume = float(gt.sum() * voxel_volume)
    pred_volume = float(pred.sum() * voxel_volume)
    return {
        "dice": dice_coefficient(gt, pred),
        "iou": iou_score(gt, pred),
        "hd95_mm": hd95_mm(gt, pred, spacing_xyz=spacing_xyz),
        "absolute_volume_difference_mm3": abs(gt_volume - pred_volume),
        "gt_volume_mm3": gt_volume,
        "prediction_volume_mm3": pred_volume,
    }
