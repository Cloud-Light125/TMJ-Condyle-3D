"""Small, reproducible figures for reports.

The figures are derived from supplied MRI/label/prediction files. No synthetic
mask is ever created by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _normalize(slice_array: np.ndarray) -> np.ndarray:
    values = np.asarray(slice_array, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _representative_index(mask: np.ndarray | None, axis: int, shape: tuple[int, ...]) -> int:
    if mask is None or not np.any(mask):
        return shape[axis] // 2
    projection = np.any(mask, axis=tuple(i for i in range(mask.ndim) if i != axis))
    indices = np.flatnonzero(projection)
    return int(indices[len(indices) // 2]) if indices.size else shape[axis] // 2


def _slice_views(array: np.ndarray, indices: tuple[int, int, int]) -> list[np.ndarray]:
    z, y, x = indices
    return [array[z, :, :], array[:, y, :], array[:, :, x]]


def generate_case_figures(
    *,
    image: Any,
    ground_truth: Any | None,
    prediction: Any | None,
    output_dir: str | Path,
) -> list[Path]:
    """Generate the requested 1-5 case views when corresponding data exists."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("matplotlib is required for report figures") from exc
    import SimpleITK as sitk

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_array = sitk.GetArrayFromImage(image)
    gt_array = sitk.GetArrayFromImage(ground_truth) == 1 if ground_truth is not None else None
    pred_array = sitk.GetArrayFromImage(prediction) == 1 if prediction is not None else None
    reference_mask = gt_array if gt_array is not None and gt_array.any() else pred_array
    indices = tuple(
        _representative_index(reference_mask, axis, image_array.shape)
        for axis in range(3)
    )
    generated: list[Path] = []

    def save_figure(fig: Any, number: int, name: str) -> None:
        path = output / f"figure_{number:02d}_{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)

    views = _slice_views(image_array, indices)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, view, title in zip(axes, views, ("Axial", "Coronal", "Sagittal")):
        ax.imshow(_normalize(view), cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("MRI three-view (anonymous case)")
    save_figure(fig, 1, "mri_three_views")

    if gt_array is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, view, mask_view, title in zip(
            axes, views, _slice_views(gt_array, indices), ("Axial", "Coronal", "Sagittal")
        ):
            ax.imshow(_normalize(view), cmap="gray", origin="lower")
            ax.imshow(np.ma.masked_where(~mask_view, mask_view), cmap="autumn", alpha=0.55, origin="lower")
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle("Manual mandibular condyle mask overlay")
        save_figure(fig, 2, "gt_overlay")

    if pred_array is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, view, mask_view, title in zip(
            axes, views, _slice_views(pred_array, indices), ("Axial", "Coronal", "Sagittal")
        ):
            ax.imshow(_normalize(view), cmap="gray", origin="lower")
            ax.imshow(np.ma.masked_where(~mask_view, mask_view), cmap="winter", alpha=0.55, origin="lower")
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle("nnU-Net prediction overlay")
        save_figure(fig, 3, "prediction_overlay")

    if gt_array is not None and pred_array is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, view, gt_view, pred_view, title in zip(
            axes,
            views,
            _slice_views(gt_array, indices),
            _slice_views(pred_array, indices),
            ("Axial", "Coronal", "Sagittal"),
        ):
            ax.imshow(_normalize(view), cmap="gray", origin="lower")
            ax.imshow(np.ma.masked_where(~gt_view, gt_view), cmap="autumn", alpha=0.5, origin="lower")
            ax.imshow(np.ma.masked_where(~pred_view, pred_view), cmap="winter", alpha=0.5, origin="lower")
            ax.set_title(f"{title}: GT orange / prediction cyan")
            ax.axis("off")
        fig.suptitle("Ground truth versus prediction")
        save_figure(fig, 4, "gt_vs_prediction")

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        for mask, color, label in (
            (gt_array, "tab:orange", "GT"),
            (pred_array, "deepskyblue", "Prediction"),
        ):
            points = np.argwhere(mask)
            if len(points) > 5000:
                points = points[:: max(1, len(points) // 5000)]
            if len(points):
                ax.scatter(points[:, 2], points[:, 1], points[:, 0], s=2, alpha=0.25, c=color, label=label)
        ax.set_title("3D mask comparison (QC preview; use Slicer for surface)")
        ax.set_xlabel("I")
        ax.set_ylabel("J")
        ax.set_zlabel("K")
        ax.legend()
        save_figure(fig, 5, "3d_mask_comparison")

        try:
            from skimage.measure import marching_cubes
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "scikit-image is required for the true 3D surface figure"
            ) from exc

        def mesh(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
            if not np.any(mask):
                return None
            # Padding prevents a surface touching a volume boundary from being
            # clipped by marching cubes.
            padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
            foreground_voxels = int(mask.sum())
            step_size = (
                1
                if foreground_voxels <= 12_000
                else 2
                if foreground_voxels <= 40_000
                else 3
            )
            vertices, faces, _, _ = marching_cubes(
                padded.astype(np.float32),
                level=0.5,
                spacing=tuple(float(value) for value in image.GetSpacing()[::-1]),
                step_size=step_size,
            )
            vertices -= np.asarray(image.GetSpacing()[::-1], dtype=float)
            # marching_cubes returns z-y-x coordinates; the plot is x-y-z.
            return vertices[:, [2, 1, 0]], faces

        meshes = (
            (mesh(gt_array), "tab:orange", "GT"),
            (mesh(pred_array), "deepskyblue", "Prediction"),
        )
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        plotted_vertices: list[np.ndarray] = []
        for surface, color, label in meshes:
            if surface is None:
                continue
            vertices, faces = surface
            plotted_vertices.append(vertices)
            collection = Poly3DCollection(
                vertices[faces],
                alpha=0.52,
                facecolor=color,
                edgecolor="none",
                label=label,
            )
            ax.add_collection3d(collection)
        if plotted_vertices:
            all_vertices = np.concatenate(plotted_vertices, axis=0)
            mins = all_vertices.min(axis=0)
            maxs = all_vertices.max(axis=0)
            centers = (mins + maxs) / 2.0
            radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
            ax.set_xlim(centers[0] - radius, centers[0] + radius)
            ax.set_ylim(centers[1] - radius, centers[1] + radius)
            ax.set_zlim(centers[2] - radius, centers[2] + radius)
            ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_title("3D surface: GT orange / prediction cyan")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.legend(loc="upper right")
        save_figure(fig, 6, "3d_surface")
    return generated
