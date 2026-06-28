"""Construct slice-balanced bbox layouts for FM augmentation.

The default FM augmentation reuses each real image's exact GT layout, so the
synthetic set inherits the real data's skewed distribution over the 27 slices
(1 class x 3 size bins x 9 position bins). This module instead builds *new*
multi-box layouts that over-represent rare slices: each synthetic box is a real
box bootstrapped from a target (size, position) slice (guaranteed realistic,
in-bounds, correct slice), lightly jittered within its 3x3 cell, and recombined
with boxes from other deficit slices into novel layouts. Placing the full
per-slice deficit pool brings every slice up to the max real slice count, so the
combined real+synth distribution is flat.

The resulting layouts are returned as ``YOLOTrainSample`` objects (same type the
FM generation path already consumes), so they plug straight into
``_write_generated_coco``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analysis.flir_subgroup.yolo_slice_stats import (
    POSITION_BIN_ORDER,
    build_slice_counts_table,
    load_yolo_slice_dataset,
)


def compute_slice_pool(
    dataset_yaml: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Load the real YOLO train split and compute per-slice instance tables."""

    ds = load_yolo_slice_dataset(dataset_yaml)
    if ds.instance_df.empty:
        raise ValueError(f"No GT instances found for slice balancing in {dataset_yaml}")
    class_labels = sorted(ds.instance_df["class_label"].astype(str).unique().tolist())
    counts = build_slice_counts_table(ds.instance_df, class_labels)
    return ds.instance_df, counts, ds.thresholds


def _cell_bounds(position_bin: str) -> Tuple[float, float, float, float]:
    """(x_lo, x_hi, y_lo, y_hi) of the 3x3 cell for a position bin label."""

    idx = POSITION_BIN_ORDER.index(position_bin)
    row, col = idx // 3, idx % 3
    return col / 3.0, (col + 1) / 3.0, row / 3.0, (row + 1) / 3.0


def _iou_cxcywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 1e-9 else 0.0


def _jitter_box(
    box: Sequence[float], position_bin: str, *, jitter_frac: float, rng: random.Random
) -> Tuple[float, float, float, float]:
    """Jitter a box center within its 3x3 cell, keeping it in-bounds.

    Width/height are preserved (so the size_bin is unchanged) and the center stays
    inside the cell (so the position_bin is unchanged).
    """

    cx, cy, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    x_lo, x_hi, y_lo, y_hi = _cell_bounds(position_bin)
    span = jitter_frac / 3.0
    cx += rng.uniform(-span, span)
    cy += rng.uniform(-span, span)
    # keep center in cell AND box in [0,1]
    cx = min(max(cx, x_lo, w / 2.0), x_hi, 1.0 - w / 2.0)
    cy = min(max(cy, y_lo, h / 2.0), y_hi, 1.0 - h / 2.0)
    return cx, cy, w, h


def build_balanced_layouts(
    *,
    instance_df: pd.DataFrame,
    counts: pd.DataFrame,
    n_obj_distribution: Sequence[int],
    target: str = "max",
    seed: int = 7,
    jitter_frac: float = 0.15,
    max_pair_iou: float = 0.5,
    placement_tries: int = 8,
    max_layouts: int | None = None,
) -> List[Any]:
    """Build slice-balanced synthetic layouts as ``YOLOTrainSample`` objects.

    ``max_layouts`` caps the number of emitted layouts (smoke testing); ``None``
    places the entire per-slice deficit pool.
    """

    # Deferred import avoids a circular import with yolo_experiment_b.
    from src.algorithms.training.yolo_experiment_b import YOLOBox, YOLOTrainSample

    rng = random.Random(int(seed))

    if str(target) == "max":
        target_count = int(counts["count"].max())
    elif str(target) == "mean":
        target_count = int(round(float(counts["count"].mean())))
    else:
        target_count = int(target)

    # Per-slice real boxes to bootstrap from, keyed by (class_label, size_bin, position_bin).
    slice_boxes: dict[tuple, np.ndarray] = {}
    grouped = instance_df.groupby(
        ["class_label", "size_bin", "position_bin"], observed=True
    )
    for key, grp in grouped:
        slice_boxes[tuple(str(k) for k in key)] = grp[
            ["bbox_center_x_norm", "bbox_center_y_norm", "bbox_w_norm", "bbox_h_norm"]
        ].to_numpy(dtype=np.float64)

    if "class_idx" in instance_df.columns:
        name_to_class_id = {
            str(lbl): int(cid)
            for lbl, cid in zip(instance_df["class_label"], instance_df["class_idx"])
        }
    else:
        name_to_class_id = {}

    # Build the deficit pool: deficit copies of each slice that has real boxes.
    pool: List[tuple] = []
    for _, row in counts.iterrows():
        key = (str(row["class_label"]), str(row["size_bin"]), str(row["position_bin"]))
        deficit = max(0, target_count - int(row["count"]))
        if deficit > 0 and key in slice_boxes:
            pool.extend([key] * deficit)
    rng.shuffle(pool)

    # n_objects sampling pool (real per-image distribution, excluding empty images).
    n_obj_choices = [int(n) for n in n_obj_distribution if int(n) >= 1]
    if not n_obj_choices:
        n_obj_choices = [1]

    layouts: List[Any] = []
    cursor = 0
    n_pool = len(pool)
    while cursor < n_pool:
        if max_layouts is not None and len(layouts) >= max_layouts:
            break
        n_obj = rng.choice(n_obj_choices)
        n_obj = min(n_obj, n_pool - cursor)
        slot_keys = pool[cursor:cursor + n_obj]
        cursor += n_obj

        placed: List[Tuple[float, float, float, float]] = []
        placed_class: List[int] = []
        for key in slot_keys:
            class_label, _size_bin, position_bin = key
            real_boxes = slice_boxes[key]
            box = None
            for _ in range(placement_tries):
                cand = real_boxes[rng.randrange(real_boxes.shape[0])]
                jb = _jitter_box(cand, position_bin, jitter_frac=jitter_frac, rng=rng)
                if all(_iou_cxcywh(jb, p) <= max_pair_iou for p in placed):
                    box = jb
                    break
            if box is None:  # accept last candidate even if it overlaps (keeps balance exact)
                box = jb
            placed.append(box)
            placed_class.append(int(name_to_class_id.get(class_label, 0)))

        idx = len(layouts)
        boxes = [
            YOLOBox(class_id=c, x_center=b[0], y_center=b[1], width=b[2], height=b[3])
            for c, b in zip(placed_class, placed)
        ]
        layouts.append(
            YOLOTrainSample(
                index=idx,
                image_path=Path(f"synthlayout_{idx:06d}.png"),
                label_path=Path(f"synthlayout_{idx:06d}.txt"),
                boxes=boxes,
            )
        )

    return layouts
