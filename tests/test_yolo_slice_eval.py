"""Tests for per-slice mAP evaluation helpers (yolo_slice_eval.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis.flir_subgroup.yolo_slice_eval import (
    assign_gt_slices,
    compute_frozen_thresholds,
    evaluate_per_slice,
    load_frozen_thresholds,
    save_frozen_thresholds,
)
from src.analysis.flir_subgroup.yolo_slice_stats import (
    POSITION_BIN_ORDER,
    SIZE_BIN_ORDER,
    YoloSliceThresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dataset_yaml(root: Path, *, split: str = "train") -> Path:
    """Write a minimal YOLO dataset YAML pointing at root/<split>/images."""
    images_dir = root / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = root / f"{split}.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                f"train: {(root / 'train' / 'images').resolve()}",
                f"val: {(root / 'val' / 'images').resolve()}",
                f"test: {(root / 'test' / 'images').resolve()}",
                "names:",
                "  0: person",
                "nc: 1",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def _write_labels(labels_dir: Path, stem: str, boxes: list[tuple]) -> None:
    """Write YOLO-format label file: class cx cy w h per line."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(cls)} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}"
             for cls, cx, cy, w, h in boxes]
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tiny_png(path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)


# ---------------------------------------------------------------------------
# compute_frozen_thresholds
# ---------------------------------------------------------------------------

def test_compute_frozen_thresholds_basic(tmp_path: Path) -> None:
    """Thresholds are computed from the training split and q33 <= q67."""
    labels_dir = tmp_path / "train" / "labels"
    _write_labels(labels_dir, "img_a", [(0, 0.2, 0.2, 0.05, 0.05)])   # small area 0.0025
    _write_labels(labels_dir, "img_b", [(0, 0.5, 0.5, 0.20, 0.20)])   # medium area 0.04
    _write_labels(labels_dir, "img_c", [(0, 0.8, 0.8, 0.50, 0.50)])   # large area 0.25
    images_dir = tmp_path / "train" / "images"
    for stem in ("img_a", "img_b", "img_c"):
        _write_tiny_png(images_dir / f"{stem}.png")
    yaml_path = _write_dataset_yaml(tmp_path)

    thresh = compute_frozen_thresholds(yaml_path)
    assert isinstance(thresh, YoloSliceThresholds)
    assert thresh.q33 <= thresh.q67


def test_frozen_thresholds_roundtrip(tmp_path: Path) -> None:
    thresh = YoloSliceThresholds(q33=0.012, q67=0.045)
    save_frozen_thresholds(thresh, tmp_path)
    loaded = load_frozen_thresholds(tmp_path / "slice_thresholds.json")
    assert abs(loaded.q33 - thresh.q33) < 1e-9
    assert abs(loaded.q67 - thresh.q67) < 1e-9


# ---------------------------------------------------------------------------
# assign_gt_slices
# ---------------------------------------------------------------------------

def test_assign_gt_slices_columns(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    _write_labels(labels_dir, "img_a", [(0, 0.5, 0.5, 0.3, 0.3)])
    names = {0: "person"}
    thresh = YoloSliceThresholds(q33=0.05, q67=0.15)
    df = assign_gt_slices(labels_dir, names, thresh)
    expected_cols = {
        "image_stem", "instance_index", "class_idx", "class_label",
        "bbox_center_x_norm", "bbox_center_y_norm", "bbox_w_norm", "bbox_h_norm",
        "bbox_area_ratio", "size_bin", "position_row_bin", "position_col_bin",
        "position_bin", "slice_key",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1


def test_assign_gt_slices_size_bin_large(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    # area = 0.5*0.5 = 0.25; q67=0.1 → large
    _write_labels(labels_dir, "img_a", [(0, 0.5, 0.5, 0.5, 0.5)])
    thresh = YoloSliceThresholds(q33=0.05, q67=0.10)
    df = assign_gt_slices(labels_dir, {0: "person"}, thresh)
    assert df.iloc[0]["size_bin"] == "large"


def test_assign_gt_slices_size_bin_small(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    # area = 0.02*0.02 = 0.0004; q33=0.01 → small
    _write_labels(labels_dir, "img_a", [(0, 0.1, 0.1, 0.02, 0.02)])
    thresh = YoloSliceThresholds(q33=0.01, q67=0.10)
    df = assign_gt_slices(labels_dir, {0: "person"}, thresh)
    assert df.iloc[0]["size_bin"] == "small"


def test_assign_gt_slices_position_top_left(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    # center at (0.1, 0.1) → top-left cell
    _write_labels(labels_dir, "img_a", [(0, 0.1, 0.1, 0.05, 0.05)])
    thresh = YoloSliceThresholds(q33=0.001, q67=0.1)
    df = assign_gt_slices(labels_dir, {0: "person"}, thresh)
    assert df.iloc[0]["position_bin"] == "top_left"


def test_assign_gt_slices_position_bottom_right(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    # center at (0.9, 0.9) → bottom-right cell
    _write_labels(labels_dir, "img_a", [(0, 0.9, 0.9, 0.05, 0.05)])
    thresh = YoloSliceThresholds(q33=0.001, q67=0.1)
    df = assign_gt_slices(labels_dir, {0: "person"}, thresh)
    assert df.iloc[0]["position_bin"] == "bottom_right"


# ---------------------------------------------------------------------------
# evaluate_per_slice (synthetic, no GPU, no real model)
# ---------------------------------------------------------------------------

def _build_test_split_yaml(tmp_path: Path) -> Path:
    images_dir = tmp_path / "test" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {tmp_path.resolve()}",
            f"train: {(tmp_path / 'train' / 'images').resolve()}",
            f"val: {(tmp_path / 'test' / 'images').resolve()}",
            f"test: {(tmp_path / 'test' / 'images').resolve()}",
            "names:",
            "  0: person",
            "nc: 1",
        ]),
        encoding="utf-8",
    )
    return yaml_path


# ---------------------------------------------------------------------------
# Internal slice attribution helpers (unit-level, no model needed)
# ---------------------------------------------------------------------------

def test_assign_pred_slices_bottom_right_large(tmp_path: Path) -> None:
    """Predicted box in bottom-right corner → 'large', 'bottom_right'."""
    from src.analysis.flir_subgroup.yolo_slice_eval import _assign_pred_slices
    import numpy as _np
    # xyxy normalised; box from 0.7 to 1.0 in both x and y → w=h=0.3, area=0.09
    pred = _np.array([[0.7, 0.7, 1.0, 1.0]], dtype=_np.float32)
    thresh = YoloSliceThresholds(q33=0.003, q67=0.05)  # 0.09 > q67 → large
    slices = _assign_pred_slices(pred, thresh, "person")
    assert len(slices) == 1
    cls, sz, pos = slices[0]
    assert cls == "person"
    assert sz == "large"
    assert pos == "bottom_right"


def test_assign_pred_slices_top_left_small(tmp_path: Path) -> None:
    from src.analysis.flir_subgroup.yolo_slice_eval import _assign_pred_slices
    import numpy as _np
    # box from 0.05 to 0.10 → w=h=0.05, area=0.0025; center at 0.075 → top-left
    pred = _np.array([[0.05, 0.05, 0.10, 0.10]], dtype=_np.float32)
    thresh = YoloSliceThresholds(q33=0.01, q67=0.05)  # 0.0025 < q33 → small
    slices = _assign_pred_slices(pred, thresh, "person")
    cls, sz, pos = slices[0]
    assert sz == "small"
    assert pos == "top_left"


def test_assign_pred_slices_empty() -> None:
    from src.analysis.flir_subgroup.yolo_slice_eval import _assign_pred_slices
    import numpy as _np
    result = _assign_pred_slices(_np.zeros((0, 4), dtype=_np.float32),
                                 YoloSliceThresholds(q33=0.01, q67=0.05), "person")
    assert result == []


# ---------------------------------------------------------------------------
# evaluate_per_slice: structure test using internal _build_slice_rows helper
# that bypasses model loading
# ---------------------------------------------------------------------------

def test_per_slice_output_has_27_slices_plus_overall() -> None:
    """Output rows must contain exactly 27 slice keys + 1 overall row."""
    from src.analysis.flir_subgroup.yolo_slice_eval import _build_per_slice_rows_from_entries

    thresh = YoloSliceThresholds(q33=0.01, q67=0.05)
    names = {0: "person"}
    # Empty inputs: all 27 slices should still appear (with NaN metrics)
    rows = _build_per_slice_rows_from_entries({}, {}, thresh, names)
    slice_rows = [r for r in rows if r["slice_key"] != "overall"]
    overall_rows = [r for r in rows if r["slice_key"] == "overall"]
    assert len(slice_rows) == 27, f"Expected 27, got {len(slice_rows)}"
    assert len(overall_rows) == 1
