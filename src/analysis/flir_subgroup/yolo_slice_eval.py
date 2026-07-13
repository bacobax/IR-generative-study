"""Per-slice mAP evaluation for YOLO detectors.

Slices are defined as ``size_bin × position_bin`` cross-products.  With a
single-class dataset (e.g. v18 person) this gives 3 × 9 = 27 slice cells.

Attribution rules
-----------------
- **TP**: inherits the slice of the *matched GT* box.
- **FN** (missed GT): counted in the *GT* box's slice.
- **FP**: binned by the *predicted box's own geometry* (size_bin and
  position_bin computed from the prediction itself, using frozen tertile
  thresholds).

This means each slice's AP curve is built from:
  - predictions whose *matched GT* (for TPs) or *own geometry* (for FPs)
    falls in that slice,
  - against the GT count for that slice (denominator for recall).

Usage::

    from src.analysis.flir_subgroup.yolo_slice_eval import (
        compute_frozen_thresholds,
        evaluate_per_slice,
    )

    thresholds = compute_frozen_thresholds("data/derived/yolo-test-ds_v18/full_train.yaml")
    results = evaluate_per_slice(
        weights="artifacts/checkpoints/yolo/exp_v18_scratch_yolo11n/default_aug/best.pt",
        test_yaml="data/derived/yolo-test-ds_v18/test.yaml",
        thresholds=thresholds,
        output_dir=Path("artifacts/analysis/yolo/exp_v18_scratch_yolo11n/default_aug"),
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.analysis.flir_subgroup.yolo_slice_stats import (
    POSITION_BIN_ORDER,
    SIZE_BIN_ORDER,
    YoloSliceThresholds,
    add_position_bin_columns,
    assign_bins_from_thresholds,
    load_yolo_slice_dataset,
)
from src.evaluation.detection_metrics import (
    DetectionPrediction,
    box_iou_matrix,
    summarize_match_arrays,
)


# ---------------------------------------------------------------------------
# Frozen thresholds
# ---------------------------------------------------------------------------

def compute_frozen_thresholds(train_dataset_yaml: str | Path) -> YoloSliceThresholds:
    """Compute size-bin tertile thresholds from the training split.

    Loads the YOLO train split, computes the 33rd and 67th percentile of
    ``bbox_area_ratio = w * h`` (normalised) across all GT boxes, and returns
    frozen thresholds to be reused for both training-time rare-slice selection
    and evaluation-time slice assignment.

    Args:
        train_dataset_yaml: Path to the YOLO dataset YAML whose *train* key
            points at the full-train split (e.g.
            ``data/derived/yolo-test-ds_v18/full_train.yaml``).

    Returns:
        A :class:`YoloSliceThresholds` with ``q33`` and ``q67`` attributes.
    """
    ds = load_yolo_slice_dataset(train_dataset_yaml)
    return ds.thresholds


def save_frozen_thresholds(thresholds: YoloSliceThresholds, output_dir: Path) -> Path:
    """Persist thresholds to ``<output_dir>/slice_thresholds.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "slice_thresholds.json"
    path.write_text(json.dumps(thresholds.to_dict(), indent=2))
    return path


def load_frozen_thresholds(path: str | Path) -> YoloSliceThresholds:
    """Load thresholds previously written by :func:`save_frozen_thresholds`."""
    payload = json.loads(Path(path).read_text())
    return YoloSliceThresholds(q33=float(payload["q33"]), q67=float(payload["q67"]))


# ---------------------------------------------------------------------------
# GT slice assignment
# ---------------------------------------------------------------------------

def assign_gt_slices(
    labels_dir: str | Path,
    names: dict[int, str],
    thresholds: YoloSliceThresholds,
) -> pd.DataFrame:
    """Parse YOLO-format label files and assign each GT box to a slice.

    Each row in the returned DataFrame describes one GT box with columns:
    ``image_stem``, ``instance_index``, ``class_idx``, ``class_label``,
    ``bbox_center_x_norm``, ``bbox_center_y_norm``, ``bbox_w_norm``,
    ``bbox_h_norm``, ``bbox_area_ratio``, ``size_bin``, ``position_row_bin``,
    ``position_col_bin``, ``position_bin``, ``slice_key``.

    Args:
        labels_dir: Directory containing ``.txt`` YOLO label files.
        names: ``{class_idx: class_name}`` mapping.
        thresholds: Frozen size-bin thresholds from the training split.
    """
    labels_dir = Path(labels_dir)
    rows: list[dict] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for ann_index, line in enumerate(text.splitlines()):
            parts = line.split()
            if len(parts) != 5:
                continue
            class_idx = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            rows.append(
                {
                    "image_stem": stem,
                    "instance_index": ann_index,
                    "class_idx": class_idx,
                    "class_label": names.get(class_idx, str(class_idx)),
                    "bbox_center_x_norm": cx,
                    "bbox_center_y_norm": cy,
                    "bbox_w_norm": w,
                    "bbox_h_norm": h,
                    "bbox_area_ratio": w * h,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = add_position_bin_columns(df)
    df["size_bin"] = assign_bins_from_thresholds(
        df["bbox_area_ratio"], thresholds.q33, thresholds.q67
    )
    df["slice_key"] = list(
        zip(
            df["class_label"].astype(str),
            df["size_bin"].astype(str),
            df["position_bin"].astype(str),
        )
    )
    return df


# ---------------------------------------------------------------------------
# Per-slice AP computation
# ---------------------------------------------------------------------------

def _assign_pred_slices(
    pred_boxes_xyxy_norm: np.ndarray,
    thresholds: YoloSliceThresholds,
    class_label: str,
) -> list[tuple[str, str, str]]:
    """Assign each predicted box to a (class_label, size_bin, position_bin) slice.

    ``pred_boxes_xyxy_norm`` has shape ``(N, 4)`` in normalised xyxy format.
    Returns a list of N slice tuples.
    """
    if len(pred_boxes_xyxy_norm) == 0:
        return []

    x1, y1, x2, y2 = (
        pred_boxes_xyxy_norm[:, 0],
        pred_boxes_xyxy_norm[:, 1],
        pred_boxes_xyxy_norm[:, 2],
        pred_boxes_xyxy_norm[:, 3],
    )
    w = np.clip(x2 - x1, 0.0, 1.0)
    h = np.clip(y2 - y1, 0.0, 1.0)
    cx = np.clip(x1 + w / 2.0, 0.0, 1.0)
    cy = np.clip(y1 + h / 2.0, 0.0, 1.0)
    area = w * h

    # size bin
    size_bins = np.select(
        [area <= thresholds.q33, area <= thresholds.q67],
        ["small", "medium"],
        default="large",
    )

    # position bin (3×3 grid)
    col_idx = np.clip(np.floor(cx * 3.0).astype(int), 0, 2)
    row_idx = np.clip(np.floor(cy * 3.0).astype(int), 0, 2)
    _row_labels = ("top", "middle", "bottom")
    _col_labels = ("left", "center", "right")

    slices = []
    for sz, ri, ci in zip(size_bins, row_idx, col_idx):
        pos = f"{_row_labels[ri]}_{_col_labels[ci]}"
        slices.append((class_label, sz, pos))
    return slices


def _iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> np.ndarray:
    """Return (N, M) IoU matrix for two sets of xyxy boxes."""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    return box_iou_matrix(boxes_a.detach().cpu().numpy(), boxes_b.detach().cpu().numpy())


def _build_per_slice_rows_from_entries(
    slice_entries: dict,
    gt_per_slice: dict,
    thresholds: YoloSliceThresholds,
    names: dict[int, str],
) -> list[dict]:
    """Build the dense per-slice metrics table from pre-aggregated entries.

    ``slice_entries`` maps ``(class_label, size_bin, position_bin)`` tuples to a
    list of ``(tp_vec, conf, pred_cls_int)`` tuples — one entry per prediction
    routed to that slice.  ``gt_per_slice`` maps the same keys to GT counts.

    Returns a list of row dicts (27 slice rows + 1 overall row).  Slices with
    neither GT nor predictions receive ``NaN`` metric values.
    """
    all_class_labels = sorted({names[k] for k in names})

    per_slice_rows: list[dict] = []
    all_tp_flat: list[np.ndarray] = []
    all_conf_flat: list[float] = []
    all_pred_cls_flat: list[int] = []
    all_gt_cls_flat: list[int] = []

    for cls_lbl in all_class_labels:
        cls_idx = next((k for k, v in names.items() if v == cls_lbl), 0)
        for sz in SIZE_BIN_ORDER:
            for pos in POSITION_BIN_ORDER:
                key = (cls_lbl, sz, pos)
                entries = slice_entries.get(key, [])
                n_gt_slice = gt_per_slice.get(key, 0)

                if n_gt_slice == 0 and len(entries) == 0:
                    per_slice_rows.append(
                        {
                            "class": cls_lbl,
                            "size_bin": sz,
                            "position_bin": pos,
                            "slice_key": str(key),
                            "n_gt": 0,
                            "n_pred": 0,
                            "AP50": float("nan"),
                            "AP50_95": float("nan"),
                            "precision": float("nan"),
                            "recall": float("nan"),
                        }
                    )
                    continue

                if len(entries) == 0:
                    per_slice_rows.append(
                        {
                            "class": cls_lbl,
                            "size_bin": sz,
                            "position_bin": pos,
                            "slice_key": str(key),
                            "n_gt": n_gt_slice,
                            "n_pred": 0,
                            "AP50": 0.0,
                            "AP50_95": 0.0,
                            "precision": 0.0,
                            "recall": 0.0,
                        }
                    )
                    all_gt_cls_flat.extend([cls_idx] * n_gt_slice)
                    continue

                tp_arr = np.stack([e[0] for e in entries])
                conf_arr = np.array([e[1] for e in entries])
                pred_cls_arr_s = np.array([e[2] for e in entries])
                target_cls_arr = np.full(n_gt_slice, cls_idx, dtype=np.int32)

                all_tp_flat.append(tp_arr)
                all_conf_flat.extend(conf_arr.tolist())
                all_pred_cls_flat.extend(pred_cls_arr_s.tolist())
                all_gt_cls_flat.extend([cls_idx] * n_gt_slice)

                result_rows = summarize_match_arrays(
                    tp=tp_arr,
                    conf=conf_arr,
                    pred_cls=pred_cls_arr_s,
                    target_cls=target_cls_arr,
                    class_ids=[cls_idx],
                )
                result_row = result_rows[0]
                ap50 = float(result_row["AP50"])
                ap50_95 = float(result_row["AP50_95"])
                p = float(result_row["precision"])
                r = float(result_row["recall"])

                per_slice_rows.append(
                    {
                        "class": cls_lbl,
                        "size_bin": sz,
                        "position_bin": pos,
                        "slice_key": str(key),
                        "n_gt": n_gt_slice,
                        "n_pred": len(entries),
                        "AP50": ap50,
                        "AP50_95": ap50_95,
                        "precision": p,
                        "recall": r,
                    }
                )

    overall_row: dict = {
        "class": "overall",
        "size_bin": "all",
        "position_bin": "all",
        "slice_key": "overall",
        "n_gt": sum(gt_per_slice.values()),
        "n_pred": sum(len(e) for e in slice_entries.values()),
        "AP50": float("nan"),
        "AP50_95": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
    }
    if all_tp_flat and all_gt_cls_flat:
        tp_all = np.concatenate(all_tp_flat, axis=0)
        conf_all = np.array(all_conf_flat)
        pred_cls_all = np.array(all_pred_cls_flat)
        tgt_cls_all = np.array(all_gt_cls_flat)
        result_rows = summarize_match_arrays(
            tp=tp_all,
            conf=conf_all,
            pred_cls=pred_cls_all,
            target_cls=tgt_cls_all,
            class_ids=sorted(names),
        )
        finite = lambda key: [float(row[key]) for row in result_rows if np.isfinite(float(row[key]))]
        for key, out_key in (("AP50", "AP50"), ("AP50_95", "AP50_95"), ("precision", "precision"), ("recall", "recall")):
            values = finite(key)
            if values:
                overall_row[out_key] = float(np.mean(values))

    per_slice_rows.append(overall_row)
    return per_slice_rows


def _resolve_eval_split_paths(test_yaml: str | Path) -> tuple[list[Path], Path, dict[int, str]]:
    """Resolve image paths, labels dir, and class names for a YOLO eval YAML."""

    from src.core.configs.config_loader import load_yaml

    test_yaml = Path(test_yaml)
    ds_payload = load_yaml(test_yaml)
    ds_root = Path(str(ds_payload.get("path", test_yaml.parent))).resolve()
    split_key = "test" if "test" in ds_payload else "val"
    test_images_rel = str(ds_payload[split_key])
    test_images_dir = Path(test_images_rel)
    if not test_images_dir.is_absolute():
        test_images_dir = (ds_root / test_images_rel).resolve()
    if test_images_dir.parent.name == "images":
        labels_dir = test_images_dir.parent.parent / "labels" / test_images_dir.name
    elif test_images_dir.name == "images":
        labels_dir = test_images_dir.parent / "labels"
    else:
        labels_dir = test_images_dir.parents[1] / "labels" / test_images_dir.name
    raw_names = ds_payload.get("names", {})
    if isinstance(raw_names, dict):
        names = {int(k): str(v) for k, v in raw_names.items()}
    else:
        names = {idx: str(value) for idx, value in enumerate(raw_names)}
    image_paths = sorted(p for p in test_images_dir.glob("*") if p.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No test images found under {test_images_dir}")
    return image_paths, labels_dir.resolve(), names


def _load_gt_arrays_for_image(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    gt_boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
    gt_cls_arr = np.zeros(0, dtype=np.int32)
    if not label_path.exists():
        return gt_boxes_xyxy, gt_cls_arr
    text = label_path.read_text(encoding="utf-8").strip()
    gt_rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_i = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = np.clip(cx - w / 2, 0.0, 1.0)
        y1 = np.clip(cy - h / 2, 0.0, 1.0)
        x2 = np.clip(cx + w / 2, 0.0, 1.0)
        y2 = np.clip(cy + h / 2, 0.0, 1.0)
        gt_rows.append((cls_i, x1, y1, x2, y2))
    if gt_rows:
        arr = np.array(gt_rows, dtype=np.float32)
        gt_cls_arr = arr[:, 0].astype(np.int32)
        gt_boxes_xyxy = arr[:, 1:]
    return gt_boxes_xyxy, gt_cls_arr


def evaluate_per_slice_from_predictions(
    predictions: list[DetectionPrediction],
    test_yaml: str | Path,
    thresholds: YoloSliceThresholds,
    *,
    output_dir: Optional[Path] = None,
    iou_threshold: float = 0.5,
) -> dict:
    """Run per-slice mAP evaluation from precomputed normalized predictions."""

    del iou_threshold
    image_paths, labels_dir, names = _resolve_eval_split_paths(test_yaml)
    predictions_by_stem = {prediction.image_id: prediction for prediction in predictions}
    all_pred_boxes: list[np.ndarray] = []
    all_pred_confs: list[np.ndarray] = []
    all_pred_cls: list[np.ndarray] = []
    all_gt_boxes: list[np.ndarray] = []
    all_gt_cls: list[np.ndarray] = []

    for img_path in image_paths:
        stem = img_path.stem
        prediction = predictions_by_stem.get(
            stem,
            DetectionPrediction(
                image_id=stem,
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                class_ids=np.zeros(0, dtype=np.int32),
            ),
        )
        gt_boxes_xyxy, gt_cls_arr = _load_gt_arrays_for_image(labels_dir / f"{stem}.txt")
        all_pred_boxes.append(np.asarray(prediction.boxes_xyxy, dtype=np.float32).reshape(-1, 4))
        all_pred_confs.append(np.asarray(prediction.scores, dtype=np.float32).reshape(-1))
        all_pred_cls.append(np.asarray(prediction.class_ids, dtype=np.int32).reshape(-1))
        all_gt_boxes.append(gt_boxes_xyxy)
        all_gt_cls.append(gt_cls_arr)

    iou_thresholds = np.linspace(0.5, 0.95, 10, dtype=np.float64)
    slice_entries: dict[tuple, list] = {}
    gt_per_slice: dict[tuple, int] = {}

    for pred_boxes, pred_confs, pred_cls_arr, gt_boxes, gt_cls_arr in zip(
        all_pred_boxes,
        all_pred_confs,
        all_pred_cls,
        all_gt_boxes,
        all_gt_cls,
    ):
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)
        gt_slice_keys: list[tuple] = []
        for gi in range(n_gt):
            cls_lbl = names.get(int(gt_cls_arr[gi]), str(int(gt_cls_arr[gi])))
            cx_gt = float((gt_boxes[gi, 0] + gt_boxes[gi, 2]) / 2.0)
            cy_gt = float((gt_boxes[gi, 1] + gt_boxes[gi, 3]) / 2.0)
            w_gt = float(gt_boxes[gi, 2] - gt_boxes[gi, 0])
            h_gt = float(gt_boxes[gi, 3] - gt_boxes[gi, 1])
            area_gt = w_gt * h_gt
            if area_gt <= thresholds.q33:
                sz = "small"
            elif area_gt <= thresholds.q67:
                sz = "medium"
            else:
                sz = "large"
            col_i = min(int(cx_gt * 3), 2)
            row_i = min(int(cy_gt * 3), 2)
            pos = f"{('top', 'middle', 'bottom')[row_i]}_{('left', 'center', 'right')[col_i]}"
            key = (cls_lbl, sz, pos)
            gt_slice_keys.append(key)
            gt_per_slice[key] = gt_per_slice.get(key, 0) + 1

        if n_pred == 0:
            continue

        pred_slice_keys = []
        for pi in range(n_pred):
            cls_lbl = names.get(int(pred_cls_arr[pi]), str(int(pred_cls_arr[pi])))
            pred_slice_keys.append(_assign_pred_slices(pred_boxes[pi:pi + 1], thresholds, cls_lbl)[0])

        iou_mat = (
            box_iou_matrix(pred_boxes, gt_boxes)
            if n_gt > 0
            else np.zeros((n_pred, 0), dtype=np.float32)
        )
        tp_mat = np.zeros((n_pred, len(iou_thresholds)), dtype=bool)
        sort_idx = np.argsort(-pred_confs)
        for t_idx, iou_th in enumerate(iou_thresholds):
            gt_matched = np.zeros(n_gt, dtype=bool)
            for pi_sorted in sort_idx:
                if n_gt == 0:
                    break
                row = iou_mat[pi_sorted]
                best_iou = -1.0
                best_gi = -1
                for gi in range(n_gt):
                    if not gt_matched[gi] and row[gi] > best_iou:
                        best_iou = row[gi]
                        best_gi = gi
                if best_iou >= iou_th:
                    tp_mat[pi_sorted, t_idx] = True
                    gt_matched[best_gi] = True

        for pi in range(n_pred):
            tp_vec = tp_mat[pi]
            if tp_vec[0] and n_gt > 0:
                best_gi = int(np.argmax(iou_mat[pi]))
                attr_key = gt_slice_keys[best_gi]
            else:
                attr_key = pred_slice_keys[pi]
            slice_entries.setdefault(attr_key, []).append((tp_vec, float(pred_confs[pi]), int(pred_cls_arr[pi])))

    per_slice_rows = _build_per_slice_rows_from_entries(slice_entries, gt_per_slice, thresholds, names)
    overall_row = next(r for r in per_slice_rows if r["slice_key"] == "overall")
    per_slice_df = pd.DataFrame(per_slice_rows)
    out = {
        "per_slice": per_slice_df.to_dict(orient="records"),
        "overall": overall_row,
        "thresholds": thresholds.to_dict(),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        per_slice_df.to_csv(output_dir / "per_slice_metrics.csv", index=False)
        with open(output_dir / "per_slice_metrics.json", "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        save_frozen_thresholds(thresholds, output_dir)
    return out


def evaluate_per_slice(
    weights: str | Path,
    test_yaml: str | Path,
    thresholds: YoloSliceThresholds,
    *,
    output_dir: Optional[Path] = None,
    imgsz: int = 640,
    conf: float = 0.001,
    iou_threshold: float = 0.5,
    device: str = "cpu",
) -> dict:
    """Run per-slice mAP evaluation on a trained YOLO detector.

    Slice attribution:
    - **TP**: matched GT box's slice.
    - **FN**: unmatched GT box's slice (handled implicitly via GT count per
      slice as the recall denominator).
    - **FP**: predicted box's own size_bin × position_bin slice.

    Args:
        weights: Path to a ``.pt`` checkpoint.
        test_yaml: YOLO dataset YAML pointing at the test images/labels.
        thresholds: Frozen size-bin tertile thresholds from the training split.
        output_dir: If provided, write ``per_slice_metrics.csv``,
            ``per_slice_metrics.json``, and ``slice_thresholds.json`` here.
        imgsz: Inference image size.
        conf: Confidence threshold for predictions (low default to get full
            PR curve).
        iou_threshold: IoU threshold for TP matching (COCO uses 0.5 for AP50).
        device: Torch device string.

    Returns:
        Dict with keys ``per_slice`` (list of per-slice dicts), ``overall``
        (aggregate dict), ``thresholds`` (q33/q67).
    """
    from ultralytics import YOLO
    from src.core.configs.config_loader import load_yaml

    # -- resolve test paths ------------------------------------------------
    test_yaml = Path(test_yaml)
    ds_payload = load_yaml(test_yaml)
    ds_root = Path(str(ds_payload.get("path", test_yaml.parent))).resolve()
    # Use the 'test' split; fall back to 'val' if missing
    split_key = "test" if "test" in ds_payload else "val"
    test_images_rel = str(ds_payload[split_key])
    test_images_dir = Path(test_images_rel)
    if not test_images_dir.is_absolute():
        test_images_dir = (ds_root / test_images_rel).resolve()
    # labels dir: sibling of images dir under 'labels'
    if test_images_dir.parent.name == "images":
        labels_dir = test_images_dir.parent.parent / "labels" / test_images_dir.name
    elif test_images_dir.name == "images":
        labels_dir = test_images_dir.parent / "labels"
    else:
        labels_dir = test_images_dir.parents[1] / "labels" / test_images_dir.name
    labels_dir = labels_dir.resolve()

    raw_names = ds_payload.get("names", {})
    names: dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}

    # -- load model --------------------------------------------------------
    model = YOLO(str(weights))

    # -- run inference on test images --------------------------------------
    image_paths = sorted(p for p in test_images_dir.glob("*") if p.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No test images found under {test_images_dir}")

    # Accumulate per-image data
    # For each image: list of pred (xyxy_norm, conf, class) + list of GT (xyxy_norm, class)
    all_pred_boxes: list[np.ndarray] = []    # (N, 4) xyxy normalised
    all_pred_confs: list[np.ndarray] = []    # (N,)
    all_pred_cls:   list[np.ndarray] = []    # (N,)
    all_gt_boxes:   list[np.ndarray] = []    # (M, 4) xyxy normalised
    all_gt_cls:     list[np.ndarray] = []    # (M,)
    all_gt_stems:   list[list[str]]  = []
    all_gt_insts:   list[list[int]]  = []

    results_list = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        save=False,
    )

    for img_path, result in zip(image_paths, results_list):
        stem = img_path.stem
        ih, iw = result.orig_shape[:2]
        label_path = labels_dir / f"{stem}.txt"

        # --- GT boxes (normalised xywh → xyxy) ---
        gt_boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        gt_cls_arr = np.zeros(0, dtype=np.int32)
        gt_stems_img: list[str] = []
        gt_insts_img: list[int] = []
        if label_path.exists():
            text = label_path.read_text(encoding="utf-8").strip()
            gt_rows = []
            for ann_idx, line in enumerate(text.splitlines()):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_i = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = np.clip(cx - w / 2, 0.0, 1.0)
                y1 = np.clip(cy - h / 2, 0.0, 1.0)
                x2 = np.clip(cx + w / 2, 0.0, 1.0)
                y2 = np.clip(cy + h / 2, 0.0, 1.0)
                gt_rows.append((cls_i, x1, y1, x2, y2))
                gt_stems_img.append(stem)
                gt_insts_img.append(ann_idx)
            if gt_rows:
                arr = np.array(gt_rows, dtype=np.float32)
                gt_cls_arr = arr[:, 0].astype(np.int32)
                gt_boxes_xyxy = arr[:, 1:]

        # --- predictions (normalised xyxy) ---
        pred_boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        pred_confs_arr = np.zeros(0, dtype=np.float32)
        pred_cls_arr = np.zeros(0, dtype=np.int32)
        if result.boxes is not None and len(result.boxes) > 0:
            # boxes.xyxyn: normalised xyxy
            pred_boxes_xyxy = result.boxes.xyxyn.cpu().numpy().astype(np.float32)
            pred_confs_arr = result.boxes.conf.cpu().numpy().astype(np.float32)
            pred_cls_arr = result.boxes.cls.cpu().numpy().astype(np.int32)

        all_pred_boxes.append(pred_boxes_xyxy)
        all_pred_confs.append(pred_confs_arr)
        all_pred_cls.append(pred_cls_arr)
        all_gt_boxes.append(gt_boxes_xyxy)
        all_gt_cls.append(gt_cls_arr)
        all_gt_stems.append(gt_stems_img)
        all_gt_insts.append(gt_insts_img)

    # -- global COCO match: IoU threshold 0.5 (single threshold for TP/FP label) ---
    # Build flat arrays for ap_per_class (needs per-prediction TP flag + conf + pred_cls
    # and target_cls for the denominator).
    # We do a proper COCO multi-threshold match using 10 IoU thresholds 0.5:0.05:0.95.
    iou_thresholds = np.linspace(0.5, 0.95, 10, dtype=np.float64)

    # These lists accumulate entries keyed by the slice the prediction is attributed to.
    # We'll collect (tp_row, conf, pred_cls, gt_slice_key) for all predictions.
    slice_entries: dict[tuple, list] = {}  # slice_key → list of (tp_vec, conf, is_pred=True)
    # GT per slice for denominator
    gt_per_slice: dict[tuple, int] = {}

    for img_idx, (pred_boxes, pred_confs, pred_cls_arr,
                  gt_boxes, gt_cls_arr, gt_stems_img, gt_insts_img) in enumerate(
        zip(all_pred_boxes, all_pred_confs, all_pred_cls,
            all_gt_boxes, all_gt_cls, all_gt_stems, all_gt_insts)
    ):
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        # Assign GT to slices
        # v18 is single-class (person), so class_label is fixed. Support multi-class too.
        gt_slice_keys: list[tuple] = []
        for gi in range(n_gt):
            cls_lbl = names.get(int(gt_cls_arr[gi]), str(int(gt_cls_arr[gi])))
            cx_gt = float((gt_boxes[gi, 0] + gt_boxes[gi, 2]) / 2.0)
            cy_gt = float((gt_boxes[gi, 1] + gt_boxes[gi, 3]) / 2.0)
            w_gt = float(gt_boxes[gi, 2] - gt_boxes[gi, 0])
            h_gt = float(gt_boxes[gi, 3] - gt_boxes[gi, 1])
            area_gt = w_gt * h_gt
            # size bin
            if area_gt <= thresholds.q33:
                sz = "small"
            elif area_gt <= thresholds.q67:
                sz = "medium"
            else:
                sz = "large"
            # position bin
            col_i = min(int(cx_gt * 3), 2)
            row_i = min(int(cy_gt * 3), 2)
            _rl = ("top", "middle", "bottom")
            _cl = ("left", "center", "right")
            pos = f"{_rl[row_i]}_{_cl[col_i]}"
            key = (cls_lbl, sz, pos)
            gt_slice_keys.append(key)
            gt_per_slice[key] = gt_per_slice.get(key, 0) + 1

        if n_pred == 0:
            continue

        # Assign predicted boxes to slices (for FP attribution)
        if len(names) == 1:
            # Single-class: use the one class name for all predictions
            cls_lbl_for_preds = names[0]
            pred_slice_keys = _assign_pred_slices(pred_boxes, thresholds, cls_lbl_for_preds)
        else:
            pred_slice_keys = []
            for pi in range(n_pred):
                cls_lbl = names.get(int(pred_cls_arr[pi]), str(int(pred_cls_arr[pi])))
                pred_slice_keys.append(
                    _assign_pred_slices(pred_boxes[pi:pi+1], thresholds, cls_lbl)[0]
                )

        # Compute IoU matrix (n_pred × n_gt)
        if n_gt > 0:
            iou_mat = _iou_matrix(
                torch.tensor(pred_boxes, dtype=torch.float32),
                torch.tensor(gt_boxes, dtype=torch.float32),
            )  # (n_pred, n_gt)
        else:
            iou_mat = np.zeros((n_pred, 0), dtype=np.float32)

        # TP matrix: (n_pred, n_iou_thresholds)
        tp_mat = np.zeros((n_pred, len(iou_thresholds)), dtype=bool)

        # Greedy matching per IoU threshold (COCO-style: descending conf, each GT once)
        # Sort predictions by confidence descending once.
        sort_idx = np.argsort(-pred_confs)
        for t_idx, iou_th in enumerate(iou_thresholds):
            gt_matched = np.zeros(n_gt, dtype=bool)
            for pi_sorted in sort_idx:
                if n_gt == 0:
                    break
                row = iou_mat[pi_sorted]
                # find best unmatched GT above threshold
                best_iou = -1.0
                best_gi = -1
                for gi in range(n_gt):
                    if not gt_matched[gi] and row[gi] > best_iou:
                        best_iou = row[gi]
                        best_gi = gi
                if best_iou >= iou_th:
                    tp_mat[pi_sorted, t_idx] = True
                    gt_matched[best_gi] = True

        # Attribute each prediction to a slice and record
        for pi in range(n_pred):
            tp_vec = tp_mat[pi]  # shape (10,)
            # Is this prediction a TP at iou=0.5 (first threshold)?
            if tp_vec[0]:
                # TP → inherit GT slice (find matched GT for this iou threshold)
                # We determine matched GT by finding the best unmatched GT above 0.5.
                # Re-derive: the greedy match at threshold 0.5 already stored in tp_mat.
                # To get the matched GT index, we redo the single-threshold match.
                # Quick path: find GT with highest IoU > 0.5 (greedy already done, but
                # we only stored the TP bool, not the matched GT index).
                # We use the IoU row directly.
                if n_gt > 0:
                    best_gi = int(np.argmax(iou_mat[pi]))
                    attr_key = gt_slice_keys[best_gi]
                else:
                    attr_key = pred_slice_keys[pi]
            else:
                # FP → predicted box geometry
                attr_key = pred_slice_keys[pi]

            if attr_key not in slice_entries:
                slice_entries[attr_key] = []
            slice_entries[attr_key].append((tp_vec, float(pred_confs[pi]), int(pred_cls_arr[pi])))

    # -- build per-slice AP table ------------------------------------------
    per_slice_rows = _build_per_slice_rows_from_entries(
        slice_entries, gt_per_slice, thresholds, names
    )
    overall_row = next(r for r in per_slice_rows if r["slice_key"] == "overall")
    per_slice_df = pd.DataFrame(per_slice_rows)

    out = {
        "per_slice": per_slice_df.to_dict(orient="records"),
        "overall": overall_row,
        "thresholds": thresholds.to_dict(),
    }

    # -- persist artifacts -------------------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        per_slice_df.to_csv(output_dir / "per_slice_metrics.csv", index=False)
        with open(output_dir / "per_slice_metrics.json", "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        save_frozen_thresholds(thresholds, output_dir)

    return out
