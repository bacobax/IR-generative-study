"""Native object-detection metrics and prediction containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DetectionPrediction:
    """One image's normalized xyxy predictions."""

    image_id: str
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray


@dataclass(frozen=True)
class DetectionGroundTruth:
    """One image's normalized xyxy ground-truth boxes."""

    image_id: str
    boxes_xyxy: np.ndarray
    class_ids: np.ndarray


def xywh_to_xyxy_np(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    converted = np.zeros_like(boxes, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return np.clip(converted, 0.0, 1.0)


def xyxy_to_xywh_np(boxes_xyxy: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    converted = np.zeros_like(boxes, dtype=np.float32)
    converted[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    converted[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    converted[:, 2] = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    converted[:, 3] = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    return converted


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return an ``(N, M)`` IoU matrix for normalized xyxy boxes."""

    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, *, iou_threshold: float) -> np.ndarray:
    """Class-local greedy non-maximum suppression."""

    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)

    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        ious = box_iou_matrix(boxes[current:current + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= float(iou_threshold)]
    return np.asarray(keep, dtype=np.int64)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute interpolated AP from recall/precision curves."""

    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    if recall.size == 0 or precision.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def summarize_match_arrays(
    *,
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    class_ids: Iterable[int],
) -> list[dict[str, float | int]]:
    """Summarize AP/P/R from already matched prediction arrays."""

    tp = np.asarray(tp, dtype=bool)
    if tp.ndim == 1:
        tp = tp[:, None]
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    pred_cls = np.asarray(pred_cls, dtype=np.int32).reshape(-1)
    target_cls = np.asarray(target_cls, dtype=np.int32).reshape(-1)
    order = conf.argsort()[::-1] if len(conf) else np.zeros(0, dtype=np.int64)

    rows: list[dict[str, float | int]] = []
    for cls_id in class_ids:
        cls_id = int(cls_id)
        cls_pred_mask = pred_cls == cls_id
        cls_target_count = int((target_cls == cls_id).sum())
        cls_order = [int(idx) for idx in order if bool(cls_pred_mask[idx])]
        if cls_target_count == 0:
            rows.append(
                {
                    "class_id": cls_id,
                    "n_gt": 0,
                    "n_pred": int(cls_pred_mask.sum()),
                    "AP50": float("nan"),
                    "AP50_95": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                }
            )
            continue
        if not cls_order:
            rows.append(
                {
                    "class_id": cls_id,
                    "n_gt": cls_target_count,
                    "n_pred": 0,
                    "AP50": 0.0,
                    "AP50_95": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                }
            )
            continue

        cls_tp = tp[np.asarray(cls_order, dtype=np.int64)]
        aps = []
        for threshold_idx in range(cls_tp.shape[1]):
            tp_cum = cls_tp[:, threshold_idx].astype(np.float64).cumsum()
            fp_cum = (~cls_tp[:, threshold_idx]).astype(np.float64).cumsum()
            recall_curve = tp_cum / max(cls_target_count, 1)
            precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            aps.append(average_precision(recall_curve, precision_curve))

        tp50 = cls_tp[:, 0].astype(np.float64)
        fp50 = (~cls_tp[:, 0]).astype(np.float64)
        n_tp50 = float(tp50.sum())
        n_fp50 = float(fp50.sum())
        rows.append(
            {
                "class_id": cls_id,
                "n_gt": cls_target_count,
                "n_pred": int(cls_pred_mask.sum()),
                "AP50": float(aps[0]),
                "AP50_95": float(np.mean(aps)),
                "precision": float(n_tp50 / max(n_tp50 + n_fp50, 1e-12)),
                "recall": float(n_tp50 / max(cls_target_count, 1)),
            }
        )
    return rows


def _match_class_predictions(
    *,
    predictions: list[DetectionPrediction],
    ground_truths: dict[str, DetectionGroundTruth],
    class_id: int,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    class_preds: list[tuple[str, float, np.ndarray]] = []
    gt_by_image: dict[str, np.ndarray] = {}
    n_gt = 0
    for gt in ground_truths.values():
        mask = gt.class_ids.astype(int) == int(class_id)
        boxes = gt.boxes_xyxy[mask]
        gt_by_image[gt.image_id] = boxes
        n_gt += int(len(boxes))
    for pred in predictions:
        mask = pred.class_ids.astype(int) == int(class_id)
        for box, score in zip(pred.boxes_xyxy[mask], pred.scores[mask]):
            class_preds.append((pred.image_id, float(score), np.asarray(box, dtype=np.float32)))

    if not class_preds:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32), n_gt

    class_preds.sort(key=lambda item: item[1], reverse=True)
    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_by_image.items()}
    tp = np.zeros(len(class_preds), dtype=bool)
    conf = np.asarray([item[1] for item in class_preds], dtype=np.float32)
    for pred_idx, (image_id, _, box) in enumerate(class_preds):
        gt_boxes = gt_by_image.get(image_id, np.zeros((0, 4), dtype=np.float32))
        if len(gt_boxes) == 0:
            continue
        ious = box_iou_matrix(box.reshape(1, 4), gt_boxes)[0]
        best_idx = int(np.argmax(ious))
        if float(ious[best_idx]) >= float(iou_threshold) and not matched[image_id][best_idx]:
            tp[pred_idx] = True
            matched[image_id][best_idx] = True
    return tp, conf, n_gt


def evaluate_detections(
    *,
    predictions: list[DetectionPrediction],
    ground_truths: list[DetectionGroundTruth],
    names: dict[int, str],
    iou_thresholds: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute deterministic AP50/AP50-95 and P/R summaries."""

    thresholds = (
        np.linspace(0.5, 0.95, 10, dtype=np.float64)
        if iou_thresholds is None
        else np.asarray(iou_thresholds, dtype=np.float64)
    )
    gt_by_image = {gt.image_id: gt for gt in ground_truths}
    class_rows: list[dict[str, object]] = []
    for class_id in sorted(names):
        ap_values: list[float] = []
        precision50 = 0.0
        recall50 = 0.0
        n_pred = int(sum((pred.class_ids.astype(int) == int(class_id)).sum() for pred in predictions))
        n_gt_final = 0
        for threshold_idx, iou_threshold in enumerate(thresholds):
            tp, conf, n_gt = _match_class_predictions(
                predictions=predictions,
                ground_truths=gt_by_image,
                class_id=int(class_id),
                iou_threshold=float(iou_threshold),
            )
            n_gt_final = int(n_gt)
            if n_gt == 0:
                ap_values.append(float("nan"))
                continue
            if len(tp) == 0:
                ap_values.append(0.0)
                if threshold_idx == 0:
                    precision50 = 0.0
                    recall50 = 0.0
                continue
            tp_cum = tp.astype(np.float64).cumsum()
            fp_cum = (~tp).astype(np.float64).cumsum()
            recall_curve = tp_cum / max(n_gt, 1)
            precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap_values.append(average_precision(recall_curve, precision_curve))
            if threshold_idx == 0:
                n_tp = float(tp.sum())
                n_fp = float((~tp).sum())
                precision50 = float(n_tp / max(n_tp + n_fp, 1e-12))
                recall50 = float(n_tp / max(n_gt, 1))
        finite_aps = [value for value in ap_values if np.isfinite(value)]
        class_rows.append(
            {
                "class_idx": int(class_id),
                "class_label": names[int(class_id)],
                "n_gt": n_gt_final,
                "n_pred": n_pred,
                "AP50": float(ap_values[0]) if ap_values else float("nan"),
                "AP75": float(ap_values[int(np.argmin(np.abs(thresholds - 0.75)))]) if ap_values else float("nan"),
                "AP50_95": float(np.mean(finite_aps)) if finite_aps else float("nan"),
                "precision": precision50 if n_gt_final > 0 else float("nan"),
                "recall": recall50 if n_gt_final > 0 else float("nan"),
            }
        )

    valid_rows = [row for row in class_rows if int(row["n_gt"]) > 0]
    def _mean_metric(key: str) -> float:
        values = [float(row[key]) for row in valid_rows if np.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else float("nan")

    return {
        "per_class": class_rows,
        "summary": {
            "map": _mean_metric("AP50_95"),
            "map50": _mean_metric("AP50"),
            "map75": _mean_metric("AP75"),
            "precision": _mean_metric("precision"),
            "recall": _mean_metric("recall"),
            "n_images": len(ground_truths),
            "n_predictions": int(sum(len(pred.scores) for pred in predictions)),
            "n_ground_truth": int(sum(len(gt.class_ids) for gt in ground_truths)),
        },
    }
