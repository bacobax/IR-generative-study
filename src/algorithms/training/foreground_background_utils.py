"""Utilities for FLIR foreground/background classifier training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch


def size_bucket_name(area_ratio: float) -> str:
    """Map normalized source area to the reporting bucket used in logs."""
    area_ratio = float(area_ratio)
    if area_ratio < 0.002:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    return "medium_large"


def compute_binary_metrics(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    positive_area_ratios: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute binary classification metrics at a fixed threshold."""
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= float(threshold)).astype(np.float32)

    tp = float(np.sum((pred == 1.0) & (labels == 1.0)))
    tn = float(np.sum((pred == 0.0) & (labels == 0.0)))
    fp = float(np.sum((pred == 1.0) & (labels == 0.0)))
    fn = float(np.sum((pred == 0.0) & (labels == 1.0)))

    accuracy = (tp + tn) / max(1.0, tp + tn + fp + fn)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    specificity = tn / max(1.0, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }

    if positive_area_ratios is None:
        return metrics

    area_ratios = np.asarray(positive_area_ratios, dtype=np.float32).reshape(-1)
    pos_mask = labels == 1.0
    pos_pred = pred[pos_mask]
    pos_areas = area_ratios[pos_mask]
    for bucket in ("tiny", "small", "medium_large"):
        bucket_mask = np.array([size_bucket_name(v) == bucket for v in pos_areas], dtype=bool)
        if not bucket_mask.any():
            metrics[f"recall_{bucket}"] = 0.0
            continue
        bucket_recall = float(np.mean(pos_pred[bucket_mask] == 1.0))
        metrics[f"recall_{bucket}"] = bucket_recall
    return metrics


def _candidate_thresholds(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if probs.size == 0:
        return np.asarray([0.5], dtype=np.float32)
    unique = np.unique(probs)
    if unique.size <= 201:
        thresholds = unique
    else:
        quantiles = np.linspace(0.0, 1.0, 201, dtype=np.float32)
        thresholds = np.quantile(unique, quantiles)
    thresholds = np.clip(thresholds, 1e-4, 1.0 - 1e-4)
    thresholds = np.unique(np.concatenate([thresholds, np.asarray([0.5], dtype=np.float32)]))
    return thresholds.astype(np.float32)


def select_best_threshold(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    positive_area_ratios: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Pick the operating threshold that maximizes validation F1."""
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))

    best_metrics: Optional[Dict[str, float]] = None
    best_threshold = 0.5
    for threshold in _candidate_thresholds(probs):
        metrics = compute_binary_metrics(
            logits=logits,
            labels=labels,
            threshold=float(threshold),
            positive_area_ratios=positive_area_ratios,
        )
        if best_metrics is None:
            best_metrics = metrics
            best_threshold = float(threshold)
            continue
        if metrics["f1"] > best_metrics["f1"] + 1e-12:
            best_metrics = metrics
            best_threshold = float(threshold)
            continue
        if abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12 and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5):
            best_metrics = metrics
            best_threshold = float(threshold)

    return {
        "threshold": float(best_threshold),
        "metrics": dict(best_metrics or compute_binary_metrics(logits=logits, labels=labels, threshold=0.5)),
    }


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
    config: Dict[str, Any],
    best_val_metric: float,
    best_threshold: float,
    best_val_metrics: Optional[Dict[str, Any]] = None,
    best_test_metrics: Optional[Dict[str, Any]] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a full training checkpoint with threshold metadata."""
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": dict(config),
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "best_val_metric": float(best_val_metric),
        "best_threshold": float(best_threshold),
        "best_val_metrics": {} if best_val_metrics is None else dict(best_val_metrics),
        "best_test_metrics": {} if best_test_metrics is None else dict(best_test_metrics),
    }
    if extra_payload:
        payload.update(dict(extra_payload))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load a full training checkpoint into the provided modules."""
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Append JSON-serializable rows to a jsonl file."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def compute_multiclass_metrics(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    background_class_index: int,
    positive_area_ratios: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (N, C), got {logits.shape}")
    if logits.shape[0] == 0:
        return {
            "accuracy": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "foreground_exact_match_rate": 0.0,
            "background_recall": 0.0,
        }

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)
    accuracy = float(np.mean(pred == labels))

    class_ids = list(range(logits.shape[1]))
    recalls = []
    f1s = []
    for class_idx in class_ids:
        gt = labels == class_idx
        tp = float(np.sum((pred == class_idx) & gt))
        fp = float(np.sum((pred == class_idx) & (~gt)))
        fn = float(np.sum((pred != class_idx) & gt))
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)
        recalls.append(recall)
        f1s.append(f1)

    foreground_mask = labels != int(background_class_index)
    foreground_exact_match_rate = float(np.mean(pred[foreground_mask] == labels[foreground_mask])) if foreground_mask.any() else 0.0
    background_mask = labels == int(background_class_index)
    background_recall = float(np.mean(pred[background_mask] == int(background_class_index))) if background_mask.any() else 0.0
    metrics = {
        "accuracy": float(accuracy),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "foreground_exact_match_rate": float(foreground_exact_match_rate),
        "background_recall": float(background_recall),
    }

    if positive_area_ratios is not None:
        area_ratios = np.asarray(positive_area_ratios, dtype=np.float32).reshape(-1)
        pos_pred = pred[foreground_mask]
        pos_labels = labels[foreground_mask]
        pos_areas = area_ratios[foreground_mask]
        for bucket in ("tiny", "small", "medium_large"):
            bucket_mask = np.array([size_bucket_name(v) == bucket for v in pos_areas], dtype=bool)
            if not bucket_mask.any():
                metrics[f"foreground_recall_{bucket}"] = 0.0
                continue
            metrics[f"foreground_recall_{bucket}"] = float(np.mean(pos_pred[bucket_mask] == pos_labels[bucket_mask]))
    return metrics


def select_best_thresholds_per_class(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    foreground_class_indices: Sequence[int],
) -> Dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    thresholds: Dict[str, float] = {}
    metrics_by_class: Dict[str, Dict[str, float]] = {}
    for class_idx in foreground_class_indices:
        class_probs = probs[:, int(class_idx)]
        class_labels = (labels == int(class_idx)).astype(np.float32)
        best_threshold = 0.5
        best_metrics = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0}
        for threshold in _candidate_thresholds(class_probs):
            pred = (class_probs >= float(threshold)).astype(np.float32)
            tp = float(np.sum((pred == 1.0) & (class_labels == 1.0)))
            fp = float(np.sum((pred == 1.0) & (class_labels == 0.0)))
            fn = float(np.sum((pred == 0.0) & (class_labels == 1.0)))
            precision = tp / max(1.0, tp + fp)
            recall = tp / max(1.0, tp + fn)
            f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)
            if f1 > best_metrics["f1"] + 1e-12 or (
                abs(f1 - best_metrics["f1"]) <= 1e-12 and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
            ):
                best_threshold = float(threshold)
                best_metrics = {
                    "f1": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "tp": float(tp),
                    "fp": float(fp),
                    "fn": float(fn),
                }
        thresholds[str(int(class_idx))] = float(best_threshold)
        metrics_by_class[str(int(class_idx))] = best_metrics

    return {
        "thresholds": thresholds,
        "metrics_by_class": metrics_by_class,
    }
