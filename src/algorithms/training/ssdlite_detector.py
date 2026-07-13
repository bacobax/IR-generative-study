"""Training, evaluation, and inference for the SSDLite detector backend."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.algorithms.training.simple_yolo_detector import (
    SimpleYoloDataset,
    YoloSplitInfo,
    _make_dataloader,
    _make_optimizer,
    _write_json,
    render_detection_overlay,
    resolve_yolo_split_info,
)
from src.algorithms.training.yolo_slice_baselines import prepare_yolo_slice_baseline
from src.core.training_utils import autocast_context, build_summary_writer
from src.core.training_runtime import setup_precision
from src.evaluation.detection_metrics import (
    DetectionGroundTruth,
    DetectionPrediction,
    evaluate_detections,
    nms_numpy,
    xywh_to_xyxy_np,
)
from src.models.simple_yolo import count_trainable_parameters
from src.models.ssdlite import SSDLiteConfig, SSDLiteDetector, build_ssdlite_model

_VARIANCES = (0.1, 0.1, 0.2, 0.2)


# ---------------------------------------------------------------------------
# Box encoding / decoding
# ---------------------------------------------------------------------------

def encode_ssdlite_offsets(
    gt_boxes_xywh: torch.Tensor,  # [..., 4]
    anchors: torch.Tensor,         # [..., 4]
) -> torch.Tensor:
    """Encode GT boxes as SSD deltas relative to anchors."""
    gcx, gcy, gw, gh = gt_boxes_xywh.unbind(-1)
    acx, acy, aw, ah = anchors.unbind(-1)
    aw = aw.clamp_min(1e-8)
    ah = ah.clamp_min(1e-8)
    gw = gw.clamp_min(1e-8)
    gh = gh.clamp_min(1e-8)
    dx = (gcx - acx) / (aw * _VARIANCES[0])
    dy = (gcy - acy) / (ah * _VARIANCES[1])
    dw = torch.log(gw / aw) / _VARIANCES[2]
    dh = torch.log(gh / ah) / _VARIANCES[3]
    return torch.stack([dx, dy, dw, dh], dim=-1)


def decode_ssdlite_offsets(
    bbox_pred: torch.Tensor,  # [B, N, 4] or [N, 4]
    anchors: torch.Tensor,    # [N, 4]
) -> torch.Tensor:
    """Decode SSD deltas back to cx/cy/w/h boxes."""
    if bbox_pred.dim() == 2:
        # Single-image path
        dx, dy, dw, dh = bbox_pred.unbind(-1)
        acx, acy, aw, ah = anchors.unbind(-1)
    else:
        dx, dy, dw, dh = bbox_pred.unbind(-1)
        acx, acy, aw, ah = anchors.unbind(-1)

    cx = dx * aw * _VARIANCES[0] + acx
    cy = dy * ah * _VARIANCES[1] + acy
    w = torch.exp(dw * _VARIANCES[2]) * aw
    h = torch.exp(dh * _VARIANCES[3]) * ah
    return torch.stack([cx, cy, w, h], dim=-1).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Pairwise IoU
# ---------------------------------------------------------------------------

def _box_iou_matrix_xywh(
    boxes_a: torch.Tensor,  # [N, 4] cx/cy/w/h
    boxes_b: torch.Tensor,  # [M, 4] cx/cy/w/h
) -> torch.Tensor:          # [N, M]
    """Pairwise IoU between two sets of cx/cy/w/h boxes."""
    def to_xyxy(b: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            b[:, 0] - b[:, 2] / 2,
            b[:, 1] - b[:, 3] / 2,
            b[:, 0] + b[:, 2] / 2,
            b[:, 1] + b[:, 3] / 2,
        ], dim=1)

    a = to_xyxy(boxes_a)  # [N, 4]
    b = to_xyxy(boxes_b)  # [M, 4]

    inter_x1 = torch.maximum(a[:, 0].unsqueeze(1), b[:, 0].unsqueeze(0))
    inter_y1 = torch.maximum(a[:, 1].unsqueeze(1), b[:, 1].unsqueeze(0))
    inter_x2 = torch.minimum(a[:, 2].unsqueeze(1), b[:, 2].unsqueeze(0))
    inter_y2 = torch.minimum(a[:, 3].unsqueeze(1), b[:, 3].unsqueeze(0))
    inter = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).clamp_min(0).unsqueeze(1)
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).clamp_min(0).unsqueeze(0)
    union = (area_a + area_b - inter).clamp_min(1e-10)
    return inter / union


# ---------------------------------------------------------------------------
# Anchor-to-GT matching
# ---------------------------------------------------------------------------

def assign_ssdlite_targets(
    anchors: torch.Tensor,        # [N, 4]
    gt_boxes_xywh: torch.Tensor,  # [M, 4]
    gt_classes: torch.Tensor,     # [M]
    *,
    iou_pos_threshold: float = 0.5,
    iou_neg_threshold: float = 0.4,
    nc: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match anchors to GT boxes using SSD bidirectional matching.

    Returns:
        cls_targets [N]:  class index for positives, -1 for ignored, -2 for negatives
        loc_targets [N, 4]: encoded box deltas (meaningful only for positives)
        pos_mask [N]:     bool mask of positive anchors
    """
    N = anchors.shape[0]
    device = anchors.device
    M = int(gt_boxes_xywh.shape[0]) if gt_boxes_xywh.numel() > 0 else 0

    cls_targets = torch.full((N,), -2, dtype=torch.long, device=device)
    loc_targets = torch.zeros(N, 4, dtype=torch.float32, device=device)
    pos_mask = torch.zeros(N, dtype=torch.bool, device=device)

    if M == 0:
        return cls_targets, loc_targets, pos_mask

    iou_mat = _box_iou_matrix_xywh(anchors, gt_boxes_xywh)  # [N, M]

    best_gt_iou, best_gt_idx = iou_mat.max(dim=1)            # [N]
    best_anchor_per_gt = iou_mat.argmax(dim=0)                # [M]

    # Threshold-based positives
    pos_mask = best_gt_iou >= iou_pos_threshold

    # Force one positive per GT box (SSD bidirectional matching)
    pos_mask[best_anchor_per_gt] = True
    best_gt_idx[best_anchor_per_gt] = torch.arange(M, device=device)

    # Fill localization targets for positives
    pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
    if pos_indices.numel() > 0:
        matched_gt = best_gt_idx[pos_indices]
        cls_targets[pos_indices] = gt_classes[matched_gt]
        loc_targets[pos_indices] = encode_ssdlite_offsets(
            gt_boxes_xywh[matched_gt], anchors[pos_indices]
        )

    # Ignore uncertain region (between thresholds)
    ignored = (best_gt_iou >= iou_neg_threshold) & (~pos_mask)
    cls_targets[ignored] = -1

    return cls_targets, loc_targets, pos_mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SSDLiteLoss(nn.Module):
    """SSD-style loss: Smooth L1 (loc) + BCE with hard negative mining (conf)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.box_weight = float(cfg.box_weight)
        self.class_weight = float(cfg.class_weight)
        self.neg_pos_ratio = float(getattr(cfg, "neg_pos_ratio", 3.0))

    def forward(
        self,
        cls_logits: torch.Tensor,      # [B, N, nc]
        bbox_pred: torch.Tensor,        # [B, N, 4]
        anchors: torch.Tensor,          # [N, 4]
        *,
        boxes_xywh: list[torch.Tensor],
        class_ids: list[torch.Tensor],
        iou_pos_threshold: float = 0.5,
        iou_neg_threshold: float = 0.4,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B = cls_logits.shape[0]
        nc = cls_logits.shape[2]
        device = cls_logits.device

        total_loc = cls_logits.new_zeros(())
        total_conf = cls_logits.new_zeros(())
        total_pos = 0
        total_assigned = 0

        for i in range(B):
            gt_boxes = boxes_xywh[i].to(device=device, dtype=torch.float32)
            gt_cls = class_ids[i].to(device=device, dtype=torch.long)
            M = int(gt_boxes.shape[0])
            total_assigned += M

            cls_tgt, loc_tgt, pos_mask = assign_ssdlite_targets(
                anchors, gt_boxes, gt_cls,
                iou_pos_threshold=iou_pos_threshold,
                iou_neg_threshold=iou_neg_threshold,
                nc=nc,
            )

            n_pos = int(pos_mask.sum().item())
            total_pos += n_pos

            # Localization loss (positives only, Smooth L1)
            if n_pos > 0:
                loc_loss = F.smooth_l1_loss(
                    bbox_pred[i][pos_mask], loc_tgt[pos_mask], reduction="sum"
                ) / n_pos
                total_loc = total_loc + loc_loss

            # Confidence loss: positives + hard-mined negatives
            neg_mask = cls_tgt == -2  # true negatives (not ignored)

            # Positive confidence: BCE with one-hot class targets
            if n_pos > 0:
                pos_logits = cls_logits[i][pos_mask]           # [n_pos, nc]
                pos_target = torch.zeros_like(pos_logits)
                pos_cls_idx = cls_tgt[pos_mask].clamp(0, nc - 1)
                pos_target.scatter_(1, pos_cls_idx.unsqueeze(1), 1.0)
                conf_pos = F.binary_cross_entropy_with_logits(
                    pos_logits, pos_target, reduction="sum"
                )
            else:
                conf_pos = cls_logits.new_zeros(())

            # Hard negative mining: top-k negatives by loss
            n_neg_avail = int(neg_mask.sum().item())
            # Mine at least a few negatives even when n_pos == 0
            n_neg_target = max(n_pos * int(self.neg_pos_ratio), min(30, n_neg_avail))
            if n_neg_avail > 0 and n_neg_target > 0:
                neg_logits = cls_logits[i][neg_mask]           # [n_neg_avail, nc]
                # Loss of predicting background (all zeros target)
                neg_loss_each = F.binary_cross_entropy_with_logits(
                    neg_logits,
                    torch.zeros_like(neg_logits),
                    reduction="none",
                ).sum(dim=1)
                top_k = min(n_neg_target, n_neg_avail)
                _, hard_idx = neg_loss_each.topk(top_k)
                conf_neg = neg_loss_each[hard_idx].sum()
            else:
                conf_neg = cls_logits.new_zeros(())

            normalizer = max(n_pos, 1)
            total_conf = total_conf + (conf_pos + conf_neg) / normalizer

        total_loc = total_loc / B
        total_conf = total_conf / B
        total_loss = self.box_weight * total_loc + self.class_weight * total_conf

        return total_loss, {
            "loss": float(total_loss.item()),
            "loc_loss": float(total_loc.item()),
            "conf_loss": float(total_conf.item()),
            "n_pos": float(total_pos),
            "assigned_targets": float(total_assigned),
            "dropped_targets": 0.0,
        }


# ---------------------------------------------------------------------------
# Inference / prediction collection
# ---------------------------------------------------------------------------

def _ssdlite_log_epoch_progress(row: dict[str, Any], *, total_epochs: int) -> None:
    epoch = int(row["epoch"])
    loss_keys = ["loss", "loc_loss", "conf_loss", "n_pos"]
    val_keys = ["val_map50", "val_map", "val_precision", "val_recall"]
    parts = [f"[epoch {epoch:>3d}/{total_epochs}]"]
    parts += [f"{k}={row[k]:.4f}" for k in loss_keys if k in row]
    val_parts = [f"{k}={row[k]:.4f}" for k in val_keys if k in row]
    if val_parts:
        parts += ["|"] + val_parts
    print("  ".join(parts), flush=True)


def _ground_truth_from_batch(batch: dict[str, Any]) -> list[DetectionGroundTruth]:
    ground_truths: list[DetectionGroundTruth] = []
    for image_id, boxes_xywh, class_ids in zip(
        batch["image_ids"], batch["boxes_xywh"], batch["class_ids"]
    ):
        boxes_np = boxes_xywh.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
        class_np = class_ids.detach().cpu().numpy().astype(np.int32).reshape(-1)
        ground_truths.append(
            DetectionGroundTruth(
                image_id=str(image_id),
                boxes_xyxy=xywh_to_xyxy_np(boxes_np),
                class_ids=class_np,
            )
        )
    return ground_truths


def _decode_ssdlite_predictions_batch(
    cls_logits: torch.Tensor,  # [B, N, nc]
    bbox_pred: torch.Tensor,   # [B, N, 4]
    anchors: torch.Tensor,     # [N, 4]
    *,
    names: dict[int, str],
    conf_threshold: float,
    nms_iou: float,
) -> list[DetectionPrediction]:
    decoded = decode_ssdlite_offsets(bbox_pred, anchors).detach().cpu()
    scores_all = torch.sigmoid(cls_logits).detach().cpu()
    anchors_cpu = anchors.detach().cpu()

    B = cls_logits.shape[0]
    predictions: list[DetectionPrediction] = []
    for b in range(B):
        boxes_list: list[np.ndarray] = []
        scores_list: list[float] = []
        class_list: list[int] = []

        for class_id in sorted(names):
            cls_score = scores_all[b, :, int(class_id)]   # [N]
            keep = cls_score >= float(conf_threshold)
            if not keep.any():
                continue
            boxes_xywh = decoded[b][keep].numpy()
            boxes_xyxy = xywh_to_xyxy_np(boxes_xywh)
            scores = cls_score[keep].numpy().astype(np.float32)
            keep_idx = nms_numpy(boxes_xyxy, scores, iou_threshold=float(nms_iou))
            boxes_list.extend([boxes_xyxy[int(idx)] for idx in keep_idx])
            scores_list.extend([float(scores[int(idx)]) for idx in keep_idx])
            class_list.extend([int(class_id)] * len(keep_idx))

        if boxes_list:
            boxes_arr = np.asarray(boxes_list, dtype=np.float32).reshape(-1, 4)
            scores_arr = np.asarray(scores_list, dtype=np.float32)
            class_arr = np.asarray(class_list, dtype=np.int32)
            order = scores_arr.argsort()[::-1]
            boxes_arr, scores_arr, class_arr = (
                boxes_arr[order], scores_arr[order], class_arr[order]
            )
        else:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            scores_arr = np.zeros(0, dtype=np.float32)
            class_arr = np.zeros(0, dtype=np.int32)

        predictions.append(DetectionPrediction("", boxes_arr, scores_arr, class_arr))
    return predictions


@torch.inference_mode()
def collect_ssdlite_predictions(
    model: SSDLiteDetector,
    dataloader: DataLoader,
    *,
    device: torch.device,
    names: dict[int, str],
    conf_threshold: float,
    nms_iou: float,
) -> tuple[list[DetectionPrediction], list[DetectionGroundTruth]]:
    model.eval()
    predictions: list[DetectionPrediction] = []
    ground_truths: list[DetectionGroundTruth] = []
    for batch in dataloader:
        images = batch["images"].to(device=device, non_blocking=True)
        cls_logits, bbox_pred, anchors = model(images)
        batch_preds = _decode_ssdlite_predictions_batch(
            cls_logits, bbox_pred, anchors,
            names=names,
            conf_threshold=float(conf_threshold),
            nms_iou=float(nms_iou),
        )
        for pred, image_id in zip(batch_preds, batch["image_ids"]):
            predictions.append(
                DetectionPrediction(
                    image_id=str(image_id),
                    boxes_xyxy=pred.boxes_xyxy,
                    scores=pred.scores,
                    class_ids=pred.class_ids,
                )
            )
        ground_truths.extend(_ground_truth_from_batch(batch))
    return predictions, ground_truths


def evaluate_ssdlite_model(
    model: SSDLiteDetector,
    dataloader: DataLoader,
    *,
    device: torch.device,
    names: dict[int, str],
    conf_threshold: float,
    nms_iou: float,
) -> dict[str, Any]:
    predictions, ground_truths = collect_ssdlite_predictions(
        model, dataloader,
        device=device, names=names,
        conf_threshold=float(conf_threshold), nms_iou=float(nms_iou),
    )
    metrics = evaluate_detections(
        predictions=predictions, ground_truths=ground_truths, names=names
    )
    return {
        "summary": metrics["summary"],
        "per_class": metrics["per_class"],
        "predictions": predictions,
        "ground_truths": ground_truths,
    }


# ---------------------------------------------------------------------------
# TensorBoard overlay rendering
# ---------------------------------------------------------------------------

@torch.inference_mode()
def render_ssdlite_detection_batch(
    model: SSDLiteDetector,
    dataset: SimpleYoloDataset,
    *,
    device: torch.device,
    names: dict[int, str],
    max_images: int,
    conf_threshold: float,
    nms_iou: float,
) -> "torch.Tensor | None":
    n_images = min(int(max_images), len(dataset))
    if n_images <= 0:
        return None
    model.eval()
    rendered: list[torch.Tensor] = []
    for index in range(n_images):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device=device)
        cls_logits, bbox_pred, anchors = model(image)
        preds = _decode_ssdlite_predictions_batch(
            cls_logits, bbox_pred, anchors,
            names=names,
            conf_threshold=float(conf_threshold),
            nms_iou=float(nms_iou),
        )[0]
        preds = DetectionPrediction(
            image_id=str(sample["image_id"]),
            boxes_xyxy=preds.boxes_xyxy,
            scores=preds.scores,
            class_ids=preds.class_ids,
        )
        rendered.append(
            render_detection_overlay(
                sample["image"],
                gt_boxes_xywh=sample["boxes_xywh"],
                gt_class_ids=sample["class_ids"],
                prediction=preds,
                names=names,
            )
        )
    return torch.stack(rendered, dim=0)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def _save_ssdlite_checkpoint(
    path: Path,
    *,
    model: SSDLiteDetector,
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    split_info: YoloSplitInfo,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "ssdlite_detector_v1",
            "epoch": int(epoch),
            "model_config": model.config.to_dict(),
            "nc": int(split_info.nc),
            "names": split_info.names,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "training_config": {
                "data": vars(cfg.data),
                "model": asdict(cfg.model),
                "training": vars(cfg.training),
                "loss": vars(cfg.loss),
                "baseline": vars(cfg.baseline),
            },
        },
        path,
    )


def load_ssdlite_checkpoint(
    weights_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SSDLiteDetector, dict[str, Any]]:
    payload = torch.load(weights_path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format") != "ssdlite_detector_v1":
        raise ValueError(f"Not an SSDLite checkpoint: {weights_path}")
    model = SSDLiteDetector(
        SSDLiteConfig.from_mapping(payload["model_config"], nc=int(payload["nc"]))
    )
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


# ---------------------------------------------------------------------------
# Helpers shared with the simple_torch trainer
# ---------------------------------------------------------------------------

def _prediction_rows(
    predictions: list[DetectionPrediction], names: dict[int, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        for box, score, class_id in zip(pred.boxes_xyxy, pred.scores, pred.class_ids):
            rows.append(
                {
                    "image_id": pred.image_id,
                    "class_idx": int(class_id),
                    "class_label": names.get(int(class_id), str(int(class_id))),
                    "confidence": float(score),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def _validate_ssdlite_config(cfg: Any, *, split_info: YoloSplitInfo) -> None:
    if int(cfg.training.epochs) <= 0:
        raise ValueError("training.epochs must be positive.")
    if int(cfg.training.val_interval) <= 0:
        raise ValueError("training.val_interval must be positive.")
    if split_info.nc <= 0:
        raise ValueError("SSDLite requires a dataset with at least one class.")


def train_ssdlite(
    cfg: Any,
    *,
    dataset_yaml: str,
    run_dir: Path,
    checkpoint_dir: Path,
    analysis_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train the SSDLite detector."""

    train_info = resolve_yolo_split_info(dataset_yaml, split="train")
    val_info = resolve_yolo_split_info(dataset_yaml, split="val")
    _validate_ssdlite_config(cfg, split_info=train_info)
    torch_device = torch.device(device)

    aug_cfg = getattr(cfg, "augment", None)
    general_aug_enabled = aug_cfg is not None and bool(aug_cfg.enabled)
    rare_slice_aug = str(cfg.baseline.mode) == "baseline_b"
    if general_aug_enabled and rare_slice_aug:
        raise ValueError(
            "augment.enabled and baseline.mode=baseline_b are mutually exclusive."
        )
    cache_images = bool(getattr(cfg.data, "cache_images", False))

    prepared_baseline = None
    baseline_artifact_paths: dict[str, Optional[str]] = {
        "slice_counts_csv": None,
        "slice_summary_json": None,
        "image_sampling_weights_csv": None,
        "sampling_weight_summary_json": None,
    }
    if str(cfg.baseline.mode) in {"baseline_a", "baseline_b"}:
        prepared_baseline = prepare_yolo_slice_baseline(
            dataset_yaml=dataset_yaml,
            analysis_dir=analysis_dir,
            baseline_cfg=cfg.baseline,
        )
        baseline_artifact_paths = {
            "slice_counts_csv": str((analysis_dir / "slice_counts.csv").resolve()),
            "slice_summary_json": str((analysis_dir / "slice_summary.json").resolve()),
            "image_sampling_weights_csv": str((analysis_dir / "image_sampling_weights.csv").resolve()),
            "sampling_weight_summary_json": str((analysis_dir / "sampling_weight_summary.json").resolve()),
        }

    input_channels = int(cfg.model.ssdlite.input_channels)
    train_dataset = SimpleYoloDataset(
        train_info,
        image_size=int(cfg.data.image_size),
        input_channels=input_channels,
        augment=general_aug_enabled or rare_slice_aug,
        prepared_baseline=prepared_baseline,
        baseline_cfg=cfg.baseline,
        aug_cfg=aug_cfg,
        cache_images=cache_images,
    )
    val_dataset = SimpleYoloDataset(
        val_info,
        image_size=int(cfg.data.image_size),
        input_channels=input_channels,
        augment=False,
        cache_images=cache_images,
    )
    train_loader = _make_dataloader(
        train_dataset, cfg=cfg, shuffle=True,
        use_weighted_sampler=prepared_baseline is not None and bool(cfg.baseline.use_weighted_sampler),
    )
    val_loader = _make_dataloader(val_dataset, cfg=cfg, shuffle=False, use_weighted_sampler=False)

    model = build_ssdlite_model(cfg, nc=train_info.nc).to(torch_device)
    criterion = SSDLiteLoss(cfg.loss)
    optimizer = _make_optimizer(cfg, model)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.training.epochs))
        if bool(cfg.training.cos_lr)
        else None
    )
    precision, scaler = setup_precision(torch_device, cfg.training.mixed_precision)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    writer = build_summary_writer(str(run_dir.resolve()))

    _write_json(
        analysis_dir / "train_stage_plan.json",
        {
            "experiment_name": cfg.output.experiment_name,
            "backend": "ssdlite",
            "dataset_yaml": dataset_yaml,
            "baseline_mode": cfg.baseline.mode,
            "baseline_artifact_paths": baseline_artifact_paths,
            "stages": [{"stage_index": 1, "stage_name": "ssdlite", "epochs": int(cfg.training.epochs)}],
        },
    )

    iou_pos = float(cfg.model.ssdlite.iou_pos_threshold)
    iou_neg = float(cfg.model.ssdlite.iou_neg_threshold)
    nms_iou_eval = float(cfg.evaluation.iou) if cfg.evaluation.iou is not None else 0.45

    history: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_epoch = -1
    bad_epochs = 0
    last_metrics: dict[str, Any] = {}
    try:
        for epoch_idx in range(1, int(cfg.training.epochs) + 1):
            model.train()
            totals: dict[str, float] = {}
            n_batches = 0
            for batch in train_loader:
                images = batch["images"].to(device=torch_device, non_blocking=True)
                boxes_xywh = [t.to(torch_device) for t in batch["boxes_xywh"]]
                class_ids = [t.to(torch_device) for t in batch["class_ids"]]
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(precision):
                    cls_logits, bbox_pred, anchors = model(images)
                    loss, loss_parts = criterion(
                        cls_logits, bbox_pred, anchors,
                        boxes_xywh=boxes_xywh, class_ids=class_ids,
                        iou_pos_threshold=iou_pos, iou_neg_threshold=iou_neg,
                    )
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if cfg.training.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(cfg.training.grad_clip_norm)
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.training.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(cfg.training.grad_clip_norm)
                        )
                    optimizer.step()
                for key, value in loss_parts.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                n_batches += 1
            if scheduler is not None:
                scheduler.step()

            row: dict[str, Any] = {
                "epoch": epoch_idx,
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{key: value / max(n_batches, 1) for key, value in totals.items()},
            }
            writer.add_scalar("train/lr", float(row["lr"]), epoch_idx)
            for key, value in row.items():
                if key not in {"epoch", "lr"} and isinstance(value, (int, float)):
                    writer.add_scalar(f"train/{key}", float(value), epoch_idx)

            should_eval = (
                epoch_idx == int(cfg.training.epochs)
                or epoch_idx % int(cfg.training.val_interval) == 0
            )
            if should_eval:
                eval_payload = evaluate_ssdlite_model(
                    model, val_loader,
                    device=torch_device,
                    names=train_info.names,
                    conf_threshold=0.001,
                    nms_iou=nms_iou_eval,
                )
                last_metrics = dict(eval_payload["summary"])
                row.update({
                    f"val_{key}": value
                    for key, value in last_metrics.items()
                    if isinstance(value, (int, float))
                })
                for key, value in last_metrics.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        writer.add_scalar(f"val/{key}", float(value), epoch_idx)
                image_interval = int(cfg.training.tensorboard_image_interval)
                if image_interval > 0 and epoch_idx % image_interval == 0:
                    image_batch = render_ssdlite_detection_batch(
                        model, val_dataset,
                        device=torch_device,
                        names=train_info.names,
                        max_images=int(cfg.training.tensorboard_max_images),
                        conf_threshold=float(cfg.training.tensorboard_prediction_conf),
                        nms_iou=nms_iou_eval,
                    )
                    if image_batch is not None:
                        writer.add_images("val/detection_overlays", image_batch, epoch_idx)
                metric = float(last_metrics.get("map50", float("nan")))
                if not math.isfinite(metric):
                    metric = -float(row.get("loss", 0.0))
                if metric > best_metric:
                    best_metric = metric
                    best_epoch = epoch_idx
                    bad_epochs = 0
                    _save_ssdlite_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=model, optimizer=optimizer,
                        cfg=cfg, split_info=train_info,
                        epoch=epoch_idx, metrics=last_metrics,
                    )
                else:
                    bad_epochs += 1
            else:
                bad_epochs += 1

            history.append(row)
            _ssdlite_log_epoch_progress(row, total_epochs=int(cfg.training.epochs))
            _save_ssdlite_checkpoint(
                checkpoint_dir / "last.pt",
                model=model, optimizer=optimizer,
                cfg=cfg, split_info=train_info,
                epoch=epoch_idx, metrics=last_metrics,
            )
            writer.flush()
            if int(cfg.training.patience) > 0 and bad_epochs >= int(cfg.training.patience):
                break
    finally:
        writer.close()

    if not (checkpoint_dir / "best.pt").exists():
        _save_ssdlite_checkpoint(
            checkpoint_dir / "best.pt",
            model=model, optimizer=optimizer,
            cfg=cfg, split_info=train_info,
            epoch=history[-1]["epoch"],
            metrics=last_metrics,
        )

    pd.DataFrame(history).to_csv(analysis_dir / "loss_history.csv", index=False)
    summary = {
        "experiment_name": cfg.output.experiment_name,
        "backend": "ssdlite",
        "dataset_yaml": dataset_yaml,
        "baseline_mode": cfg.baseline.mode,
        "baseline_settings": vars(cfg.baseline),
        "baseline_artifact_paths": baseline_artifact_paths,
        "run_dir": str(run_dir.resolve()),
        "tensorboard_log_dir": str(run_dir.resolve()),
        "best_weights_path": str((checkpoint_dir / "best.pt").resolve()),
        "last_weights_path": str((checkpoint_dir / "last.pt").resolve()),
        "loss_history_csv": str((analysis_dir / "loss_history.csv").resolve()),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "last_metrics": last_metrics,
        "model_parameters": count_trainable_parameters(model),
        "stages": [
            {
                "stage_index": 1,
                "stage_name": "ssdlite",
                "epochs": int(history[-1]["epoch"]) if history else 0,
                "best_weights_path": str((checkpoint_dir / "best.pt").resolve()),
                "last_weights_path": str((checkpoint_dir / "last.pt").resolve()),
            }
        ],
    }
    _write_json(analysis_dir / "train_summary.json", summary)
    pd.DataFrame([summary]).to_csv(analysis_dir / "train_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------

def eval_ssdlite(
    cfg: Any,
    *,
    weights_path: str,
    data_yaml: str,
    analysis_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Evaluate an SSDLite checkpoint."""

    split_info = resolve_yolo_split_info(data_yaml, split=str(cfg.evaluation.split))
    torch_device = torch.device(device)
    model, payload = load_ssdlite_checkpoint(weights_path, map_location=torch_device)
    if int(payload.get("nc", -1)) != split_info.nc:
        raise ValueError(
            f"Checkpoint nc={payload.get('nc')} does not match dataset nc={split_info.nc}."
        )
    model = model.to(torch_device)

    dataset = SimpleYoloDataset(
        split_info,
        image_size=int(cfg.data.image_size),
        input_channels=int(model.config.input_channels),
        augment=False,
    )
    dataloader = _make_dataloader(dataset, cfg=cfg, shuffle=False, use_weighted_sampler=False)
    conf_threshold = float(cfg.evaluation.conf) if cfg.evaluation.conf is not None else 0.001
    nms_iou = float(cfg.evaluation.iou) if cfg.evaluation.iou is not None else 0.45

    eval_payload = evaluate_ssdlite_model(
        model, dataloader,
        device=torch_device,
        names=split_info.names,
        conf_threshold=conf_threshold,
        nms_iou=nms_iou,
    )
    summary = {
        "experiment_name": cfg.output.experiment_name,
        "backend": "ssdlite",
        "dataset_yaml": str(Path(data_yaml).resolve()),
        "weights_path": str(Path(weights_path).resolve()),
        "split": cfg.evaluation.split,
        **eval_payload["summary"],
        "raw_results": eval_payload["summary"],
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_json(analysis_dir / "eval_summary.json", summary)
    pd.DataFrame([
        {key: value for key, value in summary.items() if key != "raw_results"}
    ]).to_csv(analysis_dir / "eval_summary.csv", index=False)
    per_class_df = pd.DataFrame(eval_payload["per_class"])
    if not per_class_df.empty:
        per_class_df.to_csv(analysis_dir / "per_class_metrics.csv", index=False)
    if bool(cfg.evaluation.save_json):
        _write_json(
            analysis_dir / "predictions.json",
            {"predictions": _prediction_rows(eval_payload["predictions"], split_info.names)},
        )
    return {
        **summary,
        "per_class": eval_payload["per_class"],
        "predictions": eval_payload["predictions"],
        "ground_truths": eval_payload["ground_truths"],
        "names": split_info.names,
    }


def eval_ssdlite_slices(
    cfg: Any,
    *,
    weights_path: str,
    data_yaml: str,
    threshold_yaml: str,
    analysis_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Run SSDLite evaluation + native per-slice breakdown."""

    from src.analysis.flir_subgroup.yolo_slice_eval import (
        compute_frozen_thresholds,
        evaluate_per_slice_from_predictions,
    )

    eval_payload = eval_ssdlite(
        cfg,
        weights_path=weights_path,
        data_yaml=data_yaml,
        analysis_dir=analysis_dir,
        device=device,
    )
    thresholds = compute_frozen_thresholds(threshold_yaml)
    per_slice_results = evaluate_per_slice_from_predictions(
        predictions=eval_payload["predictions"],
        test_yaml=data_yaml,
        thresholds=thresholds,
        output_dir=analysis_dir,
        iou_threshold=float(cfg.evaluation.iou) if cfg.evaluation.iou is not None else 0.5,
    )
    summary = {
        "experiment_name": cfg.output.experiment_name,
        "backend": "ssdlite",
        "weights_path": weights_path,
        "dataset_yaml": data_yaml,
        "per_slice_metrics_csv": str(analysis_dir / "per_slice_metrics.csv"),
        "per_slice_metrics_json": str(analysis_dir / "per_slice_metrics.json"),
        "per_slice_overall": per_slice_results.get("overall", {}),
    }
    _write_json(analysis_dir / "eval_slices_summary.json", summary)
    return summary
