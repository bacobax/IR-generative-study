"""Native PyTorch training and evaluation for the simple YOLO backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.algorithms.training.yolo_slice_baselines import (
    apply_center_scale,
    apply_constrained_crop_resize,
    apply_horizontal_flip,
    apply_translation,
    prepare_yolo_slice_baseline,
    valid_box_mask,
)
from src.core.configs.config_loader import load_yaml
from src.core.training_utils import autocast_context, build_summary_writer
from src.core.training_runtime import setup_precision
from src.evaluation.detection_metrics import (
    DetectionGroundTruth,
    DetectionPrediction,
    evaluate_detections,
    nms_numpy,
    xywh_to_xyxy_np,
)
from src.models.simple_yolo import (
    SimpleYOLOConfig,
    SimpleYOLODetector,
    count_trainable_parameters,
)


@dataclass(frozen=True)
class YoloSplitInfo:
    """Resolved YOLO-format split directories."""

    dataset_yaml: str
    split: str
    image_dir: str
    label_dir: str
    names: dict[int, str]
    nc: int


@dataclass
class SimpleYoloTargets:
    """Dense grid targets for one native YOLO batch."""

    objectness: torch.Tensor
    boxes_xywh: torch.Tensor
    classes: torch.Tensor
    assigned_count: int
    dropped_count: int


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _split_label_dir(image_dir: Path) -> Path:
    if image_dir.parent.name == "images":
        return image_dir.parent.parent / "labels" / image_dir.name
    if image_dir.name == "images":
        return image_dir.parent / "labels"
    return image_dir.parents[1] / "labels" / image_dir.name


def _parse_names(raw_names: Any, nc: Any = None) -> dict[int, str]:
    if isinstance(raw_names, dict):
        names = {int(idx): str(label) for idx, label in raw_names.items()}
    elif isinstance(raw_names, list):
        names = {idx: str(label) for idx, label in enumerate(raw_names)}
    else:
        count = int(nc or 0)
        names = {idx: str(idx) for idx in range(count)}
    if not names:
        raise ValueError("YOLO dataset YAML must define at least one class name.")
    return dict(sorted(names.items()))


def resolve_yolo_split_info(dataset_yaml: str | Path, *, split: str) -> YoloSplitInfo:
    """Resolve one split from a YOLO dataset YAML."""

    yaml_path = Path(dataset_yaml).resolve()
    payload = load_yaml(yaml_path)
    ds_root = Path(str(payload.get("path", yaml_path.parent))).resolve()
    if split not in payload:
        raise ValueError(f"YOLO dataset YAML {yaml_path} does not define split {split!r}.")
    image_dir = Path(str(payload[split]))
    if not image_dir.is_absolute():
        image_dir = (ds_root / image_dir).resolve()
    label_dir = _split_label_dir(image_dir).resolve()
    names = _parse_names(payload.get("names", {}), payload.get("nc"))
    nc = int(payload.get("nc", len(names)))
    if nc != len(names):
        raise ValueError(f"YOLO dataset nc={nc} does not match names count={len(names)} in {yaml_path}.")
    if not image_dir.exists():
        raise FileNotFoundError(f"YOLO image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"YOLO label directory not found: {label_dir}")
    return YoloSplitInfo(
        dataset_yaml=str(yaml_path),
        split=str(split),
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        names=names,
        nc=nc,
    )


def _read_label_file(label_path: Path, *, nc: int) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)
    boxes: list[list[float]] = []
    classes: list[int] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 YOLO columns in {label_path}, got: {line!r}")
        class_id = int(parts[0])
        if class_id < 0 or class_id >= int(nc):
            raise ValueError(f"Class id {class_id} in {label_path} is outside [0, {nc}).")
        cx, cy, width, height = (float(value) for value in parts[1:])
        boxes.append([cx, cy, width, height])
        classes.append(class_id)
    return (
        np.clip(np.asarray(boxes, dtype=np.float32), 0.0, 1.0).reshape(-1, 4),
        np.asarray(classes, dtype=np.int64).reshape(-1),
    )


class SimpleYoloDataset(Dataset):
    """YOLO-format image/label dataset for the native detector."""

    def __init__(
        self,
        split_info: YoloSplitInfo,
        *,
        image_size: int,
        input_channels: int,
        augment: bool = False,
        prepared_baseline: Any = None,
        baseline_cfg: Any = None,
        aug_cfg: Any = None,
        cache_images: bool = False,
    ) -> None:
        self.split_info = split_info
        self.image_size = int(image_size)
        self.input_channels = int(input_channels)
        self.augment = bool(augment)
        self.prepared_baseline = prepared_baseline
        self.baseline_cfg = baseline_cfg
        self.aug_cfg = aug_cfg
        if self.input_channels not in {1, 3}:
            raise ValueError("simple.input_channels currently supports 1 or 3 for native dataset loading.")
        image_dir = Path(split_info.image_dir)
        self.image_paths = sorted(path for path in image_dir.glob("*") if path.is_file())
        if not self.image_paths:
            raise ValueError(f"No images found for YOLO split under: {image_dir}")
        metadata = prepared_baseline.image_metadata_by_stem if prepared_baseline is not None else {}
        self.sampling_weights = np.asarray(
            [float(metadata.get(path.stem, {}).get("sampling_weight", 1.0)) for path in self.image_paths],
            dtype=np.float64,
        )

        # Build in-memory cache: decode + resize each image and parse labels once.
        self._image_cache: list[np.ndarray] | None = None
        self._label_cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        if cache_images:
            self._image_cache = []
            self._label_cache = []
            pil_mode = "L" if self.input_channels == 1 else "RGB"
            for img_path in self.image_paths:
                with Image.open(img_path) as im:
                    pil_img = im.convert(pil_mode).resize(
                        (self.image_size, self.image_size), Image.BILINEAR
                    )
                arr = np.array(pil_img, dtype=np.uint8, copy=True)
                if self.input_channels == 1:
                    arr = arr[:, :, None]
                self._image_cache.append(arr)
                lbl_path = Path(self.split_info.label_dir) / f"{img_path.stem}.txt"
                boxes_xywh, class_ids = _read_label_file(lbl_path, nc=self.split_info.nc)
                self._label_cache.append((boxes_xywh, class_ids))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[int(index)]

        if self._image_cache is not None:
            # Read from cache — copy so augmentation can mutate safely.
            array = self._image_cache[index].copy()
            boxes_xywh, class_ids = (
                self._label_cache[index][0].copy(),
                self._label_cache[index][1].copy(),
            )
        else:
            label_path = Path(self.split_info.label_dir) / f"{image_path.stem}.txt"
            boxes_xywh, class_ids = _read_label_file(label_path, nc=self.split_info.nc)
            pil_mode = "L" if self.input_channels == 1 else "RGB"
            with Image.open(image_path) as image:
                pil_image = image.convert(pil_mode).resize(
                    (self.image_size, self.image_size), Image.BILINEAR
                )
            array = np.array(pil_image, dtype=np.uint8, copy=True)
            if self.input_channels == 1:
                array = array[:, :, None]

        if self.augment:
            # General all-image augmentation (enabled via augment_cfg, no baseline required).
            if self.aug_cfg is not None and bool(self.aug_cfg.enabled):
                array, boxes_xywh, class_ids = _apply_general_aug(
                    array, boxes_xywh, class_ids, self.aug_cfg
                )
            else:
                # Rare-slice targeted augmentation (baseline_b).
                array, boxes_xywh, class_ids = self._maybe_apply_targeted_aug(image_path.stem, array, boxes_xywh, class_ids)

        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255.0
        return {
            "image": tensor,
            "boxes_xywh": torch.from_numpy(boxes_xywh.astype(np.float32)),
            "class_ids": torch.from_numpy(class_ids.astype(np.int64)),
            "image_id": image_path.stem,
            "image_path": str(image_path),
        }

    def _maybe_apply_targeted_aug(
        self,
        stem: str,
        image: np.ndarray,
        boxes_xywh: np.ndarray,
        class_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.prepared_baseline is None or self.baseline_cfg is None:
            return image, boxes_xywh, class_ids
        if str(getattr(self.baseline_cfg, "mode", "none")) != "baseline_b":
            return image, boxes_xywh, class_ids
        if len(boxes_xywh) == 0:
            return image, boxes_xywh, class_ids
        metadata = self.prepared_baseline.image_metadata_by_stem.get(stem)
        if metadata is None:
            return image, boxes_xywh, class_ids
        if float(metadata.get("image_score", 0.0)) < float(self.prepared_baseline.targeted_image_score_threshold):
            return image, boxes_xywh, class_ids
        if random.random() > float(getattr(self.baseline_cfg, "targeted_aug_probability", 0.0)):
            return image, boxes_xywh, class_ids
        protected_indices = [int(idx) for idx in metadata.get("protected_instance_indices", [])]
        if not protected_indices:
            return image, boxes_xywh, class_ids

        original_boxes = boxes_xywh.astype(np.float32).copy()
        transformed_image = image
        transformed_boxes = original_boxes
        transformed_image, transformed_boxes = apply_translation(
            transformed_image,
            transformed_boxes,
            dx=random.uniform(-float(self.baseline_cfg.translate_fraction), float(self.baseline_cfg.translate_fraction)),
            dy=random.uniform(-float(self.baseline_cfg.translate_fraction), float(self.baseline_cfg.translate_fraction)),
        )
        transformed_image, transformed_boxes = apply_center_scale(
            transformed_image,
            transformed_boxes,
            scale=random.uniform(float(self.baseline_cfg.scale_min), float(self.baseline_cfg.scale_max)),
        )
        crop_result = apply_constrained_crop_resize(
            transformed_image,
            transformed_boxes,
            protected_indices=protected_indices,
            crop_scale_min=float(self.baseline_cfg.crop_scale_min),
            crop_scale_max=float(self.baseline_cfg.crop_scale_max),
            min_retained_area=float(self.baseline_cfg.crop_min_rare_box_area_retained),
            max_attempts=int(self.baseline_cfg.crop_max_attempts),
        )
        if crop_result is not None:
            transformed_image = crop_result["img"]
            transformed_boxes = crop_result["boxes"]
        if bool(self.baseline_cfg.allow_horizontal_flip) and random.random() < 0.5:
            transformed_image, transformed_boxes = apply_horizontal_flip(transformed_image, transformed_boxes)
        keep_mask = valid_box_mask(transformed_boxes)
        if not keep_mask.any():
            return image, boxes_xywh, class_ids
        if not any(idx < len(keep_mask) and bool(keep_mask[idx]) for idx in protected_indices):
            return image, boxes_xywh, class_ids
        return (
            np.ascontiguousarray(transformed_image),
            transformed_boxes[keep_mask].astype(np.float32),
            class_ids[keep_mask],
        )


def simple_yolo_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "boxes_xywh": [item["boxes_xywh"] for item in batch],
        "class_ids": [item["class_ids"] for item in batch],
        "image_ids": [str(item["image_id"]) for item in batch],
        "image_paths": [str(item["image_path"]) for item in batch],
    }


def build_simple_yolo_targets(
    *,
    boxes_xywh: list[torch.Tensor],
    class_ids: list[torch.Tensor],
    batch_size: int,
    grid_h: int,
    grid_w: int,
    boxes_per_cell: int,
    nc: int,
    device: torch.device,
) -> SimpleYoloTargets:
    """Assign boxes to center cells and first available per-cell slot."""

    objectness = torch.zeros((batch_size, boxes_per_cell, grid_h, grid_w), dtype=torch.float32, device=device)
    target_boxes = torch.zeros((batch_size, boxes_per_cell, 4, grid_h, grid_w), dtype=torch.float32, device=device)
    target_classes = torch.zeros((batch_size, boxes_per_cell, nc, grid_h, grid_w), dtype=torch.float32, device=device)
    assigned = 0
    dropped = 0
    for batch_idx, (box_tensor, cls_tensor) in enumerate(zip(boxes_xywh, class_ids)):
        boxes = box_tensor.to(device=device, dtype=torch.float32)
        classes = cls_tensor.to(device=device, dtype=torch.long)
        for box, class_id in zip(boxes, classes):
            cx, cy, width, height = [float(value) for value in box.tolist()]
            if width <= 0.0 or height <= 0.0:
                dropped += 1
                continue
            grid_x = min(max(int(cx * grid_w), 0), grid_w - 1)
            grid_y = min(max(int(cy * grid_h), 0), grid_h - 1)
            free_slots = torch.nonzero(objectness[batch_idx, :, grid_y, grid_x] < 0.5, as_tuple=False).flatten()
            if len(free_slots) == 0:
                dropped += 1
                continue
            slot = int(free_slots[0].item())
            objectness[batch_idx, slot, grid_y, grid_x] = 1.0
            target_boxes[batch_idx, slot, :, grid_y, grid_x] = box
            target_classes[batch_idx, slot, int(class_id.item()), grid_y, grid_x] = 1.0
            assigned += 1
    return SimpleYoloTargets(
        objectness=objectness,
        boxes_xywh=target_boxes,
        classes=target_classes,
        assigned_count=assigned,
        dropped_count=dropped,
    )


def decode_simple_yolo_grid(output: torch.Tensor) -> torch.Tensor:
    """Decode raw model output to normalized xywh boxes for all cells."""

    b, a, _, grid_h, grid_w = output.shape
    device = output.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=torch.float32),
        torch.arange(grid_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cx = (torch.sigmoid(output[:, :, 0]) + grid_x.view(1, 1, grid_h, grid_w)) / float(grid_w)
    cy = (torch.sigmoid(output[:, :, 1]) + grid_y.view(1, 1, grid_h, grid_w)) / float(grid_h)
    width = torch.sigmoid(output[:, :, 2])
    height = torch.sigmoid(output[:, :, 3])
    return torch.stack([cx, cy, width, height], dim=2)


def _xywh_to_xyxy_torch(boxes_xywh: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes_xywh.unbind(dim=-1)
    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = cx + width / 2.0
    y2 = cy + height / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def _generalized_iou_diag(boxes_a_xyxy: torch.Tensor, boxes_b_xyxy: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes_a_xyxy[:, 0], boxes_b_xyxy[:, 0])
    y1 = torch.maximum(boxes_a_xyxy[:, 1], boxes_b_xyxy[:, 1])
    x2 = torch.minimum(boxes_a_xyxy[:, 2], boxes_b_xyxy[:, 2])
    y2 = torch.minimum(boxes_a_xyxy[:, 3], boxes_b_xyxy[:, 3])
    inter = (x2 - x1).clamp_min(0.0) * (y2 - y1).clamp_min(0.0)
    area_a = (boxes_a_xyxy[:, 2] - boxes_a_xyxy[:, 0]).clamp_min(0.0) * (
        boxes_a_xyxy[:, 3] - boxes_a_xyxy[:, 1]
    ).clamp_min(0.0)
    area_b = (boxes_b_xyxy[:, 2] - boxes_b_xyxy[:, 0]).clamp_min(0.0) * (
        boxes_b_xyxy[:, 3] - boxes_b_xyxy[:, 1]
    ).clamp_min(0.0)
    union = (area_a + area_b - inter).clamp_min(1e-12)
    iou = inter / union
    cx1 = torch.minimum(boxes_a_xyxy[:, 0], boxes_b_xyxy[:, 0])
    cy1 = torch.minimum(boxes_a_xyxy[:, 1], boxes_b_xyxy[:, 1])
    cx2 = torch.maximum(boxes_a_xyxy[:, 2], boxes_b_xyxy[:, 2])
    cy2 = torch.maximum(boxes_a_xyxy[:, 3], boxes_b_xyxy[:, 3])
    enclosing = ((cx2 - cx1).clamp_min(0.0) * (cy2 - cy1).clamp_min(0.0)).clamp_min(1e-12)
    return iou - (enclosing - union) / enclosing


class SimpleYoloLoss(nn.Module):
    """Loss for the anchor-free grid detector."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.box_weight = float(cfg.box_weight)
        self.giou_weight = float(cfg.giou_weight)
        self.objectness_weight = float(cfg.objectness_weight)
        self.no_object_weight = float(cfg.no_object_weight)
        self.class_weight = float(cfg.class_weight)

    def forward(
        self,
        output: torch.Tensor,
        *,
        boxes_xywh: list[torch.Tensor],
        class_ids: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        b, a, channels, grid_h, grid_w = output.shape
        nc = channels - 5
        targets = build_simple_yolo_targets(
            boxes_xywh=boxes_xywh,
            class_ids=class_ids,
            batch_size=b,
            grid_h=grid_h,
            grid_w=grid_w,
            boxes_per_cell=a,
            nc=nc,
            device=output.device,
        )
        obj_logits = output[:, :, 4]
        obj_loss_raw = F.binary_cross_entropy_with_logits(
            obj_logits,
            targets.objectness,
            reduction="none",
        )
        obj_weights = torch.where(
            targets.objectness > 0.5,
            torch.full_like(obj_loss_raw, self.objectness_weight),
            torch.full_like(obj_loss_raw, self.no_object_weight),
        )
        obj_loss = (obj_loss_raw * obj_weights).mean()

        pos_mask = targets.objectness > 0.5
        zero = output.sum() * 0.0
        if pos_mask.any():
            pred_boxes = decode_simple_yolo_grid(output).permute(0, 1, 3, 4, 2)[pos_mask]
            target_boxes = targets.boxes_xywh.permute(0, 1, 3, 4, 2)[pos_mask]
            box_loss = F.l1_loss(pred_boxes, target_boxes)
            giou = _generalized_iou_diag(_xywh_to_xyxy_torch(pred_boxes), _xywh_to_xyxy_torch(target_boxes))
            giou_loss = (1.0 - giou).mean()
            cls_logits = output[:, :, 5:].permute(0, 1, 3, 4, 2)[pos_mask]
            cls_targets = targets.classes.permute(0, 1, 3, 4, 2)[pos_mask]
            class_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets)
        else:
            box_loss = zero
            giou_loss = zero
            class_loss = zero

        total = (
            self.box_weight * box_loss
            + self.giou_weight * giou_loss
            + obj_loss
            + self.class_weight * class_loss
        )
        return total, {
            "loss": float(total.detach().cpu()),
            "box_loss": float(box_loss.detach().cpu()),
            "giou_loss": float(giou_loss.detach().cpu()),
            "objectness_loss": float(obj_loss.detach().cpu()),
            "class_loss": float(class_loss.detach().cpu()),
            "assigned_targets": float(targets.assigned_count),
            "dropped_targets": float(targets.dropped_count),
        }


def _log_epoch_progress(row: dict[str, Any], *, total_epochs: int) -> None:
    epoch = int(row["epoch"])
    loss_keys = ["loss", "box_loss", "giou_loss", "objectness_loss", "class_loss"]
    val_keys = ["val_map50", "val_map", "val_precision", "val_recall"]
    parts = [f"[epoch {epoch:>3d}/{total_epochs}]"]
    parts += [f"{k}={row[k]:.4f}" for k in loss_keys if k in row]
    val_parts = [f"{k}={row[k]:.4f}" for k in val_keys if k in row]
    if val_parts:
        parts += ["|"] + val_parts
    print("  ".join(parts), flush=True)


def _apply_general_aug(
    array: np.ndarray,
    boxes_xywh: np.ndarray,
    class_ids: np.ndarray,
    aug_cfg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply ultralytics-like general augmentation to one image.

    Geometric ops reuse existing primitives from yolo_slice_baselines.
    Photometric ops (brightness/contrast) work directly on the uint8 HWC array.
    Box validity is enforced after geometric transforms.
    """
    # --- photometric: brightness/contrast ---
    brightness = float(aug_cfg.brightness)
    contrast = float(aug_cfg.contrast)
    if brightness > 0.0 or contrast > 0.0:
        arr_f = array.astype(np.float32)
        if brightness > 0.0:
            delta = random.uniform(-brightness, brightness) * 255.0
            arr_f = arr_f + delta
        if contrast > 0.0:
            factor = 1.0 + random.uniform(-contrast, contrast)
            mean = arr_f.mean()
            arr_f = (arr_f - mean) * factor + mean
        array = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)

    # --- geometric: scale ---
    scale_gain = float(aug_cfg.scale)
    if scale_gain > 0.0 and len(boxes_xywh) > 0:
        scale = 1.0 + random.uniform(-scale_gain, scale_gain)
        array, boxes_xywh = apply_center_scale(array, boxes_xywh, scale=scale)
    elif scale_gain > 0.0:
        scale = 1.0 + random.uniform(-scale_gain, scale_gain)
        array, _ = apply_center_scale(array, boxes_xywh, scale=scale)

    # --- geometric: translate ---
    translate = float(aug_cfg.translate)
    if translate > 0.0:
        dx = random.uniform(-translate, translate)
        dy = random.uniform(-translate, translate)
        array, boxes_xywh = apply_translation(array, boxes_xywh, dx=dx, dy=dy)

    # --- geometric: horizontal flip ---
    fliplr = float(aug_cfg.fliplr)
    if fliplr > 0.0 and random.random() < fliplr:
        array, boxes_xywh = apply_horizontal_flip(array, boxes_xywh)

    # Drop degenerate boxes introduced by geometric transforms.
    if len(boxes_xywh) > 0:
        keep = valid_box_mask(boxes_xywh)
        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]

    return array, boxes_xywh, class_ids


def _config_from_yolo_cfg(cfg: Any, *, nc: int) -> SimpleYOLOConfig:
    payload = asdict(cfg.model.simple) if hasattr(cfg.model.simple, "__dataclass_fields__") else dict(cfg.model.simple)
    return SimpleYOLOConfig.from_mapping(payload, nc=nc)


def build_simple_yolo_model(cfg: Any, *, nc: int) -> SimpleYOLODetector:
    return SimpleYOLODetector(_config_from_yolo_cfg(cfg, nc=nc))


def _validate_native_config(cfg: Any, *, split_info: YoloSplitInfo) -> None:
    if str(cfg.model.backend) != "simple_torch":
        raise ValueError("Native simple YOLO trainer requires model.backend: simple_torch.")
    image_size = int(cfg.data.image_size)
    output_stride = int(cfg.model.simple.output_stride)
    if image_size <= 0:
        raise ValueError("data.image_size must be positive.")
    if image_size % output_stride != 0:
        raise ValueError("data.image_size must be divisible by model.simple.output_stride.")
    if int(cfg.training.epochs) <= 0:
        raise ValueError("training.epochs must be positive.")
    if int(cfg.training.val_interval) <= 0:
        raise ValueError("training.val_interval must be positive.")
    if int(cfg.training.tensorboard_image_interval) < 0:
        raise ValueError("training.tensorboard_image_interval must be >= 0.")
    if int(cfg.training.tensorboard_max_images) < 0:
        raise ValueError("training.tensorboard_max_images must be >= 0.")
    if not 0.0 <= float(cfg.training.tensorboard_prediction_conf) <= 1.0:
        raise ValueError("training.tensorboard_prediction_conf must be in [0, 1].")
    if int(cfg.model.simple.boxes_per_cell) <= 0:
        raise ValueError("model.simple.boxes_per_cell must be positive.")
    if split_info.nc <= 0:
        raise ValueError("Native simple YOLO requires a dataset with at least one class.")


def _make_dataloader(
    dataset: SimpleYoloDataset,
    *,
    cfg: Any,
    shuffle: bool,
    use_weighted_sampler: bool,
) -> DataLoader:
    sampler = None
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(dataset.sampling_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=bool(cfg.baseline.sampler_replacement),
        )
        shuffle = False
    num_workers = int(cfg.data.workers)
    return DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=bool(shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
        collate_fn=simple_yolo_collate,
    )


def _make_optimizer(cfg: Any, model: nn.Module) -> torch.optim.Optimizer:
    name = str(cfg.training.optimizer or "auto").lower()
    if name == "auto":
        name = "adamw"
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr0), weight_decay=float(cfg.training.weight_decay))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr0), weight_decay=float(cfg.training.weight_decay))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.training.lr0),
            momentum=float(cfg.training.momentum),
            weight_decay=float(cfg.training.weight_decay),
            nesterov=True,
        )
    raise ValueError("Native simple YOLO optimizer must be one of: auto, AdamW, Adam, SGD.")


def _load_initial_weights_if_present(model: nn.Module, weights: str | None, *, device: torch.device) -> None:
    if not weights:
        return
    path = Path(str(weights))
    if not path.exists():
        return
    payload = torch.load(path, map_location=device)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)


def _save_checkpoint(
    path: Path,
    *,
    model: SimpleYOLODetector,
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    split_info: YoloSplitInfo,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "simple_yolo_detector_v1",
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
    }
    torch.save(payload, path)


def load_simple_yolo_checkpoint(
    weights_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SimpleYOLODetector, dict[str, Any]]:
    payload = torch.load(weights_path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format") != "simple_yolo_detector_v1":
        raise ValueError(f"Not a native simple YOLO checkpoint: {weights_path}")
    model = SimpleYOLODetector(SimpleYOLOConfig.from_mapping(payload["model_config"], nc=int(payload["nc"])))
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def decode_simple_yolo_predictions(
    output: torch.Tensor,
    *,
    names: dict[int, str],
    conf_threshold: float,
    nms_iou: float,
) -> list[DetectionPrediction]:
    """Decode model output for each batch item."""

    decoded_boxes = decode_simple_yolo_grid(output).detach().cpu()
    objectness = torch.sigmoid(output[:, :, 4]).detach().cpu()
    class_probs = torch.sigmoid(output[:, :, 5:]).detach().cpu()
    b, a, nc, grid_h, grid_w = class_probs.shape
    predictions: list[DetectionPrediction] = []
    for batch_idx in range(b):
        boxes_list: list[np.ndarray] = []
        scores_list: list[float] = []
        class_list: list[int] = []
        for class_id in sorted(names):
            cls_score = class_probs[batch_idx, :, int(class_id)] * objectness[batch_idx]
            keep = cls_score >= float(conf_threshold)
            if not keep.any():
                continue
            boxes_xywh = decoded_boxes[batch_idx].permute(0, 2, 3, 1)[keep].numpy()
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
            boxes_arr = boxes_arr[order]
            scores_arr = scores_arr[order]
            class_arr = class_arr[order]
        else:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            scores_arr = np.zeros(0, dtype=np.float32)
            class_arr = np.zeros(0, dtype=np.int32)
        predictions.append(DetectionPrediction("", boxes_arr, scores_arr, class_arr))
    return predictions


def _ground_truth_from_batch(batch: dict[str, Any]) -> list[DetectionGroundTruth]:
    ground_truths: list[DetectionGroundTruth] = []
    for image_id, boxes_xywh, class_ids in zip(batch["image_ids"], batch["boxes_xywh"], batch["class_ids"]):
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


@torch.inference_mode()
def collect_simple_yolo_predictions(
    model: SimpleYOLODetector,
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
        output = model(images)
        batch_predictions = decode_simple_yolo_predictions(
            output,
            names=names,
            conf_threshold=float(conf_threshold),
            nms_iou=float(nms_iou),
        )
        for pred, image_id in zip(batch_predictions, batch["image_ids"]):
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


def evaluate_simple_yolo_model(
    model: SimpleYOLODetector,
    dataloader: DataLoader,
    *,
    device: torch.device,
    names: dict[int, str],
    conf_threshold: float,
    nms_iou: float,
) -> dict[str, Any]:
    predictions, ground_truths = collect_simple_yolo_predictions(
        model,
        dataloader,
        device=device,
        names=names,
        conf_threshold=float(conf_threshold),
        nms_iou=float(nms_iou),
    )
    metrics = evaluate_detections(predictions=predictions, ground_truths=ground_truths, names=names)
    return {
        "summary": metrics["summary"],
        "per_class": metrics["per_class"],
        "predictions": predictions,
        "ground_truths": ground_truths,
    }


def _prediction_rows(predictions: list[DetectionPrediction], names: dict[int, str]) -> list[dict[str, Any]]:
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


def _draw_normalized_box(
    draw: ImageDraw.ImageDraw,
    box_xyxy: np.ndarray,
    *,
    image_size: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    width, height = image_size
    x1 = float(np.clip(box_xyxy[0], 0.0, 1.0)) * width
    y1 = float(np.clip(box_xyxy[1], 0.0, 1.0)) * height
    x2 = float(np.clip(box_xyxy[2], 0.0, 1.0)) * width
    y2 = float(np.clip(box_xyxy[3], 0.0, 1.0)) * height
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    if label:
        text_y = max(0.0, y1 - 12.0)
        draw.rectangle([x1, text_y, min(width, x1 + 7 * len(label) + 4), text_y + 11], fill=color)
        draw.text((x1 + 2, text_y), label, fill=(0, 0, 0))


def render_detection_overlay(
    image_tensor: torch.Tensor,
    *,
    gt_boxes_xywh: torch.Tensor,
    gt_class_ids: torch.Tensor,
    prediction: DetectionPrediction,
    names: dict[int, str],
) -> torch.Tensor:
    """Render one CHW image with GT boxes in green and predictions in red."""

    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    pil_image = Image.fromarray(array)
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([0, 0, 84, 26], fill=(0, 0, 0))
    draw.text((3, 2), "GT", fill=(0, 255, 0))
    draw.text((3, 14), "Pred", fill=(255, 64, 64))

    gt_boxes_np = gt_boxes_xywh.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
    gt_classes_np = gt_class_ids.detach().cpu().numpy().astype(np.int32).reshape(-1)
    for box_xyxy, class_id in zip(xywh_to_xyxy_np(gt_boxes_np), gt_classes_np):
        _draw_normalized_box(
            draw,
            box_xyxy,
            image_size=pil_image.size,
            color=(0, 255, 0),
            label=f"gt {names.get(int(class_id), int(class_id))}",
        )
    for box_xyxy, score, class_id in zip(prediction.boxes_xyxy, prediction.scores, prediction.class_ids):
        _draw_normalized_box(
            draw,
            box_xyxy,
            image_size=pil_image.size,
            color=(255, 64, 64),
            label=f"{names.get(int(class_id), int(class_id))} {float(score):.2f}",
        )

    rendered = np.asarray(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(rendered).permute(2, 0, 1)


@torch.inference_mode()
def render_tensorboard_detection_batch(
    model: SimpleYOLODetector,
    dataset: SimpleYoloDataset,
    *,
    device: torch.device,
    names: dict[int, str],
    max_images: int,
    conf_threshold: float,
    nms_iou: float,
) -> torch.Tensor | None:
    """Render a small validation batch for TensorBoard sanity checks."""

    n_images = min(int(max_images), len(dataset))
    if n_images <= 0:
        return None
    model.eval()
    rendered: list[torch.Tensor] = []
    for index in range(n_images):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device=device)
        output = model(image)
        prediction = decode_simple_yolo_predictions(
            output,
            names=names,
            conf_threshold=float(conf_threshold),
            nms_iou=float(nms_iou),
        )[0]
        prediction = DetectionPrediction(
            image_id=str(sample["image_id"]),
            boxes_xyxy=prediction.boxes_xyxy,
            scores=prediction.scores,
            class_ids=prediction.class_ids,
        )
        rendered.append(
            render_detection_overlay(
                sample["image"],
                gt_boxes_xywh=sample["boxes_xywh"],
                gt_class_ids=sample["class_ids"],
                prediction=prediction,
                names=names,
            )
        )
    return torch.stack(rendered, dim=0)


def train_simple_yolo(
    cfg: Any,
    *,
    dataset_yaml: str,
    run_dir: Path,
    checkpoint_dir: Path,
    analysis_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train the native simple YOLO detector."""

    train_info = resolve_yolo_split_info(dataset_yaml, split="train")
    val_info = resolve_yolo_split_info(dataset_yaml, split="val")
    _validate_native_config(cfg, split_info=train_info)
    torch_device = torch.device(device)

    aug_cfg = getattr(cfg, "augment", None)
    general_aug_enabled = aug_cfg is not None and bool(aug_cfg.enabled)
    rare_slice_aug = str(cfg.baseline.mode) == "baseline_b"
    if general_aug_enabled and rare_slice_aug:
        raise ValueError(
            "augment.enabled and baseline.mode=baseline_b are mutually exclusive. "
            "Use one augmentation strategy per run."
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

    train_dataset = SimpleYoloDataset(
        train_info,
        image_size=int(cfg.data.image_size),
        input_channels=int(cfg.model.simple.input_channels),
        augment=general_aug_enabled or rare_slice_aug,
        prepared_baseline=prepared_baseline,
        baseline_cfg=cfg.baseline,
        aug_cfg=aug_cfg,
        cache_images=cache_images,
    )
    val_dataset = SimpleYoloDataset(
        val_info,
        image_size=int(cfg.data.image_size),
        input_channels=int(cfg.model.simple.input_channels),
        augment=False,
        cache_images=cache_images,
    )
    train_loader = _make_dataloader(
        train_dataset,
        cfg=cfg,
        shuffle=True,
        use_weighted_sampler=prepared_baseline is not None and bool(cfg.baseline.use_weighted_sampler),
    )
    val_loader = _make_dataloader(val_dataset, cfg=cfg, shuffle=False, use_weighted_sampler=False)
    model = build_simple_yolo_model(cfg, nc=train_info.nc).to(torch_device)
    _load_initial_weights_if_present(model, cfg.model.weights, device=torch_device)
    criterion = SimpleYoloLoss(cfg.loss)
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
            "backend": "simple_torch",
            "dataset_yaml": dataset_yaml,
            "baseline_mode": cfg.baseline.mode,
            "baseline_artifact_paths": baseline_artifact_paths,
            "stages": [{"stage_index": 1, "stage_name": "native_simple_yolo", "epochs": int(cfg.training.epochs)}],
        },
    )

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
                boxes_xywh = [tensor.to(torch_device) for tensor in batch["boxes_xywh"]]
                class_ids = [tensor.to(torch_device) for tensor in batch["class_ids"]]
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(precision):
                    output = model(images)
                    loss, loss_parts = criterion(output, boxes_xywh=boxes_xywh, class_ids=class_ids)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if cfg.training.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.training.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip_norm))
                    optimizer.step()
                for key, value in loss_parts.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                n_batches += 1
            if scheduler is not None:
                scheduler.step()

            row = {
                "epoch": epoch_idx,
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{key: value / max(n_batches, 1) for key, value in totals.items()},
            }
            writer.add_scalar("train/lr", float(row["lr"]), epoch_idx)
            for key, value in row.items():
                if key not in {"epoch", "lr"} and isinstance(value, (int, float)):
                    writer.add_scalar(f"train/{key}", float(value), epoch_idx)

            should_eval = epoch_idx == int(cfg.training.epochs) or epoch_idx % int(cfg.training.val_interval) == 0
            if should_eval:
                eval_payload = evaluate_simple_yolo_model(
                    model,
                    val_loader,
                    device=torch_device,
                    names=train_info.names,
                    conf_threshold=0.001,
                    nms_iou=float(cfg.evaluation.iou) if cfg.evaluation.iou is not None else 0.7,
                )
                last_metrics = dict(eval_payload["summary"])
                row.update({f"val_{key}": value for key, value in last_metrics.items() if isinstance(value, (int, float))})
                for key, value in last_metrics.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        writer.add_scalar(f"val/{key}", float(value), epoch_idx)
                image_interval = int(cfg.training.tensorboard_image_interval)
                if image_interval > 0 and epoch_idx % image_interval == 0:
                    image_batch = render_tensorboard_detection_batch(
                        model,
                        val_dataset,
                        device=torch_device,
                        names=train_info.names,
                        max_images=int(cfg.training.tensorboard_max_images),
                        conf_threshold=float(cfg.training.tensorboard_prediction_conf),
                        nms_iou=float(cfg.evaluation.iou) if cfg.evaluation.iou is not None else 0.7,
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
                    _save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        split_info=train_info,
                        epoch=epoch_idx,
                        metrics=last_metrics,
                    )
                else:
                    bad_epochs += 1
            else:
                bad_epochs += 1
            history.append(row)
            _log_epoch_progress(row, total_epochs=int(cfg.training.epochs))
            _save_checkpoint(
                checkpoint_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                split_info=train_info,
                epoch=epoch_idx,
                metrics=last_metrics,
            )
            writer.flush()
            if int(cfg.training.patience) > 0 and bad_epochs >= int(cfg.training.patience):
                break
    finally:
        writer.close()

    if not (checkpoint_dir / "best.pt").exists():
        _save_checkpoint(
            checkpoint_dir / "best.pt",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            split_info=train_info,
            epoch=history[-1]["epoch"],
            metrics=last_metrics,
        )

    pd.DataFrame(history).to_csv(analysis_dir / "loss_history.csv", index=False)
    summary = {
        "experiment_name": cfg.output.experiment_name,
        "backend": "simple_torch",
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
                "stage_name": "native_simple_yolo",
                "epochs": int(history[-1]["epoch"]) if history else 0,
                "best_weights_path": str((checkpoint_dir / "best.pt").resolve()),
                "last_weights_path": str((checkpoint_dir / "last.pt").resolve()),
            }
        ],
    }
    _write_json(analysis_dir / "train_summary.json", summary)
    pd.DataFrame([summary]).to_csv(analysis_dir / "train_summary.csv", index=False)
    return summary


def eval_simple_yolo(
    cfg: Any,
    *,
    weights_path: str,
    data_yaml: str,
    analysis_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Evaluate a native simple YOLO checkpoint."""

    split_info = resolve_yolo_split_info(data_yaml, split=str(cfg.evaluation.split))
    torch_device = torch.device(device)
    model, payload = load_simple_yolo_checkpoint(weights_path, map_location=torch_device)
    if int(payload.get("nc", -1)) != split_info.nc:
        raise ValueError(
            f"Checkpoint nc={payload.get('nc')} does not match dataset nc={split_info.nc} for {data_yaml}."
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
    nms_iou = float(cfg.evaluation.iou) if cfg.evaluation.iou is not None else 0.7
    eval_payload = evaluate_simple_yolo_model(
        model,
        dataloader,
        device=torch_device,
        names=split_info.names,
        conf_threshold=conf_threshold,
        nms_iou=nms_iou,
    )
    summary = {
        "experiment_name": cfg.output.experiment_name,
        "backend": "simple_torch",
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


def eval_simple_yolo_slices(
    cfg: Any,
    *,
    weights_path: str,
    data_yaml: str,
    threshold_yaml: str,
    analysis_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Run only native per-slice evaluation."""

    from src.analysis.flir_subgroup.yolo_slice_eval import (
        compute_frozen_thresholds,
        evaluate_per_slice_from_predictions,
    )

    eval_payload = eval_simple_yolo(
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
        "backend": "simple_torch",
        "weights_path": weights_path,
        "dataset_yaml": data_yaml,
        "per_slice_metrics_csv": str(analysis_dir / "per_slice_metrics.csv"),
        "per_slice_metrics_json": str(analysis_dir / "per_slice_metrics.json"),
        "per_slice_overall": per_slice_results.get("overall", {}),
    }
    _write_json(analysis_dir / "eval_slices_summary.json", summary)
    return summary
