"""Foreground/background crop dataset utilities for FLIR filtering."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.data.annotations import (
    build_category_id_to_name,
    clip_box_to_image,
    coco_bbox_to_xyxy,
    index_annotations,
    load_coco_annotations,
)
from src.core.data.dataset_targets import resolve_dataset_target
from src.core.normalization import resize_and_normalize
from src.core.visualization.layout_debug import draw_bbox_overlays


_SPLIT_SEED_OFFSETS = {
    "train": 0,
    "val": 1_000_003,
    "test": 2_000_006,
}


@dataclass(frozen=True)
class CropSampleMetadata:
    """Metadata for one foreground/background crop."""

    split: str
    file_name: str
    image_id: Any
    label: int
    sample_kind: str
    crop_box_xyxy: Tuple[float, float, float, float]
    source_bbox_xyxy: Optional[Tuple[float, float, float, float]]
    source_area_ratio: float
    crop_area_ratio: float
    image_width: int
    image_height: int
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    is_background: bool = False
    model_label_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a JSON-friendly dict."""
        return {
            "split": self.split,
            "file_name": self.file_name,
            "image_id": self.image_id,
            "label": self.label,
            "sample_kind": self.sample_kind,
            "crop_box_xyxy": [float(v) for v in self.crop_box_xyxy],
            "source_bbox_xyxy": None
            if self.source_bbox_xyxy is None
            else [float(v) for v in self.source_bbox_xyxy],
            "source_area_ratio": float(self.source_area_ratio),
            "crop_area_ratio": float(self.crop_area_ratio),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "category_id": None if self.category_id is None else int(self.category_id),
            "category_name": self.category_name,
            "is_background": bool(self.is_background),
            "model_label_index": None if self.model_label_index is None else int(self.model_label_index),
        }


def _expand_box_with_context(
    box_xyxy: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    context_ratio: float,
) -> Tuple[float, float, float, float]:
    """Expand a box around its center then clip it to image bounds."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    new_w = max(1.0, width * float(context_ratio))
    new_h = max(1.0, height * float(context_ratio))
    expanded = (
        cx - 0.5 * new_w,
        cy - 0.5 * new_h,
        cx + 0.5 * new_w,
        cy + 0.5 * new_h,
    )
    return clip_box_to_image(expanded, image_width, image_height)


def _box_area(box_xyxy: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _max_iou_xyxy(
    candidate_box: Sequence[float],
    other_boxes_xyxy: np.ndarray,
) -> float:
    """Return the maximum IoU between one box and an array of boxes."""
    if other_boxes_xyxy.size == 0:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in candidate_box]
    inter_x1 = np.maximum(x1, other_boxes_xyxy[:, 0])
    inter_y1 = np.maximum(y1, other_boxes_xyxy[:, 1])
    inter_x2 = np.minimum(x2, other_boxes_xyxy[:, 2])
    inter_y2 = np.minimum(y2, other_boxes_xyxy[:, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    candidate_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    other_areas = np.maximum(0.0, other_boxes_xyxy[:, 2] - other_boxes_xyxy[:, 0]) * np.maximum(
        0.0, other_boxes_xyxy[:, 3] - other_boxes_xyxy[:, 1]
    )
    union = np.maximum(candidate_area + other_areas - inter_area, 1e-6)
    return float((inter_area / union).max(initial=0.0))


def _build_negative_candidate_box(
    *,
    image_width: int,
    image_height: int,
    crop_width: float,
    crop_height: float,
    rng: random.Random,
) -> Optional[Tuple[float, float, float, float]]:
    crop_width = min(float(image_width), max(1.0, float(crop_width)))
    crop_height = min(float(image_height), max(1.0, float(crop_height)))
    if crop_width <= 0.0 or crop_height <= 0.0:
        return None
    max_x = max(0.0, float(image_width) - crop_width)
    max_y = max(0.0, float(image_height) - crop_height)
    x1 = rng.uniform(0.0, max_x) if max_x > 0.0 else 0.0
    y1 = rng.uniform(0.0, max_y) if max_y > 0.0 else 0.0
    x2 = min(float(image_width), x1 + crop_width)
    y2 = min(float(image_height), y1 + crop_height)
    return clip_box_to_image((x1, y1, x2, y2), image_width, image_height)


class ForegroundBackgroundCropDataset(Dataset):
    """Binary crop dataset built from real FLIR annotations."""

    def __init__(
        self,
        *,
        split: str,
        dataset_id: str = "flir_private_proxy_alignment_v18",
        dataset_root: str | Path | None = None,
        normalization_mode: str | None = None,
        input_size: int = 128,
        context_ratio: float = 1.25,
        negative_iou_threshold: float = 0.01,
        negative_max_retries: int = 64,
        seed: int = 0,
        max_samples: int = 0,
        context_preview_size: int = 192,
    ) -> None:
        self.split = str(split)
        self.input_size = int(input_size)
        self.context_ratio = float(context_ratio)
        self.negative_iou_threshold = float(negative_iou_threshold)
        self.negative_max_retries = int(negative_max_retries)
        self.context_preview_size = int(context_preview_size)

        target = resolve_dataset_target(dataset_id)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else target.root
        self.split_dir = self.dataset_root / self.split
        self.annotations_path = self.split_dir / "annotations.json"
        self.normalization_mode = str(normalization_mode or target.normalization_mode)

        if not self.annotations_path.is_file():
            raise FileNotFoundError(f"Missing annotations file: {self.annotations_path}")
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {self.split_dir}")

        coco = load_coco_annotations(self.annotations_path)
        self.category_id_to_name = build_category_id_to_name(coco)
        self.category_ids = sorted(self.category_id_to_name)
        self.images_by_id, self.anns_by_image_id, _ = index_annotations(coco)
        self.image_ids = list(self.images_by_id.keys())
        self._boxes_xyxy_by_image_id: Dict[Any, np.ndarray] = {}
        self._crop_size_pairs: List[Tuple[float, float]] = []
        self._positive_entries: List[CropSampleMetadata] = []
        self._negative_entries: List[CropSampleMetadata] = []

        split_seed = int(seed) + _SPLIT_SEED_OFFSETS.get(self.split, 0)
        self._rng = random.Random(split_seed)

        for image_id, image_info in self.images_by_id.items():
            anns = self.anns_by_image_id.get(image_id, [])
            boxes_xyxy = np.asarray(
                [coco_bbox_to_xyxy(ann["bbox"]) for ann in anns],
                dtype=np.float32,
            )
            if boxes_xyxy.size == 0:
                boxes_xyxy = boxes_xyxy.reshape(0, 4)
            self._boxes_xyxy_by_image_id[image_id] = boxes_xyxy

            width = int(image_info["width"])
            height = int(image_info["height"])
            for ann in anns:
                source_box = coco_bbox_to_xyxy(ann["bbox"])
                crop_box = _expand_box_with_context(
                    source_box,
                    image_width=width,
                    image_height=height,
                    context_ratio=self.context_ratio,
                )
                self._crop_size_pairs.append(
                    (
                        float(crop_box[2] - crop_box[0]),
                        float(crop_box[3] - crop_box[1]),
                    )
                )
                self._positive_entries.append(
                    CropSampleMetadata(
                        split=self.split,
                        file_name=str(image_info["file_name"]),
                        image_id=image_id,
                        label=1,
                        sample_kind="positive",
                        crop_box_xyxy=crop_box,
                        source_bbox_xyxy=tuple(float(v) for v in source_box),
                        source_area_ratio=_box_area(source_box) / float(width * height),
                        crop_area_ratio=_box_area(crop_box) / float(width * height),
                        image_width=width,
                        image_height=height,
                        category_id=None,
                        category_name=None,
                        is_background=False,
                        model_label_index=1,
                    )
                )

        if max_samples and max_samples > 0:
            self._positive_entries = self._positive_entries[: int(max_samples)]

        self._negative_entries = self._build_negative_entries(self._positive_entries)
        self.samples: List[CropSampleMetadata] = self._positive_entries + self._negative_entries

    def _build_negative_entries(
        self,
        positive_entries: Sequence[CropSampleMetadata],
    ) -> List[CropSampleMetadata]:
        negatives: List[CropSampleMetadata] = []
        if not positive_entries:
            return negatives

        image_ids = self.image_ids
        for positive in positive_entries:
            pos_crop = positive.crop_box_xyxy
            crop_width = float(pos_crop[2] - pos_crop[0])
            crop_height = float(pos_crop[3] - pos_crop[1])
            negative_metadata = self._sample_negative_metadata(
                crop_width=crop_width,
                crop_height=crop_height,
                primary_image_id=positive.image_id,
                image_ids=image_ids,
            )
            if negative_metadata is not None:
                negatives.append(negative_metadata)
        return negatives

    def _sample_negative_metadata(
        self,
        *,
        crop_width: float,
        crop_height: float,
        primary_image_id: Any,
        image_ids: Sequence[Any],
    ) -> Optional[CropSampleMetadata]:
        for attempt_idx in range(max(1, self.negative_max_retries)):
            image_id = primary_image_id if attempt_idx == 0 else self._rng.choice(image_ids)
            image_info = self.images_by_id[image_id]
            width = int(image_info["width"])
            height = int(image_info["height"])
            candidate = _build_negative_candidate_box(
                image_width=width,
                image_height=height,
                crop_width=crop_width,
                crop_height=crop_height,
                rng=self._rng,
            )
            if candidate is None:
                continue
            max_iou = _max_iou_xyxy(candidate, self._boxes_xyxy_by_image_id[image_id])
            if max_iou > self.negative_iou_threshold:
                continue
            crop_area_ratio = _box_area(candidate) / float(width * height)
            return CropSampleMetadata(
                split=self.split,
                file_name=str(image_info["file_name"]),
                image_id=image_id,
                label=0,
                sample_kind="negative",
                crop_box_xyxy=candidate,
                source_bbox_xyxy=None,
                source_area_ratio=crop_area_ratio,
                crop_area_ratio=crop_area_ratio,
                image_width=width,
                image_height=height,
                category_id=None,
                category_name="background",
                is_background=True,
                model_label_index=0,
            )
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        metadata = self.samples[index]
        image_path = self.split_dir / metadata.file_name
        image = np.load(image_path)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image at {image_path}, got shape={image.shape}")
        x1, y1, x2, y2 = [int(round(v)) for v in metadata.crop_box_xyxy]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Empty crop for {image_path} at {metadata.crop_box_xyxy}")
        crop_tensor = torch.from_numpy(crop[None, ...])
        crop_tensor = resize_and_normalize(
            crop_tensor,
            image_size=self.input_size,
            normalization_mode=self.normalization_mode,
        )
        return {
            "pixel_values": crop_tensor,
            "label": torch.tensor(float(metadata.label), dtype=torch.float32),
            "metadata": metadata.to_dict(),
        }

    def stats(self) -> Dict[str, Any]:
        """Return dataset statistics for logging and summaries."""
        return {
            "split": self.split,
            "dataset_root": str(self.dataset_root),
            "split_dir": str(self.split_dir),
            "annotations_path": str(self.annotations_path),
            "normalization_mode": self.normalization_mode,
            "input_size": self.input_size,
            "context_ratio": self.context_ratio,
            "negative_iou_threshold": self.negative_iou_threshold,
            "negative_max_retries": self.negative_max_retries,
            "num_positive": len(self._positive_entries),
            "num_negative": len(self._negative_entries),
            "num_total": len(self.samples),
            "category_id_to_name": {int(k): str(v) for k, v in self.category_id_to_name.items()},
        }

    def render_context_previews(
        self,
        metadata_batch: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        """Render full-frame previews with crop rectangles overlaid."""
        if not metadata_batch:
            return torch.zeros(0, 3, self.context_preview_size, self.context_preview_size, dtype=torch.float32)

        images: List[torch.Tensor] = []
        boxes: List[List[float]] = []
        labels: List[int] = []
        for metadata in metadata_batch:
            image_path = self.split_dir / str(metadata["file_name"])
            image = np.load(image_path)
            if image.ndim == 3 and image.shape[-1] == 1:
                image = image[..., 0]
            image_tensor = torch.from_numpy(image[None, ...])
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0).to(torch.float32),
                size=(self.context_preview_size, self.context_preview_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            if self.normalization_mode == "uint8_linear":
                image_tensor = (image_tensor / 255.0).clamp(0.0, 1.0)
            else:
                image_tensor = ((image_tensor - image_tensor.min()) / max(float(image_tensor.max() - image_tensor.min()), 1e-6)).clamp(0.0, 1.0)
            images.append(image_tensor.repeat(3, 1, 1))

            x1, y1, x2, y2 = [float(v) for v in metadata["crop_box_xyxy"]]
            scale_x = float(self.context_preview_size) / float(metadata["image_width"])
            scale_y = float(self.context_preview_size) / float(metadata["image_height"])
            boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
            labels.append(int(metadata["label"]))

        image_batch = torch.stack(images, dim=0)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).unsqueeze(1)
        labels_tensor = torch.tensor(labels, dtype=torch.long).unsqueeze(1)
        object_mask = torch.ones(len(metadata_batch), 1, dtype=torch.bool)
        return draw_bbox_overlays(
            image_batch,
            boxes_xyxy=boxes_tensor,
            object_mask=object_mask,
            labels=labels_tensor,
            line_width=2,
        )


def collate_foreground_background_batch(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate crop tensors and preserve metadata lists."""
    items = list(batch)
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in items], dim=0),
        "label": torch.stack([item["label"] for item in items], dim=0),
        "metadata": [item["metadata"] for item in items],
    }


class MultiClassCropDataset(Dataset):
    """Multiclass crop dataset with all foreground classes plus background."""

    def __init__(
        self,
        *,
        split: str,
        dataset_id: str = "flir_private_proxy_alignment_v18",
        dataset_root: str | Path | None = None,
        normalization_mode: str | None = None,
        input_size: int = 128,
        context_ratio: float = 1.25,
        negative_iou_threshold: float = 0.01,
        negative_max_retries: int = 64,
        seed: int = 0,
        max_samples: int = 0,
        context_preview_size: int = 192,
    ) -> None:
        self.split = str(split)
        self.input_size = int(input_size)
        self.context_ratio = float(context_ratio)
        self.negative_iou_threshold = float(negative_iou_threshold)
        self.negative_max_retries = int(negative_max_retries)
        self.context_preview_size = int(context_preview_size)

        target = resolve_dataset_target(dataset_id)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else target.root
        self.split_dir = self.dataset_root / self.split
        self.annotations_path = self.split_dir / "annotations.json"
        self.normalization_mode = str(normalization_mode or target.normalization_mode)

        if not self.annotations_path.is_file():
            raise FileNotFoundError(f"Missing annotations file: {self.annotations_path}")
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {self.split_dir}")

        coco = load_coco_annotations(self.annotations_path)
        self.category_id_to_name = build_category_id_to_name(coco)
        self.category_ids = sorted(self.category_id_to_name)
        self.category_id_to_model_index = {
            int(category_id): idx for idx, category_id in enumerate(self.category_ids)
        }
        self.model_index_to_category_id = {
            idx: int(category_id) for category_id, idx in self.category_id_to_model_index.items()
        }
        self.background_class_index = len(self.category_ids)
        self.num_classes = self.background_class_index + 1

        self.images_by_id, self.anns_by_image_id, _ = index_annotations(coco)
        self.image_ids = list(self.images_by_id.keys())
        self._boxes_xyxy_by_image_id: Dict[Any, np.ndarray] = {}
        self._positive_entries: List[CropSampleMetadata] = []
        self._negative_entries: List[CropSampleMetadata] = []

        split_seed = int(seed) + _SPLIT_SEED_OFFSETS.get(self.split, 0)
        self._rng = random.Random(split_seed)

        for image_id, image_info in self.images_by_id.items():
            anns = self.anns_by_image_id.get(image_id, [])
            boxes_xyxy = np.asarray(
                [coco_bbox_to_xyxy(ann["bbox"]) for ann in anns],
                dtype=np.float32,
            )
            if boxes_xyxy.size == 0:
                boxes_xyxy = boxes_xyxy.reshape(0, 4)
            self._boxes_xyxy_by_image_id[image_id] = boxes_xyxy

            width = int(image_info["width"])
            height = int(image_info["height"])
            for ann in anns:
                category_id = int(ann["category_id"])
                category_name = str(self.category_id_to_name.get(category_id, category_id))
                source_box = coco_bbox_to_xyxy(ann["bbox"])
                crop_box = _expand_box_with_context(
                    source_box,
                    image_width=width,
                    image_height=height,
                    context_ratio=self.context_ratio,
                )
                self._positive_entries.append(
                    CropSampleMetadata(
                        split=self.split,
                        file_name=str(image_info["file_name"]),
                        image_id=image_id,
                        label=category_id,
                        sample_kind="positive",
                        crop_box_xyxy=crop_box,
                        source_bbox_xyxy=tuple(float(v) for v in source_box),
                        source_area_ratio=_box_area(source_box) / float(width * height),
                        crop_area_ratio=_box_area(crop_box) / float(width * height),
                        image_width=width,
                        image_height=height,
                        category_id=category_id,
                        category_name=category_name,
                        is_background=False,
                        model_label_index=self.category_id_to_model_index[category_id],
                    )
                )

        if max_samples and max_samples > 0:
            self._positive_entries = self._positive_entries[: int(max_samples)]

        self._negative_entries = self._build_negative_entries(self._positive_entries)
        self.samples: List[CropSampleMetadata] = self._positive_entries + self._negative_entries

    def _build_negative_entries(
        self,
        positive_entries: Sequence[CropSampleMetadata],
    ) -> List[CropSampleMetadata]:
        negatives: List[CropSampleMetadata] = []
        if not positive_entries:
            return negatives

        for positive in positive_entries:
            pos_crop = positive.crop_box_xyxy
            crop_width = float(pos_crop[2] - pos_crop[0])
            crop_height = float(pos_crop[3] - pos_crop[1])
            negative_metadata = self._sample_negative_metadata(
                crop_width=crop_width,
                crop_height=crop_height,
                primary_image_id=positive.image_id,
                image_ids=self.image_ids,
            )
            if negative_metadata is not None:
                negatives.append(negative_metadata)
        return negatives

    def _sample_negative_metadata(
        self,
        *,
        crop_width: float,
        crop_height: float,
        primary_image_id: Any,
        image_ids: Sequence[Any],
    ) -> Optional[CropSampleMetadata]:
        for attempt_idx in range(max(1, self.negative_max_retries)):
            image_id = primary_image_id if attempt_idx == 0 else self._rng.choice(image_ids)
            image_info = self.images_by_id[image_id]
            width = int(image_info["width"])
            height = int(image_info["height"])
            candidate = _build_negative_candidate_box(
                image_width=width,
                image_height=height,
                crop_width=crop_width,
                crop_height=crop_height,
                rng=self._rng,
            )
            if candidate is None:
                continue
            max_iou = _max_iou_xyxy(candidate, self._boxes_xyxy_by_image_id[image_id])
            if max_iou > self.negative_iou_threshold:
                continue
            crop_area_ratio = _box_area(candidate) / float(width * height)
            return CropSampleMetadata(
                split=self.split,
                file_name=str(image_info["file_name"]),
                image_id=image_id,
                label=self.background_class_index,
                sample_kind="negative",
                crop_box_xyxy=candidate,
                source_bbox_xyxy=None,
                source_area_ratio=crop_area_ratio,
                crop_area_ratio=crop_area_ratio,
                image_width=width,
                image_height=height,
                category_id=None,
                category_name="background",
                is_background=True,
                model_label_index=self.background_class_index,
            )
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        metadata = self.samples[index]
        image_path = self.split_dir / metadata.file_name
        image = np.load(image_path)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image at {image_path}, got shape={image.shape}")
        x1, y1, x2, y2 = [int(round(v)) for v in metadata.crop_box_xyxy]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Empty crop for {image_path} at {metadata.crop_box_xyxy}")
        crop_tensor = torch.from_numpy(crop[None, ...])
        crop_tensor = resize_and_normalize(
            crop_tensor,
            image_size=self.input_size,
            normalization_mode=self.normalization_mode,
        )
        return {
            "pixel_values": crop_tensor,
            "label": torch.tensor(int(metadata.model_label_index), dtype=torch.long),
            "metadata": metadata.to_dict(),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "dataset_root": str(self.dataset_root),
            "split_dir": str(self.split_dir),
            "annotations_path": str(self.annotations_path),
            "normalization_mode": self.normalization_mode,
            "input_size": self.input_size,
            "context_ratio": self.context_ratio,
            "negative_iou_threshold": self.negative_iou_threshold,
            "negative_max_retries": self.negative_max_retries,
            "num_positive": len(self._positive_entries),
            "num_negative": len(self._negative_entries),
            "num_total": len(self.samples),
            "num_foreground_classes": len(self.category_ids),
            "background_class_index": int(self.background_class_index),
            "category_id_to_name": {int(k): str(v) for k, v in self.category_id_to_name.items()},
            "category_id_to_model_index": {int(k): int(v) for k, v in self.category_id_to_model_index.items()},
            "model_index_to_category_id": {int(k): int(v) for k, v in self.model_index_to_category_id.items()},
        }

    def render_context_previews(
        self,
        metadata_batch: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        helper = ForegroundBackgroundCropDataset(
            split=self.split,
            dataset_root=self.dataset_root,
            normalization_mode=self.normalization_mode,
            input_size=self.input_size,
            context_ratio=self.context_ratio,
            negative_iou_threshold=self.negative_iou_threshold,
            negative_max_retries=self.negative_max_retries,
            seed=0,
            max_samples=0,
            context_preview_size=self.context_preview_size,
        )
        return helper.render_context_previews(metadata_batch)


def build_balanced_sample_weights(
    dataset: MultiClassCropDataset,
) -> torch.Tensor:
    counts: Dict[int, int] = {}
    for sample in dataset.samples:
        key = int(sample.model_label_index)
        counts[key] = counts.get(key, 0) + 1
    weights = []
    for sample in dataset.samples:
        key = int(sample.model_label_index)
        weights.append(1.0 / max(1, counts[key]))
    return torch.tensor(weights, dtype=torch.float32)
