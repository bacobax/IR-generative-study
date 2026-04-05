"""Rare-slice sampling and targeted geometry helpers for YOLO baselines."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from src.analysis.flir_subgroup.yolo_slice_stats import (
    POSITION_BIN_ORDER,
    SIZE_BIN_ORDER,
    build_slice_counts_table,
    load_yolo_slice_dataset,
)


_DATALOADER_SEED = 6148914691236517205


@dataclass(frozen=True)
class PreparedYoloSliceBaseline:
    """Precomputed slice statistics and sampling metadata for one run."""

    mode: str
    dataset_yaml: str
    slice_counts_df: pd.DataFrame
    image_sampling_df: pd.DataFrame
    slice_summary: dict[str, Any]
    sampling_summary: dict[str, Any]
    image_metadata_by_stem: dict[str, dict[str, Any]]

    @property
    def targeted_image_score_threshold(self) -> float:
        return float(self.sampling_summary.get("targeted_image_score_threshold", 0.0))


def prepare_yolo_slice_baseline(
    *,
    dataset_yaml: str,
    analysis_dir: Path,
    baseline_cfg: Any,
) -> PreparedYoloSliceBaseline:
    """Compute slice rarity tables and persist baseline analysis artifacts."""

    dataset = load_yolo_slice_dataset(dataset_yaml)
    class_labels = [dataset.names[idx] for idx in sorted(dataset.names)]
    slice_counts_df = build_slice_counts_table(dataset.instance_df, class_labels)
    slice_counts_df["rarity"] = _rarity_from_counts(
        slice_counts_df["count"].astype(float).to_numpy(),
        alpha=float(baseline_cfg.rarity_alpha),
        eps=float(baseline_cfg.rarity_eps),
    )
    rarity_map = {
        (str(row.class_label), str(row.size_bin), str(row.position_bin)): float(row.rarity)
        for row in slice_counts_df.itertuples(index=False)
    }

    image_sampling_df = _build_image_sampling_table(
        image_df=dataset.image_df,
        instance_df=dataset.instance_df,
        rarity_map=rarity_map,
        top_k=int(baseline_cfg.image_score_top_k),
        eps=float(baseline_cfg.rarity_eps),
        normalize_weights=bool(baseline_cfg.normalize_weights),
        clip_weight_min=baseline_cfg.clip_weight_min,
        clip_weight_max=baseline_cfg.clip_weight_max,
        targeted_aug_rarity_quantile=float(baseline_cfg.targeted_aug_rarity_quantile),
    )
    image_metadata_by_stem = {
        str(row.image_stem): {
            "image_score": float(row.image_score),
            "sampling_weight": float(row.sampling_weight),
            "n_instances": int(row.n_instances),
            "protected_instance_indices": [
                int(idx) for idx in str(row.protected_instance_indices).split("|") if idx != ""
            ],
        }
        for row in image_sampling_df.itertuples(index=False)
    }

    slice_summary = {
        "baseline_mode": baseline_cfg.mode,
        "dataset_yaml": str(Path(dataset_yaml).resolve()),
        "train_image_dir": dataset.train_image_dir,
        "train_label_dir": dataset.train_label_dir,
        "n_images": int(len(dataset.image_df)),
        "n_instances": int(len(dataset.instance_df)),
        "n_classes": int(len(class_labels)),
        "classes": class_labels,
        "size_bins": list(SIZE_BIN_ORDER),
        "position_bins": list(POSITION_BIN_ORDER),
        "size_thresholds": dataset.thresholds.to_dict(),
        "active_slice_count": int((slice_counts_df["count"] > 0).sum()),
        "total_possible_slices": int(len(slice_counts_df)),
        "rarity_alpha": float(baseline_cfg.rarity_alpha),
        "rarity_eps": float(baseline_cfg.rarity_eps),
        "image_score_top_k": int(baseline_cfg.image_score_top_k),
    }
    sampling_summary = {
        "baseline_mode": baseline_cfg.mode,
        "normalize_weights": bool(baseline_cfg.normalize_weights),
        "clip_weight_min": _optional_float(baseline_cfg.clip_weight_min),
        "clip_weight_max": _optional_float(baseline_cfg.clip_weight_max),
        "sampler_replacement": bool(baseline_cfg.sampler_replacement),
        "targeted_aug_probability": float(baseline_cfg.targeted_aug_probability),
        "targeted_aug_rarity_quantile": float(baseline_cfg.targeted_aug_rarity_quantile),
        "targeted_image_score_threshold": float(image_sampling_df["targeted_threshold"].iloc[0]),
        "weight_min": float(image_sampling_df["sampling_weight"].min()),
        "weight_max": float(image_sampling_df["sampling_weight"].max()),
        "weight_mean": float(image_sampling_df["sampling_weight"].mean()),
        "weight_std": float(image_sampling_df["sampling_weight"].std(ddof=0)),
        "zero_instance_images": int((image_sampling_df["n_instances"] == 0).sum()),
        "eligible_targeted_images": int(image_sampling_df["is_targeted_eligible"].sum()),
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    slice_counts_df.to_csv(analysis_dir / "slice_counts.csv", index=False)
    image_sampling_df.to_csv(analysis_dir / "image_sampling_weights.csv", index=False)
    (analysis_dir / "slice_summary.json").write_text(json.dumps(slice_summary, indent=2), encoding="utf-8")
    (analysis_dir / "sampling_weight_summary.json").write_text(
        json.dumps(sampling_summary, indent=2),
        encoding="utf-8",
    )

    return PreparedYoloSliceBaseline(
        mode=str(baseline_cfg.mode),
        dataset_yaml=str(Path(dataset_yaml).resolve()),
        slice_counts_df=slice_counts_df,
        image_sampling_df=image_sampling_df,
        slice_summary=slice_summary,
        sampling_summary=sampling_summary,
        image_metadata_by_stem=image_metadata_by_stem,
    )


def make_slice_aware_yolo_dataset_class(yolo_dataset_cls, compose_cls):
    """Return a YOLO dataset subclass with optional targeted rare-slice augmentation."""

    class SliceAwareYOLODataset(yolo_dataset_cls):
        def __init__(
            self,
            *args,
            baseline_mode: str = "none",
            prepared_baseline: PreparedYoloSliceBaseline | None = None,
            baseline_cfg: Any = None,
            **kwargs,
        ):
            self.baseline_mode = str(baseline_mode)
            self.prepared_baseline = prepared_baseline
            self.baseline_cfg = baseline_cfg
            self._image_metadata_by_stem = (
                prepared_baseline.image_metadata_by_stem if prepared_baseline is not None else {}
            )
            super().__init__(*args, **kwargs)
            self.slice_sampling_weights = np.asarray(
                [
                    float(self._image_metadata_by_stem.get(Path(im_file).stem, {}).get("sampling_weight", 1.0))
                    for im_file in self.im_files
                ],
                dtype=np.float64,
            )

        def build_transforms(self, hyp: dict | None = None):
            transforms = super().build_transforms(hyp)
            if self.augment and self.baseline_mode == "baseline_b" and self.prepared_baseline is not None:
                transforms = compose_cls(
                    [
                        RareSliceGeometryTransform(
                            image_metadata_by_stem=self._image_metadata_by_stem,
                            probability=float(self.baseline_cfg.targeted_aug_probability),
                            image_score_threshold=float(self.prepared_baseline.targeted_image_score_threshold),
                            translate_fraction=float(self.baseline_cfg.translate_fraction),
                            scale_min=float(self.baseline_cfg.scale_min),
                            scale_max=float(self.baseline_cfg.scale_max),
                            crop_scale_min=float(self.baseline_cfg.crop_scale_min),
                            crop_scale_max=float(self.baseline_cfg.crop_scale_max),
                            crop_min_rare_box_area_retained=float(self.baseline_cfg.crop_min_rare_box_area_retained),
                            crop_max_attempts=int(self.baseline_cfg.crop_max_attempts),
                            allow_horizontal_flip=bool(self.baseline_cfg.allow_horizontal_flip),
                        ),
                        *transforms.tolist(),
                    ]
                )
            return transforms

    return SliceAwareYOLODataset


def build_weighted_train_dataloader(
    *,
    dataset,
    batch_size: int,
    workers: int,
    replacement: bool,
    drop_last: bool,
    pin_memory: bool,
    infinite_dataloader_cls,
    seed_worker_fn: Callable[[int], None],
    rank: int,
    rank_seed_offset: int,
):
    """Build a train dataloader that uses per-image sampling weights."""

    if rank != -1:
        raise ValueError("Rare-slice weighted sampling currently supports single-process YOLO training only.")
    batch = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min(os.cpu_count() // max(nd, 1), workers)
    generator = torch.Generator()
    generator.manual_seed(_DATALOADER_SEED + int(rank_seed_offset))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(dataset.slice_sampling_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=bool(replacement),
        generator=generator,
    )
    return infinite_dataloader_cls(
        dataset=dataset,
        batch_size=batch,
        shuffle=False,
        sampler=sampler,
        num_workers=nw,
        prefetch_factor=4 if nw > 0 else None,
        pin_memory=nd > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker_fn,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )


def _build_image_sampling_table(
    *,
    image_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    rarity_map: dict[tuple[str, str, str], float],
    top_k: int,
    eps: float,
    normalize_weights: bool,
    clip_weight_min: float | None,
    clip_weight_max: float | None,
    targeted_aug_rarity_quantile: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_k = max(int(top_k), 1)
    grouped = (
        {}
        if instance_df.empty
        else {str(image_stem): group.copy() for image_stem, group in instance_df.groupby("image_stem", sort=False)}
    )

    for image_row in image_df.itertuples(index=False):
        image_stem = str(image_row.image_stem)
        group = grouped.get(image_stem)
        if group is None or group.empty:
            rows.append(
                {
                    "image_index": int(image_row.image_index),
                    "image_stem": image_stem,
                    "image_path": str(image_row.image_path),
                    "n_instances": 0,
                    "image_score": 0.0,
                    "protected_instance_indices": "",
                }
            )
            continue
        instance_scores = np.asarray(
            [
                float(rarity_map[(str(row.class_label), str(row.size_bin), str(row.position_bin))])
                for row in group.itertuples(index=False)
            ],
            dtype=np.float64,
        )
        sorted_indices = np.argsort(-instance_scores, kind="stable")
        protected_indices = sorted_indices[: min(len(sorted_indices), top_k)]
        rows.append(
            {
                "image_index": int(image_row.image_index),
                "image_stem": image_stem,
                "image_path": str(image_row.image_path),
                "n_instances": int(len(group)),
                "image_score": float(instance_scores[protected_indices].mean()) if len(protected_indices) else 0.0,
                "protected_instance_indices": "|".join(str(int(idx)) for idx in protected_indices.tolist()),
            }
        )

    image_sampling_df = pd.DataFrame(rows).sort_values("image_index").reset_index(drop=True)
    positive_scores = image_sampling_df.loc[image_sampling_df["image_score"] > 0, "image_score"]
    if positive_scores.empty:
        raw_weights = np.ones(len(image_sampling_df), dtype=np.float64)
        targeted_threshold = 0.0
    else:
        min_positive = max(float(positive_scores.min()) * 0.5, float(eps))
        raw_weights = np.where(
            image_sampling_df["image_score"].to_numpy(dtype=np.float64) > 0.0,
            image_sampling_df["image_score"].to_numpy(dtype=np.float64),
            min_positive,
        )
        targeted_threshold = float(
            np.quantile(
                positive_scores.to_numpy(dtype=np.float64),
                min(max(float(targeted_aug_rarity_quantile), 0.0), 1.0),
            )
        )

    sampling_weights = raw_weights.copy()
    if normalize_weights and sampling_weights.size > 0:
        sampling_weights = sampling_weights / max(float(sampling_weights.mean()), float(eps))
    if clip_weight_min is not None:
        sampling_weights = np.maximum(sampling_weights, float(clip_weight_min))
    if clip_weight_max is not None:
        sampling_weights = np.minimum(sampling_weights, float(clip_weight_max))

    image_sampling_df["sampling_weight"] = sampling_weights
    image_sampling_df["raw_sampling_weight"] = raw_weights
    image_sampling_df["targeted_threshold"] = targeted_threshold
    image_sampling_df["is_targeted_eligible"] = (
        (image_sampling_df["image_score"] >= targeted_threshold) & (image_sampling_df["image_score"] > 0)
    )
    return image_sampling_df


def _rarity_from_counts(counts: np.ndarray, *, alpha: float, eps: float) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    return np.power(values + float(eps), -float(alpha))


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


class RareSliceGeometryTransform:
    """Targeted geometry branch for rare-slice images."""

    def __init__(
        self,
        *,
        image_metadata_by_stem: dict[str, dict[str, Any]],
        probability: float,
        image_score_threshold: float,
        translate_fraction: float,
        scale_min: float,
        scale_max: float,
        crop_scale_min: float,
        crop_scale_max: float,
        crop_min_rare_box_area_retained: float,
        crop_max_attempts: int,
        allow_horizontal_flip: bool,
    ) -> None:
        self.image_metadata_by_stem = image_metadata_by_stem
        self.probability = probability
        self.image_score_threshold = image_score_threshold
        self.translate_fraction = translate_fraction
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.crop_scale_min = crop_scale_min
        self.crop_scale_max = crop_scale_max
        self.crop_min_rare_box_area_retained = crop_min_rare_box_area_retained
        self.crop_max_attempts = crop_max_attempts
        self.allow_horizontal_flip = allow_horizontal_flip

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        stem = Path(str(labels.get("im_file", ""))).stem
        metadata = self.image_metadata_by_stem.get(stem)
        if metadata is None:
            return labels
        if float(metadata.get("image_score", 0.0)) < self.image_score_threshold:
            return labels
        if random.random() > self.probability:
            return labels

        protected_indices = [int(idx) for idx in metadata.get("protected_instance_indices", [])]
        if not protected_indices:
            return labels

        original_img = labels["img"]
        original_instances = labels["instances"]
        original_cls = labels["cls"]
        original_instances.convert_bbox(format="xywh")
        original_boxes = original_instances.bboxes.astype(np.float32).copy()
        if len(original_boxes) == 0:
            return labels

        transformed_img = original_img
        transformed_boxes = original_boxes

        dx = random.uniform(-self.translate_fraction, self.translate_fraction)
        dy = random.uniform(-self.translate_fraction, self.translate_fraction)
        transformed_img, transformed_boxes = apply_translation(transformed_img, transformed_boxes, dx=dx, dy=dy)

        scale = random.uniform(self.scale_min, self.scale_max)
        transformed_img, transformed_boxes = apply_center_scale(transformed_img, transformed_boxes, scale=scale)

        crop_result = apply_constrained_crop_resize(
            transformed_img,
            transformed_boxes,
            protected_indices=protected_indices,
            crop_scale_min=self.crop_scale_min,
            crop_scale_max=self.crop_scale_max,
            min_retained_area=self.crop_min_rare_box_area_retained,
            max_attempts=self.crop_max_attempts,
        )
        if crop_result is not None:
            transformed_img, transformed_boxes = crop_result["img"], crop_result["boxes"]

        if self.allow_horizontal_flip and random.random() < 0.5:
            transformed_img, transformed_boxes = apply_horizontal_flip(transformed_img, transformed_boxes)

        keep_mask = valid_box_mask(transformed_boxes)
        if not any(idx < len(keep_mask) and bool(keep_mask[idx]) for idx in protected_indices):
            return labels
        if not bool(keep_mask.any()):
            return labels

        kept_instances = original_instances[keep_mask]
        kept_instances.update(bboxes=transformed_boxes[keep_mask])
        labels["img"] = np.ascontiguousarray(transformed_img)
        labels["instances"] = kept_instances
        labels["cls"] = original_cls[keep_mask]
        labels["resized_shape"] = transformed_img.shape[:2]
        return labels


def apply_translation(img: np.ndarray, boxes_xywh: np.ndarray, *, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Translate an image and normalized xywh boxes."""

    h, w = img.shape[:2]
    matrix = np.array([[1.0, 0.0, dx * w], [0.0, 1.0, dy * h]], dtype=np.float32)
    translated_img = cv2.warpAffine(img, matrix, (w, h), borderValue=_border_value(img))
    translated_boxes = boxes_xywh.astype(np.float32).copy()
    translated_boxes[:, 0] += dx
    translated_boxes[:, 1] += dy
    translated_boxes = clip_boxes_xywh(translated_boxes)
    return translated_img, translated_boxes


def apply_center_scale(img: np.ndarray, boxes_xywh: np.ndarray, *, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply centered scale jitter to an image and normalized xywh boxes."""

    h, w = img.shape[:2]
    scaled_img = _resize_centered(img, scale=scale)
    boxes_xyxy = xywh_to_xyxy(boxes_xywh.astype(np.float32))
    boxes_xyxy = 0.5 + (boxes_xyxy - 0.5) * float(scale)
    scaled_boxes = xyxy_to_xywh(clip_boxes_xyxy(boxes_xyxy))
    return scaled_img.reshape(h, w, *img.shape[2:]) if img.ndim > 2 else scaled_img, scaled_boxes


def apply_horizontal_flip(img: np.ndarray, boxes_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip an image and normalized xywh boxes horizontally."""

    flipped_img = np.ascontiguousarray(np.fliplr(img))
    flipped_boxes = boxes_xywh.astype(np.float32).copy()
    flipped_boxes[:, 0] = 1.0 - flipped_boxes[:, 0]
    return flipped_img, clip_boxes_xywh(flipped_boxes)


def apply_constrained_crop_resize(
    img: np.ndarray,
    boxes_xywh: np.ndarray,
    *,
    protected_indices: list[int],
    crop_scale_min: float,
    crop_scale_max: float,
    min_retained_area: float,
    max_attempts: int,
) -> dict[str, Any] | None:
    """Crop-resize while ensuring at least one protected box survives well enough."""

    if crop_scale_min >= 1.0 or crop_scale_max >= 1.0 and math.isclose(crop_scale_min, crop_scale_max):
        return None
    h, w = img.shape[:2]
    boxes_xyxy = xywh_to_xyxy(boxes_xywh.astype(np.float32))
    protected_indices = [idx for idx in protected_indices if 0 <= idx < len(boxes_xyxy)]
    if not protected_indices:
        return None

    for _ in range(max(1, int(max_attempts))):
        protected_idx = random.choice(protected_indices)
        scale = random.uniform(crop_scale_min, crop_scale_max)
        crop_w = min(max(float(scale), 1e-3), 1.0)
        crop_h = min(max(float(scale), 1e-3), 1.0)
        px, py = boxes_xywh[protected_idx, 0], boxes_xywh[protected_idx, 1]

        x0_min = max(0.0, px - crop_w)
        x0_max = min(px, 1.0 - crop_w)
        y0_min = max(0.0, py - crop_h)
        y0_max = min(py, 1.0 - crop_h)
        x0 = x0_min if x0_max <= x0_min else random.uniform(x0_min, x0_max)
        y0 = y0_min if y0_max <= y0_min else random.uniform(y0_min, y0_max)
        crop_box = np.array([x0, y0, x0 + crop_w, y0 + crop_h], dtype=np.float32)

        intersections = _intersection_boxes(boxes_xyxy, crop_box)
        original_area = np.maximum((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]), 1e-12)
        inter_area = np.maximum(
            (intersections[:, 2] - intersections[:, 0]) * (intersections[:, 3] - intersections[:, 1]),
            0.0,
        )
        retained_ratio = inter_area / original_area
        transformed_xyxy = intersections.copy()
        transformed_xyxy[:, [0, 2]] = (transformed_xyxy[:, [0, 2]] - x0) / crop_w
        transformed_xyxy[:, [1, 3]] = (transformed_xyxy[:, [1, 3]] - y0) / crop_h
        transformed_xyxy = clip_boxes_xyxy(transformed_xyxy)
        transformed_boxes = xyxy_to_xywh(transformed_xyxy)
        keep_mask = valid_box_mask(transformed_boxes)

        protected_ok = any(
            idx < len(keep_mask)
            and bool(keep_mask[idx])
            and float(retained_ratio[idx]) >= float(min_retained_area)
            for idx in protected_indices
        )
        if not protected_ok:
            continue

        x1_px = int(round(x0 * w))
        y1_px = int(round(y0 * h))
        x2_px = max(x1_px + 1, int(round((x0 + crop_w) * w)))
        y2_px = max(y1_px + 1, int(round((y0 + crop_h) * h)))
        cropped = img[y1_px:y2_px, x1_px:x2_px]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return {"img": resized, "boxes": transformed_boxes}
    return None


def valid_box_mask(boxes_xywh: np.ndarray, *, min_size: float = 1e-4) -> np.ndarray:
    """Return a mask of boxes with positive normalized width and height."""

    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    return (boxes[:, 2] > float(min_size)) & (boxes[:, 3] > float(min_size))


def clip_boxes_xywh(boxes_xywh: np.ndarray) -> np.ndarray:
    """Clip normalized xywh boxes to image bounds."""

    clipped_xyxy = clip_boxes_xyxy(xywh_to_xyxy(boxes_xywh))
    return xyxy_to_xywh(clipped_xyxy)


def clip_boxes_xyxy(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Clip normalized xyxy boxes to image bounds."""

    clipped = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0.0, 1.0)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0.0, 1.0)
    return clipped


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """Convert normalized xywh boxes to normalized xyxy boxes."""

    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Convert normalized xyxy boxes to normalized xywh boxes."""

    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    converted = boxes.copy()
    converted[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    converted[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    converted[:, 2] = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    converted[:, 3] = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    return converted


def _intersection_boxes(boxes_xyxy: np.ndarray, crop_box_xyxy: np.ndarray) -> np.ndarray:
    intersections = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    intersections[:, 0] = np.maximum(intersections[:, 0], crop_box_xyxy[0])
    intersections[:, 1] = np.maximum(intersections[:, 1], crop_box_xyxy[1])
    intersections[:, 2] = np.minimum(intersections[:, 2], crop_box_xyxy[2])
    intersections[:, 3] = np.minimum(intersections[:, 3], crop_box_xyxy[3])
    intersections[:, 2] = np.maximum(intersections[:, 2], intersections[:, 0])
    intersections[:, 3] = np.maximum(intersections[:, 3], intersections[:, 1])
    return intersections


def _resize_centered(img: np.ndarray, *, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    scaled_h = max(1, int(round(h * scale)))
    scaled_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, *img.shape[2:]), _border_value(img), dtype=img.dtype)

    if scale >= 1.0:
        y0 = max((scaled_h - h) // 2, 0)
        x0 = max((scaled_w - w) // 2, 0)
        cropped = resized[y0:y0 + h, x0:x0 + w]
        return cropped

    y0 = max((h - scaled_h) // 2, 0)
    x0 = max((w - scaled_w) // 2, 0)
    canvas[y0:y0 + scaled_h, x0:x0 + scaled_w] = resized
    return canvas


def _border_value(img: np.ndarray) -> tuple[int, ...] | int:
    if img.ndim == 2:
        return 114
    return tuple(114 for _ in range(img.shape[2]))
