#!/usr/bin/env python3
"""Sanity-check named dataset loading and save bbox overlay previews.

This script resolves a named dataset target, instantiates the real
``AnnotationLayoutDataset``, prints per-sample stats, and saves bbox overlays
for the first sample at both the original and resized resolutions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.configs.config_loader import apply_yaml_defaults
from src.core.data.datasets import AnnotationLayoutDataset
from src.core.data.dataset_targets import resolve_dataset_target, supported_dataset_ids
from src.core.normalization import (
    DEFAULT_IMAGE_SIZE,
    uint16_to_png_uint8,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview a named dataset target.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file. CLI values override config values.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="flir_private_proxy_alignment_v18",
        choices=sorted(supported_dataset_ids()),
        help="Named dataset target to inspect.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val", "test"),
        help="Which split to inspect.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square resize target used for the preview tensor.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of deterministic samples to print.",
    )
    return parser


def validate_image_size(image_size: int) -> None:
    if image_size <= 0 or image_size % 32 != 0:
        raise ValueError(
            f"image_size must be a positive multiple of 32, got {image_size}"
        )


def image_to_display_uint8(arr: np.ndarray, normalization_mode: str) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if normalization_mode == "uint8_linear":
        return arr.astype(np.uint8)
    if normalization_mode == "raw_uint16_percentile":
        return uint16_to_png_uint8(arr.astype(np.uint16))

    arr_float = arr.astype(np.float32)
    lo = float(arr_float.min())
    hi = float(arr_float.max())
    if hi <= lo:
        return np.zeros_like(arr_float, dtype=np.uint8)
    return ((arr_float - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


def draw_boxes(
    image_uint8: np.ndarray,
    boxes: list[tuple[float, float, float, float, str]],
) -> Image.Image:
    image = Image.fromarray(image_uint8).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, y, w, h, label in boxes:
        draw.rectangle((x, y, x + w, y + h), outline=(255, 64, 64), width=2)
        if label:
            text_y = max(0.0, y - 12.0)
            draw.text((x + 2.0, text_y), label, fill=(255, 220, 64))
    return image


def collate_preview(batch: list[dict]) -> list[dict]:
    return batch


def main() -> None:
    parser = build_parser()
    config_args, _ = parser.parse_known_args()
    apply_yaml_defaults(parser, config_args.config)
    args = parser.parse_args()
    validate_image_size(args.image_size)

    target = resolve_dataset_target(args.dataset_id)
    split_dir = target.split_dir(args.split)
    annotations_path = target.annotations_path(args.split)

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Missing annotations file: {annotations_path}")

    dataset = AnnotationLayoutDataset(
        root_dir=str(split_dir),
        annotations_path=str(annotations_path),
        image_size=args.image_size,
        normalization_mode=target.normalization_mode,
        include_label_names=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_preview,
    )

    print(f"dataset_id={target.dataset_id}")
    print(f"root={target.root}")
    print(f"split={args.split}")
    print(f"normalization_mode={target.normalization_mode}")
    print(f"dataset_len={len(dataset)}")
    print(f"image_size={args.image_size}")

    for idx, batch in enumerate(loader):
        sample = batch[0]
        raw = np.load(split_dir / sample["file_name"])
        resized = sample["pixel_values"]
        boxes_xyxy = sample["boxes_xyxy"]
        labels = sample["labels"]
        label_names = sample.get("label_names", [])
        assert "pixel_values" in sample, "Missing pixel_values"
        assert "boxes_xyxy" in sample, "Missing boxes_xyxy"
        assert "labels" in sample, "Missing labels"
        assert boxes_xyxy.shape[0] == labels.shape[0], "boxes_xyxy/labels length mismatch"
        print("")
        print(f"[sample {idx}] file={sample['file_name']}")
        print(
            f"  raw shape={tuple(raw.shape)} dtype={raw.dtype} "
            f"min={raw.min()} max={raw.max()}"
        )
        print(
            f"  resized_tensor shape={tuple(resized.shape)} dtype={resized.dtype} "
            f"min={float(resized.min()):.4f} max={float(resized.max()):.4f}"
        )
        print(f"  image_id={sample['image_id']}")
        print(f"  n_bboxes={sample['n_objects']}")
        category_counts = Counter(label_names)
        print(f"  category_counts={dict(sorted(category_counts.items()))}")
        for bbox_idx in range(min(5, boxes_xyxy.shape[0])):
            x1, y1, x2, y2 = boxes_xyxy[bbox_idx].tolist()
            label = label_names[bbox_idx] if bbox_idx < len(label_names) else str(int(labels[bbox_idx]))
            print(
                f"    bbox[{bbox_idx}] label={label!r} "
                f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )
        if idx + 1 >= args.num_samples:
            break

    first = dataset[0]
    out_dir = (
        REPO / "artifacts" / "debug" / "dataset_sanity" / args.dataset_id / args.split
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_first = np.load(split_dir / first["file_name"])
    original_display = image_to_display_uint8(raw_first, target.normalization_mode)
    original_boxes = []
    label_names = first.get("label_names", [])
    for bbox_idx, box in enumerate(first["boxes_xyxy_original"].tolist()):
        x1, y1, x2, y2 = box
        label = label_names[bbox_idx] if bbox_idx < len(label_names) else ""
        original_boxes.append((x1, y1, x2 - x1, y2 - y1, label))
    original_overlay = draw_boxes(original_display, original_boxes)
    original_path = out_dir / "first_sample_bboxes_original.png"
    original_overlay.save(original_path)

    resized_uint8 = (
        ((first["pixel_values"].detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0) * 255.0
    ).byte().numpy()
    if resized_uint8.ndim == 3 and resized_uint8.shape[0] == 1:
        resized_uint8 = resized_uint8[0]
    resized_boxes = []
    for bbox_idx, box in enumerate(first["boxes_xyxy"].tolist()):
        x1, y1, x2, y2 = box
        label = label_names[bbox_idx] if bbox_idx < len(label_names) else ""
        resized_boxes.append((x1, y1, x2 - x1, y2 - y1, label))
    resized_overlay = draw_boxes(resized_uint8, resized_boxes)
    resized_path = out_dir / f"first_sample_bboxes_{args.image_size}.png"
    resized_overlay.save(resized_path)

    print("")
    print(f"saved_original_overlay={original_path}")
    print(f"saved_resized_overlay={resized_path}")


if __name__ == "__main__":
    main()
