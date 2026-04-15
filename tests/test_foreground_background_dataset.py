"""Tests for the FLIR foreground/background crop dataset."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.data.foreground_background_dataset import (
    ForegroundBackgroundCropDataset,
    _max_iou_xyxy,
)


def _write_split(split_dir: Path, *, split_name: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    annotation_id = 1
    for image_index in range(3):
        image_id = f"{split_name}-img-{image_index}"
        file_name = f"{image_id}.npy"
        width = 64
        height = 64
        image = np.full((height, width), 20 + 15 * image_index, dtype=np.uint8)
        image[6:18, 6:18] = 240
        image[40:54, 40:56] = 180
        np.save(split_dir / file_name, image)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [6, 6, 12, 12],
                "area": 144,
                "iscrowd": 0,
            }
        )
        annotation_id += 1
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 2,
                "bbox": [40, 40, 16, 14],
                "area": 224,
                "iscrowd": 0,
            }
        )
        annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "car"},
        ],
    }
    (split_dir / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "flir_like"
    for split in ("train", "val", "test"):
        _write_split(root / split, split_name=split)
    return root


def test_foreground_background_dataset_builds_bounded_positive_and_negative_crops(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path)
    dataset = ForegroundBackgroundCropDataset(
        split="train",
        dataset_root=root,
        input_size=32,
        seed=7,
        negative_max_retries=32,
    )

    assert dataset.annotations_path == root / "train" / "annotations.json"
    assert dataset.stats()["num_positive"] == 6
    assert dataset.stats()["num_negative"] <= dataset.stats()["num_positive"]

    sample = dataset[0]
    assert sample["pixel_values"].shape == (1, 32, 32)

    for metadata in dataset.samples:
        x1, y1, x2, y2 = metadata.crop_box_xyxy
        assert 0.0 <= x1 < x2 <= metadata.image_width
        assert 0.0 <= y1 < y2 <= metadata.image_height
        if metadata.label == 0:
            boxes = dataset._boxes_xyxy_by_image_id[metadata.image_id]
            assert _max_iou_xyxy(metadata.crop_box_xyxy, boxes) <= dataset.negative_iou_threshold + 1e-9


def test_foreground_background_dataset_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path)
    dataset_a = ForegroundBackgroundCropDataset(split="val", dataset_root=root, input_size=32, seed=11)
    dataset_b = ForegroundBackgroundCropDataset(split="val", dataset_root=root, input_size=32, seed=11)
    dataset_c = ForegroundBackgroundCropDataset(split="test", dataset_root=root, input_size=32, seed=11)

    metadata_a = [sample.to_dict() for sample in dataset_a.samples]
    metadata_b = [sample.to_dict() for sample in dataset_b.samples]

    assert metadata_a == metadata_b
    assert dataset_a.annotations_path == root / "val" / "annotations.json"
    assert dataset_c.annotations_path == root / "test" / "annotations.json"

