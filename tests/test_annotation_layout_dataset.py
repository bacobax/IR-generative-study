"""Tests for the bbox-output AnnotationLayoutDataset."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.data.datasets import AnnotationLayoutDataset
from src.core.data.transforms import ScheduledHorizontalFlip
from src.core.normalization import UINT8_LINEAR


def _make_layout_test_data(tmpdir: str) -> tuple[str, str]:
    img_dir = os.path.join(tmpdir, "images")
    os.makedirs(img_dir, exist_ok=True)

    np.save(
        os.path.join(img_dir, "annotated.npy"),
        np.random.randint(0, 255, size=(50, 100), dtype=np.uint8),
    )
    np.save(
        os.path.join(img_dir, "empty.npy"),
        np.random.randint(0, 255, size=(50, 100), dtype=np.uint8),
    )

    payload = {
        "images": [
            {
                "id": "img-annotated",
                "file_name": "annotated.npy",
                "width": 100,
                "height": 50,
            },
            {
                "id": "img-empty",
                "file_name": "empty.npy",
                "width": 100,
                "height": 50,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": "img-annotated",
                "category_id": 0,
                "bbox": [10, 5, 20, 10],
                "area": 200,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": "img-annotated",
                "category_id": 2,
                "bbox": [50, 10, 25, 20],
                "area": 500,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 2, "name": "car"},
        ],
    }

    annotations_path = os.path.join(tmpdir, "annotations.json")
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return img_dir, annotations_path


def test_annotation_layout_dataset_returns_aligned_boxes_and_labels() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annotations_path = _make_layout_test_data(tmpdir)
        ds = AnnotationLayoutDataset(
            root_dir=img_dir,
            annotations_path=annotations_path,
            image_size=200,
            normalization_mode=UINT8_LINEAR,
            include_label_names=True,
        )

        sample = ds[0]
        assert isinstance(sample["pixel_values"], torch.Tensor)
        assert sample["pixel_values"].shape == (1, 200, 200)
        assert sample["image_id"] == "img-annotated"
        assert sample["file_name"] == "annotated.npy"
        assert sample["n_objects"] == 2
        assert sample["boxes_xyxy"].shape == (2, 4)
        assert sample["labels"].shape == (2,)
        assert sample["label_names"] == ["person", "car"]
        assert sample["labels"].tolist() == [0, 2]
        assert sample["boxes_xyxy_original"].tolist() == [
            [10.0, 5.0, 30.0, 15.0],
            [50.0, 10.0, 75.0, 30.0],
        ]
        assert sample["boxes_xyxy"].tolist() == [
            [20.0, 20.0, 60.0, 60.0],
            [100.0, 40.0, 150.0, 120.0],
        ]


def test_annotation_layout_dataset_handles_empty_annotations() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annotations_path = _make_layout_test_data(tmpdir)
        ds = AnnotationLayoutDataset(
            root_dir=img_dir,
            annotations_path=annotations_path,
            image_size=128,
            normalization_mode=UINT8_LINEAR,
            include_label_names=True,
        )

        sample = ds[1]
        assert sample["image_id"] == "img-empty"
        assert sample["n_objects"] == 0
        assert sample["boxes_xyxy"].shape == (0, 4)
        assert sample["boxes_xyxy_original"].shape == (0, 4)
        assert sample["labels"].shape == (0,)
        assert sample["label_names"] == []


def test_annotation_layout_dataset_uses_subset_manifest_order(tmp_path) -> None:
    img_dir, annotations_path = _make_layout_test_data(str(tmp_path))
    manifest = tmp_path / "subsets" / "train_layout.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"path": "empty.npy"},
                    {"path": "annotated.npy"},
                ]
            }
        ),
        encoding="utf-8",
    )

    ds = AnnotationLayoutDataset(
        root_dir=img_dir,
        annotations_path=annotations_path,
        image_size=128,
        normalization_mode=UINT8_LINEAR,
        subset_manifest=str(manifest),
    )

    assert ds.files == ["empty.npy", "annotated.npy"]
    assert ds[0]["file_name"] == "empty.npy"


def test_annotation_layout_dataset_horizontal_flip_mirrors_boxes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annotations_path = _make_layout_test_data(tmpdir)
        flip_schedule = ScheduledHorizontalFlip(
            total_epochs=1,
            p_hflip_warmup=1.0,
            p_hflip_max=1.0,
            p_hflip_final=1.0,
        )
        ds = AnnotationLayoutDataset(
            root_dir=img_dir,
            annotations_path=annotations_path,
            image_size=200,
            normalization_mode=UINT8_LINEAR,
            include_label_names=True,
            horizontal_flip_schedule=flip_schedule,
        )

        sample = ds[0]
        assert sample["horizontal_flipped"] is True
        assert sample["boxes_xyxy"].tolist() == [
            [140.0, 20.0, 180.0, 60.0],
            [50.0, 40.0, 100.0, 120.0],
        ]
