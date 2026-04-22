"""Smoke tests for the end-to-end FLIR private-proxy builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.standalone.build_flir_private_proxy_alignment_dataset import (
    align_eval_split_to_train_categories,
    export_split_to_v18,
)
from scripts.standalone.download_and_build_flir_full_thermal_dataset import (
    build_builder_command,
    find_flir_root,
    safe_extract_zip,
)


def _write_jpg(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((8, 10), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_export_split_to_v18_writes_expected_layout(tmp_path: Path) -> None:
    flir_root = tmp_path / "flir"
    split_dir = flir_root / "images_thermal_train" / "data"
    _write_jpg(split_dir / "video-a-frame-000001-tokenA.jpg", value=11)
    _write_jpg(split_dir / "video-a-frame-000002-tokenB.jpg", value=22)

    output_root = tmp_path / "out"
    summary = export_split_to_v18(
        split="train",
        image_records=[
            {
                "id": "train::1",
                "file_name": "data/video-a-frame-000001-tokenA.jpg",
                "width": 10,
                "height": 8,
                "source_split": "train",
            },
            {
                "id": "train::2",
                "file_name": "data/video-a-frame-000002-tokenB.jpg",
                "width": 10,
                "height": 8,
                "source_split": "train",
            },
        ],
        annotation_records=[
            {
                "id": "ann-1",
                "image_id": "train::1",
                "category_id": 1,
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "area": 12.0,
                "iscrowd": 0,
            },
            {
                "id": "ann-2",
                "image_id": "train::2",
                "category_id": 3,
                "bbox": [0.0, 1.0, 2.0, 3.0],
                "area": 6.0,
                "iscrowd": 0,
            },
        ],
        source_categories=[
            {"id": 1, "name": "person"},
            {"id": 2, "name": "bike"},
            {"id": 3, "name": "car"},
        ],
        flir_root=flir_root,
        output_root=output_root,
        overwrite=True,
        write_train_captions=True,
        source_annotations_ref="synthetic/train_source.json",
    )

    assert summary.split == "train"
    assert summary.image_count == 2
    assert summary.annotation_count == 2

    train_dir = output_root / "train"
    assert (train_dir / "video-a-frame-000001-tokenA.npy").exists()
    assert (train_dir / "video-a-frame-000002-tokenB.npy").exists()
    assert (train_dir / "captions.json").exists()

    annotations_payload = json.loads((train_dir / "annotations.json").read_text(encoding="utf-8"))
    assert [cat["name"] for cat in annotations_payload["categories"]] == ["person", "bike", "car"]
    assert annotations_payload["annotations"][0]["image_id"] == "video-a-frame-000001-tokenA"
    assert annotations_payload["annotations"][0]["category_id"] == 0
    assert annotations_payload["annotations"][1]["category_id"] == 2

    captions_payload = json.loads((train_dir / "captions.json").read_text(encoding="utf-8"))
    assert captions_payload["video-a-frame-000001-tokenA"].endswith("1 person")
    assert captions_payload["video-a-frame-000002-tokenB"].startswith("overhead infrared surveillance image")


def test_align_eval_split_to_train_categories_removes_empty_images() -> None:
    pd = pytest.importorskip("pandas")

    images_df = pd.DataFrame(
        [
            {"image_id": "v1", "file_name": "a.jpg"},
            {"image_id": "v2", "file_name": "b.jpg"},
            {"image_id": "v3", "file_name": "c.jpg"},
        ]
    )
    annotations_df = pd.DataFrame(
        [
            {"image_id": "v1", "category_id": 1},
            {"image_id": "v2", "category_id": 3},
            {"image_id": "v3", "category_id": 2},
            {"image_id": "v3", "category_id": 3},
        ]
    )

    filtered_images, filtered_annotations, stats_payload = align_eval_split_to_train_categories(
        images_df,
        annotations_df,
        allowed_category_ids={1, 2},
    )

    assert filtered_images["image_id"].tolist() == ["v1", "v3"]
    assert filtered_annotations["category_id"].tolist() == [1, 2]
    assert stats_payload["images_removed"] == 1
    assert stats_payload["annotations_removed"] == 2


def test_download_wrapper_finds_nested_flir_root(tmp_path: Path) -> None:
    flir_root = tmp_path / "extract" / "FLIR_ADAS_v2"
    for split_dir in ("images_thermal_train", "images_thermal_val", "video_thermal_test"):
        path = flir_root / split_dir
        path.mkdir(parents=True)
        (path / "coco.json").write_text("{}", encoding="utf-8")

    assert find_flir_root(tmp_path / "extract") == flir_root


def test_download_wrapper_builds_full_dataset_command(tmp_path: Path) -> None:
    command = build_builder_command(
        flir_root=tmp_path / "flir",
        output_root=tmp_path / "out",
        splits=["train", "val"],
        max_images_per_split=5,
        overwrite=True,
        no_train_captions=True,
    )

    assert "build_flir_private_proxy_alignment_dataset.py" in command[1]
    assert "--use-full-dataset" in command
    assert "--flir-root" in command
    assert str(tmp_path / "flir") in command
    assert "--output-root" in command
    assert str(tmp_path / "out") in command
    assert command[command.index("--splits") + 1 : command.index("--max-images-per-split")] == ["train", "val"]
    assert command[-3:] == ["5", "--overwrite", "--no-train-captions"]


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    import zipfile

    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../escape.txt", "nope")

    with pytest.raises(ValueError, match="unsafe zip member"):
        safe_extract_zip(zip_path, tmp_path / "extract")
