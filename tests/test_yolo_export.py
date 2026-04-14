"""Tests for FLIR-to-YOLO export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.flir_subgroup.yolo_export import export_experiment_a_yolo_datasets


def _write_npy(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((32, 32), value, dtype=np.uint16))


def _image_row(image_id: str, root: Path, *, value: int) -> dict:
    file_name = f"{image_id}.npy"
    file_path = root / file_name
    _write_npy(file_path, value)
    return {
        "split": "train",
        "image_id": image_id,
        "image_key": f"train::{image_id}",
        "file_name": file_name,
        "file_path": str(file_path),
        "image_exists": True,
        "image_width": 32.0,
        "image_height": 32.0,
        "image_area": 1024.0,
        "caption": None,
        "n_annotations": 1,
    }


def _instance_row(image_id: str, ann_id: int, class_id: int, *, x: float, y: float, w: float, h: float) -> dict:
    bbox_area = w * h
    return {
        "split": "train",
        "image_id": image_id,
        "image_key": f"train::{image_id}",
        "file_name": f"{image_id}.npy",
        "file_path": "",
        "class_id": class_id,
        "class_label": "car" if class_id == 1 else "person",
        "ann_id": ann_id,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
        "bbox_area": bbox_area,
        "bbox_area_norm": bbox_area / 1024.0,
        "bbox_center_x": x + 0.5 * w,
        "bbox_center_y": y + 0.5 * h,
        "bbox_center_x_norm": (x + 0.5 * w) / 32.0,
        "bbox_center_y_norm": (y + 0.5 * h) / 32.0,
        "image_width": 32.0,
        "image_height": 32.0,
        "image_area": 1024.0,
        "iscrowd": 0,
    }


def test_export_experiment_a_yolo_datasets(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "yolo-test-ds"
    category_table = pd.DataFrame(
        [
            {"split": "train", "class_id": 1, "class_label": "car"},
            {"split": "train", "class_id": 2, "class_label": "person"},
        ]
    )

    balanced_images = pd.DataFrame([_image_row("b1", source_root, value=10), _image_row("b2", source_root, value=20)])
    unbalanced_images = pd.DataFrame([_image_row("u1", source_root, value=30)])
    full_train_images = pd.DataFrame([_image_row("f1", source_root, value=35), _image_row("f2", source_root, value=36)])
    val_images = pd.DataFrame([_image_row("v1", source_root, value=40)])
    test_images = pd.DataFrame([_image_row("t1", source_root, value=50), _image_row("t2", source_root, value=60)])

    balanced_instances = pd.DataFrame(
        [
            _instance_row("b1", 1, 1, x=4, y=5, w=8, h=10),
            _instance_row("b2", 2, 2, x=10, y=12, w=6, h=7),
        ]
    )
    unbalanced_instances = pd.DataFrame([_instance_row("u1", 3, 1, x=2, y=3, w=5, h=6)])
    full_train_instances = pd.DataFrame(
        [
            _instance_row("f1", 31, 1, x=3, y=4, w=9, h=11),
            _instance_row("f2", 32, 2, x=6, y=7, w=5, h=6),
        ]
    )
    val_instances = pd.DataFrame([_instance_row("v1", 4, 2, x=1, y=2, w=7, h=8)])
    test_instances = pd.DataFrame([_instance_row("t1", 5, 1, x=6, y=6, w=10, h=10)])

    result = export_experiment_a_yolo_datasets(
        output_root=output_root,
        category_table=category_table,
        balanced_image_df=balanced_images,
        balanced_instance_df=balanced_instances,
        unbalanced_image_df=unbalanced_images,
        unbalanced_instance_df=unbalanced_instances,
        full_train_image_df=full_train_images,
        full_train_instance_df=full_train_instances,
        val_image_df=val_images,
        val_instance_df=val_instances,
        test_image_df=test_images,
        test_instance_df=test_instances,
        overwrite=True,
    )

    summary_df = result["summary_df"]
    assert set(summary_df["dataset_name"]) == {"balanced", "unbalanced", "full_train", "val", "test"}
    assert int(summary_df.loc[summary_df["dataset_name"] == "balanced", "n_images"].iloc[0]) == 2
    assert int(summary_df.loc[summary_df["dataset_name"] == "full_train", "n_images"].iloc[0]) == 2
    assert int(summary_df.loc[summary_df["dataset_name"] == "test", "n_empty_images"].iloc[0]) == 1

    balanced_yaml = Path(result["balanced_yaml"])
    assert balanced_yaml.exists()
    yaml_text = balanced_yaml.read_text(encoding="utf-8")
    assert "balanced/images/train" in yaml_text
    assert "val/images/val" in yaml_text
    assert "test/images/test" in yaml_text

    balanced_label = output_root / "balanced" / "labels" / "train" / "b1.txt"
    assert balanced_label.exists()
    first_line = balanced_label.read_text(encoding="utf-8").strip()
    assert first_line.startswith("0 ")

    full_train_yaml = Path(result["full_train_yaml"])
    assert full_train_yaml.exists()
    full_train_yaml_text = full_train_yaml.read_text(encoding="utf-8")
    assert "full_train/images/train" in full_train_yaml_text

    empty_label = output_root / "test" / "labels" / "test" / "t2.txt"
    assert empty_label.exists()
    assert empty_label.read_text(encoding="utf-8") == ""

    png_path = output_root / "balanced" / "images" / "train" / "b1.png"
    assert png_path.exists()

    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()


def test_export_skips_missing_source_images(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "yolo-test-ds"
    category_table = pd.DataFrame(
        [
            {"split": "train", "class_id": 1, "class_label": "car"},
            {"split": "train", "class_id": 2, "class_label": "person"},
        ]
    )

    balanced_images = pd.DataFrame([_image_row("b1", source_root, value=10)])
    unbalanced_images = pd.DataFrame([_image_row("u1", source_root, value=20)])
    val_images = pd.DataFrame(
        [
            _image_row("v1", source_root, value=30),
            {
                **_image_row("v_missing", source_root, value=31),
                "file_path": None,
                "image_exists": False,
            },
        ]
    )
    test_images = pd.DataFrame([_image_row("t1", source_root, value=40)])

    balanced_instances = pd.DataFrame([_instance_row("b1", 1, 1, x=4, y=5, w=8, h=10)])
    unbalanced_instances = pd.DataFrame([_instance_row("u1", 2, 2, x=10, y=12, w=6, h=7)])
    val_instances = pd.DataFrame(
        [
            _instance_row("v1", 3, 2, x=1, y=2, w=7, h=8),
            _instance_row("v_missing", 4, 1, x=3, y=4, w=5, h=6),
        ]
    )
    test_instances = pd.DataFrame([_instance_row("t1", 5, 1, x=6, y=6, w=10, h=10)])

    result = export_experiment_a_yolo_datasets(
        output_root=output_root,
        category_table=category_table,
        balanced_image_df=balanced_images,
        balanced_instance_df=balanced_instances,
        unbalanced_image_df=unbalanced_images,
        unbalanced_instance_df=unbalanced_instances,
        val_image_df=val_images,
        val_instance_df=val_instances,
        test_image_df=test_images,
        test_instance_df=test_instances,
        overwrite=True,
    )

    summary_df = result["summary_df"]
    val_summary = summary_df.loc[summary_df["dataset_name"] == "val"].iloc[0]
    assert int(val_summary["n_images_requested"]) == 2
    assert int(val_summary["n_images"]) == 1
    assert int(val_summary["n_missing_source_images"]) == 1
    assert int(val_summary["n_annotations_requested"]) == 2
    assert int(val_summary["n_annotations_source"]) == 1
    assert int(val_summary["n_annotations_exported"]) == 1
    assert int(val_summary["n_annotations_skipped_missing_source"]) == 1

    exported_val_image = output_root / "val" / "images" / "val" / "v1.png"
    skipped_val_image = output_root / "val" / "images" / "val" / "v_missing.png"
    exported_val_label = output_root / "val" / "labels" / "val" / "v1.txt"
    skipped_val_label = output_root / "val" / "labels" / "val" / "v_missing.txt"
    assert exported_val_image.exists()
    assert exported_val_label.exists()
    assert not skipped_val_image.exists()
    assert not skipped_val_label.exists()
