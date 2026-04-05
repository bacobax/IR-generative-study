"""Tests for YOLO rare-slice baseline helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.algorithms.training.yolo_slice_baselines import (
    apply_center_scale,
    apply_constrained_crop_resize,
    apply_horizontal_flip,
    apply_translation,
    prepare_yolo_slice_baseline,
)
from src.analysis.flir_subgroup.yolo_slice_stats import load_yolo_slice_dataset


def _write_image(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((64, 64, 3), value, dtype=np.uint8)).save(path)


def _write_dataset_yaml(root: Path) -> Path:
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                f"train: {(root / 'train' / 'images').resolve()}",
                f"val: {(root / 'train' / 'images').resolve()}",
                f"test: {(root / 'train' / 'images').resolve()}",
                "names:",
                "  0: person",
                "  1: car",
                "nc: 2",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def _build_simple_dataset(root: Path) -> Path:
    image_dir = root / "train" / "images"
    label_dir = root / "train" / "labels"
    _write_image(image_dir / "img_a.png", value=10)
    _write_image(image_dir / "img_b.png", value=20)
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "img_a.txt").write_text(
        "\n".join(
            [
                "0 0.10 0.10 0.10 0.10",
                "0 0.50 0.50 0.20 0.20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (label_dir / "img_b.txt").write_text("1 0.90 0.90 0.30 0.30\n", encoding="utf-8")
    return _write_dataset_yaml(root)


def _build_weighted_dataset(root: Path) -> Path:
    image_dir = root / "train" / "images"
    label_dir = root / "train" / "labels"
    _write_image(image_dir / "img_common.png", value=30)
    _write_image(image_dir / "img_rare.png", value=40)
    _write_image(image_dir / "img_empty.png", value=50)
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "img_common.txt").write_text(
        "\n".join(
            [
                "0 0.20 0.20 0.20 0.20",
                "0 0.20 0.20 0.20 0.20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (label_dir / "img_rare.txt").write_text("1 0.80 0.80 0.10 0.10\n", encoding="utf-8")
    (label_dir / "img_empty.txt").write_text("", encoding="utf-8")
    return _write_dataset_yaml(root)


def test_load_yolo_slice_dataset_assigns_notebook_style_slices(tmp_path: Path) -> None:
    dataset_yaml = _build_simple_dataset(tmp_path)

    dataset = load_yolo_slice_dataset(dataset_yaml)

    assert round(dataset.thresholds.q33, 6) == round(0.03, 6)
    assert round(dataset.thresholds.q67, 6) == round(0.056667, 6)

    rows = dataset.instance_df.sort_values(["image_stem", "instance_index"]).reset_index(drop=True)
    assert rows.loc[0, "position_bin"] == "top_left"
    assert rows.loc[1, "position_bin"] == "middle_center"
    assert rows.loc[2, "position_bin"] == "bottom_right"
    assert rows.loc[0, "slice_key"] == ("person", "small", "top_left")
    assert rows.loc[1, "slice_key"] == ("person", "medium", "middle_center")
    assert rows.loc[2, "slice_key"] == ("car", "large", "bottom_right")


def test_prepare_yolo_slice_baseline_computes_rarity_and_weights(tmp_path: Path) -> None:
    dataset_yaml = _build_weighted_dataset(tmp_path)
    analysis_dir = tmp_path / "analysis"
    baseline_cfg = SimpleNamespace(
        mode="baseline_a",
        rarity_alpha=1.0,
        rarity_eps=1e-6,
        image_score_top_k=1,
        normalize_weights=True,
        clip_weight_min=None,
        clip_weight_max=10.0,
        sampler_replacement=True,
        targeted_aug_probability=0.5,
        targeted_aug_rarity_quantile=0.8,
    )

    prepared = prepare_yolo_slice_baseline(
        dataset_yaml=str(dataset_yaml),
        analysis_dir=analysis_dir,
        baseline_cfg=baseline_cfg,
    )

    common_row = prepared.image_sampling_df.loc[prepared.image_sampling_df["image_stem"] == "img_common"].iloc[0]
    rare_row = prepared.image_sampling_df.loc[prepared.image_sampling_df["image_stem"] == "img_rare"].iloc[0]
    empty_row = prepared.image_sampling_df.loc[prepared.image_sampling_df["image_stem"] == "img_empty"].iloc[0]

    assert rare_row["image_score"] > common_row["image_score"] > 0.0
    assert rare_row["sampling_weight"] > common_row["sampling_weight"] > empty_row["sampling_weight"] > 0.0
    assert round(float(prepared.image_sampling_df["sampling_weight"].mean()), 6) == 1.0
    assert common_row["protected_instance_indices"] == "0"
    assert rare_row["protected_instance_indices"] == "0"
    assert (analysis_dir / "slice_counts.csv").exists()
    assert (analysis_dir / "slice_summary.json").exists()
    assert (analysis_dir / "image_sampling_weights.csv").exists()
    assert (analysis_dir / "sampling_weight_summary.json").exists()


def test_geometry_helpers_update_boxes_and_preserve_validity() -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    boxes = np.array([[0.20, 0.50, 0.20, 0.20]], dtype=np.float32)

    _, translated = apply_translation(image, boxes, dx=0.10, dy=-0.05)
    assert np.allclose(translated[0], np.array([0.30, 0.45, 0.20, 0.20], dtype=np.float32), atol=1e-6)

    _, scaled = apply_center_scale(image, boxes, scale=1.10)
    assert np.allclose(scaled[0], np.array([0.17, 0.50, 0.22, 0.22], dtype=np.float32), atol=1e-6)

    _, flipped = apply_horizontal_flip(image, boxes)
    assert np.allclose(flipped[0], np.array([0.80, 0.50, 0.20, 0.20], dtype=np.float32), atol=1e-6)


def test_constrained_crop_resize_rejects_invalid_and_preserves_protected_box() -> None:
    image = np.full((48, 64, 3), 200, dtype=np.uint8)
    boxes = np.array(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.15, 0.15, 0.10, 0.10],
        ],
        dtype=np.float32,
    )

    rejected = apply_constrained_crop_resize(
        image,
        boxes,
        protected_indices=[0],
        crop_scale_min=0.20,
        crop_scale_max=0.20,
        min_retained_area=1.1,
        max_attempts=3,
    )
    assert rejected is None

    accepted = apply_constrained_crop_resize(
        image,
        boxes,
        protected_indices=[0],
        crop_scale_min=0.70,
        crop_scale_max=0.70,
        min_retained_area=0.5,
        max_attempts=3,
    )
    assert accepted is not None
    assert accepted["img"].shape == image.shape
    protected_box = accepted["boxes"][0]
    assert protected_box[2] > 0.0
    assert protected_box[3] > 0.0
