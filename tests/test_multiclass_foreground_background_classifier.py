"""Tests for the multiclass FLIR crop classifier pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.standalone.train_multiclass_fg_bg_classifier import main as train_multiclass_main
from src.algorithms.inference.rare_layout_dataset_tools import audit_generated_layout_dataset
from src.algorithms.training.foreground_background_utils import select_best_thresholds_per_class
from src.core.data.foreground_background_dataset import MultiClassCropDataset, build_balanced_sample_weights
from src.models.foreground_background_classifier import MultiClassForegroundBackgroundClassifier


def _write_split(split_dir: Path, *, split_name: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    annotation_id = 1
    for image_index in range(3):
        image_id = f"{split_name}-img-{image_index}"
        file_name = f"{image_id}.npy"
        image = np.full((64, 64), 20 + 15 * image_index, dtype=np.uint8)
        image[6:18, 6:18] = 240
        image[40:54, 40:56] = 180
        np.save(split_dir / file_name, image)
        images.append(
            {"id": image_id, "file_name": file_name, "width": 64, "height": 64}
        )
        annotations.append(
            {"id": annotation_id, "image_id": image_id, "category_id": 0, "bbox": [6, 6, 12, 12], "area": 144, "iscrowd": 0}
        )
        annotation_id += 1
        annotations.append(
            {"id": annotation_id, "image_id": image_id, "category_id": 2, "bbox": [40, 40, 16, 14], "area": 224, "iscrowd": 0}
        )
        annotation_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "person"}, {"id": 2, "name": "car"}],
    }
    (split_dir / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "flir_like"
    for split in ("train", "val", "test"):
        _write_split(root / split, split_name=split)
    return root


def test_multiclass_dataset_builds_foreground_and_background_labels(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path)
    dataset = MultiClassCropDataset(split="train", dataset_root=root, input_size=32, seed=7, negative_max_retries=32)
    assert dataset.background_class_index == 2
    assert dataset.num_classes == 3
    assert dataset.stats()["num_positive"] == 6
    assert dataset.stats()["num_negative"] <= dataset.stats()["num_positive"]
    sample = dataset[0]
    assert sample["pixel_values"].shape == (1, 32, 32)
    assert sample["label"].dtype == torch.long
    weights = build_balanced_sample_weights(dataset)
    assert weights.shape[0] == len(dataset)
    assert torch.isfinite(weights).all()
    assert any(bool(item.is_background) for item in dataset.samples)
    assert any((not bool(item.is_background)) and item.category_id == 0 for item in dataset.samples)


def test_multiclass_model_forward_and_threshold_selection() -> None:
    model = MultiClassForegroundBackgroundClassifier(num_classes=4)
    logits = model(torch.randn(5, 1, 32, 32))
    assert logits.shape == (5, 4)

    synthetic_logits = np.asarray(
        [
            [5.0, -2.0, -4.0],
            [4.0, -1.5, -3.0],
            [-2.0, 5.0, -3.0],
            [-1.0, 4.0, -2.0],
            [-4.0, -3.0, 6.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)
    payload = select_best_thresholds_per_class(
        logits=synthetic_logits,
        labels=labels,
        foreground_class_indices=[0, 1],
    )
    assert "0" in payload["thresholds"]
    assert "1" in payload["thresholds"]
    assert payload["metrics_by_class"]["0"]["f1"] >= 0.99


def test_train_multiclass_classifier_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_dataset_root(tmp_path)
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multiclass_fg_bg_classifier.py",
            "--dataset_root",
            str(root),
            "--output_dir",
            str(output_dir),
            "--input_size",
            "32",
            "--batch_size",
            "2",
            "--epochs",
            "1",
            "--num_workers",
            "0",
            "--max_logged_images",
            "4",
            "--device",
            "cpu",
            "--mixed_precision",
            "no",
            "--scheduler",
            "none",
        ],
    )
    train_multiclass_main()
    best_path = output_dir / "checkpoints" / "best.pt"
    summary_path = output_dir / "metrics" / "summary.json"
    tensorboard_dir = output_dir / "tensorboard"
    assert best_path.is_file()
    assert summary_path.is_file()
    assert tensorboard_dir.is_dir()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["classifier_mode"] == "multiclass"
    assert "per_class_thresholds" in summary
    assert "model_index_to_category_id" in summary


class _ControlledMulticlassModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for sample in x:
            mean_value = float(sample.mean())
            if mean_value > 0.8:
                out.append(torch.tensor([5.0, 0.5, -1.0], dtype=torch.float32))
            elif mean_value > 0.4:
                out.append(torch.tensor([0.2, 5.0, -1.0], dtype=torch.float32))
            else:
                out.append(torch.tensor([-1.0, 0.1, 5.0], dtype=torch.float32))
        return torch.stack(out, dim=0)


def test_multiclass_audit_marks_match_threshold_wrong_class_and_background(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "generated_ds"
    images_dir = dataset_dir / "images"
    metadata_dir = dataset_dir / "metadata"
    images_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    np.save(images_dir / "sample_000001.npy", np.full((1, 32, 32), 1.0, dtype=np.float32))
    np.save(images_dir / "sample_000002.npy", np.full((1, 32, 32), 0.6, dtype=np.float32))
    np.save(images_dir / "sample_000003.npy", np.full((1, 32, 32), 0.1, dtype=np.float32))

    annotations = {
        "images": [
            {"id": 1, "file_name": "sample_000001.npy", "width": 32, "height": 32},
            {"id": 2, "file_name": "sample_000002.npy", "width": 32, "height": 32},
            {"id": 3, "file_name": "sample_000003.npy", "width": 32, "height": 32},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 32, 32], "area": 1024, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 0, "bbox": [0, 0, 32, 32], "area": 1024, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 0, "bbox": [0, 0, 32, 32], "area": 1024, "iscrowd": 0},
        ],
        "categories": [{"id": 0, "name": "person"}, {"id": 1, "name": "bike"}],
    }
    (dataset_dir / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    (metadata_dir / "summary.json").write_text(json.dumps({"size_bin_thresholds": [0.1, 0.2]}), encoding="utf-8")

    classifier_summary = {
        "classifier_mode": "multiclass",
        "background_class_index": 2,
        "category_id_to_model_index": {"0": 0, "1": 1},
        "model_index_to_category_id": {"0": 0, "1": 1},
        "category_id_to_name": {"0": "person", "1": "bike"},
        "per_class_thresholds": {"0": 0.97, "1": 0.6},
    }
    instance_rows, image_rows, stats = audit_generated_layout_dataset(
        dataset_dir=dataset_dir,
        filter_model=_ControlledMulticlassModel(),
        threshold=classifier_summary["per_class_thresholds"],
        filter_input_size=32,
        context_ratio=1.0,
        min_valid_object_fraction=1.0,
        device="cpu",
        crop_batch_size=4,
        classifier_summary=classifier_summary,
    )
    assert [row["is_positive"] for row in instance_rows] == [True, False, False]
    assert instance_rows[1]["is_class_match"] is False
    assert instance_rows[2]["is_background_prediction"] is True
    assert stats["multiclass"]["wrong_class_count"] == 1
    assert stats["multiclass"]["background_prediction_count"] == 1

