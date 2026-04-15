"""Tests for the FLIR foreground/background classifier training helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.standalone.train_fg_bg_classifier import main as train_fg_bg_main
from src.algorithms.training.foreground_background_utils import (
    load_training_checkpoint,
    save_training_checkpoint,
    select_best_threshold,
)
from src.models.foreground_background_classifier import ForegroundBackgroundClassifier


def _write_split(split_dir: Path, *, split_name: str) -> None:
    import json

    split_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    annotation_id = 1
    for image_index in range(2):
        image_id = f"{split_name}-{image_index}"
        file_name = f"{image_id}.npy"
        image = np.zeros((64, 64), dtype=np.uint8)
        image[8:20, 8:20] = 220
        image[36:52, 40:56] = 150
        np.save(split_dir / file_name, image)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": 64,
                "height": 64,
            }
        )
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [8, 8, 12, 12],
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
                "bbox": [40, 36, 16, 16],
                "area": 256,
                "iscrowd": 0,
            }
        )
        annotation_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
    }
    (split_dir / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "flir_like"
    for split in ("train", "val", "test"):
        _write_split(root / split, split_name=split)
    return root


def test_foreground_background_classifier_forward_shape() -> None:
    model = ForegroundBackgroundClassifier()
    logits = model(torch.randn(4, 1, 32, 32))
    assert logits.shape == (4,)


def test_threshold_selection_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    logits = np.asarray([-4.0, -2.0, 2.0, 4.0], dtype=np.float32)
    labels = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    areas = np.asarray([0.001, 0.003, 0.02, 0.015], dtype=np.float32)
    threshold_payload = select_best_threshold(logits=logits, labels=labels, positive_area_ratios=areas)

    assert threshold_payload["metrics"]["f1"] == pytest.approx(1.0)
    assert 0.05 <= threshold_payload["threshold"] <= 0.95

    model = ForegroundBackgroundClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        global_step=17,
        config={"hello": "world"},
        best_val_metric=0.9,
        best_threshold=threshold_payload["threshold"],
        best_val_metrics=threshold_payload["metrics"],
        best_test_metrics={"f1": 0.8},
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    payload = load_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location="cpu",
    )
    assert payload["epoch"] == 3
    assert payload["global_step"] == 17
    assert payload["best_threshold"] == pytest.approx(threshold_payload["threshold"])
    assert payload["best_val_metrics"]["f1"] == pytest.approx(1.0)


def test_train_fg_bg_classifier_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_dataset_root(tmp_path)
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_fg_bg_classifier.py",
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

    train_fg_bg_main()

    best_path = output_dir / "checkpoints" / "best.pt"
    latest_path = output_dir / "checkpoints" / "latest.pt"
    summary_path = output_dir / "metrics" / "summary.json"
    per_epoch_path = output_dir / "metrics" / "per_epoch.jsonl"
    tensorboard_dir = output_dir / "tensorboard"

    assert best_path.is_file()
    assert latest_path.is_file()
    assert summary_path.is_file()
    assert per_epoch_path.is_file()
    assert tensorboard_dir.is_dir()
    assert any(path.name.startswith("events.out.tfevents") for path in tensorboard_dir.iterdir())

    summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    assert "chosen_threshold" in summary
    assert "best_val_metrics" in summary
    assert "best_test_metrics" in summary
