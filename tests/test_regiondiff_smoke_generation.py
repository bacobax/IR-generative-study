"""Tests for RegionDiff smoke synthetic export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.algorithms.inference.regiondiff_smoke_generation import export_generated_candidate_dataset
from src.algorithms.training.yolo_experiment_b import (
    YOLOBox,
    YOLOTrainSample,
    prepare_experiment_b_dataset,
    validate_experiment_b_config,
)
from src.core.configs.yolo_experiment_config import YOLOExperimentConfig


def test_export_generated_candidate_dataset_writes_experiment_b_shape(tmp_path: Path) -> None:
    image_path = tmp_path / "real.png"
    label_path = tmp_path / "real.txt"
    image_path.write_bytes(b"fake")
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    sample = YOLOTrainSample(
        index=3,
        image_path=image_path,
        label_path=label_path,
        boxes=[YOLOBox(0, 0.5, 0.5, 0.25, 0.25)],
    )
    output_dir = tmp_path / "generated"

    export_generated_candidate_dataset(
        output_dir=output_dir,
        source_samples=[sample],
        generated_arrays=[np.zeros((8, 8), dtype=np.float32)],
        dataset_payload={"names": {0: "person"}, "_yaml_path": "dataset.yaml"},
        generator_kind="regiondiff_test",
    )

    assert (output_dir / "images" / "sample_000001.npy").is_file()
    assert (output_dir / "annotations.json").is_file()
    assert (output_dir / "metadata" / "provenance.jsonl").is_file()
    summary = json.loads((output_dir / "metadata" / "summary.json").read_text(encoding="utf-8"))
    assert summary["generator_kind"] == "regiondiff_test"
    assert summary["n_generated_samples"] == 1


def test_yolo_experiment_b_accepts_precomputed_aug_without_filter(tmp_path: Path) -> None:
    from PIL import Image
    import yaml

    yolo_root = tmp_path / "yolo"
    for split in ("train", "val", "test"):
        (yolo_root / "images" / split).mkdir(parents=True)
        (yolo_root / "labels" / split).mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(yolo_root / "images" / "train" / "real.png")
    (yolo_root / "labels" / "train" / "real.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    dataset_yaml = yolo_root / "full_train.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(yolo_root),
                "train": str((yolo_root / "images" / "train").resolve()),
                "val": str((yolo_root / "images" / "val").resolve()),
                "test": str((yolo_root / "images" / "test").resolve()),
                "names": {0: "person"},
                "nc": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    export_generated_candidate_dataset(
        output_dir=generated_dir,
        source_samples=[
            YOLOTrainSample(
                index=0,
                image_path=yolo_root / "images" / "train" / "real.png",
                label_path=yolo_root / "labels" / "train" / "real.txt",
                boxes=[YOLOBox(0, 0.5, 0.5, 0.25, 0.25)],
            )
        ],
        generated_arrays=[np.ones((8, 8), dtype=np.float32)],
        dataset_payload={"names": {0: "person"}, "_yaml_path": str(dataset_yaml)},
        generator_kind="regiondiff_test",
    )

    cfg = YOLOExperimentConfig()
    cfg.data.dataset_yaml = str(dataset_yaml)
    cfg.data.full_train_dataset_yaml = str(dataset_yaml)
    cfg.experiment_b.mode = "precomputed_aug"
    cfg.experiment_b.precomputed_dataset_dir = str(generated_dir)
    cfg.experiment_b.augmented_yolo_root = str(tmp_path / "augmented")
    cfg.experiment_b.filter.enabled = False
    cfg.output.experiment_name = "smoked_yolo_test"

    validate_experiment_b_config(cfg)
    summary = prepare_experiment_b_dataset(cfg, device="cpu")
    augmented_yaml = Path(summary["augmented_dataset_yaml"])
    assert augmented_yaml.is_file()
    assert summary["n_generated_images"] == 1
    assert summary["n_kept_synthetic_images"] == 1
