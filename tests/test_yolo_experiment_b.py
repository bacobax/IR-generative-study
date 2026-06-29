"""Tests for YOLO Experiment B synthetic augmentation wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.algorithms.training import yolo_experiment_b
from src.algorithms.training.yolo_experiment_b import (
    build_instance_discard_summary,
    classify_generated_image_rows,
    export_augmented_yolo_dataset,
    load_full_train_samples,
    validate_experiment_b_config,
)
from src.cli import train_yolo
from src.core.configs.yolo_experiment_config import YOLOExperimentConfig


def _write_png(path: Path, value: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16), value, dtype=np.uint8), mode="L").convert("RGB").save(path)


def _make_yolo_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "yolo_ds"
    train_images = root / "full_train" / "images" / "train"
    train_labels = root / "full_train" / "labels" / "train"
    val_images = root / "val" / "images" / "val"
    test_images = root / "test" / "images" / "test"
    train_labels.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        _write_png(train_images / f"img{idx}.png", value=50 + idx)
    _write_png(val_images / "val0.png")
    _write_png(test_images / "test0.png")
    (train_labels / "img0.txt").write_text("0 0.50000000 0.50000000 0.50000000 0.50000000\n", encoding="utf-8")
    (train_labels / "img1.txt").write_text("", encoding="utf-8")
    yaml_path = root / "full_train.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": str(train_images),
                "val": str(val_images),
                "test": str(test_images),
                "names": {0: "person", 1: "car"},
                "nc": 2,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path


def _cfg_for_dataset(dataset_yaml: Path) -> YOLOExperimentConfig:
    cfg = YOLOExperimentConfig()
    cfg.data.dataset_yaml = str(dataset_yaml)
    cfg.data.full_train_dataset_yaml = str(dataset_yaml)
    cfg.output.experiment_name = "exp_test_b"
    cfg.experiment_b.generated_dataset_dir = str(dataset_yaml.parent / "generated")
    cfg.experiment_b.augmented_yolo_root = str(dataset_yaml.parent / "augmented")
    return cfg


def test_validate_experiment_b_rejects_non_full_train_dataset(tmp_path: Path) -> None:
    full_train_yaml = _make_yolo_dataset(tmp_path)
    other_yaml = tmp_path / "other.yaml"
    other_yaml.write_text(full_train_yaml.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = _cfg_for_dataset(full_train_yaml)
    cfg.data.dataset_yaml = str(other_yaml)

    with pytest.raises(ValueError, match="full-train only"):
        validate_experiment_b_config(cfg)


def test_validate_experiment_b_rejects_invalid_source_combinations(tmp_path: Path) -> None:
    cfg = _cfg_for_dataset(_make_yolo_dataset(tmp_path))
    cfg.experiment_b.mode = "sd_aug"
    cfg.experiment_b.sd.stage1_dir = "stage1"
    cfg.experiment_b.sd.lora_dir = "lora"

    with pytest.raises(ValueError, match="exactly one"):
        validate_experiment_b_config(cfg)

    cfg = _cfg_for_dataset(_make_yolo_dataset(tmp_path / "b"))
    cfg.experiment_b.mode = "fm_aug"
    with pytest.raises(ValueError, match="checkpoint_path"):
        validate_experiment_b_config(cfg)


def test_discard_generated_rows_uses_invalid_instance_ratio_threshold() -> None:
    rows = [
        {"generated_image_id": 1, "n_instances": 4, "n_negative_instances": 2},
        {"generated_image_id": 2, "n_instances": 4, "n_negative_instances": 3},
        {"generated_image_id": 3, "n_instances": 0, "n_negative_instances": 0},
    ]

    classified = classify_generated_image_rows(rows, invalid_instance_ratio_threshold=0.5)

    assert classified[0]["invalid_instance_ratio"] == 0.5
    assert classified[0]["discarded_by_invalid_instance_ratio"] is False
    assert classified[1]["invalid_instance_ratio"] == 0.75
    assert classified[1]["discarded_by_invalid_instance_ratio"] is True
    assert classified[2]["invalid_instance_ratio"] == 0.0
    assert classified[2]["discarded_by_invalid_instance_ratio"] is False


def test_instance_discard_summary_groups_global_category_size_and_combo() -> None:
    image_rows = classify_generated_image_rows(
        [
            {"generated_image_id": 1, "n_instances": 2, "n_negative_instances": 1},
            {"generated_image_id": 2, "n_instances": 2, "n_negative_instances": 2},
        ],
        invalid_instance_ratio_threshold=0.5,
    )
    instance_rows = [
        {"generated_image_id": 1, "category_name": "person", "size_bin": "small", "is_positive": True},
        {"generated_image_id": 1, "category_name": "car", "size_bin": "big", "is_positive": False},
        {"generated_image_id": 2, "category_name": "person", "size_bin": "small", "is_positive": False},
        {"generated_image_id": 2, "category_name": "car", "size_bin": "big", "is_positive": False},
    ]

    summary = build_instance_discard_summary(
        instance_rows=instance_rows,
        classified_image_rows=image_rows,
        invalid_instance_ratio_threshold=0.5,
    )

    assert summary["discarded_image_count"] == 1
    assert summary["global"]["total_instance_count"] == 4
    assert summary["global"]["discarded_instance_count"] == 2
    assert summary["global"]["discarded_valid_instance_count"] == 0
    assert summary["global"]["discarded_invalid_instance_count"] == 2
    by_category = {row["group"]: row for row in summary["by_category"]}
    assert by_category["person"]["discarded_invalid_instance_count"] == 1
    assert by_category["car"]["discarded_invalid_instance_count"] == 1
    by_size = {row["group"]: row for row in summary["by_size"]}
    assert by_size["small"]["discarded_instance_count"] == 1
    by_combo = {row["group"]: row for row in summary["by_category_size"]}
    assert by_combo["person | small"]["discarded_instance_count"] == 1


def test_export_augmented_yolo_dataset_writes_real_and_kept_synthetic(tmp_path: Path) -> None:
    dataset_yaml = _make_yolo_dataset(tmp_path)
    cfg = _cfg_for_dataset(dataset_yaml)
    cfg.experiment_b.mode = "fm_aug"
    samples, payload = load_full_train_samples(dataset_yaml)
    generated_dir = tmp_path / "generated_candidates"
    (generated_dir / "images").mkdir(parents=True)
    np.save(generated_dir / "images" / "sample_000001.npy", np.ones((16, 16), dtype=np.uint8) * 100)
    np.save(generated_dir / "images" / "sample_000002.npy", np.ones((16, 16), dtype=np.uint8) * 150)
    (generated_dir / "annotations.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "sample_000001.npy", "width": 16, "height": 16},
                    {"id": 2, "file_name": "sample_000002.npy", "width": 16, "height": 16},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [4, 4, 4, 4], "area": 16, "iscrowd": 0},
                ],
                "categories": [{"id": 0, "name": "person"}, {"id": 1, "name": "car"}],
            },
        ),
        encoding="utf-8",
    )
    classified_rows = [
        {
            "generated_image_id": 1,
            "generated_file_name": "sample_000001.npy",
            "discarded_by_invalid_instance_ratio": False,
            "invalid_instance_ratio": 0.0,
        },
        {
            "generated_image_id": 2,
            "generated_file_name": "sample_000002.npy",
            "discarded_by_invalid_instance_ratio": True,
            "invalid_instance_ratio": 1.0,
        },
    ]

    augmented_yaml = export_augmented_yolo_dataset(
        cfg=cfg,
        source_samples=samples,
        dataset_payload=payload,
        generated_dataset_dir=generated_dir,
        classified_rows=classified_rows,
    )

    augmented_payload = yaml.safe_load(augmented_yaml.read_text(encoding="utf-8"))
    train_dir = Path(augmented_payload["train"])
    label_dir = train_dir.parent.parent / "labels" / "train"
    assert len(list(train_dir.glob("*.png"))) == 4
    assert len(list(label_dir.glob("*.txt"))) == 4
    assert any(path.name.startswith("synthetic_000000_img0") for path in train_dir.glob("*.png"))
    synthetic_label_0 = label_dir / "synthetic_000000_img0.txt"
    synthetic_label_1 = label_dir / "synthetic_000001_img1.txt"
    assert synthetic_label_0.read_text(encoding="utf-8").startswith("1 ")
    assert synthetic_label_1.read_text(encoding="utf-8") == ""
    assert augmented_payload["val"] == yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))["val"]
    assert augmented_payload["test"] == yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))["test"]
    assert augmented_payload["names"] == {0: "person", 1: "car"}


def test_run_exp_b_plain_skips_synthetic_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg_for_dataset(_make_yolo_dataset(tmp_path))
    cfg.experiment_b.mode = "plain"
    calls: list[str] = []

    monkeypatch.setattr(train_yolo, "prepare_experiment_b_dataset", lambda *args, **kwargs: pytest.fail("no prep"))
    monkeypatch.setattr(
        train_yolo,
        "run_train",
        lambda cfg: calls.append(cfg.data.dataset_yaml) or {"best_weights_path": "/tmp/best.pt", "tensorboard_log_dir": "/tmp/run"},
    )
    monkeypatch.setattr(
        train_yolo,
        "run_eval",
        lambda cfg, *, weights_path=None: {
            "dataset_yaml": cfg.data.dataset_yaml,
            "split": "val",
            "map": 0.1,
            "map50": 0.2,
            "map75": 0.15,
            "precision": 0.3,
            "recall": 0.4,
        },
    )

    summary = train_yolo.run_experiment_b(cfg)

    assert calls == [str(Path(cfg.data.dataset_yaml))]
    assert summary["mode"] == "plain"


def test_run_exp_b_augmented_modes_use_prepared_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_yaml = _make_yolo_dataset(tmp_path)
    prepared_yaml = tmp_path / "prepared.yaml"
    prepared_yaml.write_text(dataset_yaml.read_text(encoding="utf-8"), encoding="utf-8")
    modes_seen: list[str] = []
    trained_datasets: list[str] = []

    def fake_prepare(cfg: YOLOExperimentConfig, *, device: str):
        modes_seen.append(cfg.experiment_b.mode)
        return {
            "mode": cfg.experiment_b.mode,
            "augmented_dataset_yaml": str(prepared_yaml),
            "n_generated_images": 2,
        }

    monkeypatch.setattr(train_yolo, "prepare_experiment_b_dataset", fake_prepare)
    monkeypatch.setattr(
        train_yolo,
        "run_train",
        lambda cfg: trained_datasets.append(cfg.data.dataset_yaml)
        or {"best_weights_path": "/tmp/best.pt", "tensorboard_log_dir": "/tmp/run"},
    )
    monkeypatch.setattr(
        train_yolo,
        "run_eval",
        lambda cfg, *, weights_path=None: {
            "dataset_yaml": cfg.data.dataset_yaml,
            "split": "val",
            "map": 0.1,
            "map50": 0.2,
            "map75": 0.15,
            "precision": 0.3,
            "recall": 0.4,
        },
    )

    fm_cfg = _cfg_for_dataset(dataset_yaml)
    fm_cfg.experiment_b.mode = "fm_aug"
    fm_cfg.experiment_b.fm.checkpoint_path = "fm.pt"
    sd_cfg = _cfg_for_dataset(dataset_yaml)
    sd_cfg.experiment_b.mode = "sd_aug"
    sd_cfg.experiment_b.sd.stage1_dir = "stage1"

    train_yolo.run_experiment_b(fm_cfg)
    train_yolo.run_experiment_b(sd_cfg)

    assert modes_seen == ["fm_aug", "sd_aug"]
    assert trained_datasets == [str(prepared_yaml), str(prepared_yaml)]


def test_threshold_from_config_reaches_audit_discard_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg_for_dataset(_make_yolo_dataset(tmp_path))
    cfg.experiment_b.invalid_instance_ratio_threshold = 0.25
    seen_thresholds: list[float] = []

    def fake_classify(rows, *, invalid_instance_ratio_threshold: float):
        seen_thresholds.append(invalid_instance_ratio_threshold)
        return []

    monkeypatch.setattr(yolo_experiment_b, "classify_generated_image_rows", fake_classify)
    monkeypatch.setattr(
        yolo_experiment_b,
        "_resolve_filter_source",
        lambda cfg: (None, tmp_path / "fake.pt"),
    )

    class FakeModel:
        def eval(self):
            return None

    monkeypatch.setattr(
        yolo_experiment_b,
        "load_filter_from_run_or_checkpoint",
        lambda **kwargs: (FakeModel(), {"classifier_mode": "multiclass"}, {"0": 0.5}, 16, 1.0, None),
    )
    monkeypatch.setattr(
        yolo_experiment_b,
        "audit_generated_layout_dataset",
        lambda **kwargs: ([], [{"generated_image_id": 1, "n_instances": 1, "n_negative_instances": 1}], {}),
    )
    monkeypatch.setattr(yolo_experiment_b, "export_audit_results", lambda **kwargs: None)
    monkeypatch.setattr(
        yolo_experiment_b,
        "_write_filtered_annotations_from_audit",
        lambda **kwargs: {"n_invalid_annotations_removed": 1, "n_annotations_unfiltered": 1},
    )

    yolo_experiment_b.audit_generated_candidates(
        cfg=cfg,
        generated_dataset_dir=tmp_path / "generated",
        device="cpu",
    )

    assert seen_thresholds == [0.25]


def test_experiment_b_filter_accepts_binary_classifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Binary fg/bg classifier must be accepted by audit_generated_candidates (no multiclass-only restriction)."""
    cfg = _cfg_for_dataset(_make_yolo_dataset(tmp_path))
    cfg.experiment_b.invalid_instance_ratio_threshold = 1.0
    seen_modes: list[str] = []

    class FakeModel:
        def eval(self):
            return None

    monkeypatch.setattr(
        yolo_experiment_b,
        "_resolve_filter_source",
        lambda cfg: (None, tmp_path / "fake.pt"),
    )
    monkeypatch.setattr(
        yolo_experiment_b,
        "load_filter_from_run_or_checkpoint",
        lambda **kwargs: (FakeModel(), {"classifier_mode": "binary"}, 0.5, 16, 1.0, None),
    )
    monkeypatch.setattr(
        yolo_experiment_b,
        "audit_generated_layout_dataset",
        lambda **kwargs: ([], [{"generated_image_id": 1, "n_instances": 1, "n_negative_instances": 0}], {}),
    )
    monkeypatch.setattr(yolo_experiment_b, "export_audit_results", lambda **kwargs: None)
    monkeypatch.setattr(
        yolo_experiment_b,
        "_write_filtered_annotations_from_audit",
        lambda **kwargs: {"n_invalid_annotations_removed": 0, "n_annotations_unfiltered": 1},
    )

    # Must not raise — binary is now allowed
    yolo_experiment_b.audit_generated_candidates(
        cfg=cfg,
        generated_dataset_dir=tmp_path / "generated",
        device="cpu",
    )
