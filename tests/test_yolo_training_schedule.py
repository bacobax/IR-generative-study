"""Tests for YOLO staged training config validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cli.train_yolo import _build_train_stages, _ordered_experiment_entries, _validate_config_yaml_keys
from src.core.configs.yolo_experiment_config import YOLOExperimentConfig


def test_build_train_stages_creates_frozen_and_finetune_phases() -> None:
    cfg = YOLOExperimentConfig()
    cfg.output.experiment_name = "exp_balanced"
    cfg.training.epochs = 12
    cfg.training.freeze_backbone_epochs = 3
    cfg.training.freeze_backbone_layers = 10
    cfg.training.backbone_lr_multiplier = 0.1
    cfg.training.backbone_param_prefixes = ["model.0.", "model.1."]

    stages = _build_train_stages(cfg)

    assert [stage["stage_name"] for stage in stages] == ["backbone_frozen", "full_finetune"]
    assert stages[0]["epochs"] == 3
    assert stages[0]["freeze"] == 10
    assert stages[1]["epochs"] == 9
    assert stages[1]["freeze"] is None
    assert stages[0]["run_name"] == "exp_balanced_phase1_frozen"
    assert stages[1]["run_name"] == "exp_balanced"


def test_validate_config_yaml_keys_rejects_unknown_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_yolo_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "training:",
                "  epochs: 10",
                "  lr0: 0.01",
                "  typo_lr: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training.typo_lr"):
        _validate_config_yaml_keys(str(config_path))


def test_validate_config_yaml_keys_accepts_baseline_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "good_yolo_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "baseline:",
                "  mode: baseline_b",
                "  rarity_alpha: 1.0",
                "  rarity_eps: 1.0e-6",
                "  image_score_top_k: 3",
                "  normalize_weights: true",
                "  targeted_aug_probability: 0.5",
                "  translate_fraction: 0.05",
                "  scale_min: 0.9",
                "  scale_max: 1.1",
                "  crop_scale_min: 0.85",
                "  crop_scale_max: 1.0",
                "  crop_min_rare_box_area_retained: 0.5",
                "  crop_max_attempts: 10",
                "  allow_horizontal_flip: true",
                "  seed: 7",
            ]
        ),
        encoding="utf-8",
    )

    assert _validate_config_yaml_keys(str(config_path))["baseline"]["mode"] == "baseline_b"


def test_validate_config_yaml_keys_rejects_unknown_baseline_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_baseline_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "baseline:",
                "  mode: baseline_a",
                "  typo_probability: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline.typo_probability"):
        _validate_config_yaml_keys(str(config_path))


def test_ordered_experiment_entries_follow_launcher_order() -> None:
    cfg = YOLOExperimentConfig()
    cfg.launcher.ordered_config_paths = [
        "configs/yolo/exp_a/flir/exp_balanced.yaml",
        "configs/yolo/exp_a/flir/exp_unbalanced.yaml",
        "configs/yolo/exp_a/flir/exp_full_train.yaml",
        "configs/yolo/exp_a/flir/exp_full_train_baseline_a.yaml",
        "configs/yolo/exp_a/flir/exp_full_train_baseline_b.yaml",
    ]
    cfg.launcher.ordered_labels = [
        "balanced",
        "unbalanced",
        "full_train",
        "full_train_baseline_a",
        "full_train_baseline_b",
    ]

    assert _ordered_experiment_entries(cfg) == [
        ("balanced", "configs/yolo/exp_a/flir/exp_balanced.yaml"),
        ("unbalanced", "configs/yolo/exp_a/flir/exp_unbalanced.yaml"),
        ("full_train", "configs/yolo/exp_a/flir/exp_full_train.yaml"),
        ("full_train_baseline_a", "configs/yolo/exp_a/flir/exp_full_train_baseline_a.yaml"),
        ("full_train_baseline_b", "configs/yolo/exp_a/flir/exp_full_train_baseline_b.yaml"),
    ]
