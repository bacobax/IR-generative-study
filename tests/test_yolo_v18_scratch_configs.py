"""Validate that the exp_v18_scratch_yolo11n configs load and resolve correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cli.train_yolo import _validate_config_yaml_keys, build_parser
from src.core.configs.config_loader import load_yaml, merge_config_and_cli
from src.core.configs.yolo_experiment_config import YOLOExperimentConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs" / "yolo" / "exp_v18_scratch_yolo11n"

CONCRETE_CONFIGS = ["default_aug.yaml", "plain.yaml", "rare_aug.yaml"]


@pytest.mark.parametrize("config_name", CONCRETE_CONFIGS)
def test_config_yaml_keys_are_valid(config_name: str) -> None:
    """All keys in the concrete configs (after extends resolution) map to known fields."""
    config_path = str(CONFIGS_DIR / config_name)
    _validate_config_yaml_keys(config_path)  # raises if unknown key found


@pytest.mark.parametrize("config_name", CONCRETE_CONFIGS)
def test_config_resolves_to_yolo11n_from_scratch(config_name: str) -> None:
    """Each config must resolve to yolo11n.yaml (from-scratch) as the model weights."""
    parser = build_parser()
    args = parser.parse_args([])
    cfg = merge_config_and_cli(
        YOLOExperimentConfig,
        str(CONFIGS_DIR / config_name),
        parser,
        args,
        flat_to_nested={},
    )
    assert cfg.model.weights == "yolo11n.yaml", (
        f"{config_name}: expected model.weights='yolo11n.yaml', got '{cfg.model.weights}'"
    )
    assert cfg.training.backbone_param_prefixes == [], (
        f"{config_name}: expected empty backbone_param_prefixes for from-scratch training"
    )
    assert cfg.training.freeze_backbone_epochs == 0


@pytest.mark.parametrize("config_name", CONCRETE_CONFIGS)
def test_config_has_per_slice_eval_enabled(config_name: str) -> None:
    """All three configs must have per_slice_enabled=True."""
    parser = build_parser()
    args = parser.parse_args([])
    cfg = merge_config_and_cli(
        YOLOExperimentConfig,
        str(CONFIGS_DIR / config_name),
        parser,
        args,
        flat_to_nested={},
    )
    assert cfg.evaluation.per_slice_enabled is True, (
        f"{config_name}: per_slice_enabled should be True"
    )
    assert cfg.evaluation.slice_threshold_dataset_yaml is not None


def test_plain_config_disables_aug() -> None:
    """plain.yaml must disable Ultralytics augmentation."""
    parser = build_parser()
    args = parser.parse_args([])
    cfg = merge_config_and_cli(
        YOLOExperimentConfig,
        str(CONFIGS_DIR / "plain.yaml"),
        parser,
        args,
        flat_to_nested={},
    )
    assert cfg.experiment_b.disable_ultralytics_augmentations is True
    assert cfg.baseline.mode == "none"


def test_default_aug_config_leaves_aug_on() -> None:
    """default_aug.yaml must not disable Ultralytics augmentation."""
    parser = build_parser()
    args = parser.parse_args([])
    cfg = merge_config_and_cli(
        YOLOExperimentConfig,
        str(CONFIGS_DIR / "default_aug.yaml"),
        parser,
        args,
        flat_to_nested={},
    )
    assert cfg.experiment_b.disable_ultralytics_augmentations is False
    assert cfg.baseline.mode == "none"


def test_rare_aug_config_has_targeted_aug_no_oversampling() -> None:
    """rare_aug.yaml must use baseline_b with use_weighted_sampler=False."""
    parser = build_parser()
    args = parser.parse_args([])
    cfg = merge_config_and_cli(
        YOLOExperimentConfig,
        str(CONFIGS_DIR / "rare_aug.yaml"),
        parser,
        args,
        flat_to_nested={},
    )
    assert cfg.baseline.mode == "baseline_b"
    assert cfg.baseline.use_weighted_sampler is False
    assert cfg.experiment_b.disable_ultralytics_augmentations is True
