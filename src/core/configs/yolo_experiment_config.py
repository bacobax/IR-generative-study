"""Structured configuration for YOLO export, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from src.core.paths import (
    yolo_analysis_root,
    yolo_checkpoints_root,
    yolo_runs_root,
    yolo_test_ds_root,
)


@dataclass
class YOLODataConfig:
    """Dataset paths and loader settings for YOLO experiments."""

    dataset_yaml: str = str(yolo_test_ds_root() / "balanced.yaml")
    balanced_dataset_yaml: str = str(yolo_test_ds_root() / "balanced.yaml")
    unbalanced_dataset_yaml: str = str(yolo_test_ds_root() / "unbalanced.yaml")
    full_train_dataset_yaml: str = str(yolo_test_ds_root() / "full_train.yaml")
    test_dataset_yaml: Optional[str] = None
    batch_size: int = 16
    workers: int = 4
    image_size: int = 640


@dataclass
class YOLOModelConfig:
    """Model weights and Ultralytics task selection."""

    weights: str = "yolov8n.pt"
    task: str = "detect"


@dataclass
class YOLOTrainConfig:
    """Core optimization and reproducibility controls."""

    epochs: int = 50
    lr0: float = 0.01
    optimizer: str = "auto"
    momentum: float = 0.937
    weight_decay: float = 0.0005
    seed: int = 7
    deterministic: bool = True
    patience: int = 20
    cos_lr: bool = False
    freeze_backbone_epochs: int = 0
    freeze_backbone_layers: Any = None
    backbone_lr_multiplier: float = 1.0
    backbone_param_prefixes: list[str] = field(default_factory=list)


@dataclass
class YOLOBaselineConfig:
    """Slice-aware baseline controls for full-train YOLO experiments."""

    mode: str = "none"
    rarity_alpha: float = 1.0
    rarity_eps: float = 1e-6
    image_score_top_k: int = 3
    normalize_weights: bool = True
    clip_weight_min: Optional[float] = None
    clip_weight_max: Optional[float] = 10.0
    sampler_replacement: bool = True
    targeted_aug_probability: float = 0.5
    targeted_aug_rarity_quantile: float = 0.8
    translate_fraction: float = 0.05
    scale_min: float = 0.9
    scale_max: float = 1.1
    crop_scale_min: float = 0.85
    crop_scale_max: float = 1.0
    crop_min_rare_box_area_retained: float = 0.5
    crop_max_attempts: int = 10
    allow_horizontal_flip: bool = True
    seed: int = 7


@dataclass
class YOLOEvalConfig:
    """Evaluation-time settings."""

    split: str = "test"
    save_json: bool = True
    save_hybrid: bool = False
    conf: Optional[float] = None
    iou: Optional[float] = None


@dataclass
class YOLOOutputConfig:
    """Artifact locations and experiment naming."""

    experiment_name: str = "exp_balanced"
    runs_root: str = field(default_factory=lambda: str(yolo_runs_root()))
    checkpoints_root: str = field(default_factory=lambda: str(yolo_checkpoints_root()))
    analysis_root: str = field(default_factory=lambda: str(yolo_analysis_root()))

    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.experiment_name

    def checkpoint_dir(self) -> Path:
        return Path(self.checkpoints_root) / self.experiment_name

    def analysis_dir(self) -> Path:
        return Path(self.analysis_root) / self.experiment_name


@dataclass
class YOLOLauncherConfig:
    """Names for the paired Experiment A runs."""

    balanced_experiment_name: str = "exp_balanced"
    unbalanced_experiment_name: str = "exp_unbalanced"
    full_train_experiment_name: str = "exp_full_train"
    comparison_filename: str = "comparison_summary.csv"
    ordered_config_paths: list[str] = field(default_factory=list)
    ordered_labels: list[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class YOLOExperimentConfig:
    """Complete configuration for YOLO train/eval workflows."""

    data: YOLODataConfig = field(default_factory=YOLODataConfig)
    model: YOLOModelConfig = field(default_factory=YOLOModelConfig)
    training: YOLOTrainConfig = field(default_factory=YOLOTrainConfig)
    baseline: YOLOBaselineConfig = field(default_factory=YOLOBaselineConfig)
    evaluation: YOLOEvalConfig = field(default_factory=YOLOEvalConfig)
    output: YOLOOutputConfig = field(default_factory=YOLOOutputConfig)
    launcher: YOLOLauncherConfig = field(default_factory=YOLOLauncherConfig)
    device: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"
