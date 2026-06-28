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
    cache_images: bool = False


@dataclass
class SimpleYOLOModelConfig:
    """Small native PyTorch YOLO architecture controls."""

    input_channels: int = 3
    base_channels: int = 16
    width_multiplier: float = 1.0
    channel_multipliers: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    blocks_per_stage: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    output_stride: int = 16
    boxes_per_cell: int = 2
    activation: str = "silu"
    dropout: float = 0.0


@dataclass
class YOLOModelConfig:
    """Model weights/backend selection and task settings."""

    weights: str = "yolov8n.pt"
    task: str = "detect"
    backend: str = "ultralytics"
    simple: SimpleYOLOModelConfig = field(default_factory=SimpleYOLOModelConfig)


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
    mixed_precision: str = "no"
    grad_clip_norm: Optional[float] = None
    val_interval: int = 1
    tensorboard_image_interval: int = 1
    tensorboard_max_images: int = 4
    tensorboard_prediction_conf: float = 0.25


@dataclass
class YOLOLossConfig:
    """Native simple YOLO loss weights."""

    box_weight: float = 5.0
    giou_weight: float = 2.0
    objectness_weight: float = 1.0
    no_object_weight: float = 0.5
    class_weight: float = 1.0


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
    use_weighted_sampler: bool = True
    seed: int = 7


@dataclass
class YOLOEvalConfig:
    """Evaluation-time settings."""

    dataset_yaml: Optional[str] = None
    split: str = "test"
    save_json: bool = True
    save_hybrid: bool = False
    conf: Optional[float] = None
    iou: Optional[float] = None
    per_slice_enabled: bool = False
    slice_threshold_dataset_yaml: Optional[str] = None


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
class YOLOExperimentBFilterConfig:
    """Filter settings for Experiment B synthetic images.

    ``kind`` selects the validation mechanism:
      - ``"fgbg"``: multiclass foreground/background crop classifier (default).
      - ``"yolo"``: run the trained YOLO detector and keep a GT box only when a
        prediction matches it at IoU >= ``adherence_iou``.
    """

    enabled: bool = True
    kind: str = "fgbg"
    checkpoint_dir: str = (
        "artifacts/checkpoints/multiclass_foreground_background_filter/"
        "runs/multiclass_fgbg_20260415_210440/checkpoints"
    )
    checkpoint_path: Optional[str] = None
    batch_size: int = 64
    # YOLO-adherence filter (kind="yolo") settings.
    yolo_weights: str = (
        "artifacts/checkpoints/yolo/exp_v18_scratch_yolo11n/default_aug/best.pt"
    )
    adherence_iou: float = 0.5
    yolo_conf: float = 0.25
    discard_empty_images: bool = True


@dataclass
class YOLOExperimentBFMConfig:
    """Layout-conditioned flow-matching generation settings for Experiment B."""

    checkpoint_path: Optional[str] = None
    preset_path: str = "configs/fm/train/presets/stay_layout_latent_flir_sd15_512.yaml"
    steps: int = 50
    batch_size: int = 8


@dataclass
class YOLOExperimentBSDConfig:
    """Stage-1 Stable Diffusion generation settings for Experiment B."""

    stage1_dir: Optional[str] = None
    lora_dir: Optional[str] = None
    base_model: str = "runwayml/stable-diffusion-v1-5"
    sd_steps: int = 100
    guidance: float = 7.5
    precision: str = "fp16"
    max_tries: int = 25
    prompt_mode: str = "constant"
    negative_prompt: Optional[str] = None


@dataclass
class YOLOExperimentBBalancedConfig:
    """Slice-balanced layout-generation settings (mode=fm_balanced_aug).

    Builds new multi-box layouts that over-represent rare 27-slices
    (1 class x 3 size bins x 9 position bins) to flatten the combined real+synth
    distribution. ``target`` selects the per-slice goal: "max" (every slice up to
    the largest real slice count), "mean", or an integer count.
    """

    target: str = "max"
    n_objects_source: str = "real"
    jitter_frac: float = 0.15
    max_pair_iou: float = 0.5
    placement_tries: int = 8
    seed: int = 7


@dataclass
class YOLOExperimentBConfig:
    """Full-train Experiment B controls."""

    mode: str = "plain"
    invalid_instance_ratio_threshold: float = 0.5
    generated_dataset_dir: str = "artifacts/generated/yolo/exp_b/generated_candidates"
    precomputed_dataset_dir: Optional[str] = None
    augmented_yolo_root: str = "artifacts/generated/yolo/exp_b/augmented_yolo"
    disable_ultralytics_augmentations: bool = False
    filter: YOLOExperimentBFilterConfig = field(default_factory=YOLOExperimentBFilterConfig)
    fm: YOLOExperimentBFMConfig = field(default_factory=YOLOExperimentBFMConfig)
    sd: YOLOExperimentBSDConfig = field(default_factory=YOLOExperimentBSDConfig)
    balanced: YOLOExperimentBBalancedConfig = field(default_factory=YOLOExperimentBBalancedConfig)


@dataclass
class YOLOAugConfig:
    """General per-image augmentation for the native simple_torch YOLO trainer.

    Defaults are all OFF so existing configs are unaffected.  Set ``enabled: true``
    in a YAML preset to activate.  Mutually exclusive with ``baseline.mode: baseline_b``
    (rare-slice targeted aug) — the trainer raises an error if both are enabled.

    All magnitudes use the same conventions as Ultralytics YOLO hyps.
    HSV hue/sat are intentionally omitted: v18 images are grayscale (R=G=B).
    """

    enabled: bool = False
    fliplr: float = 0.5       # probability of horizontal flip
    scale: float = 0.5        # ± scale gain fraction (e.g. 0.5 → [0.5x, 1.5x])
    translate: float = 0.1    # ± translate as fraction of image dimension
    brightness: float = 0.4   # ± brightness gain applied to uint8 [0,255] pixel values
    contrast: float = 0.0     # ± contrast gain (0 = disabled)


@dataclass
class YOLOExperimentConfig:
    """Complete configuration for YOLO train/eval workflows."""

    data: YOLODataConfig = field(default_factory=YOLODataConfig)
    model: YOLOModelConfig = field(default_factory=YOLOModelConfig)
    training: YOLOTrainConfig = field(default_factory=YOLOTrainConfig)
    loss: YOLOLossConfig = field(default_factory=YOLOLossConfig)
    baseline: YOLOBaselineConfig = field(default_factory=YOLOBaselineConfig)
    augment: YOLOAugConfig = field(default_factory=YOLOAugConfig)
    evaluation: YOLOEvalConfig = field(default_factory=YOLOEvalConfig)
    output: YOLOOutputConfig = field(default_factory=YOLOOutputConfig)
    launcher: YOLOLauncherConfig = field(default_factory=YOLOLauncherConfig)
    experiment_b: YOLOExperimentBConfig = field(default_factory=YOLOExperimentBConfig)
    device: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"
