"""Structured configuration for text-conditioned flow-matching (train + sample).

Follows the same pattern as :mod:`src.core.configs.fm_config` but adds
fields for text conditioning, CFG conditioning dropout, and guidance
scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch


from src.core.configs.fm_config import CurriculumConfig, CountFilterConfig


# ═══════════════════════════════════════════════════════════════════════
# Sub-configs
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TextDataConfig:
    """Paths and loader settings for text+image datasets."""

    train_dir: str = "./data/raw/v18/train/"
    val_dir: str = "./data/raw/v18/val/"
    annotations_path: Optional[str] = None
    text_caption_source: str = "annotations"  # only "annotations" supported
    fallback_text: Optional[str] = None
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class TextModelConfig:
    """Model architecture and pretrained weight paths."""

    unet_config: str = "configs/models/fm/text_unet_config.json"
    vae_config: str = "configs/models/fm/vae_config.json"
    vae_weights: Optional[str] = None
    pretrained_unet_path: Optional[str] = None
    model_builder_name: Optional[str] = None


@dataclass
class ConditioningConfig:
    """Text encoder and CFG conditioning dropout."""

    text_encoder: str = "openai/clip-vit-large-patch14"
    max_text_length: int = 77
    cond_drop_prob: float = 0.1
    return_pooled: bool = False


@dataclass
class TextAugmentConfig:
    """Augmentation schedule for ScheduledAugment256."""

    warmup_frac: float = 0.1
    ramp_frac: float = 0.3
    p_crop_warmup: float = 0.05
    p_crop_max: float = 0.20
    p_crop_final: float = 0.05
    p_rot_warmup: float = 0.05
    p_rot_max: float = 0.30
    p_rot_final: float = 0.05


@dataclass
class TextTrainHyperConfig:
    """Core training hyper-parameters."""

    epochs: int = 100
    lr: float = 1e-4
    t_scale: float = 1000.0
    train_target: str = "v"
    save_every_n_epochs: int = 10
    patience: Optional[int] = None
    min_delta: float = 0.0
    strict_load: bool = True


@dataclass
class TextSampleConfig:
    """Per-epoch sampling during training."""

    sample_every: int = 1
    sample_steps: int = 50
    sample_batch_size: int = 4
    sample_shape: Optional[Tuple[int, int, int]] = None


@dataclass
class AttentionVisConfig:
    """Configuration for cross-attention heatmap visualization.

    Controls which timesteps, layers, and tokens are captured,
    how attention heads are reduced, and how heatmaps are rendered.
    """

    enabled: bool = False
    target_tokens: List[str] = field(
        default_factory=lambda: ["person", "people"],
    )
    num_vis_steps: int = 8
    layer_filter: Union[str, List[str]] = "all"
    head_reduction: str = "mean"
    overlay: bool = True
    colormap: str = "jet"
    per_layer: bool = False
    vis_guidance_scale: float = 7.5


@dataclass
class TextOutputConfig:
    """Checkpoint, log directory, and model output paths."""

    model_dir: str = "./artifacts/checkpoints/flow_matching/text_fm/"
    log_dir: Optional[str] = None
    resume: Optional[str] = None

    def resolved_log_dir(self) -> str:
        if self.log_dir is not None:
            return self.log_dir
        return f"{self.model_dir}/runs/text_fm_logs/"


# ═══════════════════════════════════════════════════════════════════════════
# Top-level composite configs
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TextFMTrainConfig:
    """Complete configuration for text-conditioned FM training with CFG."""

    data: TextDataConfig = field(default_factory=TextDataConfig)
    model: TextModelConfig = field(default_factory=TextModelConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    augment: TextAugmentConfig = field(default_factory=TextAugmentConfig)
    training: TextTrainHyperConfig = field(default_factory=TextTrainHyperConfig)
    sampling: TextSampleConfig = field(default_factory=TextSampleConfig)
    attention_vis: AttentionVisConfig = field(default_factory=AttentionVisConfig)
    output: TextOutputConfig = field(default_factory=TextOutputConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    count_filter: CountFilterConfig = field(default_factory=CountFilterConfig)
    trainer_name: Optional[str] = "text_fm_cfg"
    sampler_name: Optional[str] = "cfg_fm"
    device: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TextFMSampleConfig:
    """Configuration for text-conditioned FM sampling with CFG."""

    pipeline_dir: str = "./artifacts/checkpoints/flow_matching/text_fm/"
    vae_weights: Optional[str] = None
    t_scale: float = 1000.0
    train_target: str = "v"
    steps: int = 50
    batch_size: int = 8
    guidance_scale: float = 7.5
    prompt: str = ""
    sample_shape: Optional[Tuple[int, int, int]] = None
    device: Optional[str] = None
    text_encoder: str = "openai/clip-vit-large-patch14"
    max_text_length: int = 77
    sampler_name: Optional[str] = "cfg_fm"
    model_builder_name: Optional[str] = "text_fm_unet"

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"
