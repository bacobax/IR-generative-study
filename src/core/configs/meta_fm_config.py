"""Minimal config for a single-episode meta FM training run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from src.core.configs.text_fm_config import (
    TextDataConfig,
    TextAugmentConfig,
    ConditioningConfig,
    TextModelConfig,
    TextOutputConfig,
    TextTrainHyperConfig,
)
from src.core.configs.fm_config import CurriculumConfig, CountFilterConfig


@dataclass
class MetaPhaseConfig:
    """Basic phase configuration with explicit module trainability."""

    epochs: int = 1
    lr: float = 1e-4
    lambda_corr: float = 1.0
    mlp_trainable: bool = True
    router_trainable: bool = True
    moe_trainable: bool = True
    unet_trainable: bool = True
    unfreeze_unet_policy: str = "all"  # none|all|mid|up


@dataclass
class MetaPhaseCConfig(MetaPhaseConfig):
    """Phase C configuration with replay and optional router LR scaling."""

    replay_every: int = 1
    router_lr_scale: float = 1.0


@dataclass
class MetaCheckpointConfig:
    """Full-state checkpoint policy for curriculum meta training."""

    enabled: bool = True
    save_every_epochs: int = 1
    save_latest: bool = True
    latest_filename: str = "meta_fm_latest.pt"
    dir_name: str = "meta_checkpoints"

    def resolved_dir(self, model_dir: str) -> str:
        return f"{model_dir}/{self.dir_name}"


@dataclass
class ConditionSplitConfig:
    """Explicit curriculum condition split."""

    base: List[int] = field(default_factory=list)
    incremental: List[int] = field(default_factory=list)
    test: List[int] = field(default_factory=list)


@dataclass
class RouterRegConfig:
    """Optional routing regularization settings."""

    sparsity_weight: float = 0.0
    smoothness_weight: float = 0.0
    balance_weight: float = 0.0


@dataclass
class EvaluationConfig:
    """Final evaluation settings for unseen conditions."""

    enabled: bool = True
    samples_per_condition: int = 4
    steps: int = 50
    guidance_scale: float = 7.5
    output_dir: str = "./artifacts/generated/meta_fm/test_conditions"


@dataclass
class MetaSamplingConfig:
    """Sanity-check sampling during curriculum training."""

    enabled: bool = False
    phase_a_every: int = 0
    steps: int = 50
    guidance_scale: float = 7.5
    samples_per_condition: int = 4
    output_dir: str = "./artifacts/generated/meta_fm/sanity_samples"


@dataclass
class MetaFMTrainConfig:
    """Configuration for a single incremental episode.

    Uses the same model/conditioning definitions as text FM training, plus
    three simple phase configs.
    """

    data: TextDataConfig = field(default_factory=TextDataConfig)
    model: TextModelConfig = field(default_factory=TextModelConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    augment: TextAugmentConfig = field(default_factory=TextAugmentConfig)
    training: TextTrainHyperConfig = field(default_factory=TextTrainHyperConfig)
    output: TextOutputConfig = field(default_factory=TextOutputConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    count_filter: CountFilterConfig = field(default_factory=CountFilterConfig)

    phase_a: MetaPhaseConfig = field(
        default_factory=lambda: MetaPhaseConfig(
            mlp_trainable=True,
            router_trainable=True,
            moe_trainable=True,
            unet_trainable=True,
            unfreeze_unet_policy="all",
        ),
    )
    phase_b: MetaPhaseConfig = field(
        default_factory=lambda: MetaPhaseConfig(
            mlp_trainable=True,
            router_trainable=True,
            moe_trainable=False,
            unet_trainable=False,
            unfreeze_unet_policy="none",
        ),
    )
    phase_c: MetaPhaseCConfig = field(
        default_factory=lambda: MetaPhaseCConfig(
            mlp_trainable=True,
            router_trainable=True,
            moe_trainable=True,
            unet_trainable=False,
            unfreeze_unet_policy="none",
        ),
    )

    condition_split: ConditionSplitConfig = field(default_factory=ConditionSplitConfig)
    router_reg: RouterRegConfig = field(default_factory=RouterRegConfig)
    sampling: MetaSamplingConfig = field(default_factory=MetaSamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: MetaCheckpointConfig = field(default_factory=MetaCheckpointConfig)

    trainer_name: Optional[str] = "meta_fm"
    device: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"
