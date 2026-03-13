"""Minimal config for a single-episode meta FM training run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.core.configs.text_fm_config import (
    TextDataConfig,
    TextAugmentConfig,
    ConditioningConfig,
    TextModelConfig,
    TextOutputConfig,
    TextTrainHyperConfig,
)


@dataclass
class MetaPhaseConfig:
    """Basic phase configuration (epochs + lr)."""

    epochs: int = 1
    lr: float = 1e-4


@dataclass
class MetaPhaseCConfig(MetaPhaseConfig):
    """Phase C configuration with replay and unfreezing policy."""

    replay_every: int = 1
    unfreeze_unet_policy: str = "none"  # none|all|mid|up
    router_trainable: bool = True
    router_lr_scale: float = 1.0


@dataclass
class ConditionSplitConfig:
    """Explicit curriculum condition split."""

    base: List[int] = field(default_factory=list)
    incremental: List[int] = field(default_factory=list)
    test: List[int] = field(default_factory=list)
    prompt_template: str = "IR image with {count} persons"


@dataclass
class RouterRegConfig:
    """Optional routing regularization settings."""

    sparsity_weight: float = 0.0
    smoothness_weight: float = 0.0


@dataclass
class EvaluationConfig:
    """Final evaluation settings for unseen conditions."""

    enabled: bool = True
    samples_per_condition: int = 4
    steps: int = 50
    guidance_scale: float = 7.5
    output_dir: str = "./artifacts/generated/meta_fm/test_conditions"


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

    phase_a: MetaPhaseConfig = field(default_factory=MetaPhaseConfig)
    phase_b: MetaPhaseConfig = field(default_factory=MetaPhaseConfig)
    phase_c: MetaPhaseCConfig = field(default_factory=MetaPhaseCConfig)

    condition_split: ConditionSplitConfig = field(default_factory=ConditionSplitConfig)
    router_reg: RouterRegConfig = field(default_factory=RouterRegConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    trainer_name: Optional[str] = "meta_fm"
