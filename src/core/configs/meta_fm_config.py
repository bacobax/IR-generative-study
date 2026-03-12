"""Minimal config for a single-episode meta FM training run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.core.configs.text_fm_config import (
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
class MetaFMTrainConfig:
    """Configuration for a single incremental episode.

    Uses the same model/conditioning definitions as text FM training, plus
    three simple phase configs.
    """

    model: TextModelConfig = field(default_factory=TextModelConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    training: TextTrainHyperConfig = field(default_factory=TextTrainHyperConfig)
    output: TextOutputConfig = field(default_factory=TextOutputConfig)

    phase_a: MetaPhaseConfig = field(default_factory=MetaPhaseConfig)
    phase_b: MetaPhaseConfig = field(default_factory=MetaPhaseConfig)
    phase_c: MetaPhaseCConfig = field(default_factory=MetaPhaseCConfig)

    trainer_name: Optional[str] = "meta_fm"
