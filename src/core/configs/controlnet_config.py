"""Structured configuration objects for ControlNet flow-matching training.

Mirrors the pattern of :mod:`src.core.configs.fm_config` but tailored to
ControlNet stage-2 training (frozen UNet + VAE, trainable ControlNet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Sub-configs
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CNDataConfig:
    """Paths and loader settings for ControlNet training / validation data."""

    train_dir: str = "./data/raw/v18/train/"
    val_dir: str = "./data/raw/v18/val/"
    train_annotations: str = "./data/raw/v18/train/annotations.json"
    val_annotations: str = "./data/raw/v18/val/annotations.json"
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class CNStage1Config:
    """Paths to the frozen stage-1 pipeline (UNet + VAE)."""

    stage1_pipeline_dir: Optional[str] = None
    vae_weights_override: Optional[str] = None


@dataclass
class CNControlNetConfig:
    """ControlNet architecture parameters."""

    conditioning_channels: int = 1
    conditioning_downscale_factor: int = 4


@dataclass
class CNTrainHyperConfig:
    """Core training hyper-parameters for ControlNet."""

    epochs: int = 100
    lr: float = 1e-4
    t_scale: float = 1000.0
    conditioning_scale: float = 1.0
    conditioning_dropout: float = 0.1
    save_every_n_epochs: int = 10
    patience: Optional[int] = None
    min_delta: float = 0.0


@dataclass
class CNSampleConfig:
    """Parameters controlling per-epoch sampling."""

    sample_every: int = 1
    sample_steps: int = 50


@dataclass
class CNOutputConfig:
    """Checkpoint, log directory, and model output paths."""

    model_dir: str = "./controlnet_runs/bbox_controlnet/"
    log_dir: Optional[str] = None
    resume: Optional[str] = None

    def resolved_log_dir(self) -> str:
        if self.log_dir is not None:
            return self.log_dir
        return f"{self.model_dir}/runs/controlnet_logs/"


# ═══════════════════════════════════════════════════════════════════════════
# Top-level composite config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CNTrainConfig:
    """Complete configuration for a ControlNet training run.

    Aggregates all sub-configs.  Designed for 3-layer merge:
    dataclass defaults → YAML → CLI overrides.
    """

    data: CNDataConfig = field(default_factory=CNDataConfig)
    stage1: CNStage1Config = field(default_factory=CNStage1Config)
    controlnet: CNControlNetConfig = field(default_factory=CNControlNetConfig)
    training: CNTrainHyperConfig = field(default_factory=CNTrainHyperConfig)
    sampling: CNSampleConfig = field(default_factory=CNSampleConfig)
    output: CNOutputConfig = field(default_factory=CNOutputConfig)
