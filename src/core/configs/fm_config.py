"""Structured configuration objects for flow-matching training and sampling.

These dataclasses replace the loose argparse values that were previously
threaded through ``train_sfm.py``, ``FlowMatchingTrainer``, and
``FlowMatchingSampler``.  They are **not** Hydra configs — just plain
``dataclasses.dataclass`` objects with default values that match the
existing CLI defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch


# ═══════════════════════════════════════════════════════════════════════════
# Sub-configs
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    """Paths and loader settings for training / validation data."""

    train_dir: str = "./data/raw/v18/train/"
    val_dir: str = "./data/raw/v18/val/"
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Paths to model architecture configs and pretrained weights."""

    unet_config: str = "configs/models/fm/stable_unet_config.json"
    vae_config: str = "configs/models/fm/vae_config.json"
    vae_weights: Optional[str] = "./vae_best.pt"
    pretrained_unet_path: Optional[str] = None
    # Registry component names (None → use default)
    model_builder_name: Optional[str] = None


@dataclass
class AugmentConfig:
    """Augmentation schedule for ``ScheduledAugment256``."""

    warmup_frac: float = 0.1
    ramp_frac: float = 0.3
    p_crop_warmup: float = 0.05
    p_crop_max: float = 0.20
    p_crop_final: float = 0.05
    p_rot_warmup: float = 0.05
    p_rot_max: float = 0.30
    p_rot_final: float = 0.05


@dataclass
class TrainHyperConfig:
    """Core training hyper-parameters."""

    epochs: int = 100
    t_scale: float = 1000.0
    train_target: str = "v"          # "v" | "x0"
    save_every_n_epochs: int = 10
    patience: Optional[int] = None
    min_delta: float = 0.0
    strict_load: bool = True


@dataclass
class SampleConfig:
    """Parameters controlling per-epoch and stand-alone sampling."""

    sample_every_epoch: bool = True
    sample_steps: int = 50
    sample_batch_size: int = 4
    sample_shape: Optional[Tuple[int, int, int]] = None


@dataclass
class OutputConfig:
    """Checkpoint, log directory, and model output paths."""

    model_dir: str = "./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled/"
    log_dir: Optional[str] = None    # derived from model_dir if None
    resume: Optional[str] = None

    def resolved_log_dir(self) -> str:
        if self.log_dir is not None:
            return self.log_dir
        return f"{self.model_dir}/runs/stable_flow_matching_logs/"


# ═══════════════════════════════════════════════════════════════════════════
# Top-level composite configs
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FMTrainConfig:
    """Complete configuration for a flow-matching training run.

    Aggregates all sub-configs and provides a ``from_args`` factory that
    mirrors the existing ``argparse`` interface in ``train_sfm.py``.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: TrainHyperConfig = field(default_factory=TrainHyperConfig)
    sampling: SampleConfig = field(default_factory=SampleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Registry component names (None → use default)
    trainer_name: Optional[str] = None
    sampler_name: Optional[str] = None

    @classmethod
    def from_args(cls, args) -> "FMTrainConfig":
        """Build an ``FMTrainConfig`` from an ``argparse.Namespace``."""
        return cls(
            data=DataConfig(
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            ),
            model=ModelConfig(
                unet_config=args.unet_config,
                vae_config=args.vae_config,
                vae_weights=args.vae_weights,
            ),
            augment=AugmentConfig(
                warmup_frac=args.warmup_frac,
                ramp_frac=args.ramp_frac,
                p_crop_warmup=args.p_crop_warmup,
                p_crop_max=args.p_crop_max,
                p_crop_final=args.p_crop_final,
                p_rot_warmup=args.p_rot_warmup,
                p_rot_max=args.p_rot_max,
                p_rot_final=args.p_rot_final,
            ),
            training=TrainHyperConfig(
                epochs=args.epochs,
                t_scale=args.t_scale,
                train_target=args.train_target,
                save_every_n_epochs=args.save_every_n_epochs,
            ),
            sampling=SampleConfig(
                sample_batch_size=args.sample_batch_size,
            ),
            output=OutputConfig(
                model_dir=args.model_dir,
                resume=args.resume,
            ),
        )


@dataclass
class FMSampleConfig:
    """Configuration for stand-alone flow-matching sampling / generation.

    Mirrors the FM-related CLI flags from ``generate_datasets.py``.
    """

    pipeline_dir: str = "./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled/"
    vae_weights: Optional[str] = None
    t_scale: float = 1000.0
    train_target: str = "v"
    steps: int = 50
    batch_size: int = 8
    sample_shape: Optional[Tuple[int, int, int]] = None
    device: Optional[str] = None
    # Registry component names (None → use default)
    sampler_name: Optional[str] = None
    model_builder_name: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"
