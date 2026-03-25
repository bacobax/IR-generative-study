"""Structured configuration objects for flow-matching training and sampling.

These dataclasses replace the loose argparse values that were previously
threaded through ``train_sfm.py``, ``FlowMatchingTrainer``, and
``FlowMatchingSampler``.  They are **not** Hydra configs — just plain
``dataclasses.dataclass`` objects with default values that match the
existing CLI defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ═══════════════════════════════════════════════════════════════════════════
# Sub-configs
# ═══════════════════════════════════════════════════════════════════════════


def _validate_image_size(image_size: int) -> int:
    """Require a positive square image size divisible by 32."""
    image_size = int(image_size)
    if image_size <= 0 or image_size % 32 != 0:
        raise ValueError(
            f"image_size must be a positive multiple of 32, got {image_size}"
        )
    return image_size

@dataclass
class CountFilterConfig:
    """Filter training images by person count for generalization testing.

    Specify either ``seen_counts`` (whitelist) or ``unseen_counts``
    (blacklist).  If both are None, no filtering is applied.  Setting
    both is an error.

    After cropping, the resulting count is also checked.  If the crop
    produces an excluded count, the pipeline retries up to
    ``max_crop_retries`` times before falling back to the full image.
    """

    seen_counts: Optional[List[int]] = None
    unseen_counts: Optional[List[int]] = None
    max_crop_retries: int = 5


@dataclass
class CurriculumConfig:
    """Curriculum learning settings for person-centered cropping.

    When ``enabled`` is False (default), no curriculum crop is applied.
    When enabled, crops are sampled around annotated people with
    probability driven by a schedule over training epochs.
    """

    enabled: bool = False
    crop_prob_start: float = 0.0
    crop_prob_end: float = 0.5
    schedule: str = "linear"          # "linear" | "fixed"
    margin_min: float = 1.2
    margin_max: float = 2.0
    center_jitter: float = 0.15
    force_square: bool = False
    total_epochs: int = 100           # set automatically from training.epochs


@dataclass
class DataConfig:
    """Paths and loader settings for training / validation data."""

    dataset_id: Optional[str] = None
    train_dir: str = "./data/raw/v18/train/"
    val_dir: str = "./data/raw/v18/val/"
    annotations_path: Optional[str] = None
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    subset_strategy: str = "first_n"

    def __post_init__(self) -> None:
        self.image_size = _validate_image_size(self.image_size)


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
    lr: float = 1e-4
    t_scale: float = 1000.0
    train_target: str = "v"          # "v" | "x0"
    save_every_n_epochs: int = 10
    patience: Optional[int] = None
    min_delta: float = 0.0
    strict_load: bool = True


@dataclass
class LayoutConditioningConfig:
    """Settings for the bbox-conditioned pixel-space FM path."""

    enabled: bool = False
    num_classes: Optional[int] = None
    category_id_to_name: Dict[int, str] = field(default_factory=dict)
    class_embed_dim: int = 32
    bbox_embed_dim: int = 32
    spatial_channels: int = 8
    raster_mode: str = "box_fill_mean"
    log_internal_maps: bool = True


@dataclass
class LoggingConfig:
    """Step-based TensorBoard logging cadence for layout-conditioned FM."""

    scalar_every_steps: int = 10
    image_every_steps: int = 200
    max_logged_images: int = 4


@dataclass
class SampleConfig:
    """Parameters controlling per-epoch and stand-alone sampling."""

    sample_every: int = 1
    sample_steps: int = 50
    sample_batch_size: int = 4
    sample_shape: Optional[Tuple[int, int, int]] = None
    fixed_validation_examples: int = 4
    sample_every_steps: int = 0
    save_debug_images: bool = False


@dataclass
class OutputConfig:
    """Checkpoint, log directory, and model output paths."""

    model_dir: str = "./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled/"
    log_dir: Optional[str] = None    # derived from model_dir if None
    debug_dir: Optional[str] = None  # derived from model_dir if None
    resume: Optional[str] = None

    def resolved_log_dir(self) -> str:
        if self.log_dir is not None:
            return self.log_dir
        return f"{self.model_dir}/runs/stable_flow_matching_logs/"

    def resolved_debug_dir(self) -> str:
        if self.debug_dir is not None:
            return self.debug_dir
        return f"{self.model_dir}/debug_samples/"


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
    layout_conditioning: LayoutConditioningConfig = field(default_factory=LayoutConditioningConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    sampling: SampleConfig = field(default_factory=SampleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    count_filter: CountFilterConfig = field(default_factory=CountFilterConfig)
    # Registry component names (None → use default)
    trainer_name: Optional[str] = None
    sampler_name: Optional[str] = None
    device: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_args(cls, args) -> "FMTrainConfig":
        """Build an ``FMTrainConfig`` from an ``argparse.Namespace``."""
        return cls(
            data=DataConfig(
                dataset_id=getattr(args, "dataset_id", None),
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                annotations_path=getattr(args, "annotations_path", None),
                image_size=getattr(args, "image_size", 256),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_train_samples=getattr(args, "max_train_samples", None),
                max_val_samples=getattr(args, "max_val_samples", None),
                subset_strategy=getattr(args, "subset_strategy", "first_n"),
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
                lr=getattr(args, "lr", 1e-4),
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
