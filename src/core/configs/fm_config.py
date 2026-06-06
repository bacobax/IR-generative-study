"""Structured configuration objects for flow-matching training and sampling.

These dataclasses replace the loose argparse values that were previously
threaded through ``train_flow_matching.py``, ``FlowMatchingTrainer``, and
``FlowMatchingSampler``.  They are **not** Hydra configs — just plain
``dataclasses.dataclass`` objects with default values that match the
existing CLI defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
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
    subset_manifest: Optional[str] = None

    def __post_init__(self) -> None:
        self.image_size = _validate_image_size(self.image_size)


@dataclass
class ModelConfig:
    """Paths to model architecture configs and pretrained weights."""

    architecture: str = "unet"
    unet_config: str = "configs/models/fm/stable_unet_config.json"
    dit_config: str = "configs/models/dit/dit_b4_latent.json"
    vae_config: Optional[str] = "configs/models/fm/vae_config.json"
    vae_weights: Optional[str] = "./vae_best.pt"
    vae_pretrained_model_name_or_path: Optional[str] = None
    vae_pretrained_subfolder: Optional[str] = "vae"
    vae_revision: Optional[str] = None
    vae_variant: Optional[str] = None
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
    p_hflip_warmup: float = 0.0
    p_hflip_max: float = 0.0
    p_hflip_final: float = 0.0


@dataclass
class TrainHyperConfig:
    """Core training hyper-parameters."""

    epochs: int = 100
    lr: float = 1e-4
    gradient_accumulation_steps: int = 1
    t_scale: float = 1000.0
    train_target: str = "v"          # "v" | "x0"
    save_every_n_epochs: int = 10
    eval_every: int = 1
    patience: Optional[int] = None
    min_delta: float = 0.0
    strict_load: bool = True
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive.")
        self.t_scale = float(self.t_scale)
        if not math.isfinite(self.t_scale) or self.t_scale <= 0.0:
            raise ValueError(
                f"training.t_scale must be a positive finite float, got {self.t_scale!r}"
            )


@dataclass
class OptimizerConfig:
    """Optimizer settings for FM training."""

    name: str = "adamw"
    lr: Optional[float] = None
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class SchedulerConfig:
    """Learning-rate scheduler settings."""

    name: str = "warmup_cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1


@dataclass
class EMAConfig:
    """Exponential moving average settings."""

    enabled: bool = True
    decay: float = 0.999
    start_step: int = 100


@dataclass
class PrecisionConfig:
    """Mixed-precision settings."""

    mixed_precision: str = "auto"


@dataclass
class PathConfig:
    """Transport-path configuration for FM training."""

    mode: str = "independent"  # independent | minibatch_ot | conditional_ot
    solver: str = "hungarian"
    layout_cost_resolution: int = 16
    condition_weight: float = 1.0


@dataclass
class LayoutConditioningConfig:
    """Settings for the bbox-conditioned pixel-space FM path."""

    enabled: bool = False
    variant: str = "raster_v1"
    num_classes: Optional[int] = None
    category_id_to_name: Dict[int, str] = field(default_factory=dict)
    class_embed_dim: int = 32
    bbox_embed_dim: int = 32
    object_embed_dim: int = 64
    use_style_latent: bool = True
    style_latent_dim: int = 16
    style_seed: int = 1234
    spatial_channels: int = 8
    raster_mode: str = "box_fill_mean"
    mask_resolution: int = 16
    mask_hidden_channels: int = 32
    mask_threshold: float = 0.5
    edge_dilation: int = 1
    injection_mode: str = "ea_norm"
    use_masked_context: bool = True
    mask_overlap_loss_weight: float = 0.05
    mask_sharpness_loss_weight: float = 0.01
    mask_activation_loss_weight: float = 0.01
    log_internal_maps: bool = True
    active_region_resolutions: List[int] = field(default_factory=lambda: [64, 32, 16])
    layout_token_dim: int = 256
    bbox_fourier_dim: int = 64
    same_class_position_slots: int = 128
    use_background_token: bool = True
    use_spatial_adapter: bool = False
    train_mode: str = "adapters_only"
    partial_backbone_modules: List[str] = field(default_factory=lambda: ["mid_block", "up_blocks"])
    adapter_learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    area_loss_enabled: bool = False
    area_loss_alpha: float = 1.0
    area_loss_background_weight: float = 0.5
    area_loss_min_weight: float = 0.5
    area_loss_max_weight: float = 4.0
    area_x0_loss_weight: float = 1.0


@dataclass
class DistillationConfig:
    """Optional RegionDiff attention-map distillation settings for FM training."""

    enabled: bool = False
    teacher_checkpoint: Optional[str] = None
    teacher_model_type: str = "sd15_regiondiff"
    loss_type: str = "attention_kl"
    lambda_attn: float = 0.05
    warmup_epochs: int = 80
    selected_categories: List[str] = field(default_factory=list)
    selected_region_layers: List[str] = field(default_factory=list)
    timestep_range: Tuple[float, float] = (0.2, 0.8)
    detach_teacher: bool = True
    normalize_attention: bool = True
    bbox_mask_only: bool = True
    log_attention_maps_every: int = 1000

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.detach_teacher = bool(self.detach_teacher)
        self.normalize_attention = bool(self.normalize_attention)
        self.bbox_mask_only = bool(self.bbox_mask_only)

        self.teacher_model_type = str(self.teacher_model_type)
        if self.teacher_model_type not in {"sd15_regiondiff"}:
            raise ValueError(
                "distillation.teacher_model_type must be 'sd15_regiondiff', "
                f"got {self.teacher_model_type!r}"
            )

        self.loss_type = str(self.loss_type)
        if self.loss_type not in {"attention_kl", "attention_l2"}:
            raise ValueError(
                "distillation.loss_type must be 'attention_kl' or 'attention_l2', "
                f"got {self.loss_type!r}"
            )

        self.lambda_attn = float(self.lambda_attn)
        if not math.isfinite(self.lambda_attn) or self.lambda_attn < 0.0:
            raise ValueError(
                "distillation.lambda_attn must be a finite non-negative float, "
                f"got {self.lambda_attn!r}"
            )

        self.warmup_epochs = int(self.warmup_epochs)
        if self.warmup_epochs < 0:
            raise ValueError(
                "distillation.warmup_epochs must be non-negative, "
                f"got {self.warmup_epochs!r}"
            )

        if len(self.timestep_range) != 2:
            raise ValueError(
                "distillation.timestep_range must contain exactly two values, "
                f"got {self.timestep_range!r}"
            )
        start, end = (float(self.timestep_range[0]), float(self.timestep_range[1]))
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(
                "distillation.timestep_range values must be finite, "
                f"got {self.timestep_range!r}"
            )
        if start < 0.0 or end > 1.0 or start > end:
            raise ValueError(
                "distillation.timestep_range must satisfy 0 <= start <= end <= 1, "
                f"got {(start, end)!r}"
            )
        self.timestep_range = (start, end)

        self.log_attention_maps_every = int(self.log_attention_maps_every)
        if self.log_attention_maps_every < 0:
            raise ValueError(
                "distillation.log_attention_maps_every must be non-negative, "
                f"got {self.log_attention_maps_every!r}"
            )

        self.selected_categories = [
            str(item) for item in (self.selected_categories or [])
        ]
        self.selected_region_layers = [
            str(item) for item in (self.selected_region_layers or [])
        ]


@dataclass
class LoggingConfig:
    """Scalar cadence plus legacy step-based image logging for layout FM."""

    scalar_every_steps: int = 10
    image_every_steps: int = 200
    max_logged_images: int = 4


@dataclass
class SampleConfig:
    """Parameters controlling sampling and image logging cadence."""

    sample_every: int = 1
    sample_steps: int = 50
    sample_batch_size: int = 4
    sample_shape: Optional[Tuple[int, int, int]] = None
    fixed_validation_examples: int = 4
    sample_every_steps: int = 0
    early_sanity_sample_epoch: int = 0
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
    mirrors the existing ``argparse`` interface in ``train_flow_matching.py``.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: TrainHyperConfig = field(default_factory=TrainHyperConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    path: PathConfig = field(default_factory=PathConfig)
    layout_conditioning: LayoutConditioningConfig = field(default_factory=LayoutConditioningConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    sampling: SampleConfig = field(default_factory=SampleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    count_filter: CountFilterConfig = field(default_factory=CountFilterConfig)
    architecture_mode: str = "legacy"
    # Registry component names (None → use default)
    trainer_name: Optional[str] = None
    sampler_name: Optional[str] = None
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.architecture_mode = str(self.architecture_mode or "legacy")
        if self.architecture_mode not in {"legacy", "adapter_v1"}:
            raise ValueError(
                "architecture_mode must be 'legacy' or 'adapter_v1', "
                f"got {self.architecture_mode!r}"
            )

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolved_lr(self) -> float:
        """Return the effective optimizer learning rate."""
        if self.optimizer.lr is not None:
            return float(self.optimizer.lr)
        return float(self.training.lr)

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
                subset_manifest=getattr(args, "subset_manifest", None),
            ),
            model=ModelConfig(
                unet_config=args.unet_config,
                vae_config=args.vae_config,
                vae_weights=args.vae_weights,
                vae_pretrained_model_name_or_path=getattr(args, "vae_pretrained_model_name_or_path", None),
                vae_pretrained_subfolder=getattr(args, "vae_pretrained_subfolder", "vae"),
                vae_revision=getattr(args, "vae_revision", None),
                vae_variant=getattr(args, "vae_variant", None),
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
                p_hflip_warmup=getattr(args, "p_hflip_warmup", 0.0),
                p_hflip_max=getattr(args, "p_hflip_max", 0.0),
                p_hflip_final=getattr(args, "p_hflip_final", 0.0),
            ),
            training=TrainHyperConfig(
                epochs=args.epochs,
                lr=getattr(args, "lr", 1e-4),
                gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
                t_scale=args.t_scale,
                train_target=args.train_target,
                save_every_n_epochs=args.save_every_n_epochs,
                max_grad_norm=getattr(args, "max_grad_norm", 1.0),
            ),
            optimizer=OptimizerConfig(
                name=getattr(args, "optimizer_name", "adamw"),
                lr=getattr(args, "optimizer_lr", None),
                weight_decay=getattr(args, "weight_decay", 0.01),
                beta1=getattr(args, "beta1", 0.9),
                beta2=getattr(args, "beta2", 0.999),
            ),
            scheduler=SchedulerConfig(
                name=getattr(args, "scheduler_name", "warmup_cosine"),
                warmup_ratio=getattr(args, "warmup_ratio", 0.05),
                min_lr_ratio=getattr(args, "min_lr_ratio", 0.1),
            ),
            ema=EMAConfig(
                enabled=getattr(args, "ema_enabled", True),
                decay=getattr(args, "ema_decay", 0.999),
                start_step=getattr(args, "ema_start_step", 100),
            ),
            precision=PrecisionConfig(
                mixed_precision=getattr(args, "mixed_precision", "auto"),
            ),
            path=PathConfig(
                mode=getattr(args, "path_mode", "independent"),
                solver=getattr(args, "path_solver", "hungarian"),
                layout_cost_resolution=getattr(args, "layout_cost_resolution", 16),
                condition_weight=getattr(args, "condition_weight", 1.0),
            ),
            distillation=DistillationConfig(
                enabled=getattr(args, "distillation_enabled", False),
                teacher_checkpoint=getattr(args, "distillation_teacher_checkpoint", None),
                teacher_model_type=(
                    getattr(args, "distillation_teacher_model_type", None)
                    or "sd15_regiondiff"
                ),
                loss_type=getattr(args, "distillation_loss_type", None) or "attention_kl",
                lambda_attn=(
                    getattr(args, "distillation_lambda_attn", None)
                    if getattr(args, "distillation_lambda_attn", None) is not None
                    else 0.05
                ),
                warmup_epochs=(
                    getattr(args, "distillation_warmup_epochs", None)
                    if getattr(args, "distillation_warmup_epochs", None) is not None
                    else 80
                ),
                selected_categories=getattr(args, "distillation_selected_categories", None) or [],
                selected_region_layers=getattr(args, "distillation_selected_region_layers", None) or [],
                timestep_range=tuple(
                    getattr(args, "distillation_timestep_range", None) or (0.2, 0.8)
                ),
            ),
            sampling=SampleConfig(
                sample_batch_size=args.sample_batch_size,
            ),
            output=OutputConfig(
                model_dir=args.model_dir,
                resume=args.resume,
            ),
            architecture_mode=getattr(args, "architecture_mode", "legacy"),
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
