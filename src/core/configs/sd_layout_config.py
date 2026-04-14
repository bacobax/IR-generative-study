"""Structured configuration for RegionDiff-style SD layout stage-2 training."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.algorithms.stable_diffusion.config import DEFAULT_PARTIAL_UNET_MODULES
from src.core.configs.config_loader import merge_config_and_cli
from src.core.data.dataset_targets import supported_dataset_ids
from src.core.paths import sd_layout_runs_dir, sd_unet_runs_dir


DEFAULT_STAGE1_DIR = str(sd_unet_runs_dir() / "flir_sd15_unet_full_stage1")
DEFAULT_OUTPUT_DIR = str(sd_layout_runs_dir() / "flir_sd15_regiondiff_stage2")


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


@dataclass
class SDLayoutDataConfig:
    """Dataset paths and loader settings."""

    dataset_id: str = "flir_private_proxy_alignment_v18"
    train_data_dir: Optional[str] = None
    train_annotations: Optional[str] = None
    val_data_dir: Optional[str] = None
    val_annotations: Optional[str] = None
    train_split: str = "train"
    val_split: str = "val"
    resolution: int = 512
    batch_size: int = 4
    num_workers: int = 4
    max_train_samples: Optional[int] = None
    max_val_samples: int = 8


@dataclass
class SDLayoutStage1Config:
    """Base stage-1 checkpoint configuration."""

    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None
    variant: Optional[str] = None
    stage1_dir: str = DEFAULT_STAGE1_DIR
    stage1_checkpoint: Optional[str] = "latest"


@dataclass
class SDLayoutPromptConfig:
    """Prompt-construction options."""

    prompt_mode: str = "class_list"
    constant_prompt: str = "thermal image"
    thermal_scene_suffix: str = "in thermal scene."
    use_captions_if_available: bool = False


@dataclass
class SDLayoutRegionConfig:
    """RegionDiff-style layout tokenization and region attention settings."""

    active_region_resolutions: List[int] = field(default_factory=lambda: [64, 32, 16])
    layout_token_dim: int = 256
    bbox_fourier_dim: int = 64
    same_class_position_slots: int = 128
    use_background_token: bool = True


@dataclass
class SDLayoutAreaLossConfig:
    """Area-based loss re-weighting settings."""

    enabled: bool = True
    alpha: float = 0.5
    background_weight: float = 0.5
    min_weight: float = 0.5
    max_weight: float = 4.0


@dataclass
class SDLayoutTrainingConfig:
    """Optimization, checkpointing, and trainability settings."""

    train_mode: str = "adapters_only"
    partial_unet_modules: List[str] = field(
        default_factory=lambda: list(DEFAULT_PARTIAL_UNET_MODULES)
    )
    adapter_learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    num_train_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    scale_lr: bool = False
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    snr_gamma: Optional[float] = None
    noise_offset: float = 0.0
    prediction_type: Optional[str] = None
    mixed_precision: Optional[str] = None
    allow_tf32: bool = False
    enable_xformers_memory_efficient_attention: bool = False
    checkpointing_steps: int = 1000
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    seed: Optional[int] = None
    report_to: str = "tensorboard"


@dataclass
class SDLayoutValidationConfig:
    """Validation sample generation settings."""

    validation_epochs: int = 1
    num_validation_images: int = 4
    validation_num_inference_steps: int = 30
    guidance_scale: float = 7.5


@dataclass
class SDLayoutOutputConfig:
    """Filesystem outputs."""

    output_dir: str = DEFAULT_OUTPUT_DIR
    logging_dir: str = "logs"


@dataclass
class SDLayoutTrainConfig:
    """Complete Stage-2 layout training configuration."""

    data: SDLayoutDataConfig = field(default_factory=SDLayoutDataConfig)
    stage1: SDLayoutStage1Config = field(default_factory=SDLayoutStage1Config)
    prompt: SDLayoutPromptConfig = field(default_factory=SDLayoutPromptConfig)
    region: SDLayoutRegionConfig = field(default_factory=SDLayoutRegionConfig)
    area_loss: SDLayoutAreaLossConfig = field(default_factory=SDLayoutAreaLossConfig)
    training: SDLayoutTrainingConfig = field(default_factory=SDLayoutTrainingConfig)
    validation: SDLayoutValidationConfig = field(default_factory=SDLayoutValidationConfig)
    output: SDLayoutOutputConfig = field(default_factory=SDLayoutOutputConfig)

    def validate(self) -> None:
        if self.data.dataset_id not in set(supported_dataset_ids()):
            raise ValueError(
                f"Unknown dataset_id={self.data.dataset_id!r}. "
                f"Expected one of: {', '.join(sorted(supported_dataset_ids()))}"
            )
        if self.prompt.prompt_mode not in {"constant", "class_list"}:
            raise ValueError(
                f"Unknown prompt_mode={self.prompt.prompt_mode!r}. "
                "Expected 'constant' or 'class_list'."
            )
        if self.training.train_mode not in {
            "adapters_only",
            "adapters_plus_partial_unet",
        }:
            raise ValueError(
                f"Unknown train_mode={self.training.train_mode!r}. "
                "Expected 'adapters_only' or 'adapters_plus_partial_unet'."
            )
        if (
            self.training.train_mode == "adapters_plus_partial_unet"
            and not self.training.partial_unet_modules
        ):
            raise ValueError(
                "partial_unet_modules must be non-empty when train_mode="
                "'adapters_plus_partial_unet'."
            )
        if not self.region.active_region_resolutions:
            raise ValueError("active_region_resolutions must be non-empty.")
        if self.region.layout_token_dim <= 0:
            raise ValueError("layout_token_dim must be positive.")
        if self.region.bbox_fourier_dim <= 0:
            raise ValueError("bbox_fourier_dim must be positive.")
        if self.region.same_class_position_slots <= 0:
            raise ValueError("same_class_position_slots must be positive.")
        if self.data.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.training.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive.")
        if self.training.max_train_steps is not None and self.training.max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive when set.")
        if self.training.checkpointing_steps <= 0:
            raise ValueError("checkpointing_steps must be positive.")
        if self.validation.num_validation_images <= 0:
            raise ValueError("num_validation_images must be positive.")
        if self.validation.validation_num_inference_steps <= 0:
            raise ValueError("validation_num_inference_steps must be positive.")
        if self.output.output_dir is None or not str(self.output.output_dir).strip():
            raise ValueError("output_dir must be set.")


_FLAT_TO_NESTED = {
    "dataset_id": "data.dataset_id",
    "train_data_dir": "data.train_data_dir",
    "train_annotations": "data.train_annotations",
    "val_data_dir": "data.val_data_dir",
    "val_annotations": "data.val_annotations",
    "train_split": "data.train_split",
    "val_split": "data.val_split",
    "resolution": "data.resolution",
    "batch_size": "data.batch_size",
    "num_workers": "data.num_workers",
    "max_train_samples": "data.max_train_samples",
    "max_val_samples": "data.max_val_samples",
    "pretrained_model_name_or_path": "stage1.pretrained_model_name_or_path",
    "revision": "stage1.revision",
    "variant": "stage1.variant",
    "stage1_dir": "stage1.stage1_dir",
    "stage1_checkpoint": "stage1.stage1_checkpoint",
    "prompt_mode": "prompt.prompt_mode",
    "constant_prompt": "prompt.constant_prompt",
    "thermal_scene_suffix": "prompt.thermal_scene_suffix",
    "use_captions_if_available": "prompt.use_captions_if_available",
    "active_region_resolutions": "region.active_region_resolutions",
    "layout_token_dim": "region.layout_token_dim",
    "bbox_fourier_dim": "region.bbox_fourier_dim",
    "same_class_position_slots": "region.same_class_position_slots",
    "use_background_token": "region.use_background_token",
    "area_loss_enabled": "area_loss.enabled",
    "area_loss_alpha": "area_loss.alpha",
    "area_loss_background_weight": "area_loss.background_weight",
    "area_loss_min_weight": "area_loss.min_weight",
    "area_loss_max_weight": "area_loss.max_weight",
    "train_mode": "training.train_mode",
    "partial_unet_modules": "training.partial_unet_modules",
    "adapter_learning_rate": "training.adapter_learning_rate",
    "backbone_learning_rate": "training.backbone_learning_rate",
    "num_train_epochs": "training.num_train_epochs",
    "max_train_steps": "training.max_train_steps",
    "gradient_accumulation_steps": "training.gradient_accumulation_steps",
    "gradient_checkpointing": "training.gradient_checkpointing",
    "lr_scheduler": "training.lr_scheduler",
    "lr_warmup_steps": "training.lr_warmup_steps",
    "scale_lr": "training.scale_lr",
    "use_8bit_adam": "training.use_8bit_adam",
    "adam_beta1": "training.adam_beta1",
    "adam_beta2": "training.adam_beta2",
    "adam_weight_decay": "training.adam_weight_decay",
    "adam_epsilon": "training.adam_epsilon",
    "max_grad_norm": "training.max_grad_norm",
    "snr_gamma": "training.snr_gamma",
    "noise_offset": "training.noise_offset",
    "prediction_type": "training.prediction_type",
    "mixed_precision": "training.mixed_precision",
    "allow_tf32": "training.allow_tf32",
    "enable_xformers_memory_efficient_attention": "training.enable_xformers_memory_efficient_attention",
    "checkpointing_steps": "training.checkpointing_steps",
    "checkpoints_total_limit": "training.checkpoints_total_limit",
    "resume_from_checkpoint": "training.resume_from_checkpoint",
    "seed": "training.seed",
    "report_to": "training.report_to",
    "validation_epochs": "validation.validation_epochs",
    "num_validation_images": "validation.num_validation_images",
    "validation_num_inference_steps": "validation.validation_num_inference_steps",
    "guidance_scale": "validation.guidance_scale",
    "output_dir": "output.output_dir",
    "logging_dir": "output.logging_dir",
}


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for Stage-2 SD layout training."""
    parser = argparse.ArgumentParser(
        description="RegionDiff-style layout-conditioned SD1.5 stage-2 training.",
    )
    parser.add_argument("--config", type=str, default=None)

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--dataset_id", type=str, default=SDLayoutDataConfig.dataset_id)
    data_group.add_argument("--train_data_dir", type=str, default=None)
    data_group.add_argument("--train_annotations", type=str, default=None)
    data_group.add_argument("--val_data_dir", type=str, default=None)
    data_group.add_argument("--val_annotations", type=str, default=None)
    data_group.add_argument("--train_split", type=str, default=SDLayoutDataConfig.train_split)
    data_group.add_argument("--val_split", type=str, default=SDLayoutDataConfig.val_split)
    data_group.add_argument("--resolution", type=int, default=SDLayoutDataConfig.resolution)
    data_group.add_argument("--batch_size", type=int, default=SDLayoutDataConfig.batch_size)
    data_group.add_argument("--num_workers", type=int, default=SDLayoutDataConfig.num_workers)
    data_group.add_argument("--max_train_samples", type=int, default=None)
    data_group.add_argument("--max_val_samples", type=int, default=SDLayoutDataConfig.max_val_samples)

    stage1_group = parser.add_argument_group("Stage-1")
    stage1_group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=SDLayoutStage1Config.pretrained_model_name_or_path,
    )
    stage1_group.add_argument("--revision", type=str, default=None)
    stage1_group.add_argument("--variant", type=str, default=None)
    stage1_group.add_argument("--stage1_dir", type=str, default=DEFAULT_STAGE1_DIR)
    stage1_group.add_argument("--stage1_checkpoint", type=str, default="latest")

    prompt_group = parser.add_argument_group("Prompt")
    prompt_group.add_argument("--prompt_mode", type=str, default=SDLayoutPromptConfig.prompt_mode)
    prompt_group.add_argument("--constant_prompt", type=str, default=SDLayoutPromptConfig.constant_prompt)
    prompt_group.add_argument(
        "--thermal_scene_suffix",
        type=str,
        default=SDLayoutPromptConfig.thermal_scene_suffix,
    )
    prompt_group.add_argument(
        "--use_captions_if_available",
        type=_str2bool,
        default=SDLayoutPromptConfig.use_captions_if_available,
    )

    region_group = parser.add_argument_group("RegionDiff")
    region_group.add_argument(
        "--active_region_resolutions",
        type=_csv_ints,
        default=list(SDLayoutRegionConfig().active_region_resolutions),
        help="Comma-separated latent resolutions, e.g. 64,32,16",
    )
    region_group.add_argument("--layout_token_dim", type=int, default=SDLayoutRegionConfig.layout_token_dim)
    region_group.add_argument("--bbox_fourier_dim", type=int, default=SDLayoutRegionConfig.bbox_fourier_dim)
    region_group.add_argument(
        "--same_class_position_slots",
        type=int,
        default=SDLayoutRegionConfig.same_class_position_slots,
    )
    region_group.add_argument(
        "--use_background_token",
        type=_str2bool,
        default=SDLayoutRegionConfig.use_background_token,
    )

    area_group = parser.add_argument_group("Area Loss")
    area_group.add_argument("--area_loss_enabled", type=_str2bool, default=SDLayoutAreaLossConfig.enabled)
    area_group.add_argument("--area_loss_alpha", type=float, default=SDLayoutAreaLossConfig.alpha)
    area_group.add_argument(
        "--area_loss_background_weight",
        type=float,
        default=SDLayoutAreaLossConfig.background_weight,
    )
    area_group.add_argument("--area_loss_min_weight", type=float, default=SDLayoutAreaLossConfig.min_weight)
    area_group.add_argument("--area_loss_max_weight", type=float, default=SDLayoutAreaLossConfig.max_weight)

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--train_mode", type=str, default=SDLayoutTrainingConfig.train_mode)
    train_group.add_argument(
        "--partial_unet_modules",
        nargs="+",
        default=list(SDLayoutTrainingConfig().partial_unet_modules),
    )
    train_group.add_argument(
        "--adapter_learning_rate",
        type=float,
        default=SDLayoutTrainingConfig.adapter_learning_rate,
    )
    train_group.add_argument(
        "--backbone_learning_rate",
        type=float,
        default=SDLayoutTrainingConfig.backbone_learning_rate,
    )
    train_group.add_argument("--num_train_epochs", type=int, default=SDLayoutTrainingConfig.num_train_epochs)
    train_group.add_argument("--max_train_steps", type=int, default=None)
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=SDLayoutTrainingConfig.gradient_accumulation_steps,
    )
    train_group.add_argument(
        "--gradient_checkpointing",
        type=_str2bool,
        default=SDLayoutTrainingConfig.gradient_checkpointing,
    )
    train_group.add_argument("--lr_scheduler", type=str, default=SDLayoutTrainingConfig.lr_scheduler)
    train_group.add_argument("--lr_warmup_steps", type=int, default=SDLayoutTrainingConfig.lr_warmup_steps)
    train_group.add_argument("--scale_lr", type=_str2bool, default=SDLayoutTrainingConfig.scale_lr)
    train_group.add_argument("--use_8bit_adam", type=_str2bool, default=SDLayoutTrainingConfig.use_8bit_adam)
    train_group.add_argument("--adam_beta1", type=float, default=SDLayoutTrainingConfig.adam_beta1)
    train_group.add_argument("--adam_beta2", type=float, default=SDLayoutTrainingConfig.adam_beta2)
    train_group.add_argument("--adam_weight_decay", type=float, default=SDLayoutTrainingConfig.adam_weight_decay)
    train_group.add_argument("--adam_epsilon", type=float, default=SDLayoutTrainingConfig.adam_epsilon)
    train_group.add_argument("--max_grad_norm", type=float, default=SDLayoutTrainingConfig.max_grad_norm)
    train_group.add_argument("--snr_gamma", type=float, default=None)
    train_group.add_argument("--noise_offset", type=float, default=SDLayoutTrainingConfig.noise_offset)
    train_group.add_argument("--prediction_type", type=str, default=None)
    train_group.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    train_group.add_argument("--allow_tf32", type=_str2bool, default=SDLayoutTrainingConfig.allow_tf32)
    train_group.add_argument(
        "--enable_xformers_memory_efficient_attention",
        type=_str2bool,
        default=SDLayoutTrainingConfig.enable_xformers_memory_efficient_attention,
    )
    train_group.add_argument(
        "--checkpointing_steps",
        type=int,
        default=SDLayoutTrainingConfig.checkpointing_steps,
    )
    train_group.add_argument("--checkpoints_total_limit", type=int, default=None)
    train_group.add_argument("--resume_from_checkpoint", type=str, default=None)
    train_group.add_argument("--seed", type=int, default=None)
    train_group.add_argument("--report_to", type=str, default=SDLayoutTrainingConfig.report_to)

    validation_group = parser.add_argument_group("Validation")
    validation_group.add_argument(
        "--validation_epochs",
        type=int,
        default=SDLayoutValidationConfig.validation_epochs,
    )
    validation_group.add_argument(
        "--num_validation_images",
        type=int,
        default=SDLayoutValidationConfig.num_validation_images,
    )
    validation_group.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=SDLayoutValidationConfig.validation_num_inference_steps,
    )
    validation_group.add_argument(
        "--guidance_scale",
        type=float,
        default=SDLayoutValidationConfig.guidance_scale,
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    output_group.add_argument("--logging_dir", type=str, default=SDLayoutOutputConfig.logging_dir)

    return parser


def parse_args(argv: Optional[List[str]] = None) -> SDLayoutTrainConfig:
    """Parse CLI arguments into a validated structured config."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config = merge_config_and_cli(
        SDLayoutTrainConfig,
        args.config,
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
    )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        _ = env_local_rank  # kept for symmetry with the stage-1 parser

    config.validate()
    return config

