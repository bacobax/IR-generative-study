#!/usr/bin/env python
# coding=utf-8
"""Configuration module for Stage-1 Stable Diffusion IR adaptation."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from src.core.configs.config_loader import apply_yaml_defaults, load_yaml
from src.core.data.dataset_targets import supported_dataset_ids


DEFAULT_PROMPT_TEXT = "thermal image"
LEGACY_GENERIC_PROMPT = "overhead infrared surveillance image with any people or objects"
DEFAULT_LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"]
DEFAULT_PARTIAL_UNET_MODULES = ["mid_block", "up_blocks"]
DEFAULT_NUM_TRAIN_EPOCHS = 80


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


@dataclass
class TrainingConfig:
    """Configuration dataclass for SD IR adaptation training."""

    # Model configuration
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    revision: Optional[str] = None
    variant: Optional[str] = None

    # Adaptation mode
    baseline_mode: str = "sd_ir_lora"
    unet_train_mode: str = "full"
    unet_trainable_modules: List[str] = field(
        default_factory=lambda: list(DEFAULT_PARTIAL_UNET_MODULES)
    )

    # Dataset configuration
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_data_dir: Optional[str] = None
    train_split: str = "train"
    image_column: str = "image"
    caption_column: str = "text"
    prompt_text: Optional[str] = DEFAULT_PROMPT_TEXT
    generic_prompt: bool = False
    max_train_samples: Optional[int] = None
    cache_dir: Optional[str] = None

    # Image preprocessing
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    image_interpolation_mode: str = "lanczos"
    use_ir_preprocessing: bool = True

    # Train/freeze toggles
    freeze_vae: bool = True
    freeze_text_encoder: bool = True

    # Training hyperparameters
    train_batch_size: int = 16
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500

    # LoRA configuration
    rank: int = 4
    lora_alpha_scale: float = 1.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )

    # RegionDiff layout conditioning (opt-in)
    layout_conditioning_enabled: bool = False
    layout_conditioning_variant: str = "regiondiff_v1"
    layout_annotations_path: Optional[str] = None
    active_region_resolutions: List[int] = field(default_factory=lambda: [64, 32, 16])
    layout_token_dim: int = 256
    bbox_fourier_dim: int = 64
    same_class_position_slots: int = 128
    use_background_token: bool = True
    layout_train_mode: str = "adapters_only"
    partial_backbone_modules: List[str] = field(default_factory=lambda: list(DEFAULT_PARTIAL_UNET_MODULES))
    adapter_learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    layout_category_id_to_name: Optional[dict] = None

    # Optimizer configuration
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0

    # Loss configuration
    snr_gamma: Optional[float] = None
    noise_offset: float = 0.0
    prediction_type: Optional[str] = None
    domainstudio_enabled: bool = False
    domainstudio_source_prompt: Optional[str] = None
    domainstudio_target_prompt: Optional[str] = None
    domainstudio_lambda_prior: float = 1.0
    domainstudio_lambda_img: float = 100.0
    domainstudio_lambda_hf: float = 100.0
    domainstudio_lambda_hfmse: float = 0.5
    domainstudio_similarity_temperature: float = 1.0
    domainstudio_loss_every_n_steps: int = 1
    domainstudio_min_pairwise_batch: int = 2
    domainstudio_prior_data_dir: Optional[str] = None
    domainstudio_prior_cache_dir: Optional[str] = None
    domainstudio_num_prior_images: int = 200
    domainstudio_generate_prior_if_missing: bool = False

    # Validation configuration
    validation_prompt: Optional[str] = None
    num_validation_images: int = 4
    validation_epochs: int = 1
    validation_num_inference_steps: int = 30

    # Output and logging
    output_dir: str = "sd-model-finetuned-lora"
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    seed: Optional[int] = None

    # Checkpointing
    checkpointing_epochs: int = 1
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    # Hub configuration
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

    # Performance optimization
    mixed_precision: Optional[str] = None
    allow_tf32: bool = False
    enable_xformers_memory_efficient_attention: bool = False
    dataloader_num_workers: int = 0

    # Distributed training
    local_rank: int = -1

    def resolved_prompt_text(self) -> Optional[str]:
        """Return the effective constant prompt, if any."""
        if self.generic_prompt:
            return LEGACY_GENERIC_PROMPT
        if self.prompt_text:
            return self.prompt_text
        return None

    def resolved_training_prompt_text(self) -> Optional[str]:
        """Return the prompt used by the target training dataloader."""
        if self.domainstudio_target_prompt:
            return self.domainstudio_target_prompt
        return self.resolved_prompt_text()

    def validate(self):
        """Validate configuration parameters."""
        if self.dataset_name is None and self.train_data_dir is None and self.dataset_id is None:
            raise ValueError("Need either dataset_id, dataset_name, or train_data_dir.")

        if self.dataset_id is not None and self.dataset_id not in set(supported_dataset_ids()):
            raise ValueError(
                f"Unknown dataset_id={self.dataset_id!r}. "
                f"Expected one of: {', '.join(sorted(supported_dataset_ids()))}"
            )

        if self.baseline_mode not in {"sd_ir_lora", "sd_ir_unet"}:
            raise ValueError(
                f"Unknown baseline_mode={self.baseline_mode!r}. "
                "Expected 'sd_ir_lora' or 'sd_ir_unet'."
            )

        if not (0.0 <= self.lora_alpha_scale <= 1.0):
            raise ValueError("--lora_alpha_scale must be between 0 and 1.")

        if self.baseline_mode == "sd_ir_lora":
            if not self.freeze_vae:
                raise ValueError("LoRA baseline requires freeze_vae=True.")
            if not self.freeze_text_encoder:
                raise ValueError("LoRA baseline requires freeze_text_encoder=True.")
            if self.rank <= 0:
                raise ValueError("--rank must be positive for LoRA baseline.")
            if not self.lora_target_modules:
                raise ValueError("LoRA baseline requires at least one lora_target_modules entry.")
        else:
            if self.unet_train_mode not in {"full", "partial"}:
                raise ValueError(
                    f"Unknown unet_train_mode={self.unet_train_mode!r}. "
                    "Expected 'full' or 'partial'."
                )
            if self.unet_train_mode == "partial" and not self.unet_trainable_modules:
                raise ValueError(
                    "Partial U-Net baseline requires unet_trainable_modules to be non-empty."
                )

        if self.layout_conditioning_enabled:
            if self.layout_conditioning_variant != "regiondiff_v1":
                raise ValueError(
                    "Only layout_conditioning_variant='regiondiff_v1' is supported for SD1.5."
                )
            if not self.active_region_resolutions:
                raise ValueError("active_region_resolutions must be non-empty when layout conditioning is enabled.")
            if self.layout_train_mode not in {"adapters_only", "adapters_plus_partial_unet"}:
                raise ValueError(
                    "layout_train_mode must be 'adapters_only' or 'adapters_plus_partial_unet'."
                )
            if self.layout_train_mode == "adapters_plus_partial_unet" and not self.partial_backbone_modules:
                raise ValueError(
                    "partial_backbone_modules must be non-empty for adapters_plus_partial_unet."
                )
            if self.dataset_id is None and self.layout_annotations_path is None:
                raise ValueError(
                    "layout_annotations_path is required for SD1.5 layout conditioning "
                    "when dataset_id is not set."
                )

        if self.domainstudio_enabled:
            if self.layout_conditioning_enabled:
                raise ValueError(
                    "DomainStudio Stage-1 losses are not currently supported with "
                    "layout_conditioning_enabled=True because source-prior batches "
                    "do not carry RegionDiff layout annotations."
                )
            if not self.domainstudio_source_prompt:
                raise ValueError(
                    "domainstudio_source_prompt is required when domainstudio_enabled=True."
                )
            if not self.freeze_vae:
                raise ValueError("DomainStudio requires freeze_vae=True.")
            if not self.freeze_text_encoder:
                raise ValueError("DomainStudio requires freeze_text_encoder=True.")
            if self.prediction_type not in {None, "epsilon"}:
                raise NotImplementedError(
                    "DomainStudio Stage-1 losses currently support only epsilon prediction."
                )
            if self.domainstudio_similarity_temperature <= 0:
                raise ValueError("domainstudio_similarity_temperature must be positive.")
            if self.domainstudio_loss_every_n_steps <= 0:
                raise ValueError("domainstudio_loss_every_n_steps must be positive.")
            if self.domainstudio_min_pairwise_batch <= 0:
                raise ValueError("domainstudio_min_pairwise_batch must be positive.")
            if self.domainstudio_num_prior_images <= 0:
                raise ValueError("domainstudio_num_prior_images must be positive.")
            if self.domainstudio_generate_prior_if_missing and not self.domainstudio_prior_cache_dir:
                raise ValueError(
                    "domainstudio_prior_cache_dir is required when "
                    "domainstudio_generate_prior_if_missing=True."
                )

        if self.report_to == "wandb" and self.hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk."
                " Please use `hf auth login` to authenticate with the Hub."
            )

        if self.num_train_epochs <= 0:
            raise ValueError("--num_train_epochs must be positive.")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("--max_train_steps must be positive when set.")
        if self.lr_warmup_steps < 0:
            raise ValueError("--lr_warmup_steps must be >= 0.")
        if self.num_validation_images <= 0:
            raise ValueError("--num_validation_images must be positive.")
        if self.validation_epochs <= 0:
            raise ValueError("--validation_epochs must be positive.")
        if self.validation_num_inference_steps <= 0:
            raise ValueError("--validation_num_inference_steps must be positive.")
        if self.checkpointing_epochs <= 0:
            raise ValueError("--checkpointing_epochs must be positive.")


def _validate_yaml_config_keys(
    parser: argparse.ArgumentParser,
    config_path: str | Path | None,
) -> None:
    """Fail fast when a SD YAML config contains unsupported keys."""
    if not config_path:
        return

    data = load_yaml(config_path)
    if not data:
        return
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in SD config {config_path!s}.")

    known_dests = {action.dest for action in parser._actions}
    unknown = sorted(key for key in data if key not in known_dests)
    if unknown:
        unknown_list = ", ".join(unknown)
        raise ValueError(
            f"Unknown keys in SD config {config_path}: {unknown_list}. "
            "Remove them or add matching argparse/config fields."
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for SD IR adaptation training."""
    parser = argparse.ArgumentParser(
        description="Stage-1 Stable Diffusion IR adaptation training."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. CLI flags override config values.",
    )

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    model_group.add_argument("--revision", type=str, default=None)
    model_group.add_argument("--variant", type=str, default=None)
    model_group.add_argument(
        "--baseline_mode",
        type=str,
        default="sd_ir_lora",
        choices=["sd_ir_lora", "sd_ir_unet"],
        help="Stage-1 adaptation baseline strategy.",
    )
    model_group.add_argument(
        "--unet_train_mode",
        type=str,
        default="full",
        choices=["full", "partial"],
        help="How strongly to adapt the U-Net when baseline_mode=sd_ir_unet.",
    )
    model_group.add_argument(
        "--unet_trainable_modules",
        nargs="+",
        default=list(DEFAULT_PARTIAL_UNET_MODULES),
        help="Module-name prefixes to train when using partial U-Net adaptation.",
    )
    model_group.add_argument(
        "--freeze_vae",
        type=_str2bool,
        default=True,
        help="Whether to keep the VAE frozen.",
    )
    model_group.add_argument(
        "--freeze_text_encoder",
        type=_str2bool,
        default=True,
        help="Whether to keep the text encoder frozen.",
    )

    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        choices=sorted(supported_dataset_ids()),
        help="Named repo dataset target.",
    )
    data_group.add_argument("--dataset_name", type=str, default=None)
    data_group.add_argument("--dataset_config_name", type=str, default=None)
    data_group.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    data_group.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Dataset split to use when dataset_id resolves a repo dataset root.",
    )
    data_group.add_argument("--image_column", type=str, default="image")
    data_group.add_argument("--caption_column", type=str, default="text")
    data_group.add_argument(
        "--prompt_text",
        type=str,
        default=DEFAULT_PROMPT_TEXT,
        help="Constant prompt text used for all samples when set.",
    )
    data_group.add_argument(
        "--generic_prompt",
        action="store_true",
        help="Use the legacy fixed generic surveillance prompt.",
    )
    data_group.add_argument("--max_train_samples", type=int, default=None)
    data_group.add_argument("--cache_dir", type=str, default=None)

    preprocess_group = parser.add_argument_group("Image Preprocessing")
    preprocess_group.add_argument("--resolution", type=int, default=512)
    preprocess_group.add_argument("--center_crop", action="store_true")
    preprocess_group.add_argument("--random_flip", action="store_true")
    preprocess_group.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=["nearest", "nearest_exact", "box", "bilinear", "hamming", "bicubic", "lanczos"],
    )
    preprocess_group.add_argument(
        "--use_ir_preprocessing",
        type=_str2bool,
        default=True,
        help="Apply repo-native IR preprocessing for local .npy datasets.",
    )

    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--train_batch_size", type=int, default=16)
    train_group.add_argument("--num_train_epochs", type=int, default=DEFAULT_NUM_TRAIN_EPOCHS)
    train_group.add_argument("--max_train_steps", type=int, default=None)
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train_group.add_argument("--gradient_checkpointing", action="store_true")
    train_group.add_argument("--learning_rate", type=float, default=1e-4)
    train_group.add_argument("--scale_lr", action="store_true")
    train_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    train_group.add_argument("--lr_warmup_steps", type=int, default=500)

    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument("--rank", type=int, default=4)
    lora_group.add_argument("--lora_alpha_scale", type=float, default=1.0)
    lora_group.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=list(DEFAULT_LORA_TARGET_MODULES),
        help="Target U-Net module names for LoRA injection.",
    )

    layout_group = parser.add_argument_group("RegionDiff Layout Conditioning")
    layout_group.add_argument("--layout_conditioning_enabled", type=_str2bool, default=False)
    layout_group.add_argument("--layout_conditioning_variant", type=str, default="regiondiff_v1")
    layout_group.add_argument("--layout_annotations_path", type=str, default=None)
    layout_group.add_argument(
        "--active_region_resolutions",
        type=lambda value: [int(item.strip()) for item in str(value).split(",") if item.strip()],
        default=[64, 32, 16],
    )
    layout_group.add_argument("--layout_token_dim", type=int, default=256)
    layout_group.add_argument("--bbox_fourier_dim", type=int, default=64)
    layout_group.add_argument("--same_class_position_slots", type=int, default=128)
    layout_group.add_argument("--use_background_token", type=_str2bool, default=True)
    layout_group.add_argument(
        "--layout_train_mode",
        type=str,
        default="adapters_only",
        choices=["adapters_only", "adapters_plus_partial_unet"],
    )
    layout_group.add_argument(
        "--partial_backbone_modules",
        nargs="+",
        default=list(DEFAULT_PARTIAL_UNET_MODULES),
    )
    layout_group.add_argument("--adapter_learning_rate", type=float, default=1e-4)
    layout_group.add_argument("--backbone_learning_rate", type=float, default=1e-5)

    optim_group = parser.add_argument_group("Optimizer Configuration")
    optim_group.add_argument("--use_8bit_adam", action="store_true")
    optim_group.add_argument("--adam_beta1", type=float, default=0.9)
    optim_group.add_argument("--adam_beta2", type=float, default=0.999)
    optim_group.add_argument("--adam_weight_decay", type=float, default=1e-2)
    optim_group.add_argument("--adam_epsilon", type=float, default=1e-08)
    optim_group.add_argument("--max_grad_norm", type=float, default=1.0)

    loss_group = parser.add_argument_group("Loss Configuration")
    loss_group.add_argument("--snr_gamma", type=float, default=None)
    loss_group.add_argument("--noise_offset", type=float, default=0.0)
    loss_group.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        choices=["epsilon", "v_prediction", None],
    )
    loss_group.add_argument("--domainstudio_enabled", type=_str2bool, default=False)
    loss_group.add_argument("--domainstudio_source_prompt", type=str, default=None)
    loss_group.add_argument("--domainstudio_target_prompt", type=str, default=None)
    loss_group.add_argument("--domainstudio_lambda_prior", type=float, default=1.0)
    loss_group.add_argument("--domainstudio_lambda_img", type=float, default=100.0)
    loss_group.add_argument("--domainstudio_lambda_hf", type=float, default=100.0)
    loss_group.add_argument("--domainstudio_lambda_hfmse", type=float, default=0.5)
    loss_group.add_argument("--domainstudio_similarity_temperature", type=float, default=1.0)
    loss_group.add_argument("--domainstudio_loss_every_n_steps", type=int, default=1)
    loss_group.add_argument("--domainstudio_min_pairwise_batch", type=int, default=2)
    loss_group.add_argument("--domainstudio_prior_data_dir", type=str, default=None)
    loss_group.add_argument("--domainstudio_prior_cache_dir", type=str, default=None)
    loss_group.add_argument("--domainstudio_num_prior_images", type=int, default=200)
    loss_group.add_argument("--domainstudio_generate_prior_if_missing", type=_str2bool, default=False)

    val_group = parser.add_argument_group("Validation Configuration")
    val_group.add_argument("--validation_prompt", type=str, default=None)
    val_group.add_argument("--num_validation_images", type=int, default=4)
    val_group.add_argument("--validation_epochs", type=int, default=1)
    val_group.add_argument("--validation_num_inference_steps", type=int, default=30)

    output_group = parser.add_argument_group("Output and Logging")
    output_group.add_argument("--output_dir", type=str, default="sd-model-finetuned-lora")
    output_group.add_argument("--logging_dir", type=str, default="logs")
    output_group.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "comet_ml", "all"],
    )
    output_group.add_argument("--seed", type=int, default=None)

    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument("--checkpointing_epochs", type=int, default=1)
    ckpt_group.add_argument("--checkpoints_total_limit", type=int, default=None)
    ckpt_group.add_argument("--resume_from_checkpoint", type=str, default=None)

    hub_group = parser.add_argument_group("Hugging Face Hub")
    hub_group.add_argument("--push_to_hub", action="store_true")
    hub_group.add_argument("--hub_token", type=str, default=None)
    hub_group.add_argument("--hub_model_id", type=str, default=None)

    perf_group = parser.add_argument_group("Performance Optimization")
    perf_group.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    perf_group.add_argument("--allow_tf32", action="store_true")
    perf_group.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    perf_group.add_argument("--dataloader_num_workers", type=int, default=0)

    dist_group = parser.add_argument_group("Distributed Training")
    dist_group.add_argument("--local_rank", type=int, default=-1)

    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    """Parse CLI arguments into a validated TrainingConfig."""
    parser = build_parser()

    preliminary, _ = parser.parse_known_args(argv)
    _validate_yaml_config_keys(parser, preliminary.config)
    apply_yaml_defaults(parser, preliminary.config)
    args = parser.parse_args(argv)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    data = vars(args).copy()
    data.pop("config", None)
    config = TrainingConfig(**data)
    config.validate()
    return config
