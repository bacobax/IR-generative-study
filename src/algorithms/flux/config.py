#!/usr/bin/env python
# coding=utf-8
"""Configuration for FLUX.1-dev QLoRA fine-tuning."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from src.algorithms.stable_diffusion.config import _str2bool
from src.core.configs.config_loader import apply_yaml_defaults, load_yaml
from src.core.data.dataset_targets import supported_dataset_ids


DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_FLUX_LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
DEFAULT_NUM_TRAIN_EPOCHS = 100


@dataclass
class TrainingConfig:
    """Configuration dataclass for FLUX.1-dev QLoRA LoRA fine-tuning."""

    # Model
    pretrained_model_name_or_path: str = DEFAULT_FLUX_MODEL

    # Dataset
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_data_dir: Optional[str] = None
    train_split: str = "train"
    image_column: str = "image"
    caption_column: str = "text"
    prompt_text: str = "image"
    max_train_samples: Optional[int] = None
    subset_manifest: Optional[str] = None
    cache_dir: Optional[str] = None

    # Image preprocessing
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    image_interpolation_mode: str = "lanczos"
    use_ir_preprocessing: bool = True

    # Training hyperparameters
    train_batch_size: int = 1
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0

    # QLoRA / quantization
    quantize_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    cache_latents: bool = True

    # LoRA
    rank: int = 8
    lora_alpha_scale: float = 1.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: list(DEFAULT_FLUX_LORA_TARGET_MODULES)
    )

    # FLUX-specific
    guidance_scale: float = 1.0
    weighting_scheme: str = "none"
    max_sequence_length: int = 512

    # Optimizer
    use_8bit_adam: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Validation
    validation_prompt: Optional[str] = None
    num_validation_images: int = 4
    validation_epochs: int = 1
    validation_num_inference_steps: int = 28

    # Output and logging
    output_dir: str = "flux-model-finetuned-lora"
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    seed: Optional[int] = None

    # Checkpointing
    checkpointing_epochs: int = 1
    checkpoints_total_limit: Optional[int] = None
    save_optimizer_state: bool = True
    resume_from_checkpoint: Optional[str] = None

    # Hub
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

    # Performance
    mixed_precision: Optional[str] = None
    allow_tf32: bool = False
    dataloader_num_workers: int = 0
    local_rank: int = -1

    def accelerator_mixed_precision(self) -> Optional[str]:
        if self.mixed_precision in {None, "no", "fp32"}:
            return None
        return self.mixed_precision

    def validate(self) -> None:
        if self.dataset_name is None and self.train_data_dir is None and self.dataset_id is None:
            raise ValueError("Need either dataset_id, dataset_name, or train_data_dir.")
        if self.subset_manifest is not None and self.dataset_name is not None:
            raise ValueError(
                "subset_manifest is only supported for local repo datasets, "
                "not Hugging Face dataset_name inputs."
            )
        if self.dataset_id is not None and self.dataset_id not in set(supported_dataset_ids()):
            raise ValueError(
                f"Unknown dataset_id={self.dataset_id!r}. "
                f"Expected one of: {', '.join(sorted(supported_dataset_ids()))}"
            )
        if self.rank <= 0:
            raise ValueError("--rank must be positive for LoRA training.")
        if not (0.0 <= self.lora_alpha_scale <= 1.0):
            raise ValueError("--lora_alpha_scale must be between 0 and 1.")
        if not self.lora_target_modules:
            raise ValueError("FLUX LoRA training requires lora_target_modules.")
        if self.resolution <= 0:
            raise ValueError("--resolution must be positive.")
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
        if self.guidance_scale <= 0:
            raise ValueError("--guidance_scale must be positive.")
        if self.max_sequence_length <= 0:
            raise ValueError("--max_sequence_length must be positive.")
        if self.bnb_4bit_quant_type not in {"nf4", "fp4"}:
            raise ValueError("--bnb_4bit_quant_type must be 'nf4' or 'fp4'.")
        if self.weighting_scheme not in {"sigma_sqrt", "logit_normal", "mode", "cosmap", "none"}:
            raise ValueError(
                "--weighting_scheme must be one of: sigma_sqrt, logit_normal, mode, cosmap, none."
            )
        if self.report_to == "wandb" and self.hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk."
            )


def _validate_yaml_config_keys(parser: argparse.ArgumentParser, config_path: str | Path | None) -> None:
    if not config_path:
        return
    data = load_yaml(config_path)
    if not data:
        return
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in FLUX config {config_path!s}.")
    known_dests = {action.dest for action in parser._actions}
    unknown = sorted(key for key in data if key not in known_dests)
    if unknown:
        raise ValueError(
            f"Unknown keys in FLUX config {config_path}: {', '.join(unknown)}. "
            "Remove them or add matching argparse/config fields."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FLUX.1-dev QLoRA LoRA fine-tuning.")
    parser.add_argument("--config", type=str, default=None)

    model = parser.add_argument_group("Model Configuration")
    model.add_argument("--pretrained_model_name_or_path", type=str, default=DEFAULT_FLUX_MODEL)

    data = parser.add_argument_group("Dataset Configuration")
    data.add_argument("--dataset_id", type=str, default=None, choices=sorted(supported_dataset_ids()))
    data.add_argument("--dataset_name", type=str, default=None)
    data.add_argument("--dataset_config_name", type=str, default=None)
    data.add_argument("--train_data_dir", type=str, default=None)
    data.add_argument("--train_split", type=str, default="train")
    data.add_argument("--image_column", type=str, default="image")
    data.add_argument("--caption_column", type=str, default="text")
    data.add_argument("--prompt_text", type=str, default="image")
    data.add_argument("--max_train_samples", type=int, default=None)
    data.add_argument("--subset_manifest", type=str, default=None)
    data.add_argument("--cache_dir", type=str, default=None)

    preprocess = parser.add_argument_group("Image Preprocessing")
    preprocess.add_argument("--resolution", type=int, default=512)
    preprocess.add_argument("--center_crop", action="store_true")
    preprocess.add_argument("--random_flip", action="store_true")
    preprocess.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=["nearest", "nearest_exact", "box", "bilinear", "hamming", "bicubic", "lanczos"],
    )
    preprocess.add_argument("--use_ir_preprocessing", type=_str2bool, default=True)

    train = parser.add_argument_group("Training Hyperparameters")
    train.add_argument("--train_batch_size", type=int, default=1)
    train.add_argument("--num_train_epochs", type=int, default=DEFAULT_NUM_TRAIN_EPOCHS)
    train.add_argument("--max_train_steps", type=int, default=None)
    train.add_argument("--gradient_accumulation_steps", type=int, default=4)
    train.add_argument("--gradient_checkpointing", type=_str2bool, default=True)
    train.add_argument("--learning_rate", type=float, default=1e-4)
    train.add_argument("--scale_lr", action="store_true")
    train.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    train.add_argument("--lr_warmup_steps", type=int, default=0)

    qlora = parser.add_argument_group("QLoRA / Quantization")
    qlora.add_argument("--quantize_4bit", type=_str2bool, default=True)
    qlora.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    qlora.add_argument("--cache_latents", type=_str2bool, default=True)

    lora = parser.add_argument_group("LoRA Configuration")
    lora.add_argument("--rank", type=int, default=8)
    lora.add_argument("--lora_alpha_scale", type=float, default=1.0)
    lora.add_argument("--lora_target_modules", nargs="+", default=list(DEFAULT_FLUX_LORA_TARGET_MODULES))

    flux = parser.add_argument_group("FLUX-specific")
    flux.add_argument("--guidance_scale", type=float, default=1.0)
    flux.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    flux.add_argument("--max_sequence_length", type=int, default=512)

    optim = parser.add_argument_group("Optimizer Configuration")
    optim.add_argument("--use_8bit_adam", type=_str2bool, default=True)
    optim.add_argument("--adam_beta1", type=float, default=0.9)
    optim.add_argument("--adam_beta2", type=float, default=0.999)
    optim.add_argument("--adam_weight_decay", type=float, default=1e-4)
    optim.add_argument("--adam_epsilon", type=float, default=1e-8)
    optim.add_argument("--max_grad_norm", type=float, default=1.0)

    val = parser.add_argument_group("Validation Configuration")
    val.add_argument("--validation_prompt", type=str, default=None)
    val.add_argument("--num_validation_images", type=int, default=4)
    val.add_argument("--validation_epochs", type=int, default=1)
    val.add_argument("--validation_num_inference_steps", type=int, default=28)

    output = parser.add_argument_group("Output and Logging")
    output.add_argument("--output_dir", type=str, default="flux-model-finetuned-lora")
    output.add_argument("--logging_dir", type=str, default="logs")
    output.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "comet_ml", "all"],
    )
    output.add_argument("--seed", type=int, default=None)

    ckpt = parser.add_argument_group("Checkpointing")
    ckpt.add_argument("--checkpointing_epochs", type=int, default=1)
    ckpt.add_argument("--checkpoints_total_limit", type=int, default=None)
    ckpt.add_argument("--save_optimizer_state", type=_str2bool, default=True)
    ckpt.add_argument("--resume_from_checkpoint", type=str, default=None)

    hub = parser.add_argument_group("Hugging Face Hub")
    hub.add_argument("--push_to_hub", action="store_true")
    hub.add_argument("--hub_token", type=str, default=None)
    hub.add_argument("--hub_model_id", type=str, default=None)

    perf = parser.add_argument_group("Performance Optimization")
    perf.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp32", "fp16", "bf16"])
    perf.add_argument("--allow_tf32", action="store_true")
    perf.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
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
