#!/usr/bin/env python
# coding=utf-8
"""Configuration for Stage-1 Stable Diffusion XL IR adaptation."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from src.algorithms.stable_diffusion.config import (
    DEFAULT_PROMPT_TEXT,
    LEGACY_GENERIC_PROMPT,
    _str2bool,
)
from src.core.configs.config_loader import apply_yaml_defaults, load_yaml
from src.core.data.dataset_targets import supported_dataset_ids


DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SDXL_LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
DEFAULT_TEXT_ENCODER_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
DEFAULT_NUM_TRAIN_EPOCHS = 80


@dataclass
class TrainingConfig:
    """Configuration dataclass for SDXL LoRA stage-1 training."""

    pretrained_model_name_or_path: str = DEFAULT_SDXL_MODEL
    revision: Optional[str] = None
    variant: Optional[str] = None

    baseline_mode: str = "sdxl_ir_lora"

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
    subset_manifest: Optional[str] = None
    cache_dir: Optional[str] = None

    resolution: int = 1024
    center_crop: bool = False
    random_flip: bool = False
    image_interpolation_mode: str = "lanczos"
    use_ir_preprocessing: bool = True

    freeze_vae: bool = True
    freeze_text_encoder: bool = True

    train_batch_size: int = 4
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0

    rank: int = 4
    lora_alpha_scale: float = 1.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: list(DEFAULT_SDXL_LORA_TARGET_MODULES)
    )
    text_encoder_lora_enabled: bool = False
    text_encoder_lora_target_modules: List[str] = field(
        default_factory=lambda: list(DEFAULT_TEXT_ENCODER_LORA_TARGET_MODULES)
    )

    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0

    snr_gamma: Optional[float] = None
    noise_offset: float = 0.0
    prediction_type: Optional[str] = None

    validation_prompt: Optional[str] = None
    num_validation_images: int = 4
    validation_epochs: int = 1
    validation_num_inference_steps: int = 30

    output_dir: str = "sdxl-model-finetuned-lora"
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    seed: Optional[int] = None

    checkpointing_epochs: int = 1
    checkpoints_total_limit: Optional[int] = None
    save_optimizer_state: bool = True
    resume_from_checkpoint: Optional[str] = None

    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

    mixed_precision: Optional[str] = None
    allow_tf32: bool = False
    enable_xformers_memory_efficient_attention: bool = False
    dataloader_num_workers: int = 0
    local_rank: int = -1

    def resolved_prompt_text(self) -> Optional[str]:
        if self.generic_prompt:
            return LEGACY_GENERIC_PROMPT
        if self.prompt_text:
            return self.prompt_text
        return None

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
        if self.baseline_mode != "sdxl_ir_lora":
            raise ValueError(
                f"Unknown baseline_mode={self.baseline_mode!r}. Expected 'sdxl_ir_lora'."
            )
        if not self.freeze_vae:
            raise ValueError("SDXL LoRA training keeps the VAE frozen.")
        if not self.freeze_text_encoder and not self.text_encoder_lora_enabled:
            raise ValueError(
                "Set text_encoder_lora_enabled=True to train SDXL text encoder adapters."
            )
        if self.rank <= 0:
            raise ValueError("--rank must be positive for LoRA training.")
        if not (0.0 <= self.lora_alpha_scale <= 1.0):
            raise ValueError("--lora_alpha_scale must be between 0 and 1.")
        if not self.lora_target_modules:
            raise ValueError("SDXL LoRA training requires lora_target_modules.")
        if self.text_encoder_lora_enabled and not self.text_encoder_lora_target_modules:
            raise ValueError("text_encoder_lora_target_modules must be non-empty.")
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
        raise ValueError(f"Expected top-level mapping in SDXL config {config_path!s}.")
    known_dests = {action.dest for action in parser._actions}
    unknown = sorted(key for key in data if key not in known_dests)
    if unknown:
        raise ValueError(
            f"Unknown keys in SDXL config {config_path}: {', '.join(unknown)}. "
            "Remove them or add matching argparse/config fields."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1 SDXL LoRA IR adaptation training.")
    parser.add_argument("--config", type=str, default=None)

    model = parser.add_argument_group("Model Configuration")
    model.add_argument("--pretrained_model_name_or_path", type=str, default=DEFAULT_SDXL_MODEL)
    model.add_argument("--revision", type=str, default=None)
    model.add_argument("--variant", type=str, default=None)
    model.add_argument("--baseline_mode", type=str, default="sdxl_ir_lora", choices=["sdxl_ir_lora"])
    model.add_argument("--freeze_vae", type=_str2bool, default=True)
    model.add_argument("--freeze_text_encoder", type=_str2bool, default=True)

    data = parser.add_argument_group("Dataset Configuration")
    data.add_argument("--dataset_id", type=str, default=None, choices=sorted(supported_dataset_ids()))
    data.add_argument("--dataset_name", type=str, default=None)
    data.add_argument("--dataset_config_name", type=str, default=None)
    data.add_argument("--train_data_dir", type=str, default=None)
    data.add_argument("--train_split", type=str, default="train")
    data.add_argument("--image_column", type=str, default="image")
    data.add_argument("--caption_column", type=str, default="text")
    data.add_argument("--prompt_text", type=str, default=DEFAULT_PROMPT_TEXT)
    data.add_argument("--generic_prompt", action="store_true")
    data.add_argument("--max_train_samples", type=int, default=None)
    data.add_argument("--subset_manifest", type=str, default=None)
    data.add_argument("--cache_dir", type=str, default=None)

    preprocess = parser.add_argument_group("Image Preprocessing")
    preprocess.add_argument("--resolution", type=int, default=1024)
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
    train.add_argument("--train_batch_size", type=int, default=4)
    train.add_argument("--num_train_epochs", type=int, default=DEFAULT_NUM_TRAIN_EPOCHS)
    train.add_argument("--max_train_steps", type=int, default=None)
    train.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train.add_argument("--gradient_checkpointing", action="store_true")
    train.add_argument("--learning_rate", type=float, default=1e-4)
    train.add_argument("--scale_lr", action="store_true")
    train.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    train.add_argument("--lr_warmup_steps", type=int, default=0)

    lora = parser.add_argument_group("LoRA Configuration")
    lora.add_argument("--rank", type=int, default=4)
    lora.add_argument("--lora_alpha_scale", type=float, default=1.0)
    lora.add_argument("--lora_target_modules", nargs="+", default=list(DEFAULT_SDXL_LORA_TARGET_MODULES))
    lora.add_argument("--text_encoder_lora_enabled", type=_str2bool, default=False)
    lora.add_argument(
        "--text_encoder_lora_target_modules",
        nargs="+",
        default=list(DEFAULT_TEXT_ENCODER_LORA_TARGET_MODULES),
    )

    optim = parser.add_argument_group("Optimizer Configuration")
    optim.add_argument("--use_8bit_adam", action="store_true")
    optim.add_argument("--adam_beta1", type=float, default=0.9)
    optim.add_argument("--adam_beta2", type=float, default=0.999)
    optim.add_argument("--adam_weight_decay", type=float, default=1e-2)
    optim.add_argument("--adam_epsilon", type=float, default=1e-08)
    optim.add_argument("--max_grad_norm", type=float, default=1.0)

    loss = parser.add_argument_group("Loss Configuration")
    loss.add_argument("--snr_gamma", type=float, default=None)
    loss.add_argument("--noise_offset", type=float, default=0.0)
    loss.add_argument("--prediction_type", type=str, default=None, choices=["epsilon", "v_prediction", None])

    val = parser.add_argument_group("Validation Configuration")
    val.add_argument("--validation_prompt", type=str, default=None)
    val.add_argument("--num_validation_images", type=int, default=4)
    val.add_argument("--validation_epochs", type=int, default=1)
    val.add_argument("--validation_num_inference_steps", type=int, default=30)

    output = parser.add_argument_group("Output and Logging")
    output.add_argument("--output_dir", type=str, default="sdxl-model-finetuned-lora")
    output.add_argument("--logging_dir", type=str, default="logs")
    output.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "comet_ml", "all"])
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
    perf.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
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

