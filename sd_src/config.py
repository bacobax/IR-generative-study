#!/usr/bin/env python
# coding=utf-8
"""
Configuration module for Stable Diffusion LoRA fine-tuning.

Handles argument parsing and configuration management.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, List

from torchvision import transforms


@dataclass
class TrainingConfig:
    """Configuration dataclass for training parameters."""
    
    # Model configuration
    pretrained_model_name_or_path: str = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    
    # Dataset configuration
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_data_dir: Optional[str] = None
    image_column: str = "image"
    caption_column: str = "text"
    generic_prompt: bool = False
    max_train_samples: Optional[int] = None
    cache_dir: Optional[str] = None
    
    # Image preprocessing
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    image_interpolation_mode: str = "lanczos"
    use_ir_preprocessing: bool = False
    
    # Training hyperparameters
    train_batch_size: int = 16
    num_train_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    
    # LoRA configuration
    rank: int = 4
    
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
    
    # Validation configuration
    validation_prompt: Optional[str] = None
    num_validation_images: int = 4
    validation_epochs: int = 1
    
    # Output and logging
    output_dir: str = "sd-model-finetuned-lora"
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    seed: Optional[int] = None
    
    # Checkpointing
    checkpointing_steps: int = 500
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
    
    def validate(self):
        """Validate configuration parameters."""
        if self.dataset_name is None and self.train_data_dir is None:
            raise ValueError("Need either a dataset name or a training folder.")
        
        if self.report_to == "wandb" and self.hub_token is not None:
            raise ValueError(
                "You cannot use both --report_to=wandb and --hub_token due to a security risk."
                " Please use `hf auth login` to authenticate with the Hub."
            )
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create config from parsed arguments."""
        return cls(**{k: v for k, v in vars(args).items() if hasattr(cls, k) or k in cls.__dataclass_fields__})


def parse_args() -> TrainingConfig:
    """Parse command-line arguments and return a TrainingConfig object."""
    parser = argparse.ArgumentParser(description="Fine-tuning script for Stable Diffusion with LoRA.")
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    model_group.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    model_group.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files (e.g., 'fp16').",
    )
    
    # Dataset arguments
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub) to train on.",
    )
    data_group.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset.",
    )
    data_group.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data with a metadata.jsonl file.",
    )
    data_group.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    data_group.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption.",
    )
    data_group.add_argument(
        "--generic_prompt",
        action="store_true",
        help=(
            "Use a fixed prompt for all samples: "
            "'overhead infrared surveillance image, circular field of view'."
        ),
    )
    data_group.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Truncate the number of training examples for debugging.",
    )
    data_group.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for storing downloaded models and datasets.",
    )
    
    # Image preprocessing arguments
    preprocess_group = parser.add_argument_group("Image Preprocessing")
    preprocess_group.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for input images.",
    )
    preprocess_group.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images.",
    )
    preprocess_group.add_argument(
        "--random_flip",
        action="store_true",
        help="Whether to randomly flip images horizontally.",
    )
    preprocess_group.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[f.lower() for f in dir(transforms.InterpolationMode) 
                 if not f.startswith("__") and not f.endswith("__")],
        help="Interpolation method for resizing images.",
    )
    preprocess_group.add_argument(
        "--use_ir_preprocessing",
        action="store_true",
        help="Enable infrared preprocessing for images.",
    )
    
    # Training hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size per device.",
    )
    train_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    train_group.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps (overrides num_train_epochs).",
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    train_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory.",
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    train_group.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale learning rate by batch size and accumulation steps.",
    )
    train_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", 
                 "constant", "constant_with_warmup"],
        help="Learning rate scheduler type.",
    )
    train_group.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps.",
    )
    
    # LoRA configuration
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument(
        "--rank",
        type=int,
        default=4,
        help="LoRA rank (dimension of update matrices).",
    )
    
    # Optimizer arguments
    optim_group = parser.add_argument_group("Optimizer Configuration")
    optim_group.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam from bitsandbytes.",
    )
    optim_group.add_argument("--adam_beta1", type=float, default=0.9)
    optim_group.add_argument("--adam_beta2", type=float, default=0.999)
    optim_group.add_argument("--adam_weight_decay", type=float, default=1e-2)
    optim_group.add_argument("--adam_epsilon", type=float, default=1e-08)
    optim_group.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Loss configuration
    loss_group = parser.add_argument_group("Loss Configuration")
    loss_group.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma (recommended: 5.0).",
    )
    loss_group.add_argument(
        "--noise_offset",
        type=float,
        default=0.0,
        help="Scale of noise offset.",
    )
    loss_group.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        choices=["epsilon", "v_prediction", None],
        help="Prediction type for training.",
    )
    
    # Validation arguments
    val_group = parser.add_argument_group("Validation Configuration")
    val_group.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="Prompt for validation inference.",
    )
    val_group.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of validation images to generate.",
    )
    val_group.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    
    # Output and logging
    output_group = parser.add_argument_group("Output and Logging")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="Output directory for checkpoints and model.",
    )
    output_group.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    output_group.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "comet_ml", "all"],
        help="Logging integration to use.",
    )
    output_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    
    # Checkpointing
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps.",
    )
    ckpt_group.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep.",
    )
    ckpt_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint or 'latest' to resume training.",
    )
    
    # Hub configuration
    hub_group = parser.add_argument_group("Hugging Face Hub")
    hub_group.add_argument("--push_to_hub", action="store_true")
    hub_group.add_argument("--hub_token", type=str, default=None)
    hub_group.add_argument("--hub_model_id", type=str, default=None)
    
    # Performance optimization
    perf_group = parser.add_argument_group("Performance Optimization")
    perf_group.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode.",
    )
    perf_group.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 on Ampere GPUs.",
    )
    perf_group.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xformers memory efficient attention.",
    )
    perf_group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    
    # Distributed training
    dist_group = parser.add_argument_group("Distributed Training")
    dist_group.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    
    # Handle environment variable for local rank
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    # Create config and validate
    config = TrainingConfig(**vars(args))
    config.validate()
    
    return config
