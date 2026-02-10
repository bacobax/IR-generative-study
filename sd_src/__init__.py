# Stable Diffusion 1.5 LoRA Fine-tuning Package
"""
Modular package for fine-tuning Stable Diffusion 1.5 with LoRA.

Modules:
    - config: Argument parsing and configuration management
    - data: Dataset loading, preprocessing, and data augmentation
    - models: Model loading, LoRA configuration, and model utilities
    - training: Training loop, validation, and checkpointing
    - utils: Helper functions and utilities
"""

from .config import parse_args, TrainingConfig
from .data import create_dataloader, get_transforms
from .models import load_models, setup_lora, ModelComponents
from .training import Trainer
from .utils import ir_to_3ch_with_stretch, trainable_params, generate_prompt

__all__ = [
    "parse_args",
    "TrainingConfig",
    "create_dataloader",
    "get_transforms",
    "load_models",
    "setup_lora",
    "ModelComponents",
    "Trainer",
    "ir_to_3ch_with_stretch",
    "trainable_params",
    "generate_prompt",
]
