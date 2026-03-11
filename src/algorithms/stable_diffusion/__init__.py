# Stable Diffusion 1.5 LoRA Fine-tuning Package
"""
Modular package for fine-tuning Stable Diffusion 1.5 with LoRA.

This is the source-of-truth location under the ``src`` namespace.
The legacy ``sd_src`` package re-exports from here for backward compatibility.

Modules:
    - config: Argument parsing and configuration management
    - data: Dataset loading, preprocessing, and data augmentation
    - models: Model loading, LoRA configuration, and model utilities
    - training: Training loop, validation, and checkpointing
    - utils: Helper functions and utilities

Imports are lazy to avoid triggering heavy dependency loading
(diffusers, transformers, bitsandbytes) at package import time.
Use explicit sub-module imports, e.g.::

    from src.algorithms.stable_diffusion.config import TrainingConfig
"""

__all__ = [
    "config",
    "data",
    "models",
    "training",
    "utils",
    "helpers",
]
