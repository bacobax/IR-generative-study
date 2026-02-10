#!/usr/bin/env python
# coding=utf-8
"""
Utility functions for Stable Diffusion LoRA fine-tuning.

Contains helper functions for image processing, model utilities, and hub operations.
"""

import logging
import os
from typing import List, Optional

import numpy as np
from PIL import Image

from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card


logger = logging.getLogger(__name__)


def ir_to_3ch_with_stretch(pil_img: Image.Image) -> Image.Image:
    """
    Convert infrared grayscale image to 3-channel RGB with contrast stretching.
    
    Stable Diffusion expects 3-channel RGB images. This function converts
    grayscale infrared images and applies contrast stretching to normalize
    the intensity range.
    
    Args:
        pil_img: Input PIL image (can be grayscale or RGB).
    
    Returns:
        3-channel RGB PIL image with stretched contrast.
    """
    # Convert to numpy array
    arr = np.array(pil_img)
    
    # If already 3-channel, take first channel for luminance
    if arr.ndim == 3:
        arr = arr[..., 0]
    
    arr = arr.astype(np.uint8)
    
    # Compute min/max for contrast stretching
    mn = int(arr.min())
    mx = int(arr.max())
    
    # Apply contrast stretching (avoid divide-by-zero)
    if mx <= mn:
        stretched = np.zeros_like(arr, dtype=np.uint8)
    else:
        stretched = ((arr - mn) * 255.0 / (mx - mn)).clip(0, 255).astype(np.uint8)
    
    # Convert to 3-channel RGB
    return Image.fromarray(stretched, mode="L").convert("RGB")


def trainable_params(model) -> List:
    """
    Get list of trainable parameters from a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        List of parameters that have requires_grad=True.
    """
    return [p for p in model.parameters() if p.requires_grad]


def generate_prompt(num_persons: int) -> str:
    """
    Generate a text prompt for infrared surveillance images.
    
    Args:
        num_persons: Number of people in the image.
    
    Returns:
        Text prompt describing the image.
    """
    base = "overhead infrared surveillance image, circular field of view"
    
    if num_persons == 0:
        return base
    elif num_persons == 1:
        return base + ", one person"
    else:
        return base + f", {num_persons} people"


def count_parameters(model, trainable_only: bool = False) -> int:
    """
    Count parameters in a model.
    
    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.
    
    Returns:
        Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def log_model_info(model, model_name: str = "Model") -> None:
    """
    Log model parameter information.
    
    Args:
        model: PyTorch model.
        model_name: Name for logging.
    """
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    frozen = total - trainable
    
    logger.info(f"{model_name} parameters:")
    logger.info(f"  Total: {total:,}")
    logger.info(f"  Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
    logger.info(f"  Frozen: {frozen:,}")


def save_model_card(
    repo_id: str,
    images: Optional[List] = None,
    base_model: Optional[str] = None,
    dataset_name: Optional[str] = None,
    repo_folder: Optional[str] = None,
) -> None:
    """
    Create and save a model card for the trained LoRA.
    
    Args:
        repo_id: Repository ID on HuggingFace hub.
        images: List of sample images to include.
        base_model: Name of the base model.
        dataset_name: Name of the training dataset.
        repo_folder: Local folder to save the model card.
    """
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}

These are LoRA adaption weights for {base_model}. 
The weights were fine-tuned on the {dataset_name} dataset.

## Example Images

{img_str}

## Usage

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("{base_model}")
pipeline.load_lora_weights("{repo_id}")

image = pipeline("your prompt here").images[0]
```
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))
    
    logger.info(f"Saved model card to {repo_folder}/README.md")


def setup_logging(
    accelerator,
    log_level: str = "INFO",
) -> None:
    """
    Setup logging configuration for training.
    
    Args:
        accelerator: Accelerator instance.
        log_level: Logging level.
    """
    import datasets
    import transformers
    import diffusers
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=getattr(logging, log_level.upper()),
    )
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
