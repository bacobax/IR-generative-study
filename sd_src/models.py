#!/usr/bin/env python
# coding=utf-8
"""
Model module for Stable Diffusion LoRA fine-tuning.

Handles model loading, LoRA configuration, and model utilities.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


logger = logging.getLogger(__name__)


@dataclass
class ModelComponents:
    """Container for all model components."""
    unet: UNet2DConditionModel
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    noise_scheduler: DDPMScheduler
    weight_dtype: torch.dtype


def get_weight_dtype(mixed_precision: Optional[str]) -> torch.dtype:
    """Get weight dtype based on mixed precision setting."""
    if mixed_precision == "fp16":
        return torch.float16
    elif mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_models(
    pretrained_model_name_or_path: str,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
    device: torch.device = None,
    mixed_precision: Optional[str] = None,
) -> ModelComponents:
    """
    Load all model components for Stable Diffusion.
    
    Args:
        pretrained_model_name_or_path: Path or HF identifier for the model.
        revision: Model revision to use.
        variant: Model variant (e.g., 'fp16').
        device: Target device for models.
        mixed_precision: Mixed precision mode.
    
    Returns:
        ModelComponents containing all loaded models.
    """
    logger.info("Loading model components...")
    
    # Load scheduler and tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    
    # Load models
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=revision,
        variant=variant,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
        variant=variant,
    )
    
    # Freeze non-trainable models
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Determine weight dtype
    weight_dtype = get_weight_dtype(mixed_precision)
    
    # Move models to device
    if device is not None:
        unet.to(device, dtype=weight_dtype)
        vae.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
    
    logger.info("Model components loaded successfully")
    
    return ModelComponents(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
    )


def get_lora_config(rank: int = 4) -> LoraConfig:
    """
    Create LoRA configuration for UNet.
    
    Args:
        rank: LoRA rank (dimension of update matrices).
    
    Returns:
        LoraConfig for PEFT.
    """
    return LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"],
    )


def setup_lora(
    unet: UNet2DConditionModel,
    lora_config: LoraConfig,
    mixed_precision: Optional[str] = None,
    gradient_checkpointing: bool = False,
    enable_xformers: bool = False,
) -> UNet2DConditionModel:
    """
    Setup LoRA adapter on UNet.
    
    Args:
        unet: UNet model to add LoRA to.
        lora_config: LoRA configuration.
        mixed_precision: Mixed precision mode.
        gradient_checkpointing: Enable gradient checkpointing.
        enable_xformers: Enable xformers memory efficient attention.
    
    Returns:
        UNet with LoRA adapter added.
    """
    logger.info("Setting up LoRA adapter...")
    
    # Add LoRA adapter
    unet.add_adapter(lora_config)
    
    # Cast trainable params to fp32 if using fp16 mixed precision
    if mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)
    
    # Enable xformers if requested
    if enable_xformers:
        _enable_xformers(unet)
    
    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    
    return unet


def _enable_xformers(unet: UNet2DConditionModel) -> None:
    """Enable xformers memory efficient attention."""
    if is_xformers_available():
        import xformers
        
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warning(
                "xFormers 0.0.16 cannot be used for training in some GPUs. "
                "Please update to at least 0.0.17."
            )
        unet.enable_xformers_memory_efficient_attention()
        logger.info("xFormers memory efficient attention enabled")
    else:
        raise ValueError("xformers is not available. Please install it correctly.")


def unwrap_model(model, accelerator=None):
    """Unwrap a model from DDP or compiled wrapper."""
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def get_trainable_params(model) -> list:
    """Get list of trainable parameters."""
    return [p for p in model.parameters() if p.requires_grad]


def create_save_model_hook(unet, accelerator):
    """Create a hook for saving LoRA weights during checkpointing."""
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet, accelerator))):
                    unet_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                weights.pop()

            StableDiffusionPipeline.save_lora_weights(
                save_directory=output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                safe_serialization=True,
            )
    
    return save_model_hook


def create_load_model_hook(unet, accelerator, mixed_precision: Optional[str] = None):
    """Create a hook for loading LoRA weights during checkpoint resume."""
    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(unet, accelerator))):
                unet_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = StableDiffusionPipeline.lora_state_dict(input_dir)

        unet_state_dict = {
            f"{k.replace('unet.', '')}": v 
            for k, v in lora_state_dict.items() 
            if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights led to unexpected keys: {unexpected_keys}"
                )

        if mixed_precision == "fp16":
            cast_training_params([unet_], dtype=torch.float32)
    
    return load_model_hook
