#!/usr/bin/env python
# coding=utf-8
"""Model utilities for Stage-1 Stable Diffusion IR adaptation."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from src.core.normalization import RAW_UINT16_PERCENTILE
from src.core.paths import sd_lora_runs_dir, sd_unet_runs_dir

from .config import TrainingConfig


logger = logging.getLogger(__name__)

STAGE1_MANIFEST_NAME = "stage1_manifest.json"
UNET_EXPORT_DIRNAME = "unet"
TEXT_ENCODER_EXPORT_DIRNAME = "text_encoder"
VAE_EXPORT_DIRNAME = "vae"


@dataclass
class ModelComponents:
    """Container for all model components."""

    unet: UNet2DConditionModel
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    noise_scheduler: DDPMScheduler
    weight_dtype: torch.dtype
    normalization_mode: str = RAW_UINT16_PERCENTILE


def get_weight_dtype(mixed_precision: Optional[str]) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _component_dtype(is_trainable: bool, weight_dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if is_trainable else weight_dtype


def is_unet_trainable(config: TrainingConfig) -> bool:
    return config.baseline_mode == "sd_ir_unet"


def is_text_encoder_trainable(config: TrainingConfig) -> bool:
    return config.baseline_mode == "sd_ir_unet" and not config.freeze_text_encoder


def is_vae_trainable(config: TrainingConfig) -> bool:
    return config.baseline_mode == "sd_ir_unet" and not config.freeze_vae


def load_models(
    *,
    config: TrainingConfig,
    device: torch.device | None = None,
) -> ModelComponents:
    """Load all model components with trainable modules kept in fp32."""
    logger.info("Loading model components...")

    noise_scheduler = DDPMScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=config.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=config.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=config.revision,
        variant=config.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="unet",
        revision=config.revision,
        variant=config.variant,
    )

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = get_weight_dtype(config.mixed_precision)
    if device is not None:
        unet.to(device, dtype=_component_dtype(is_unet_trainable(config), weight_dtype))
        vae.to(device, dtype=_component_dtype(is_vae_trainable(config), weight_dtype))
        text_encoder.to(device, dtype=_component_dtype(is_text_encoder_trainable(config), weight_dtype))

    return ModelComponents(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
    )


def get_lora_config(
    *,
    rank: int = 4,
    lora_alpha_scale: float = 1.0,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=rank * lora_alpha_scale,
        init_lora_weights="gaussian",
        target_modules=target_modules or [],
    )


def _enable_xformers(unet: UNet2DConditionModel) -> None:
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


def _set_requires_grad_for_prefixes(module: torch.nn.Module, prefixes: List[str]) -> List[str]:
    matched: List[str] = []
    for name, param in module.named_parameters():
        is_match = any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        param.requires_grad_(is_match)
        if is_match:
            matched.append(name)
    return matched


def _trainable_param_names(module: torch.nn.Module) -> List[str]:
    return [name for name, param in module.named_parameters() if param.requires_grad]


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def count_total_parameters(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def describe_component_trainability(name: str, module: torch.nn.Module) -> None:
    total = count_total_parameters(module)
    trainable = count_trainable_parameters(module)
    logger.info(
        "%s parameters: trainable=%s / total=%s (%.2f%%)",
        name,
        f"{trainable:,}",
        f"{total:,}",
        (100.0 * trainable / max(total, 1)),
    )


def configure_trainable_components(
    *,
    models: ModelComponents,
    config: TrainingConfig,
) -> Dict[str, object]:
    """Apply LoRA or U-Net adaptation settings and report the resolved setup."""
    info: Dict[str, object] = {
        "baseline_mode": config.baseline_mode,
        "freeze_vae": config.freeze_vae,
        "freeze_text_encoder": config.freeze_text_encoder,
        "lora_active": config.baseline_mode == "sd_ir_lora",
        "unet_train_mode": config.unet_train_mode if config.baseline_mode == "sd_ir_unet" else None,
        "lora_target_modules": list(config.lora_target_modules),
        "unet_trainable_modules": [],
        "trainable_parameter_names": {},
    }

    if config.baseline_mode == "sd_ir_lora":
        models.unet.add_adapter(
            get_lora_config(
                rank=config.rank,
                lora_alpha_scale=config.lora_alpha_scale,
                target_modules=config.lora_target_modules,
            )
        )
        if config.mixed_precision == "fp16":
            cast_training_params(models.unet, dtype=torch.float32)
    else:
        models.unet.requires_grad_(False)
        if config.unet_train_mode == "full":
            models.unet.requires_grad_(True)
            info["unet_trainable_modules"] = ["<all_unet_parameters>"]
        else:
            matched = _set_requires_grad_for_prefixes(
                models.unet,
                list(config.unet_trainable_modules),
            )
            if not matched:
                raise ValueError(
                    "Partial U-Net mode resolved to zero trainable parameters. "
                    f"Provided prefixes: {config.unet_trainable_modules!r}"
                )
            info["unet_trainable_modules"] = list(config.unet_trainable_modules)

    if config.freeze_text_encoder:
        models.text_encoder.requires_grad_(False)
    else:
        models.text_encoder.requires_grad_(True)

    if config.freeze_vae:
        models.vae.requires_grad_(False)
    else:
        models.vae.requires_grad_(True)

    if config.enable_xformers_memory_efficient_attention:
        _enable_xformers(models.unet)
    if config.gradient_checkpointing:
        models.unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")

    info["trainable_parameter_names"] = {
        "unet": _trainable_param_names(models.unet),
        "text_encoder": _trainable_param_names(models.text_encoder),
        "vae": _trainable_param_names(models.vae),
    }

    describe_component_trainability("UNet", models.unet)
    describe_component_trainability("Text encoder", models.text_encoder)
    describe_component_trainability("VAE", models.vae)
    return info


def unwrap_model(model, accelerator=None):
    """Unwrap a model from DDP or compiled wrapper."""
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def get_trainable_params(models: ModelComponents) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for module in (models.unet, models.text_encoder, models.vae):
        params.extend(param for param in module.parameters() if param.requires_grad)
    return params


def get_trainable_models(models: ModelComponents) -> List[torch.nn.Module]:
    trainable = []
    for module in (models.unet, models.text_encoder, models.vae):
        if any(param.requires_grad for param in module.parameters()):
            trainable.append(module)
    return trainable


def trainable_component_names(models: ModelComponents) -> List[str]:
    names = []
    if any(param.requires_grad for param in models.unet.parameters()):
        names.append("unet")
    if any(param.requires_grad for param in models.text_encoder.parameters()):
        names.append("text_encoder")
    if any(param.requires_grad for param in models.vae.parameters()):
        names.append("vae")
    return names


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

        lora_state_dict, _ = StableDiffusionPipeline.lora_state_dict(input_dir)

        unet_state_dict = {
            f"{key.replace('unet.', '')}": value
            for key, value in lora_state_dict.items()
            if key.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    "Loading adapter weights led to unexpected keys: %s",
                    unexpected_keys,
                )

        if mixed_precision == "fp16":
            cast_training_params([unet_], dtype=torch.float32)

    return load_model_hook


def get_canonical_output_dir(config: TrainingConfig) -> str:
    """Return a canonical output directory if the caller leaves the placeholder default."""
    if config.output_dir != "sd-model-finetuned-lora":
        return config.output_dir

    base_root = sd_lora_runs_dir() if config.baseline_mode == "sd_ir_lora" else sd_unet_runs_dir()
    dataset_key = config.dataset_id or "custom"
    if config.baseline_mode == "sd_ir_lora":
        run_name = f"{dataset_key}_sd15_lora_r{config.rank}"
    else:
        run_name = f"{dataset_key}_sd15_unet_{config.unet_train_mode}"
    return str(base_root / run_name)


def build_stage1_manifest(
    *,
    config: TrainingConfig,
    normalization_mode: str,
    adaptation_info: Dict[str, object],
) -> Dict[str, object]:
    """Create a JSON-serializable manifest for the exported stage-1 artifact."""
    return {
        "baseline_mode": config.baseline_mode,
        "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
        "revision": config.revision,
        "variant": config.variant,
        "dataset_id": config.dataset_id,
        "train_data_dir": config.train_data_dir,
        "train_split": config.train_split,
        "normalization_mode": normalization_mode,
        "prompt_text": config.resolved_prompt_text(),
        "learning_rate": config.learning_rate,
        "max_train_steps": config.max_train_steps,
        "freeze_vae": config.freeze_vae,
        "freeze_text_encoder": config.freeze_text_encoder,
        "rank": config.rank,
        "lora_alpha_scale": config.lora_alpha_scale,
        "lora_target_modules": list(config.lora_target_modules),
        "unet_train_mode": config.unet_train_mode,
        "unet_trainable_modules": list(config.unet_trainable_modules),
        "validation_prompt": config.validation_prompt,
        "validation_num_inference_steps": config.validation_num_inference_steps,
        "checkpointing_steps": config.checkpointing_steps,
        "resume_from_checkpoint": config.resume_from_checkpoint,
        "output_dir": config.output_dir,
        "adaptation_info": adaptation_info,
    }


def save_stage1_manifest(output_dir: str, manifest: Dict[str, object]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, STAGE1_MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    logger.info("Saved stage-1 manifest to %s", manifest_path)
    return manifest_path


def load_stage1_manifest(stage1_dir: str) -> Dict[str, object]:
    manifest_path = Path(stage1_dir) / STAGE1_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing stage-1 manifest at {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_stage1_pipeline(
    *,
    stage1_dir: str,
    base_model: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> tuple[StableDiffusionPipeline, Dict[str, object]]:
    """Load a reusable stage-1 artifact into a StableDiffusionPipeline."""
    manifest = load_stage1_manifest(stage1_dir)
    base_model_id = base_model or manifest["pretrained_model_name_or_path"]
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        revision=manifest.get("revision"),
        variant=manifest.get("variant"),
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if manifest["baseline_mode"] == "sd_ir_lora":
        pipeline.load_lora_weights(stage1_dir)
    else:
        unet_dir = Path(stage1_dir) / UNET_EXPORT_DIRNAME
        if not unet_dir.is_dir():
            raise FileNotFoundError(f"Missing exported U-Net at {unet_dir}")
        pipeline.unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=torch_dtype)

        text_encoder_dir = Path(stage1_dir) / TEXT_ENCODER_EXPORT_DIRNAME
        if text_encoder_dir.is_dir():
            pipeline.text_encoder = CLIPTextModel.from_pretrained(text_encoder_dir, torch_dtype=torch_dtype)

        vae_dir = Path(stage1_dir) / VAE_EXPORT_DIRNAME
        if vae_dir.is_dir():
            pipeline.vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=torch_dtype)

    return pipeline, manifest
