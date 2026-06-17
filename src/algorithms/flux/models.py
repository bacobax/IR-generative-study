#!/usr/bin/env python
# coding=utf-8
"""Model utilities for FLUX.1-dev QLoRA fine-tuning."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.core.diffusers_compat import disable_diffusers_optional_scipy

disable_diffusers_optional_scipy(lightweight_diffusers_imports=False)

from diffusers import (
    AutoencoderKL,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module

from src.core.artifacts import ArtifactManifest, write_artifact_manifest
from src.core.normalization import RAW_UINT16_PERCENTILE
from src.core.paths import checkpoints_root

from .config import TrainingConfig


logger = logging.getLogger(__name__)

STAGE1_MANIFEST_NAME = "stage1_manifest.json"
LORA_WEIGHT_FILENAMES = ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")


@dataclass
class ModelComponents:
    """Container for FLUX model components kept in memory during training."""

    transformer: FluxTransformer2DModel
    vae: AutoencoderKL
    noise_scheduler: FlowMatchEulerDiscreteScheduler
    weight_dtype: torch.dtype
    normalization_mode: str = RAW_UINT16_PERCENTILE


def get_weight_dtype(mixed_precision: Optional[str]) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_models(*, config: TrainingConfig, device: Optional[torch.device] = None) -> ModelComponents:
    """Load FLUX components; quantize the transformer to NF4 when quantize_4bit is set."""
    from peft import prepare_model_for_kbit_training

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="vae"
    )

    weight_dtype = get_weight_dtype(config.mixed_precision)

    if config.quantize_4bit:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.float16,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.float16,
        )
        # Prepare for k-bit training: disables quantized layers from getting
        # gradients (they stay 4-bit); LoRA adapters added later will be fp32.
        transformer = prepare_model_for_kbit_training(
            transformer, use_gradient_checkpointing=False
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    if device is not None:
        if not config.quantize_4bit:
            transformer.to(device, dtype=weight_dtype)
        # VAE stays fp32 for numerical stability (same convention as SDXL).
        vae.to(device, dtype=torch.float32)

    return ModelComponents(
        transformer=transformer,
        vae=vae,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
    )


@torch.no_grad()
def precompute_prompt_embeds(
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the single fixed prompt once and free the text encoders.

    FLUX has two text encoders: CLIP (~340 MB) and T5-XXL (~9 GB in fp16).
    Because every sample in a repo dataset uses the same fixed prompt string,
    we encode it once here and broadcast the result to the batch during training.
    This is the same memory trick as the blog's parquet precomputation, but
    trivially simple when there is only one prompt.

    Returns (prompt_embeds, pooled_prompt_embeds, text_ids) on CPU.
    """
    logger.info(
        "Pre-computing prompt embeddings for prompt=%r (max_sequence_length=%d)…",
        config.prompt_text,
        config.max_sequence_length,
    )
    pipeline = FluxPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        device_map="balanced",
        torch_dtype=torch.float16,
    )
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=config.prompt_text,
        prompt_2=None,
        max_sequence_length=config.max_sequence_length,
    )
    # Move to CPU so they don't occupy GPU memory during training setup.
    prompt_embeds = prompt_embeds.cpu()
    pooled_prompt_embeds = pooled_prompt_embeds.cpu()
    text_ids = text_ids.cpu()

    del pipeline
    free_memory()
    logger.info(
        "Prompt embeddings ready: embeds=%s pooled=%s text_ids=%s",
        tuple(prompt_embeds.shape),
        tuple(pooled_prompt_embeds.shape),
        tuple(text_ids.shape),
    )
    return prompt_embeds, pooled_prompt_embeds, text_ids


def get_lora_config(
    *,
    rank: int,
    lora_alpha_scale: float,
    target_modules: Sequence[str],
) -> object:
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=rank * lora_alpha_scale,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
    )


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
        100.0 * trainable / max(total, 1),
    )


def configure_trainable_components(
    *, models: ModelComponents, config: TrainingConfig
) -> Dict[str, object]:
    """Attach LoRA adapters to the FLUX transformer and report trainable setup."""
    models.transformer.requires_grad_(False)
    models.vae.requires_grad_(False)

    models.transformer.add_adapter(
        get_lora_config(
            rank=config.rank,
            lora_alpha_scale=config.lora_alpha_scale,
            target_modules=config.lora_target_modules,
        )
    )

    # When using fp16 mixed precision the LoRA params must stay fp32 so they
    # can accumulate small gradients without underflowing.
    if config.mixed_precision == "fp16":
        cast_training_params(models.transformer, dtype=torch.float32)

    if config.gradient_checkpointing:
        models.transformer.enable_gradient_checkpointing()

    info: Dict[str, object] = {
        "model_family": "flux",
        "lora_active": True,
        "lora_target_modules": list(config.lora_target_modules),
        "quantize_4bit": bool(config.quantize_4bit),
        "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
        "cache_latents": bool(config.cache_latents),
        "trainable_parameter_names": {
            "transformer": _trainable_param_names(models.transformer),
        },
    }
    describe_component_trainability("Transformer", models.transformer)
    describe_component_trainability("VAE", models.vae)
    return info


def unwrap_model(model, accelerator=None):
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def get_trainable_params(models: ModelComponents) -> List[torch.nn.Parameter]:
    return [p for p in models.transformer.parameters() if p.requires_grad]


def get_trainable_models(models: ModelComponents) -> List[torch.nn.Module]:
    return [models.transformer]


def create_save_model_hook(models_ref: ModelComponents, accelerator):
    """Accelerate save hook: serialise LoRA weights via FluxPipeline."""

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            from peft.utils import get_peft_model_state_dict

            transformer_lora_layers = None
            for model in models:
                unwrapped = unwrap_model(model, accelerator)
                if isinstance(unwrapped, type(unwrap_model(models_ref.transformer, accelerator))):
                    transformer_lora_layers = get_peft_model_state_dict(unwrapped)
                    weights.pop()
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

            FluxPipeline.save_lora_weights(
                save_directory=output_dir,
                transformer_lora_layers=transformer_lora_layers,
                text_encoder_lora_layers=None,
                safe_serialization=True,
            )

    return save_model_hook


def create_load_model_hook(models_ref: ModelComponents, accelerator, mixed_precision: Optional[str] = None):
    """Accelerate load hook: restore LoRA weights from a checkpoint."""

    def load_model_hook(models, input_dir):
        from peft.utils import set_peft_model_state_dict

        transformer_model = None
        expected_type = type(unwrap_model(models_ref.transformer, accelerator))
        while models:
            model = models.pop()
            unwrapped = unwrap_model(model, accelerator)
            if isinstance(unwrapped, expected_type):
                transformer_model = model
            else:
                raise ValueError(f"Unexpected load model: {model.__class__}")

        lora_state_dict, _ = FluxPipeline.lora_state_dict(input_dir)

        if transformer_model is not None:
            transformer_state = {
                key.replace("transformer.", ""): value
                for key, value in lora_state_dict.items()
                if key.startswith("transformer.")
            }
            set_peft_model_state_dict(
                transformer_model,
                convert_unet_state_dict_to_peft(transformer_state),
                adapter_name="default",
            )
        if mixed_precision == "fp16":
            cast_training_params([transformer_model], dtype=torch.float32)

    return load_model_hook


def get_canonical_output_dir(config: TrainingConfig) -> str:
    if config.output_dir != "flux-model-finetuned-lora":
        return config.output_dir
    dataset_key = config.dataset_id or "custom"
    return str(checkpoints_root() / "flux" / "lora_runs" / f"{dataset_key}_flux_lora_r{config.rank}")


def _git_metadata() -> Dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {}


def build_stage1_manifest(
    *,
    config: TrainingConfig,
    normalization_mode: str,
    adaptation_info: Dict[str, object],
) -> Dict[str, object]:
    return {
        "model_family": "flux",
        "training_mode": "qlora",
        "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
        "dataset_id": config.dataset_id,
        "train_data_dir": config.train_data_dir,
        "train_split": config.train_split,
        "resolution": config.resolution,
        "normalization_mode": normalization_mode,
        "prompt_text": config.prompt_text,
        "max_sequence_length": config.max_sequence_length,
        "learning_rate": config.learning_rate,
        "max_train_steps": config.max_train_steps,
        "rank": config.rank,
        "lora_alpha_scale": config.lora_alpha_scale,
        "lora_target_modules": list(config.lora_target_modules),
        "quantize_4bit": config.quantize_4bit,
        "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
        "cache_latents": config.cache_latents,
        "guidance_scale": config.guidance_scale,
        "weighting_scheme": config.weighting_scheme,
        "validation_prompt": config.validation_prompt,
        "validation_num_inference_steps": config.validation_num_inference_steps,
        "checkpointing_epochs": config.checkpointing_epochs,
        "resume_from_checkpoint": config.resume_from_checkpoint,
        "output_dir": config.output_dir,
        "adaptation_info": adaptation_info,
        "code": _git_metadata(),
    }


def save_stage1_manifest(output_dir: str, manifest: Dict[str, object]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, STAGE1_MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    artifact = ArtifactManifest(
        model_kind="flux_stage1_lora",
        model_family="flux",
        base_model=manifest.get("pretrained_model_name_or_path"),
        adapters=[
            {
                "kind": "qlora",
                "rank": manifest.get("rank"),
                "alpha_scale": manifest.get("lora_alpha_scale"),
                "quantize_4bit": manifest.get("quantize_4bit"),
                "bnb_4bit_quant_type": manifest.get("bnb_4bit_quant_type"),
            }
        ],
        task={"kind": "flux_stage1_lora"},
        dataset={"dataset_id": manifest.get("dataset_id"), "train_split": manifest.get("train_split")},
        normalization={"mode": manifest.get("normalization_mode")},
        metadata={"stage1_manifest": STAGE1_MANIFEST_NAME},
    )
    write_artifact_manifest(output_dir, artifact)
    logger.info("Saved FLUX stage-1 manifest to %s", manifest_path)
    return manifest_path


def load_stage1_manifest(stage1_dir: str | Path) -> Dict[str, object]:
    manifest_path = Path(stage1_dir) / STAGE1_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing FLUX stage-1 manifest at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_flux_stage1_pipeline(
    *,
    stage1_dir: str | Path,
    base_model: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[FluxPipeline, Dict[str, object]]:
    """Load a FLUX stage-1 LoRA artifact into a FluxPipeline for inference."""
    manifest = load_stage1_manifest(stage1_dir)
    if manifest.get("model_family") not in {None, "flux"}:
        raise ValueError(f"Expected FLUX stage-1 manifest, got {manifest.get('model_family')!r}")
    base_model_id = base_model or manifest["pretrained_model_name_or_path"]
    pipeline = FluxPipeline.from_pretrained(base_model_id, torch_dtype=torch_dtype)
    pipeline.load_lora_weights(str(stage1_dir))
    return pipeline, manifest
