#!/usr/bin/env python
# coding=utf-8
"""Model utilities for Stage-1 Stable Diffusion XL IR adaptation."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from packaging import version

from src.core.diffusers_compat import disable_diffusers_optional_scipy

disable_diffusers_optional_scipy(lightweight_diffusers_imports=False)

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from src.core.artifacts import ArtifactManifest, write_artifact_manifest
from src.core.normalization import RAW_UINT16_PERCENTILE
from src.core.paths import checkpoints_root
from src.core.training_utils import cast_training_params

from .config import TrainingConfig


logger = logging.getLogger(__name__)

STAGE1_MANIFEST_NAME = "stage1_manifest.json"
LORA_WEIGHT_FILENAMES = ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")


@dataclass
class ModelComponents:
    """Container for all SDXL model components."""

    unet: UNet2DConditionModel
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    text_encoder_2: CLIPTextModelWithProjection
    tokenizer: object
    tokenizer_2: object
    noise_scheduler: DDPMScheduler
    weight_dtype: torch.dtype
    normalization_mode: str = RAW_UINT16_PERCENTILE


def get_weight_dtype(mixed_precision: Optional[str]) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _component_dtype(is_trainable: bool, weight_dtype: torch.dtype, *, force_fp32: bool = False) -> torch.dtype:
    if force_fp32 or is_trainable:
        return torch.float32
    return weight_dtype


def load_models(*, config: TrainingConfig, device: torch.device | None = None) -> ModelComponents:
    """Load SDXL components with both tokenizers and both text encoders explicit."""
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=config.revision,
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=config.revision,
        use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=config.revision,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
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
    text_encoder_2.requires_grad_(False)

    weight_dtype = get_weight_dtype(config.mixed_precision)
    if device is not None:
        unet.to(device, dtype=weight_dtype)
        vae.to(device, dtype=torch.float32)
        text_dtype = _component_dtype(config.text_encoder_lora_enabled, weight_dtype)
        text_encoder.to(device, dtype=text_dtype)
        text_encoder_2.to(device, dtype=text_dtype)

    return ModelComponents(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        noise_scheduler=noise_scheduler,
        weight_dtype=weight_dtype,
    )


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


def _enable_xformers(unet: UNet2DConditionModel) -> None:
    if is_xformers_available():
        import xformers

        if version.parse(xformers.__version__) == version.parse("0.0.16"):
            logger.warning("xFormers 0.0.16 can be unstable for training; update if possible.")
        unet.enable_xformers_memory_efficient_attention()
        return
    raise ValueError("xformers is not available. Please install it correctly.")


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


def configure_trainable_components(*, models: ModelComponents, config: TrainingConfig) -> Dict[str, object]:
    """Freeze SDXL backbones, attach LoRA adapters, and report trainable setup."""
    models.unet.requires_grad_(False)
    models.vae.requires_grad_(False)
    models.text_encoder.requires_grad_(False)
    models.text_encoder_2.requires_grad_(False)

    models.unet.add_adapter(
        get_lora_config(
            rank=config.rank,
            lora_alpha_scale=config.lora_alpha_scale,
            target_modules=config.lora_target_modules,
        )
    )
    if config.text_encoder_lora_enabled:
        text_lora = get_lora_config(
            rank=config.rank,
            lora_alpha_scale=config.lora_alpha_scale,
            target_modules=config.text_encoder_lora_target_modules,
        )
        models.text_encoder.add_adapter(text_lora)
        models.text_encoder_2.add_adapter(text_lora)

    if config.mixed_precision == "fp16":
        cast_training_params(models.unet, dtype=torch.float32)
        if config.text_encoder_lora_enabled:
            cast_training_params([models.text_encoder, models.text_encoder_2], dtype=torch.float32)

    if config.enable_xformers_memory_efficient_attention:
        _enable_xformers(models.unet)
    if config.gradient_checkpointing:
        models.unet.enable_gradient_checkpointing()
        if config.text_encoder_lora_enabled:
            if hasattr(models.text_encoder, "gradient_checkpointing_enable"):
                models.text_encoder.gradient_checkpointing_enable()
            if hasattr(models.text_encoder_2, "gradient_checkpointing_enable"):
                models.text_encoder_2.gradient_checkpointing_enable()

    info: Dict[str, object] = {
        "model_family": "sdxl",
        "baseline_mode": config.baseline_mode,
        "lora_active": True,
        "lora_target_modules": list(config.lora_target_modules),
        "text_encoder_lora_enabled": bool(config.text_encoder_lora_enabled),
        "text_encoder_lora_target_modules": (
            list(config.text_encoder_lora_target_modules)
            if config.text_encoder_lora_enabled
            else []
        ),
        "trainable_parameter_names": {
            "unet": _trainable_param_names(models.unet),
            "text_encoder": _trainable_param_names(models.text_encoder),
            "text_encoder_2": _trainable_param_names(models.text_encoder_2),
            "vae": _trainable_param_names(models.vae),
        },
    }
    describe_component_trainability("UNet", models.unet)
    describe_component_trainability("Text encoder 1", models.text_encoder)
    describe_component_trainability("Text encoder 2", models.text_encoder_2)
    describe_component_trainability("VAE", models.vae)
    return info


def unwrap_model(model, accelerator=None):
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def get_trainable_params(models: ModelComponents) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for module in (models.unet, models.text_encoder, models.text_encoder_2):
        params.extend(param for param in module.parameters() if param.requires_grad)
    return params


def get_trainable_models(models: ModelComponents) -> List[torch.nn.Module]:
    return [
        module
        for module in (models.unet, models.text_encoder, models.text_encoder_2)
        if any(param.requires_grad for param in module.parameters())
    ]


def trainable_component_names(models: ModelComponents) -> List[str]:
    names = []
    for name, module in (
        ("unet", models.unet),
        ("text_encoder", models.text_encoder),
        ("text_encoder_2", models.text_encoder_2),
        ("vae", models.vae),
    ):
        if any(param.requires_grad for param in module.parameters()):
            names.append(name)
    return names


def _extract_hidden_states(output) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if hasattr(output, "hidden_states"):
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output[0]
        return pooled, output.hidden_states
    return output[0], output[-1]


def encode_prompt(
    *,
    text_encoder: torch.nn.Module,
    text_encoder_2: torch.nn.Module,
    input_ids_one: torch.Tensor,
    input_ids_two: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode SDXL prompts with both text encoders."""
    prompt_embeds_list = []
    pooled_prompt_embeds = None
    for input_ids, encoder in ((input_ids_one, text_encoder), (input_ids_two, text_encoder_2)):
        output = encoder(input_ids, output_hidden_states=True, return_dict=False)
        pooled, hidden_states = _extract_hidden_states(output)
        prompt_embeds = hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)
        pooled_prompt_embeds = pooled

    assert pooled_prompt_embeds is not None
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(pooled_prompt_embeds.shape[0], -1)
    return prompt_embeds, pooled_prompt_embeds


def build_time_ids(
    original_sizes: Sequence[Tuple[int, int]],
    crop_top_lefts: Sequence[Tuple[int, int]],
    target_sizes: Sequence[Tuple[int, int]],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = [
        list(tuple(original_size) + tuple(crop_top_left) + tuple(target_size))
        for original_size, crop_top_left, target_size in zip(original_sizes, crop_top_lefts, target_sizes)
    ]
    return torch.tensor(values, device=device, dtype=dtype)


def create_save_model_hook(models_ref: ModelComponents, accelerator):
    """Create an Accelerate save hook for SDXL LoRA checkpointing."""

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            from peft.utils import get_peft_model_state_dict

            unet_lora_layers = None
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None
            expected = {
                id(unwrap_model(models_ref.unet, accelerator)): "unet",
                id(unwrap_model(models_ref.text_encoder, accelerator)): "text_encoder",
                id(unwrap_model(models_ref.text_encoder_2, accelerator)): "text_encoder_2",
            }
            for model in models:
                unwrapped = unwrap_model(model, accelerator)
                role = expected.get(id(unwrapped))
                if role == "unet":
                    unet_lora_layers = get_peft_model_state_dict(unwrapped)
                elif role == "text_encoder":
                    text_encoder_lora_layers = get_peft_model_state_dict(unwrapped)
                elif role == "text_encoder_2":
                    text_encoder_2_lora_layers = get_peft_model_state_dict(unwrapped)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")
                weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=output_dir,
                unet_lora_layers=unet_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
                text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                safe_serialization=True,
            )

    return save_model_hook


def _resolve_lora_weights_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    for filename in LORA_WEIGHT_FILENAMES:
        candidate = path / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing SDXL LoRA weights under {path}")


def create_load_model_hook(models_ref: ModelComponents, accelerator, mixed_precision: Optional[str] = None):
    """Create an Accelerate load hook for SDXL LoRA checkpoint resume."""

    def load_model_hook(models, input_dir):
        model_by_role = {}
        expected_types = (
            type(unwrap_model(models_ref.unet, accelerator)),
            type(unwrap_model(models_ref.text_encoder, accelerator)),
            type(unwrap_model(models_ref.text_encoder_2, accelerator)),
        )
        while models:
            model = models.pop()
            unwrapped = unwrap_model(model, accelerator)
            if isinstance(unwrapped, expected_types[0]):
                model_by_role["unet"] = model
            elif isinstance(unwrapped, expected_types[1]):
                model_by_role["text_encoder"] = model
            elif isinstance(unwrapped, expected_types[2]):
                model_by_role["text_encoder_2"] = model
            else:
                raise ValueError(f"Unexpected load model: {model.__class__}")

        lora_state_dict, _ = StableDiffusionXLPipeline.lora_state_dict(input_dir)
        from peft.utils import set_peft_model_state_dict

        if "unet" in model_by_role:
            unet_state = {
                key.replace("unet.", ""): value
                for key, value in lora_state_dict.items()
                if key.startswith("unet.")
            }
            set_peft_model_state_dict(
                model_by_role["unet"],
                convert_unet_state_dict_to_peft(unet_state),
                adapter_name="default",
            )
        for role in ("text_encoder", "text_encoder_2"):
            if role not in model_by_role:
                continue
            prefix = f"{role}."
            state = {
                key.replace(prefix, ""): value
                for key, value in lora_state_dict.items()
                if key.startswith(prefix)
            }
            if state:
                set_peft_model_state_dict(model_by_role[role], state, adapter_name="default")
        if mixed_precision == "fp16":
            cast_training_params(list(model_by_role.values()), dtype=torch.float32)

    return load_model_hook


def get_canonical_output_dir(config: TrainingConfig) -> str:
    if config.output_dir != "sdxl-model-finetuned-lora":
        return config.output_dir
    dataset_key = config.dataset_id or "custom"
    return str(checkpoints_root() / "stable_diffusion_xl" / "lora_runs" / f"{dataset_key}_sdxl_lora_r{config.rank}")


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
        "model_family": "sdxl",
        "baseline_mode": config.baseline_mode,
        "training_mode": "lora",
        "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
        "revision": config.revision,
        "variant": config.variant,
        "dataset_id": config.dataset_id,
        "train_data_dir": config.train_data_dir,
        "train_split": config.train_split,
        "resolution": config.resolution,
        "normalization_mode": normalization_mode,
        "prompt_text": config.resolved_prompt_text(),
        "learning_rate": config.learning_rate,
        "max_train_steps": config.max_train_steps,
        "rank": config.rank,
        "lora_alpha_scale": config.lora_alpha_scale,
        "lora_target_modules": list(config.lora_target_modules),
        "text_encoder_lora_enabled": bool(config.text_encoder_lora_enabled),
        "text_encoder_lora_target_modules": list(config.text_encoder_lora_target_modules),
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
        model_kind="sdxl_stage1_lora",
        model_family="sdxl",
        base_model=manifest.get("pretrained_model_name_or_path"),
        adapters=[
            {
                "kind": "lora",
                "rank": manifest.get("rank"),
                "alpha_scale": manifest.get("lora_alpha_scale"),
                "text_encoder_lora_enabled": manifest.get("text_encoder_lora_enabled"),
            }
        ],
        task={"kind": "stable_diffusion_stage1"},
        dataset={"dataset_id": manifest.get("dataset_id"), "train_split": manifest.get("train_split")},
        normalization={"mode": manifest.get("normalization_mode")},
        metadata={"stage1_manifest": STAGE1_MANIFEST_NAME},
    )
    write_artifact_manifest(output_dir, artifact)
    logger.info("Saved SDXL stage-1 manifest to %s", manifest_path)
    return manifest_path


def load_stage1_manifest(stage1_dir: str | Path) -> Dict[str, object]:
    manifest_path = Path(stage1_dir) / STAGE1_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing SDXL stage-1 manifest at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_sdxl_stage1_pipeline(
    *,
    stage1_dir: str | Path,
    base_model: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> tuple[StableDiffusionXLPipeline, Dict[str, object]]:
    """Load an SDXL stage-1 LoRA artifact into a StableDiffusionXLPipeline."""
    manifest = load_stage1_manifest(stage1_dir)
    if manifest.get("model_family") not in {None, "sdxl"}:
        raise ValueError(f"Expected SDXL stage-1 manifest, got {manifest.get('model_family')!r}")
    base_model_id = base_model or manifest["pretrained_model_name_or_path"]
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        revision=manifest.get("revision"),
        variant=manifest.get("variant"),
        torch_dtype=torch_dtype,
    )
    pipeline.load_lora_weights(str(stage1_dir))
    return pipeline, manifest

