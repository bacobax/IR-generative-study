"""Model loading, trainability, and artifact helpers for SD layout stage-2."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import CLIPTextModel

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available

from src.algorithms.stable_diffusion.models import (
    STAGE1_MANIFEST_NAME,
    TEXT_ENCODER_EXPORT_DIRNAME,
    UNET_EXPORT_DIRNAME,
    VAE_EXPORT_DIRNAME,
    get_weight_dtype,
)
from src.core.configs.sd_layout_config import SDLayoutTrainConfig
from src.core.training_utils import cast_training_params
from src.models.regiondiffusion import (
    RegionDiffusionUNetWrapper,
    iter_regiondiff_adapter_parameters,
    load_regiondiff_config,
    regiondiff_config_dict,
    save_regiondiff_config,
)
from src.models.regiondiffusion_factory import (
    configure_regiondiff_trainability,
    regiondiff_optimizer_param_groups,
)


logger = logging.getLogger(__name__)

STAGE2_MANIFEST_NAME = "stage2_layout_manifest.json"
STAGE2_BASE_UNET_CONFIG = "base_unet_config.json"
STAGE2_REGIONDIFF_CONFIG = "regiondiff_config.json"
STAGE2_UNET_WEIGHTS = "regiondiff_unet.safetensors"
STAGE2_CHECKPOINT_UNET_WEIGHTS = "regiondiff_unet_checkpoint.safetensors"
LEGACY_ACCELERATE_MODEL_WEIGHTS = "model.safetensors"


try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except ImportError:  # pragma: no cover - fallback when safetensors is unavailable
    safe_load_file = None
    safe_save_file = None


@dataclass
class SDLayoutModelComponents:
    """Container for Stage-2 layout-conditioned SD modules."""

    unet: RegionDiffusionUNetWrapper
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    tokenizer: object
    noise_scheduler: DDPMScheduler
    weight_dtype: torch.dtype


def _load_json(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save_json(path: str | Path, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)


def _config_to_dict(config) -> Dict[str, object]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    return dict(config)


def _load_state_dict(path: str | Path) -> Dict[str, torch.Tensor]:
    path = str(path)
    if path.endswith(".safetensors") and safe_load_file is not None:
        return safe_load_file(path)
    return torch.load(path, map_location="cpu")


def _save_state_dict(path: str | Path, state_dict: Dict[str, torch.Tensor]) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".safetensors") and safe_save_file is not None:
        safe_save_file(state_dict, path)
        return
    torch.save(state_dict, path)


def _find_stage1_manifest_dir(stage1_dir: str | Path) -> Path:
    stage1_dir = Path(stage1_dir)
    if (stage1_dir / STAGE1_MANIFEST_NAME).is_file():
        return stage1_dir
    if (stage1_dir.parent / STAGE1_MANIFEST_NAME).is_file():
        return stage1_dir.parent
    raise FileNotFoundError(
        f"Could not find {STAGE1_MANIFEST_NAME} in {stage1_dir} or its parent."
    )


def _load_stage1_manifest(stage1_dir: str | Path) -> Dict[str, object]:
    manifest_dir = _find_stage1_manifest_dir(stage1_dir)
    return _load_json(manifest_dir / STAGE1_MANIFEST_NAME)


def _resolve_latest_checkpoint(stage1_dir: Path) -> Path:
    checkpoint_dirs = sorted(
        (path for path in stage1_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")),
        key=lambda path: int(path.name.split("-")[1]),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-* directories found under {stage1_dir}")
    return checkpoint_dirs[-1]


def _resolve_stage1_checkpoint_path(stage1_dir: Path, checkpoint: Optional[str]) -> Optional[Path]:
    if checkpoint is None:
        return None
    if checkpoint == "latest":
        return _resolve_latest_checkpoint(stage1_dir)
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return checkpoint_path
    joined = stage1_dir / checkpoint
    if joined.exists():
        return joined
    raise FileNotFoundError(f"Could not resolve stage1 checkpoint {checkpoint!r}")


def resolve_stage1_initialization(
    *,
    stage1_dir: str,
    stage1_checkpoint: Optional[str],
) -> Dict[str, object]:
    """Resolve the concrete Stage-1 source used to initialize Stage-2."""
    manifest_dir = _find_stage1_manifest_dir(stage1_dir)
    manifest = _load_stage1_manifest(manifest_dir)
    baseline_mode = str(manifest["baseline_mode"])
    checkpoint_path = _resolve_stage1_checkpoint_path(manifest_dir, stage1_checkpoint)

    if checkpoint_path is not None:
        return {
            "manifest_dir": str(manifest_dir),
            "baseline_mode": baseline_mode,
            "source_kind": "checkpoint",
            "resolved_path": str(checkpoint_path),
            "manifest": manifest,
        }

    if baseline_mode == "sd_ir_unet" and (manifest_dir / UNET_EXPORT_DIRNAME).is_dir():
        return {
            "manifest_dir": str(manifest_dir),
            "baseline_mode": baseline_mode,
            "source_kind": "export",
            "resolved_path": str(manifest_dir),
            "manifest": manifest,
        }

    if baseline_mode == "sd_ir_lora":
        return {
            "manifest_dir": str(manifest_dir),
            "baseline_mode": baseline_mode,
            "source_kind": "export",
            "resolved_path": str(manifest_dir),
            "manifest": manifest,
        }

    raise FileNotFoundError(
        "Unable to resolve a Stage-1 export or checkpoint for Stage-2 initialization. "
        f"stage1_dir={stage1_dir!r}, stage1_checkpoint={stage1_checkpoint!r}"
    )


def _load_unet_weights_from_checkpoint(unet: UNet2DConditionModel, checkpoint_path: Path) -> None:
    if checkpoint_path.is_dir():
        if (checkpoint_path / "model.safetensors").is_file():
            weights_path = checkpoint_path / "model.safetensors"
        elif (checkpoint_path / "pytorch_model.bin").is_file():
            weights_path = checkpoint_path / "pytorch_model.bin"
        else:
            raise FileNotFoundError(f"No UNet weights found under {checkpoint_path}")
    else:
        weights_path = checkpoint_path

    state_dict = _load_state_dict(weights_path)
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Stage-1 UNet checkpoint did not match the SD1.5 UNet architecture. "
            f"Missing keys={missing[:5]}, unexpected keys={unexpected[:5]}"
        )


def _load_exported_components_if_present(
    pipeline: StableDiffusionPipeline,
    export_dir: Path,
    *,
    torch_dtype: Optional[torch.dtype],
) -> StableDiffusionPipeline:
    text_encoder_dir = export_dir / TEXT_ENCODER_EXPORT_DIRNAME
    vae_dir = export_dir / VAE_EXPORT_DIRNAME
    if text_encoder_dir.is_dir():
        pipeline.text_encoder = CLIPTextModel.from_pretrained(
            text_encoder_dir,
            torch_dtype=torch_dtype,
        )
    if vae_dir.is_dir():
        pipeline.vae = AutoencoderKL.from_pretrained(
            vae_dir,
            torch_dtype=torch_dtype,
        )
    return pipeline


def load_stage1_pipeline_for_stage2(
    *,
    config: SDLayoutTrainConfig,
    device: torch.device | None = None,
) -> Tuple[StableDiffusionPipeline, Dict[str, object]]:
    """Load the base SD1.5 pipeline and initialize it from the requested Stage-1 source."""
    weight_dtype = get_weight_dtype(config.training.mixed_precision)
    init_info = resolve_stage1_initialization(
        stage1_dir=config.stage1.stage1_dir,
        stage1_checkpoint=config.stage1.stage1_checkpoint,
    )
    manifest = init_info["manifest"]

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.stage1.pretrained_model_name_or_path,
        revision=config.stage1.revision,
        variant=config.stage1.variant,
        torch_dtype=weight_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    resolved_path = Path(str(init_info["resolved_path"]))
    if init_info["baseline_mode"] == "sd_ir_unet":
        if init_info["source_kind"] == "export":
            unet_dir = resolved_path / UNET_EXPORT_DIRNAME
            pipeline.unet = UNet2DConditionModel.from_pretrained(
                unet_dir,
                torch_dtype=weight_dtype,
            )
            pipeline = _load_exported_components_if_present(
                pipeline,
                resolved_path,
                torch_dtype=weight_dtype,
            )
        else:
            _load_unet_weights_from_checkpoint(pipeline.unet, resolved_path)
    else:
        pipeline.load_lora_weights(str(resolved_path))
        if hasattr(pipeline, "fuse_lora"):
            pipeline.fuse_lora()
        if hasattr(pipeline, "unload_lora_weights"):
            pipeline.unload_lora_weights()

    if device is not None:
        pipeline.to(device)

    init_info = {
        **init_info,
        "resolved_stage1_checkpoint": str(resolved_path),
        "stage1_manifest": manifest,
    }
    return pipeline, init_info


def build_class_text_features(
    *,
    tokenizer,
    text_encoder: CLIPTextModel,
    category_id_to_name: Dict[int, str],
    device: torch.device,
) -> torch.Tensor:
    """Encode category names with the frozen CLIP text tower used by SD1.5."""
    num_classes = max(category_id_to_name.keys(), default=-1) + 1
    names = [
        str(category_id_to_name.get(class_id, f"class {class_id}")).replace("_", " ")
        for class_id in range(max(num_classes, 1))
    ]
    tokenized = tokenizer(
        names,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = getattr(tokenized, "attention_mask", None)

    with torch.no_grad():
        outputs = text_encoder(
            tokenized.input_ids.to(device),
            attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            return_dict=True,
        )

    hidden_states = outputs.last_hidden_state.detach().to(torch.float32)
    if attention_mask is None:
        attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.long)
    else:
        attention_mask = attention_mask.to(hidden_states.device)
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.cpu()


def build_regiondiff_unet_wrapper(
    *,
    base_unet: UNet2DConditionModel,
    tokenizer,
    text_encoder: CLIPTextModel,
    category_id_to_name: Dict[int, str],
    config: SDLayoutTrainConfig,
    device: torch.device,
) -> RegionDiffusionUNetWrapper:
    """Construct the layout-conditioned RegionDiff wrapper around an SD U-Net."""
    class_text_features = build_class_text_features(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        category_id_to_name=category_id_to_name,
        device=device,
    )
    return RegionDiffusionUNetWrapper(
        base_unet=base_unet,
        class_text_features=class_text_features,
        category_id_to_name=category_id_to_name,
        layout_token_dim=config.region.layout_token_dim,
        bbox_fourier_dim=config.region.bbox_fourier_dim,
        same_class_position_slots=config.region.same_class_position_slots,
        use_background_token=config.region.use_background_token,
        active_region_resolutions=config.region.active_region_resolutions,
    )


def load_layout_model_components(
    *,
    config: SDLayoutTrainConfig,
    category_id_to_name: Dict[int, str],
    device: torch.device,
) -> Tuple[SDLayoutModelComponents, Dict[str, object]]:
    """Load the Stage-2 training components and initialize from Stage-1."""
    pipeline, init_info = load_stage1_pipeline_for_stage2(config=config, device=device)
    weight_dtype = get_weight_dtype(config.training.mixed_precision)
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.stage1.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.to(device=device, dtype=weight_dtype)
    pipeline.vae.to(device=device, dtype=weight_dtype)
    pipeline.text_encoder.to(device=device, dtype=weight_dtype)

    wrapped_unet = build_regiondiff_unet_wrapper(
        base_unet=pipeline.unet,
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        category_id_to_name=category_id_to_name,
        config=config,
        device=device,
    )
    wrapped_unet.to(device=device)
    wrapped_unet.base_unet.to(device=device, dtype=weight_dtype)

    return (
        SDLayoutModelComponents(
            unet=wrapped_unet,
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
        ),
        init_info,
    )


def _set_requires_grad_for_prefixes(module: torch.nn.Module, prefixes: List[str]) -> List[str]:
    matched: List[str] = []
    for name, param in module.named_parameters():
        is_match = any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        if is_match:
            param.requires_grad_(True)
            matched.append(name)
    return matched


def configure_layout_trainability(
    *,
    models: SDLayoutModelComponents,
    config: SDLayoutTrainConfig,
) -> Dict[str, object]:
    """Freeze the base model and unfreeze RegionDiff adapters plus optional U-Net blocks."""
    models.vae.requires_grad_(False)
    models.text_encoder.requires_grad_(False)

    if config.training.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError("xformers is not available. Please install it correctly.")
        models.unet.base_unet.enable_xformers_memory_efficient_attention()
        logger.info("xFormers memory efficient attention enabled for stage-2 layout training")

    if config.training.gradient_checkpointing:
        models.unet.base_unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled for stage-2 layout training")

    info = configure_regiondiff_trainability(
        wrapper=models.unet,
        train_mode=config.training.train_mode,
        partial_backbone_modules=config.training.partial_unet_modules,
        mixed_precision=config.training.mixed_precision,
    )
    info["active_region_resolutions"] = list(config.region.active_region_resolutions)
    info["prompt_mode"] = config.prompt.prompt_mode
    return info


def build_optimizer_param_groups(
    *,
    models: SDLayoutModelComponents,
    config: SDLayoutTrainConfig,
    accelerator_processes: int,
) -> List[Dict[str, object]]:
    """Build optimizer param groups for adapters and optional backbone blocks."""
    adapter_lr = config.training.adapter_learning_rate
    backbone_lr = config.training.backbone_learning_rate
    if config.training.scale_lr:
        scale = config.training.gradient_accumulation_steps * config.data.batch_size * accelerator_processes
        adapter_lr *= scale
        backbone_lr *= scale

    return regiondiff_optimizer_param_groups(
        wrapper=models.unet,
        adapter_learning_rate=adapter_lr,
        backbone_learning_rate=backbone_lr,
    )


def build_stage2_manifest(
    *,
    config: SDLayoutTrainConfig,
    init_info: Dict[str, object],
    trainability_info: Dict[str, object],
    category_id_to_name: Dict[int, str],
) -> Dict[str, object]:
    """Build the persisted stage-2 artifact manifest."""
    return {
        "model_type": "regiondiff_sd_layout",
        "pretrained_model_name_or_path": config.stage1.pretrained_model_name_or_path,
        "revision": config.stage1.revision,
        "variant": config.stage1.variant,
        "dataset_id": config.data.dataset_id,
        "train_split": config.data.train_split,
        "val_split": config.data.val_split,
        "resolution": config.data.resolution,
        "prompt_mode": config.prompt.prompt_mode,
        "constant_prompt": config.prompt.constant_prompt,
        "thermal_scene_suffix": config.prompt.thermal_scene_suffix,
        "use_captions_if_available": config.prompt.use_captions_if_available,
        "train_mode": config.training.train_mode,
        "partial_unet_modules": list(config.training.partial_unet_modules),
        "adapter_learning_rate": config.training.adapter_learning_rate,
        "backbone_learning_rate": config.training.backbone_learning_rate,
        "area_loss": {
            "enabled": config.area_loss.enabled,
            "alpha": config.area_loss.alpha,
            "background_weight": config.area_loss.background_weight,
            "min_weight": config.area_loss.min_weight,
            "max_weight": config.area_loss.max_weight,
        },
        "region": {
            "active_region_resolutions": list(config.region.active_region_resolutions),
            "layout_token_dim": config.region.layout_token_dim,
            "bbox_fourier_dim": config.region.bbox_fourier_dim,
            "same_class_position_slots": config.region.same_class_position_slots,
            "use_background_token": config.region.use_background_token,
        },
        "category_id_to_name": {str(key): value for key, value in category_id_to_name.items()},
        "stage1_initialization": init_info,
        "trainability": trainability_info,
    }


def create_stage2_save_model_hook(unet, accelerator):
    """Create a save hook so Accelerate checkpoints can handle the wrapped RegionDiff UNet."""

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        for model in list(models):
            if not isinstance(model, type(unet)):
                raise ValueError(f"Unexpected save model: {model.__class__}")

            state_dict = {
                key: value.detach().cpu().to(torch.float32)
                for key, value in accelerator.unwrap_model(model).state_dict().items()
            }
            _save_state_dict(os.path.join(output_dir, STAGE2_CHECKPOINT_UNET_WEIGHTS), state_dict)

            if weights:
                weights.pop()

    return save_model_hook


def create_stage2_load_model_hook(unet, accelerator):
    """Create a load hook so Accelerate checkpoints can restore the wrapped RegionDiff UNet."""

    def load_model_hook(models, input_dir):
        target_model = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unet)):
                target_model = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

        if target_model is None:
            raise ValueError("No RegionDiff UNet model was provided to the load hook.")

        checkpoint_path = os.path.join(input_dir, STAGE2_CHECKPOINT_UNET_WEIGHTS)
        if not os.path.isfile(checkpoint_path):
            legacy_path = os.path.join(input_dir, LEGACY_ACCELERATE_MODEL_WEIGHTS)
            if os.path.isfile(legacy_path):
                checkpoint_path = legacy_path
            else:
                raise FileNotFoundError(
                    "Could not find a saved RegionDiff UNet checkpoint under "
                    f"{input_dir}. Expected either {STAGE2_CHECKPOINT_UNET_WEIGHTS!r} "
                    f"or {LEGACY_ACCELERATE_MODEL_WEIGHTS!r}."
                )

        state_dict = _load_state_dict(checkpoint_path)
        target_model.load_state_dict(state_dict)

    return load_model_hook


def save_stage2_layout_artifact(
    *,
    output_dir: str,
    unet: RegionDiffusionUNetWrapper,
    config: SDLayoutTrainConfig,
    init_info: Dict[str, object],
    trainability_info: Dict[str, object],
) -> None:
    """Save the final stage-2 artifact for later reuse."""
    os.makedirs(output_dir, exist_ok=True)
    state_dict = {
        key: value.detach().cpu().to(torch.float32)
        for key, value in unet.state_dict().items()
    }
    _save_state_dict(os.path.join(output_dir, STAGE2_UNET_WEIGHTS), state_dict)
    _save_json(
        os.path.join(output_dir, STAGE2_BASE_UNET_CONFIG),
        _config_to_dict(unet.base_unet.config),
    )
    save_regiondiff_config(
        regiondiff_config_dict(unet),
        os.path.join(output_dir, STAGE2_REGIONDIFF_CONFIG),
    )
    _save_json(
        os.path.join(output_dir, STAGE2_MANIFEST_NAME),
        build_stage2_manifest(
            config=config,
            init_info=init_info,
            trainability_info=trainability_info,
            category_id_to_name=unet.category_id_to_name,
        ),
    )
    logger.info("Saved stage-2 layout artifact to %s", output_dir)


def load_stage2_layout_pipeline(
    *,
    stage2_dir: str,
    torch_dtype: Optional[torch.dtype] = None,
    base_model: Optional[str] = None,
) -> Tuple[StableDiffusionPipeline, Dict[str, object]]:
    """Load a saved stage-2 RegionDiff artifact as a StableDiffusionPipeline."""
    stage2_dir_path = Path(stage2_dir)
    manifest = _load_json(stage2_dir_path / STAGE2_MANIFEST_NAME)
    region_config = load_regiondiff_config(stage2_dir_path / STAGE2_REGIONDIFF_CONFIG)
    base_model_id = base_model or manifest["pretrained_model_name_or_path"]

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        revision=manifest.get("revision"),
        variant=manifest.get("variant"),
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    category_id_to_name = {
        int(key): str(value)
        for key, value in region_config.get("category_id_to_name", {}).items()
    }

    base_unet_config = _load_json(stage2_dir_path / STAGE2_BASE_UNET_CONFIG)
    base_unet = UNet2DConditionModel.from_config(base_unet_config)
    if torch_dtype is not None:
        base_unet.to(dtype=torch_dtype)

    class_text_features = build_class_text_features(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        category_id_to_name=category_id_to_name,
        device=next(pipeline.text_encoder.parameters()).device,
    )
    wrapped_unet = RegionDiffusionUNetWrapper(
        base_unet=base_unet,
        class_text_features=class_text_features,
        category_id_to_name=category_id_to_name,
        layout_token_dim=int(region_config["layout_token_dim"]),
        bbox_fourier_dim=int(region_config["bbox_fourier_dim"]),
        same_class_position_slots=int(region_config["same_class_position_slots"]),
        use_background_token=bool(region_config["use_background_token"]),
        active_region_resolutions=region_config["active_region_resolutions"],
    )
    state_dict = _load_state_dict(stage2_dir_path / STAGE2_UNET_WEIGHTS)
    missing, unexpected = wrapped_unet.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Stage-2 RegionDiff artifact did not load cleanly. "
            f"Missing keys={missing[:5]}, unexpected keys={unexpected[:5]}"
        )
    if torch_dtype is not None:
        wrapped_unet.base_unet.to(dtype=torch_dtype)
    pipeline.unet = wrapped_unet
    return pipeline, manifest
