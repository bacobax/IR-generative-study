"""Shared RegionDiff construction, trainability, and metadata helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from diffusers.training_utils import cast_training_params

from src.models.regiondiffusion import (
    RegionDiffusionModelWrapper,
    RegionDiffusionUNetWrapper,
    iter_regiondiff_adapter_parameters,
    regiondiff_config_dict,
    save_regiondiff_config,
)


REGIONDIFF_METADATA_NAME = "regiondiff_config.json"


def build_identity_class_features(
    category_id_to_name: Dict[int, str],
    *,
    min_dim: int = 1,
) -> torch.Tensor:
    """Build deterministic class features for non-text-conditioned backbones."""
    num_classes = max((int(key) for key in category_id_to_name), default=-1) + 1
    num_classes = max(num_classes, 1)
    feature_dim = max(int(min_dim), num_classes)
    features = torch.zeros(num_classes, feature_dim, dtype=torch.float32)
    features[:, :num_classes] = torch.eye(num_classes, dtype=torch.float32)
    return features


def build_text_class_features(
    *,
    tokenizer: Any,
    text_encoder: nn.Module,
    category_id_to_name: Dict[int, str],
    device: torch.device,
) -> torch.Tensor:
    """Encode category names with a frozen text encoder for SD-style RegionDiff."""
    num_classes = max((int(key) for key in category_id_to_name), default=-1) + 1
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
        attention_mask = torch.ones(
            hidden_states.shape[:2],
            device=hidden_states.device,
            dtype=torch.long,
        )
    else:
        attention_mask = attention_mask.to(hidden_states.device)
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.cpu()


def _config_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _category_mapping(category_id_to_name: Optional[Dict[int, str]], num_classes: Optional[int]) -> Dict[int, str]:
    if category_id_to_name:
        return {int(key): str(value) for key, value in category_id_to_name.items()}
    count = int(num_classes or 1)
    return {idx: f"class {idx}" for idx in range(max(count, 1))}


def build_regiondiff_wrapper(
    *,
    base_model: nn.Module,
    region_config: Any,
    category_id_to_name: Optional[Dict[int, str]] = None,
    class_text_features: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    backbone_kind: str,
    attachment_kind: Optional[str] = None,
) -> nn.Module:
    """Attach RegionDiff layout conditioning to a compatible denoiser."""
    categories = _category_mapping(category_id_to_name, num_classes)
    layout_token_dim = int(_config_get(region_config, "layout_token_dim", 256))
    if class_text_features is None:
        class_text_features = build_identity_class_features(
            categories,
            min_dim=layout_token_dim,
        )

    common_kwargs = dict(
        class_text_features=class_text_features,
        category_id_to_name=categories,
        layout_token_dim=layout_token_dim,
        bbox_fourier_dim=int(_config_get(region_config, "bbox_fourier_dim", 64)),
        same_class_position_slots=int(_config_get(region_config, "same_class_position_slots", 128)),
        use_background_token=bool(_config_get(region_config, "use_background_token", True)),
        active_region_resolutions=list(_config_get(region_config, "active_region_resolutions", [64, 32, 16])),
    )

    if backbone_kind == "sd_conditional_unet":
        return RegionDiffusionUNetWrapper(base_unet=base_model, **common_kwargs)

    return RegionDiffusionModelWrapper(
        base_model=base_model,
        backbone_kind=backbone_kind,
        attachment_kind=attachment_kind or "attention",
        **common_kwargs,
    )


def set_requires_grad_for_prefixes(module: nn.Module, prefixes: Iterable[str]) -> List[str]:
    """Set parameters matching any prefix trainable and return matched names."""
    prefix_list = list(prefixes)
    matched: List[str] = []
    for name, param in module.named_parameters():
        is_match = any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefix_list)
        if is_match:
            param.requires_grad_(True)
            matched.append(name)
    return matched


def configure_regiondiff_trainability(
    *,
    wrapper: nn.Module,
    train_mode: str = "adapters_only",
    partial_backbone_modules: Optional[Iterable[str]] = None,
    mixed_precision: Optional[str] = None,
) -> Dict[str, object]:
    """Freeze a RegionDiff model and unfreeze the requested parameter groups."""
    wrapper.requires_grad_(False)
    for parameter in iter_regiondiff_adapter_parameters(wrapper):
        parameter.requires_grad_(True)

    backbone_matches: List[str] = []
    if train_mode in {"adapters_plus_partial_unet", "adapters_plus_partial_backbone"}:
        base = getattr(wrapper, "base_unet", getattr(wrapper, "base_model", None))
        if base is None:
            raise ValueError("RegionDiff wrapper has no base model for partial backbone training.")
        backbone_matches = set_requires_grad_for_prefixes(base, list(partial_backbone_modules or []))
        if not backbone_matches:
            raise ValueError(
                "RegionDiff partial-backbone mode resolved to zero trainable parameters. "
                f"Provided prefixes: {list(partial_backbone_modules or [])!r}"
            )
    elif train_mode != "adapters_only":
        raise ValueError(
            f"Unknown RegionDiff train_mode={train_mode!r}. Expected 'adapters_only' "
            "or 'adapters_plus_partial_unet'."
        )

    if mixed_precision == "fp16":
        cast_training_params(wrapper, dtype=torch.float32)

    adapter_param_ids = {id(param) for param in iter_regiondiff_adapter_parameters(wrapper)}
    trainable_groups = {"adapters": [], "backbone": []}
    for name, param in wrapper.named_parameters():
        if not param.requires_grad:
            continue
        group_name = "adapters" if id(param) in adapter_param_ids else "backbone"
        trainable_groups[group_name].append(name)

    return {
        "train_mode": train_mode,
        "adapter_parameter_count": sum(
            param.numel()
            for param in wrapper.parameters()
            if param.requires_grad and id(param) in adapter_param_ids
        ),
        "backbone_parameter_count": sum(
            param.numel()
            for param in wrapper.parameters()
            if param.requires_grad and id(param) not in adapter_param_ids
        ),
        "trainable_parameter_groups": trainable_groups,
        "backbone_matches": backbone_matches,
        "num_region_blocks": int(getattr(wrapper, "num_region_blocks", 0)),
        "backbone_kind": str(getattr(wrapper, "backbone_kind", "unknown")),
        "attachment_kind": str(getattr(wrapper, "attachment_kind", "unknown")),
    }


def regiondiff_optimizer_param_groups(
    *,
    wrapper: nn.Module,
    adapter_learning_rate: float,
    backbone_learning_rate: float,
) -> List[Dict[str, object]]:
    """Build optimizer groups for trainable RegionDiff adapters and backbone params."""
    adapter_param_ids = {id(param) for param in iter_regiondiff_adapter_parameters(wrapper)}
    adapter_params = []
    backbone_params = []
    for param in wrapper.parameters():
        if not param.requires_grad:
            continue
        if id(param) in adapter_param_ids:
            adapter_params.append(param)
        else:
            backbone_params.append(param)

    groups: List[Dict[str, object]] = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": float(adapter_learning_rate), "name": "adapters"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(backbone_learning_rate), "name": "backbone"})
    if not groups:
        raise ValueError("No trainable parameters were configured for RegionDiff.")
    return groups


def regiondiff_state_dict(wrapper: nn.Module, *, adapters_only: bool = False) -> Dict[str, torch.Tensor]:
    """Return a RegionDiff state dict, optionally filtering to adapter weights."""
    state = wrapper.state_dict()
    if not adapters_only:
        return state
    return {
        key: value
        for key, value in state.items()
        if key.startswith("layout_tokenizer.") or ".region_adapter." in key
    }


def save_regiondiff_metadata(wrapper: nn.Module, output_dir: str, *, extra: Optional[Dict[str, Any]] = None) -> str:
    """Persist RegionDiff config metadata and return its path."""
    payload = regiondiff_config_dict(wrapper)
    if extra:
        payload.update(extra)
    path = os.path.join(output_dir, REGIONDIFF_METADATA_NAME)
    save_regiondiff_config(payload, path)
    return path


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def config_to_dict(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if isinstance(config, dict):
        return dict(config)
    return dict(vars(config))
