"""DiT construction helpers for unconditional latent diffusion."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional, Union

import torch

from src.core.diffusers_compat import import_diffusers_attr


_METADATA_KEYS = {
    "architecture",
    "variant",
    "hidden_size",
    "unconditional_class_label",
}


class UnconditionalDiTWrapper(torch.nn.Module):
    """Adapt ``DiTTransformer2DModel`` to the repo's unconditional UNet contract."""

    def __init__(self, transformer: torch.nn.Module, *, class_label: int = 0) -> None:
        super().__init__()
        self.transformer = transformer
        self.class_label = int(class_label)

    @property
    def config(self):
        return self.transformer.config

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        class_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if class_labels is None:
            class_labels = torch.full(
                (sample.shape[0],),
                self.class_label,
                dtype=torch.long,
                device=sample.device,
            )
        return self.transformer(
            sample,
            timestep=timestep,
            class_labels=class_labels,
            **kwargs,
        )


def load_dit_config(
    path: Optional[str] = None,
    *,
    config_dict: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load and return a DiT config dictionary."""
    if config_dict is not None:
        return dict(config_dict)
    if path is None:
        raise ValueError("Provide either path or config_dict.")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dit_config(config: Mapping[str, Any], path: str) -> None:
    """Write a DiT config dictionary to *path* as formatted JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(config), handle, indent=2, sort_keys=True)


def resolve_dit_config_for_latent_diffusion(
    config: Mapping[str, Any],
    *,
    sample_size: int,
    latent_channels: int,
    num_train_timesteps: int,
) -> Dict[str, Any]:
    """Resolve train-config-dependent DiT kwargs while preserving metadata."""
    resolved = dict(config)
    resolved["architecture"] = "dit"
    resolved["sample_size"] = int(sample_size)
    resolved["in_channels"] = int(latent_channels)
    resolved["out_channels"] = int(latent_channels)
    resolved["num_embeds_ada_norm"] = int(num_train_timesteps)

    if "depth" in resolved:
        if "num_layers" not in resolved:
            resolved["num_layers"] = int(resolved["depth"])
        resolved.pop("depth", None)

    patch_size = int(resolved.get("patch_size", 1))
    if patch_size <= 0:
        raise ValueError(f"DiT patch_size must be positive, got {patch_size}.")
    if int(sample_size) % patch_size != 0:
        raise ValueError(
            f"DiT sample_size={sample_size} must be divisible by patch_size={patch_size}."
        )

    hidden_size = resolved.get("hidden_size")
    if hidden_size is not None:
        expected = int(resolved["num_attention_heads"]) * int(resolved["attention_head_dim"])
        if int(hidden_size) != expected:
            raise ValueError(
                "DiT hidden_size must equal num_attention_heads * attention_head_dim, "
                f"got hidden_size={hidden_size}, expected {expected}."
            )
    return resolved


def _constructor_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    kwargs = dict(config)
    if "depth" in kwargs and "num_layers" not in kwargs:
        kwargs["num_layers"] = int(kwargs["depth"])
    kwargs.pop("depth", None)
    for key in _METADATA_KEYS:
        kwargs.pop(key, None)
    return kwargs


def build_dit_from_config(
    config: Mapping[str, Any],
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> UnconditionalDiTWrapper:
    """Instantiate a wrapped ``DiTTransformer2DModel`` from a config dict."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        DiTTransformer2DModel = import_diffusers_attr("diffusers", "DiTTransformer2DModel")
    except AttributeError as exc:
        raise AttributeError(
            "diffusers.DiTTransformer2DModel is required for model.architecture='dit'. "
            "Use the repo's diffusers-dev environment or install a compatible diffusers version."
        ) from exc

    class_label = int(config.get("unconditional_class_label", 0))
    transformer = DiTTransformer2DModel(**_constructor_kwargs(config))
    wrapped = UnconditionalDiTWrapper(transformer, class_label=class_label)
    return torch.nn.Module.to(wrapped, device)


from src.core.registry import REGISTRIES  # noqa: E402


REGISTRIES.model_builder.register("dit_transformer_2d")(build_dit_from_config)
