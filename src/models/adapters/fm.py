"""Flow-matching model adapter for native UNet/VAE construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import torch

from src.core.registry import REGISTRIES
from src.models.adapters.base import ModelAdapterBase, ModelBuildRequest, ModelBundle


def _register_once(registry, name: str, value: Any) -> None:
    if name not in registry:
        registry.register(name)(value)


def _infer_vae_downsample_factor(vae_config: Mapping[str, Any]) -> int:
    """Infer the VAE spatial downsample factor from its channel schedule."""
    num_channels = vae_config.get("num_channels")
    if isinstance(num_channels, (list, tuple)) and num_channels:
        return 2 ** max(0, len(num_channels) - 1)

    block_out_channels = vae_config.get("block_out_channels")
    if isinstance(block_out_channels, (list, tuple)) and block_out_channels:
        return 2 ** max(0, len(block_out_channels) - 1)

    down_block_types = vae_config.get("down_block_types")
    if isinstance(down_block_types, (list, tuple)) and down_block_types:
        return 2 ** max(0, len(down_block_types) - 1)

    raise ValueError(
        "VAE config must define a non-empty num_channels, block_out_channels, "
        "or down_block_types sequence to infer sample_size."
    )


def resolve_fm_unet_sample_size(config: Any, vae_config: Mapping[str, Any] | None) -> int | None:
    """Resolve pixel or latent UNet sample size from a train config."""
    data_cfg = getattr(config, "data", None)
    image_size = getattr(data_cfg, "image_size", None)
    if image_size is None:
        return None

    image_size = int(image_size)
    if vae_config is None:
        return image_size

    downsample_factor = _infer_vae_downsample_factor(vae_config)
    if image_size % downsample_factor != 0:
        raise ValueError(
            f"image_size={image_size} is not divisible by VAE downsample factor "
            f"{downsample_factor}"
        )
    return image_size // downsample_factor


@dataclass
class BuiltFMModelAdapter:
    """Per-build handle exposing native FM components and convenience helpers."""

    unet: Any
    vae: Any = None
    unet_config: Dict[str, Any] | None = None
    vae_config: Dict[str, Any] | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images with the VAE, or pass pixel-space tensors through."""
        if self.vae is None:
            return x
        z_mu, z_sigma = self.vae.encode(x)
        if hasattr(self.vae, "sampling"):
            return self.vae.sampling(z_mu, z_sigma)
        return z_mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents with the VAE, or pass pixel-space tensors through."""
        if self.vae is None:
            return z
        return self.vae.decode(z)


class FMModelAdapter(ModelAdapterBase):
    """Build the repo's native FM UNet and optional native/diffusers VAE."""

    def __init__(
        self,
        *,
        load_unet_config_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        build_unet_fn: Optional[Callable[..., Any]] = None,
        resolve_vae_config_fn: Optional[Callable[..., Dict[str, Any] | None]] = None,
        build_vae_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._load_unet_config_fn = load_unet_config_fn
        self._build_unet_fn = build_unet_fn
        self._resolve_vae_config_fn = resolve_vae_config_fn
        self._build_vae_fn = build_vae_fn

    def build_from_train_config(self, config: Any, *, device: Any = None) -> ModelBundle:
        """Build native FM components from the structured training config."""
        if device is None:
            device = config.resolved_device() if hasattr(config, "resolved_device") else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        model_cfg = getattr(config, "model", None)
        if model_cfg is None:
            raise ValueError("FMModelAdapter requires a config with a model section.")

        load_unet_config_fn = self._load_unet_config_fn
        build_unet_fn = self._build_unet_fn
        resolve_vae_config_fn = self._resolve_vae_config_fn
        build_vae_fn = self._build_vae_fn
        if (
            load_unet_config_fn is None
            or build_unet_fn is None
            or resolve_vae_config_fn is None
            or build_vae_fn is None
        ):
            from src.models.fm_unet import build_fm_unet_from_config, load_unet_config
            from src.models.vae import build_vae_from_config, resolve_vae_config_from_model_config

            load_unet_config_fn = load_unet_config_fn or load_unet_config
            build_unet_fn = build_unet_fn or build_fm_unet_from_config
            resolve_vae_config_fn = resolve_vae_config_fn or resolve_vae_config_from_model_config
            build_vae_fn = build_vae_fn or build_vae_from_config

        unet_cfg = load_unet_config_fn(getattr(model_cfg, "unet_config", None))
        vae_cfg = None
        vae = None
        has_vae = (
            getattr(model_cfg, "vae_config", None) is not None
            or getattr(model_cfg, "vae_pretrained_model_name_or_path", None) is not None
        )
        if has_vae:
            vae_cfg = resolve_vae_config_fn(model_cfg)

        resolved_sample_size = resolve_fm_unet_sample_size(config, vae_cfg)
        if resolved_sample_size is not None:
            unet_cfg = dict(unet_cfg)
            unet_cfg["sample_size"] = resolved_sample_size

        unet = build_unet_fn(unet_cfg, device=device)
        if vae_cfg is not None:
            vae = build_vae_fn(vae_cfg, device=device)

        fm_adapter = BuiltFMModelAdapter(
            unet=unet,
            vae=vae,
            unet_config=dict(unet_cfg),
            vae_config=dict(vae_cfg) if vae_cfg is not None else None,
        )
        return ModelBundle(
            primary=unet,
            components={
                "unet": unet,
                "fm_adapter": fm_adapter,
                **({"vae": vae} if vae is not None else {}),
            },
            adapter_name="fm",
            model_name="native_fm",
            metadata={
                "unet_config": fm_adapter.unet_config,
                "vae_config": fm_adapter.vae_config,
            },
        )

    def build(self, request: ModelBuildRequest) -> ModelBundle:
        config = request.config
        if isinstance(config, Mapping) and "train_config" in config:
            config = config["train_config"]
        if config is None:
            config = request.options.get("train_config")
        if config is None:
            raise ValueError("FMModelAdapter.build requires a train config.")
        return self.build_from_train_config(config, device=request.device)


_fm_model_adapter = FMModelAdapter()
_register_once(REGISTRIES.model_adapter, "fm", _fm_model_adapter)
_register_once(REGISTRIES.model_adapter, "native_fm", _fm_model_adapter)


__all__ = [
    "BuiltFMModelAdapter",
    "FMModelAdapter",
    "resolve_fm_unet_sample_size",
]
