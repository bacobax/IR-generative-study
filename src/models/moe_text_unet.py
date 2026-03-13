"""Text-conditioned UNet wrapper with fixed-weight MOE adapters.

Wraps a diffusers.UNet2DConditionModel and injects a small adapter bank
at selected internal locations. Routing is fixed and uniform (1/K).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

from src.conditioning.condition_router import ConditionRouter
from src.models.fm_text_unet import load_text_unet_config, build_text_fm_unet
from src.models.moe_adapter import AdapterBank


# ── canonical config paths ─────────────────────────────────────────────────
TEXT_MOE_UNET_CONFIG = "configs/models/fm/text_unet_config.json"


@dataclass
class TextMOEConfig:
    """Configuration for TextMOEUNet wrapper and adapter bank."""

    num_experts: int = 4
    adapter_hidden_dim: Optional[int] = None
    adapter_dropout: float = 0.0
    router_hidden_dims: Optional[Tuple[int, ...]] = None


def load_text_moe_unet_config(
    path: Optional[str] = None,
    *,
    config_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load UNet and MOE configs from a JSON path or dict.

    If the loaded dict contains ``"UNET"`` or ``"unet"`` keys, the
    corresponding sub-dict is used as the UNet config, and ``"MOE"`` or
    ``"moe"`` is used for MOE settings. Otherwise, the dict is treated as
    a plain UNet config and MOE defaults are used.
    """
    if config_dict is not None:
        data = config_dict
    elif path is not None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Provide either 'path' or 'config_dict'.")

    if "UNET" in data or "unet" in data:
        unet_cfg = data.get("UNET") or data.get("unet")
        moe_cfg = data.get("MOE") or data.get("moe") or {}
        return dict(unet_cfg), dict(moe_cfg)

    return dict(data), {}


def save_text_moe_unet_config(
    unet_config: Dict[str, Any],
    moe_config: Dict[str, Any],
    path: str,
) -> None:
    """Save combined UNet+MOE config to *path*.

    Stores a JSON object with ``UNET`` and ``MOE`` top-level keys.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"UNET": unet_config, "MOE": moe_config}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


class TextMOEUNet(nn.Module):
    """Wrapper for UNet2DConditionModel with fixed uniform MOE adapters.

    The first version injects a single AdapterBank at the UNet mid-block.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        *,
        num_experts: int = 4,
        adapter_hidden_dim: Optional[int] = None,
        adapter_dropout: float = 0.0,
        router_hidden_dims: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()

        self.unet = unet
        self.num_experts = num_experts
        self.adapter_hidden_dim = adapter_hidden_dim
        self.adapter_dropout = adapter_dropout
        self.router_hidden_dims = router_hidden_dims

        cross_dim = self._resolve_cross_attention_dim(unet)
        router_dims = list(router_hidden_dims) if router_hidden_dims is not None else None
        self.router = ConditionRouter(
            input_dim=cross_dim,
            num_experts=num_experts,
            hidden_dims=router_dims,
        )

        mid_channels = self._resolve_mid_block_channels(unet)
        self.mid_adapter = AdapterBank(
            mid_channels,
            num_experts,
            hidden_dim=adapter_hidden_dim,
            dropout=adapter_dropout,
        )

        self._hooks = []
        self._register_mid_block_hook()
        self._active_router_weights: Optional[torch.Tensor] = None

    @property
    def config(self):
        """Expose the wrapped UNet config for compatibility."""
        return getattr(self.unet, "config", None)

    def _resolve_mid_block_channels(self, unet: UNet2DConditionModel) -> int:
        if hasattr(unet, "config") and hasattr(unet.config, "block_out_channels"):
            return int(unet.config.block_out_channels[-1])
        raise ValueError("Unable to infer mid-block channels from UNet config")

    def _resolve_cross_attention_dim(self, unet: UNet2DConditionModel) -> int:
        if hasattr(unet, "config") and hasattr(unet.config, "cross_attention_dim"):
            return int(unet.config.cross_attention_dim)
        raise ValueError("Unable to infer cross_attention_dim from UNet config")

    def _register_mid_block_hook(self) -> None:
        if not hasattr(self.unet, "mid_block"):
            raise ValueError("UNet does not have a mid_block to attach adapters")

        def _hook(_module, _inputs, output):
            return self._apply_adapter(output)

        handle = self.unet.mid_block.register_forward_hook(_hook)
        self._hooks.append(handle)

    def _apply_adapter(self, output):
        weights = self._active_router_weights
        if isinstance(output, tuple):
            if not output:
                return output
            first = output[0]
            adapted = self.mid_adapter(first, weights=weights)
            return (adapted,) + output[1:]
        return self.mid_adapter(output, weights=weights)

    def compute_router_weights(self, pooled_text_embeds: torch.Tensor) -> torch.Tensor:
        """Compute router weights from pooled text embeddings."""
        return self.router(pooled_text_embeds)

    def forward(self, sample: torch.Tensor, timestep, **kwargs):
        """Forward pass with the same signature as UNet2DConditionModel.

        Accepts optional ``pooled_text_embeds`` or ``router_weights`` for
        routing. If neither is provided, uniform weights are used.
        """
        pooled = kwargs.pop("pooled_text_embeds", None)
        router_weights = kwargs.pop("router_weights", None)

        if router_weights is None and pooled is not None:
            router_weights = self.compute_router_weights(pooled)

        self._active_router_weights = router_weights
        try:
            return self.unet(sample, timestep, **kwargs)
        finally:
            self._active_router_weights = None


def build_text_moe_unet(
    config: Dict[str, Any],
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> TextMOEUNet:
    """Instantiate a TextMOEUNet from a config dict.

    The *config* can be either a plain UNet2DConditionModel config, or a
    combined config with ``UNET`` and ``MOE`` keys.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    unet_cfg, moe_cfg = load_text_moe_unet_config(config_dict=config)
    unet = build_text_fm_unet(unet_cfg, device=device)

    moe_defaults = TextMOEConfig()
    merged = {
        "num_experts": moe_cfg.get("num_experts", moe_defaults.num_experts),
        "adapter_hidden_dim": moe_cfg.get("adapter_hidden_dim", moe_defaults.adapter_hidden_dim),
        "adapter_dropout": moe_cfg.get("adapter_dropout", moe_defaults.adapter_dropout),
        "router_hidden_dims": moe_cfg.get("router_hidden_dims", moe_defaults.router_hidden_dims),
    }

    model = TextMOEUNet(unet, **merged)
    model.to(device)
    return model


# ── registry ──────────────────────────────────────────────────────────────
from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.model_builder.register("text_moe_unet")(build_text_moe_unet)
