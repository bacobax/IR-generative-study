"""Builder for text-conditioned flow-matching UNet (UNet2DConditionModel).

Mirrors ``fm_unet.py`` but instantiates ``diffusers.UNet2DConditionModel``
which includes cross-attention layers for consuming text encoder hidden
states.  The ``cross_attention_dim`` in the JSON config **must** match
the text encoder's ``hidden_size`` (e.g. 768 for CLIP ViT-L/14).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Union

import torch
from diffusers import UNet2DConditionModel


# ── canonical config paths ─────────────────────────────────────────────────
TEXT_UNET_CONFIG = "configs/models/fm/text_unet_config.json"


# ═══════════════════════════════════════════════════════════════════════════
# Config I/O
# ═══════════════════════════════════════════════════════════════════════════

def load_text_unet_config(
    path: Optional[str] = None,
    *,
    config_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a UNet2DConditionModel config from *path* or a dict.

    Resolution order: *config_dict* → *path*.
    """
    if config_dict is not None:
        return config_dict
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("Provide either 'path' or 'config_dict'.")


def save_text_unet_config(config: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


# ═══════════════════════════════════════════════════════════════════════════
# Model builder
# ═══════════════════════════════════════════════════════════════════════════

def build_text_fm_unet(
    config: Dict[str, Any],
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> UNet2DConditionModel:
    """Instantiate a ``UNet2DConditionModel`` from a JSON-style config dict."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return UNet2DConditionModel(**config).to(device)


# ── registry ──────────────────────────────────────────────────────────────
from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.model_builder.register("text_fm_unet")(build_text_fm_unet)
