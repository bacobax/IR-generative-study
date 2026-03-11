"""VAE construction helpers for the latent-space flow-matching pipeline.

Mirrors the pattern in :mod:`src.models.fm_unet` but for
``generative.networks.nets.AutoencoderKL``.
"""

import json
import os
from typing import Any, Dict, Optional, Union

import torch
from generative.networks.nets import AutoencoderKL

from src.core.paths import vae_config_path, vae_config_x8_path


# ── Known built-in config paths (resolved via src.core.paths) ────────────────
VAE_CONFIG = str(vae_config_path())
VAE_CONFIG_X8 = str(vae_config_x8_path())


def load_vae_config(
    path: Optional[str] = None,
    *,
    config_dict: Optional[Dict[str, Any]] = None,
    combined_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and return a VAE config dictionary.

    Resolution order (first non-``None`` wins):
      1. *config_dict* – already a dict, returned as-is (shallow copy).
      2. *path* – path to a single JSON file with VAE kwargs.
      3. *combined_json* – path to a JSON that has a ``"VAE"`` key.

    Raises
    ------
    ValueError
        If none of the three sources is provided.
    """
    if config_dict is not None:
        return dict(config_dict)

    if path is not None:
        return _read_json(path)

    if combined_json is not None:
        combined = _read_json(combined_json)
        if "VAE" not in combined:
            raise KeyError(
                f"Combined JSON at {combined_json!r} has no 'VAE' key. "
                f"Available keys: {list(combined.keys())}"
            )
        return dict(combined["VAE"])

    raise ValueError(
        "Provide at least one of: config_dict, path, or combined_json"
    )


def build_vae_from_config(
    config: Dict[str, Any],
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> AutoencoderKL:
    """Instantiate an ``AutoencoderKL`` from a config dict."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return AutoencoderKL(**config).to(device)


def save_vae_config(config: Dict[str, Any], path: str) -> None:
    """Write a VAE config dict to *path* as formatted JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def load_vae_weights(
    vae: AutoencoderKL,
    path: str,
    *,
    strict: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
) -> AutoencoderKL:
    """Load state-dict into an existing VAE instance."""
    state = torch.load(path, map_location=map_location or "cpu")
    missing, unexpected = vae.load_state_dict(state, strict=strict)
    if (not strict) or missing or unexpected:
        print(f"[load_vae_weights] strict={strict}")
        if missing:
            print("  Missing keys:", missing)
        if unexpected:
            print("  Unexpected keys:", unexpected)
    return vae


def save_vae_weights(vae: AutoencoderKL, path: str) -> None:
    """Save VAE state-dict to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(vae.state_dict(), path)


def freeze_vae(vae: AutoencoderKL) -> AutoencoderKL:
    """Set VAE to eval mode and freeze all parameters."""
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


# ── internal ──────────────────────────────────────────────────────────────────

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
