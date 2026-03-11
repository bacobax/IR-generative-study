"""Flow-matching UNet construction helpers.

Centralises the logic for loading a UNet config JSON and instantiating
a ``diffusers.UNet2DModel``.  Both *stable* (latent-space, 4-channel)
and *non-stable* (pixel-space, 1-channel) configs are supported.

Usage::

    from src.models.fm_unet import load_unet_config, build_fm_unet_from_config

    cfg = load_unet_config(str(stable_unet_config_path()))
    unet = build_fm_unet_from_config(cfg, device="cuda")
"""

import json
import os
from typing import Any, Dict, Optional, Union

import torch
from diffusers import UNet2DModel

from src.core.paths import stable_unet_config_path, non_stable_unet_config_path


# ── Known built-in config paths (resolved via src.core.paths) ────────────────
STABLE_UNET_CONFIG = str(stable_unet_config_path())
NON_STABLE_UNET_CONFIG = str(non_stable_unet_config_path())


def load_unet_config(
    path: Optional[str] = None,
    *,
    config_dict: Optional[Dict[str, Any]] = None,
    combined_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and return a UNet config dictionary.

    Resolution order (first non-``None`` wins):
      1. *config_dict* – already a dict, returned as-is (shallow copy).
      2. *path* – path to a single JSON file with UNet kwargs.
      3. *combined_json* – path to a JSON file that has a ``"UNET"`` key
         containing the UNet kwargs (legacy convenience format).

    Returns
    -------
    dict
        A plain dict suitable for ``UNet2DModel(**cfg)``.

    Raises
    ------
    ValueError
        If none of the three sources is provided.
    FileNotFoundError
        If the specified JSON file does not exist.
    """
    if config_dict is not None:
        return dict(config_dict)

    if path is not None:
        return _read_json(path)

    if combined_json is not None:
        combined = _read_json(combined_json)
        if "UNET" not in combined:
            raise KeyError(
                f"Combined JSON at {combined_json!r} has no 'UNET' key. "
                f"Available keys: {list(combined.keys())}"
            )
        return dict(combined["UNET"])

    raise ValueError(
        "Provide at least one of: config_dict, path, or combined_json"
    )


def build_fm_unet_from_config(
    config: Dict[str, Any],
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> UNet2DModel:
    """Instantiate a ``UNet2DModel`` from a config dict.

    Parameters
    ----------
    config : dict
        UNet kwargs (e.g. from :func:`load_unet_config`).
    device : str or torch.device, optional
        Target device.  Defaults to ``"cuda"`` if available, else ``"cpu"``.

    Returns
    -------
    UNet2DModel
        The freshly initialised (random weights) model on *device*.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return UNet2DModel(**config).to(device)


def save_unet_config(config: Dict[str, Any], path: str) -> None:
    """Write a UNet config dict to *path* as formatted JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


# ── internal ──────────────────────────────────────────────────────────────────

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── registry ──────────────────────────────────────────────────────────────────
from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.model_builder.register("default_unet", default=True)(build_fm_unet_from_config)
