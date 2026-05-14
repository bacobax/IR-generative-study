"""Shared utilities for active inference samplers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch


def make_vae_latent_codec(vae) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Return no-grad VAE encode/decode closures used by latent samplers."""

    @torch.no_grad()
    def encode(x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = vae.encode(x)
        return vae.sampling(z_mu, z_sigma)

    @torch.no_grad()
    def decode(z: torch.Tensor) -> torch.Tensor:
        return vae.decode(z)

    return encode, decode


def pick_latest_checkpoint(folder: str | os.PathLike[str], prefix: str, suffix: str = ".pt") -> str | None:
    """Pick the highest numbered checkpoint matching ``prefix{N}suffix``."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return None

    best_i: int | None = None
    best_path: Path | None = None
    for path in folder_path.iterdir():
        name = path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        middle = name[len(prefix) : -len(suffix)]
        try:
            idx = int(middle)
        except ValueError:
            continue
        if best_i is None or idx > best_i:
            best_i = idx
            best_path = path
    return str(best_path) if best_path is not None else None


def resolve_preferred_or_latest_checkpoint(
    folder: str | os.PathLike[str],
    preferred_filename: str,
    latest_prefix: str,
    suffix: str = ".pt",
) -> str | None:
    """Return preferred checkpoint if present, otherwise latest numbered fallback."""
    preferred = Path(folder) / preferred_filename
    if preferred.is_file():
        return str(preferred)
    return pick_latest_checkpoint(folder, latest_prefix, suffix=suffix)


def load_checkpoint_state(path: str | os.PathLike[str], map_location, nested_key: str = "unet_state"):
    """Load a PyTorch checkpoint and unwrap a nested state dict when present."""
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and nested_key in state:
        return state[nested_key]
    return state


def get_unet_sample_shape(
    unet,
    override: Optional[Tuple[int, int, int]] = None,
) -> Tuple[int, int, int]:
    """Return ``(C, H, W)`` from the UNet config or an explicit override."""
    if override is not None:
        return override
    in_channels = unet.config.in_channels
    sample_size = unet.config.sample_size
    if isinstance(sample_size, Sequence):
        h, w = sample_size
    else:
        h = w = sample_size
    return (in_channels, h, w)
