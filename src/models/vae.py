"""VAE construction helpers for the latent-space flow-matching pipeline.

Supports both the in-repo MONAI/generative ``AutoencoderKL`` used by the
original FM pipeline and a diffusers-backed Stable Diffusion VAE adapter.
"""

import json
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from generative.networks.nets import AutoencoderKL

from src.core.diffusers_compat import import_diffusers_attr
from src.core.paths import vae_config_path, vae_config_x4_path, vae_config_x8_path


# ── Known built-in config paths (resolved via src.core.paths) ────────────────
VAE_CONFIG = str(vae_config_path())
VAE_CONFIG_X4 = str(vae_config_x4_path())
VAE_CONFIG_X8 = str(vae_config_x8_path())
DIFFUSERS_VAE_BACKEND = "diffusers_autoencoder_kl"


def _posterior_std(posterior) -> torch.Tensor:
    if hasattr(posterior, "std") and posterior.std is not None:
        return posterior.std
    if hasattr(posterior, "stddev") and posterior.stddev is not None:
        return posterior.stddev
    if hasattr(posterior, "logvar") and posterior.logvar is not None:
        return torch.exp(0.5 * posterior.logvar)
    if hasattr(posterior, "var") and posterior.var is not None:
        return torch.sqrt(posterior.var)
    raise AttributeError(
        "Unable to extract posterior standard deviation. "
        "Expected one of: .std, .stddev, .logvar, or .var"
    )

class DiffusersAutoencoderAdapter(nn.Module):
    """Expose a diffusers ``AutoencoderKL`` through the FM VAE API."""

    def __init__(self, autoencoder: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder

    @property
    def config(self):
        return self.autoencoder.config

    @property
    def scaling_factor(self) -> float:
        return float(getattr(self.autoencoder.config, "scaling_factor", 1.0) or 1.0)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_channels = int(getattr(self.autoencoder.config, "in_channels", x.shape[1]))
        x = _match_channel_count(x, expected_channels)
        posterior = self.autoencoder.encode(x).latent_dist
        scale = self.scaling_factor
        return posterior.mean * scale, _posterior_std(posterior) * scale

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        return z_mu + torch.randn_like(z_mu) * z_sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        scale = self.scaling_factor
        return self.autoencoder.decode(z / scale).sample

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstructions and posterior stats like the repo VAE API."""
        z_mu, z_sigma = self.encode(x)
        z = self.sampling(z_mu, z_sigma)
        recon = self.decode(z)
        recon = _match_channel_count(recon, int(x.shape[1]))
        return recon, z_mu, z_sigma

    def state_dict(self, *args, **kwargs):
        return self.autoencoder.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.autoencoder.load_state_dict(state_dict, strict=strict)


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


def load_diffusers_vae_config(
    pretrained_model_name_or_path: str,
    *,
    subfolder: Optional[str] = "vae",
    revision: Optional[str] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a diffusers VAE config into serializable FM metadata."""
    diffusers_autoencoder_kl = _import_diffusers_autoencoder_kl()
    raw_config: Dict[str, Any]
    load_config = getattr(diffusers_autoencoder_kl, "load_config", None)
    if callable(load_config):
        load_kwargs: Dict[str, Any] = {}
        if subfolder:
            load_kwargs["subfolder"] = subfolder
        if revision is not None:
            load_kwargs["revision"] = revision
        raw = load_config(pretrained_model_name_or_path, **load_kwargs)
        raw_config = dict(raw)
    else:
        pretrained_kwargs: Dict[str, Any] = {}
        if subfolder:
            pretrained_kwargs["subfolder"] = subfolder
        if revision is not None:
            pretrained_kwargs["revision"] = revision
        if variant is not None:
            pretrained_kwargs["variant"] = variant
        model = diffusers_autoencoder_kl.from_pretrained(
            pretrained_model_name_or_path,
            **pretrained_kwargs,
        )
        config_to_dict = getattr(model.config, "to_dict", None)
        raw_config = dict(config_to_dict() if callable(config_to_dict) else model.config)

    raw_config["_backend"] = DIFFUSERS_VAE_BACKEND
    raw_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path
    raw_config["pretrained_subfolder"] = subfolder
    raw_config["pretrained_revision"] = revision
    raw_config["pretrained_variant"] = variant
    return raw_config


def resolve_vae_config_from_model_config(model_config) -> Optional[Dict[str, Any]]:
    """Resolve either a JSON VAE config or a pretrained diffusers VAE spec."""
    pretrained_model_name_or_path = getattr(
        model_config,
        "vae_pretrained_model_name_or_path",
        None,
    )
    if pretrained_model_name_or_path:
        return load_diffusers_vae_config(
            pretrained_model_name_or_path,
            subfolder=getattr(model_config, "vae_pretrained_subfolder", "vae"),
            revision=getattr(model_config, "vae_revision", None),
            variant=getattr(model_config, "vae_variant", None),
        )

    vae_config = getattr(model_config, "vae_config", None)
    if vae_config is None:
        return None
    return load_vae_config(vae_config)


def build_vae_from_config(
    config: Dict[str, Any],
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """Instantiate a VAE from a config dict."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_diffusers_vae_config(config):
        diffusers_autoencoder_kl = _import_diffusers_autoencoder_kl()
        pretrained_kwargs: Dict[str, Any] = {}
        subfolder = config.get("pretrained_subfolder", "vae")
        if subfolder:
            pretrained_kwargs["subfolder"] = subfolder
        if config.get("pretrained_revision") is not None:
            pretrained_kwargs["revision"] = config["pretrained_revision"]
        if config.get("pretrained_variant") is not None:
            pretrained_kwargs["variant"] = config["pretrained_variant"]
        autoencoder = diffusers_autoencoder_kl.from_pretrained(
            config["pretrained_model_name_or_path"],
            **pretrained_kwargs,
        )
        return DiffusersAutoencoderAdapter(autoencoder).to(device)
    return AutoencoderKL(**config).to(device)


def save_vae_config(config: Dict[str, Any], path: str) -> None:
    """Write a VAE config dict to *path* as formatted JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def load_vae_weights(
    vae,
    path: str,
    *,
    strict: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """Load state-dict into an existing VAE instance."""
    state = torch.load(path, map_location=map_location or "cpu")
    if isinstance(state, dict) and "vae_state" in state:
        state = state["vae_state"]
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


def freeze_vae(vae):
    """Set VAE to eval mode and freeze all parameters."""
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


# ── internal ──────────────────────────────────────────────────────────────────

def is_diffusers_vae_config(config: Dict[str, Any]) -> bool:
    return str(config.get("_backend", "")).lower() == DIFFUSERS_VAE_BACKEND

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _import_diffusers_autoencoder_kl():
    return import_diffusers_attr("diffusers", "AutoencoderKL")


def _match_channel_count(x: torch.Tensor, expected_channels: int) -> torch.Tensor:
    if int(x.shape[1]) == int(expected_channels):
        return x
    if int(x.shape[1]) == 1 and int(expected_channels) == 3:
        return x.repeat(1, 3, 1, 1)
    if int(x.shape[1]) == 3 and int(expected_channels) == 1:
        return x.mean(dim=1, keepdim=True)
    raise ValueError(
        f"Cannot adapt input with {int(x.shape[1])} channels to "
        f"expected VAE channels={int(expected_channels)}."
    )
