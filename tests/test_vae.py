"""Tests for VAE adapters and config resolution helpers."""

from __future__ import annotations

import torch

from src.core.configs.fm_config import ModelConfig
from src.models.vae import DiffusersAutoencoderAdapter, resolve_vae_config_from_model_config


class _FakePosterior:
    def __init__(self, mean: torch.Tensor, stddev: torch.Tensor) -> None:
        self.mean = mean
        self.stddev = stddev


class _FakeEncodeOutput:
    def __init__(self, latent_dist: _FakePosterior) -> None:
        self.latent_dist = latent_dist


class _FakeDecodeOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class _FakeDiffusersVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type(
            "Cfg",
            (),
            {"in_channels": 3, "scaling_factor": 0.5},
        )()
        self.proj = torch.nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.decode_proj = torch.nn.Conv2d(4, 3, kernel_size=1, bias=False)
        self.last_encode_input = None
        self.last_decode_input = None

    def encode(self, x: torch.Tensor) -> _FakeEncodeOutput:
        self.last_encode_input = x.detach().clone()
        mean = self.proj(x)
        stddev = torch.ones_like(mean) * 0.25
        return _FakeEncodeOutput(_FakePosterior(mean=mean, stddev=stddev))

    def decode(self, z: torch.Tensor) -> _FakeDecodeOutput:
        self.last_decode_input = z.detach().clone()
        return _FakeDecodeOutput(self.decode_proj(z))


def test_diffusers_adapter_repeats_single_channel_input_and_scales_latents() -> None:
    autoencoder = _FakeDiffusersVAE()
    adapter = DiffusersAutoencoderAdapter(autoencoder)

    x = torch.randn(2, 1, 8, 8)
    z_mu, z_sigma = adapter.encode(x)

    assert autoencoder.last_encode_input is not None
    assert autoencoder.last_encode_input.shape == (2, 3, 8, 8)
    assert torch.allclose(autoencoder.last_encode_input[:, 0], x[:, 0])
    assert torch.allclose(autoencoder.last_encode_input[:, 1], x[:, 0])
    assert torch.allclose(autoencoder.last_encode_input[:, 2], x[:, 0])
    assert z_mu.shape == (2, 4, 8, 8)
    assert torch.allclose(z_sigma, torch.full_like(z_sigma, 0.125))

    z = torch.ones(2, 4, 8, 8)
    _ = adapter.decode(z)
    assert autoencoder.last_decode_input is not None
    assert torch.allclose(autoencoder.last_decode_input, torch.full_like(z, 2.0))


def test_resolve_vae_config_from_model_config_prefers_pretrained_sd_spec(monkeypatch) -> None:
    captured = {}

    def _fake_load(pretrained_model_name_or_path, *, subfolder="vae", revision=None, variant=None):
        captured["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        captured["subfolder"] = subfolder
        captured["revision"] = revision
        captured["variant"] = variant
        return {"_backend": "diffusers_autoencoder_kl", "latent_channels": 4}

    monkeypatch.setattr("src.models.vae.load_diffusers_vae_config", _fake_load)

    model_cfg = ModelConfig(
        vae_config=None,
        vae_weights=None,
        vae_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        vae_pretrained_subfolder="vae",
        vae_revision="main",
        vae_variant=None,
    )

    resolved = resolve_vae_config_from_model_config(model_cfg)

    assert resolved == {"_backend": "diffusers_autoencoder_kl", "latent_channels": 4}
    assert captured == {
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "subfolder": "vae",
        "revision": "main",
        "variant": None,
    }
