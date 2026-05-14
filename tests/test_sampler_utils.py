from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler, _pick_latest, get_unet_sample_shape
from src.algorithms.inference.layout_flow_matching_sampler import LayoutFlowMatchingSampler
from src.algorithms.inference.sampler_utils import (
    load_checkpoint_state,
    make_vae_latent_codec,
    pick_latest_checkpoint,
    resolve_preferred_or_latest_checkpoint,
)
from src.algorithms.inference.unconditional_sd_sampler import UnconditionalStableDiffusionSampler


class _TinyUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=4, sample_size=8)


class _FakeLatentVAE:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, x: torch.Tensor):
        self.calls.append("encode")
        latents = F.interpolate(x[:, :1], size=(8, 8), mode="bilinear", align_corners=False)
        latents = latents.repeat(1, 4, 1, 1)
        sigma = torch.ones_like(latents) * 0.5
        return latents, sigma

    def sampling(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        self.calls.append("sampling")
        del sigma
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.calls.append("decode")
        decoded = z[:, :1]
        return F.interpolate(decoded, size=(16, 16), mode="bilinear", align_corners=False)


def test_make_vae_latent_codec_uses_existing_encode_sampling_decode_contract() -> None:
    vae = _FakeLatentVAE()
    encode, decode = make_vae_latent_codec(vae)

    latents = encode(torch.randn(2, 1, 16, 16))
    decoded = decode(latents)

    assert vae.calls == ["encode", "sampling", "decode"]
    assert latents.shape == (2, 4, 8, 8)
    assert decoded.shape == (2, 1, 16, 16)
    assert not latents.requires_grad
    assert not decoded.requires_grad


def test_checkpoint_helpers_preserve_latest_and_preferred_resolution(tmp_path) -> None:
    (tmp_path / "unet_fm_epoch_2.pt").write_bytes(b"early")
    (tmp_path / "unet_fm_epoch_bad.pt").write_bytes(b"bad")
    latest = tmp_path / "unet_fm_epoch_10.pt"
    latest.write_bytes(b"latest")

    assert pick_latest_checkpoint(tmp_path, "unet_fm_epoch_") == str(latest)
    assert _pick_latest(str(tmp_path), "unet_fm_epoch_") == str(latest)
    assert resolve_preferred_or_latest_checkpoint(tmp_path, "unet_fm_best.pt", "unet_fm_epoch_") == str(latest)

    preferred = tmp_path / "unet_fm_best.pt"
    preferred.write_bytes(b"best")
    assert resolve_preferred_or_latest_checkpoint(tmp_path, "unet_fm_best.pt", "unet_fm_epoch_") == str(preferred)


def test_load_checkpoint_state_unwraps_nested_unet_state(tmp_path) -> None:
    nested = tmp_path / "nested.pt"
    plain = tmp_path / "plain.pt"
    nested_state = {"weight": torch.ones(1)}
    plain_state = {"bias": torch.zeros(1)}
    torch.save({"unet_state": nested_state, "epoch": 3}, nested)
    torch.save(plain_state, plain)

    loaded_nested = load_checkpoint_state(nested, map_location="cpu")
    loaded_plain = load_checkpoint_state(plain, map_location="cpu")
    assert torch.equal(loaded_nested["weight"], nested_state["weight"])
    assert torch.equal(loaded_plain["bias"], plain_state["bias"])


def test_sample_shape_compatibility_exports() -> None:
    unet = _TinyUNet()
    assert get_unet_sample_shape(unet) == (4, 8, 8)
    assert get_unet_sample_shape(unet, override=(1, 2, 3)) == (1, 2, 3)


def test_from_stable_sampler_codecs_decode_with_fake_vae() -> None:
    unet = _TinyUNet()
    vae = _FakeLatentVAE()
    latents = torch.randn(2, 4, 8, 8)

    fm_sampler = FlowMatchingSampler.from_stable(unet, vae, device="cpu", t_scale=1.0)
    layout_sampler = LayoutFlowMatchingSampler.from_stable(unet, vae, device="cpu", t_scale=1.0)
    sd_sampler = UnconditionalStableDiffusionSampler.from_stable(
        unet,
        vae,
        noise_scheduler=object(),
        device="cpu",
    )

    assert fm_sampler.decode(latents).shape == (2, 1, 16, 16)
    assert layout_sampler.decode(latents).shape == (2, 1, 16, 16)
    assert sd_sampler.decode(latents).shape == (2, 1, 16, 16)
