"""Tests for unconditional latent Stable Diffusion training helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel

from src.algorithms.inference.unconditional_sd_sampler import UnconditionalStableDiffusionSampler
from src.algorithms.training.unconditional_sd_trainer import UnconditionalStableDiffusionTrainer
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.sd_uncond_config import (
    SDUncondTrainConfig,
    SDUncondOutputConfig,
    _FLAT_TO_NESTED,
    build_parser,
)


class _FakeLatentVAE(torch.nn.Module):
    def encode(self, x: torch.Tensor):
        latents = F.interpolate(x[:, :1], scale_factor=0.125, mode="bilinear", align_corners=False)
        latents = latents.repeat(1, 4, 1, 1)
        sigma = torch.ones_like(latents) * 0.5
        return latents, sigma

    def sampling(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        del sigma
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = z[:, :1]
        return F.interpolate(decoded, scale_factor=8.0, mode="bilinear", align_corners=False)


def _tiny_unet(sample_size: int = 8) -> UNet2DModel:
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        attention_head_dim=8,
        norm_num_groups=8,
    )


def test_sd_uncond_config_yaml_and_cli_override(tmp_path: Path) -> None:
    config_path = tmp_path / "sd_uncond.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_id: flir_private_proxy_alignment_v18",
                "training:",
                "  epochs: 12",
                "diffusion:",
                "  prediction_type: v_prediction",
            ]
        ),
        encoding="utf-8",
    )

    parser = build_parser()
    cli_argv = ["--config", str(config_path), "--epochs", "3", "--eval_every", "1"]
    args = parser.parse_args(cli_argv)
    cfg = merge_config_and_cli(
        SDUncondTrainConfig,
        str(config_path),
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
        cli_argv=cli_argv,
    )

    assert cfg.data.dataset_id == "flir_private_proxy_alignment_v18"
    assert cfg.training.epochs == 3
    assert cfg.training.eval_every == 1
    assert cfg.diffusion.prediction_type == "v_prediction"


def test_sd_uncond_trainer_from_config_resolves_latent_sample_size(monkeypatch, tmp_path: Path) -> None:
    unet_config_path = tmp_path / "tiny_unet.json"
    unet_config_path.write_text(
        json.dumps(
            {
                "sample_size": 128,
                "in_channels": 4,
                "out_channels": 4,
                "layers_per_block": 1,
                "block_out_channels": [32, 64],
                "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
                "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
                "norm_num_groups": 8,
                "attention_head_dim": 8,
            }
        ),
        encoding="utf-8",
    )

    fake_vae_cfg = {
        "_backend": "diffusers_autoencoder_kl",
        "latent_channels": 4,
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "down_block_types": [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
    }
    monkeypatch.setattr(
        "src.models.vae.resolve_vae_config_from_model_config",
        lambda model_cfg: dict(fake_vae_cfg),
    )
    monkeypatch.setattr(
        "src.models.vae.build_vae_from_config",
        lambda cfg, device=None: _FakeLatentVAE(),
    )

    cfg = SDUncondTrainConfig(
        output=SDUncondOutputConfig(model_dir=str(tmp_path / "sd_uncond_run")),
        device="cpu",
    )
    cfg.data.image_size = 512
    cfg.model.unet_config = str(unet_config_path)
    cfg.model.vae_config = None
    cfg.model.vae_weights = None
    cfg.model.vae_pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

    trainer = UnconditionalStableDiffusionTrainer.from_config(cfg)

    assert trainer.vae is not None
    assert trainer.unet.config.sample_size == 64
    assert trainer.unet.config.in_channels == 4
    assert trainer.noise_scheduler.config.prediction_type == "epsilon"


def test_sd_uncond_diffusion_step_supports_epsilon_and_v_prediction() -> None:
    latents = torch.randn(2, 4, 8, 8)

    epsilon_cfg = SimpleNamespace(
        num_train_timesteps=32,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.012,
        prediction_type="epsilon",
        noise_offset=0.0,
        snr_gamma=None,
    )
    trainer_epsilon = UnconditionalStableDiffusionTrainer(
        _tiny_unet(),
        noise_scheduler=UnconditionalStableDiffusionTrainer.build_noise_scheduler(epsilon_cfg),
        diffusion_config=epsilon_cfg,
        device="cpu",
        model_dir="/tmp/sd_uncond_epsilon",
        vae=_FakeLatentVAE(),
    )
    loss_epsilon = trainer_epsilon.diffusion_step(latents)

    vpred_cfg = SimpleNamespace(
        num_train_timesteps=32,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.012,
        prediction_type="v_prediction",
        noise_offset=0.0,
        snr_gamma=5.0,
    )
    trainer_vpred = UnconditionalStableDiffusionTrainer(
        _tiny_unet(),
        noise_scheduler=UnconditionalStableDiffusionTrainer.build_noise_scheduler(vpred_cfg),
        diffusion_config=vpred_cfg,
        device="cpu",
        model_dir="/tmp/sd_uncond_vpred",
        vae=_FakeLatentVAE(),
    )
    loss_vpred = trainer_vpred.diffusion_step(latents)

    assert torch.isfinite(loss_epsilon)
    assert torch.isfinite(loss_vpred)
    assert float(loss_epsilon.item()) > 0.0
    assert float(loss_vpred.item()) > 0.0


def test_sd_uncond_sampler_runs_without_text_conditioning() -> None:
    sampler = UnconditionalStableDiffusionSampler.from_stable(
        _tiny_unet(),
        _FakeLatentVAE(),
        UnconditionalStableDiffusionTrainer.build_noise_scheduler(
            SimpleNamespace(
                num_train_timesteps=16,
                beta_schedule="scaled_linear",
                beta_start=0.00085,
                beta_end=0.012,
                prediction_type="epsilon",
            )
        ),
        device="cpu",
    )

    latents = sampler.sample(steps=3, batch_size=2, sample_shape=(4, 8, 8))
    decoded = sampler.decode(latents)

    assert latents.shape == (2, 4, 8, 8)
    assert decoded.shape == (2, 1, 64, 64)


def test_sd_uncond_preset_uses_shared_unet_and_sd15_vae() -> None:
    preset_path = Path("configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512.yaml")
    parser = build_parser()
    args = parser.parse_args(["--config", str(preset_path)])
    cfg = merge_config_and_cli(
        SDUncondTrainConfig,
        str(preset_path),
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
    )

    assert cfg.data.dataset_id == "flir_private_proxy_alignment_v18"
    assert cfg.data.image_size == 512
    assert cfg.model.unet_config == "configs/models/fm/stable_unet_x4_512.json"
    assert cfg.model.vae_config is None
    assert cfg.model.vae_weights is None
    assert cfg.model.vae_pretrained_model_name_or_path == "runwayml/stable-diffusion-v1-5"
    assert cfg.diffusion.prediction_type == "epsilon"
    assert cfg.sampling.sample_every == 10
    assert cfg.sampling.sample_steps == 40
    assert cfg.device is None
    assert cfg.output.model_dir.endswith("uncond_latent_flir_sd15_512")


def test_sd_uncond_train_from_config_skips_vae_weights_for_pretrained_vae(tmp_path: Path) -> None:
    trainer = UnconditionalStableDiffusionTrainer(
        _tiny_unet(),
        noise_scheduler=UnconditionalStableDiffusionTrainer.build_noise_scheduler(
            SimpleNamespace(
                num_train_timesteps=16,
                beta_schedule="scaled_linear",
                beta_start=0.00085,
                beta_end=0.012,
                prediction_type="epsilon",
                noise_offset=0.0,
                snr_gamma=None,
            )
        ),
        diffusion_config=SimpleNamespace(noise_offset=0.0, snr_gamma=None),
        device="cpu",
        model_dir=str(tmp_path / "sd_uncond_train"),
        vae=_FakeLatentVAE(),
    )

    calls = {}

    def fake_train(**kwargs):
        calls.update(kwargs)

    trainer.train = fake_train
    cfg = SDUncondTrainConfig(
        output=SDUncondOutputConfig(model_dir=str(tmp_path / "sd_uncond_train")),
        device="cpu",
    )
    cfg.model.vae_weights = "./vae_best.pt"
    cfg.model.vae_pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

    trainer.train_from_config(cfg, dataloader=object(), eval_dataloader=None)

    assert calls["pretrained_vae_path"] is None
