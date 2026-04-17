"""Sampling utilities for unconditional latent Stable Diffusion."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.tensorboard import SummaryWriter

from src.algorithms.inference.flow_matching_sampler import (
    _default_from_norm_to_display,
    _pick_latest,
    get_unet_sample_shape,
)
from src.models.fm_unet import build_fm_unet_from_config, load_unet_config
from src.models.vae import (
    build_vae_from_config,
    freeze_vae,
    is_diffusers_vae_config,
    load_vae_config,
    load_vae_weights,
)


class UnconditionalStableDiffusionSampler:
    """Standalone sampler for unconditional latent diffusion checkpoints."""

    def __init__(
        self,
        unet: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        *,
        device: Optional[Union[str, torch.device]] = None,
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.from_norm_to_display = from_norm_to_display or _default_from_norm_to_display
        self._sample_shape = sample_shape
        self._encoder = encoder or (lambda x: x)
        self._decoder = decoder or (lambda z: z)

    @classmethod
    def from_stable(
        cls,
        unet: UNet2DModel,
        vae,
        noise_scheduler: DDPMScheduler,
        **kwargs,
    ) -> "UnconditionalStableDiffusionSampler":
        """Build a sampler wired to a frozen VAE for latent decoding."""

        @torch.no_grad()
        def _encode(x: torch.Tensor) -> torch.Tensor:
            z_mu, z_sigma = vae.encode(x)
            return vae.sampling(z_mu, z_sigma)

        @torch.no_grad()
        def _decode(z: torch.Tensor) -> torch.Tensor:
            return vae.decode(z)

        return cls(
            unet,
            noise_scheduler,
            encoder=_encode,
            decoder=_decode,
            **kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config,
        *,
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> "UnconditionalStableDiffusionSampler":
        """Load a saved unconditional latent SD checkpoint tree."""
        import os

        device = config.resolved_device()
        pipeline_dir = config.output.model_dir if hasattr(config, "output") else config.pipeline_dir
        unet_dir = os.path.join(pipeline_dir, "UNET")
        vae_dir = os.path.join(pipeline_dir, "VAE")
        scheduler_dir = os.path.join(pipeline_dir, "SCHEDULER")

        unet_cfg = load_unet_config(os.path.join(unet_dir, "config.json"))
        unet = build_fm_unet_from_config(unet_cfg, device=device)
        unet_w = os.path.join(unet_dir, "unet_sd_uncond_best.pt")
        if not os.path.isfile(unet_w):
            unet_w = _pick_latest(unet_dir, "unet_sd_uncond_epoch_")
        if unet_w is None or not os.path.isfile(unet_w):
            raise FileNotFoundError(f"No unconditional SD UNet weights found in {unet_dir}")
        state = torch.load(unet_w, map_location=device)
        if isinstance(state, dict) and "unet_state" in state:
            state = state["unet_state"]
        unet.load_state_dict(state)
        unet.eval()

        vae_cfg = load_vae_config(os.path.join(vae_dir, "config.json"))
        vae = build_vae_from_config(vae_cfg, device=device)
        vae_w = os.path.join(vae_dir, "vae_best.pt")
        if not os.path.isfile(vae_w):
            vae_w = _pick_latest(vae_dir, "vae_epoch_")
        if vae_w is None or not os.path.isfile(vae_w):
            if not is_diffusers_vae_config(vae_cfg):
                raise FileNotFoundError(f"No VAE weights found in {vae_dir}")
        else:
            load_vae_weights(vae, vae_w, map_location=device)
        freeze_vae(vae)

        noise_scheduler = DDPMScheduler.from_pretrained(scheduler_dir)
        return cls.from_stable(
            unet,
            vae,
            noise_scheduler,
            device=device,
            from_norm_to_display=from_norm_to_display,
            sample_shape=getattr(getattr(config, "sampling", None), "sample_shape", None),
        )

    def _shape(self, override: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
        return get_unet_sample_shape(self.unet, override=override or self._sample_shape)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(z)

    @torch.no_grad()
    def sample(
        self,
        *,
        steps: int = 50,
        batch_size: int = 4,
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """Draw latent samples by iterating the discrete diffusion scheduler."""
        self.unet.eval()
        latents = torch.randn(batch_size, *self._shape(sample_shape), device=self.device)
        self.noise_scheduler.set_timesteps(steps, device=self.device)

        for timestep in self.noise_scheduler.timesteps:
            model_pred = self.unet(
                latents,
                timestep.expand(batch_size),
            ).sample
            latents = self.noise_scheduler.step(model_pred, timestep, latents).prev_sample
        return latents

    @torch.no_grad()
    def log_samples_to_tensorboard(
        self,
        writer: SummaryWriter,
        epoch: int,
        *,
        steps: int = 50,
        batch_size: int = 4,
        tag: str = "sd_uncond/generated",
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        latents = self.sample(steps=steps, batch_size=batch_size, sample_shape=sample_shape)
        images = self.decode(latents)
        images = self.from_norm_to_display(images).clamp(0.0, 1.0)
        writer.add_images(tag, images, epoch)


from src.core.registry import REGISTRIES  # noqa: E402


REGISTRIES.sampler.register("sd_uncond")(UnconditionalStableDiffusionSampler)
