"""Sampling utilities for unconditional latent Stable Diffusion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import torch

from src.algorithms.inference.flow_matching_sampler import (
    _default_from_norm_to_display,
    _maybe_wrap_regiondiff_unet,
)
from src.algorithms.inference.sampler_utils import (
    get_unet_sample_shape,
    load_checkpoint_state,
    make_vae_latent_codec,
    resolve_preferred_or_latest_checkpoint,
)
from src.core.diffusers_compat import import_diffusers_attr
from src.models.dit import build_dit_from_config, load_dit_config
from src.models.fm_unet import build_fm_unet_from_config, load_unet_config
from src.models.vae import (
    build_vae_from_config,
    freeze_vae,
    is_diffusers_vae_config,
    load_vae_config,
    load_vae_weights,
)

if TYPE_CHECKING:
    from diffusers import DDPMScheduler, UNet2DModel
    from torch.utils.tensorboard import SummaryWriter


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
        encoder, decoder = make_vae_latent_codec(vae)

        return cls(
            unet,
            noise_scheduler,
            encoder=encoder,
            decoder=decoder,
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

        config_path = os.path.join(unet_dir, "config.json")
        unet_cfg = load_unet_config(config_path)
        architecture = str(unet_cfg.get("architecture", "unet") or "unet").lower()
        if architecture == "dit":
            unet = build_dit_from_config(load_dit_config(config_path), device=device)
        elif architecture == "unet":
            unet = build_fm_unet_from_config(unet_cfg, device=device)
            unet = _maybe_wrap_regiondiff_unet(
                unet,
                pipeline_dir=pipeline_dir,
                backbone_kind="sd_uncond_unet2d",
            )
        else:
            raise ValueError(
                f"Unsupported unconditional latent SD backbone architecture {architecture!r} "
                f"in {config_path}."
            )
        unet = torch.nn.Module.to(unet, device)
        unet_w = resolve_preferred_or_latest_checkpoint(
            unet_dir,
            "unet_sd_uncond_best.pt",
            "unet_sd_uncond_epoch_",
        )
        if unet_w is None or not os.path.isfile(unet_w):
            raise FileNotFoundError(f"No unconditional SD backbone weights found in {unet_dir}")
        state = load_checkpoint_state(unet_w, map_location=device)
        unet.load_state_dict(state)
        unet.eval()

        vae_cfg = load_vae_config(os.path.join(vae_dir, "config.json"))
        vae = build_vae_from_config(vae_cfg, device=device)
        vae_w = resolve_preferred_or_latest_checkpoint(
            vae_dir,
            "vae_best.pt",
            "vae_epoch_",
        )
        if vae_w is None or not os.path.isfile(vae_w):
            if not is_diffusers_vae_config(vae_cfg):
                raise FileNotFoundError(f"No VAE weights found in {vae_dir}")
        else:
            load_vae_weights(vae, vae_w, map_location=device)
        freeze_vae(vae)

        DDPMScheduler = import_diffusers_attr("diffusers", "DDPMScheduler")
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
    def sample_layout(
        self,
        batch: dict[str, torch.Tensor],
        *,
        steps: int = 50,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Draw latent samples with RegionDiff layout kwargs."""
        self.unet.eval()
        batch_size = int(batch["pixel_values"].shape[0])
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        latents = torch.randn(
            batch_size,
            *self._shape(sample_shape),
            generator=generator,
            device=self.device,
        )
        self.noise_scheduler.set_timesteps(steps, device=self.device)
        cond_kw = {
            "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(self.device),
            "labels": batch["labels"].to(self.device),
            "object_mask": batch["object_mask"].to(self.device),
        }

        for timestep in self.noise_scheduler.timesteps:
            model_pred = self.unet(
                latents,
                timestep.expand(batch_size),
                **cond_kw,
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
