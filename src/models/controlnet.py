"""ControlNet model for flow-matching UNets.

Implements the ControlNet architecture from `Adding Conditional Control to
Text-to-Image Diffusion Models <https://arxiv.org/abs/2302.05543>`_
(Zhang et al., 2023) adapted to work with ``diffusers.UNet2DModel``.

Architecture overview (per the paper):
  1. The encoder half of a pre-trained UNet (down-blocks + mid-block) is
     **deep-copied** so that training starts from a good initialisation.
  2. A lightweight **conditioning encoder** downsamples a spatial conditioning
     signal (e.g. bounding-box mask at image resolution) to the latent
     resolution and adds it to the ``conv_in`` output.
  3. **Zero convolutions** (1x1 convolutions initialised to zero) are placed
     on every skip-connection output and on the mid-block output.  This
     guarantees that the ControlNet contributes *nothing* at the start of
     training, so the augmented UNet behaves identically to the base UNet.
  4. During the forward pass the frozen UNet receives the ControlNet
     residuals via :func:`unet_forward_with_controlnet`, which injects
     ``conditioning_scale * residual`` into each skip tensor and the
     mid-block tensor before the decoder consumes them.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import UNet2DModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize all parameters of *module* and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ---------------------------------------------------------------------------
# ControlNet model
# ---------------------------------------------------------------------------

class ControlNetModel(nn.Module):
    """ControlNet that mirrors the encoder half of a ``UNet2DModel``.

    Following the ControlNet paper:
    * A small conditioning encoder converts a spatial conditioning image
      (e.g. a bounding-box mask at image resolution) into features at the
      latent resolution and adds them to the ``conv_in`` output.
    * The encoder (down-blocks + mid-block) is deep-copied from the
      pre-trained UNet so that training starts from a good initialisation.
    * Zero convolutions on every skip-connection output guarantee that the
      ControlNet contributes *nothing* at the start of training — the
      augmented UNet therefore behaves identically to the base UNet.
    """

    def __init__(
        self,
        unet: UNet2DModel,
        conditioning_channels: int = 1,
        conditioning_downscale_factor: int = 4,
    ):
        super().__init__()

        block_out_ch0 = unet.config.block_out_channels[0]

        # --- conditioning encoder -------------------------------------------
        # Downsamples from image resolution to latent resolution.
        # Default factor = 4  (256x256 -> 64x64 for VAE with 2 downsamples).
        cond_layers: list[nn.Module] = [
            nn.Conv2d(conditioning_channels, 16, 3, stride=1, padding=1),
            nn.SiLU(),
        ]
        ch_in = 16
        factor = conditioning_downscale_factor
        while factor > 1:
            ch_out = min(ch_in * 2, 96)
            cond_layers += [
                nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                nn.SiLU(),
            ]
            ch_in = ch_out
            factor //= 2
        # Final projection to match conv_in output channels.
        cond_layers.append(
            nn.Conv2d(ch_in, block_out_ch0, 3, stride=1, padding=1)
        )
        self.controlnet_cond_embedding = nn.Sequential(*cond_layers)

        # --- encoder copied from UNet ---------------------------------------
        self.time_proj = copy.deepcopy(unet.time_proj)
        self.time_embedding = copy.deepcopy(unet.time_embedding)
        self.conv_in = copy.deepcopy(unet.conv_in)
        self.down_blocks = copy.deepcopy(unet.down_blocks)
        self.mid_block = copy.deepcopy(unet.mid_block)

        # --- zero convolutions ----------------------------------------------
        down_channels, mid_ch = self._compute_residual_channels(unet.config)

        self.controlnet_down_blocks = nn.ModuleList(
            [zero_module(nn.Conv2d(ch, ch, 1)) for ch in down_channels]
        )
        self.controlnet_mid_block = zero_module(nn.Conv2d(mid_ch, mid_ch, 1))

    # --------------------------------------------------------------------- #

    @staticmethod
    def _compute_residual_channels(
        config,
    ) -> Tuple[list, int]:
        """Return the channel count for every skip-connection tensor and
        the mid-block output channel count."""
        block_out_channels = list(config.block_out_channels)
        layers_per_block = getattr(config, "layers_per_block", 2)
        n_blocks = len(block_out_channels)

        # First entry comes from conv_in.
        channels = [block_out_channels[0]]
        for i, ch in enumerate(block_out_channels):
            for _ in range(layers_per_block):
                channels.append(ch)
            # All blocks except the last one have a down-sampler.
            if i < n_blocks - 1:
                channels.append(ch)

        mid_ch = block_out_channels[-1]
        return channels, mid_ch

    # --------------------------------------------------------------------- #

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        conditioning_image: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Parameters
        ----------
        sample : (B, C_latent, H_lat, W_lat)
            Noisy latent tensor.
        timestep : (B,)
            Scaled timesteps.
        conditioning_image : (B, cond_channels, H_img, W_img)
            Spatial conditioning (e.g. bbox mask at image resolution).

        Returns
        -------
        (down_block_residuals, mid_block_residual)
            Residuals to be injected into the frozen UNet.
        """
        # -- time embedding ---------------------------------------------------
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # -- pre-process + conditioning ---------------------------------------
        sample = self.conv_in(sample)
        conditioning = self.controlnet_cond_embedding(conditioning_image)
        sample = sample + conditioning

        # -- encoder ----------------------------------------------------------
        down_block_res_samples: Tuple[torch.Tensor, ...] = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb
            )
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb)

        # -- zero convolutions ------------------------------------------------
        controlnet_down = tuple(
            zc(s)
            for zc, s in zip(self.controlnet_down_blocks, down_block_res_samples)
        )
        controlnet_mid = self.controlnet_mid_block(sample)

        return controlnet_down, controlnet_mid


# ---------------------------------------------------------------------------
# Custom UNet forward with ControlNet residuals
# ---------------------------------------------------------------------------

def unet_forward_with_controlnet(
    unet: UNet2DModel,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    down_block_additional_residuals: Tuple[torch.Tensor, ...],
    mid_block_additional_residual: torch.Tensor,
    conditioning_scale: float = 1.0,
) -> torch.Tensor:
    """Run the frozen ``UNet2DModel`` forward pass while injecting ControlNet
    residuals into the skip connections and mid-block output.

    This replicates the logic of ``UNet2DModel.forward`` but adds
    ``conditioning_scale * residual`` to each skip tensor before the decoder
    consumes it — faithfully implementing the ControlNet paper's architecture.
    """
    # -- time embedding -------------------------------------------------------
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(
            [timesteps], dtype=torch.long, device=sample.device
        )
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = unet.time_proj(timesteps)
    t_emb = t_emb.to(dtype=sample.dtype)
    emb = unet.time_embedding(t_emb)

    # -- encoder --------------------------------------------------------------
    sample = unet.conv_in(sample)

    down_block_res_samples: Tuple[torch.Tensor, ...] = (sample,)
    for downsample_block in unet.down_blocks:
        sample, res_samples = downsample_block(
            hidden_states=sample, temb=emb
        )
        down_block_res_samples += res_samples

    sample = unet.mid_block(sample, emb)

    # -- inject ControlNet residuals ------------------------------------------
    sample = sample + mid_block_additional_residual * conditioning_scale

    down_block_res_samples = tuple(
        s + r * conditioning_scale
        for s, r in zip(down_block_res_samples, down_block_additional_residuals)
    )

    # -- decoder --------------------------------------------------------------
    for upsample_block in unet.up_blocks:
        n_res = len(upsample_block.resnets)
        res_samples = down_block_res_samples[-n_res:]
        down_block_res_samples = down_block_res_samples[:-n_res]
        sample = upsample_block(sample, res_samples, emb)

    # -- post-process ---------------------------------------------------------
    sample = unet.conv_norm_out(sample)
    sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)

    return sample


# ---------------------------------------------------------------------------
# Config save / load helpers
# ---------------------------------------------------------------------------

def save_controlnet_config(config: Dict[str, Any], path: str) -> None:
    """Write a ControlNet config dict to *path* as formatted JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def load_controlnet_config(path: str) -> Dict[str, Any]:
    """Load a ControlNet config dict from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
