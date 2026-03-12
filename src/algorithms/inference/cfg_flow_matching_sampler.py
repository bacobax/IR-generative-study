"""Flow-matching sampler with classifier-free guidance (CFG) support.

Extends :class:`FlowMatchingSampler` with :meth:`sample_euler_cfg`
which performs the two-pass velocity prediction:

    v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

CFG is skipped (single forward pass) when ``guidance_scale == 1.0``,
recovering pure conditional generation.  ``guidance_scale == 0.0``
recovers pure unconditional generation.

Existing sampling methods from the parent class (``sample_euler``,
``sample_euler_guided``, etc.) remain available and behave identically;
they use the conditioner's ``prepare_for_sampling`` which returns the
null embedding for unconditional generation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import UNet2DConditionModel
from torch.utils.tensorboard import SummaryWriter

from src.algorithms.inference.flow_matching_sampler import (
    FlowMatchingSampler,
    get_unet_sample_shape,
)
from src.analysis.cross_attention_maps import (
    AttentionExtractionConfig,
    AttentionHeatmapVisualizer,
    CrossAttentionExtractor,
)
from src.conditioning.text_conditioner import TextConditioner


class CFGFlowMatchingSampler(FlowMatchingSampler):
    """Flow-matching sampler with classifier-free guidance.

    This class adds :meth:`sample_euler_cfg` on top of the existing
    sampling methods.  The ``conditioner`` **must** be a
    :class:`TextConditioner` (or any conditioner that implements
    :meth:`prepare_cfg_pair`).

    Parameters
    ----------
    unet : UNet2DConditionModel
        Text-conditioned UNet.
    conditioner : TextConditioner
        Provides conditional / unconditional embedding pairs.
    device, t_scale, train_target, from_norm_to_display, sample_shape,
    encoder, decoder, guidance :
        Forwarded to :class:`FlowMatchingSampler`.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        *,
        conditioner: TextConditioner,
        device: Optional[Union[str, torch.device]] = None,
        t_scale: float = 1.0,
        train_target: str = "v",
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        guidance=None,
    ):
        super().__init__(
            unet,
            device=device,
            t_scale=t_scale,
            train_target=train_target,
            from_norm_to_display=from_norm_to_display,
            sample_shape=sample_shape,
            encoder=encoder,
            decoder=decoder,
            guidance=guidance,
            conditioner=conditioner,
        )

    # ------------------------------------------------------------------
    # VAE convenience constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_stable(
        cls,
        unet: UNet2DConditionModel,
        vae,
        *,
        conditioner: TextConditioner,
        **kwargs,
    ) -> "CFGFlowMatchingSampler":
        """Build a latent-space CFG sampler wired to a frozen VAE."""

        @torch.no_grad()
        def _encode(x: torch.Tensor) -> torch.Tensor:
            z_mu, z_sigma = vae.encode(x)
            return vae.sampling(z_mu, z_sigma)

        @torch.no_grad()
        def _decode(z: torch.Tensor) -> torch.Tensor:
            return vae.decode(z)

        return cls(
            unet,
            conditioner=conditioner,
            encoder=_encode,
            decoder=_decode,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Config-driven constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config,
        *,
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> "CFGFlowMatchingSampler":
        """Build a CFG sampler from a :class:`TextFMSampleConfig`.

        Loads the UNet2DConditionModel, VAE, and text conditioner from
        the pipeline directory.
        """
        from src.models.fm_text_unet import load_text_unet_config, build_text_fm_unet
        from src.models.vae import (
            load_vae_config,
            build_vae_from_config,
            load_vae_weights,
            freeze_vae,
        )
        import json
        import os

        device = config.resolved_device()
        pipeline_dir = config.pipeline_dir
        unet_dir = os.path.join(pipeline_dir, "UNET")
        vae_dir = os.path.join(pipeline_dir, "VAE")

        # UNet
        unet_cfg = load_text_unet_config(os.path.join(unet_dir, "config.json"))
        unet = build_text_fm_unet(unet_cfg, device=device)
        unet_w = os.path.join(unet_dir, "unet_fm_best.pt")
        if not os.path.isfile(unet_w):
            from src.algorithms.inference.flow_matching_sampler import _pick_latest
            unet_w = _pick_latest(unet_dir, "unet_fm_epoch_")
        if unet_w is None or not os.path.isfile(unet_w):
            raise FileNotFoundError(f"No UNET weights in {unet_dir}")
        unet.load_state_dict(torch.load(unet_w, map_location=device))
        unet.eval()

        # VAE
        vae_cfg = load_vae_config(os.path.join(vae_dir, "config.json"))
        vae = build_vae_from_config(vae_cfg, device=device)
        vae_w = os.path.join(vae_dir, "vae_best.pt")
        if not os.path.isfile(vae_w):
            from src.algorithms.inference.flow_matching_sampler import _pick_latest
            vae_w = _pick_latest(vae_dir, "vae_epoch_")
        if vae_w is not None and os.path.isfile(vae_w):
            load_vae_weights(vae, vae_w, map_location=device)
        if config.vae_weights is not None:
            load_vae_weights(vae, config.vae_weights, map_location=device)
        freeze_vae(vae)

        # Text conditioner
        cond_path = os.path.join(pipeline_dir, "conditioner.json")
        if os.path.isfile(cond_path):
            with open(cond_path, "r") as f:
                cond_info = json.load(f)
            encoder_name = cond_info["encoder_name"]
            max_length = cond_info["max_length"]
        else:
            encoder_name = config.text_encoder
            max_length = config.max_text_length

        conditioner = TextConditioner(
            encoder_name=encoder_name,
            max_length=max_length,
            cond_drop_prob=0.0,  # no dropout at inference
            device=device,
        )

        return cls.from_stable(
            unet,
            vae,
            conditioner=conditioner,
            device=device,
            t_scale=config.t_scale,
            train_target=config.train_target,
            from_norm_to_display=from_norm_to_display,
            sample_shape=config.sample_shape,
        )

    # ------------------------------------------------------------------
    # CFG Euler sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_euler_cfg(
        self,
        prompts: List[str],
        steps: int = 50,
        guidance_scale: float = 7.5,
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """Sample with classifier-free guidance.

        At each Euler step the velocity is computed as::

            v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

        When ``guidance_scale == 1.0`` the unconditional pass is
        skipped (pure conditional generation).

        Parameters
        ----------
        prompts : list[str]
            Text prompts (one per sample).  Batch size is ``len(prompts)``.
        steps : int
            Number of Euler integration steps.
        guidance_scale : float
            CFG scale.  ``1.0`` → no guidance.  ``0.0`` → unconditional.
            Typical values: 3.0–15.0.
        sample_shape : (C, H, W), optional
            Override spatial size.

        Returns
        -------
        torch.Tensor of shape ``(B, C, H, W)`` — generated latents.
        Call ``self.decode(z)`` to get pixel-space images.
        """
        self.unet.eval()
        batch_size = len(prompts)
        shape = self._shape(sample_shape)
        z = torch.randn(batch_size, *shape, device=self.device)
        dt = 1.0 / steps

        cond_kw, uncond_kw = self.conditioner.prepare_cfg_pair(
            prompts, self.device,
        )

        for i in range(steps):
            t_val = i / steps
            t = torch.full((batch_size,), t_val, device=self.device)
            t_scaled = t * self.t_scale

            out_cond = self.unet(z, t_scaled, **cond_kw).sample

            if guidance_scale == 1.0:
                out = out_cond
            elif guidance_scale == 0.0:
                out = self.unet(z, t_scaled, **uncond_kw).sample
            else:
                out_uncond = self.unet(z, t_scaled, **uncond_kw).sample
                out = out_uncond + guidance_scale * (out_cond - out_uncond)

            # Convert to velocity if train target is x0
            if self.train_target == "x0":
                t_exp = t[:, None, None, None]
                v = (out - z) / (1 - t_exp).clamp(min=1e-5)
            else:
                v = out

            z = z + v * dt

        return z

    # ------------------------------------------------------------------
    # CFG Euler sampling with cross-attention extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_euler_cfg_with_attention(
        self,
        prompts: List[str],
        steps: int = 50,
        guidance_scale: float = 7.5,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        attn_config: Optional[AttentionExtractionConfig] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
        """Sample with CFG and capture cross-attention maps.

        Identical to :meth:`sample_euler_cfg` but additionally installs
        :class:`CrossAttentionExtractor` hooks to capture attention
        maps at uniformly spaced timesteps.

        Parameters
        ----------
        prompts : list[str]
            Text prompts.
        steps : int
            Number of Euler steps.
        guidance_scale : float
            CFG scale.
        sample_shape : (C, H, W), optional
        attn_config : AttentionExtractionConfig, optional
            Controls which layers to capture, how many steps, target
            tokens, head reduction, etc.

        Returns
        -------
        (z, captured_maps) where:
          - ``z``: (B, C, H, W) generated latents
          - ``captured_maps``: dict[step, dict[layer_name, Tensor]]
            from :meth:`CrossAttentionExtractor.collect`.
        """
        self.unet.eval()
        batch_size = len(prompts)
        shape = self._shape(sample_shape)
        z = torch.randn(batch_size, *shape, device=self.device)
        dt = 1.0 / steps

        cond_kw, uncond_kw = self.conditioner.prepare_cfg_pair(
            prompts, self.device,
        )

        # Set up attention extraction
        extractor = CrossAttentionExtractor(
            self.unet, self.conditioner, config=attn_config,
        )
        extractor.bind(total_steps=steps)

        try:
            for i in range(steps):
                extractor.notify_step(i)

                t_val = i / steps
                t = torch.full((batch_size,), t_val, device=self.device)
                t_scaled = t * self.t_scale

                # Conditional pass (attention captured on vis steps)
                out_cond = self.unet(z, t_scaled, **cond_kw).sample

                if guidance_scale == 1.0:
                    out = out_cond
                elif guidance_scale == 0.0:
                    out = self.unet(z, t_scaled, **uncond_kw).sample
                else:
                    out_uncond = self.unet(z, t_scaled, **uncond_kw).sample
                    out = out_uncond + guidance_scale * (out_cond - out_uncond)

                if self.train_target == "x0":
                    t_exp = t[:, None, None, None]
                    v = (out - z) / (1 - t_exp).clamp(min=1e-5)
                else:
                    v = out

                z = z + v * dt

            captured_maps = extractor.collect()
        finally:
            extractor.unbind()

        return z, captured_maps

    # ------------------------------------------------------------------
    # TensorBoard: attention heatmaps
    # ------------------------------------------------------------------
    def log_attention_heatmaps_to_tensorboard(
        self,
        writer: SummaryWriter,
        epoch: int,
        prompts: List[str],
        steps: int = 50,
        guidance_scale: float = 7.5,
        attn_config: Optional[AttentionExtractionConfig] = None,
        tag_prefix: str = "cross_attn",
        per_layer: bool = False,
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Generate samples with CFG, extract attention maps, and log heatmaps.

        Combines :meth:`sample_euler_cfg_with_attention` with
        :class:`AttentionHeatmapVisualizer` to produce TensorBoard
        entries showing which spatial patches attend to target tokens
        at selected timesteps.

        Parameters
        ----------
        writer : SummaryWriter
        epoch : int
        prompts : list[str]
        steps : int
        guidance_scale : float
        attn_config : AttentionExtractionConfig, optional
        tag_prefix : str
        per_layer : bool
            If True, log per-layer heatmaps in addition to aggregated.
        sample_shape : (C, H, W), optional
        """
        if attn_config is None:
            attn_config = AttentionExtractionConfig()

        z, captured_maps = self.sample_euler_cfg_with_attention(
            prompts,
            steps=steps,
            guidance_scale=guidance_scale,
            sample_shape=sample_shape,
            attn_config=attn_config,
        )

        # Decode to pixel space
        x = self.decode(z)
        vis = self.from_norm_to_display(x).clamp(0, 1)

        # Also log the generated images themselves
        for i in range(vis.shape[0]):
            writer.add_image(
                f"{tag_prefix}/generated/{i}",
                vis[i],
                global_step=epoch,
            )

        # Render and log heatmaps
        visualizer = AttentionHeatmapVisualizer(
            config=attn_config,
            image_size=(vis.shape[2], vis.shape[3]),
        )

        extractor_for_tokens = CrossAttentionExtractor(
            self.unet, self.conditioner, config=attn_config,
        )

        visualizer.log_to_tensorboard(
            writer, epoch, captured_maps,
            generated_images=vis,
            prompts=prompts,
            extractor=extractor_for_tokens,
            tag_prefix=tag_prefix,
            per_layer=per_layer,
        )
    def log_samples_to_tensorboard_cfg(
        self,
        writer: SummaryWriter,
        epoch: int,
        prompts: List[str],
        steps: int = 50,
        guidance_scale: float = 7.5,
        tag: str = "text_fm/cfg_generated",
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Generate CFG samples and log to TensorBoard."""
        z = self.sample_euler_cfg(
            prompts,
            steps=steps,
            guidance_scale=guidance_scale,
            sample_shape=sample_shape,
        )
        x = self.decode(z)
        vis = self.from_norm_to_display(x)
        vis = vis.clamp(0, 1)
        for i in range(vis.shape[0]):
            writer.add_image(f"{tag}/{i}", vis[i], global_step=epoch)


# ── registry ──────────────────────────────────────────────────────────────
from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.sampler.register("cfg_fm")(CFGFlowMatchingSampler)
