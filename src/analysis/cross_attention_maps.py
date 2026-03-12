"""Cross-attention map extraction and visualization for text-conditioned FM.

This module provides two main components:

1. :class:`CrossAttentionExtractor` — hooks into a UNet2DConditionModel's
   cross-attention layers via custom :class:`AttnProcessor` wrappers.
   Captures attention maps at selected sampling timesteps.

2. :class:`AttentionHeatmapVisualizer` — takes captured attention maps,
   isolates attention to specific text tokens, and produces heatmaps
   suitable for TensorBoard logging.

Usage::

    extractor = CrossAttentionExtractor(
        unet, conditioner,
        target_tokens=["person", "people"],
        num_vis_steps=8,
        layer_filter="all",           # or "up", "down", "mid", list
        head_reduction="mean",        # or "none" for per-head maps
    )
    extractor.bind(total_steps=50)

    # ... run sample_euler_cfg loop, extractor captures maps automatically ...

    maps = extractor.collect()        # dict[step -> dict[layer -> Tensor]]
    extractor.unbind()                # restore original processors

    visualizer = AttentionHeatmapVisualizer(overlay=True)
    visualizer.log_to_tensorboard(writer, epoch, maps, generated_images)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention


# ═══════════════════════════════════════════════════════════════════════════
# Storing attention processor — wraps any existing processor
# ═══════════════════════════════════════════════════════════════════════════

class _StoringAttnProcessor:
    """Wraps an existing processor and optionally stores attention probs.

    When ``active`` is True, the attention probabilities are computed
    explicitly (even if the original processor uses
    ``F.scaled_dot_product_attention``) and stored in ``store`` keyed
    by ``(step_index, layer_name)``.

    When ``active`` is False, the original processor runs unmodified.
    """

    def __init__(self, original_processor, layer_name: str, store: dict):
        self._original = original_processor
        self._layer_name = layer_name
        self._store = store
        self.active = False
        self.step_index: int = -1

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if not self.active:
            return self._original(
                attn, hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb, *args, **kwargs,
            )

        # --- Explicit attention computation to capture probs ---
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        sequence_length = (
            hidden_states.shape[1]
            if encoder_hidden_states is None
            else encoder_hidden_states.shape[1]
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size,
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Store cross-attention probs only (not self-attention)
        if is_cross:
            # attention_probs shape: (B*heads, num_patches, seq_len)
            self._store[(self.step_index, self._layer_name)] = (
                attention_probs.detach().cpu()
            )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# ═══════════════════════════════════════════════════════════════════════════
# CrossAttentionExtractor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AttentionExtractionConfig:
    """Configuration knobs for attention extraction.

    Parameters
    ----------
    target_tokens : list[str]
        Token strings to isolate (e.g. ``["person", "people"]``).
    num_vis_steps : int
        Number of sampling timesteps at which to capture maps.  They
        are selected uniformly across the full trajectory.
    layer_filter : str or list[str]
        Which attention layers to capture:
        - ``"all"`` — every cross-attention layer
        - ``"up"`` / ``"down"`` / ``"mid"`` — only that block group
        - A list of exact module name substrings to match.
    head_reduction : str
        ``"mean"`` to average across attention heads,
        ``"none"`` to keep per-head maps.
    overlay : bool
        If True, overlay heatmap on generated image. If False, log as
        standalone grayscale heatmap.
    colormap : str
        Matplotlib colormap name for heatmap rendering.
    """

    target_tokens: List[str] = field(default_factory=lambda: ["person", "people"])
    num_vis_steps: int = 8
    layer_filter: Union[str, List[str]] = "all"
    head_reduction: str = "mean"
    overlay: bool = True
    colormap: str = "jet"


class CrossAttentionExtractor:
    """Hooks into a UNet2DConditionModel to capture cross-attention maps.

    Parameters
    ----------
    unet : UNet2DConditionModel
        The text-conditioned UNet.
    conditioner : TextConditioner
        Text conditioner providing the tokenizer for token lookup.
    config : AttentionExtractionConfig, optional
        All configuration knobs.  Defaults are sensible.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        conditioner,
        config: Optional[AttentionExtractionConfig] = None,
    ):
        self.unet = unet
        self.conditioner = conditioner
        self.config = config or AttentionExtractionConfig()
        self._store: Dict[Tuple[int, str], torch.Tensor] = {}
        self._processors: Dict[str, _StoringAttnProcessor] = {}
        self._original_processors: Dict[str, Any] = {}
        self._bound = False
        self._vis_steps: Set[int] = set()
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Layer selection
    # ------------------------------------------------------------------
    def _should_capture(self, module_name: str) -> bool:
        """Check if the module matches the layer_filter."""
        filt = self.config.layer_filter
        if filt == "all":
            return True
        if filt == "down":
            return "down_blocks" in module_name
        if filt == "up":
            return "up_blocks" in module_name
        if filt == "mid":
            return "mid_block" in module_name
        if isinstance(filt, list):
            return any(f in module_name for f in filt)
        return False

    # ------------------------------------------------------------------
    # Bind / unbind — swap processors
    # ------------------------------------------------------------------
    def bind(self, total_steps: int) -> None:
        """Install storing processors on all matching cross-attention layers.

        Call this BEFORE starting the sampling loop.

        Parameters
        ----------
        total_steps : int
            Total number of Euler steps so we can pick uniform vis steps.
        """
        if self._bound:
            self.unbind()

        self._total_steps = total_steps
        self._store.clear()
        self._processors.clear()
        self._original_processors.clear()

        # Choose uniformly spaced steps
        n = min(self.config.num_vis_steps, total_steps)
        if n >= total_steps:
            self._vis_steps = set(range(total_steps))
        else:
            self._vis_steps = {
                round(i * (total_steps - 1) / (n - 1)) for i in range(n)
            }

        # Find all attn2 modules and wrap their processors
        for name, module in self.unet.named_modules():
            if not name.endswith(".attn2"):
                continue
            if not isinstance(module, Attention):
                continue
            if not self._should_capture(name):
                continue

            original = module.get_processor()
            wrapper = _StoringAttnProcessor(original, name, self._store)
            module.set_processor(wrapper)
            self._processors[name] = wrapper
            self._original_processors[name] = original

        self._bound = True

    def unbind(self) -> None:
        """Restore original processors and release references."""
        for name, module in self.unet.named_modules():
            if name in self._original_processors:
                if isinstance(module, Attention):
                    module.set_processor(self._original_processors[name])
        self._processors.clear()
        self._original_processors.clear()
        self._bound = False

    # ------------------------------------------------------------------
    # Step notification — called by the sampling loop
    # ------------------------------------------------------------------
    def notify_step(self, step_index: int) -> None:
        """Activate/deactivate processors for this step.

        The sampling loop should call this at the START of each Euler step.
        """
        active = step_index in self._vis_steps
        for proc in self._processors.values():
            proc.active = active
            proc.step_index = step_index

    # ------------------------------------------------------------------
    # Collect captured maps
    # ------------------------------------------------------------------
    def collect(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Return captured attention maps grouped by step.

        Returns
        -------
        dict[int, dict[str, Tensor]]
            ``{step_index: {layer_name: attn_probs}}``
            where ``attn_probs`` has shape ``(B*heads, num_patches, seq_len)``.
        """
        result: Dict[int, Dict[str, torch.Tensor]] = {}
        for (step, layer), tensor in self._store.items():
            result.setdefault(step, {})[layer] = tensor
        return result

    def clear(self) -> None:
        """Clear the stored maps (frees memory)."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Token index resolution
    # ------------------------------------------------------------------
    def resolve_token_indices(self, prompt: str) -> List[int]:
        """Find positions of target tokens in the tokenized prompt.

        Returns a list of token indices (0-based into the tokenized
        sequence including special tokens).
        """
        tokenizer = self.conditioner.tokenizer
        tokens = tokenizer.tokenize(prompt)
        # Add offset of 1 for the [CLS] / <|startoftext|> token
        indices = []
        for target in self.config.target_tokens:
            target_lower = target.lower()
            # CLIP tokenizer uses </w> suffix and lowercases
            for i, tok in enumerate(tokens):
                # Strip CLIP's </w> suffix for comparison
                clean = tok.replace("</w>", "").lower()
                if clean == target_lower:
                    indices.append(i + 1)  # +1 for start token
        return indices


# ═══════════════════════════════════════════════════════════════════════════
# AttentionHeatmapVisualizer
# ═══════════════════════════════════════════════════════════════════════════

class AttentionHeatmapVisualizer:
    """Renders cross-attention maps as heatmaps and logs to TensorBoard.

    Parameters
    ----------
    config : AttentionExtractionConfig
        Shared config with overlay, colormap, head_reduction settings.
    image_size : (H, W), optional
        Target image resolution for heatmap upsampling.  If None, uses
        the native attention resolution.
    """

    def __init__(
        self,
        config: Optional[AttentionExtractionConfig] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        self.config = config or AttentionExtractionConfig()
        self.image_size = image_size

    # ------------------------------------------------------------------
    # Core: extract token-specific heatmap from attention maps
    # ------------------------------------------------------------------
    def extract_token_heatmap(
        self,
        attn_probs: torch.Tensor,
        token_indices: List[int],
        batch_size: int,
        num_heads: int,
    ) -> torch.Tensor:
        """Extract and aggregate heatmap for specific token indices.

        Parameters
        ----------
        attn_probs : Tensor of shape (B*heads, num_patches, seq_len)
            Raw attention probabilities from the storing processor.
        token_indices : list[int]
            Token positions to aggregate.
        batch_size : int
            Original batch size.
        num_heads : int
            Number of attention heads.

        Returns
        -------
        Tensor of shape (B, 1, H_attn, W_attn) — token attention heatmap.
        If head_reduction == "none", shape is (B, heads, H_attn, W_attn).
        """
        # attn_probs: (B*heads, num_patches, seq_len)
        total = attn_probs.shape[0]
        num_patches = attn_probs.shape[1]
        h_attn = w_attn = int(math.isqrt(num_patches))

        # Select token columns and average across them
        if not token_indices:
            # No matching tokens — return uniform attention
            heatmap = torch.ones(total, num_patches, device=attn_probs.device)
            heatmap = heatmap / num_patches
        else:
            # Average across target token positions
            heatmap = attn_probs[:, :, token_indices].mean(dim=-1)
            # shape: (B*heads, num_patches)

        # Reshape to spatial grid
        heatmap = heatmap.view(total, h_attn, w_attn)
        # Reshape to (B, heads, H, W)
        heatmap = heatmap.view(batch_size, num_heads, h_attn, w_attn)

        if self.config.head_reduction == "mean":
            heatmap = heatmap.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        return heatmap

    # ------------------------------------------------------------------
    # Upscale heatmap to image resolution
    # ------------------------------------------------------------------
    def upscale_heatmap(
        self,
        heatmap: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Bilinear upscale heatmap to target image size.

        Parameters
        ----------
        heatmap : (B, C, H_attn, W_attn)
        target_h, target_w : int

        Returns
        -------
        (B, C, target_h, target_w) — normalized to [0, 1].
        """
        up = F.interpolate(
            heatmap.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        # Normalize per-sample to [0, 1]
        b, c = up.shape[:2]
        flat = up.view(b, c, -1)
        lo = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
        hi = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
        up = (up - lo) / (hi - lo + 1e-8)
        return up

    # ------------------------------------------------------------------
    # Apply colormap
    # ------------------------------------------------------------------
    def apply_colormap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Convert single-channel heatmap to RGB using matplotlib colormap.

        Parameters
        ----------
        heatmap : (B, 1, H, W) with values in [0, 1]

        Returns
        -------
        (B, 3, H, W) RGB tensor.
        """
        import matplotlib.cm as cm

        cmap = cm.get_cmap(self.config.colormap)
        # Process per-sample
        b, _, h, w = heatmap.shape
        flat = heatmap.view(b, h * w).numpy()
        rgb_list = []
        for i in range(b):
            rgba = cmap(flat[i])  # (h*w, 4)
            rgb = torch.from_numpy(rgba[:, :3]).float().view(h, w, 3)
            rgb = rgb.permute(2, 0, 1)  # (3, H, W)
            rgb_list.append(rgb)
        return torch.stack(rgb_list)  # (B, 3, H, W)

    # ------------------------------------------------------------------
    # Overlay heatmap on image
    # ------------------------------------------------------------------
    def overlay_on_image(
        self,
        image: torch.Tensor,
        heatmap_rgb: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Blend heatmap onto generated image.

        Parameters
        ----------
        image : (B, 3, H, W) in [0, 1]
        heatmap_rgb : (B, 3, H, W) in [0, 1]
        alpha : blend weight for heatmap

        Returns
        -------
        (B, 3, H, W) blended image.
        """
        return (1 - alpha) * image + alpha * heatmap_rgb

    # ------------------------------------------------------------------
    # Full pipeline: maps → heatmaps for one timestep + one layer
    # ------------------------------------------------------------------
    def render_heatmaps(
        self,
        attn_probs: torch.Tensor,
        token_indices: List[int],
        batch_size: int,
        num_heads: int,
        generated_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Complete pipeline: extract → upscale → colormap → optionally overlay.

        Returns
        -------
        (B, 3, H, W) tensor ready for TensorBoard.
        """
        heatmap = self.extract_token_heatmap(
            attn_probs, token_indices, batch_size, num_heads,
        )

        # Determine target size
        if generated_images is not None:
            target_h, target_w = generated_images.shape[2], generated_images.shape[3]
        elif self.image_size is not None:
            target_h, target_w = self.image_size
        else:
            target_h, target_w = heatmap.shape[2], heatmap.shape[3]

        if self.config.head_reduction == "mean":
            heatmap = self.upscale_heatmap(heatmap, target_h, target_w)
            heatmap_rgb = self.apply_colormap(heatmap)
            if self.config.overlay and generated_images is not None:
                return self.overlay_on_image(generated_images.cpu(), heatmap_rgb)
            return heatmap_rgb
        else:
            # Per-head: return shape (B*heads, 3, H, W)
            b, heads, h, w = heatmap.shape
            heatmap = heatmap.view(b * heads, 1, h, w)
            heatmap = self.upscale_heatmap(heatmap, target_h, target_w)
            heatmap_rgb = self.apply_colormap(heatmap)
            if self.config.overlay and generated_images is not None:
                # Repeat images for each head
                imgs_rep = generated_images.cpu().repeat_interleave(heads, dim=0)
                return self.overlay_on_image(imgs_rep, heatmap_rgb)
            return heatmap_rgb

    # ------------------------------------------------------------------
    # Aggregate across layers
    # ------------------------------------------------------------------
    def aggregate_layer_heatmaps(
        self,
        step_maps: Dict[str, torch.Tensor],
        token_indices: List[int],
        batch_size: int,
        num_heads: int,
    ) -> torch.Tensor:
        """Average heatmaps across all captured layers for a single step.

        Returns single-channel heatmap (B, 1, H_max, W_max).
        """
        heatmaps = []
        max_h = max_w = 0
        for layer_name, attn_probs in step_maps.items():
            hm = self.extract_token_heatmap(
                attn_probs, token_indices, batch_size, num_heads,
            )
            # hm: (B, 1, h, w) if mean reduction
            if self.config.head_reduction == "mean":
                max_h = max(max_h, hm.shape[2])
                max_w = max(max_w, hm.shape[3])
                heatmaps.append(hm)

        if not heatmaps:
            return torch.zeros(batch_size, 1, 1, 1)

        # Upscale all to max resolution then average
        upscaled = []
        for hm in heatmaps:
            if hm.shape[2] != max_h or hm.shape[3] != max_w:
                hm = F.interpolate(
                    hm.float(), size=(max_h, max_w),
                    mode="bilinear", align_corners=False,
                )
            upscaled.append(hm)

        return torch.stack(upscaled).mean(dim=0)

    # ------------------------------------------------------------------
    # TensorBoard logging — full pipeline
    # ------------------------------------------------------------------
    def log_to_tensorboard(
        self,
        writer,
        epoch: int,
        captured_maps: Dict[int, Dict[str, torch.Tensor]],
        generated_images: Optional[torch.Tensor],
        prompts: List[str],
        extractor: CrossAttentionExtractor,
        tag_prefix: str = "cross_attn",
        per_layer: bool = False,
    ) -> None:
        """Log cross-attention heatmaps to TensorBoard.

        Parameters
        ----------
        writer : SummaryWriter
        epoch : int
        captured_maps : dict from :meth:`CrossAttentionExtractor.collect`
        generated_images : (B, 3, H, W) in [0, 1], or None
        prompts : list[str] — the prompts used for sampling
        extractor : CrossAttentionExtractor — for token index resolution
        tag_prefix : str
        per_layer : bool
            If True, also log separate heatmaps for each layer.
        """
        if not captured_maps or not prompts:
            return

        # Resolve token indices from first prompt (assume consistent)
        token_indices = extractor.resolve_token_indices(prompts[0])
        batch_size = len(prompts)

        # Determine num_heads from first captured tensor
        first_step = next(iter(captured_maps))
        first_layer = next(iter(captured_maps[first_step]))
        first_tensor = captured_maps[first_step][first_layer]
        total_b_heads = first_tensor.shape[0]
        num_heads = total_b_heads // batch_size

        sorted_steps = sorted(captured_maps.keys())

        for step in sorted_steps:
            step_maps = captured_maps[step]

            # Aggregate across layers
            agg_heatmap = self.aggregate_layer_heatmaps(
                step_maps, token_indices, batch_size, num_heads,
            )

            # Determine target size
            if generated_images is not None:
                th, tw = generated_images.shape[2], generated_images.shape[3]
            elif self.image_size is not None:
                th, tw = self.image_size
            else:
                th, tw = agg_heatmap.shape[2], agg_heatmap.shape[3]

            agg_up = self.upscale_heatmap(agg_heatmap, th, tw)
            agg_rgb = self.apply_colormap(agg_up)

            if self.config.overlay and generated_images is not None:
                vis = self.overlay_on_image(generated_images.cpu(), agg_rgb)
            else:
                vis = agg_rgb

            vis = vis.clamp(0, 1)
            for i in range(min(vis.shape[0], batch_size)):
                writer.add_image(
                    f"{tag_prefix}/step_{step:03d}/sample_{i}",
                    vis[i],
                    global_step=epoch,
                )

            # Per-layer logging
            if per_layer:
                for layer_name, attn_probs in step_maps.items():
                    short_name = layer_name.replace(".", "_")
                    rendered = self.render_heatmaps(
                        attn_probs, token_indices,
                        batch_size, num_heads, generated_images,
                    )
                    rendered = rendered.clamp(0, 1)
                    for i in range(min(rendered.shape[0], batch_size)):
                        writer.add_image(
                            f"{tag_prefix}/{short_name}/step_{step:03d}/sample_{i}",
                            rendered[i],
                            global_step=epoch,
                        )
