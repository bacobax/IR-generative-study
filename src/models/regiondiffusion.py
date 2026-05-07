"""RegionDiff-style layout-conditioning modules for Stable Diffusion."""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from src.core.diffusers_compat import disable_diffusers_optional_scipy

disable_diffusers_optional_scipy()

from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention


def compute_same_class_positions(
    labels: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    max_slots: int,
) -> torch.Tensor:
    """Return per-object occurrence indices for repeated categories."""
    batch_size, max_objects = labels.shape
    positions = torch.zeros_like(labels, dtype=torch.long)

    for batch_idx in range(batch_size):
        counts: Dict[int, int] = {}
        for object_idx in range(max_objects):
            if not bool(object_mask[batch_idx, object_idx]):
                continue
            label = int(labels[batch_idx, object_idx].item())
            occurrence = counts.get(label, 0)
            positions[batch_idx, object_idx] = min(occurrence, max_slots - 1)
            counts[label] = occurrence + 1
    return positions


def build_region_token_mask(
    *,
    boxes_xyxy_norm: torch.Tensor,
    object_mask: torch.Tensor,
    resolution: int,
    use_background_token: bool,
) -> torch.Tensor:
    """Build token-to-layout accessibility masks for one latent resolution."""
    batch_size, max_objects, _ = boxes_xyxy_norm.shape
    device = boxes_xyxy_norm.device
    region_masks = torch.zeros(
        batch_size,
        resolution,
        resolution,
        max_objects,
        dtype=torch.bool,
        device=device,
    )

    for batch_idx in range(batch_size):
        for object_idx in range(max_objects):
            if not bool(object_mask[batch_idx, object_idx]):
                continue
            x1, y1, x2, y2 = boxes_xyxy_norm[batch_idx, object_idx]
            ix1 = max(0, min(resolution - 1, int(torch.floor(x1 * resolution).item())))
            iy1 = max(0, min(resolution - 1, int(torch.floor(y1 * resolution).item())))
            ix2 = max(ix1 + 1, min(resolution, int(torch.ceil(x2 * resolution).item())))
            iy2 = max(iy1 + 1, min(resolution, int(torch.ceil(y2 * resolution).item())))
            region_masks[batch_idx, iy1:iy2, ix1:ix2, object_idx] = True

    token_mask = region_masks.view(batch_size, resolution * resolution, max_objects)
    if not use_background_token:
        return token_mask

    background_mask = ~token_mask.any(dim=-1, keepdim=True)
    return torch.cat([token_mask, background_mask], dim=-1)


def build_area_weight_map(
    *,
    boxes_xyxy_norm: torch.Tensor,
    object_mask: torch.Tensor,
    latent_height: int,
    latent_width: int,
    alpha: float,
    background_weight: float,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    """Construct RegionDiff-style area-based spatial loss weights."""
    batch_size, max_objects, _ = boxes_xyxy_norm.shape
    device = boxes_xyxy_norm.device
    dtype = boxes_xyxy_norm.dtype
    weights = torch.full(
        (batch_size, 1, latent_height, latent_width),
        float(background_weight),
        device=device,
        dtype=dtype,
    )

    sizes = (boxes_xyxy_norm[..., 2:] - boxes_xyxy_norm[..., :2]).clamp(min=0.0)
    areas = sizes[..., 0] * sizes[..., 1] * float(latent_height * latent_width)

    for batch_idx in range(batch_size):
        valid_areas = areas[batch_idx][object_mask[batch_idx]]
        if valid_areas.numel() == 0:
            continue
        mean_area = valid_areas.mean().clamp(min=1e-6)
        object_weights = torch.clamp(
            (mean_area / valid_areas.clamp(min=1e-6)).pow(float(alpha)),
            min=float(min_weight),
            max=float(max_weight),
        )

        valid_index = 0
        for object_idx in range(max_objects):
            if not bool(object_mask[batch_idx, object_idx]):
                continue
            x1, y1, x2, y2 = boxes_xyxy_norm[batch_idx, object_idx]
            ix1 = max(0, min(latent_width - 1, int(torch.floor(x1 * latent_width).item())))
            iy1 = max(0, min(latent_height - 1, int(torch.floor(y1 * latent_height).item())))
            ix2 = max(ix1 + 1, min(latent_width, int(torch.ceil(x2 * latent_width).item())))
            iy2 = max(iy1 + 1, min(latent_height, int(torch.ceil(y2 * latent_height).item())))
            weights[batch_idx, :, iy1:iy2, ix1:ix2] = torch.maximum(
                weights[batch_idx, :, iy1:iy2, ix1:ix2],
                object_weights[valid_index].view(1, 1, 1),
            )
            valid_index += 1

    mean_weight = weights.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    return weights / mean_weight


class LayoutTokenizer(nn.Module):
    """Convert ``(bbox, class)`` pairs into RegionDiff-style layout tokens."""

    def __init__(
        self,
        *,
        class_text_features: torch.Tensor,
        layout_token_dim: int,
        bbox_fourier_dim: int,
        same_class_position_slots: int,
        use_background_token: bool,
    ) -> None:
        super().__init__()
        text_features = torch.as_tensor(class_text_features, dtype=torch.float32)
        if text_features.ndim != 2:
            raise ValueError(
                "class_text_features must have shape (num_classes, feature_dim), "
                f"got {tuple(text_features.shape)}"
            )

        self.layout_token_dim = int(layout_token_dim)
        self.bbox_fourier_dim = int(bbox_fourier_dim)
        self.same_class_position_slots = int(same_class_position_slots)
        self.use_background_token = bool(use_background_token)
        self.num_classes = int(text_features.shape[0])
        self.text_feature_dim = int(text_features.shape[1])

        self.register_buffer("class_text_features", text_features, persistent=True)

        self.class_projection = nn.Linear(self.text_feature_dim, self.layout_token_dim)
        self.position_embedding = nn.Embedding(self.same_class_position_slots, self.layout_token_dim)
        self.bbox_projection = nn.Sequential(
            nn.Linear(8 * self.bbox_fourier_dim, self.layout_token_dim),
            nn.SiLU(),
            nn.Linear(self.layout_token_dim, self.layout_token_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.layout_token_dim * 3, self.layout_token_dim),
            nn.SiLU(),
            nn.Linear(self.layout_token_dim, self.layout_token_dim),
        )

        if self.use_background_token:
            self.background_token = nn.Parameter(torch.zeros(1, 1, self.layout_token_dim))
        else:
            self.register_parameter("background_token", None)

    def _bbox_fourier_features(self, boxes_xyxy_norm: torch.Tensor) -> torch.Tensor:
        frequencies = torch.pow(
            2.0,
            torch.arange(
                self.bbox_fourier_dim,
                device=boxes_xyxy_norm.device,
                dtype=boxes_xyxy_norm.dtype,
            ),
        ) * math.pi
        projected = boxes_xyxy_norm.unsqueeze(-1) * frequencies.view(1, 1, 1, -1)
        return torch.cat([projected.sin(), projected.cos()], dim=-1).flatten(start_dim=2)

    def forward(
        self,
        *,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_labels = labels.clamp(min=0, max=max(self.num_classes - 1, 0))
        class_features = self.class_text_features.index_select(
            0,
            safe_labels.view(-1),
        ).view(*safe_labels.shape, self.text_feature_dim)
        class_tokens = self.class_projection(class_features)

        bbox_tokens = self.bbox_projection(self._bbox_fourier_features(boxes_xyxy_norm))
        same_class_positions = compute_same_class_positions(
            safe_labels,
            object_mask,
            max_slots=self.same_class_position_slots,
        )
        position_tokens = self.position_embedding(same_class_positions)

        fused = self.fusion(torch.cat([class_tokens, bbox_tokens, position_tokens], dim=-1))
        fused = fused * object_mask.unsqueeze(-1).to(fused.dtype)
        if not self.use_background_token:
            return fused

        background = self.background_token.expand(fused.shape[0], -1, -1)
        return torch.cat([fused, background], dim=1)


class RegionSelfAttentionAdapter(nn.Module):
    """Masked attention from visual tokens into layout tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        layout_token_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layout_token_dim = int(layout_token_dim)
        self.num_heads = int(max(num_heads, 1))
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}"
            )
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.to_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.to_k = nn.Linear(self.layout_token_dim, self.hidden_dim, bias=False)
        self.to_v = nn.Linear(self.layout_token_dim, self.hidden_dim, bias=False)
        self.to_out = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self._attention_recorder = None
        nn.init.zeros_(self.to_out.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        layout_tokens: torch.Tensor,
        region_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if layout_tokens.shape[1] == 0:
            return torch.zeros_like(hidden_states)

        residual_dtype = hidden_states.dtype
        hidden_states = self.norm(hidden_states)
        query = self.to_q(hidden_states)
        key = self.to_k(layout_tokens)
        value = self.to_v(layout_tokens)

        batch_size, seq_len, _ = query.shape
        key_len = key.shape[1]

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        access_mask = region_token_mask.unsqueeze(1)
        scores = scores.masked_fill(~access_mask, -1e4)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * access_mask.to(dtype=attention.dtype)
        recorder = getattr(self, "_attention_recorder", None)
        if recorder is not None:
            recorder.record(
                self,
                attention,
                region_token_mask=region_token_mask,
                layout_tokens=layout_tokens,
                hidden_states=hidden_states,
            )

        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        return self.to_out(attended).to(dtype=residual_dtype)


class RegionSpatialAdapter(nn.Module):
    """Inject RegionDiff layout tokens as a full-resolution latent residual."""

    def __init__(
        self,
        *,
        sample_channels: int,
        layout_token_dim: int,
    ) -> None:
        super().__init__()
        self.sample_channels = int(sample_channels)
        self.layout_token_dim = int(layout_token_dim)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.layout_token_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(self.layout_token_dim, self.layout_token_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(self.layout_token_dim, self.sample_channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.proj[-1].weight)

    @staticmethod
    def _layout_map_from_context(
        *,
        regiondiff_context: Dict[str, Any],
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if height != width:
            raise ValueError(
                "RegionSpatialAdapter currently expects square latent maps, "
                f"got height={height}, width={width}."
            )

        mask_cache = regiondiff_context.setdefault("region_token_masks", {})
        region_token_mask = mask_cache.get(height)
        if region_token_mask is None:
            region_token_mask = build_region_token_mask(
                boxes_xyxy_norm=regiondiff_context["boxes_xyxy_norm"],
                object_mask=regiondiff_context["object_mask"],
                resolution=height,
                use_background_token=bool(regiondiff_context["use_background_token"]),
            )
            mask_cache[height] = region_token_mask

        layout_tokens = regiondiff_context["layout_tokens"].to(dtype=dtype)
        weights = region_token_mask.to(dtype=layout_tokens.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        spatial_tokens = torch.bmm(weights, layout_tokens)
        batch_size, _, channels = spatial_tokens.shape
        return spatial_tokens.view(batch_size, height, width, channels).permute(0, 3, 1, 2)

    def forward(
        self,
        sample: torch.Tensor,
        *,
        regiondiff_context: Dict[str, Any],
    ) -> torch.Tensor:
        layout_map = self._layout_map_from_context(
            regiondiff_context=regiondiff_context,
            height=int(sample.shape[-2]),
            width=int(sample.shape[-1]),
            dtype=sample.dtype,
        )
        return self.proj(self.norm(layout_map)).to(dtype=sample.dtype)


class RegionDiffTransformerBlock(nn.Module):
    """Wrap a diffusers transformer block and insert Region Self-Attention."""

    def __init__(
        self,
        base_block: BasicTransformerBlock,
        *,
        layout_token_dim: int,
        active_region_resolutions: Iterable[int],
    ) -> None:
        super().__init__()
        self.base_block = base_block
        self.active_region_resolutions = frozenset(int(value) for value in active_region_resolutions)
        hidden_dim = int(base_block.attn1.to_q.out_features)
        num_heads = int(getattr(base_block.attn1, "heads", 1))
        self.region_adapter = RegionSelfAttentionAdapter(
            hidden_dim=hidden_dim,
            layout_token_dim=layout_token_dim,
            num_heads=num_heads,
        )

    @staticmethod
    def _resolution_from_hidden_states(hidden_states: torch.Tensor) -> Optional[int]:
        if hidden_states.ndim != 3:
            return None
        seq_len = int(hidden_states.shape[1])
        resolution = int(round(seq_len**0.5))
        if resolution * resolution != seq_len:
            return None
        return resolution

    def _maybe_apply_region_attention(
        self,
        hidden_states: torch.Tensor,
        *,
        regiondiff_context: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        if regiondiff_context is None:
            return hidden_states

        resolution = self._resolution_from_hidden_states(hidden_states)
        if resolution is None or resolution not in self.active_region_resolutions:
            return hidden_states

        mask_cache = regiondiff_context.setdefault("region_token_masks", {})
        region_token_mask = mask_cache.get(resolution)
        if region_token_mask is None:
            region_token_mask = build_region_token_mask(
                boxes_xyxy_norm=regiondiff_context["boxes_xyxy_norm"],
                object_mask=regiondiff_context["object_mask"],
                resolution=resolution,
                use_background_token=bool(regiondiff_context["use_background_token"]),
            )
            mask_cache[resolution] = region_token_mask

        delta = self.region_adapter(
            hidden_states,
            layout_tokens=regiondiff_context["layout_tokens"].to(dtype=hidden_states.dtype),
            region_token_mask=region_token_mask,
        )
        return hidden_states + delta

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        regiondiff_context = cross_attention_kwargs.pop("regiondiff_context", None)

        if cross_attention_kwargs.get("scale", None) is not None:
            from diffusers.models.attention import logger as attention_logger

            attention_logger.warning(
                "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
            )

        batch_size = hidden_states.shape[0]

        if self.base_block.norm_type == "ada_norm":
            norm_hidden_states = self.base_block.norm1(hidden_states, timestep)
        elif self.base_block.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.base_block.norm1(
                hidden_states,
                timestep,
                class_labels,
                hidden_dtype=hidden_states.dtype,
            )
        elif self.base_block.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.base_block.norm1(hidden_states)
        elif self.base_block.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.base_block.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.base_block.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.base_block.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.base_block.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.base_block.pos_embed is not None:
            norm_hidden_states = self.base_block.pos_embed(norm_hidden_states)

        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.base_block.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.base_block.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.base_block.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.base_block.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if gligen_kwargs is not None:
            hidden_states = self.base_block.fuser(hidden_states, gligen_kwargs["objs"])

        hidden_states = self._maybe_apply_region_attention(
            hidden_states,
            regiondiff_context=regiondiff_context,
        )

        if self.base_block.attn2 is not None:
            if self.base_block.norm_type == "ada_norm":
                norm_hidden_states = self.base_block.norm2(hidden_states, timestep)
            elif self.base_block.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.base_block.norm2(hidden_states)
            elif self.base_block.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.base_block.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.base_block.norm2(
                    hidden_states,
                    added_cond_kwargs["pooled_text_emb"],
                )
            else:
                raise ValueError("Incorrect norm")

            if self.base_block.pos_embed is not None and self.base_block.norm_type != "ada_norm_single":
                norm_hidden_states = self.base_block.pos_embed(norm_hidden_states)

            attn_output = self.base_block.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        if self.base_block.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.base_block.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.base_block.norm_type != "ada_norm_single":
            norm_hidden_states = self.base_block.norm3(hidden_states)

        if self.base_block.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.base_block.norm_type == "ada_norm_single":
            norm_hidden_states = self.base_block.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self.base_block._chunk_size is not None:
            from diffusers.models.attention import _chunked_feed_forward

            ff_output = _chunked_feed_forward(
                self.base_block.ff,
                norm_hidden_states,
                self.base_block._chunk_dim,
                self.base_block._chunk_size,
            )
        else:
            ff_output = self.base_block.ff(norm_hidden_states)

        if self.base_block.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.base_block.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class RegionDiffAttentionBlock(nn.Module):
    """Wrap a diffusers spatial Attention module and add RegionDiff attention."""

    def __init__(
        self,
        base_attention: Attention,
        *,
        layout_token_dim: int,
        active_region_resolutions: Iterable[int],
    ) -> None:
        super().__init__()
        self.base_attention = base_attention
        self.active_region_resolutions = frozenset(int(value) for value in active_region_resolutions)
        hidden_dim = int(getattr(base_attention, "query_dim", 0) or getattr(base_attention, "out_dim", 0))
        if hidden_dim <= 0:
            raise ValueError("Could not infer hidden_dim from diffusers Attention module.")
        num_heads = int(getattr(base_attention, "heads", 1))
        self.region_adapter = RegionSelfAttentionAdapter(
            hidden_dim=hidden_dim,
            layout_token_dim=layout_token_dim,
            num_heads=num_heads,
        )
        self.regiondiff_context: Optional[Dict[str, Any]] = None

    @staticmethod
    def _flatten_spatial(hidden_states: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if hidden_states.ndim != 4:
            raise ValueError(
                "RegionDiffAttentionBlock expects 4D spatial hidden states, "
                f"got shape={tuple(hidden_states.shape)}"
            )
        batch_size, channels, height, width = hidden_states.shape
        tokens = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        return tokens, height, width

    @staticmethod
    def _unflatten_spatial(tokens: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        batch_size, _, channels = tokens.shape
        return tokens.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

    def _maybe_apply_region_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        regiondiff_context = self.regiondiff_context
        if regiondiff_context is None or hidden_states.ndim != 4:
            return hidden_states

        tokens, height, width = self._flatten_spatial(hidden_states)
        if height != width or height not in self.active_region_resolutions:
            return hidden_states

        mask_cache = regiondiff_context.setdefault("region_token_masks", {})
        region_token_mask = mask_cache.get(height)
        if region_token_mask is None:
            region_token_mask = build_region_token_mask(
                boxes_xyxy_norm=regiondiff_context["boxes_xyxy_norm"],
                object_mask=regiondiff_context["object_mask"],
                resolution=height,
                use_background_token=bool(regiondiff_context["use_background_token"]),
            )
            mask_cache[height] = region_token_mask

        delta = self.region_adapter(
            tokens,
            layout_tokens=regiondiff_context["layout_tokens"].to(dtype=tokens.dtype),
            region_token_mask=region_token_mask,
        )
        return hidden_states + self._unflatten_spatial(delta, height=height, width=width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        hidden_states = self.base_attention(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        return self._maybe_apply_region_attention(hidden_states)


def patch_unet_regiondiff_blocks(
    base_unet: nn.Module,
    *,
    layout_token_dim: int,
    active_region_resolutions: Iterable[int],
) -> int:
    """Recursively wrap all diffusers transformer blocks in-place."""
    wrapped = 0
    for child_name, child in list(base_unet.named_children()):
        if isinstance(child, RegionDiffTransformerBlock):
            continue
        if isinstance(child, BasicTransformerBlock):
            setattr(
                base_unet,
                child_name,
                RegionDiffTransformerBlock(
                    child,
                    layout_token_dim=layout_token_dim,
                    active_region_resolutions=active_region_resolutions,
                ),
            )
            wrapped += 1
            continue
        wrapped += patch_unet_regiondiff_blocks(
            child,
            layout_token_dim=layout_token_dim,
            active_region_resolutions=active_region_resolutions,
        )
    return wrapped


def patch_attention_regiondiff_blocks(
    base_model: nn.Module,
    *,
    layout_token_dim: int,
    active_region_resolutions: Iterable[int],
) -> int:
    """Recursively wrap diffusers spatial Attention modules in-place."""
    wrapped = 0
    for child_name, child in list(base_model.named_children()):
        if isinstance(child, RegionDiffAttentionBlock):
            continue
        if isinstance(child, RegionDiffTransformerBlock):
            continue
        if isinstance(child, BasicTransformerBlock):
            continue
        if isinstance(child, Attention):
            setattr(
                base_model,
                child_name,
                RegionDiffAttentionBlock(
                    child,
                    layout_token_dim=layout_token_dim,
                    active_region_resolutions=active_region_resolutions,
                ),
            )
            wrapped += 1
            continue
        wrapped += patch_attention_regiondiff_blocks(
            child,
            layout_token_dim=layout_token_dim,
            active_region_resolutions=active_region_resolutions,
        )
    return wrapped


def iter_regiondiff_transformer_blocks(module: nn.Module) -> Iterable[RegionDiffTransformerBlock]:
    """Yield every wrapped RegionDiff transformer block under *module*."""
    for child in module.modules():
        if isinstance(child, RegionDiffTransformerBlock):
            yield child


def iter_regiondiff_attention_blocks(module: nn.Module) -> Iterable[RegionDiffAttentionBlock]:
    """Yield every wrapped RegionDiff attention block under *module*."""
    for child in module.modules():
        if isinstance(child, RegionDiffAttentionBlock):
            yield child


def iter_regiondiff_adapter_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    """Yield trainable RegionDiff parameters only."""
    if isinstance(module, (RegionDiffusionUNetWrapper, RegionDiffusionModelWrapper)):
        yield from module.layout_tokenizer.parameters()
        layout_input_adapter = getattr(module, "layout_input_adapter", None)
        if layout_input_adapter is not None:
            yield from layout_input_adapter.parameters()
    for block in iter_regiondiff_transformer_blocks(module):
        yield from block.region_adapter.parameters()
    for block in iter_regiondiff_attention_blocks(module):
        yield from block.region_adapter.parameters()


def regiondiff_config_dict(wrapper: "RegionDiffusionUNetWrapper") -> Dict[str, Any]:
    """Serialize the RegionDiff wrapper hyperparameters."""
    return {
        "backbone_kind": str(getattr(wrapper, "backbone_kind", "sd_conditional_unet")),
        "attachment_kind": str(getattr(wrapper, "attachment_kind", "transformer")),
        "layout_token_dim": int(wrapper.layout_tokenizer.layout_token_dim),
        "bbox_fourier_dim": int(wrapper.layout_tokenizer.bbox_fourier_dim),
        "same_class_position_slots": int(wrapper.layout_tokenizer.same_class_position_slots),
        "use_background_token": bool(wrapper.layout_tokenizer.use_background_token),
        "use_spatial_adapter": bool(getattr(wrapper, "use_spatial_adapter", False)),
        "active_region_resolutions": list(wrapper.active_region_resolutions),
        "category_id_to_name": {
            str(key): value for key, value in wrapper.category_id_to_name.items()
        },
        "num_region_blocks": int(wrapper.num_region_blocks),
    }


def save_regiondiff_config(config: Dict[str, Any], path: str) -> None:
    """Persist RegionDiff wrapper config for later loading."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def load_regiondiff_config(path: str) -> Dict[str, Any]:
    """Load a persisted RegionDiff wrapper config."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class RegionDiffusionUNetWrapper(nn.Module):
    """Stable Diffusion U-Net wrapper with RegionDiff-style layout conditioning."""

    def __init__(
        self,
        *,
        base_unet: nn.Module,
        class_text_features: torch.Tensor,
        category_id_to_name: Dict[int, str],
        layout_token_dim: int,
        bbox_fourier_dim: int,
        same_class_position_slots: int,
        use_background_token: bool,
        active_region_resolutions: Iterable[int],
    ) -> None:
        super().__init__()
        self.base_unet = base_unet
        self.backbone_kind = "sd_conditional_unet"
        self.attachment_kind = "transformer"
        self.category_id_to_name = dict(category_id_to_name)
        self.active_region_resolutions = tuple(sorted(int(value) for value in active_region_resolutions))
        self.layout_tokenizer = LayoutTokenizer(
            class_text_features=class_text_features,
            layout_token_dim=layout_token_dim,
            bbox_fourier_dim=bbox_fourier_dim,
            same_class_position_slots=same_class_position_slots,
            use_background_token=use_background_token,
        )
        self.num_region_blocks = patch_unet_regiondiff_blocks(
            self.base_unet,
            layout_token_dim=layout_token_dim,
            active_region_resolutions=self.active_region_resolutions,
        )
        self.config = self.base_unet.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _broadcast_layout_inputs(
        self,
        sample_batch_size: int,
        *,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        layout_batch_size = int(boxes_xyxy_norm.shape[0])
        if layout_batch_size == sample_batch_size:
            return {
                "boxes_xyxy_norm": boxes_xyxy_norm,
                "labels": labels,
                "object_mask": object_mask,
            }

        if layout_batch_size <= 0 or sample_batch_size % layout_batch_size != 0:
            raise ValueError(
                "Layout batch size must either match the latent batch size or divide it "
                f"(got sample_batch_size={sample_batch_size}, layout_batch_size={layout_batch_size})."
            )

        repeat_factor = sample_batch_size // layout_batch_size
        return {
            "boxes_xyxy_norm": boxes_xyxy_norm.repeat_interleave(repeat_factor, dim=0),
            "labels": labels.repeat_interleave(repeat_factor, dim=0),
            "object_mask": object_mask.repeat_interleave(repeat_factor, dim=0),
        }

    def _build_regiondiff_context(
        self,
        sample_batch_size: int,
        *,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        broadcasted = self._broadcast_layout_inputs(
            sample_batch_size,
            boxes_xyxy_norm=boxes_xyxy_norm,
            labels=labels,
            object_mask=object_mask,
        )
        layout_tokens = self.layout_tokenizer(
            boxes_xyxy_norm=broadcasted["boxes_xyxy_norm"],
            labels=broadcasted["labels"],
            object_mask=broadcasted["object_mask"],
        )
        return {
            "layout_tokens": layout_tokens,
            "boxes_xyxy_norm": broadcasted["boxes_xyxy_norm"],
            "labels": broadcasted["labels"],
            "object_mask": broadcasted["object_mask"],
            "use_background_token": self.layout_tokenizer.use_background_token,
            "region_token_masks": {},
        }

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[torch.Tensor] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        region_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        layout_inputs = region_kwargs.pop("layout_inputs", None)
        if layout_inputs is not None and isinstance(layout_inputs, dict):
            for key, value in layout_inputs.items():
                region_kwargs.setdefault(key, value)

        boxes_xyxy_norm = region_kwargs.pop("boxes_xyxy_norm", None)
        labels = region_kwargs.pop("labels", None)
        object_mask = region_kwargs.pop("object_mask", None)

        if boxes_xyxy_norm is not None and labels is not None and object_mask is not None:
            region_kwargs["regiondiff_context"] = self._build_regiondiff_context(
                int(sample.shape[0]),
                boxes_xyxy_norm=boxes_xyxy_norm.to(device=sample.device, dtype=sample.dtype),
                labels=labels.to(device=sample.device),
                object_mask=object_mask.to(device=sample.device),
            )

        return self.base_unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=region_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )


class RegionDiffusionModelWrapper(nn.Module):
    """RegionDiff wrapper for denoisers without SD cross-attention kwargs."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        class_text_features: torch.Tensor,
        category_id_to_name: Dict[int, str],
        layout_token_dim: int,
        bbox_fourier_dim: int,
        same_class_position_slots: int,
        use_background_token: bool,
        active_region_resolutions: Iterable[int],
        backbone_kind: str = "unet2d",
        attachment_kind: str = "attention",
        use_spatial_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        object.__setattr__(self, "base_unet", base_model)
        self.backbone_kind = str(backbone_kind)
        self.attachment_kind = str(attachment_kind)
        self.use_spatial_adapter = bool(use_spatial_adapter)
        self.category_id_to_name = dict(category_id_to_name)
        self.active_region_resolutions = tuple(sorted(int(value) for value in active_region_resolutions))
        self.layout_tokenizer = LayoutTokenizer(
            class_text_features=class_text_features,
            layout_token_dim=layout_token_dim,
            bbox_fourier_dim=bbox_fourier_dim,
            same_class_position_slots=same_class_position_slots,
            use_background_token=use_background_token,
        )
        if self.attachment_kind == "attention":
            self.num_region_blocks = patch_attention_regiondiff_blocks(
                self.base_model,
                layout_token_dim=layout_token_dim,
                active_region_resolutions=self.active_region_resolutions,
            )
        elif self.attachment_kind == "transformer":
            self.num_region_blocks = patch_unet_regiondiff_blocks(
                self.base_model,
                layout_token_dim=layout_token_dim,
                active_region_resolutions=self.active_region_resolutions,
            )
        else:
            raise ValueError(f"Unknown RegionDiff attachment_kind={attachment_kind!r}")
        self.config = getattr(
            self.base_model,
            "config",
            SimpleNamespace(in_channels=None, out_channels=None, sample_size=None),
        )
        sample_channels = int(
            getattr(self.config, "in_channels", 0)
            or getattr(getattr(self.base_model, "conv_in", None), "in_channels", 0)
            or 0
        )
        if self.use_spatial_adapter:
            if sample_channels <= 0:
                raise ValueError("Could not infer sample channel count for RegionDiff spatial adapter.")
            self.layout_input_adapter = RegionSpatialAdapter(
                sample_channels=sample_channels,
                layout_token_dim=layout_token_dim,
            )
        else:
            self.layout_input_adapter = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _broadcast_layout_inputs(
        self,
        sample_batch_size: int,
        *,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return RegionDiffusionUNetWrapper._broadcast_layout_inputs(
            self,
            sample_batch_size,
            boxes_xyxy_norm=boxes_xyxy_norm,
            labels=labels,
            object_mask=object_mask,
        )

    def _build_regiondiff_context(
        self,
        sample_batch_size: int,
        *,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        return RegionDiffusionUNetWrapper._build_regiondiff_context(
            self,
            sample_batch_size,
            boxes_xyxy_norm=boxes_xyxy_norm,
            labels=labels,
            object_mask=object_mask,
        )

    @contextmanager
    def _regiondiff_context_scope(self, context: Optional[Dict[str, Any]]):
        blocks = list(iter_regiondiff_attention_blocks(self.base_model))
        try:
            for block in blocks:
                block.regiondiff_context = context
            yield
        finally:
            for block in blocks:
                block.regiondiff_context = None

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *args,
        boxes_xyxy_norm: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        layout_inputs: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        if layout_inputs is not None:
            boxes_xyxy_norm = boxes_xyxy_norm if boxes_xyxy_norm is not None else layout_inputs.get("boxes_xyxy_norm")
            labels = labels if labels is not None else layout_inputs.get("labels")
            object_mask = object_mask if object_mask is not None else layout_inputs.get("object_mask")

        context = None
        if boxes_xyxy_norm is not None and labels is not None and object_mask is not None:
            context = self._build_regiondiff_context(
                int(sample.shape[0]),
                boxes_xyxy_norm=boxes_xyxy_norm.to(device=sample.device, dtype=sample.dtype),
                labels=labels.to(device=sample.device),
                object_mask=object_mask.to(device=sample.device),
            )

        if context is not None and self.layout_input_adapter is not None:
            sample = sample + self.layout_input_adapter(
                sample,
                regiondiff_context=context,
            )

        with self._regiondiff_context_scope(context):
            return self.base_model(sample, timestep, *args, **kwargs)
