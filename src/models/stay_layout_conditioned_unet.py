"""STAY-inspired layout-conditioned pixel-space UNet wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fm_unet import build_fm_unet_from_config


def _resolve_num_groups(num_channels: int, preferred: int = 32) -> int:
    num_channels = int(num_channels)
    for groups in range(min(preferred, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _bbox_features(boxes_xyxy_norm: torch.Tensor) -> torch.Tensor:
    centers = 0.5 * (boxes_xyxy_norm[..., :2] + boxes_xyxy_norm[..., 2:])
    sizes = (boxes_xyxy_norm[..., 2:] - boxes_xyxy_norm[..., :2]).clamp(min=0.0)
    return torch.cat([boxes_xyxy_norm, centers, sizes], dim=-1)


def _dilate_masks(masks: torch.Tensor, dilation: int) -> torch.Tensor:
    if masks.numel() == 0:
        return masks
    if int(dilation) <= 0:
        return masks
    batch_size, max_objects, height, width = masks.shape
    dilated = masks.reshape(batch_size * max_objects, 1, height, width)
    for _ in range(int(dilation)):
        dilated = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
    return dilated.reshape(batch_size, max_objects, height, width)


def _empty_scalar(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((), device=device, dtype=dtype)


@dataclass
class STAYLayoutUNetOutput:
    """UNet-style output enriched with STAY-v2 debug tensors."""

    sample: torch.Tensor
    conditioning_maps: Optional[torch.Tensor] = None
    objectness_map: Optional[torch.Tensor] = None
    class_layout_map: Optional[torch.Tensor] = None
    feature_energy_map: Optional[torch.Tensor] = None
    object_embeddings: Optional[torch.Tensor] = None
    style_latents: Optional[torch.Tensor] = None
    soft_masks_full: Optional[torch.Tensor] = None
    hard_masks_full: Optional[torch.Tensor] = None
    non_overlap_masks_full: Optional[torch.Tensor] = None
    semantic_map: Optional[torch.Tensor] = None
    edge_map: Optional[torch.Tensor] = None
    overlap_map: Optional[torch.Tensor] = None
    occupancy_map: Optional[torch.Tensor] = None
    masked_context_map: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    mask_overlap_loss: Optional[torch.Tensor] = None
    mask_sharpness_loss: Optional[torch.Tensor] = None
    mask_activation_loss: Optional[torch.Tensor] = None


class ObjectRepresentationEncoder(nn.Module):
    """Encode explicit per-object representations from class, box, and style."""

    def __init__(
        self,
        *,
        num_classes: int,
        class_embed_dim: int,
        bbox_embed_dim: int,
        object_embed_dim: int,
        use_style_latent: bool,
        style_latent_dim: int,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.class_embed_dim = int(class_embed_dim)
        self.bbox_embed_dim = int(bbox_embed_dim)
        self.object_embed_dim = int(object_embed_dim)
        self.use_style_latent = bool(use_style_latent)
        self.style_latent_dim = int(style_latent_dim)

        self.class_embedding = nn.Embedding(max(self.num_classes, 1), self.class_embed_dim)
        self.class_projection = nn.Linear(self.class_embed_dim, self.object_embed_dim)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(8, self.bbox_embed_dim),
            nn.SiLU(),
            nn.Linear(self.bbox_embed_dim, self.object_embed_dim),
        )

        fused_input_dim = self.object_embed_dim * 2
        if self.use_style_latent:
            self.style_projection = nn.Linear(self.style_latent_dim, self.object_embed_dim)
            fused_input_dim += self.object_embed_dim
        else:
            self.style_projection = None

        self.fusion = nn.Sequential(
            nn.Linear(fused_input_dim, self.object_embed_dim),
            nn.SiLU(),
            nn.Linear(self.object_embed_dim, self.object_embed_dim),
        )

    def forward(
        self,
        *,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
        style_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_objects, _ = boxes_xyxy_norm.shape
        device = boxes_xyxy_norm.device
        dtype = boxes_xyxy_norm.dtype

        if max_objects == 0:
            empty_embeddings = torch.zeros(
                batch_size,
                0,
                self.object_embed_dim,
                device=device,
                dtype=dtype,
            )
            empty_style = torch.zeros(
                batch_size,
                0,
                self.style_latent_dim if self.use_style_latent else 0,
                device=device,
                dtype=dtype,
            )
            return empty_embeddings, empty_style

        safe_labels = labels.clamp(min=0)
        class_features = self.class_projection(self.class_embedding(safe_labels))
        bbox_features = self.bbox_mlp(_bbox_features(boxes_xyxy_norm))

        fusion_parts = [class_features, bbox_features]
        if self.use_style_latent:
            if style_noise is None:
                style_noise = torch.randn(
                    batch_size,
                    max_objects,
                    self.style_latent_dim,
                    device=device,
                    dtype=dtype,
                )
            else:
                style_noise = style_noise.to(device=device, dtype=dtype)
            style_features = self.style_projection(style_noise)
            fusion_parts.append(style_features)
            style_latents = style_noise
        else:
            style_latents = torch.zeros(batch_size, max_objects, 0, device=device, dtype=dtype)

        object_embeddings = self.fusion(torch.cat(fusion_parts, dim=-1))
        object_embeddings = object_embeddings * object_mask.unsqueeze(-1).to(dtype)

        return object_embeddings, style_latents


class ObjectMaskPredictor(nn.Module):
    """Predict local probabilistic masks from per-object embeddings."""

    def __init__(
        self,
        *,
        object_embed_dim: int,
        mask_resolution: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.object_embed_dim = int(object_embed_dim)
        self.mask_resolution = int(mask_resolution)
        self.hidden_channels = int(hidden_channels)

        self.seed_projection = nn.Linear(
            self.object_embed_dim,
            self.hidden_channels * self.mask_resolution * self.mask_resolution,
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        object_embeddings: torch.Tensor,
        *,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_objects, _ = object_embeddings.shape
        device = object_embeddings.device
        dtype = object_embeddings.dtype

        if max_objects == 0:
            return torch.zeros(
                batch_size,
                0,
                self.mask_resolution,
                self.mask_resolution,
                device=device,
                dtype=dtype,
            )

        seeds = self.seed_projection(object_embeddings)
        seeds = seeds.view(
            batch_size * max_objects,
            self.hidden_channels,
            self.mask_resolution,
            self.mask_resolution,
        )
        local_masks = torch.sigmoid(self.decoder(seeds))
        local_masks = local_masks.view(batch_size, max_objects, self.mask_resolution, self.mask_resolution)
        return local_masks * object_mask.unsqueeze(-1).unsqueeze(-1).to(dtype)


class LayoutMapAssembler(nn.Module):
    """Assemble predicted object masks into STAY-inspired spatial conditioning maps."""

    def __init__(
        self,
        *,
        num_classes: int,
        object_embed_dim: int,
        mask_threshold: float,
        edge_dilation: int,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.object_embed_dim = int(object_embed_dim)
        self.mask_threshold = float(mask_threshold)
        self.edge_dilation = int(edge_dilation)
        self.edge_gate = nn.Linear(self.object_embed_dim, self.object_embed_dim)

    def _place_masks(
        self,
        *,
        local_masks: torch.Tensor,
        boxes_xyxy_norm: torch.Tensor,
        object_mask: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        batch_size, max_objects, _, _ = local_masks.shape
        height, width = spatial_size
        device = local_masks.device
        dtype = local_masks.dtype
        soft_masks_full = torch.zeros(batch_size, max_objects, height, width, device=device, dtype=dtype)

        for batch_idx in range(batch_size):
            for object_idx in range(max_objects):
                if not bool(object_mask[batch_idx, object_idx]):
                    continue

                x1, y1, x2, y2 = boxes_xyxy_norm[batch_idx, object_idx]
                ix1 = max(0, min(width - 1, int(torch.floor(x1 * width).item())))
                iy1 = max(0, min(height - 1, int(torch.floor(y1 * height).item())))
                ix2 = max(ix1 + 1, min(width, int(torch.ceil(x2 * width).item())))
                iy2 = max(iy1 + 1, min(height, int(torch.ceil(y2 * height).item())))

                resized_mask = F.interpolate(
                    local_masks[batch_idx, object_idx].view(1, 1, local_masks.shape[-2], local_masks.shape[-1]),
                    size=(iy2 - iy1, ix2 - ix1),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                soft_masks_full[batch_idx, object_idx, iy1:iy2, ix1:ix2] = resized_mask

        return soft_masks_full

    def _compute_owner_masks(
        self,
        *,
        hard_masks_full: torch.Tensor,
        soft_masks_full: torch.Tensor,
        boxes_xyxy_norm: torch.Tensor,
        object_mask: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        batch_size, max_objects, height, width = hard_masks_full.shape
        device = hard_masks_full.device
        dtype = hard_masks_full.dtype

        if max_objects == 0:
            return torch.zeros_like(hard_masks_full)

        bbox_area = (
            (boxes_xyxy_norm[..., 2] - boxes_xyxy_norm[..., 0]).clamp(min=1.0 / width)
            * (boxes_xyxy_norm[..., 3] - boxes_xyxy_norm[..., 1]).clamp(min=1.0 / height)
            * float(height * width)
        )
        mask_area = hard_masks_full.sum(dim=(2, 3))
        priority_area = torch.where(mask_area > 0.0, mask_area, bbox_area).clamp(min=1.0)
        priority_area = torch.where(
            object_mask,
            priority_area,
            torch.full_like(priority_area, float("inf")),
        )

        owner_masks = torch.zeros_like(hard_masks_full)
        for batch_idx in range(batch_size):
            active_masks = hard_masks_full[batch_idx] > 0.5
            if not bool(active_masks.any()):
                continue
            scores = priority_area[batch_idx].view(max_objects, 1, 1).expand(max_objects, height, width)
            scores = torch.where(active_masks, scores, torch.full_like(scores, float("inf")))
            owner_idx = scores.argmin(dim=0)
            owner_valid = active_masks.any(dim=0)
            owner_one_hot = F.one_hot(owner_idx, num_classes=max_objects).permute(2, 0, 1).to(dtype)
            owner_masks[batch_idx] = owner_one_hot * owner_valid.unsqueeze(0).to(dtype)

        priority_soft = soft_masks_full * (1.0 / priority_area.clamp(min=1.0)).unsqueeze(-1).unsqueeze(-1)
        priority_soft = priority_soft * object_mask.unsqueeze(-1).unsqueeze(-1).to(dtype)
        priority_soft_sum = priority_soft.sum(dim=1, keepdim=True)
        owner_soft = torch.where(
            priority_soft_sum > 0,
            priority_soft / priority_soft_sum.clamp(min=1e-6),
            torch.zeros_like(priority_soft),
        )
        return owner_masks + owner_soft - owner_soft.detach()

    def forward(
        self,
        *,
        object_embeddings: torch.Tensor,
        local_masks: torch.Tensor,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        batch_size, max_objects, _ = object_embeddings.shape
        height, width = spatial_size
        device = object_embeddings.device
        dtype = object_embeddings.dtype

        if max_objects == 0:
            zeros_map = torch.zeros(batch_size, self.object_embed_dim, height, width, device=device, dtype=dtype)
            zeros_mask = torch.zeros(batch_size, 0, height, width, device=device, dtype=dtype)
            zeros_1 = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
            return {
                "soft_masks_full": zeros_mask,
                "hard_masks_full": zeros_mask,
                "non_overlap_masks_full": zeros_mask,
                "semantic_map": zeros_map,
                "edge_map": zeros_map,
                "overlap_map": zeros_1,
                "occupancy_map": zeros_1,
                "masked_context_map": zeros_map,
                "conditioning_maps": torch.cat([zeros_map, zeros_map], dim=1),
                "objectness_map": zeros_1,
                "class_layout_map": zeros_1,
                "feature_energy_map": zeros_1,
            }

        soft_masks_full = self._place_masks(
            local_masks=local_masks,
            boxes_xyxy_norm=boxes_xyxy_norm,
            object_mask=object_mask,
            spatial_size=spatial_size,
        )
        valid_mask_f = object_mask.unsqueeze(-1).unsqueeze(-1).to(dtype)
        soft_masks_full = soft_masks_full * valid_mask_f
        hard_masks_full = (soft_masks_full >= self.mask_threshold).to(dtype) * valid_mask_f

        non_overlap_masks_full = self._compute_owner_masks(
            hard_masks_full=hard_masks_full,
            soft_masks_full=soft_masks_full,
            boxes_xyxy_norm=boxes_xyxy_norm,
            object_mask=object_mask,
            spatial_size=spatial_size,
        )

        semantic_map = torch.einsum("bnhw,bnd->bdhw", non_overlap_masks_full, object_embeddings)
        occupancy_map = non_overlap_masks_full.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        overlap_map = (hard_masks_full.sum(dim=1, keepdim=True) > 1.0).to(dtype)

        dilated_soft = _dilate_masks(soft_masks_full, self.edge_dilation)
        dilated_hard = _dilate_masks(hard_masks_full, self.edge_dilation)
        edge_soft = (dilated_soft - soft_masks_full).clamp(min=0.0, max=1.0) * valid_mask_f
        edge_hard = ((dilated_hard - hard_masks_full) > 0.0).to(dtype) * valid_mask_f

        edge_gate = torch.sigmoid(self.edge_gate(object_embeddings)) * object_mask.unsqueeze(-1).to(dtype)
        gated_embeddings = object_embeddings * edge_gate
        edge_weights = edge_hard + edge_soft - edge_soft.detach()
        edge_map = torch.einsum("bnhw,bnd->bdhw", edge_weights, gated_embeddings)

        soft_context_denominator = soft_masks_full.sum(dim=1, keepdim=True)
        masked_context_map = torch.where(
            soft_context_denominator > 0,
            torch.einsum("bnhw,bnd->bdhw", soft_masks_full, object_embeddings)
            / soft_context_denominator.clamp(min=1e-6),
            torch.zeros(batch_size, self.object_embed_dim, height, width, device=device, dtype=dtype),
        )

        class_values = ((labels.clamp(min=0).to(dtype) + 1.0) / float(max(self.num_classes, 1)))
        class_layout_map = torch.einsum("bnhw,bn->bhw", non_overlap_masks_full, class_values).unsqueeze(1)

        conditioning_maps = torch.cat([semantic_map, edge_map], dim=1)
        feature_energy_map = conditioning_maps.pow(2).mean(dim=1, keepdim=True).sqrt()

        return {
            "soft_masks_full": soft_masks_full,
            "hard_masks_full": hard_masks_full,
            "non_overlap_masks_full": non_overlap_masks_full,
            "semantic_map": semantic_map,
            "edge_map": edge_map,
            "overlap_map": overlap_map,
            "occupancy_map": occupancy_map,
            "masked_context_map": masked_context_map,
            "conditioning_maps": conditioning_maps,
            "objectness_map": occupancy_map,
            "class_layout_map": class_layout_map,
            "feature_energy_map": feature_energy_map,
        }


class EANormAdapter(nn.Module):
    """Approximate STAY Edge-Aware Normalization with spatially varying gamma and beta."""

    def __init__(
        self,
        *,
        feature_channels: int,
        condition_channels: int,
        hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels or max(feature_channels, condition_channels))
        self.norm = nn.GroupNorm(
            num_groups=_resolve_num_groups(feature_channels),
            num_channels=feature_channels,
            eps=1e-5,
        )
        self.to_gamma_beta = nn.Sequential(
            nn.Conv2d(condition_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, feature_channels * 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(conditioning).chunk(2, dim=1)
        normalized = self.norm(hidden_states)
        return normalized * (1.0 + gamma) + beta


class MaskedObjectContextBlock(nn.Module):
    """Residual object-context adapter approximating STAY Styled Mask Attention."""

    def __init__(self, *, feature_channels: int, object_embed_dim: int) -> None:
        super().__init__()
        self.object_projection = nn.Linear(object_embed_dim, feature_channels)
        self.residual = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        object_embeddings: torch.Tensor,
        soft_masks_full: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        if object_embeddings.shape[1] == 0:
            context = torch.zeros_like(hidden_states)
            return hidden_states, context

        masks = F.interpolate(
            soft_masks_full,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ) * object_mask.unsqueeze(-1).unsqueeze(-1).to(dtype)
        projected_objects = self.object_projection(object_embeddings).to(dtype)
        context = torch.einsum("bnhw,bnc->bchw", masks, projected_objects)
        denom = masks.sum(dim=1, keepdim=True)
        context = torch.where(
            denom > 0,
            context / denom.clamp(min=1e-6),
            torch.zeros(batch_size, hidden_states.shape[1], height, width, device=device, dtype=dtype),
        )
        return hidden_states + self.residual(context), context


class STAYLayoutConditionedPixelUNet(nn.Module):
    """STAY-inspired object-mask conditioned pixel-space UNet wrapper."""

    def __init__(
        self,
        base_unet: nn.Module,
        *,
        image_in_channels: int,
        num_classes: int,
        class_embed_dim: int,
        bbox_embed_dim: int,
        object_embed_dim: int,
        use_style_latent: bool,
        style_latent_dim: int,
        style_seed: int,
        mask_resolution: int,
        mask_hidden_channels: int,
        mask_threshold: float,
        edge_dilation: int,
        injection_mode: str,
        use_masked_context: bool,
        mask_overlap_loss_weight: float,
        mask_sharpness_loss_weight: float,
        mask_activation_loss_weight: float,
        category_id_to_name: Optional[Dict[int, str]] = None,
        base_unet_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.base_unet = base_unet
        self.image_in_channels = int(image_in_channels)
        self.num_classes = int(num_classes)
        self.class_embed_dim = int(class_embed_dim)
        self.bbox_embed_dim = int(bbox_embed_dim)
        self.object_embed_dim = int(object_embed_dim)
        self.use_style_latent = bool(use_style_latent)
        self.style_latent_dim = int(style_latent_dim)
        self.style_seed = int(style_seed)
        self.mask_resolution = int(mask_resolution)
        self.mask_hidden_channels = int(mask_hidden_channels)
        self.mask_threshold = float(mask_threshold)
        self.edge_dilation = int(edge_dilation)
        self.injection_mode = str(injection_mode)
        self.use_masked_context = bool(use_masked_context)
        self.mask_overlap_loss_weight = float(mask_overlap_loss_weight)
        self.mask_sharpness_loss_weight = float(mask_sharpness_loss_weight)
        self.mask_activation_loss_weight = float(mask_activation_loss_weight)
        self.category_id_to_name = dict(category_id_to_name or {})
        self.base_unet_config = dict(base_unet_config or {})
        if self.injection_mode != "ea_norm":
            raise ValueError(
                f"STAYLayoutConditionedPixelUNet only supports injection_mode='ea_norm', got {self.injection_mode!r}"
            )
        self.config = SimpleNamespace(
            in_channels=self.image_in_channels,
            out_channels=self.image_in_channels,
            sample_size=self.base_unet.config.sample_size,
        )

        self.object_encoder = ObjectRepresentationEncoder(
            num_classes=self.num_classes,
            class_embed_dim=self.class_embed_dim,
            bbox_embed_dim=self.bbox_embed_dim,
            object_embed_dim=self.object_embed_dim,
            use_style_latent=self.use_style_latent,
            style_latent_dim=self.style_latent_dim,
        )
        self.mask_predictor = ObjectMaskPredictor(
            object_embed_dim=self.object_embed_dim,
            mask_resolution=self.mask_resolution,
            hidden_channels=self.mask_hidden_channels,
        )
        self.map_assembler = LayoutMapAssembler(
            num_classes=self.num_classes,
            object_embed_dim=self.object_embed_dim,
            mask_threshold=self.mask_threshold,
            edge_dilation=self.edge_dilation,
        )

        block_channels = list(self.base_unet.config.block_out_channels)
        up_channels = list(reversed(block_channels))
        condition_channels = self.object_embed_dim * 2 + 1

        self.ea_norm_adapters = nn.ModuleDict(
            {"stem": EANormAdapter(feature_channels=block_channels[0], condition_channels=condition_channels)}
        )
        for idx, channels in enumerate(block_channels):
            self.ea_norm_adapters[f"down_{idx}"] = EANormAdapter(
                feature_channels=channels,
                condition_channels=condition_channels,
            )
        if self.base_unet.mid_block is not None:
            self.ea_norm_adapters["mid"] = EANormAdapter(
                feature_channels=block_channels[-1],
                condition_channels=condition_channels,
            )
        for idx, channels in enumerate(up_channels):
            self.ea_norm_adapters[f"up_{idx}"] = EANormAdapter(
                feature_channels=channels,
                condition_channels=condition_channels,
            )

        self.masked_context_blocks = nn.ModuleDict()
        if self.use_masked_context:
            for idx, down_block in enumerate(self.base_unet.down_blocks):
                if hasattr(down_block, "attentions"):
                    self.masked_context_blocks[f"down_{idx}"] = MaskedObjectContextBlock(
                        feature_channels=block_channels[idx],
                        object_embed_dim=self.object_embed_dim,
                    )
            if self.base_unet.mid_block is not None and hasattr(self.base_unet.mid_block, "attentions"):
                self.masked_context_blocks["mid"] = MaskedObjectContextBlock(
                    feature_channels=block_channels[-1],
                    object_embed_dim=self.object_embed_dim,
                )
            for idx, up_block in enumerate(self.base_unet.up_blocks):
                if hasattr(up_block, "attentions"):
                    self.masked_context_blocks[f"up_{idx}"] = MaskedObjectContextBlock(
                        feature_channels=up_channels[idx],
                        object_embed_dim=self.object_embed_dim,
                    )

    def sample_style_noise(
        self,
        *,
        batch_size: int,
        max_objects: int,
        device: torch.device | str,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if not self.use_style_latent:
            return torch.zeros(batch_size, max_objects, 0, device=device, dtype=dtype)
        return torch.randn(
            batch_size,
            max_objects,
            self.style_latent_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _compute_aux_losses(
        self,
        *,
        local_masks: torch.Tensor,
        soft_masks_full: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = soft_masks_full.device
        dtype = soft_masks_full.dtype
        if local_masks.shape[1] == 0 or not bool(object_mask.any()):
            zero = _empty_scalar(device, dtype)
            return {
                "mask_overlap_loss": zero,
                "mask_sharpness_loss": zero,
                "mask_activation_loss": zero,
                "aux_loss": zero,
            }

        valid_local_masks = local_masks[object_mask]
        mask_sharpness_loss = (valid_local_masks * (1.0 - valid_local_masks)).mean()
        mask_activation_loss = torch.relu(0.05 - valid_local_masks.mean(dim=(-1, -2))).mean()

        max_objects = soft_masks_full.shape[1]
        if max_objects <= 1:
            mask_overlap_loss = _empty_scalar(device, dtype)
        else:
            pairwise_overlap = torch.einsum("bihw,bjhw->bij", soft_masks_full, soft_masks_full)
            pairwise_overlap = pairwise_overlap / float(max(1, soft_masks_full.shape[-2] * soft_masks_full.shape[-1]))
            pair_mask = object_mask.unsqueeze(1) & object_mask.unsqueeze(2)
            upper = torch.triu(
                torch.ones(max_objects, max_objects, device=device, dtype=torch.bool),
                diagonal=1,
            )
            pair_mask = pair_mask & upper.unsqueeze(0)
            if bool(pair_mask.any()):
                mask_overlap_loss = pairwise_overlap[pair_mask].mean()
            else:
                mask_overlap_loss = _empty_scalar(device, dtype)

        aux_loss = (
            self.mask_overlap_loss_weight * mask_overlap_loss
            + self.mask_sharpness_loss_weight * mask_sharpness_loss
            + self.mask_activation_loss_weight * mask_activation_loss
        )
        return {
            "mask_overlap_loss": mask_overlap_loss,
            "mask_sharpness_loss": mask_sharpness_loss,
            "mask_activation_loss": mask_activation_loss,
            "aux_loss": aux_loss,
        }

    def build_conditioning(
        self,
        *,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
        spatial_size: Tuple[int, int],
        style_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        object_embeddings, style_latents = self.object_encoder(
            boxes_xyxy_norm=boxes_xyxy_norm,
            labels=labels,
            object_mask=object_mask,
            style_noise=style_noise,
        )
        local_masks = self.mask_predictor(object_embeddings, object_mask=object_mask)
        maps = self.map_assembler(
            object_embeddings=object_embeddings,
            local_masks=local_masks,
            boxes_xyxy_norm=boxes_xyxy_norm,
            labels=labels,
            object_mask=object_mask,
            spatial_size=spatial_size,
        )
        aux_losses = self._compute_aux_losses(
            local_masks=local_masks,
            soft_masks_full=maps["soft_masks_full"],
            object_mask=object_mask,
        )

        maps.update(
            {
                "object_embeddings": object_embeddings,
                "style_latents": style_latents,
                "local_masks": local_masks,
                **aux_losses,
            }
        )
        return maps

    def _compute_time_embedding(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        class_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)
        t_emb = self.base_unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.base_unet.time_embedding(t_emb)

        if self.base_unet.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")
            if self.base_unet.config.class_embed_type == "timestep":
                class_labels = self.base_unet.time_proj(class_labels)
            class_emb = self.base_unet.class_embedding(class_labels).to(dtype=sample.dtype)
            emb = emb + class_emb
        elif class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        return emb, timesteps

    def _resized_conditioning(
        self,
        debug: Dict[str, torch.Tensor],
        *,
        spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        semantic = F.interpolate(debug["semantic_map"], size=spatial_size, mode="bilinear", align_corners=False)
        edge = F.interpolate(debug["edge_map"], size=spatial_size, mode="bilinear", align_corners=False)
        occupancy = F.interpolate(debug["occupancy_map"], size=spatial_size, mode="nearest")
        return torch.cat([semantic, edge, occupancy], dim=1)

    def _apply_conditioning_stage(
        self,
        stage_name: str,
        hidden_states: torch.Tensor,
        debug: Dict[str, torch.Tensor],
        *,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        conditioning = self._resized_conditioning(debug, spatial_size=(hidden_states.shape[-2], hidden_states.shape[-1]))
        hidden_states = self.ea_norm_adapters[stage_name](hidden_states, conditioning)
        if stage_name in self.masked_context_blocks:
            hidden_states, _ = self.masked_context_blocks[stage_name](
                hidden_states,
                object_embeddings=debug["object_embeddings"],
                soft_masks_full=debug["soft_masks_full"],
                object_mask=object_mask,
            )
        return hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
        style_noise: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        return_layout_debug: bool = False,
    ) -> STAYLayoutUNetOutput:
        if self.base_unet.config.center_input_sample:
            sample = 2.0 * sample - 1.0

        debug = self.build_conditioning(
            boxes_xyxy_norm=boxes_xyxy_norm,
            labels=labels,
            object_mask=object_mask,
            spatial_size=(sample.shape[-2], sample.shape[-1]),
            style_noise=style_noise,
        )
        emb, timesteps = self._compute_time_embedding(sample, timestep, class_labels=class_labels)

        skip_sample = sample
        sample = self.base_unet.conv_in(sample)
        sample = self._apply_conditioning_stage("stem", sample, debug, object_mask=object_mask)

        down_block_res_samples = (sample,)
        for idx, downsample_block in enumerate(self.base_unet.down_blocks):
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    skip_sample=skip_sample,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            sample = self._apply_conditioning_stage(f"down_{idx}", sample, debug, object_mask=object_mask)

            if len(res_samples) > 0:
                res_samples = list(res_samples)
                res_samples[-1] = sample
                res_samples = tuple(res_samples)

            down_block_res_samples += res_samples

        if self.base_unet.mid_block is not None:
            sample = self.base_unet.mid_block(sample, emb)
            sample = self._apply_conditioning_stage("mid", sample, debug, object_mask=object_mask)

        skip_sample = None
        for idx, upsample_block in enumerate(self.base_unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

            sample = self._apply_conditioning_stage(f"up_{idx}", sample, debug, object_mask=object_mask)

        if self.base_unet.conv_norm_out:
            sample = self.base_unet.conv_norm_out(sample)
            sample = self.base_unet.conv_act(sample)
        sample = self.base_unet.conv_out(sample)

        if skip_sample is not None:
            sample = sample + skip_sample

        if self.base_unet.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_layout_debug:
            return STAYLayoutUNetOutput(
                sample=sample,
                aux_loss=debug["aux_loss"],
                mask_overlap_loss=debug["mask_overlap_loss"],
                mask_sharpness_loss=debug["mask_sharpness_loss"],
                mask_activation_loss=debug["mask_activation_loss"],
            )

        return STAYLayoutUNetOutput(
            sample=sample,
            conditioning_maps=debug["conditioning_maps"],
            objectness_map=debug["objectness_map"],
            class_layout_map=debug["class_layout_map"],
            feature_energy_map=debug["feature_energy_map"],
            object_embeddings=debug["object_embeddings"],
            style_latents=debug["style_latents"],
            soft_masks_full=debug["soft_masks_full"],
            hard_masks_full=debug["hard_masks_full"],
            non_overlap_masks_full=debug["non_overlap_masks_full"],
            semantic_map=debug["semantic_map"],
            edge_map=debug["edge_map"],
            overlap_map=debug["overlap_map"],
            occupancy_map=debug["occupancy_map"],
            masked_context_map=debug["masked_context_map"],
            aux_loss=debug["aux_loss"],
            mask_overlap_loss=debug["mask_overlap_loss"],
            mask_sharpness_loss=debug["mask_sharpness_loss"],
            mask_activation_loss=debug["mask_activation_loss"],
        )


def build_stay_layout_conditioned_pixel_unet(
    config: Dict[str, Any],
    *,
    image_in_channels: int,
    num_classes: int,
    class_embed_dim: int,
    bbox_embed_dim: int,
    object_embed_dim: int,
    use_style_latent: bool,
    style_latent_dim: int,
    style_seed: int,
    mask_resolution: int,
    mask_hidden_channels: int,
    mask_threshold: float,
    edge_dilation: int,
    injection_mode: str,
    use_masked_context: bool,
    mask_overlap_loss_weight: float,
    mask_sharpness_loss_weight: float,
    mask_activation_loss_weight: float,
    category_id_to_name: Optional[Dict[int, str]] = None,
    device: Optional[str] = None,
) -> STAYLayoutConditionedPixelUNet:
    """Instantiate the STAY-inspired layout-conditioned wrapper around a base pixel UNet."""
    base_unet = build_fm_unet_from_config(config, device=device)
    wrapped_unet = STAYLayoutConditionedPixelUNet(
        base_unet,
        image_in_channels=image_in_channels,
        num_classes=num_classes,
        class_embed_dim=class_embed_dim,
        bbox_embed_dim=bbox_embed_dim,
        object_embed_dim=object_embed_dim,
        use_style_latent=use_style_latent,
        style_latent_dim=style_latent_dim,
        style_seed=style_seed,
        mask_resolution=mask_resolution,
        mask_hidden_channels=mask_hidden_channels,
        mask_threshold=mask_threshold,
        edge_dilation=edge_dilation,
        injection_mode=injection_mode,
        use_masked_context=use_masked_context,
        mask_overlap_loss_weight=mask_overlap_loss_weight,
        mask_sharpness_loss_weight=mask_sharpness_loss_weight,
        mask_activation_loss_weight=mask_activation_loss_weight,
        category_id_to_name=category_id_to_name,
        base_unet_config=config,
    )
    if device is not None:
        wrapped_unet = wrapped_unet.to(device)
    return wrapped_unet
