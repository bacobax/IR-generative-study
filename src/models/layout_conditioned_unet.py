"""Layout-conditioned pixel-space UNet wrapper for flow matching."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.fm_unet import build_fm_unet_from_config


@dataclass
class LayoutUNetOutput:
    """UNet-style output with optional debug tensors."""

    sample: torch.Tensor
    conditioning_maps: Optional[torch.Tensor] = None
    objectness_map: Optional[torch.Tensor] = None
    class_layout_map: Optional[torch.Tensor] = None
    feature_energy_map: Optional[torch.Tensor] = None


class LayoutConditionedPixelUNet(nn.Module):
    """Wrap a pixel-space UNet with a trainable bbox/layout conditioning path."""

    def __init__(
        self,
        base_unet: nn.Module,
        *,
        image_in_channels: int,
        num_classes: int,
        class_embed_dim: int = 32,
        bbox_embed_dim: int = 32,
        spatial_channels: int = 8,
        raster_mode: str = "box_fill_mean",
        category_id_to_name: Optional[Dict[int, str]] = None,
        base_unet_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.base_unet = base_unet
        self.image_in_channels = int(image_in_channels)
        self.num_classes = int(num_classes)
        self.class_embed_dim = int(class_embed_dim)
        self.bbox_embed_dim = int(bbox_embed_dim)
        self.spatial_channels = int(spatial_channels)
        self.raster_mode = raster_mode
        self.category_id_to_name = dict(category_id_to_name or {})
        self.base_unet_config = dict(base_unet_config or {})
        self.config = SimpleNamespace(
            in_channels=self.image_in_channels,
            out_channels=self.image_in_channels,
            sample_size=self.base_unet.config.sample_size,
        )

        self.class_embedding = nn.Embedding(max(self.num_classes, 1), self.class_embed_dim)
        self.class_projection = nn.Linear(self.class_embed_dim, self.spatial_channels)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(8, self.bbox_embed_dim),
            nn.SiLU(),
            nn.Linear(self.bbox_embed_dim, self.spatial_channels),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.spatial_channels * 2, self.spatial_channels),
            nn.SiLU(),
            nn.Linear(self.spatial_channels, self.spatial_channels),
        )
        self.condition_projection = nn.Sequential(
            nn.Conv2d(self.spatial_channels + 1, self.spatial_channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(self.spatial_channels, self.spatial_channels, kernel_size=3, padding=1, bias=False),
        )

    @staticmethod
    def _bbox_features(boxes_xyxy_norm: torch.Tensor) -> torch.Tensor:
        centers = 0.5 * (boxes_xyxy_norm[..., :2] + boxes_xyxy_norm[..., 2:])
        sizes = (boxes_xyxy_norm[..., 2:] - boxes_xyxy_norm[..., :2]).clamp(min=0.0)
        return torch.cat([boxes_xyxy_norm, centers, sizes], dim=-1)

    def build_conditioning(
        self,
        *,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Rasterize object embeddings into spatial conditioning maps."""
        batch_size, max_objects, _ = boxes_xyxy_norm.shape
        height, width = spatial_size
        device = boxes_xyxy_norm.device
        dtype = boxes_xyxy_norm.dtype

        safe_labels = labels.clamp(min=0)
        class_features = self.class_projection(self.class_embedding(safe_labels))
        bbox_features = self.bbox_mlp(self._bbox_features(boxes_xyxy_norm))
        object_features = self.fusion(torch.cat([class_features, bbox_features], dim=-1))
        object_features = object_features * object_mask.unsqueeze(-1).to(object_features.dtype)

        layout_sum = torch.zeros(batch_size, self.spatial_channels, height, width, device=device, dtype=dtype)
        class_layout_map = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
        counts = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)

        for batch_idx in range(batch_size):
            for obj_idx in range(max_objects):
                if not bool(object_mask[batch_idx, obj_idx]):
                    continue

                x1, y1, x2, y2 = boxes_xyxy_norm[batch_idx, obj_idx]
                ix1 = max(0, min(width - 1, int(torch.floor(x1 * width).item())))
                iy1 = max(0, min(height - 1, int(torch.floor(y1 * height).item())))
                ix2 = max(ix1 + 1, min(width, int(torch.ceil(x2 * width).item())))
                iy2 = max(iy1 + 1, min(height, int(torch.ceil(y2 * height).item())))

                feature = object_features[batch_idx, obj_idx].view(self.spatial_channels, 1, 1)
                layout_sum[batch_idx, :, iy1:iy2, ix1:ix2] += feature
                counts[batch_idx, :, iy1:iy2, ix1:ix2] += 1.0
                class_value = float(labels[batch_idx, obj_idx].item() + 1) / float(max(self.num_classes, 1))
                class_layout_map[batch_idx, :, iy1:iy2, ix1:ix2] = class_value

        objectness_map = (counts > 0).to(dtype)
        averaged = layout_sum / counts.clamp(min=1.0)
        conditioning_maps = self.condition_projection(torch.cat([averaged, objectness_map], dim=1))
        feature_energy_map = conditioning_maps.pow(2).mean(dim=1, keepdim=True).sqrt()

        return {
            "conditioning_maps": conditioning_maps,
            "objectness_map": objectness_map,
            "class_layout_map": class_layout_map,
            "feature_energy_map": feature_energy_map,
        }

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
        object_mask: torch.Tensor,
        return_layout_debug: bool = False,
    ) -> LayoutUNetOutput:
        debug = self.build_conditioning(
            boxes_xyxy_norm=boxes_xyxy_norm,
            labels=labels,
            object_mask=object_mask,
            spatial_size=(sample.shape[-2], sample.shape[-1]),
        )
        model_input = torch.cat([sample, debug["conditioning_maps"]], dim=1)
        output = self.base_unet(model_input, timestep).sample

        if not return_layout_debug:
            return LayoutUNetOutput(sample=output)

        return LayoutUNetOutput(
            sample=output,
            conditioning_maps=debug["conditioning_maps"],
            objectness_map=debug["objectness_map"],
            class_layout_map=debug["class_layout_map"],
            feature_energy_map=debug["feature_energy_map"],
        )


def build_layout_conditioned_pixel_unet(
    config: Dict[str, Any],
    *,
    image_in_channels: int,
    num_classes: int,
    class_embed_dim: int,
    bbox_embed_dim: int,
    spatial_channels: int,
    raster_mode: str = "box_fill_mean",
    category_id_to_name: Optional[Dict[int, str]] = None,
    device: Optional[str] = None,
) -> LayoutConditionedPixelUNet:
    """Instantiate the layout-conditioned wrapper around a base pixel UNet."""
    base_unet = build_fm_unet_from_config(config, device=device)
    wrapped_unet = LayoutConditionedPixelUNet(
        base_unet,
        image_in_channels=image_in_channels,
        num_classes=num_classes,
        class_embed_dim=class_embed_dim,
        bbox_embed_dim=bbox_embed_dim,
        spatial_channels=spatial_channels,
        raster_mode=raster_mode,
        category_id_to_name=category_id_to_name,
        base_unet_config=config,
    )
    if device is not None:
        wrapped_unet = wrapped_unet.to(device)
    return wrapped_unet


def save_layout_conditioning_metadata(metadata: Dict[str, Any], path: str) -> None:
    """Persist layout-conditioning metadata for later inspection."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
