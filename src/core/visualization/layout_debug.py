"""Visualization helpers for layout-conditioned flow matching debug logs."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch


_PALETTE = torch.tensor(
    [
        [0.95, 0.35, 0.35],
        [0.20, 0.70, 0.95],
        [0.35, 0.85, 0.45],
        [0.95, 0.80, 0.25],
        [0.85, 0.45, 0.95],
        [0.25, 0.90, 0.85],
        [0.98, 0.55, 0.20],
        [0.65, 0.70, 0.98],
    ],
    dtype=torch.float32,
)


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().to(torch.float32)


def ensure_rgb(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a ``(B, C, H, W)`` tensor to RGB on CPU."""
    tensor = _to_cpu_float(tensor)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    elif tensor.shape[1] > 3:
        tensor = tensor[:, :3]
    return tensor.clamp(0.0, 1.0)


def normalize_feature_map(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize each sample independently into ``[0, 1]`` for display."""
    tensor = _to_cpu_float(tensor)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    flat = tensor.flatten(start_dim=1)
    min_vals = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_vals = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    denom = (max_vals - min_vals).clamp(min=1e-6)
    normalized = (tensor - min_vals) / denom
    return ensure_rgb(normalized)


def render_class_layout(
    *,
    boxes_xyxy: torch.Tensor,
    labels: torch.Tensor,
    object_mask: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Render color-coded box fills for a batch of padded layout annotations."""
    boxes_xyxy = _to_cpu_float(boxes_xyxy)
    labels = labels.detach().cpu().to(torch.long)
    object_mask = object_mask.detach().cpu().to(torch.bool)

    batch_size = boxes_xyxy.shape[0]
    canvas = torch.zeros(batch_size, 3, image_size, image_size, dtype=torch.float32)

    for batch_idx in range(batch_size):
        for obj_idx in range(boxes_xyxy.shape[1]):
            if not object_mask[batch_idx, obj_idx]:
                continue
            x1, y1, x2, y2 = boxes_xyxy[batch_idx, obj_idx].tolist()
            ix1 = max(0, min(image_size - 1, int(round(x1))))
            iy1 = max(0, min(image_size - 1, int(round(y1))))
            ix2 = max(ix1 + 1, min(image_size, int(round(x2))))
            iy2 = max(iy1 + 1, min(image_size, int(round(y2))))
            color = _PALETTE[int(labels[batch_idx, obj_idx].item()) % len(_PALETTE)]
            canvas[batch_idx, :, iy1:iy2, ix1:ix2] = (
                0.65 * canvas[batch_idx, :, iy1:iy2, ix1:ix2] + 0.35 * color.view(3, 1, 1)
            )
    return canvas.clamp(0.0, 1.0)


def draw_bbox_overlays(
    images: torch.Tensor,
    *,
    boxes_xyxy: torch.Tensor,
    object_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    line_width: int = 2,
) -> torch.Tensor:
    """Draw light RGB bbox edges over images already normalized into ``[0, 1]``."""
    images = ensure_rgb(images).clone()
    boxes_xyxy = _to_cpu_float(boxes_xyxy)
    object_mask = object_mask.detach().cpu().to(torch.bool)
    if labels is None:
        labels = torch.zeros_like(object_mask, dtype=torch.long)
    else:
        labels = labels.detach().cpu().to(torch.long)

    _, _, image_h, image_w = images.shape

    for batch_idx in range(images.shape[0]):
        for obj_idx in range(boxes_xyxy.shape[1]):
            if not object_mask[batch_idx, obj_idx]:
                continue
            x1, y1, x2, y2 = boxes_xyxy[batch_idx, obj_idx].tolist()
            ix1 = max(0, min(image_w - 1, int(round(x1))))
            iy1 = max(0, min(image_h - 1, int(round(y1))))
            ix2 = max(ix1 + 1, min(image_w, int(round(x2))))
            iy2 = max(iy1 + 1, min(image_h, int(round(y2))))
            color = _PALETTE[int(labels[batch_idx, obj_idx].item()) % len(_PALETTE)].view(3, 1, 1)

            for offset in range(line_width):
                top = min(image_h - 1, iy1 + offset)
                bottom = max(0, iy2 - 1 - offset)
                left = min(image_w - 1, ix1 + offset)
                right = max(0, ix2 - 1 - offset)

                images[batch_idx, :, top:top + 1, ix1:ix2] = color
                images[batch_idx, :, bottom:bottom + 1, ix1:ix2] = color
                images[batch_idx, :, iy1:iy2, left:left + 1] = color
                images[batch_idx, :, iy1:iy2, right:right + 1] = color

    return images.clamp(0.0, 1.0)


def make_side_by_side_panel(images: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of image batches horizontally."""
    rgb_batches = [ensure_rgb(image) for image in images]
    return torch.cat(rgb_batches, dim=-1)


def save_image_batch(
    images: torch.Tensor,
    *,
    output_dir: str,
    prefix: str,
) -> List[str]:
    """Persist a batch of RGB images as PNGs."""
    from PIL import Image

    images = ensure_rgb(images)
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []

    for idx in range(images.shape[0]):
        arr = (images[idx].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        path = os.path.join(output_dir, f"{prefix}_{idx:02d}.png")
        Image.fromarray(arr).save(path)
        saved_paths.append(path)

    return saved_paths
