"""Reusable image transforms and augmentation schedules."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.core.constants import DEFAULT_IMAGE_SIZE
from src.core.normalization import (
    RAW_UINT16_PERCENTILE,
    resize_and_normalize,
    resize_and_normalize_256,
)


# ── Geometric primitives ─────────────────────────────────────────────────────

def center_crop_square(x: torch.Tensor) -> torch.Tensor:
    """Crop the spatial centre to the largest inscribed square."""
    _, h, w = x.shape
    crop = min(h, w)
    top = (h - crop) // 2
    left = (w - crop) // 2
    return x[:, top : top + crop, left : left + crop]


def rotate_90(x: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate by ``k × 90°`` in the spatial plane."""
    return torch.rot90(x, k, dims=(1, 2))


def random_rotate_90(x: torch.Tensor) -> torch.Tensor:
    """Randomly rotate by 0/90/180/270°."""
    k = int(torch.randint(0, 4, (1,)).item())
    return rotate_90(x, k)


def horizontal_flip(x: torch.Tensor) -> torch.Tensor:
    """Flip an image tensor along the width dimension."""
    return torch.flip(x, dims=(2,))


# ── I/O helper ────────────────────────────────────────────────────────────────

def save_tensor_image(x: torch.Tensor, out_base: str) -> None:
    """Save a tensor image as ``.npy`` (uint16) and ``.png`` (uint8 grayscale).

    Handles both raw uint16 tensors and [-1, 1] normalised tensors.
    """
    x = x.detach().cpu().float()

    if x.min() < 0:  # already in [-1, 1]
        x_01 = ((x + 1) / 2).clamp(0, 1)
    else:  # raw integer-like values
        raw_scale = 255.0 if float(x.max()) <= 255.0 else 65535.0
        x_01 = (x / raw_scale).clamp(0, 1)

    # .npy in uint16 domain
    x_uint16 = (x_01 * 65535.0).numpy().astype(np.uint16)
    if x_uint16.ndim == 3 and x_uint16.shape[0] == 1:
        x_uint16 = x_uint16[0]
    np.save(f"{out_base}.npy", x_uint16)

    # .png in uint8 for visualisation
    x_uint8 = (x_01 * 255).clamp(0, 255).byte()
    if x_uint8.shape[0] == 1:
        img = x_uint8[0].numpy()
    else:
        img = x_uint8.permute(1, 2, 0).numpy()
    try:
        from PIL import Image
        Image.fromarray(img).save(f"{out_base}.png")
    except Exception:
        pass


# ── Scheduled augmentation ────────────────────────────────────────────────────

class ScheduledAugment256:
    """Epoch-aware augmentation followed by resize-and-normalise.

    Probabilities follow a warmup → ramp → decay schedule.
    """

    def __init__(
        self,
        *,
        total_epochs: int,
        warmup_frac: float = 0.15,
        ramp_frac: float = 0.4,
        p_crop_warmup: float = 0.05,
        p_crop_max: float = 0.20,
        p_crop_final: float = 0.05,
        p_rot_warmup: float = 0.05,
        p_rot_max: float = 0.30,
        p_rot_final: float = 0.05,
        p_hflip_warmup: float = 0.0,
        p_hflip_max: float = 0.0,
        p_hflip_final: float = 0.0,
        image_size: int = DEFAULT_IMAGE_SIZE,
        normalization_mode: str = RAW_UINT16_PERCENTILE,
    ):
        self.total_epochs = int(total_epochs)
        self.warmup_frac = float(warmup_frac)
        self.ramp_frac = float(ramp_frac)
        self.image_size = int(image_size)
        self.normalization_mode = normalization_mode

        self.p_crop_warmup = float(p_crop_warmup)
        self.p_crop_max = float(p_crop_max)
        self.p_crop_final = float(p_crop_final)
        self.p_rot_warmup = float(p_rot_warmup)
        self.p_rot_max = float(p_rot_max)
        self.p_rot_final = float(p_rot_final)
        self.p_hflip_warmup = float(p_hflip_warmup)
        self.p_hflip_max = float(p_hflip_max)
        self.p_hflip_final = float(p_hflip_final)
        self.epoch = 0
        self._last_phase = None

        self._log_probs(phase="start")

    def _log_probs(self, *, phase: str) -> None:
        p_crop, p_rot, p_hflip = self._get_probs()
        print(
            f"[AugmentSchedule] {phase}: epoch={self.epoch} "
            f"p_crop={p_crop:.3f} p_rot={p_rot:.3f} p_hflip={p_hflip:.3f}"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        phase = self._get_phase()
        if phase != self._last_phase:
            self._log_probs(phase=phase)
            self._last_phase = phase

    def _get_phase(self) -> str:
        warmup_end = int(self.total_epochs * self.warmup_frac)
        ramp_end = int(self.total_epochs * (self.warmup_frac + self.ramp_frac))
        if self.epoch < warmup_end:
            return "warmup"
        if self.epoch < ramp_end:
            return "ramp"
        return "decay"

    def _get_probs(self) -> tuple[float, float, float]:
        warmup_end = int(self.total_epochs * self.warmup_frac)
        ramp_end = int(self.total_epochs * (self.warmup_frac + self.ramp_frac))

        if self.epoch < warmup_end:
            return self.p_crop_warmup, self.p_rot_warmup, self.p_hflip_warmup

        if self.epoch < ramp_end:
            denom = max(1, ramp_end - warmup_end)
            alpha = (self.epoch - warmup_end) / denom
            p_crop = self.p_crop_warmup + alpha * (self.p_crop_max - self.p_crop_warmup)
            p_rot = self.p_rot_warmup + alpha * (self.p_rot_max - self.p_rot_warmup)
            p_hflip = self.p_hflip_warmup + alpha * (self.p_hflip_max - self.p_hflip_warmup)
            return p_crop, p_rot, p_hflip

        denom = max(1, self.total_epochs - ramp_end)
        alpha = (self.epoch - ramp_end) / denom
        p_crop = self.p_crop_max + alpha * (self.p_crop_final - self.p_crop_max)
        p_rot = self.p_rot_max + alpha * (self.p_rot_final - self.p_rot_max)
        p_hflip = self.p_hflip_max + alpha * (self.p_hflip_final - self.p_hflip_max)
        return p_crop, p_rot, p_hflip

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p_crop, p_rot, p_hflip = self._get_probs()
        if torch.rand(()) < p_hflip:
            x = horizontal_flip(x)
        if torch.rand(()) < p_crop:
            x = center_crop_square(x)
        if torch.rand(()) < p_rot:
            x = random_rotate_90(x)
        return resize_and_normalize(
            x,
            image_size=self.image_size,
            normalization_mode=self.normalization_mode,
        )


def _ramped_value(
    *,
    epoch: int,
    total_epochs: int,
    warmup_frac: float,
    ramp_frac: float,
    v_warmup: float,
    v_max: float,
    v_final: float,
) -> float:
    """Shared warmup → ramp → decay interpolation used by aug schedules."""
    warmup_end = int(total_epochs * warmup_frac)
    ramp_end = int(total_epochs * (warmup_frac + ramp_frac))
    if epoch < warmup_end:
        return v_warmup
    if epoch < ramp_end:
        denom = max(1, ramp_end - warmup_end)
        alpha = (epoch - warmup_end) / denom
        return v_warmup + alpha * (v_max - v_warmup)
    denom = max(1, total_epochs - ramp_end)
    alpha = (epoch - ramp_end) / denom
    return v_max + alpha * (v_final - v_max)


def _phase_for_epoch(*, epoch: int, total_epochs: int, warmup_frac: float, ramp_frac: float) -> str:
    warmup_end = int(total_epochs * warmup_frac)
    ramp_end = int(total_epochs * (warmup_frac + ramp_frac))
    if epoch < warmup_end:
        return "warmup"
    if epoch < ramp_end:
        return "ramp"
    return "decay"


def _hflip_boxes(boxes_xyxy: torch.Tensor, *, size: float) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    out = boxes_xyxy.clone()
    out[:, 0] = size - boxes_xyxy[:, 2]
    out[:, 2] = size - boxes_xyxy[:, 0]
    return out


def _rotate_point_ccw(x: torch.Tensor, y: torch.Tensor, *, k: int, size: float):
    """Map a point under a ``k×90°`` CCW image rotation (matches ``torch.rot90``)."""
    k = k % 4
    if k == 0:
        return x, y
    if k == 1:      # 90° CCW: (x, y) -> (y, size - x)
        return y, size - x
    if k == 2:      # 180°
        return size - x, size - y
    return size - y, x  # 270° CCW == 90° CW


def _rotate_boxes_90(boxes_xyxy: torch.Tensor, *, k: int, size: float) -> torch.Tensor:
    """Rotate axis-aligned boxes by ``k×90°`` (exact; recomputes the AABB)."""
    if boxes_xyxy.numel() == 0 or k % 4 == 0:
        return boxes_xyxy
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    corners_x = torch.stack([x1, x2, x1, x2], dim=-1)
    corners_y = torch.stack([y1, y1, y2, y2], dim=-1)
    rx, ry = _rotate_point_ccw(corners_x, corners_y, k=k, size=size)
    new = torch.stack(
        [rx.min(dim=-1).values, ry.min(dim=-1).values, rx.max(dim=-1).values, ry.max(dim=-1).values],
        dim=-1,
    )
    return new


class LayoutScheduledAugment:
    """Epoch-aware, bbox-aware augmentation for layout FM training.

    Applies (each probability-gated, on a square ``size×size`` normalized image
    and matching ``boxes_xyxy``): horizontal flip, ``90°`` rotation, random-scale
    square crop, and image-only photometric jitter. Geometric ops transform the
    boxes jointly so layout annotations stay aligned. Crop drops boxes that fall
    (almost) out of view. Probabilities follow a warmup → ramp → decay schedule;
    ``set_epoch`` is driven automatically by ``set_epoch_for_dataloader``.
    """

    def __init__(
        self,
        *,
        total_epochs: int,
        warmup_frac: float = 0.05,
        ramp_frac: float = 0.15,
        p_hflip_warmup: float = 0.0,
        p_hflip_max: float = 0.5,
        p_hflip_final: float = 0.5,
        p_crop_warmup: float = 0.0,
        p_crop_max: float = 0.3,
        p_crop_final: float = 0.3,
        p_rot_warmup: float = 0.0,
        p_rot_max: float = 0.2,
        p_rot_final: float = 0.2,
        p_photometric_warmup: float = 0.0,
        p_photometric_max: float = 0.5,
        p_photometric_final: float = 0.5,
        photometric_brightness: float = 0.2,
        photometric_contrast: float = 0.2,
        photometric_gamma: float = 0.2,
        photometric_noise_std: float = 0.03,
        crop_min_scale: float = 0.8,
        crop_min_visibility: float = 0.3,
        rot_allow_90: bool = True,
    ) -> None:
        self.total_epochs = int(total_epochs)
        self.warmup_frac = float(warmup_frac)
        self.ramp_frac = float(ramp_frac)
        self.p_hflip = (float(p_hflip_warmup), float(p_hflip_max), float(p_hflip_final))
        self.p_crop = (float(p_crop_warmup), float(p_crop_max), float(p_crop_final))
        self.p_rot = (float(p_rot_warmup), float(p_rot_max), float(p_rot_final))
        self.p_photometric = (
            float(p_photometric_warmup),
            float(p_photometric_max),
            float(p_photometric_final),
        )
        self.photometric_brightness = float(photometric_brightness)
        self.photometric_contrast = float(photometric_contrast)
        self.photometric_gamma = float(photometric_gamma)
        self.photometric_noise_std = float(photometric_noise_std)
        self.crop_min_scale = float(crop_min_scale)
        self.crop_min_visibility = float(crop_min_visibility)
        self.rot_allow_90 = bool(rot_allow_90)
        self.epoch = 0
        self._last_phase = None
        self._log_probs(phase="start")

    # ── schedule ──────────────────────────────────────────────────────────
    def _prob(self, triple: tuple[float, float, float]) -> float:
        return _ramped_value(
            epoch=self.epoch,
            total_epochs=self.total_epochs,
            warmup_frac=self.warmup_frac,
            ramp_frac=self.ramp_frac,
            v_warmup=triple[0],
            v_max=triple[1],
            v_final=triple[2],
        )

    def current_probs(self) -> dict:
        return {
            "p_hflip": self._prob(self.p_hflip),
            "p_crop": self._prob(self.p_crop),
            "p_rot": self._prob(self.p_rot),
            "p_photometric": self._prob(self.p_photometric),
        }

    def _get_phase(self) -> str:
        return _phase_for_epoch(
            epoch=self.epoch,
            total_epochs=self.total_epochs,
            warmup_frac=self.warmup_frac,
            ramp_frac=self.ramp_frac,
        )

    def _log_probs(self, *, phase: str) -> None:
        p = self.current_probs()
        print(
            f"[LayoutAugment] {phase}: epoch={self.epoch} "
            f"p_hflip={p['p_hflip']:.3f} p_crop={p['p_crop']:.3f} "
            f"p_rot={p['p_rot']:.3f} p_photometric={p['p_photometric']:.3f}"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        phase = self._get_phase()
        if phase != self._last_phase:
            self._log_probs(phase=phase)
            self._last_phase = phase

    # ── geometric ops ─────────────────────────────────────────────────────
    def _apply_crop(self, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, *, size: int):
        scale = float(torch.empty(()).uniform_(self.crop_min_scale, 1.0).item())
        crop = max(1, int(round(size * scale)))
        if crop >= size:
            return image, boxes, labels
        top = int(torch.randint(0, size - crop + 1, (1,)).item())
        left = int(torch.randint(0, size - crop + 1, (1,)).item())
        cropped = image[:, top : top + crop, left : left + crop]
        resized = F.interpolate(
            cropped.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        )[0]
        if boxes.numel() == 0:
            return resized, boxes, labels
        scale_factor = size / float(crop)
        moved = boxes.clone()
        moved[:, [0, 2]] = (boxes[:, [0, 2]] - left) * scale_factor
        moved[:, [1, 3]] = (boxes[:, [1, 3]] - top) * scale_factor
        clipped = moved.clamp(0.0, float(size))
        orig_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        new_area = (clipped[:, 2] - clipped[:, 0]).clamp(min=0) * (clipped[:, 3] - clipped[:, 1]).clamp(min=0)
        visibility = new_area / orig_area.clamp(min=1e-6)
        keep = visibility >= self.crop_min_visibility
        kept_labels = labels[keep] if labels is not None and labels.numel() == boxes.shape[0] else labels
        return resized, clipped[keep], kept_labels

    def _apply_photometric(self, image: torch.Tensor) -> torch.Tensor:
        lo, hi = (-1.0, 1.0) if float(image.min()) < 0.0 else (0.0, 1.0)
        x = image
        if self.photometric_brightness > 0:
            b = float(torch.empty(()).uniform_(-self.photometric_brightness, self.photometric_brightness).item())
            x = x + b
        if self.photometric_contrast > 0:
            c = float(torch.empty(()).uniform_(1.0 - self.photometric_contrast, 1.0 + self.photometric_contrast).item())
            mean = x.mean()
            x = (x - mean) * c + mean
        if self.photometric_gamma > 0:
            g = float(torch.empty(()).uniform_(1.0 - self.photometric_gamma, 1.0 + self.photometric_gamma).item())
            x01 = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
            x01 = x01.pow(max(g, 1e-3))
            x = x01 * (hi - lo) + lo
        if self.photometric_noise_std > 0:
            x = x + torch.randn_like(x) * self.photometric_noise_std
        return x.clamp(lo, hi)

    def __call__(self, image: torch.Tensor, boxes_xyxy: torch.Tensor, labels: torch.Tensor = None):
        """Augment ``image`` + ``boxes_xyxy`` (+ optional ``labels``) jointly.

        Returns ``(image, boxes)`` when ``labels`` is ``None`` (back-compat), else
        ``(image, boxes, labels)`` with labels filtered to surviving boxes.
        """
        size = int(image.shape[-1])
        probs = self.current_probs()
        if torch.rand(()) < probs["p_hflip"]:
            image = horizontal_flip(image)
            boxes_xyxy = _hflip_boxes(boxes_xyxy, size=float(size))
        if self.rot_allow_90 and torch.rand(()) < probs["p_rot"]:
            k = int(torch.randint(1, 4, (1,)).item())
            image = rotate_90(image, k)
            boxes_xyxy = _rotate_boxes_90(boxes_xyxy, k=k, size=float(size))
        if torch.rand(()) < probs["p_crop"]:
            image, boxes_xyxy, labels = self._apply_crop(image, boxes_xyxy, labels, size=size)
        if torch.rand(()) < probs["p_photometric"]:
            image = self._apply_photometric(image)
        if labels is None:
            return image, boxes_xyxy
        return image, boxes_xyxy, labels


class ScheduledHorizontalFlip:
    """Epoch-aware horizontal flip schedule for bbox-aligned layout datasets."""

    def __init__(
        self,
        *,
        total_epochs: int,
        warmup_frac: float = 0.15,
        ramp_frac: float = 0.4,
        p_hflip_warmup: float = 0.0,
        p_hflip_max: float = 0.0,
        p_hflip_final: float = 0.0,
    ):
        self.total_epochs = int(total_epochs)
        self.warmup_frac = float(warmup_frac)
        self.ramp_frac = float(ramp_frac)
        self.p_hflip_warmup = float(p_hflip_warmup)
        self.p_hflip_max = float(p_hflip_max)
        self.p_hflip_final = float(p_hflip_final)
        self.epoch = 0
        self._last_phase = None
        self._log_prob(phase="start")

    def _log_prob(self, *, phase: str) -> None:
        print(
            f"[HorizontalFlipSchedule] {phase}: epoch={self.epoch} "
            f"p_hflip={self._get_prob():.3f}"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        phase = self._get_phase()
        if phase != self._last_phase:
            self._log_prob(phase=phase)
            self._last_phase = phase

    def _get_phase(self) -> str:
        warmup_end = int(self.total_epochs * self.warmup_frac)
        ramp_end = int(self.total_epochs * (self.warmup_frac + self.ramp_frac))
        if self.epoch < warmup_end:
            return "warmup"
        if self.epoch < ramp_end:
            return "ramp"
        return "decay"

    def _get_prob(self) -> float:
        warmup_end = int(self.total_epochs * self.warmup_frac)
        ramp_end = int(self.total_epochs * (self.warmup_frac + self.ramp_frac))

        if self.epoch < warmup_end:
            return self.p_hflip_warmup

        if self.epoch < ramp_end:
            denom = max(1, ramp_end - warmup_end)
            alpha = (self.epoch - warmup_end) / denom
            return self.p_hflip_warmup + alpha * (self.p_hflip_max - self.p_hflip_warmup)

        denom = max(1, self.total_epochs - ramp_end)
        alpha = (self.epoch - ramp_end) / denom
        return self.p_hflip_max + alpha * (self.p_hflip_final - self.p_hflip_max)

    def should_flip(self) -> bool:
        return bool(torch.rand(()) < self._get_prob())


# ── Debug / visualisation helper ──────────────────────────────────────────────

def save_transform_examples(dataset: Dataset, out_dir: str) -> None:
    """Save a few example transforms for quick visual sanity-checking."""
    os.makedirs(out_dir, exist_ok=True)

    x = dataset[0]
    if x.ndim == 2:
        x = x.unsqueeze(0)

    x_crop = center_crop_square(x)
    x_rot = rotate_90(x_crop, k=1)

    transform = getattr(dataset, "transform", None)
    if transform is not None and hasattr(transform, "set_epoch"):
        transform.set_epoch(0)
    x_final = transform(x) if transform is not None else resize_and_normalize_256(x)

    save_tensor_image(x_crop, os.path.join(out_dir, "example_center_crop"))
    save_tensor_image(x_rot, os.path.join(out_dir, "example_rotate_90"))
    save_tensor_image(x_final, os.path.join(out_dir, "example_final"))
