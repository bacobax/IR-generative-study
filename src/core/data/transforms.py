"""Reusable image transforms and augmentation schedules."""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.core.normalization import resize_and_normalize_256


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


# ── I/O helper ────────────────────────────────────────────────────────────────

def save_tensor_image(x: torch.Tensor, out_base: str) -> None:
    """Save a tensor image as ``.npy`` (uint16) and ``.png`` (uint8 grayscale).

    Handles both raw uint16 tensors and [-1, 1] normalised tensors.
    """
    x = x.detach().cpu().float()

    if x.min() < 0:  # already in [-1, 1]
        x_01 = ((x + 1) / 2).clamp(0, 1)
    else:  # raw uint16 values
        x_01 = (x / 65535.0).clamp(0, 1)

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
    """Epoch-aware augmentation: optional centre-crop + random 90° rotation,
    followed by resize-and-normalise to 256×256.

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
    ):
        self.total_epochs = int(total_epochs)
        self.warmup_frac = float(warmup_frac)
        self.ramp_frac = float(ramp_frac)

        self.p_crop_warmup = float(p_crop_warmup)
        self.p_crop_max = float(p_crop_max)
        self.p_crop_final = float(p_crop_final)
        self.p_rot_warmup = float(p_rot_warmup)
        self.p_rot_max = float(p_rot_max)
        self.p_rot_final = float(p_rot_final)
        self.epoch = 0
        self._last_phase = None

        self._log_probs(phase="start")

    def _log_probs(self, *, phase: str) -> None:
        p_crop, p_rot = self._get_probs()
        print(
            f"[AugmentSchedule] {phase}: epoch={self.epoch} "
            f"p_crop={p_crop:.3f} p_rot={p_rot:.3f}"
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

    def _get_probs(self) -> tuple[float, float]:
        warmup_end = int(self.total_epochs * self.warmup_frac)
        ramp_end = int(self.total_epochs * (self.warmup_frac + self.ramp_frac))

        if self.epoch < warmup_end:
            return self.p_crop_warmup, self.p_rot_warmup

        if self.epoch < ramp_end:
            denom = max(1, ramp_end - warmup_end)
            alpha = (self.epoch - warmup_end) / denom
            p_crop = self.p_crop_warmup + alpha * (self.p_crop_max - self.p_crop_warmup)
            p_rot = self.p_rot_warmup + alpha * (self.p_rot_max - self.p_rot_warmup)
            return p_crop, p_rot

        denom = max(1, self.total_epochs - ramp_end)
        alpha = (self.epoch - ramp_end) / denom
        p_crop = self.p_crop_max + alpha * (self.p_crop_final - self.p_crop_max)
        p_rot = self.p_rot_max + alpha * (self.p_rot_final - self.p_rot_max)
        return p_crop, p_rot

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p_crop, p_rot = self._get_probs()
        if torch.rand(()) < p_crop:
            x = center_crop_square(x)
        if torch.rand(()) < p_rot:
            x = random_rotate_90(x)
        return resize_and_normalize_256(x)


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
