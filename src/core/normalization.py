"""Centralised normalization and denormalization helpers.

Every function here preserves the *exact* numerical behaviour of the original
per-script implementations.  Two distinct normalization families exist in this
repo and are kept separate:

1. **Percentile-based** (``raw_to_norm``, ``norm_to_display``, ``norm_to_uint16``)
   Maps raw uint16 sensor values through the [A, B] percentile window to [-1, 1].

2. **Per-image min/max** (``per_image_minmax``)
   Maps each image individually to [-1, 1] using its own min/max.

Both families are used in different parts of the pipeline and must NOT be
confused or merged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.core.constants import (
    P0001_PERCENTILE_RAW_IMAGES as _A,
    P9999_PERCENTILE_RAW_IMAGES as _B,
    RAW_RANGE as _S,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_IMAGE_SIZE,
)

RAW_UINT16_PERCENTILE = "raw_uint16_percentile"
UINT8_LINEAR = "uint8_linear"
SENTINEL2_REFLECTANCE = "sentinel2_reflectance"


# ── Percentile-based normalization ────────────────────────────────────────────


def raw_to_norm(x: torch.Tensor) -> torch.Tensor:
    """Raw uint16 sensor tensor → [-1, 1] using the global percentile window.

    Formula::

        clamp((x - A) / S, 0, 1) * 2 - 1

    where A = 11667, S = 2277.
    """
    return torch.clamp((x.to(torch.float32) - _A) / _S, 0, 1) * 2 - 1


def norm_to_display(x: torch.Tensor) -> torch.Tensor:
    """Normalized [-1, 1] tensor → [0, 1] for TensorBoard / display.

    Formula::

        (x + 1) / 2
    """
    return (x + 1) / 2


def norm_to_uint16(x: torch.Tensor) -> torch.Tensor:
    """Normalized [-1, 1] tensor → raw uint16 scale (float).

    Formula::

        ((x + 1) / 2) * S + A
    """
    return ((x + 1) / 2) * _S + _A


def raw_to_norm_numpy(arr: np.ndarray) -> np.ndarray:
    """Numpy variant of :func:`raw_to_norm`.

    Maps uint16 values through the percentile window to [-1, 1].

    Formula::

        (arr - A) / S * 2 - 1

    .. note:: Unlike the torch version this does **not** clamp; caller is
       responsible for any clipping, matching the original
       ``_normalize_uint16_to_m1p1`` in *analyze_distribution_shift.py*.
    """
    return (arr.astype(np.float32) - _A) / _S * 2.0 - 1.0


def uint8_to_norm(x: torch.Tensor) -> torch.Tensor:
    """Linearly map uint8-like values from [0, 255] to [-1, 1]."""
    return (x.to(torch.float32) / 255.0).clamp(0.0, 1.0) * 2.0 - 1.0


def sentinel2_reflectance_to_norm(x: torch.Tensor) -> torch.Tensor:
    """Map Sentinel-2 reflectance-scaled integer values to [-1, 1]."""
    return (x.to(torch.float32) / 10000.0).clamp(0.0, 1.0) * 2.0 - 1.0


# ── Per-image min/max normalization ───────────────────────────────────────────


def per_image_minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-image min/max normalization → [-1, 1].

    Parameters
    ----------
    x : torch.Tensor
        Batched tensor of shape ``(B, C, H, W)``.
    eps : float
        Epsilon to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Same shape, values in [-1, 1].
    """
    B = x.shape[0]
    flat = x.view(B, -1)
    lo = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    hi = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    x = (x - lo) / (hi - lo + eps)
    return 2.0 * x - 1.0


# ── Resize-and-normalize composites ──────────────────────────────────────────


def normalize_image_tensor(
    x: torch.Tensor,
    *,
    normalization_mode: str = RAW_UINT16_PERCENTILE,
) -> torch.Tensor:
    """Normalize one image tensor into [-1, 1] with the requested mode."""
    if normalization_mode == RAW_UINT16_PERCENTILE:
        return raw_to_norm(x)
    if normalization_mode == UINT8_LINEAR:
        return uint8_to_norm(x)
    if normalization_mode == SENTINEL2_REFLECTANCE:
        return sentinel2_reflectance_to_norm(x)
    raise ValueError(
        f"Unknown normalization_mode={normalization_mode!r}. "
        f"Expected one of: {RAW_UINT16_PERCENTILE!r}, "
        f"{UINT8_LINEAR!r}, {SENTINEL2_REFLECTANCE!r}"
    )


def resize_and_normalize(
    x: torch.Tensor,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    normalization_mode: str = RAW_UINT16_PERCENTILE,
) -> torch.Tensor:
    """Resize a single image to a square size then normalize it.

    Parameters
    ----------
    x : torch.Tensor
        Single image tensor ``(C, H, W)`` with raw image values.
    image_size : int
        Target square resolution.
    normalization_mode : str
        Which normalization family to apply after resizing.

    Returns
    -------
    torch.Tensor
        ``(C, image_size, image_size)`` in [-1, 1].
    """
    x = F.interpolate(
        x.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return normalize_image_tensor(x, normalization_mode=normalization_mode)


def resize_and_normalize_256(x: torch.Tensor) -> torch.Tensor:
    """Backward-compatible 256x256 percentile-based normalization wrapper."""
    return resize_and_normalize(
        x,
        image_size=DEFAULT_IMAGE_SIZE,
        normalization_mode=RAW_UINT16_PERCENTILE,
    )


# ── Output conversion helpers ────────────────────────────────────────────────


def fm_output_to_uint16(tensor: torch.Tensor) -> np.ndarray:
    """Flow-matching output tensor (1, H, W) in [-1, 1] → uint16 ndarray.

    Applies the inverse percentile mapping with rounding to nearest int.

    Parameters
    ----------
    tensor : torch.Tensor
        Single-image tensor, typically ``(1, H, W)`` in [-1, 1].

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint16 array.
    """
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    uint16_val = ((np.clip(arr, -1.0, 1.0) + 1.0) / 2.0) * _S + _A
    return np.rint(uint16_val).astype(np.uint16)


def sd_output_to_uint16(image_pil) -> np.ndarray:
    """SD 1.5 PIL output → 1-channel uint16 ndarray.

    Reverses the percentile normalization used during SD training:
    ``uint16 = A + grey_01 * S`` where ``grey_01`` is the [0, 1] grayscale.

    Parameters
    ----------
    image_pil : PIL.Image.Image
        Output image from the Stable Diffusion pipeline.

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint16 array.
    """
    raw = np.asarray(image_pil).astype(np.float32) / 255.0
    if raw.ndim == 3:
        raw = raw.mean(axis=-1)
    uint16_val = _A + np.clip(raw, 0.0, 1.0) * _S
    return np.clip(uint16_val, 0, 65535).astype(np.uint16)


def sd_output_to_npy(
    image_pil,
    *,
    normalization_mode: str = RAW_UINT16_PERCENTILE,
) -> np.ndarray:
    """SD output PIL image -> raw-domain numpy array for the requested dataset mode."""
    raw = np.asarray(image_pil).astype(np.float32) / 255.0
    if raw.ndim == 3:
        raw = raw.mean(axis=-1)

    if normalization_mode == RAW_UINT16_PERCENTILE:
        uint16_val = _A + np.clip(raw, 0.0, 1.0) * _S
        return np.clip(uint16_val, 0, 65535).astype(np.uint16)

    if normalization_mode == UINT8_LINEAR:
        return np.clip(np.rint(raw * 255.0), 0, 255).astype(np.uint8)

    if normalization_mode == SENTINEL2_REFLECTANCE:
        return np.clip(np.rint(raw * 10000.0), 0, 10000).astype(np.uint16)

    raise ValueError(
        f"Unknown normalization_mode={normalization_mode!r}. "
        f"Expected one of: {RAW_UINT16_PERCENTILE!r}, "
        f"{UINT8_LINEAR!r}, {SENTINEL2_REFLECTANCE!r}"
    )


def uint16_to_png_uint8(arr_uint16: np.ndarray) -> np.ndarray:
    """uint16 image → uint8 for PNG visualization (per-image p1–p99 stretch).

    Parameters
    ----------
    arr_uint16 : np.ndarray
        ``(H, W)`` uint16 array.

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8 array.
    """
    arr = arr_uint16.astype(np.float32)
    p1 = float(np.percentile(arr, 1.0))
    p99 = float(np.percentile(arr, 99.0))
    if p99 <= p1:
        return np.zeros_like(arr_uint16, dtype=np.uint8)
    norm = (arr - p1) / (p99 - p1)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def raw_array_to_png_uint8(
    arr: np.ndarray,
    *,
    normalization_mode: str = RAW_UINT16_PERCENTILE,
) -> np.ndarray:
    """Raw-domain array -> uint8 PNG visualization for the requested dataset mode."""
    if normalization_mode == RAW_UINT16_PERCENTILE:
        return uint16_to_png_uint8(arr.astype(np.uint16, copy=False))

    if normalization_mode == UINT8_LINEAR:
        return np.clip(arr, 0, 255).astype(np.uint8, copy=False)

    if normalization_mode == SENTINEL2_REFLECTANCE:
        x01 = np.clip(arr.astype(np.float32, copy=False) / 10000.0, 0.0, 1.0)
        return np.rint(x01 * 255.0).astype(np.uint8)

    raise ValueError(
        f"Unknown normalization_mode={normalization_mode!r}. "
        f"Expected one of: {RAW_UINT16_PERCENTILE!r}, "
        f"{UINT8_LINEAR!r}, {SENTINEL2_REFLECTANCE!r}"
    )
