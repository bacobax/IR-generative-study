#!/usr/bin/env python3
"""Lightweight equivalence checks for the centralized normalization functions.

Compares:
  - src.core.normalization functions against inline reference implementations
  - Round-trip consistency (raw -> norm -> uint16 ≈ raw)
  - Edge cases (at percentile boundaries, out-of-range values)

Usage:
    python scripts/check_normalization_equivalence.py
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from src.core.constants import (
    P0001_PERCENTILE_RAW_IMAGES as A,
    P9999_PERCENTILE_RAW_IMAGES as B,
    RAW_RANGE as S,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.core.normalization import (
    raw_to_norm,
    norm_to_display,
    norm_to_uint16,
    raw_to_norm_numpy,
    per_image_minmax,
    fm_output_to_uint16,
    sd_output_to_uint16,
    uint16_to_png_uint8,
)

ATOL = 1e-5
PASS = 0
FAIL = 0


def check(name: str, condition: bool):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def ref_to_sd_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x.to(torch.float32) - A) / S, 0, 1) * 2 - 1


def ref_from_norm_to_display(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


def ref_from_norm_to_uint16(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1) / 2) * S + A


def ref_normalize_uint16_to_m1p1(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.float32) - A) / S * 2.0 - 1.0


def ref_per_image_minmax(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    B = x.shape[0]
    flat = x.view(B, -1)
    lo = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    hi = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    x = (x - lo) / (hi - lo + eps)
    return 2.0 * x - 1.0


print("=" * 60)
print("Normalization equivalence checks")
print("=" * 60)

print("\n1. raw_to_norm vs reference")
for val in [11667.0, 12000.0, 13000.0, 13944.0, 10000.0, 15000.0]:
    x = torch.tensor([val])
    check(f"raw_to_norm({val})", torch.allclose(raw_to_norm(x), ref_to_sd_tensor(x), atol=ATOL))

print("\n2. norm_to_display vs reference")
for val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    x = torch.tensor([val])
    check(f"norm_to_display({val})", torch.allclose(norm_to_display(x), ref_from_norm_to_display(x), atol=ATOL))

print("\n3. norm_to_uint16 vs reference")
for val in [-1.0, 0.0, 1.0]:
    x = torch.tensor([val])
    check(f"norm_to_uint16({val})", torch.allclose(norm_to_uint16(x), ref_from_norm_to_uint16(x), atol=ATOL))

print("\n4. Round-trip: raw -> norm -> uint16 ≈ raw")
for raw_val in [11667.0, 12000.0, 12500.0, 13000.0, 13944.0]:
    x = torch.tensor([raw_val])
    roundtrip = norm_to_uint16(raw_to_norm(x))
    check(f"roundtrip({raw_val})", torch.allclose(roundtrip, x, atol=0.01))

print("\n5. raw_to_norm_numpy vs reference")
for val in [11667, 12000, 13944]:
    arr = np.array([val], dtype=np.uint16)
    check(f"raw_to_norm_numpy({val})", np.allclose(raw_to_norm_numpy(arr), ref_normalize_uint16_to_m1p1(arr), atol=ATOL))

print("\n6. per_image_minmax vs reference")
torch.manual_seed(42)
x = torch.randn(4, 1, 8, 8)
check("per_image_minmax batch", torch.allclose(per_image_minmax(x), ref_per_image_minmax(x), atol=ATOL))

print("\n7. fm_output_to_uint16 consistency")
t = torch.tensor([[[0.0]]])
result = fm_output_to_uint16(t)
expected = np.rint(((0.0 + 1.0) / 2.0) * S + A).astype(np.uint16)
check("fm_output_to_uint16(0.0)", result.item() == expected.item())

t_neg1 = torch.tensor([[[-1.0]]])
result_neg1 = fm_output_to_uint16(t_neg1)
check("fm_output_to_uint16(-1.0)", result_neg1.item() == int(np.rint(A)))

t_pos1 = torch.tensor([[[1.0]]])
result_pos1 = fm_output_to_uint16(t_pos1)
check("fm_output_to_uint16(1.0)", result_pos1.item() == int(np.rint(B)))

print("\n8. uint16_to_png_uint8 smoke test")
arr = np.array([[11000, 12000, 13000, 14000]], dtype=np.uint16)
out = uint16_to_png_uint8(arr)
check("uint16_to_png_uint8 dtype", out.dtype == np.uint8)
check("uint16_to_png_uint8 range", out.min() >= 0 and out.max() <= 255)

print("\n9. Constants sanity")
check("A == 11667.0", A == 11667.0)
check("B == 13944.0", B == 13944.0)
check("S == 2277.0", S == 2277.0)
check("IMAGENET_MEAN", IMAGENET_MEAN == (0.485, 0.456, 0.406))
check("IMAGENET_STD", IMAGENET_STD == (0.229, 0.224, 0.225))

print("\n" + "=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed")
print("=" * 60)

sys.exit(1 if FAIL > 0 else 0)
