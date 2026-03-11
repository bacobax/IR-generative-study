#!/usr/bin/env python3
"""Quick sanity-check that the extracted dataset/transform modules are
importable and behaviorally equivalent to the originals."""

import ast
import importlib
import os
import sys
import tempfile

import numpy as np
import torch

# ── ensure repo root is on the path ─────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

passed = 0
failed = 0


def check(label: str, cond: bool) -> None:
    global passed, failed
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        passed += 1
    else:
        failed += 1


# ── 1. Imports ───────────────────────────────────────────────────────────────
print("=== 1. Module imports ===")
from src.core.data.datasets import NPYImageDataset, NPYStemDataset
check("NPYImageDataset importable", True)
check("NPYStemDataset importable", True)

from src.core.data.transforms import (
    center_crop_square,
    rotate_90,
    random_rotate_90,
    save_tensor_image,
    ScheduledAugment256,
    save_transform_examples,
)
check("All transform functions importable", True)


# ── 2. Geometric transforms ─────────────────────────────────────────────────
print("\n=== 2. Geometric transforms ===")
x = torch.randn(1, 480, 640)

cropped = center_crop_square(x)
check("center_crop_square → square", cropped.shape[1] == cropped.shape[2] == 480)

rotated = rotate_90(x, k=1)
check("rotate_90 k=1 shape", rotated.shape == (1, 640, 480))

rr = random_rotate_90(x)
check("random_rotate_90 returns tensor", isinstance(rr, torch.Tensor))


# ── 3. ScheduledAugment256 ──────────────────────────────────────────────────
print("\n=== 3. ScheduledAugment256 ===")
aug = ScheduledAugment256(total_epochs=100)
check("Instantiation", aug is not None)

aug.set_epoch(0)
check("set_epoch(0) runs", True)

# Apply to a raw-valued tensor (simulating uint16 .npy values)
raw = torch.full((1, 300, 400), 12500.0)
out = aug(raw)
check("Output shape 256×256", out.shape == (1, 256, 256))
check("Output in [-1, 1]", out.min() >= -1.0 and out.max() <= 1.0)


# ── 4. NPYImageDataset ──────────────────────────────────────────────────────
print("\n=== 4. NPYImageDataset ===")
with tempfile.TemporaryDirectory() as tmpdir:
    for i in range(3):
        np.save(os.path.join(tmpdir, f"img_{i:04d}.npy"),
                np.random.randint(10000, 14000, size=(300, 400), dtype=np.uint16).astype(np.float32))

    ds = NPYImageDataset(tmpdir)
    check("len == 3", len(ds) == 3)
    sample = ds[0]
    check("sample ndim == 3", sample.ndim == 3)
    check("sample shape (1, 300, 400)", sample.shape == (1, 300, 400))

    ds_t = NPYImageDataset(tmpdir, transform=aug)
    sample_t = ds_t[0]
    check("With transform → (1,256,256)", sample_t.shape == (1, 256, 256))


# ── 5. NPYStemDataset ───────────────────────────────────────────────────────
print("\n=== 5. NPYStemDataset ===")
with tempfile.TemporaryDirectory() as tmpdir:
    for i in range(5):
        np.save(os.path.join(tmpdir, f"sample_{i:04d}.npy"),
                np.random.rand(300, 400).astype(np.float32) * 14000)

    ds2 = NPYStemDataset(tmpdir)
    check("len == 5", len(ds2) == 5)
    tensor, stem = ds2[0]
    check("Returns (tensor, stem)", isinstance(stem, str))
    check("stem is filename w/o ext", stem.startswith("sample_"))

    ds3 = NPYStemDataset(tmpdir, stem_list=["sample_0001", "sample_0003"])
    check("stem_list filters to 2", len(ds3) == 2)

    ds4 = NPYStemDataset(tmpdir, max_items=2)
    check("max_items caps to 2", len(ds4) == 2)


# ── 6. save_tensor_image ────────────────────────────────────────────────────
print("\n=== 6. save_tensor_image ===")
with tempfile.TemporaryDirectory() as tmpdir:
    t = torch.randn(1, 64, 64)  # normalised [-1,1]
    base = os.path.join(tmpdir, "test_out")
    save_tensor_image(t, base)
    check(".npy written", os.path.exists(f"{base}.npy"))
    check(".png written", os.path.exists(f"{base}.png"))


# ── 7. Syntax check updated scripts ─────────────────────────────────────────
print("\n=== 7. Syntax check scripts ===")
for script in ["train_sfm.py", "scripts/standalone/train_fm.py", "train_vae.py", "scripts/standalone/build_surprise_pred_dataset.py"]:
    path = os.path.join(REPO, script)
    with open(path) as f:
        try:
            ast.parse(f.read(), filename=script)
            check(f"{script} parses OK", True)
        except SyntaxError as e:
            check(f"{script} parses OK — {e}", False)


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  {passed}/{passed + failed} checks passed")
if failed:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("  All OK!")
