#!/usr/bin/env python3
"""Smoke-check: load each known FM UNet config and instantiate the model."""

import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch
from src.models.fm_unet import (
    load_unet_config,
    build_fm_unet_from_config,
    save_unet_config,
    STABLE_UNET_CONFIG,
    NON_STABLE_UNET_CONFIG,
)

passed = 0
failed = 0


def check(label, cond):
    global passed, failed
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        passed += 1
    else:
        failed += 1


device = "cpu"

# ── 1. load_unet_config from JSON path ──────────────────────────────────────
print("=== 1. load_unet_config ===")

cfg_stable = load_unet_config(os.path.join(REPO, STABLE_UNET_CONFIG))
check("Stable config loaded", isinstance(cfg_stable, dict))
check("Stable in_channels == 4", cfg_stable.get("in_channels") == 4)
check("Stable sample_size == 64", cfg_stable.get("sample_size") == 64)

cfg_pixel = load_unet_config(os.path.join(REPO, NON_STABLE_UNET_CONFIG))
check("Non-stable config loaded", isinstance(cfg_pixel, dict))
check("Non-stable in_channels == 1", cfg_pixel.get("in_channels") == 1)
check("Non-stable sample_size == 256", cfg_pixel.get("sample_size") == 256)

# load from dict
cfg_from_dict = load_unet_config(config_dict=cfg_stable)
check("load from dict", cfg_from_dict == cfg_stable)


# ── 2. build_fm_unet_from_config ─────────────────────────────────────────────
print("\n=== 2. Build UNet (stable config) ===")
unet_stable = build_fm_unet_from_config(cfg_stable, device=device)
check("Instantiated UNet2DModel", unet_stable is not None)
check("Type is UNet2DModel", type(unet_stable).__name__ == "UNet2DModel")

# Quick forward pass
x = torch.randn(1, 4, 64, 64)
t = torch.tensor([500.0])
with torch.no_grad():
    out = unet_stable(x, t)
y = out.sample if hasattr(out, "sample") else out
check("Forward pass output shape", y.shape == (1, 4, 64, 64))

print("\n=== 3. Build UNet (non-stable / pixel config) ===")
unet_pixel = build_fm_unet_from_config(cfg_pixel, device=device)
check("Instantiated pixel UNet", unet_pixel is not None)

x2 = torch.randn(1, 1, 256, 256)
t2 = torch.tensor([500.0])
with torch.no_grad():
    out2 = unet_pixel(x2, t2)
y2 = out2.sample if hasattr(out2, "sample") else out2
check("Pixel forward pass output shape", y2.shape == (1, 1, 256, 256))


# ── 4. save_unet_config round-trip ──────────────────────────────────────────
print("\n=== 4. save_unet_config round-trip ===")
import tempfile, json
with tempfile.TemporaryDirectory() as tmpdir:
    p = os.path.join(tmpdir, "test_unet_config.json")
    save_unet_config(cfg_stable, p)
    check("Config file created", os.path.isfile(p))
    reloaded = load_unet_config(p)
    check("Round-trip matches", reloaded == cfg_stable)


# ── 5. Pipeline still works with updated code ───────────────────────────────
print("\n=== 5. Pipeline import & build ===")
import sys as _sys
from src.core.paths import legacy_code_root
_sys.path.insert(0, str(legacy_code_root()))
from fm_src.pipelines.flow_matching_pipeline import FlowMatchingPipeline, StableFlowMatchingPipeline

with tempfile.TemporaryDirectory() as tmpdir:
    pipe = FlowMatchingPipeline(device=device, model_dir=tmpdir)
    pipe.build_from_configs(unet_json=os.path.join(REPO, NON_STABLE_UNET_CONFIG))
    check("FlowMatchingPipeline.build_from_configs OK", hasattr(pipe, "unet"))
    check("Pipeline unet_config saved", os.path.isfile(os.path.join(tmpdir, "UNET", "config.json")))

with tempfile.TemporaryDirectory() as tmpdir:
    spipe = StableFlowMatchingPipeline(device=device, model_dir=tmpdir)
    spipe.build_from_configs(
        unet_json=os.path.join(REPO, STABLE_UNET_CONFIG),
        vae_json=os.path.join(REPO, "configs/models/fm/vae_config.json"),
    )
    check("StableFlowMatchingPipeline has unet", hasattr(spipe, "unet"))
    check("StableFlowMatchingPipeline has vae", hasattr(spipe, "vae"))


# ── 6. Syntax check pipeline file ───────────────────────────────────────────
print("\n=== 6. Syntax checks ===")
for f in [
    "archive/legacy_code/fm_src/pipelines/flow_matching_pipeline.py",
    "src/models/fm_unet.py",
    "train_sfm.py",
    "scripts/standalone/train_fm.py",
    "train_vae.py",
]:
    path = os.path.join(REPO, f)
    with open(path) as fh:
        try:
            ast.parse(fh.read(), filename=f)
            check(f"{f} parses OK", True)
        except SyntaxError as e:
            check(f"{f} syntax error: {e}", False)


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  {passed}/{passed + failed} checks passed")
if failed:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("  All OK!")
