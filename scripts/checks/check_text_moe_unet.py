#!/usr/bin/env python3
"""Smoke-check: TextMOEUNet wrapper with fixed uniform adapters.

Checks:
  A. Build base text UNet and MOE-wrapped UNet
  B. Forward pass with dummy inputs (conditioning + timestep)
  C. Output shapes match
  D. No NaNs in outputs
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch

from src.models.fm_text_unet import load_text_unet_config, build_text_fm_unet
from src.models.moe_text_unet import build_text_moe_unet

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


print("=== A. Build models ===")
config_path = os.path.join(REPO, "configs/models/fm/text_unet_config.json")
unet_cfg = load_text_unet_config(config_path)

base_unet = build_text_fm_unet(unet_cfg, device="cpu")
check("Base text UNet built", base_unet is not None)

moe_unet = build_text_moe_unet(unet_cfg, device="cpu")
check("MOE text UNet built", moe_unet is not None)


print("\n=== B. Forward pass (dummy inputs) ===")
batch_size = 2
in_channels = unet_cfg.get("in_channels", 4)
sample_size = unet_cfg.get("sample_size", 64)
seq_len = 77
cross_dim = unet_cfg.get("cross_attention_dim", 768)

x = torch.randn(batch_size, in_channels, sample_size, sample_size)
t = torch.tensor([500.0, 250.0])
encoder_hidden_states = torch.randn(batch_size, seq_len, cross_dim)

with torch.no_grad():
    base_out = base_unet(x, t, encoder_hidden_states=encoder_hidden_states)
    moe_out = moe_unet(x, t, encoder_hidden_states=encoder_hidden_states)

base_sample = base_out.sample if hasattr(base_out, "sample") else base_out
moe_sample = moe_out.sample if hasattr(moe_out, "sample") else moe_out

check("Base output is tensor", isinstance(base_sample, torch.Tensor))
check("MOE output is tensor", isinstance(moe_sample, torch.Tensor))


print("\n=== C. Output shapes ===")
check(
    base_sample.shape == (batch_size, in_channels, sample_size, sample_size),
    f"Base output shape matches ({batch_size}, {in_channels}, {sample_size}, {sample_size})",
)
check(
    moe_sample.shape == (batch_size, in_channels, sample_size, sample_size),
    f"MOE output shape matches ({batch_size}, {in_channels}, {sample_size}, {sample_size})",
)


print("\n=== D. Output sanity ===")
check("Base output no NaNs", not torch.isnan(base_sample).any().item())
check("MOE output no NaNs", not torch.isnan(moe_sample).any().item())


print("\n=== Summary ===")
print(f"  {passed}/{passed + failed} checks passed")
if failed:
    sys.exit(1)
else:
    print("  All OK!")
