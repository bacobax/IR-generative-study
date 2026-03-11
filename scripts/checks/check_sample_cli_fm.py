#!/usr/bin/env python3
"""Smoke-check: verify the modular sampling CLI (src.cli.sample).

Checks:
  1. Module and key symbols import cleanly.
  2. Argument parser exposes expected flags with correct defaults.
  3. FMSampleConfig is built correctly from parsed args.
  4. Sampler is resolved through the registry (default and explicit).
  5. Dry-run: sampler instantiates and produces output of correct shape.
  6. run_sampling uses registry.
  7. Syntax check on all touched files.
"""

import ast
import inspect
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

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


# ======================================================================
# 1. Imports
# ======================================================================
print("\n=== 1. Module imports ===")
from src.cli.sample import build_parser, main, run_sampling
check("build_parser importable", callable(build_parser))
check("main importable", callable(main))
check("run_sampling importable", callable(run_sampling))

from src.core.configs.fm_config import FMSampleConfig
check("FMSampleConfig importable", FMSampleConfig is not None)

from src.core.registry import REGISTRIES
check("REGISTRIES importable", REGISTRIES is not None)

# ======================================================================
# 2. Argument parser flags and defaults
# ======================================================================
print("\n=== 2. Argument parser ===")
parser = build_parser()
defaults = vars(parser.parse_args([]))

expected = {
    "pipeline_dir": "./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled/",
    "vae_weights": None,
    "steps": 50,
    "batch_size": 8,
    "max_samples": 64,
    "t_scale": 1000.0,
    "train_target": "v",
    "device": None,
    "output_dir": "./artifacts/generated/main/fm_cli",
}
for key, val in expected.items():
    actual = defaults.get(key)
    check(f"--{key} default == {val!r}", actual == val)

# Custom overrides
custom = parser.parse_args(["--steps", "100", "--batch_size", "4",
                             "--max_samples", "16", "--train_target", "x0"])
check("--steps override", custom.steps == 100)
check("--batch_size override", custom.batch_size == 4)
check("--max_samples override", custom.max_samples == 16)
check("--train_target override", custom.train_target == "x0")

# ======================================================================
# 3. Config construction
# ======================================================================
print("\n=== 3. Config from args ===")
args = parser.parse_args(["--pipeline_dir", "/tmp/test", "--steps", "25",
                           "--device", "cpu"])
cfg = FMSampleConfig(
    pipeline_dir=args.pipeline_dir,
    vae_weights=args.vae_weights,
    t_scale=args.t_scale,
    train_target=args.train_target,
    steps=args.steps,
    batch_size=args.batch_size,
    device=args.device,
)
check("cfg.pipeline_dir == '/tmp/test'", cfg.pipeline_dir == "/tmp/test")
check("cfg.steps == 25", cfg.steps == 25)
check("cfg.device == 'cpu'", cfg.device == "cpu")
check("cfg.resolved_device() == 'cpu'", cfg.resolved_device() == "cpu")
check("cfg.sampler_name is None (default)", cfg.sampler_name is None)

# ======================================================================
# 4. Registry resolution
# ======================================================================
print("\n=== 4. Registry resolution ===")
from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler

SamplerCls = REGISTRIES.sampler.get(cfg.sampler_name)
check("Default sampler resolves to FlowMatchingSampler",
      SamplerCls is FlowMatchingSampler)

SamplerCls2 = REGISTRIES.sampler.get("default_fm")
check("Explicit 'default_fm' resolves to FlowMatchingSampler",
      SamplerCls2 is FlowMatchingSampler)

# ======================================================================
# 5. Dry-run: build sampler + sample on CPU
# ======================================================================
print("\n=== 5. Dry-run ===")
import torch
from diffusers import UNet2DModel

# Tiny UNet for testing
tiny_cfg = {
    "in_channels": 4, "out_channels": 4, "sample_size": 32,
    "block_out_channels": [32], "down_block_types": ["DownBlock2D"],
    "up_block_types": ["UpBlock2D"],
}
tiny_unet = UNet2DModel(**tiny_cfg).to("cpu").eval()

sampler = FlowMatchingSampler(tiny_unet, device="cpu", t_scale=1000.0,
                               train_target="v")
check("Sampler instantiated", isinstance(sampler, FlowMatchingSampler))

# sample_euler produces latents of correct shape
z = sampler.sample_euler(steps=2, batch_size=2)
check("sample_euler returns tensor", isinstance(z, torch.Tensor))
check("sample_euler shape B==2", z.shape[0] == 2)
check("sample_euler shape C==4 (latent)", z.shape[1] == 4)
check("sample_euler no NaN", not torch.isnan(z).any())

# ======================================================================
# 6. run_sampling uses registry
# ======================================================================
print("\n=== 6. run_sampling uses registry ===")
run_src = inspect.getsource(run_sampling)
check("run_sampling calls REGISTRIES.sampler.get", "REGISTRIES.sampler.get" in run_src)
check("run_sampling uses cfg.sampler_name", "cfg.sampler_name" in run_src)
check("run_sampling calls from_config", ".from_config(" in run_src)
check("run_sampling calls sample_euler", ".sample_euler(" in run_src)
check("run_sampling calls decode", ".decode(" in run_src)

# ======================================================================
# 7. Syntax check
# ======================================================================
print("\n=== 7. Syntax check ===")
files_to_check = [
    "src/cli/sample.py",
    "src/core/configs/fm_config.py",
    "src/core/registry.py",
    "src/algorithms/inference/flow_matching_sampler.py",
]
for rel in files_to_check:
    fpath = os.path.join(REPO, rel)
    try:
        with open(fpath) as f:
            ast.parse(f.read(), filename=rel)
        check(f"Syntax OK: {rel}", True)
    except SyntaxError as e:
        check(f"Syntax OK: {rel} ({e})", False)

# ======================================================================
# Summary
# ======================================================================
print(f"\n{'='*60}")
print(f"  {passed} passed, {failed} failed ({passed + failed} total)")
if failed:
    sys.exit(1)
