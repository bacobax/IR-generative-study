#!/usr/bin/env python3
"""Smoke-check: verify the FM-component registry works correctly.

Checks:
  1. Registry module imports cleanly.
  2. After importing component modules, default entries are present.
  3. Lookup by name resolves to the correct class/function.
  4. Default lookup (name=None) resolves correctly.
  5. Missing-key errors are clear and list available keys.
  6. Config objects carry optional component-name fields.
  7. Duplicate-registration is rejected.
  8. Instantiation via registry produces correct types.
"""

import ast
import os
import sys

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
# 1. Registry module imports
# ======================================================================
print("\n=== 1. Registry imports ===")
from src.core.registry import Registry, REGISTRIES, _Registries
check("Registry class importable", Registry is not None)
check("REGISTRIES singleton importable", isinstance(REGISTRIES, _Registries))
check("Five named registries exist",
      all(hasattr(REGISTRIES, a) for a in
          ("model_builder", "trainer", "sampler", "guidance", "conditioning")))

# ======================================================================
# 2. Registry API basics (on a fresh Registry)
# ======================================================================
print("\n=== 2. Registry API basics ===")
r = Registry("test")
check("list() starts empty", r.list() == [])
check("'foo' not in empty registry", "foo" not in r)

@r.register("alpha", default=True)
def _alpha():
    return "alpha"

@r.register("beta")
def _beta():
    return "beta"

check("alpha registered", "alpha" in r)
check("beta registered", "beta" in r)
check("list() returns sorted keys", r.list() == ["alpha", "beta"])
check("__getitem__ alpha", r["alpha"] is _alpha)
check("__getitem__ beta", r["beta"] is _beta)
check("get(None) returns default", r.get() is _alpha)
check("get('beta') returns beta", r.get("beta") is _beta)

# Missing key → clear error
try:
    r["nonexistent"]
    check("Missing key raises KeyError", False)
except KeyError as e:
    msg = str(e)
    check("Missing key raises KeyError", True)
    check("Error message mentions registry name", "test" in msg)
    check("Error message lists available keys", "alpha" in msg and "beta" in msg)

# No default → clear error
r2 = Registry("no_default")
r2.register("only")( lambda: None)
try:
    r2.get()
    check("No default raises KeyError", False)
except KeyError as e:
    check("No default raises KeyError", True)
    check("Error mentions 'no default'", "no default" in str(e).lower() or "No name given" in str(e))

# Duplicate registration rejected
try:
    r.register("alpha")(lambda: "dup")
    check("Duplicate registration raises ValueError", False)
except ValueError as e:
    check("Duplicate registration raises ValueError", True)
    check("Dup error mentions key name", "alpha" in str(e))

check("repr works", "test" in repr(r))

# ======================================================================
# 3. Default FM components are registered
# ======================================================================
print("\n=== 3. Default FM components ===")

# Import modules to trigger registration
from src.models.fm_unet import build_fm_unet_from_config
from src.algorithms.training.flow_matching_trainer import FlowMatchingTrainer
from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler

check("model_builder: 'default_unet' registered",
      "default_unet" in REGISTRIES.model_builder)
check("trainer: 'default_fm' registered",
      "default_fm" in REGISTRIES.trainer)
check("sampler: 'default_fm' registered",
      "default_fm" in REGISTRIES.sampler)

check("model_builder default resolves to build_fm_unet_from_config",
      REGISTRIES.model_builder.get() is build_fm_unet_from_config)
check("trainer default resolves to FlowMatchingTrainer",
      REGISTRIES.trainer.get() is FlowMatchingTrainer)
check("sampler default resolves to FlowMatchingSampler",
      REGISTRIES.sampler.get() is FlowMatchingSampler)

check("model_builder['default_unet'] is build_fm_unet_from_config",
      REGISTRIES.model_builder["default_unet"] is build_fm_unet_from_config)
check("trainer['default_fm'] is FlowMatchingTrainer",
      REGISTRIES.trainer["default_fm"] is FlowMatchingTrainer)
check("sampler['default_fm'] is FlowMatchingSampler",
      REGISTRIES.sampler["default_fm"] is FlowMatchingSampler)

# ======================================================================
# 4. Guidance and conditioning registries are empty but available
# ======================================================================
print("\n=== 4. Guidance/conditioning registries ===")
check("guidance registry exists and is empty", REGISTRIES.guidance.list() == [])
check("conditioning registry exists and is empty", REGISTRIES.conditioning.list() == [])

# ======================================================================
# 5. Config objects have component-name fields
# ======================================================================
print("\n=== 5. Config component-name fields ===")
from src.core.configs.fm_config import (
    FMTrainConfig, FMSampleConfig, ModelConfig,
)

mc = ModelConfig()
check("ModelConfig.model_builder_name defaults to None", mc.model_builder_name is None)

tc = FMTrainConfig()
check("FMTrainConfig.trainer_name defaults to None", tc.trainer_name is None)
check("FMTrainConfig.sampler_name defaults to None", tc.sampler_name is None)

sc = FMSampleConfig()
check("FMSampleConfig.sampler_name defaults to None", sc.sampler_name is None)
check("FMSampleConfig.model_builder_name defaults to None", sc.model_builder_name is None)

# ======================================================================
# 6. summary() output
# ======================================================================
print("\n=== 6. REGISTRIES.summary() ===")
summary = REGISTRIES.summary()
check("summary contains 'model_builder'", "model_builder" in summary)
check("summary contains 'trainer'", "trainer" in summary)
check("summary contains 'sampler'", "sampler" in summary)
check("summary contains 'guidance'", "guidance" in summary)
check("summary contains 'conditioning'", "conditioning" in summary)
print(f"\n{summary}\n")

# ======================================================================
# 7. Build default UNet through registry
# ======================================================================
print("\n=== 7. Instantiate through registry ===")
import torch

builder = REGISTRIES.model_builder.get()
unet = builder({"in_channels": 4, "out_channels": 4, "sample_size": 32,
                 "block_out_channels": [32], "down_block_types": ["DownBlock2D"],
                 "up_block_types": ["UpBlock2D"]}, device="cpu")
check("UNet built via registry has correct type",
      type(unet).__name__ == "UNet2DModel")

TrainerCls = REGISTRIES.trainer.get()
check("Trainer from registry is FlowMatchingTrainer",
      TrainerCls is FlowMatchingTrainer)

SamplerCls = REGISTRIES.sampler.get()
check("Sampler from registry is FlowMatchingSampler",
      SamplerCls is FlowMatchingSampler)

# Create sampler instance through registry
sampler = SamplerCls(unet, device="cpu", t_scale=1000.0, train_target="v")
check("Sampler instance created through registry",
      isinstance(sampler, FlowMatchingSampler))

# ======================================================================
# 8. Syntax check of all touched files
# ======================================================================
print("\n=== 8. Syntax check ===")
files_to_check = [
    "src/core/registry.py",
    "src/core/configs/fm_config.py",
    "src/models/fm_unet.py",
    "src/algorithms/training/flow_matching_trainer.py",
    "src/algorithms/inference/flow_matching_sampler.py",
]
for rel in files_to_check:
    full = os.path.join(REPO, rel)
    try:
        with open(full) as f:
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
