#!/usr/bin/env python
"""Phase 16 checks – Conditioning abstraction integration.

Tests:
  A. Module structure
  B. BaseConditioner ABC
  C. NoConditioner implementation
  D. NoConditioner registry registration
  E. FlowMatchingTrainer conditioning support
  F. FlowMatchingSampler conditioning support
  G. Runtime conditioner injection
  H. Source inspection (trainer + sampler UNet calls)
"""

import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

ok = fail = 0


def check(cond: bool, msg: str):
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {msg}")
    else:
        fail += 1
        print(f"  [FAIL] {msg}")


# ══════════════════════════════════════════════════════════════════════════
# A. Module structure
# ══════════════════════════════════════════════════════════════════════════
print("\n=== A. Module structure ===")
cond_pkg = ROOT / "src" / "conditioning"
check(cond_pkg.is_dir(), "src/conditioning/ exists")
check((cond_pkg / "__init__.py").is_file(), "src/conditioning/__init__.py exists")
check((cond_pkg / "base_conditioner.py").is_file(), "src/conditioning/base_conditioner.py exists")
check((cond_pkg / "no_conditioner.py").is_file(), "src/conditioning/no_conditioner.py exists")

# ══════════════════════════════════════════════════════════════════════════
# B. BaseConditioner ABC
# ══════════════════════════════════════════════════════════════════════════
print("\n=== B. BaseConditioner ABC ===")
from src.conditioning.base_conditioner import BaseConditioner
import abc

check(inspect.isabstract(BaseConditioner), "BaseConditioner is abstract")
check(issubclass(BaseConditioner, abc.ABC), "BaseConditioner extends ABC")

abstract_methods = set(BaseConditioner.__abstractmethods__)
check("prepare_for_training" in abstract_methods, "prepare_for_training is abstract")
check("prepare_for_sampling" in abstract_methods, "prepare_for_sampling is abstract")
check(len(abstract_methods) == 2, f"exactly 2 abstract methods (got {len(abstract_methods)})")

# Check signatures
sig_train = inspect.signature(BaseConditioner.prepare_for_training)
check("batch" in sig_train.parameters, "prepare_for_training has batch param")
check("device" in sig_train.parameters, "prepare_for_training has device param")

sig_sample = inspect.signature(BaseConditioner.prepare_for_sampling)
check("batch_size" in sig_sample.parameters, "prepare_for_sampling has batch_size param")
check("device" in sig_sample.parameters, "prepare_for_sampling has device param")

# Return type annotations (check source for Dict)
bc_src = (cond_pkg / "base_conditioner.py").read_text()
check("Dict[str, Any]" in bc_src, "Return type uses Dict[str, Any]")


# ══════════════════════════════════════════════════════════════════════════
# C. NoConditioner implementation
# ══════════════════════════════════════════════════════════════════════════
print("\n=== C. NoConditioner implementation ===")
from src.conditioning.no_conditioner import NoConditioner
import torch

check(issubclass(NoConditioner, BaseConditioner), "NoConditioner extends BaseConditioner")
check(not inspect.isabstract(NoConditioner), "NoConditioner is concrete (not abstract)")

# Instantiate
nc = NoConditioner()
check(isinstance(nc, BaseConditioner), "NoConditioner instance passes isinstance check")

# Test prepare_for_training
train_result = nc.prepare_for_training(batch=torch.randn(2, 4, 8, 8), device="cpu")
check(isinstance(train_result, dict), "prepare_for_training returns dict")
check(len(train_result) == 0, "prepare_for_training returns empty dict (no-op)")

# Test prepare_for_sampling
sample_result = nc.prepare_for_sampling(batch_size=4, device="cpu")
check(isinstance(sample_result, dict), "prepare_for_sampling returns dict")
check(len(sample_result) == 0, "prepare_for_sampling returns empty dict (no-op)")


# ══════════════════════════════════════════════════════════════════════════
# D. Registry registration
# ══════════════════════════════════════════════════════════════════════════
print("\n=== D. Registry registration ===")
from src.core.registry import REGISTRIES

check("no_conditioning" in REGISTRIES.conditioning.list(),
      "NoConditioner registered as 'no_conditioning'")

default_cls = REGISTRIES.conditioning.get()
check(default_cls is NoConditioner, "NoConditioner is the default conditioning")

resolved = REGISTRIES.conditioning.get("no_conditioning")
check(resolved is NoConditioner, "REGISTRIES.conditioning.get('no_conditioning') → NoConditioner")


# ══════════════════════════════════════════════════════════════════════════
# E. FlowMatchingTrainer conditioning support
# ══════════════════════════════════════════════════════════════════════════
print("\n=== E. FlowMatchingTrainer conditioning support ===")
from src.algorithms.training.flow_matching_trainer import FlowMatchingTrainer

sig_t = inspect.signature(FlowMatchingTrainer.__init__)
check("conditioner" in sig_t.parameters, "Trainer __init__ has conditioner parameter")
check(sig_t.parameters["conditioner"].default is None, "Trainer conditioner default is None")

trainer_src = (ROOT / "src" / "algorithms" / "training" / "flow_matching_trainer.py").read_text()
check("self.conditioner = conditioner" in trainer_src, "Trainer stores self.conditioner")
check("self.conditioner.prepare_for_training" in trainer_src,
      "Trainer calls conditioner.prepare_for_training")

# flow_matching_step signature
sig_fms = inspect.signature(FlowMatchingTrainer.flow_matching_step)
check("cond_kwargs" in sig_fms.parameters,
      "flow_matching_step has cond_kwargs parameter")
check(sig_fms.parameters["cond_kwargs"].default is None,
      "cond_kwargs default is None")

# flow_matching_step passes kwargs to UNet
fms_src = inspect.getsource(FlowMatchingTrainer.flow_matching_step)
check("**cond_kwargs" in fms_src, "flow_matching_step unpacks **cond_kwargs to UNet")

# Training loop passes cond_kw
check("self.flow_matching_step(x_fm, cond_kw)" in trainer_src,
      "Training loop passes cond_kw to flow_matching_step")


# ══════════════════════════════════════════════════════════════════════════
# F. FlowMatchingSampler conditioning support
# ══════════════════════════════════════════════════════════════════════════
print("\n=== F. FlowMatchingSampler conditioning support ===")
from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler

sig_s = inspect.signature(FlowMatchingSampler.__init__)
check("conditioner" in sig_s.parameters, "Sampler __init__ has conditioner parameter")
check(sig_s.parameters["conditioner"].default is None, "Sampler conditioner default is None")

sampler_src = (ROOT / "src" / "algorithms" / "inference" / "flow_matching_sampler.py").read_text()
check("self.conditioner = conditioner" in sampler_src, "Sampler stores self.conditioner")

# set_conditioner method
check(hasattr(FlowMatchingSampler, "set_conditioner"), "set_conditioner method exists")
check(callable(FlowMatchingSampler.set_conditioner), "set_conditioner is callable")

# _cond_kwargs helper
check(hasattr(FlowMatchingSampler, "_cond_kwargs"), "_cond_kwargs helper exists")
check("self.conditioner.prepare_for_sampling" in sampler_src,
      "_cond_kwargs calls conditioner.prepare_for_sampling")

# UNet calls in all sampling methods use cond kwargs
euler_src = inspect.getsource(FlowMatchingSampler.sample_euler)
check("_cond_kwargs" in euler_src, "sample_euler uses _cond_kwargs")
check("**cond_kw" in euler_src, "sample_euler unpacks **cond_kw to UNet")

guided_src = inspect.getsource(FlowMatchingSampler.sample_euler_guided)
check("_cond_kwargs" in guided_src, "sample_euler_guided uses _cond_kwargs")
check("**cond_kw" in guided_src, "sample_euler_guided unpacks **cond_kw to UNet")


# ══════════════════════════════════════════════════════════════════════════
# G. Runtime conditioner injection
# ══════════════════════════════════════════════════════════════════════════
print("\n=== G. Runtime conditioner injection ===")


class _MockUNet:
    class config:
        in_channels = 4
        sample_size = 8

    def __call__(self, z, t, **kwargs):
        class _Out:
            sample = torch.randn_like(z)
        return _Out()

    def eval(self):
        pass

    def parameters(self):
        return iter([])

    def train(self, mode=True):
        pass


mock_unet = _MockUNet()

# Sampler without conditioner
sampler_no_c = FlowMatchingSampler(mock_unet, device="cpu", t_scale=1000.0)
check(sampler_no_c.conditioner is None, "Sampler without conditioner: conditioner is None")

kw = sampler_no_c._cond_kwargs(2)
check(kw == {}, "_cond_kwargs returns {} when conditioner is None")

# Sampler with NoConditioner
sampler_with_c = FlowMatchingSampler(mock_unet, device="cpu", t_scale=1000.0, conditioner=nc)
check(sampler_with_c.conditioner is nc, "Sampler with conditioner: conditioner is set")

kw2 = sampler_with_c._cond_kwargs(2)
check(kw2 == {}, "_cond_kwargs returns {} with NoConditioner")

# set_conditioner at runtime
sampler_no_c.set_conditioner(nc)
check(sampler_no_c.conditioner is nc, "set_conditioner injects conditioner at runtime")

# sample_euler still works with conditioner
z_plain = sampler_no_c.sample_euler(steps=3, batch_size=2)
check(z_plain.shape == (2, 4, 8, 8), "sample_euler output shape correct with conditioner set")

# sample_euler_guided still works with conditioner
z_guided = sampler_with_c.sample_euler_guided(steps=3, batch_size=2, guidance_scale=0.0)
check(z_guided.shape == (2, 4, 8, 8), "sample_euler_guided output shape correct with conditioner")


# Custom conditioner test
class _LabelConditioner(BaseConditioner):
    def prepare_for_training(self, batch, device="cpu"):
        return {"class_labels": torch.zeros(2, dtype=torch.long, device=device)}

    def prepare_for_sampling(self, batch_size, device="cpu"):
        return {"class_labels": torch.ones(batch_size, dtype=torch.long, device=device)}


lc = _LabelConditioner()
check(isinstance(lc, BaseConditioner), "Custom conditioner extends BaseConditioner")

tr = lc.prepare_for_training(None, device="cpu")
check("class_labels" in tr, "Custom conditioner returns class_labels in training")
check(tr["class_labels"].shape == (2,), "Training class_labels shape correct")

sr = lc.prepare_for_sampling(4, device="cpu")
check("class_labels" in sr, "Custom conditioner returns class_labels in sampling")
check(sr["class_labels"].shape == (4,), "Sampling class_labels shape correct")


# ══════════════════════════════════════════════════════════════════════════
# H. Source inspection (UNet calls + wiring)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== H. Source inspection ===")

# base_conditioner.py
check("abc.ABC" in bc_src, "base_conditioner uses abc.ABC")
check("@abc.abstractmethod" in bc_src, "base_conditioner has @abc.abstractmethod")

# no_conditioner.py
nc_src = (cond_pkg / "no_conditioner.py").read_text()
check("BaseConditioner" in nc_src, "no_conditioner imports BaseConditioner")
check("REGISTRIES.conditioning.register" in nc_src, "no_conditioner registers with REGISTRIES")

# Trainer source: cond wiring
check("cond_kw = self.conditioner.prepare_for_training" in trainer_src,
      "Trainer derives cond_kw from conditioner")
check("if self.conditioner is not None else {}" in trainer_src,
      "Trainer gracefully handles None conditioner")

# Sampler source: all UNet calls have cond_kw
# Count occurrences of the pattern
import re
unet_calls = re.findall(r"self\.unet\(.*?\)\.sample", sampler_src)
cond_unet_calls = [c for c in unet_calls if "cond_kw" in c]
check(len(unet_calls) == len(cond_unet_calls),
      f"All {len(unet_calls)} UNet calls in sampler include cond kwargs")

# beam sampling also wired
if hasattr(FlowMatchingSampler, "sample_euler_beam"):
    beam_src = inspect.getsource(FlowMatchingSampler.sample_euler_beam)
    check("_cond_kwargs" in beam_src, "sample_euler_beam uses _cond_kwargs")
    check("**cond_kw" in beam_src, "sample_euler_beam unpacks **cond_kw to UNet")


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Conditioning integration checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    sys.exit(1)
else:
    print("All checks passed!")
