#!/usr/bin/env python
"""Phase 15 checks – Guidance abstraction integration.

Tests:
  A. Module structure
  B. BaseGuidance ABC
  C. NoGuidance implementation
  D. NoGuidance registry registration
  E. FlowMatchingSampler accepts guidance
  F. Default sampling unchanged (structural)
  G. Guidance injection at runtime
  H. Source inspection
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
guidance_pkg = ROOT / "src" / "guidance"
check(guidance_pkg.is_dir(), "src/guidance/ exists")
check((guidance_pkg / "__init__.py").is_file(), "src/guidance/__init__.py exists")
check((guidance_pkg / "base_guidance.py").is_file(), "src/guidance/base_guidance.py exists")
check((guidance_pkg / "no_guidance.py").is_file(), "src/guidance/no_guidance.py exists")

# ══════════════════════════════════════════════════════════════════════════
# B. BaseGuidance ABC
# ══════════════════════════════════════════════════════════════════════════
print("\n=== B. BaseGuidance ABC ===")
from src.guidance.base_guidance import BaseGuidance
import abc

check(inspect.isabstract(BaseGuidance), "BaseGuidance is abstract")
check(issubclass(BaseGuidance, abc.ABC), "BaseGuidance extends ABC")

# Check required abstract methods
abstract_methods = set(BaseGuidance.__abstractmethods__)
check("energy" in abstract_methods, "energy is abstract")
check("guidance_grad" in abstract_methods, "guidance_grad is abstract")
check("log_scores" in abstract_methods, "log_scores is abstract")
check(len(abstract_methods) == 3, f"exactly 3 abstract methods (got {len(abstract_methods)})")

# Check signatures
sig_energy = inspect.signature(BaseGuidance.energy)
check("z" in sig_energy.parameters, "energy has z param")
check("t" in sig_energy.parameters, "energy has t param")
check("pipeline" in sig_energy.parameters, "energy has pipeline param")

sig_grad = inspect.signature(BaseGuidance.guidance_grad)
check("z" in sig_grad.parameters, "guidance_grad has z param")
check("velocity" in sig_grad.parameters, "guidance_grad has velocity param")

sig_scores = inspect.signature(BaseGuidance.log_scores)
check("z" in sig_scores.parameters, "log_scores has z param")

# ══════════════════════════════════════════════════════════════════════════
# C. NoGuidance implementation
# ══════════════════════════════════════════════════════════════════════════
print("\n=== C. NoGuidance implementation ===")
from src.guidance.no_guidance import NoGuidance
import torch

check(issubclass(NoGuidance, BaseGuidance), "NoGuidance extends BaseGuidance")
check(not inspect.isabstract(NoGuidance), "NoGuidance is concrete (not abstract)")

# Instantiate
ng = NoGuidance()
check(isinstance(ng, BaseGuidance), "NoGuidance instance passes isinstance check")

# Test energy
z_test = torch.randn(2, 4, 8, 8)
e = ng.energy(z_test, t=0.5)
check(e.shape == (2,), f"energy returns shape (B,): {e.shape}")
check(torch.all(e == 0.0).item(), "energy returns all zeros")

# Test guidance_grad
g = ng.guidance_grad(z_test, t=0.5, velocity=torch.randn_like(z_test))
check(g.shape == z_test.shape, f"guidance_grad returns same shape: {g.shape}")
check(torch.all(g == 0.0).item(), "guidance_grad returns all zeros")

# Test log_scores
scores = ng.log_scores(z_test)
check(isinstance(scores, dict), "log_scores returns dict")
check(len(scores) == 0, "log_scores returns empty dict (no-op)")


# ══════════════════════════════════════════════════════════════════════════
# D. Registry registration
# ══════════════════════════════════════════════════════════════════════════
print("\n=== D. Registry registration ===")
from src.core.registry import REGISTRIES

# Before accessing, ensure no_guidance module is imported (registration is at module level)
check("no_guidance" in REGISTRIES.guidance.list(),
      "NoGuidance registered as 'no_guidance'")

default_cls = REGISTRIES.guidance.get()
check(default_cls is NoGuidance, "NoGuidance is the default guidance")

resolved = REGISTRIES.guidance.get("no_guidance")
check(resolved is NoGuidance, "REGISTRIES.guidance.get('no_guidance') → NoGuidance")


# ══════════════════════════════════════════════════════════════════════════
# E. FlowMatchingSampler accepts guidance
# ══════════════════════════════════════════════════════════════════════════
print("\n=== E. FlowMatchingSampler guidance support ===")
from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler

# Check __init__ signature accepts guidance
sig = inspect.signature(FlowMatchingSampler.__init__)
check("guidance" in sig.parameters, "__init__ has guidance parameter")

# Check default is None
default_val = sig.parameters["guidance"].default
check(default_val is None, "guidance default is None")

# Check set_guidance method exists
check(hasattr(FlowMatchingSampler, "set_guidance"), "set_guidance method exists")
check(callable(FlowMatchingSampler.set_guidance), "set_guidance is callable")

# Check sampler stores self.guidance
sampler_src = (ROOT / "src" / "algorithms" / "inference" / "flow_matching_sampler.py").read_text()
check("self.guidance = guidance" in sampler_src, "sampler stores self.guidance")
check("self.guidance" in sampler_src, "self.guidance is referenced")


# ══════════════════════════════════════════════════════════════════════════
# F. Default sampling structurally unchanged
# ══════════════════════════════════════════════════════════════════════════
print("\n=== F. Default sampling structurally unchanged ===")

# sample_euler should NOT reference self.guidance at all (fast path)
# Extract sample_euler method source
euler_src = inspect.getsource(FlowMatchingSampler.sample_euler)
check("self.guidance" not in euler_src,
      "sample_euler does NOT use self.guidance (fast, unmodified path)")
check("guidance" not in euler_src.lower() or "guidance" not in euler_src,
      "sample_euler has no guidance logic")

# sample_euler_guided should fall back to self.guidance
guided_src = inspect.getsource(FlowMatchingSampler.sample_euler_guided)
check("active_guidance" in guided_src,
      "sample_euler_guided uses active_guidance variable")
check("self.guidance" in guided_src,
      "sample_euler_guided falls back to self.guidance")

# ══════════════════════════════════════════════════════════════════════════
# G. Guidance injection at runtime
# ══════════════════════════════════════════════════════════════════════════
print("\n=== G. Runtime guidance injection ===")

# Create a mock UNet for structural testing
class _MockUNet:
    class config:
        in_channels = 4
        sample_size = 8

    def __call__(self, z, t):
        class _Out:
            sample = torch.randn_like(z)
        return _Out()

    def eval(self):
        pass

mock_unet = _MockUNet()

# Build sampler without guidance
sampler_no_g = FlowMatchingSampler(mock_unet, device="cpu", t_scale=1000.0)
check(sampler_no_g.guidance is None, "sampler without guidance: guidance is None")

# Build sampler with NoGuidance
sampler_with_g = FlowMatchingSampler(mock_unet, device="cpu", t_scale=1000.0, guidance=ng)
check(sampler_with_g.guidance is ng, "sampler with guidance: guidance is set")

# set_guidance at runtime
sampler_no_g.set_guidance(ng)
check(sampler_no_g.guidance is ng, "set_guidance injects guidance at runtime")

# Replace guidance
class _DummyGuidance(BaseGuidance):
    def energy(self, z, t=None, pipeline=None):
        return torch.ones(z.shape[0])

    def guidance_grad(self, z, t=None, pipeline=None, velocity=None):
        return torch.ones_like(z) * 0.01

    def log_scores(self, z):
        return {"dummy": 1.0}

dummy = _DummyGuidance()
sampler_with_g.set_guidance(dummy)
check(sampler_with_g.guidance is dummy, "set_guidance can replace guidance")

# Verify sample_euler still works (no guidance interference)
z_plain = sampler_no_g.sample_euler(steps=3, batch_size=2)
check(z_plain.shape == (2, 4, 8, 8), "sample_euler output shape correct with guidance set")

# Verify sample_euler_guided picks up instance guidance
z_guided = sampler_with_g.sample_euler_guided(
    steps=3, batch_size=2, guidance_scale=1.0
)
check(z_guided.shape == (2, 4, 8, 8), "sample_euler_guided output shape correct with instance guidance")

# Verify explicit guidance overrides instance guidance
z_explicit = sampler_with_g.sample_euler_guided(
    steps=3, batch_size=2, guidance=ng, guidance_scale=1.0
)
check(z_explicit.shape == (2, 4, 8, 8), "explicit guidance param overrides instance guidance")


# ══════════════════════════════════════════════════════════════════════════
# H. Source inspection
# ══════════════════════════════════════════════════════════════════════════
print("\n=== H. Source inspection ===")

# base_guidance.py
bg_src = (guidance_pkg / "base_guidance.py").read_text()
check("abc.ABC" in bg_src or "abc.ABC" in bg_src, "base_guidance uses abc.ABC")
check("@abc.abstractmethod" in bg_src, "base_guidance has @abc.abstractmethod")

# no_guidance.py
ng_src = (guidance_pkg / "no_guidance.py").read_text()
check("BaseGuidance" in ng_src, "no_guidance imports BaseGuidance")
check("REGISTRIES.guidance.register" in ng_src, "no_guidance registers with REGISTRIES")
check("torch.zeros_like" in ng_src, "no_guidance returns zeros for grad")
check("torch.zeros(" in ng_src, "no_guidance returns zeros for energy")

# sampler
check("active_guidance = guidance if guidance is not None else self.guidance" in sampler_src,
      "guided method has proper fallback logic")


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Guidance integration checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    sys.exit(1)
else:
    print("All checks passed!")
