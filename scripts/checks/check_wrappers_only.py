#!/usr/bin/env python
"""Phase 17 checks – Root scripts are thin wrappers.

Tests:
  A. Root wrapper files exist and are short (≤ 15 lines of code)
  B. Each root wrapper imports from the correct src.cli.* module
  C. Source-of-truth files exist under src/cli/
  D. Root wrappers contain no core logic (no training loops, no pipeline classes)
  E. src/cli/train_vae.py has no StableFlowMatchingPipeline dependency
  F. BBoxConditioningDataset lives in src/core/data/datasets.py
"""

import ast
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


# Mapping: root wrapper file -> (expected import source, source-of-truth file)
WRAPPER_MAP = {
    "train_sfm.py":         ("src.cli.train",            "src/cli/train.py"),
    "train_sd.py":          ("src.cli.train_sd",         "src/cli/train_sd.py"),
    "train_vae.py":         ("src.cli.train_vae",        "src/cli/train_vae.py"),
    "train_controlnet.py":  ("src.cli.train_controlnet", "src/cli/train_controlnet.py"),
    "generate_datasets.py": ("src.cli.generate",         "src/cli/generate.py"),
}

# Patterns that should NOT appear in a thin wrapper
FORBIDDEN_PATTERNS = [
    "class ",
    "def train",
    "def generate",
    "FlowMatchingPipeline",
    "StableFlowMatchingPipeline",
    "ControlNetFlowMatchingPipeline",
    "torch.optim",
    "backward()",
    "loss.item()",
]

# ══════════════════════════════════════════════════════════════════════════
# A. Root wrappers exist and are short
# ══════════════════════════════════════════════════════════════════════════
print("\n=== A. Root wrappers exist and are short ===")
for wrapper_name in WRAPPER_MAP:
    wrapper_path = ROOT / wrapper_name
    check(wrapper_path.is_file(), f"{wrapper_name} exists")
    if wrapper_path.is_file():
        lines = wrapper_path.read_text().splitlines()
        # Count non-blank, non-comment lines
        code_lines = [
            ln for ln in lines
            if ln.strip() and not ln.strip().startswith("#")
        ]
        # Filter out docstring blocks (rough: lines that are just triple-quotes)
        check(len(code_lines) <= 15,
              f"{wrapper_name} has ≤15 code lines (got {len(code_lines)})")

# ══════════════════════════════════════════════════════════════════════════
# B. Each root wrapper imports from the correct src.cli module
# ══════════════════════════════════════════════════════════════════════════
print("\n=== B. Root wrappers import from correct src.cli modules ===")
for wrapper_name, (expected_import, _) in WRAPPER_MAP.items():
    wrapper_path = ROOT / wrapper_name
    if not wrapper_path.is_file():
        check(False, f"{wrapper_name}: file missing, cannot check import")
        continue

    source = wrapper_path.read_text()
    # Look for e.g. "from src.cli.train_sd import main"
    found_import = False
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(expected_import):
                    found_import = True
                    break
    except SyntaxError:
        pass

    check(found_import,
          f"{wrapper_name} imports from {expected_import}")

# ══════════════════════════════════════════════════════════════════════════
# C. Source-of-truth files exist under src/cli/
# ══════════════════════════════════════════════════════════════════════════
print("\n=== C. Source-of-truth files exist ===")
for wrapper_name, (_, sot_relpath) in WRAPPER_MAP.items():
    sot_path = ROOT / sot_relpath
    check(sot_path.is_file(),
          f"{sot_relpath} exists (source of truth for {wrapper_name})")

# ══════════════════════════════════════════════════════════════════════════
# D. Root wrappers contain no core logic
# ══════════════════════════════════════════════════════════════════════════
print("\n=== D. Root wrappers contain no core logic ===")
for wrapper_name in WRAPPER_MAP:
    wrapper_path = ROOT / wrapper_name
    if not wrapper_path.is_file():
        check(False, f"{wrapper_name}: file missing, cannot check content")
        continue

    source = wrapper_path.read_text()
    clean = True
    violations = []
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in source:
            clean = False
            violations.append(pattern)

    check(clean,
          f"{wrapper_name} has no forbidden patterns"
          + (f" (found: {violations})" if violations else ""))

# ══════════════════════════════════════════════════════════════════════════
# E. src/cli/train_vae.py has no old pipeline dependency
# ══════════════════════════════════════════════════════════════════════════
print("\n=== E. train_vae.py source of truth – no old pipeline ===")
vae_sot = ROOT / "src" / "cli" / "train_vae.py"
if vae_sot.is_file():
    vae_source = vae_sot.read_text()
    check("StableFlowMatchingPipeline" not in vae_source,
          "src/cli/train_vae.py has no StableFlowMatchingPipeline import")
    check("FlowMatchingPipeline" not in vae_source,
          "src/cli/train_vae.py has no FlowMatchingPipeline import")
    has_fm_src_import = ("import fm_src" in vae_source
                         or "from fm_src" in vae_source)
    check(not has_fm_src_import,
          "src/cli/train_vae.py has no fm_src import")
    # Positive: uses src.models.vae
    check("src.models.vae" in vae_source,
          "src/cli/train_vae.py imports from src.models.vae")
else:
    for _ in range(4):
        check(False, "src/cli/train_vae.py missing")

# ══════════════════════════════════════════════════════════════════════════
# F. BBoxConditioningDataset in src/core/data/datasets.py
# ══════════════════════════════════════════════════════════════════════════
print("\n=== F. BBoxConditioningDataset location ===")
datasets_path = ROOT / "src" / "core" / "data" / "datasets.py"
if datasets_path.is_file():
    ds_source = datasets_path.read_text()
    check("class BBoxConditioningDataset" in ds_source,
          "BBoxConditioningDataset class defined in src/core/data/datasets.py")
else:
    check(False, "src/core/data/datasets.py missing")

# Check that root train_controlnet.py does NOT define it
tc_root = ROOT / "train_controlnet.py"
if tc_root.is_file():
    tc_source = tc_root.read_text()
    check("class BBoxConditioningDataset" not in tc_source,
          "train_controlnet.py (root) does NOT define BBoxConditioningDataset")
else:
    check(False, "train_controlnet.py missing for BBoxConditioningDataset check")

# ══════════════════════════════════════════════════════════════════════════
# G. Source-of-truth files have main() callable
# ══════════════════════════════════════════════════════════════════════════
print("\n=== G. Source-of-truth files expose main() ===")
for wrapper_name, (_, sot_relpath) in WRAPPER_MAP.items():
    sot_path = ROOT / sot_relpath
    if not sot_path.is_file():
        check(False, f"{sot_relpath} missing")
        continue

    source = sot_path.read_text()
    try:
        tree = ast.parse(source)
        has_main = any(
            isinstance(node, ast.FunctionDef) and node.name == "main"
            for node in ast.walk(tree)
        )
        check(has_main, f"{sot_relpath} defines main()")
    except SyntaxError as e:
        check(False, f"{sot_relpath} has syntax error: {e}")

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Phase 17 wrapper checks: {ok} passed, {fail} failed "
      f"(total {ok + fail})")
if fail:
    sys.exit(1)
else:
    print("All checks passed!")
