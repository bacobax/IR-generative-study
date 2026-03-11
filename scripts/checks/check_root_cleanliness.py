#!/usr/bin/env python3
"""Check that the repository root is intentionally clean.

Validates:
  1. Root directory tree at depth 1
  2. Only expected top-level entries present
  3. No unexpected heavy folders at root
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ok = fail = 0


def check(label, cond):
    global ok, fail
    status = "PASS" if cond else "FAIL"
    if not cond:
        fail += 1
    else:
        ok += 1
    print(f"  [{status}] {label}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Print root tree at depth 1
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 1. Root tree (depth 1) ===")
entries = sorted(REPO.iterdir())
for e in entries:
    kind = "DIR " if e.is_dir() else "FILE"
    print(f"  {kind}  {e.name}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Expected root entries (whitelist)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 2. Root entry whitelist ===")

EXPECTED_DIRS = {
    "src", "configs", "scripts", "docs", "data", "artifacts", "archive",
    "tests", "__pycache__", ".git",
}

EXPECTED_FILES = {
    "README.md", "pyproject.toml", ".gitignore",
    # Thin wrappers
    "train_sfm.py", "train_sd.py", "train_vae.py",
    "train_controlnet.py", "generate_datasets.py",
}

actual_dirs = {e.name for e in entries if e.is_dir()}
actual_files = {e.name for e in entries if e.is_file()}

# Unexpected dirs
unexpected_dirs = actual_dirs - EXPECTED_DIRS
check(f"No unexpected directories ({len(unexpected_dirs)} found)", len(unexpected_dirs) == 0)
if unexpected_dirs:
    for d in sorted(unexpected_dirs):
        print(f"    → unexpected dir: {d}/")

# Unexpected files
unexpected_files = actual_files - EXPECTED_FILES
check(f"No unexpected files ({len(unexpected_files)} found)", len(unexpected_files) == 0)
if unexpected_files:
    for f in sorted(unexpected_files):
        print(f"    → unexpected file: {f}")

# Required dirs present
for d in ("src", "configs", "scripts", "docs", "data", "artifacts", "archive", "tests"):
    check(f"Required dir {d}/ present", d in actual_dirs)

# Required files present
for f in ("README.md", "pyproject.toml", ".gitignore"):
    check(f"Required file {f} present", f in actual_files)

# Thin wrappers present
for f in ("train_sfm.py", "train_sd.py", "train_vae.py", "train_controlnet.py", "generate_datasets.py"):
    check(f"Thin wrapper {f} present", f in actual_files)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Flag heavy folders that should NOT be at root
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 3. No heavy clutter at root ===")
FORBIDDEN_ROOT_DIRS = {
    "generated", "generated_test", "old_generated", "debug_samples",
    "analysis_results", "analysis_results_test",
    "serious_runs", "vae_runs", "stable_diffusion_15_out",
    "count_adapter_runs", "pipeline_model", "UNET", "VAE",
    "runs", "runs_test", "v18", "surprise_pred_dataset",
    "fm_src", "sd_src",
}

for forbidden in sorted(FORBIDDEN_ROOT_DIRS):
    check(f"{forbidden}/ NOT at root", forbidden not in actual_dirs)

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  {ok} passed, {fail} failed ({ok + fail} total)")
sys.exit(1 if fail else 0)
