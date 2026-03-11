#!/usr/bin/env python3
"""Check that dataset / cache directories are in their canonical locations.

Validates:
  1. v18 is at data/raw/v18/ (not project root)
  2. surprise_pred_dataset is at data/derived/surprise_pred_dataset/ (not root)
  3. dino_cache is at data/cache/dino_cache/ (not .dino_cache at root)
  4. Old root-level locations do NOT exist (moves were successful)
  5. Key subdirectories exist inside moved dataset dirs
"""

import os
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
# 1. Old root-level locations should NOT exist
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 1. Old locations removed ===")
check("v18/ NOT at root", not (REPO / "v18").exists())
check("surprise_pred_dataset/ NOT at root", not (REPO / "surprise_pred_dataset").exists())
check(".dino_cache/ NOT at root", not (REPO / ".dino_cache").exists())


# ═══════════════════════════════════════════════════════════════════════════
# 2. New canonical locations exist
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 2. New locations exist ===")
check("data/raw/v18/ exists", (REPO / "data" / "raw" / "v18").is_dir())
check("data/derived/surprise_pred_dataset/ exists",
      (REPO / "data" / "derived" / "surprise_pred_dataset").is_dir())
check("data/cache/dino_cache/ exists", (REPO / "data" / "cache" / "dino_cache").is_dir())


# ═══════════════════════════════════════════════════════════════════════════
# 3. Key subdirectories / files inside moved dirs
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 3. Subdirectory integrity ===")
v18 = REPO / "data" / "raw" / "v18"
check("v18/train/ exists", (v18 / "train").is_dir())
check("v18/val/ exists", (v18 / "val").is_dir())
check("v18/test/ exists", (v18 / "test").is_dir())
check("v18/images/ exists", (v18 / "images").is_dir())

spd = REPO / "data" / "derived" / "surprise_pred_dataset"
# At least the dir itself exists (contents depend on whether build was run)
check("surprise_pred_dataset/ is dir", spd.is_dir())
# Check for expected subdirs if populated
has_contents = any(spd.iterdir())
check("surprise_pred_dataset/ is non-empty", has_contents)

dc = REPO / "data" / "cache" / "dino_cache"
check("dino_cache/ is dir", dc.is_dir())


# ═══════════════════════════════════════════════════════════════════════════
# 4. src.core.paths helpers point to moved locations
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 4. paths.py helpers agree with physical layout ===")
from src.core.paths import v18_root, surprise_pred_dataset_root, dino_cache_dir

check("v18_root() matches physical", v18_root() == v18)
check("surprise_pred_dataset_root() matches physical", surprise_pred_dataset_root() == spd)
check("dino_cache_dir() matches physical", dino_cache_dir() == dc)


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Dataset-location checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    sys.exit(1)
else:
    print("All checks passed!")
