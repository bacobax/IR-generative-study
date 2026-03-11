#!/usr/bin/env python3
"""Check that src.core.paths canonical helpers resolve to real locations.

Validates:
  1. Every helper returns a Path under repo_root()
  2. Config JSON files (copied to configs/models/fm/) exist
  3. Data directory helpers point to the correct subdirectories
  4. Legacy-code archive root exists
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
# 1. Import all helpers
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 1. Import canonical helpers ===")
try:
    from src.core.paths import (
        repo_root,
        data_root,
        raw_data_root,
        derived_data_root,
        cache_root,
        artifacts_root,
        archive_root,
        legacy_code_root,
        configs_root,
        v18_root,
        default_data_dir,
        surprise_pred_dataset_root,
        dino_cache_dir,
        fm_model_configs_dir,
        stable_unet_config_path,
        non_stable_unet_config_path,
        vae_config_path,
        vae_config_x8_path,
        default_outputs_dir,
        default_models_dir,
        default_analysis_dir,
    )
    check("All helpers imported", True)
except Exception as e:
    check(f"Import error: {e}", False)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Root helpers resolve correctly
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 2. Root helpers ===")
check("repo_root() matches REPO", repo_root() == REPO)
check("data_root() == repo/data", data_root() == REPO / "data")
check("raw_data_root() == repo/data/raw", raw_data_root() == REPO / "data" / "raw")
check("derived_data_root() == repo/data/derived", derived_data_root() == REPO / "data" / "derived")
check("cache_root() == repo/data/cache", cache_root() == REPO / "data" / "cache")
check("artifacts_root() == repo/artifacts", artifacts_root() == REPO / "artifacts")
check("archive_root() == repo/archive", archive_root() == REPO / "archive")
check("legacy_code_root() == repo/archive/legacy_code", legacy_code_root() == REPO / "archive" / "legacy_code")
check("configs_root() == repo/configs", configs_root() == REPO / "configs")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Model config JSON files exist
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 3. Config JSON existence ===")
check("stable_unet_config.json exists", stable_unet_config_path().is_file())
check("non_stable_unet_config.json exists", non_stable_unet_config_path().is_file())
check("vae_config.json exists", vae_config_path().is_file())
check("vae_config_x8.json exists", vae_config_x8_path().is_file())
check("fm_model_configs_dir() is dir", fm_model_configs_dir().is_dir())


# ═══════════════════════════════════════════════════════════════════════════
# 4. Data directory helpers
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 4. Data directory helpers ===")
check("v18_root() == data/raw/v18", v18_root() == REPO / "data" / "raw" / "v18")
check("default_data_dir('train') == data/raw/v18/train", default_data_dir("train") == REPO / "data" / "raw" / "v18" / "train")
check("default_data_dir('val') == data/raw/v18/val", default_data_dir("val") == REPO / "data" / "raw" / "v18" / "val")
check("surprise_pred_dataset_root() == data/derived/surprise_pred_dataset",
      surprise_pred_dataset_root() == REPO / "data" / "derived" / "surprise_pred_dataset")
check("dino_cache_dir() == data/cache/dino_cache",
      dino_cache_dir() == REPO / "data" / "cache" / "dino_cache")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Module-level constants use canonical paths
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 5. Module constants ===")
from src.models.fm_unet import STABLE_UNET_CONFIG, NON_STABLE_UNET_CONFIG
from src.models.vae import VAE_CONFIG, VAE_CONFIG_X8

check("STABLE_UNET_CONFIG resolves via paths", STABLE_UNET_CONFIG == str(stable_unet_config_path()))
check("NON_STABLE_UNET_CONFIG resolves via paths", NON_STABLE_UNET_CONFIG == str(non_stable_unet_config_path()))
check("VAE_CONFIG resolves via paths", VAE_CONFIG == str(vae_config_path()))
check("VAE_CONFIG_X8 resolves via paths", VAE_CONFIG_X8 == str(vae_config_x8_path()))


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Repo-paths checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    sys.exit(1)
else:
    print("All checks passed!")
