#!/usr/bin/env python3
"""Check that checkpoint canonical paths resolve correctly.

Validates:
  1. Helpers import and return Paths under artifacts_root()
  2. Physical directories exist on disk
  3. CLI and config defaults reference new paths
  4. No stale root-level checkpoint references in shell scripts
"""

import re
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
# 1. Import helpers
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 1. Import checkpoint helpers ===")
from src.core.paths import (
    artifacts_root,
    checkpoints_root,
    fm_checkpoints_root,
    default_models_dir,
    vae_checkpoints_root,
    vae_runs_dir,
    sd_checkpoints_root,
    sd_lora_runs_dir,
    count_adapter_checkpoints_root,
    count_adapter_runs_dir,
    legacy_checkpoints_root,
    repo_root,
)

check("checkpoints_root under artifacts", str(checkpoints_root()).startswith(str(artifacts_root())))
check("fm_checkpoints_root under checkpoints", str(fm_checkpoints_root()).startswith(str(checkpoints_root())))
check("default_models_dir under fm_checkpoints", str(default_models_dir()).startswith(str(fm_checkpoints_root())))
check("vae_runs_dir under vae_checkpoints", str(vae_runs_dir()).startswith(str(vae_checkpoints_root())))
check("sd_lora_runs_dir under sd_checkpoints", str(sd_lora_runs_dir()).startswith(str(sd_checkpoints_root())))
check("count_adapter_runs_dir under count_adapter", str(count_adapter_runs_dir()).startswith(str(count_adapter_checkpoints_root())))
check("legacy_checkpoints_root under checkpoints", str(legacy_checkpoints_root()).startswith(str(checkpoints_root())))

# ═══════════════════════════════════════════════════════════════════════════
# 2. Expected relative paths
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 2. Relative path checks ===")
check("checkpoints_root == artifacts/checkpoints", checkpoints_root() == artifacts_root() / "checkpoints")
check("fm serious_runs dir", default_models_dir() == fm_checkpoints_root() / "serious_runs")
check("vae_runs_dir", vae_runs_dir() == vae_checkpoints_root() / "vae_runs")
check("sd_lora_runs_dir", sd_lora_runs_dir() == sd_checkpoints_root() / "lora_runs")
check("count_adapter_runs_dir", count_adapter_runs_dir() == count_adapter_checkpoints_root() / "runs")
check("legacy_checkpoints_root", legacy_checkpoints_root() == checkpoints_root() / "legacy")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Directories exist on disk
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 3. Directories exist ===")
check("fm serious_runs exists", default_models_dir().is_dir())
check("vae_runs exists", vae_runs_dir().is_dir())
check("sd lora_runs exists", sd_lora_runs_dir().is_dir())
check("count_adapter runs exists", count_adapter_runs_dir().is_dir())
check("legacy checkpoints exists", legacy_checkpoints_root().is_dir())

# Old roots must NOT exist at repo root
for old in ("serious_runs", "vae_runs", "stable_diffusion_15_out", "count_adapter_runs", "pipeline_model", "UNET", "VAE"):
    check(f"{old}/ NOT at repo root", not (repo_root() / old).is_dir())

# ═══════════════════════════════════════════════════════════════════════════
# 4. Config defaults reference new paths
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 4. Config defaults ===")
from src.core.configs.fm_config import OutputConfig, FMSampleConfig

o = OutputConfig()
check("OutputConfig.model_dir contains artifacts/", "artifacts/" in o.model_dir)

s = FMSampleConfig()
check("FMSampleConfig.pipeline_dir contains artifacts/", "artifacts/" in s.pipeline_dir)

# ═══════════════════════════════════════════════════════════════════════════
# 5. No stale shell-script references
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 5. No stale shell-script references ===")
stale_pattern = re.compile(r'"\$ROOT_DIR/(serious_runs|vae_runs|stable_diffusion_15_out|count_adapter_runs|pipeline_model)/')
scripts_dir = REPO / "scripts"
stale_found = []
for sh in sorted(scripts_dir.rglob("*.sh")):
    text = sh.read_text()
    for m in stale_pattern.finditer(text):
        if "artifacts/" not in text[max(0, m.start() - 30): m.end()]:
            stale_found.append(f"{sh.name}:{m.group()}")
check(f"No stale checkpoint refs in shell scripts ({len(stale_found)} found)", len(stale_found) == 0)
if stale_found:
    for s in stale_found[:10]:
        print(f"    → {s}")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  {ok} passed, {fail} failed ({ok + fail} total)")
sys.exit(1 if fail else 0)
