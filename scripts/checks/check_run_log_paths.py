#!/usr/bin/env python3
"""Check that run/log canonical paths resolve correctly.

Validates:
  1. Helpers import and return Paths under artifacts_root()
  2. Physical directories exist on disk
  3. CLI defaults reference new paths
  4. No stale root-level run/log references in shell scripts
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
print("\n=== 1. Import run/log helpers ===")
from src.core.paths import (
    artifacts_root,
    runs_root,
    default_runs_dir,
    runs_test_dir,
    repo_root,
)

check("runs_root under artifacts", str(runs_root()).startswith(str(artifacts_root())))
check("default_runs_dir under runs_root", str(default_runs_dir()).startswith(str(runs_root())))
check("runs_test_dir under runs_root", str(runs_test_dir()).startswith(str(runs_root())))

# ═══════════════════════════════════════════════════════════════════════════
# 2. Expected relative paths
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 2. Relative path checks ===")
check("runs_root == artifacts/runs", runs_root() == artifacts_root() / "runs")
check("default_runs_dir == artifacts/runs/main", default_runs_dir() == artifacts_root() / "runs" / "main")
check("runs_test_dir == artifacts/runs/test", runs_test_dir() == artifacts_root() / "runs" / "test")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Directories exist on disk
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 3. Directories exist ===")
check("artifacts/runs/main exists", default_runs_dir().is_dir())
check("artifacts/runs/test exists", runs_test_dir().is_dir())

# Old roots must NOT exist at repo root
check("runs/ NOT at repo root", not (repo_root() / "runs").is_dir())
check("runs_test/ NOT at repo root", not (repo_root() / "runs_test").is_dir())

# ═══════════════════════════════════════════════════════════════════════════
# 4. CLI defaults reference new paths (source inspection, avoids heavy imports)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 4. CLI defaults ===")
vae_src = (REPO / "src" / "cli" / "train_vae.py").read_text()
check("train_vae --log-dir default contains artifacts/", "artifacts/runs/main" in vae_src)

# ═══════════════════════════════════════════════════════════════════════════
# 5. No stale shell-script references
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 5. No stale shell-script references ===")
stale_pattern = re.compile(r'"\$ROOT_DIR/(runs/|runs_test)')
scripts_dir = REPO / "scripts"
stale_found = []
for sh in sorted(scripts_dir.glob("*.sh")):
    text = sh.read_text()
    for m in stale_pattern.finditer(text):
        if "artifacts/" not in text[max(0, m.start() - 30): m.end()]:
            stale_found.append(f"{sh.name}:{m.group()}")
check(f"No stale runs/runs_test refs in shell scripts ({len(stale_found)} found)", len(stale_found) == 0)
if stale_found:
    for s in stale_found[:5]:
        print(f"    → {s}")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  {ok} passed, {fail} failed ({ok + fail} total)")
sys.exit(1 if fail else 0)
