#!/usr/bin/env python3
"""Check that analysis-output canonical paths resolve correctly.

Validates:
  1. Helpers import and return Paths under artifacts_root()
  2. Physical directories exist on disk
  3. CLI defaults reference new paths
  4. Shell scripts reference new artifact paths (no stale roots)
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
print("\n=== 1. Import analysis-output helpers ===")
from src.core.paths import (
    artifacts_root,
    analysis_root,
    default_analysis_dir,
    analysis_test_dir,
    repo_root,
)

check("analysis_root under artifacts", str(analysis_root()).startswith(str(artifacts_root())))
check("default_analysis_dir under analysis_root", str(default_analysis_dir()).startswith(str(analysis_root())))
check("analysis_test_dir under analysis_root", str(analysis_test_dir()).startswith(str(analysis_root())))

# ═══════════════════════════════════════════════════════════════════════════
# 2. Expected relative paths
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 2. Relative path checks ===")
check("analysis_root == artifacts/analysis", analysis_root() == artifacts_root() / "analysis")
check("default_analysis_dir == artifacts/analysis/main", default_analysis_dir() == artifacts_root() / "analysis" / "main")
check("analysis_test_dir == artifacts/analysis/test", analysis_test_dir() == artifacts_root() / "analysis" / "test")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Directories exist on disk
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 3. Directories exist ===")
check("artifacts/analysis/main exists", default_analysis_dir().is_dir())
check("artifacts/analysis/test exists", analysis_test_dir().is_dir())

# Old roots must NOT exist at repo root
check("analysis_results/ NOT at repo root", not (repo_root() / "analysis_results").is_dir())
check("analysis_results_test/ NOT at repo root", not (repo_root() / "analysis_results_test").is_dir())

# ═══════════════════════════════════════════════════════════════════════════
# 4. Python defaults reference new paths
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 4. Python defaults ===")
src = (REPO / "scripts" / "standalone" / "analyze_distribution_shift.py").read_text()
check("analyze_distribution_shift.py uses artifacts/ for output_dir", "artifacts/analysis/main" in src)
check("analyze_distribution_shift.py uses artifacts/ for generated_dir", "artifacts/generated/main" in src)

# ═══════════════════════════════════════════════════════════════════════════
# 5. No stale shell-script references
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 5. No stale shell-script references ===")
stale_pattern = re.compile(r'"\$ROOT_DIR/(analysis_results)')
scripts_dir = REPO / "scripts"
stale_found = []
for sh in sorted(scripts_dir.rglob("*.sh")):
    text = sh.read_text()
    for m in stale_pattern.finditer(text):
        if "artifacts/" not in text[max(0, m.start() - 30): m.end()]:
            stale_found.append(f"{sh.name}:{m.group()}")
check(f"No stale analysis_results refs in shell scripts ({len(stale_found)} found)", len(stale_found) == 0)

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  {ok} passed, {fail} failed ({ok + fail} total)")
sys.exit(1 if fail else 0)
