#!/usr/bin/env python3
"""Check that generated-output canonical paths resolve correctly.

Validates:
  1. Helpers import and return Paths under artifacts_root()
  2. Physical directories exist on disk
  3. CLI defaults match canonical helpers
  4. Shell scripts reference new artifact paths (no stale roots)
"""

import os
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
print("\n=== 1. Import generated-output helpers ===")
from src.core.paths import (
    artifacts_root,
    generated_root,
    default_outputs_dir,
    generated_test_dir,
    debug_root,
    debug_samples_dir,
)

check("generated_root under artifacts", str(generated_root()).startswith(str(artifacts_root())))
check("default_outputs_dir under generated_root", str(default_outputs_dir()).startswith(str(generated_root())))
check("generated_test_dir under generated_root", str(generated_test_dir()).startswith(str(generated_root())))
check("debug_samples_dir under debug_root", str(debug_samples_dir()).startswith(str(debug_root())))

# ═══════════════════════════════════════════════════════════════════════════
# 2. Expected relative paths
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 2. Relative path checks ===")
check("generated_root == artifacts/generated", generated_root() == artifacts_root() / "generated")
check("default_outputs_dir == artifacts/generated/main", default_outputs_dir() == artifacts_root() / "generated" / "main")
check("generated_test_dir == artifacts/generated/test", generated_test_dir() == artifacts_root() / "generated" / "test")
check("debug_root == artifacts/debug", debug_root() == artifacts_root() / "debug")
check("debug_samples_dir == artifacts/debug/debug_samples", debug_samples_dir() == artifacts_root() / "debug" / "debug_samples")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Directories exist on disk
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 3. Directories exist ===")
check("artifacts/generated/main exists", default_outputs_dir().is_dir())
check("artifacts/generated/test exists", generated_test_dir().is_dir())
check("artifacts/debug/debug_samples exists", debug_samples_dir().is_dir())

# Old roots must NOT exist at repo root
from src.core.paths import repo_root
check("generated/ NOT at repo root", not (repo_root() / "generated").is_dir())
check("generated_test/ NOT at repo root", not (repo_root() / "generated_test").is_dir())
check("debug_samples/ NOT at repo root", not (repo_root() / "debug_samples").is_dir())

# ═══════════════════════════════════════════════════════════════════════════
# 4. CLI defaults reference new paths (source inspection, avoids heavy imports)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 4. CLI defaults ===")
gen_src = (REPO / "src" / "cli" / "generate.py").read_text()
check("generate CLI --output_dir default contains artifacts/", "artifacts/generated/main" in gen_src)

sample_src = (REPO / "src" / "cli" / "sample.py").read_text()
check("sample CLI --output_dir default contains artifacts/", "artifacts/generated/main" in sample_src)

# ═══════════════════════════════════════════════════════════════════════════
# 5. No stale root references in shell scripts
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 5. No stale shell-script references ===")
stale_pattern = re.compile(r'"\$ROOT_DIR/(generated|debug_samples)/')
scripts_dir = REPO / "scripts"
stale_found = []
for sh in sorted(scripts_dir.glob("*.sh")):
    text = sh.read_text()
    for m in stale_pattern.finditer(text):
        if "artifacts/" not in text[max(0, m.start() - 30): m.end()]:
            stale_found.append(f"{sh.name}:{m.group()}")
check(f"No stale generated/debug refs in shell scripts ({len(stale_found)} found)", len(stale_found) == 0)
if stale_found:
    for s in stale_found[:5]:
        print(f"    → {s}")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  {ok} passed, {fail} failed ({ok + fail} total)")
sys.exit(1 if fail else 0)
