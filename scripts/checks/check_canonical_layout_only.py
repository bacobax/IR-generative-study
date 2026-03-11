#!/usr/bin/env python3
"""Check that code/configs/scripts use only canonical artifact paths.

Validates:
  1. No Python source in src/ references old root-level output locations
  2. No YAML configs reference old root-level output locations
  3. No shell scripts reference old root-level output locations
  4. src/core/paths.py has no fallback logic for old locations
  5. Docstrings in CLIs use new artifact paths
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


# Old root-level directory names that should never appear as path references
OLD_ROOTS = [
    "serious_runs", "vae_runs", "stable_diffusion_15_out",
    "count_adapter_runs", "pipeline_model",
    "analysis_results", "debug_samples",
    "generated_test", "old_generated", "runs_test",
]

# Pattern: ./old_root/ or $ROOT_DIR/old_root/ (but NOT inside artifacts/)
OLD_ROOT_PATTERN = re.compile(
    r'(?:\.\/|"\$ROOT_DIR\/)(?:' + "|".join(re.escape(r) for r in OLD_ROOTS) + r')(?:\/|")',
)


def scan_files(glob_pattern, base_dir, label):
    """Scan files for stale old-root references, excluding comment-only matches."""
    stale = []
    for fpath in sorted(base_dir.rglob(glob_pattern)):
        if "__pycache__" in str(fpath):
            continue
        text = fpath.read_text(errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if OLD_ROOT_PATTERN.search(line) and "artifacts/" not in line:
                stale.append(f"{fpath.relative_to(REPO)}:{i}: {line.strip()[:100]}")
    check(f"No stale root refs in {label} ({len(stale)} found)", len(stale) == 0)
    if stale:
        for s in stale[:10]:
            print(f"    → {s}")
    return stale


# ═══════════════════════════════════════════════════════════════════════════
# 1. Python source in src/
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 1. Python source in src/ ===")
scan_files("*.py", REPO / "src", "src/**/*.py")

# ═══════════════════════════════════════════════════════════════════════════
# 2. YAML configs
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 2. YAML configs ===")
stale_yaml = []
for ext in ("*.yaml", "*.yml"):
    for fpath in sorted((REPO / "configs").rglob(ext)):
        text = fpath.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            for old in OLD_ROOTS:
                if f"./{old}/" in line and "artifacts/" not in line:
                    stale_yaml.append(f"{fpath.relative_to(REPO)}:{i}: {line.strip()[:100]}")
check(f"No stale root refs in configs/ ({len(stale_yaml)} found)", len(stale_yaml) == 0)
if stale_yaml:
    for s in stale_yaml[:5]:
        print(f"    → {s}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Shell scripts
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 3. Shell scripts ===")
scan_files("*.sh", REPO / "scripts", "scripts/**/*.sh")

# ═══════════════════════════════════════════════════════════════════════════
# 4. paths.py has no fallback logic
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 4. paths.py clean of fallback logic ===")
paths_src = (REPO / "src" / "core" / "paths.py").read_text()
check("No 'if.*exists' fallback in paths.py", "if" not in paths_src or ".exists()" not in paths_src)
check("No 'fallback' mention in paths.py", "fallback" not in paths_src.lower())

for old in OLD_ROOTS:
    # These names are reused as subdirectory names within artifacts/ hierarchy.
    # Only flag if they appear as a top-level path reference (./old_root or
    # ROOT_DIR/old_root), not as a Path() child segment like `parent / "name"`.
    stale_in_paths = False
    for line in paths_src.splitlines():
        stripped = line.strip()
        # Skip docstrings (start with """), comments, function signatures
        if stripped.startswith(('"""', "#", "def ")):
            continue
        # Skip Path child construction: `parent() / "name"`
        if f'/ "{old}"' in stripped or f"/ '{old}'" in stripped:
            continue
        # Now check for actual stale path reference
        if f'"{old}/' in stripped or f"'{old}/" in stripped or f"./{old}" in stripped:
            stale_in_paths = True
            break
    check(f"paths.py has no stale top-level ref to '{old}'", not stale_in_paths)

# ═══════════════════════════════════════════════════════════════════════════
# 5. CLI docstrings use new paths
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 5. CLI docstrings ===")
for cli_file in ("generate.py", "sample.py", "train.py", "train_controlnet.py"):
    fpath = REPO / "src" / "cli" / cli_file
    if fpath.exists():
        src = fpath.read_text()
        # Check docstring (first triple-quoted block)
        doc_end = src.find('"""', src.find('"""') + 3)
        docstring = src[:doc_end] if doc_end > 0 else ""
        has_stale = any(old in docstring for old in OLD_ROOTS if "artifacts/" not in docstring)
        # More precise: check each old root appears without artifacts/ context
        stale_in_doc = []
        for old in OLD_ROOTS:
            if old in docstring:
                # Check if it appears without artifacts/ prefix on the same line
                for line in docstring.splitlines():
                    if old in line and "artifacts/" not in line:
                        stale_in_doc.append(old)
                        break
        check(f"{cli_file} docstring uses canonical paths ({len(stale_in_doc)} stale)",
              len(stale_in_doc) == 0)

# Also check standalone scripts that were moved
print("\n=== 6. Standalone scripts in scripts/ ===")
scan_files("*.py", REPO / "scripts", "scripts/**/*.py (non-check)")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  {ok} passed, {fail} failed ({ok + fail} total)")
sys.exit(1 if fail else 0)
