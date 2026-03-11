#!/usr/bin/env python3
"""Check that every shell wrapper references an existing YAML config preset,
and every YAML preset is referenced by at least one shell wrapper.

Exit 0 if all mappings are valid, exit 1 otherwise.
"""
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

SHELL_DIRS = [
    REPO / "scripts" / "train",
    REPO / "scripts" / "generate",
    REPO / "scripts" / "analyze",
]

CONFIG_ROOTS = [
    REPO / "configs" / "fm",
    REPO / "configs" / "sd",
    REPO / "configs" / "vae",
    REPO / "configs" / "controlnet",
    REPO / "configs" / "auxiliary",
    REPO / "configs" / "analysis",
]

_passed = 0
_failed = 0


def check(label: str, ok: bool):
    global _passed, _failed
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}")
    if ok:
        _passed += 1
    else:
        _failed += 1


# ── 1. Collect shell wrappers and extract --config references ──────────────
print("=== 1. Shell wrapper → config mapping ===")
CONFIG_RE = re.compile(r"--config\s+(\S+)")
shell_to_config: dict[str, str] = {}

for d in SHELL_DIRS:
    if not d.is_dir():
        continue
    for sh in sorted(d.glob("*.sh")):
        text = sh.read_text()
        m = CONFIG_RE.search(text)
        if m:
            config_rel = m.group(1)
            # Strip shell variable interpolation if any
            config_rel = config_rel.strip('"').strip("'")
            shell_to_config[str(sh.relative_to(REPO))] = config_rel

# Verify each shell wrapper's config exists
for sh_rel, cfg_rel in sorted(shell_to_config.items()):
    cfg_path = REPO / cfg_rel
    check(f"{sh_rel} → {cfg_rel}", cfg_path.is_file())

# ── 2. Collect all YAML presets (only inside presets/ subdirs) ──────────────
print("\n=== 2. YAML presets with shell wrappers ===")
all_presets: set[str] = set()
for root in CONFIG_ROOTS:
    if not root.is_dir():
        continue
    for yaml_file in root.rglob("presets/*.yaml"):
        all_presets.add(str(yaml_file.relative_to(REPO)))

# Check which YAML presets are referenced by at least one wrapper
referenced = set(shell_to_config.values())
for preset in sorted(all_presets):
    has_wrapper = preset in referenced
    check(f"{preset} referenced by a wrapper", has_wrapper)

# ── 3. Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {_passed} passed, {_failed} failed")
print(f"  Shell wrappers: {len(shell_to_config)}")
print(f"  YAML presets:   {len(all_presets)}")
sys.exit(1 if _failed else 0)
