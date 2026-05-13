#!/usr/bin/env python3
"""Check that active src/ modules have NO hard-coded runtime dependency on fm_src/ or sd_src/.

This script scans all Python files under ``src/`` for forbidden path
patterns that would break now that ``fm_src`` and ``sd_src`` live in
``archive/legacy_code/``.

Allowed exceptions:
  - Comments and docstrings mentioning the old paths
"""

import ast
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

ok = fail = 0


def check(label, cond):
    global ok, fail
    status = "PASS" if cond else "FAIL"
    if not cond:
        fail += 1
    else:
        ok += 1
    print(f"  [{status}] {label}")


# Files with sanctioned transitional fm_src imports
SANCTIONED = set()

EXCLUDED_SRC_SUBTREES = {
    SRC / "diffusers",
    SRC / "flow-matching-trial",
}


def _is_excluded_src_path(path: str | Path) -> bool:
    path = Path(path)
    return any(path == excluded or excluded in path.parents for excluded in EXCLUDED_SRC_SUBTREES)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Scan for forbidden string literals containing old paths
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 1. No hard-coded fm_src/ or sd_src/ string literals in src/ ===")
FORBIDDEN_PATTERNS = [
    re.compile(r'"fm_src/'),
    re.compile(r"'fm_src/"),
    re.compile(r'"sd_src/'),
    re.compile(r"'sd_src/"),
]

violations = []
for root, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if not _is_excluded_src_path(Path(root) / d)]
    if _is_excluded_src_path(root):
        continue
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        with open(fpath) as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.lstrip()
                # Skip comments
                if stripped.startswith("#"):
                    continue
                for pat in FORBIDDEN_PATTERNS:
                    if pat.search(line):
                        violations.append(f"{os.path.relpath(fpath, REPO)}:{lineno}: {line.rstrip()}")

check(f"No forbidden string literals ({len(violations)} violations)", len(violations) == 0)
for v in violations:
    print(f"    ! {v}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Scan for import statements referencing fm_src or sd_src
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 2. No unsanctioned fm_src/sd_src imports ===")
import_violations = []
guidance_import_violations = []
for root, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if not _is_excluded_src_path(Path(root) / d)]
    if _is_excluded_src_path(root):
        continue
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        if fpath in SANCTIONED:
            continue
        with open(fpath) as fh:
            try:
                tree = ast.parse(fh.read(), filename=fpath)
            except SyntaxError:
                continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("fm_src", "sd_src")):
                        import_violations.append(
                            f"{os.path.relpath(fpath, REPO)}:{node.lineno}: import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("fm_src", "sd_src")):
                    import_violations.append(
                        f"{os.path.relpath(fpath, REPO)}:{node.lineno}: from {node.module} import ..."
                    )
                if node.module.startswith("fm_src" + ".guidance"):
                    guidance_import_violations.append(
                        f"{os.path.relpath(fpath, REPO)}:{node.lineno}: from {node.module} import ..."
                    )

check(f"No unsanctioned imports ({len(import_violations)} violations)", len(import_violations) == 0)
for v in import_violations:
    print(f"    ! {v}")
check(f"No src imports from legacy guidance ({len(guidance_import_violations)} violations)",
      len(guidance_import_violations) == 0)
for v in guidance_import_violations:
    print(f"    ! {v}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Sanctioned imports use legacy_code_root()
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 3. Sanctioned imports use legacy_code_root() ===")
for fpath in sorted(SANCTIONED):
    relpath = os.path.relpath(fpath, REPO)
    with open(fpath) as fh:
        content = fh.read()
    has_legacy_import = "legacy_code_root" in content
    has_syspath = "sys.path" in content or "_sys.path" in content
    check(f"{relpath}: uses legacy_code_root()", has_legacy_import)
    check(f"{relpath}: uses sys.path insertion", has_syspath)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Legacy code still parseable in archive
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 4. Archived legacy code exists ===")
check("archive/legacy_code/fm_src/ exists", (REPO / "archive" / "legacy_code" / "fm_src").is_dir())
check("archive/legacy_code/sd_src/ exists", (REPO / "archive" / "legacy_code" / "sd_src").is_dir())


# ═══════════════════════════════════════════════════════════════════════════
# 5. Active runtime files do not import legacy guidance
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== 5. No active legacy guidance imports ===")
excluded_parts = {
    ".git",
    "archive",
    "artifacts",
    "data",
    "docs",
    "ControlNet",
    "__pycache__",
}
active_guidance_refs = []
for root_name in ("src", "scripts", "tests"):
    root_path = REPO / root_name
    if not root_path.exists():
        continue
    for fpath in root_path.rglob("*"):
        if not fpath.is_file() or fpath.suffix not in {".py", ".sh"}:
            continue
        rel_parts = set(fpath.relative_to(REPO).parts)
        if rel_parts & excluded_parts:
            continue
        rel = str(fpath.relative_to(REPO))
        if rel.startswith("src/diffusers/") or rel.startswith("src/flow-matching-trial/"):
            continue
        text = fpath.read_text(errors="ignore")
        legacy_from = "from " + "fm_src" + ".guidance"
        legacy_module = "fm_src" + ".guidance.score_predictor_guidance"
        if legacy_from in text or legacy_module in text:
            active_guidance_refs.append(rel)

check(f"No active legacy guidance references ({len(active_guidance_refs)} found)",
      len(active_guidance_refs) == 0)
for ref in active_guidance_refs:
    print(f"    ! {ref}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Legacy-dependency checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    sys.exit(1)
else:
    print("All checks passed!")
