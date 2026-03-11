#!/usr/bin/env python
"""Phase 13 checks – SD code migrated into src.algorithms.stable_diffusion."""

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

ok = fail = 0


def check(cond: bool, msg: str):
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {msg}")
    else:
        fail += 1
        print(f"  [FAIL] {msg}")


# ── 1. File structure ─────────────────────────────────────────────────
print("\n=== 1. File structure ===")
sd_pkg = ROOT / "src" / "algorithms" / "stable_diffusion"
for name in ("__init__.py", "config.py", "data.py", "models.py",
             "training.py", "utils.py", "helpers.py"):
    check((sd_pkg / name).is_file(),
          f"src/algorithms/stable_diffusion/{name} exists")

check((sd_pkg / "scripts" / "sd_demo.py").is_file(),
      "scripts/sd_demo.py exists inside SD package")

# ── 2. Lightweight package import (no heavy deps loaded eagerly) ──────
print("\n=== 2. Package-level import (lazy) ===")
try:
    import src.algorithms.stable_diffusion as sd_pkg_mod
    check(True, "import src.algorithms.stable_diffusion succeeds")
except Exception as exc:
    check(False, f"import src.algorithms.stable_diffusion: {exc}")

# Verify __init__ does NOT eagerly pull heavy symbols
check(not hasattr(sd_pkg_mod, "Trainer"),
      "__init__ does NOT eagerly export Trainer (lazy)")
check(not hasattr(sd_pkg_mod, "ModelComponents"),
      "__init__ does NOT eagerly export ModelComponents (lazy)")

# ── 3. Direct sub-module imports (config – lightweight) ───────────────
print("\n=== 3. config sub-module ===")
try:
    from src.algorithms.stable_diffusion.config import TrainingConfig
    check(True, "TrainingConfig importable")
except Exception as exc:
    check(False, f"TrainingConfig import: {exc}")

try:
    from src.algorithms.stable_diffusion.config import parse_args
    check(callable(parse_args), "parse_args is callable")
except Exception as exc:
    check(False, f"parse_args import: {exc}")

# ── 4. helpers sub-module ──────────────────────────────────────────────
print("\n=== 4. helpers sub-module ===")
try:
    from src.algorithms.stable_diffusion.helpers import (
        ir_to_3ch_with_stretch,
        trainable_params,
        generate_prompt,
    )
    check(callable(ir_to_3ch_with_stretch), "ir_to_3ch_with_stretch callable")
    check(callable(trainable_params), "trainable_params callable")
    check(callable(generate_prompt), "generate_prompt callable")
except Exception as exc:
    check(False, f"helpers import: {exc}")

# Helpers bridge constants from src.core.constants
try:
    from src.algorithms.stable_diffusion.helpers import A1, B1
    check(isinstance(A1, (int, float)), "A1 constant available via helpers")
    check(isinstance(B1, (int, float)), "B1 constant available via helpers")
except Exception as exc:
    check(False, f"helpers constants: {exc}")

# ── 5. Source inspection: train_sd.py ──────────────────────────────────
# After Phase 17 the root train_sd.py is a thin wrapper; source of truth
# is src/cli/train_sd.py.
print("\n=== 5. train_sd.py uses new namespace ===")
train_sd_sot = (ROOT / "src" / "cli" / "train_sd.py").read_text()
tree = ast.parse(train_sd_sot)
imports_from = [
    node.module for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom) and node.module
]
check(any(m.startswith("src.algorithms.stable_diffusion") for m in imports_from),
      "train_sd.py imports from src.algorithms.stable_diffusion")
check(not any(m.startswith("sd_src.") for m in imports_from),
      "train_sd.py does NOT import from sd_src.*")

# Specific imports in src/cli/train_sd.py
check("src.algorithms.stable_diffusion.config" in imports_from,
      "train_sd.py → config")
check("src.algorithms.stable_diffusion.data" in imports_from,
      "train_sd.py → data")
check("src.algorithms.stable_diffusion.models" in imports_from,
      "train_sd.py → models")
check("src.algorithms.stable_diffusion.training" in imports_from,
      "train_sd.py → training")
check("src.algorithms.stable_diffusion.utils" in imports_from,
      "train_sd.py → utils")

# ── 6. Source inspection: sd_src/__init__.py is thin compat layer ──────
print("\n=== 6. sd_src backward-compat layer ===")
sd_init = (ROOT / "archive" / "legacy_code" / "sd_src" / "__init__.py").read_text()
check("src.algorithms.stable_diffusion" in sd_init,
      "sd_src/__init__.py references src.algorithms.stable_diffusion")
check("__getattr__" in sd_init or "import_module" in sd_init,
      "sd_src/__init__.py uses lazy import mechanism")

# ── 7. Source inspection: data.py import fix ───────────────────────────
print("\n=== 7. data.py uses src.core.constants ===")
data_src = (sd_pkg / "data.py").read_text()
check("from src.core.constants import" in data_src,
      "data.py imports from src.core.constants")
check("from sd_src." not in data_src,
      "data.py does NOT import from sd_src.*")

# ── 8. Internal relative imports in training.py ────────────────────────
print("\n=== 8. training.py relative imports ===")
training_src = (sd_pkg / "training.py").read_text()
check("from .config" in training_src or "from .config " in training_src,
      "training.py uses relative import from .config")
check("from .models" in training_src or "from .models " in training_src,
      "training.py uses relative import from .models")

# ── 9. No stale sd_src absolute imports in new package ─────────────────
print("\n=== 9. No stale sd_src imports in new package ===")
for py in sd_pkg.rglob("*.py"):
    rel = py.relative_to(sd_pkg)
    src_text = py.read_text()
    if rel.name == "__init__.py" and rel.parent == Path("."):
        continue  # skip package init
    check("from sd_src." not in src_text and "import sd_src" not in src_text,
          f"{rel}: no sd_src imports")

# ── 10. generate_datasets.py does NOT import sd_src ────────────────────
print("\n=== 10. generate_datasets.py clean ===")
gen = (ROOT / "generate_datasets.py").read_text()
# generate_datasets uses sd_src only in the SD pipeline path which is
# separate from the FM path. Check there's no sd_src import at all.
gen_tree = ast.parse(gen)
gen_imports = [
    node.module for node in ast.walk(gen_tree)
    if isinstance(node, ast.ImportFrom) and node.module
]
check(not any(m.startswith("sd_src") for m in gen_imports),
      "generate_datasets.py has no sd_src imports")

# ── Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Phase 13 checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    sys.exit(1)
else:
    print("All checks passed!")
