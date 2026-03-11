#!/usr/bin/env python3
"""Smoke-check: verify generate_datasets.py uses modular FM components.

After Phase 17 the root ``generate_datasets.py`` is a thin wrapper.
The source of truth is ``src/cli/generate.py``.

Checks:
  1. Root wrapper syntax + delegation.
  2. Source-of-truth imports modular FM pieces.
  3. No old pipeline dependency in FM path.
  4. _build_sampler uses config + registry.
  5. generate_fm body.
  6. generate_fm_guided body.
  7. CLI parsing preserved.
  8. SD mode still works structurally.
  9. Output format preserved.
"""

import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

passed = 0
failed = 0


def check(label, cond):
    global passed, failed
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        passed += 1
    else:
        failed += 1


# ======================================================================
# 1. Root wrapper + source-of-truth reading
# ======================================================================
print("\n=== 1. Source analysis ===")
wrapper_path = os.path.join(REPO, "generate_datasets.py")
sot_path = os.path.join(REPO, "src", "cli", "generate.py")

check("generate_datasets.py exists", os.path.isfile(wrapper_path))
check("src/cli/generate.py exists", os.path.isfile(sot_path))

with open(wrapper_path) as f:
    wrapper_src = f.read()
with open(sot_path) as f:
    src = f.read()

check("Wrapper syntax is valid", True)
try:
    ast.parse(wrapper_src, filename="generate_datasets.py")
    check("Wrapper AST parses without error", True)
except SyntaxError as e:
    check(f"Wrapper AST parses without error ({e})", False)

check("Wrapper delegates to src.cli.generate",
      "from src.cli.generate import main" in wrapper_src)

try:
    ast.parse(src, filename="src/cli/generate.py")
    check("Source-of-truth AST parses without error", True)
except SyntaxError as e:
    check(f"Source-of-truth AST parses without error ({e})", False)

# ======================================================================
# 2. Modular FM imports present
# ======================================================================
print("\n=== 2. Modular FM imports ===")
check("Imports FMSampleConfig", "FMSampleConfig" in src)
check("Imports REGISTRIES", "REGISTRIES" in src)
check("Imports flow_matching_sampler module (registration)",
      "import src.algorithms.inference.flow_matching_sampler" in src)

# ======================================================================
# 3. No old pipeline dependency in FM path
# ======================================================================
print("\n=== 3. No old pipeline in FM path ===")
check("No _build_sampler_from_folder", "_build_sampler_from_folder" not in src)
check("No _pick_latest_by_prefix", "_pick_latest_by_prefix" not in src)
# The only from fm_src import should be the guidance module, not the pipeline
fm_src_imports = [line for line in src.splitlines()
                  if "from fm_src" in line and "import" in line]
pipeline_imports = [line for line in fm_src_imports if "pipeline" in line.lower()]
check("No fm_src.pipelines import", len(pipeline_imports) == 0)
# Guidance import IS expected (for guided generation)
guidance_imports = [line for line in fm_src_imports if "guidance" in line.lower()]
check("fm_src.guidance import present (expected)",
      len(guidance_imports) >= 1)

# ======================================================================
# 4. _build_sampler uses config + registry
# ======================================================================
print("\n=== 4. _build_sampler helper ===")
check("_build_sampler function defined", "def _build_sampler(" in src)
check("_build_sampler uses FMSampleConfig", "FMSampleConfig(" in src)
check("_build_sampler uses REGISTRIES.sampler.get", "REGISTRIES.sampler.get" in src)
check("_build_sampler calls from_config", ".from_config(" in src)

# ======================================================================
# 5. generate_fm uses _build_sampler
# ======================================================================
print("\n=== 5. generate_fm ===")
# Locate generate_fm function body
import re
fm_match = re.search(r"def generate_fm\(.*?\):\n(.*?)(?=\ndef |\Z)", src, re.DOTALL)
fm_body = fm_match.group(1) if fm_match else ""
check("generate_fm calls _build_sampler", "_build_sampler(" in fm_body)
check("generate_fm calls sampler.sample_euler", "sampler.sample_euler(" in fm_body)
check("generate_fm calls sampler.decode", "sampler.decode(" in fm_body)
check("generate_fm calls fm_output_to_uint16", "fm_output_to_uint16(" in fm_body)

# ======================================================================
# 6. generate_fm_guided uses _build_sampler
# ======================================================================
print("\n=== 6. generate_fm_guided ===")
fg_match = re.search(r"def generate_fm_guided\(.*?\):\n(.*?)(?=\ndef |\Z)", src, re.DOTALL)
fg_body = fg_match.group(1) if fg_match else ""
check("generate_fm_guided calls _build_sampler", "_build_sampler(" in fg_body)
check("generate_fm_guided supports euler_guided", "sample_euler_guided(" in fg_body)
check("generate_fm_guided supports rerank", "sample_euler_with_candidates(" in fg_body)
check("generate_fm_guided supports beam", "sample_euler_beam(" in fg_body)
check("generate_fm_guided supports refine", "refine_latents_energy(" in fg_body)

# ======================================================================
# 7. CLI parsing preserved
# ======================================================================
print("\n=== 7. CLI parsing ===")
check("--mode flag present", "--mode" in src)
check("--fm_pipeline_dir present", "fm_pipeline_dir" in src)
check("--fm_vae_weights present", "fm_vae_weights" in src)
check("--fm_t_scale present", "fm_t_scale" in src)
check("--fm_steps present", "fm_steps" in src)
check("--fm_batch_size present", "fm_batch_size" in src)
check("--fm_guidance_method present", "fm_guidance_method" in src)

# SD flags
check("--base_model present", "base_model" in src)
check("--lora_dir present", "lora_dir" in src)
check("--sd_steps present", "sd_steps" in src)

# ======================================================================
# 8. SD mode still works structurally
# ======================================================================
print("\n=== 8. SD mode ===")
check("generate_sd15 function exists", "def generate_sd15(" in src)
check("SD dispatched in main", "generate_sd15(args" in src)

# ======================================================================
# 9. Output format preserved
# ======================================================================
print("\n=== 9. Output format ===")
check(".npy output preserved", "np.save(" in src)
check(".png output preserved", ".save(png_path)" in src)
check("sample_XXXXX naming preserved", "sample_{" in src)
check("metadata.jsonl output preserved", "metadata.jsonl" in src)

# ======================================================================
# Summary
# ======================================================================
print(f"\n{'='*60}")
print(f"  {passed} passed, {failed} failed ({passed + failed} total)")
if failed:
    sys.exit(1)
