#!/usr/bin/env python3
"""Smoke-check: verify the unconditional latent SD training CLI."""

import ast
import os
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


print("\n=== 1. Module imports ===")
from src.cli.train_sd_uncond import build_parser, main, run_training
from src.core.configs.sd_uncond_config import SDUncondTrainConfig, _FLAT_TO_NESTED

check("build_parser importable", callable(build_parser))
check("main importable", callable(main))
check("run_training importable", callable(run_training))
check("SDUncondTrainConfig importable", SDUncondTrainConfig is not None)

print("\n=== 2. Argument parser ===")
parser = build_parser()
defaults = vars(parser.parse_args([]))
check("--dataset_id default == None", defaults.get("dataset_id") is None)
check("--image_size default == 256", defaults.get("image_size") == 256)
check("--model_dir default contains uncond_runs", "uncond_runs" in defaults.get("model_dir", ""))
check("--prediction_type default == 'epsilon'", defaults.get("prediction_type") == "epsilon")
check("--sample_steps default == 50", defaults.get("sample_steps") == 50)

print("\n=== 3. CLI mapping ===")
all_cli_args = {a.dest for a in parser._actions if a.dest not in ("help", "config")}
check(
    "_FLAT_TO_NESTED covers all CLI args",
    set(_FLAT_TO_NESTED.keys()) == all_cli_args,
)

print("\n=== 4. Wrapper forwarding ===")
wrapper_path = os.path.join(REPO, "train_sd_uncond.py")
wrapper_src = open(wrapper_path, encoding="utf-8").read()
check("train_sd_uncond.py imports from src.cli.train_sd_uncond",
      "from src.cli.train_sd_uncond import main" in wrapper_src)
check("train_sd_uncond.py calls main()", "main()" in wrapper_src)
check("train_sd_uncond.py has no argparse", "argparse" not in wrapper_src)

print("\n=== 5. Syntax check ===")
for rel in (
    "src/cli/train_sd_uncond.py",
    "train_sd_uncond.py",
    "src/core/configs/sd_uncond_config.py",
    "src/algorithms/training/unconditional_sd_trainer.py",
    "src/algorithms/inference/unconditional_sd_sampler.py",
):
    try:
        with open(os.path.join(REPO, rel), encoding="utf-8") as handle:
            ast.parse(handle.read(), filename=rel)
        check(f"Syntax OK: {rel}", True)
    except SyntaxError as exc:
        check(f"Syntax OK: {rel} ({exc})", False)

print(f"\n{'=' * 60}")
print(f"  {passed} passed, {failed} failed ({passed + failed} total)")
if failed:
    sys.exit(1)
