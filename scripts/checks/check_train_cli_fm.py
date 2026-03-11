#!/usr/bin/env python3
"""Smoke-check: verify the modular training CLI (src.cli.train).

Checks:
  1. Module and key symbols import cleanly.
  2. Argument parser exposes all expected flags with correct defaults.
  3. FMTrainConfig is built correctly from parsed args.
  4. Trainer is resolved through the registry (default and explicit).
  5. train_sfm.py forwards to src.cli.train.main.
  6. run_training builds the trainer via registry (mock check).
  7. Syntax check on all touched files.
"""

import ast
import os
import sys
from types import SimpleNamespace

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
# 1. Imports
# ======================================================================
print("\n=== 1. Module imports ===")
from src.cli.train import build_parser, main, run_training
check("build_parser importable", callable(build_parser))
check("main importable", callable(main))
check("run_training importable", callable(run_training))

from src.core.configs.fm_config import FMTrainConfig
check("FMTrainConfig importable", FMTrainConfig is not None)

from src.core.registry import REGISTRIES
check("REGISTRIES importable", REGISTRIES is not None)

# ======================================================================
# 2. Parser flags and defaults
# ======================================================================
print("\n=== 2. Argument parser ===")
parser = build_parser()
defaults = vars(parser.parse_args([]))

expected_defaults = {
    "train_dir": "./data/raw/v18/train/",
    "val_dir": "./data/raw/v18/val/",
    "unet_config": "configs/models/fm/stable_unet_config.json",
    "vae_config": "configs/models/fm/vae_config.json",
    "vae_weights": "./vae_best.pt",
    "model_dir": "./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled/",
    "epochs": 100,
    "batch_size": 8,
    "num_workers": 4,
    "save_every_n_epochs": 10,
    "sample_batch_size": 4,
    "t_scale": 1000.0,
    "warmup_frac": 0.1,
    "ramp_frac": 0.3,
    "p_crop_warmup": 0.05,
    "p_crop_max": 0.20,
    "p_crop_final": 0.05,
    "p_rot_warmup": 0.05,
    "p_rot_max": 0.30,
    "p_rot_final": 0.05,
    "resume": None,
    "train_target": "v",
}
for key, val in expected_defaults.items():
    actual = defaults.get(key)
    check(f"--{key} default == {val!r}", actual == val)

# Custom flag values
custom_args = parser.parse_args(["--epochs", "42", "--batch_size", "16",
                                  "--train-target", "x0"])
check("--epochs override", custom_args.epochs == 42)
check("--batch_size override", custom_args.batch_size == 16)
check("--train-target override", custom_args.train_target == "x0")

# ======================================================================
# 3. Config construction from args
# ======================================================================
print("\n=== 3. Config from args ===")
args = parser.parse_args(["--epochs", "50", "--model_dir", "/tmp/test_run"])
cfg = FMTrainConfig.from_args(args)
check("cfg.training.epochs == 50", cfg.training.epochs == 50)
check("cfg.output.model_dir == '/tmp/test_run'", cfg.output.model_dir == "/tmp/test_run")
check("cfg.data.train_dir matches default", cfg.data.train_dir == "./data/raw/v18/train/")
check("cfg.trainer_name is None (default)", cfg.trainer_name is None)
check("cfg.sampler_name is None (default)", cfg.sampler_name is None)

# ======================================================================
# 4. Trainer resolution through registry
# ======================================================================
print("\n=== 4. Registry resolution ===")
from src.algorithms.training.flow_matching_trainer import FlowMatchingTrainer

# Default (name=None)
TrainerCls = REGISTRIES.trainer.get(cfg.trainer_name)
check("Default trainer resolves to FlowMatchingTrainer",
      TrainerCls is FlowMatchingTrainer)

# Explicit name
TrainerCls2 = REGISTRIES.trainer.get("default_fm")
check("Explicit 'default_fm' resolves to FlowMatchingTrainer",
      TrainerCls2 is FlowMatchingTrainer)

# ======================================================================
# 5. train_sfm.py forwards to src.cli.train
# ======================================================================
print("\n=== 5. Wrapper forwarding ===")
wrapper_path = os.path.join(REPO, "train_sfm.py")
with open(wrapper_path) as f:
    wrapper_src = f.read()

check("train_sfm.py imports from src.cli.train",
      "from src.cli.train import main" in wrapper_src)
check("train_sfm.py calls main()",
      "main()" in wrapper_src)
check("train_sfm.py has no argparse (delegated)",
      "argparse" not in wrapper_src)

# ======================================================================
# 6. run_training uses registry
# ======================================================================
print("\n=== 6. run_training uses registry ===")
import inspect
run_src = inspect.getsource(run_training)
check("run_training calls REGISTRIES.trainer.get",
      "REGISTRIES.trainer.get" in run_src)
check("run_training uses cfg.trainer_name",
      "cfg.trainer_name" in run_src)
check("run_training calls from_config",
      ".from_config(" in run_src)
check("run_training calls train_from_config",
      ".train_from_config(" in run_src)

# ======================================================================
# 7. Syntax check
# ======================================================================
print("\n=== 7. Syntax check ===")
files_to_check = [
    "src/cli/train.py",
    "train_sfm.py",
    "src/core/registry.py",
    "src/core/configs/fm_config.py",
]
for rel in files_to_check:
    fpath = os.path.join(REPO, rel)
    try:
        with open(fpath) as f:
            ast.parse(f.read(), filename=rel)
        check(f"Syntax OK: {rel}", True)
    except SyntaxError as e:
        check(f"Syntax OK: {rel} ({e})", False)

# ======================================================================
# Summary
# ======================================================================
print(f"\n{'='*60}")
print(f"  {passed} passed, {failed} failed ({passed + failed} total)")
if failed:
    sys.exit(1)
