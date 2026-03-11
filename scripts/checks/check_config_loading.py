#!/usr/bin/env python
"""Phase 14 checks – Config-file-driven experiments.

Tests:
  A. Config directory structure exists
  B. YAML files parse correctly
  C. config_loader utilities (load_yaml, dataclass_to_dict, dict_to_dataclass, merge)
  D. FM training config: defaults → YAML → CLI override chain
  E. FM sampling config: defaults → YAML → CLI override chain
  F. CLIs accept --config flag
  G. Wrapper scripts still work (source inspection)
  H. YAML defaults match dataclass defaults
"""

import ast
import copy
import sys
from dataclasses import fields, asdict
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


# ══════════════════════════════════════════════════════════════════════════
# A. Directory structure
# ══════════════════════════════════════════════════════════════════════════
print("\n=== A. Config directory structure ===")
for d in ("configs/fm/train", "configs/fm/sample",
          "configs/sd/train", "configs/models/fm", "configs/models/sd"):
    check((ROOT / d).is_dir(), f"{d}/ exists")


# ══════════════════════════════════════════════════════════════════════════
# B. YAML files parse correctly
# ══════════════════════════════════════════════════════════════════════════
print("\n=== B. YAML files parseable ===")
import yaml

yaml_files = {
    "configs/fm/train/default.yaml": "fm_train",
    "configs/fm/sample/default.yaml": "fm_sample",
    "configs/sd/train/default.yaml": "sd_train",
    "configs/models/fm/stable_unet.yaml": "fm_unet",
    "configs/models/fm/vae.yaml": "fm_vae",
    "configs/models/sd/sd15.yaml": "sd_model",
}
loaded_yamls = {}
for rel, tag in yaml_files.items():
    p = ROOT / rel
    check(p.is_file(), f"{rel} exists")
    try:
        data = yaml.safe_load(p.read_text())
        loaded_yamls[tag] = data
        check(isinstance(data, dict), f"{rel} parses as dict")
    except Exception as exc:
        check(False, f"{rel} parse: {exc}")


# ══════════════════════════════════════════════════════════════════════════
# C. config_loader utilities
# ══════════════════════════════════════════════════════════════════════════
print("\n=== C. config_loader module ===")
from src.core.configs.config_loader import (
    load_yaml,
    dataclass_to_dict,
    dict_to_dataclass,
    _deep_merge,
    merge_config_and_cli,
    _cli_overrides,
    _set_dotted,
)
check(callable(load_yaml), "load_yaml importable")
check(callable(dataclass_to_dict), "dataclass_to_dict importable")
check(callable(dict_to_dataclass), "dict_to_dataclass importable")
check(callable(merge_config_and_cli), "merge_config_and_cli importable")

# Test _deep_merge
base = {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
over = {"b": {"y": 99, "z": 30}, "d": 4}
merged = _deep_merge(copy.deepcopy(base), over)
check(merged == {"a": 1, "b": {"x": 10, "y": 99, "z": 30}, "c": 3, "d": 4},
      "_deep_merge works correctly")

# Test _set_dotted
d = {}
_set_dotted(d, "a.b.c", 42)
check(d == {"a": {"b": {"c": 42}}}, "_set_dotted creates nested dicts")

# Test dataclass round-trip
from src.core.configs.fm_config import (
    FMTrainConfig, FMSampleConfig, DataConfig, ModelConfig,
    AugmentConfig, TrainHyperConfig, SampleConfig, OutputConfig,
)
default_train = FMTrainConfig()
d = dataclass_to_dict(default_train)
check(isinstance(d, dict), "dataclass_to_dict returns dict")
check("data" in d and "model" in d, "nested sub-configs present")
check(d["data"]["train_dir"] == "./data/raw/v18/train/", "nested value correct")

roundtrip = dict_to_dataclass(FMTrainConfig, d)
check(roundtrip.data.train_dir == "./data/raw/v18/train/", "round-trip preserves values")
check(roundtrip.training.epochs == 100, "round-trip preserves nested values")

# Test with FMSampleConfig (flat dataclass)
default_sample = FMSampleConfig()
ds = dataclass_to_dict(default_sample)
check(ds["steps"] == 50, "flat dataclass_to_dict correct")
rs = dict_to_dataclass(FMSampleConfig, ds)
check(rs.steps == 50, "flat round-trip correct")


# ══════════════════════════════════════════════════════════════════════════
# D. FM training: defaults → YAML → CLI merge
# ══════════════════════════════════════════════════════════════════════════
print("\n=== D. FM train config merge chain ===")
from src.cli.train import build_parser as train_build_parser, _FLAT_TO_NESTED as TRAIN_MAP

train_parser = train_build_parser()

# D1: No config, no CLI overrides → pure defaults
args_d1 = train_parser.parse_args([])
cfg_d1 = merge_config_and_cli(FMTrainConfig, None, train_parser, args_d1,
                                flat_to_nested=TRAIN_MAP)
check(cfg_d1.data.train_dir == "./data/raw/v18/train/", "no-config: train_dir default")
check(cfg_d1.training.epochs == 100, "no-config: epochs default")
check(cfg_d1.augment.p_rot_max == 0.30, "no-config: p_rot_max default")

# D2: YAML only → values from YAML
yaml_path = ROOT / "configs/fm/train/default.yaml"
args_d2 = train_parser.parse_args([])
cfg_d2 = merge_config_and_cli(FMTrainConfig, str(yaml_path), train_parser, args_d2,
                                flat_to_nested=TRAIN_MAP)
check(cfg_d2.data.train_dir == "./data/raw/v18/train/", "yaml-only: train_dir from yaml")
check(cfg_d2.training.t_scale == 1000.0, "yaml-only: t_scale from yaml")

# D3: YAML + CLI override → CLI wins
args_d3 = train_parser.parse_args(["--epochs", "200", "--batch_size", "16"])
cfg_d3 = merge_config_and_cli(FMTrainConfig, str(yaml_path), train_parser, args_d3,
                                flat_to_nested=TRAIN_MAP)
check(cfg_d3.training.epochs == 200, "yaml+cli: CLI epochs overrides yaml")
check(cfg_d3.data.batch_size == 16, "yaml+cli: CLI batch_size overrides yaml")
check(cfg_d3.data.train_dir == "./data/raw/v18/train/", "yaml+cli: unset fields keep yaml value")

# D4: CLI only (no config file) with some overrides
args_d4 = train_parser.parse_args(["--epochs", "50"])
cfg_d4 = merge_config_and_cli(FMTrainConfig, None, train_parser, args_d4,
                                flat_to_nested=TRAIN_MAP)
check(cfg_d4.training.epochs == 50, "cli-only: epochs overridden")
check(cfg_d4.data.batch_size == 8, "cli-only: batch_size stays default")

# D5: Check _FLAT_TO_NESTED covers all CLI args (except --config)
all_cli_args = {a.dest for a in train_parser._actions if a.dest not in ("help", "config")}
mapped_args = set(TRAIN_MAP.keys())
check(mapped_args == all_cli_args,
      f"_FLAT_TO_NESTED covers all CLI args (mapped={len(mapped_args)}, cli={len(all_cli_args)})")


# ══════════════════════════════════════════════════════════════════════════
# E. FM sampling: defaults → YAML → CLI merge
# ══════════════════════════════════════════════════════════════════════════
print("\n=== E. FM sample config merge chain ===")
from src.cli.sample import build_parser as sample_build_parser

sample_parser = sample_build_parser()

# E1: No config → defaults
args_e1 = sample_parser.parse_args([])
cfg_e1 = merge_config_and_cli(FMSampleConfig, None, sample_parser, args_e1)
check(cfg_e1.steps == 50, "sample no-config: steps default")
check(cfg_e1.batch_size == 8, "sample no-config: batch_size default")
check(cfg_e1.t_scale == 1000.0, "sample no-config: t_scale default")

# E2: YAML only
sample_yaml = ROOT / "configs/fm/sample/default.yaml"
args_e2 = sample_parser.parse_args([])
cfg_e2 = merge_config_and_cli(FMSampleConfig, str(sample_yaml), sample_parser, args_e2)
check(cfg_e2.steps == 50, "sample yaml-only: steps from yaml")
check(cfg_e2.pipeline_dir == "./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled/",
      "sample yaml-only: pipeline_dir from yaml")

# E3: YAML + CLI override
args_e3 = sample_parser.parse_args(["--steps", "100", "--batch_size", "4"])
cfg_e3 = merge_config_and_cli(FMSampleConfig, str(sample_yaml), sample_parser, args_e3)
check(cfg_e3.steps == 100, "sample yaml+cli: CLI steps overrides")
check(cfg_e3.batch_size == 4, "sample yaml+cli: CLI batch_size overrides")
check(cfg_e3.t_scale == 1000.0, "sample yaml+cli: t_scale from yaml (not overridden)")


# ══════════════════════════════════════════════════════════════════════════
# F. CLIs accept --config flag
# ══════════════════════════════════════════════════════════════════════════
print("\n=== F. CLI --config flag ===")
# train parser
check(any(a.dest == "config" for a in train_parser._actions),
      "train parser has --config flag")
# sample parser
check(any(a.dest == "config" for a in sample_parser._actions),
      "sample parser has --config flag")

# Source inspection: train.py imports merge_config_and_cli
train_src = (ROOT / "src" / "cli" / "train.py").read_text()
check("merge_config_and_cli" in train_src, "train.py uses merge_config_and_cli")
check("_FLAT_TO_NESTED" in train_src, "train.py defines _FLAT_TO_NESTED")

sample_src = (ROOT / "src" / "cli" / "sample.py").read_text()
check("merge_config_and_cli" in sample_src, "sample.py uses merge_config_and_cli")
check("load_yaml" in sample_src, "sample.py handles YAML for extra fields")


# ══════════════════════════════════════════════════════════════════════════
# G. Wrapper scripts still work (source inspection)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== G. Wrapper scripts ===")
wrapper = (ROOT / "train_sfm.py").read_text()
check("from src.cli.train import main" in wrapper,
      "train_sfm.py still delegates to src.cli.train")

tree = ast.parse(wrapper)
# Should not have its own argparse or config loading
check("argparse" not in wrapper, "train_sfm.py has no argparse (thin wrapper)")


# ══════════════════════════════════════════════════════════════════════════
# H. YAML defaults match dataclass defaults
# ══════════════════════════════════════════════════════════════════════════
print("\n=== H. YAML ↔ dataclass default consistency ===")

# FM train
ft_yaml = loaded_yamls.get("fm_train", {})
ft_default = dataclass_to_dict(FMTrainConfig())
for section in ("data", "model", "augment", "training", "sampling", "output"):
    if section in ft_yaml:
        for key, yaml_val in ft_yaml[section].items():
            dc_val = ft_default.get(section, {}).get(key, "__MISSING__")
            check(yaml_val == dc_val,
                  f"fm_train yaml {section}.{key}: {yaml_val!r} == {dc_val!r}")

# FM sample (flat)
fs_yaml = loaded_yamls.get("fm_sample", {})
fs_default = dataclass_to_dict(FMSampleConfig())
for key in ("pipeline_dir", "t_scale", "train_target", "steps", "batch_size"):
    check(fs_yaml.get(key) == fs_default.get(key),
          f"fm_sample yaml {key}: {fs_yaml.get(key)!r} == {fs_default.get(key)!r}")


# ══════════════════════════════════════════════════════════════════════════
# I. Resolved settings printout
# ══════════════════════════════════════════════════════════════════════════
print("\n=== I. Resolved settings (config-file scenario) ===")
# Simulate: load FM train config from file, override epochs via CLI
args_final = train_parser.parse_args(
    ["--config", str(yaml_path), "--epochs", "42"]
)
cfg_final = merge_config_and_cli(
    FMTrainConfig, args_final.config, train_parser, args_final,
    flat_to_nested=TRAIN_MAP,
)
print(f"  train_dir   = {cfg_final.data.train_dir}")
print(f"  epochs      = {cfg_final.training.epochs}")
print(f"  batch_size  = {cfg_final.data.batch_size}")
print(f"  t_scale     = {cfg_final.training.t_scale}")
print(f"  model_dir   = {cfg_final.output.model_dir}")
check(cfg_final.training.epochs == 42, "resolved: epochs=42 from CLI override")
check(cfg_final.data.train_dir == "./data/raw/v18/train/", "resolved: train_dir from YAML")


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Config-loading checks: {ok} passed, {fail} failed, {ok + fail} total")
if fail:
    sys.exit(1)
else:
    print("All checks passed!")
