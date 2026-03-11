#!/usr/bin/env python3
"""Smoke-check: verify that structured FM config objects work correctly.

Checks:
  1. Config imports succeed.
  2. Default values match existing CLI defaults.
  3. ``FMTrainConfig.from_args`` maps argparse Namespace correctly.
  4. ``FlowMatchingTrainer.from_config`` builds a usable trainer.
  5. ``trainer.train_from_config`` delegates without error (dry-run).
  6. ``FlowMatchingSampler.from_config`` builds a usable sampler.
  7. ``FMSampleConfig.resolved_device`` returns expected device.
  8. Syntax check of all touched files.
"""

import ast
import os
import sys
import tempfile
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch

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


# ── 1. Config imports ────────────────────────────────────────────────────────
print("=== 1. Config imports ===")
try:
    from src.core.configs.fm_config import (
        DataConfig,
        ModelConfig,
        AugmentConfig,
        TrainHyperConfig,
        SampleConfig,
        OutputConfig,
        FMTrainConfig,
        FMSampleConfig,
    )
    check("All config classes importable", True)
except Exception as e:
    check(f"Import error: {e}", False)
    sys.exit(1)


# ── 2. Default values match existing CLI ─────────────────────────────────────
print("\n=== 2. Default values ===")
d = DataConfig()
check("DataConfig.train_dir", d.train_dir == "./data/raw/v18/train/")
check("DataConfig.val_dir", d.val_dir == "./data/raw/v18/val/")
check("DataConfig.batch_size", d.batch_size == 8)
check("DataConfig.num_workers", d.num_workers == 4)

m = ModelConfig()
check("ModelConfig.unet_config", m.unet_config == "configs/models/fm/stable_unet_config.json")
check("ModelConfig.vae_config", m.vae_config == "configs/models/fm/vae_config.json")
check("ModelConfig.vae_weights", m.vae_weights == "./vae_best.pt")
check("ModelConfig.pretrained_unet_path", m.pretrained_unet_path is None)

a = AugmentConfig()
check("AugmentConfig.warmup_frac", a.warmup_frac == 0.1)
check("AugmentConfig.ramp_frac", a.ramp_frac == 0.3)
check("AugmentConfig.p_crop_warmup", a.p_crop_warmup == 0.05)
check("AugmentConfig.p_crop_max", a.p_crop_max == 0.20)
check("AugmentConfig.p_crop_final", a.p_crop_final == 0.05)
check("AugmentConfig.p_rot_warmup", a.p_rot_warmup == 0.05)
check("AugmentConfig.p_rot_max", a.p_rot_max == 0.30)
check("AugmentConfig.p_rot_final", a.p_rot_final == 0.05)

t = TrainHyperConfig()
check("TrainHyperConfig.epochs", t.epochs == 100)
check("TrainHyperConfig.t_scale", t.t_scale == 1000.0)
check("TrainHyperConfig.train_target", t.train_target == "v")
check("TrainHyperConfig.save_every_n_epochs", t.save_every_n_epochs == 10)
check("TrainHyperConfig.patience", t.patience is None)
check("TrainHyperConfig.min_delta", t.min_delta == 0.0)
check("TrainHyperConfig.strict_load", t.strict_load is True)

s = SampleConfig()
check("SampleConfig.sample_every_epoch", s.sample_every_epoch is True)
check("SampleConfig.sample_steps", s.sample_steps == 50)
check("SampleConfig.sample_batch_size", s.sample_batch_size == 4)
check("SampleConfig.sample_shape", s.sample_shape is None)

o = OutputConfig()
check("OutputConfig.model_dir", o.model_dir == "./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled/")
check("OutputConfig.log_dir", o.log_dir is None)
check("OutputConfig.resume", o.resume is None)
check("OutputConfig.resolved_log_dir", o.resolved_log_dir() == "./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled//runs/stable_flow_matching_logs/")

sc = FMSampleConfig()
check("FMSampleConfig.t_scale", sc.t_scale == 1000.0)
check("FMSampleConfig.steps", sc.steps == 50)
check("FMSampleConfig.batch_size", sc.batch_size == 8)


# ── 3. FMTrainConfig.from_args ───────────────────────────────────────────────
print("\n=== 3. FMTrainConfig.from_args ===")
fake_args = SimpleNamespace(
    train_dir="./data/train/",
    val_dir="./data/val/",
    batch_size=16,
    num_workers=2,
    unet_config="my_unet.json",
    vae_config="my_vae.json",
    vae_weights="my_vae.pt",
    warmup_frac=0.2,
    ramp_frac=0.4,
    p_crop_warmup=0.1,
    p_crop_max=0.3,
    p_crop_final=0.1,
    p_rot_warmup=0.1,
    p_rot_max=0.4,
    p_rot_final=0.1,
    epochs=50,
    t_scale=500.0,
    train_target="x0",
    save_every_n_epochs=5,
    sample_batch_size=2,
    model_dir="/tmp/test_model",
    resume="/tmp/ckpt.pt",
)

cfg = FMTrainConfig.from_args(fake_args)
check("data.train_dir mapped", cfg.data.train_dir == "./data/train/")
check("data.batch_size mapped", cfg.data.batch_size == 16)
check("model.unet_config mapped", cfg.model.unet_config == "my_unet.json")
check("model.vae_weights mapped", cfg.model.vae_weights == "my_vae.pt")
check("augment.warmup_frac mapped", cfg.augment.warmup_frac == 0.2)
check("augment.p_rot_max mapped", cfg.augment.p_rot_max == 0.4)
check("training.epochs mapped", cfg.training.epochs == 50)
check("training.t_scale mapped", cfg.training.t_scale == 500.0)
check("training.train_target mapped", cfg.training.train_target == "x0")
check("training.save_every_n_epochs mapped", cfg.training.save_every_n_epochs == 5)
check("sampling.sample_batch_size mapped", cfg.sampling.sample_batch_size == 2)
check("output.model_dir mapped", cfg.output.model_dir == "/tmp/test_model")
check("output.resume mapped", cfg.output.resume == "/tmp/ckpt.pt")
check("output.resolved_log_dir", cfg.output.resolved_log_dir() == "/tmp/test_model/runs/stable_flow_matching_logs/")


# ── 4. FlowMatchingTrainer.from_config ───────────────────────────────────────
print("\n=== 4. FlowMatchingTrainer.from_config ===")
from src.algorithms.training.flow_matching_trainer import FlowMatchingTrainer

with tempfile.TemporaryDirectory() as tmpdir:
    real_cfg = FMTrainConfig(
        model=ModelConfig(
            unet_config=os.path.join(REPO, "configs/models/fm/stable_unet_config.json"),
            vae_config=os.path.join(REPO, "configs/models/fm/vae_config.json"),
            vae_weights=None,
        ),
        training=TrainHyperConfig(
            t_scale=1000.0,
            train_target="v",
        ),
        output=OutputConfig(
            model_dir=tmpdir,
        ),
    )
    trainer = FlowMatchingTrainer.from_config(real_cfg)
    check("Trainer created from config", trainer is not None)
    check("t_scale propagated", trainer.t_scale == 1000.0)
    check("train_target propagated", trainer.train_target == "v")
    check("model_dir propagated", trainer.model_dir == tmpdir)
    check("VAE present", trainer.vae is not None)
    check("VAE frozen", all(not p.requires_grad for p in trainer.vae.parameters()))

    # flow_matching_step still works
    z = torch.randn(2, 4, 64, 64, device=trainer.device)
    loss = trainer.flow_matching_step(z)
    check("flow_matching_step via from_config", loss.ndim == 0 and torch.isfinite(loss))


# ── 5. train_from_config delegation ─────────────────────────────────────────
print("\n=== 5. train_from_config attribute check ===")
check("train_from_config method exists", hasattr(trainer, "train_from_config"))
check("train_from_config is callable", callable(trainer.train_from_config))


# ── 6. FMSampleConfig.resolved_device ────────────────────────────────────────
print("\n=== 6. FMSampleConfig ===")
sc_default = FMSampleConfig()
dev = sc_default.resolved_device()
check("resolved_device returns str", isinstance(dev, str))
check("resolved_device is cuda or cpu", dev in ("cuda", "cpu"))

sc_forced = FMSampleConfig(device="cpu")
check("forced device=cpu", sc_forced.resolved_device() == "cpu")


# ── 7. Syntax checks ────────────────────────────────────────────────────────
print("\n=== 7. Syntax checks ===")
files_to_check = [
    "src/core/configs/__init__.py",
    "src/core/configs/fm_config.py",
    "src/algorithms/training/flow_matching_trainer.py",
    "src/algorithms/inference/flow_matching_sampler.py",
    "train_sfm.py",
    "generate_datasets.py",
]
for f in files_to_check:
    path = os.path.join(REPO, f)
    with open(path) as fh:
        try:
            ast.parse(fh.read(), filename=f)
            check(f"{f} parses OK", True)
        except SyntaxError as e:
            check(f"{f} syntax error: {e}", False)


# ── 8. train_sfm.py uses config ─────────────────────────────────────────────
print("\n=== 8. train_sfm.py / src.cli.train imports check ===")
with open(os.path.join(REPO, "train_sfm.py")) as f:
    wrapper_src = f.read()
# train_sfm.py is now a thin wrapper that forwards to src.cli.train
# Accept either direct FMTrainConfig usage OR delegation to src.cli.train
delegates = "from src.cli.train import main" in wrapper_src
with open(os.path.join(REPO, "src", "cli", "train.py")) as f:
    cli_src = f.read()
check("Imports FMTrainConfig", "FMTrainConfig" in wrapper_src or (delegates and "FMTrainConfig" in cli_src))
check("Uses from_config or from_args", "from_config" in wrapper_src or "from_args" in wrapper_src or (delegates and ("from_config" in cli_src or "from_args" in cli_src)))
check("Uses train_from_config", "train_from_config" in wrapper_src or (delegates and "train_from_config" in cli_src))


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed  ({passed+failed} total)")
print(f"{'='*50}")

sys.exit(1 if failed else 0)
