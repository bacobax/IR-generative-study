#!/usr/bin/env python3
"""Smoke-check: verify the Trainer / Sampler split (Phase 6).

Checks:
  1. Imports of new modules succeed.
  2. FlowMatchingSampler instantiation (pixel & stable).
  3. FlowMatchingTrainer instantiation (pixel & stable).
  4. Trainer.flow_matching_step produces a scalar loss.
  5. Sampler.sample_euler produces correct-shape output.
  6. train_sfm.py no longer imports from the old pipeline.
  7. Old pipeline classes still importable (legacy / ControlNet).
  8. Syntax check of all touched files.
"""

import ast
import os
import sys
import tempfile

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


device = "cpu"

# ── 1. Imports ───────────────────────────────────────────────────────────────
print("=== 1. Module imports ===")
try:
    from src.models.fm_unet import load_unet_config, build_fm_unet_from_config
    check("src.models.fm_unet imports", True)
except Exception as e:
    check(f"src.models.fm_unet imports: {e}", False)

try:
    from src.models.vae import load_vae_config, build_vae_from_config, freeze_vae
    check("src.models.vae imports", True)
except Exception as e:
    check(f"src.models.vae imports: {e}", False)

try:
    from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler
    check("FlowMatchingSampler import", True)
except Exception as e:
    check(f"FlowMatchingSampler import: {e}", False)

try:
    from src.algorithms.training.flow_matching_trainer import FlowMatchingTrainer
    check("FlowMatchingTrainer import", True)
except Exception as e:
    check(f"FlowMatchingTrainer import: {e}", False)


# ── 2. Build models ─────────────────────────────────────────────────────────
print("\n=== 2. Build models ===")
unet_cfg = load_unet_config(os.path.join(REPO, "configs/models/fm/stable_unet_config.json"))
unet = build_fm_unet_from_config(unet_cfg, device=device)
check("UNet built", unet is not None)

vae_cfg = load_vae_config(os.path.join(REPO, "configs/models/fm/vae_config.json"))
vae = build_vae_from_config(vae_cfg, device=device)
check("VAE built", vae is not None)


# ── 3. FlowMatchingSampler (pixel-space) ────────────────────────────────────
print("\n=== 3. FlowMatchingSampler (pixel-space) ===")
pixel_unet_cfg = load_unet_config(os.path.join(REPO, "configs/models/fm/non_stable_unet_config.json"))
pixel_unet = build_fm_unet_from_config(pixel_unet_cfg, device=device)

sampler_px = FlowMatchingSampler(pixel_unet, device=device, t_scale=1000.0)
check("Pixel sampler created", sampler_px is not None)

with torch.no_grad():
    imgs_px = sampler_px.sample_euler(steps=5, batch_size=2, sample_shape=(1, 256, 256))
check("Pixel sample_euler shape", imgs_px.shape == (2, 1, 256, 256))


# ── 4. FlowMatchingSampler (stable / from_stable) ───────────────────────────
print("\n=== 4. FlowMatchingSampler (stable) ===")
sampler_st = FlowMatchingSampler.from_stable(unet, vae, device=device, t_scale=1000.0)
check("Stable sampler created", sampler_st is not None)

with torch.no_grad():
    imgs_st = sampler_st.sample_euler(steps=5, batch_size=2)
check("Stable sample_euler shape ok (4D)", imgs_st.ndim == 4)
check("Stable sample_euler returns latent C=4", imgs_st.shape[1] == 4)

# Decode latents to image space
with torch.no_grad():
    decoded = sampler_st.decode(imgs_st)
check("Stable decode returns C=1", decoded.shape[1] == 1)


# ── 5. FlowMatchingTrainer ──────────────────────────────────────────────────
print("\n=== 5. FlowMatchingTrainer ===")
with tempfile.TemporaryDirectory() as tmpdir:
    trainer = FlowMatchingTrainer(
        unet,
        device=device,
        t_scale=1000.0,
        train_target="v",
        model_dir=tmpdir,
        unet_config=unet_cfg,
        vae=vae,
        vae_config=vae_cfg,
    )
    check("Trainer created", trainer is not None)
    check("VAE frozen (requires_grad=False)", all(not p.requires_grad for p in trainer.vae.parameters()))

    # flow_matching_step
    z_latent = torch.randn(2, 4, 64, 64)
    loss = trainer.flow_matching_step(z_latent)
    check("flow_matching_step returns scalar", loss.ndim == 0)
    check("Loss is finite", torch.isfinite(loss).item())

    # encode_fm_input
    x_input = torch.randn(2, 1, 256, 256)
    z_enc = trainer.encode_fm_input(x_input)
    check("encode_fm_input produces 4-ch latent", z_enc.shape[1] == 4)

    # Configs saved
    trainer._ensure_dirs()
    trainer._save_configs()
    check("UNET dir created", os.path.isdir(os.path.join(tmpdir, "UNET")))
    check("VAE dir created", os.path.isdir(os.path.join(tmpdir, "VAE")))
    check("UNet config saved", os.path.isfile(os.path.join(tmpdir, "UNET", "config.json")))
    check("VAE config saved", os.path.isfile(os.path.join(tmpdir, "VAE", "config.json")))


# ── 6. train_sfm.py uses new modules ────────────────────────────────────────
print("\n=== 6. train_sfm.py imports ===")
train_sfm_path = os.path.join(REPO, "train_sfm.py")
with open(train_sfm_path) as f:
    src = f.read()
# train_sfm.py may be a thin wrapper forwarding to src.cli.train
delegates = "from src.cli.train import main" in src
cli_src = ""
if delegates:
    with open(os.path.join(REPO, "src", "cli", "train.py")) as f:
        cli_src = f.read()

check("No StableFlowMatchingPipeline import", "StableFlowMatchingPipeline" not in src)
check("No fm_src.pipelines import", "from fm_src.pipelines" not in src)
check("Imports FlowMatchingTrainer", "FlowMatchingTrainer" in src or (delegates and ("FlowMatchingTrainer" in cli_src or "flow_matching_trainer" in cli_src)))
check("Uses config-driven approach", "FMTrainConfig" in src or "load_unet_config" in src or (delegates and "FMTrainConfig" in cli_src))
check("Uses config-driven approach (vae)", "FMTrainConfig" in src or "load_vae_config" in src or (delegates and "FMTrainConfig" in cli_src))


# ── 7. Old pipeline still importable (ControlNet etc.) ──────────────────────
print("\n=== 7. Old pipeline backward-compat ===")
try:
    import sys as _sys
    from src.core.paths import legacy_code_root
    _sys.path.insert(0, str(legacy_code_root()))
    from fm_src.pipelines.flow_matching_pipeline import FlowMatchingPipeline, StableFlowMatchingPipeline
    check("FlowMatchingPipeline importable", True)
    check("StableFlowMatchingPipeline importable", True)
except Exception as e:
    check(f"Old pipeline import error: {e}", False)


# ── 8. Syntax checks ────────────────────────────────────────────────────────
print("\n=== 8. Syntax checks ===")
files_to_check = [
    "src/algorithms/inference/flow_matching_sampler.py",
    "src/algorithms/training/flow_matching_trainer.py",
    "src/models/vae.py",
    "src/models/fm_unet.py",
    "train_sfm.py",
    "archive/legacy_code/fm_src/pipelines/flow_matching_pipeline.py",
]
for f in files_to_check:
    path = os.path.join(REPO, f)
    with open(path) as fh:
        try:
            ast.parse(fh.read(), filename=f)
            check(f"{f} parses OK", True)
        except SyntaxError as e:
            check(f"{f} syntax error: {e}", False)


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  {passed} passed, {failed} failed  ({passed+failed} total)")
print(f"{'='*50}")

sys.exit(1 if failed else 0)
