"""Modular CLI entrypoint for ControlNet flow-matching training (stage 2).

This module is the **source of truth** for launching ControlNet training.
The root-level ``train_controlnet.py`` is a thin compatibility wrapper that
forwards to :func:`main` here.

Uses the personalized :class:`ControlNetTrainer` from
``src.algorithms.training.controlnet_trainer`` — a self-contained trainer
that follows the ControlNet architecture from the paper
(`Zhang et al., 2023 <https://arxiv.org/abs/2302.05543>`_)
adapted for ``diffusers.UNet2DModel`` flow-matching pipelines.

Configuration uses a 3-layer merge (like ``src.cli.train``):
    1. Dataclass defaults (``CNTrainConfig``)
    2. YAML config file (``--config``)
    3. CLI flag overrides

Usage::

    python -m src.cli.train_controlnet \\
        --config configs/controlnet/train/default.yaml \\
        --epochs 50

    # or via the legacy wrapper:
    python train_controlnet.py --config configs/controlnet/train/default.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.core.data.datasets import BBoxConditioningDataset
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.controlnet_config import CNTrainConfig
from src.models.controlnet import ControlNetModel, save_controlnet_config
from src.models.fm_unet import load_unet_config, build_fm_unet_from_config
from src.models.vae import (
    load_vae_config,
    build_vae_from_config,
    load_vae_weights,
    freeze_vae,
)
from src.algorithms.training.controlnet_trainer import ControlNetTrainer


# ---------------------------------------------------------------------------
# Helpers: load stage-1 UNet + VAE from a pipeline folder
# ---------------------------------------------------------------------------

def _pick_latest(folder: str, prefix: str, suffix: str = ".pt"):
    """Pick the highest-numbered checkpoint matching *prefix* in *folder*."""
    if not os.path.isdir(folder):
        return None
    best_i, best_path = None, None
    for fn in os.listdir(folder):
        if not (fn.startswith(prefix) and fn.endswith(suffix)):
            continue
        mid = fn[len(prefix):-len(suffix)]
        try:
            i = int(mid)
        except ValueError:
            continue
        if best_i is None or i > best_i:
            best_i = i
            best_path = os.path.join(folder, fn)
    return best_path


def _load_stage1_unet(pipeline_dir: str, device: str):
    """Load and freeze the stage-1 UNet from a pipeline folder."""
    unet_dir = os.path.join(pipeline_dir, "UNET")
    cfg_path = os.path.join(unet_dir, "config.json")
    unet_cfg = load_unet_config(cfg_path)
    unet = build_fm_unet_from_config(unet_cfg, device=device)

    # Load weights (prefer best, fall back to latest epoch).
    w_path = os.path.join(unet_dir, "unet_fm_best.pt")
    if not os.path.isfile(w_path):
        w_path = _pick_latest(unet_dir, "unet_fm_epoch_")
    if w_path is None or not os.path.isfile(w_path):
        raise FileNotFoundError(f"No UNET weights found in {unet_dir}")

    unet.load_state_dict(torch.load(w_path, map_location=device))
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    print(f"[Stage-1] Loaded UNet from {w_path}")
    return unet


def _load_stage1_vae(pipeline_dir: str, device: str, weights_override=None):
    """Load and freeze the stage-1 VAE from a pipeline folder."""
    vae_dir = os.path.join(pipeline_dir, "VAE")
    cfg_path = os.path.join(vae_dir, "config.json")
    vae_cfg = load_vae_config(cfg_path)
    vae = build_vae_from_config(vae_cfg, device=device)

    # Load weights (prefer best, fall back to latest epoch).
    w_path = os.path.join(vae_dir, "vae_best.pt")
    if not os.path.isfile(w_path):
        w_path = _pick_latest(vae_dir, "vae_epoch_")
    if w_path is None or not os.path.isfile(w_path):
        raise FileNotFoundError(f"No VAE weights found in {vae_dir}")

    load_vae_weights(vae, w_path, map_location=device)
    print(f"[Stage-1] Loaded VAE from {w_path}")

    if weights_override is not None:
        load_vae_weights(vae, weights_override, map_location=device)
        print(f"[Stage-1] VAE weights overridden from {weights_override}")

    freeze_vae(vae)
    return vae


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ControlNet training."""
    p = argparse.ArgumentParser(
        description="ControlNet Flow Matching Training (stage 2)"
    )

    # Config file (optional)
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file. CLI flags override config values.")

    # Data
    p.add_argument("--train_dir", type=str, default="./data/raw/v18/train/")
    p.add_argument("--val_dir", type=str, default="./data/raw/v18/val/")
    p.add_argument("--train_annotations", type=str, default="./data/raw/v18/train/annotations.json")
    p.add_argument("--val_annotations", type=str, default="./data/raw/v18/val/annotations.json")

    # Stage-1 pipeline (frozen UNet + VAE)
    p.add_argument("--stage1_pipeline_dir", type=str, default=None,
                    help="Path to the stage-1 pipeline folder (contains UNET/ and VAE/)")
    p.add_argument("--vae_weights_override", type=str, default=None,
                    help="Explicit path to VAE weights (overrides auto-detection)")

    # ControlNet architecture
    p.add_argument("--conditioning_channels", type=int, default=1)
    p.add_argument("--conditioning_downscale_factor", type=int, default=4)

    # Output
    p.add_argument("--model_dir", type=str, default="./controlnet_runs/bbox_controlnet/")

    # Training params
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--conditioning_scale", type=float, default=1.0)
    p.add_argument("--conditioning_dropout", type=float, default=0.1)
    p.add_argument("--save_every_n_epochs", type=int, default=10)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--t_scale", type=float, default=1000)
    p.add_argument("--patience", type=int, default=None)

    # Resume
    p.add_argument("--resume", type=str, default=None,
                    help="Path to ControlNet checkpoint to resume from")

    return p


# Mapping from flat CLI argument names -> dotted dataclass paths.
_FLAT_TO_NESTED = {
    # Data
    "train_dir":                    "data.train_dir",
    "val_dir":                      "data.val_dir",
    "train_annotations":            "data.train_annotations",
    "val_annotations":              "data.val_annotations",
    "batch_size":                   "data.batch_size",
    "num_workers":                  "data.num_workers",
    # Stage-1
    "stage1_pipeline_dir":          "stage1.stage1_pipeline_dir",
    "vae_weights_override":         "stage1.vae_weights_override",
    # ControlNet architecture
    "conditioning_channels":        "controlnet.conditioning_channels",
    "conditioning_downscale_factor": "controlnet.conditioning_downscale_factor",
    # Training
    "epochs":                       "training.epochs",
    "lr":                           "training.lr",
    "t_scale":                      "training.t_scale",
    "conditioning_scale":           "training.conditioning_scale",
    "conditioning_dropout":         "training.conditioning_dropout",
    "save_every_n_epochs":          "training.save_every_n_epochs",
    "patience":                     "training.patience",
    # Sampling
    "sample_steps":                 "sampling.sample_steps",
    # Output
    "model_dir":                    "output.model_dir",
    "resume":                       "output.resume",
}


# ═══════════════════════════════════════════════════════════════════════════
# Training pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_training(cfg: CNTrainConfig) -> None:
    """Execute ControlNet training from a structured config."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- datasets --------------------------------------------------------
    train_dataset = BBoxConditioningDataset(
        cfg.data.train_dir,
        cfg.data.train_annotations,
        conditioning_dropout=cfg.training.conditioning_dropout,
    )
    val_dataset = BBoxConditioningDataset(
        cfg.data.val_dir,
        cfg.data.val_annotations,
        conditioning_dropout=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # --- load stage-1 models (frozen) ------------------------------------
    if cfg.stage1.stage1_pipeline_dir is None:
        raise ValueError(
            "stage1.stage1_pipeline_dir (--stage1_pipeline_dir) is required. "
            "Point it to a trained stage-1 pipeline folder."
        )

    unet = _load_stage1_unet(cfg.stage1.stage1_pipeline_dir, device)
    vae = _load_stage1_vae(
        cfg.stage1.stage1_pipeline_dir, device,
        weights_override=cfg.stage1.vae_weights_override,
    )

    # --- build ControlNet from frozen UNet --------------------------------
    controlnet = ControlNetModel(
        unet,
        conditioning_channels=cfg.controlnet.conditioning_channels,
        conditioning_downscale_factor=cfg.controlnet.conditioning_downscale_factor,
    ).to(device)

    # Save ControlNet config.
    cn_dir = os.path.join(cfg.output.model_dir, "CONTROLNET")
    os.makedirs(cn_dir, exist_ok=True)
    save_controlnet_config(
        {
            "conditioning_channels": cfg.controlnet.conditioning_channels,
            "conditioning_downscale_factor": cfg.controlnet.conditioning_downscale_factor,
        },
        os.path.join(cn_dir, "config.json"),
    )

    n_cn = sum(p.numel() for p in controlnet.parameters())
    n_unet = sum(p.numel() for p in unet.parameters())
    print(f"[ControlNet] Trainable params : {n_cn:,}")
    print(f"[UNet]       Frozen params    : {n_unet:,}")

    # --- build trainer ---------------------------------------------------
    trainer = ControlNetTrainer(
        unet=unet,
        controlnet=controlnet,
        device=device,
        t_scale=cfg.training.t_scale,
        model_dir=cfg.output.model_dir,
        vae=vae,
    )

    # --- train -----------------------------------------------------------
    trainer.train(
        dataloader=train_loader,
        epochs=cfg.training.epochs,
        eval_dataloader=val_loader,
        lr=cfg.training.lr,
        conditioning_scale=cfg.training.conditioning_scale,
        log_dir=cfg.output.resolved_log_dir(),
        sample_steps=cfg.sampling.sample_steps,
        patience=cfg.training.patience,
        save_every_n_epochs=cfg.training.save_every_n_epochs,
        resume_from_checkpoint=cfg.output.resume,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[list] = None) -> None:
    """Parse CLI flags and launch ControlNet training."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = merge_config_and_cli(
        CNTrainConfig, args.config, parser, args,
        flat_to_nested=_FLAT_TO_NESTED,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
