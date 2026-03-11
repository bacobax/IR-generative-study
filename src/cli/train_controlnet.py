"""Modular CLI entrypoint for ControlNet flow-matching training (stage 2).

This module is the **source of truth** for launching ControlNet training.
The root-level ``train_controlnet.py`` is a thin compatibility wrapper that
forwards to :func:`main` here.

.. note::

   This module still imports ``ControlNetFlowMatchingPipeline`` from
   ``fm_src.pipelines`` as a transitional dependency.  A full extraction
   of ControlNet training into ``src/algorithms/`` is planned for a future
   phase.

Usage::

    python -m src.cli.train_controlnet \\
        --stage1_pipeline_dir ./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled \\
        --epochs 100

    # or via the legacy wrapper:
    python train_controlnet.py --stage1_pipeline_dir ...
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.core.data.datasets import BBoxConditioningDataset
from src.core.paths import legacy_code_root
from src.core.configs.config_loader import apply_yaml_defaults

# Transitional: ControlNet pipeline still lives in archived fm_src.
import sys as _sys
_sys.path.insert(0, str(legacy_code_root()))
from fm_src.pipelines.controlnet_flow_matching_pipeline import (  # noqa: E402
    ControlNetFlowMatchingPipeline,
)


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
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

    preliminary, _ = p.parse_known_args()
    apply_yaml_defaults(p, preliminary.config)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- datasets --------------------------------------------------------
    train_dataset = BBoxConditioningDataset(
        args.train_dir,
        args.train_annotations,
        conditioning_dropout=args.conditioning_dropout,
    )
    val_dataset = BBoxConditioningDataset(
        args.val_dir,
        args.val_annotations,
        conditioning_dropout=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- pipeline --------------------------------------------------------
    pipe = ControlNetFlowMatchingPipeline(
        device=device,
        t_scale=args.t_scale,
        model_dir=args.model_dir,
    )

    pipe.load_from_pipeline_folder_auto(
        args.stage1_pipeline_dir,
        set_eval=True,
    )

    if args.vae_weights_override is not None:
        pipe.load_vae_weights(args.vae_weights_override)

    pipe.build_controlnet(conditioning_channels=1)

    n_cn = sum(p.numel() for p in pipe.controlnet.parameters())
    n_unet = sum(p.numel() for p in pipe.unet.parameters())
    print(f"[ControlNet] Trainable params : {n_cn:,}")
    print(f"[UNet]       Frozen params    : {n_unet:,}")

    # --- train -----------------------------------------------------------
    pipe.train_controlnet_flow_matching(
        dataloader=train_loader,
        epochs=args.epochs,
        eval_dataloader=val_loader,
        lr=args.lr,
        conditioning_scale=args.conditioning_scale,
        log_dir=os.path.join(args.model_dir, "runs", "controlnet_logs"),
        sample_steps=args.sample_steps,
        patience=args.patience,
        save_every_n_epochs=args.save_every_n_epochs,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
