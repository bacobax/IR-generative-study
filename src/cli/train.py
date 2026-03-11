"""Modular CLI entrypoint for flow-matching training.

This module is the **source of truth** for launching FM training.
The root-level ``train_sfm.py`` is now a thin convenience wrapper that
simply forwards to :func:`main` here.

Usage::

    python -m src.cli.train --train_dir ./v18/train/ --epochs 50
    # or via the legacy wrapper:
    python train_sfm.py --train_dir ./v18/train/ --epochs 50
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.core.configs.fm_config import FMTrainConfig
from src.core.configs.config_loader import merge_config_and_cli
from src.core.normalization import norm_to_display as from_norm_to_display
from src.core.data.datasets import NPYImageDataset
from src.core.data.transforms import ScheduledAugment256, save_transform_examples
from src.core.registry import REGISTRIES

# Ensure default components are registered
import src.models.fm_unet  # noqa: F401 — registers model_builder
import src.algorithms.training.flow_matching_trainer  # noqa: F401 — registers trainer
import src.algorithms.inference.flow_matching_sampler  # noqa: F401 — registers sampler


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for FM training.

    Returns the parser (not the parsed args) so callers can extend it or
    inspect it without triggering ``sys.argv`` parsing.
    """
    parser = argparse.ArgumentParser(description="Stable Flow Matching Training")

    # Config file (optional)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI flags override config values.")

    # Data paths
    parser.add_argument("--train_dir", type=str, default="./data/raw/v18/train/",
                        help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="./data/raw/v18/val/",
                        help="Path to validation data")

    # Model configs
    parser.add_argument("--unet_config", type=str, default="configs/models/fm/stable_unet_config.json",
                        help="UNet config JSON")
    parser.add_argument("--vae_config", type=str, default="configs/models/fm/vae_config.json",
                        help="VAE config JSON")
    parser.add_argument("--vae_weights", type=str, default="./vae_best.pt",
                        help="Pretrained VAE weights")

    # Output
    parser.add_argument("--model_dir", type=str,
                        default="./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled/",
                        help="Model output directory")

    # Training params
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_every_n_epochs", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--sample_batch_size", type=int, default=4,
                        help="Batch size for sampling")
    parser.add_argument("--t_scale", type=float, default=1000,
                        help="Time scale for UNet")

    # Augmentation schedule
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--ramp_frac", type=float, default=0.3)
    parser.add_argument("--p_crop_warmup", type=float, default=0.05)
    parser.add_argument("--p_crop_max", type=float, default=0.20)
    parser.add_argument("--p_crop_final", type=float, default=0.05)
    parser.add_argument("--p_rot_warmup", type=float, default=0.05)
    parser.add_argument("--p_rot_max", type=float, default=0.30)
    parser.add_argument("--p_rot_final", type=float, default=0.05)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Prediction target
    parser.add_argument("--train-target", type=str, default="v",
                        choices=["v", "x0"],
                        help="Prediction target: 'v' (velocity) or 'x0' (clean sample)")

    return parser


# Mapping from flat CLI argument names → dotted dataclass paths.
# Used by merge_config_and_cli to place CLI overrides in the right
# nested sub-config.
_FLAT_TO_NESTED = {
    # Data
    "train_dir":           "data.train_dir",
    "val_dir":             "data.val_dir",
    "batch_size":          "data.batch_size",
    "num_workers":         "data.num_workers",
    # Model
    "unet_config":         "model.unet_config",
    "vae_config":          "model.vae_config",
    "vae_weights":         "model.vae_weights",
    # Output
    "model_dir":           "output.model_dir",
    "resume":              "output.resume",
    # Training hyper-params
    "epochs":              "training.epochs",
    "t_scale":             "training.t_scale",
    "train_target":        "training.train_target",
    "save_every_n_epochs": "training.save_every_n_epochs",
    # Augmentation
    "warmup_frac":         "augment.warmup_frac",
    "ramp_frac":           "augment.ramp_frac",
    "p_crop_warmup":       "augment.p_crop_warmup",
    "p_crop_max":          "augment.p_crop_max",
    "p_crop_final":        "augment.p_crop_final",
    "p_rot_warmup":        "augment.p_rot_warmup",
    "p_rot_max":           "augment.p_rot_max",
    "p_rot_final":         "augment.p_rot_final",
    # Sampling
    "sample_batch_size":   "sampling.sample_batch_size",
}


# ═══════════════════════════════════════════════════════════════════════════
# Training pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_training(cfg: FMTrainConfig) -> None:
    """Execute FM training from a structured config.

    This function encapsulates the full pipeline:
    dataset construction → DataLoader creation → trainer instantiation
    (via registry) → training loop.
    """
    total_epochs = cfg.training.epochs

    # ── Augmentation transforms ──
    aug_kwargs = dict(
        total_epochs=total_epochs,
        warmup_frac=cfg.augment.warmup_frac,
        ramp_frac=cfg.augment.ramp_frac,
        p_crop_warmup=cfg.augment.p_crop_warmup,
        p_crop_max=cfg.augment.p_crop_max,
        p_crop_final=cfg.augment.p_crop_final,
        p_rot_warmup=cfg.augment.p_rot_warmup,
        p_rot_max=cfg.augment.p_rot_max,
        p_rot_final=cfg.augment.p_rot_final,
    )
    train_transform = ScheduledAugment256(**aug_kwargs)
    eval_transform = ScheduledAugment256(**aug_kwargs)

    # ── Datasets / loaders ──
    train_dataset = NPYImageDataset(root_dir=cfg.data.train_dir, transform=train_transform)
    eval_dataset = NPYImageDataset(root_dir=cfg.data.val_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # ── Resolve trainer class through registry ──
    TrainerCls = REGISTRIES.trainer.get(cfg.trainer_name)
    trainer = TrainerCls.from_config(cfg, from_norm_to_display=from_norm_to_display)

    # ── Save transform examples for fresh runs ──
    if cfg.output.resume is None:
        save_transform_examples(
            train_dataset,
            os.path.join(cfg.output.model_dir, "transform_examples"),
        )

    # ── Train ──
    trainer.train_from_config(cfg, train_loader, eval_loader)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[list] = None) -> None:
    """Parse CLI flags and launch FM training.

    Parameters
    ----------
    argv : list[str], optional
        Explicit argument list (for testing). ``None`` → ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = merge_config_and_cli(
        FMTrainConfig, args.config, parser, args,
        flat_to_nested=_FLAT_TO_NESTED,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
