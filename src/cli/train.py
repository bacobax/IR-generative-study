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
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

from src.core.configs.fm_config import FMTrainConfig
from src.core.configs.config_loader import merge_config_and_cli
from src.core.data.dataset_targets import resolve_dataset_target
from src.core.normalization import (
    RAW_UINT16_PERCENTILE,
    norm_to_display as from_norm_to_display,
)
from src.core.data import collate_layout_batch
from src.core.data.datasets import AnnotationLayoutDataset, NPYImageDataset
from src.core.data.annotation_dataset import AnnotationFMDataset
from src.core.data.transforms import ScheduledAugment256, save_transform_examples
from src.core.registry import REGISTRIES

# Ensure default components are registered
import src.models.fm_unet  # noqa: F401 — registers model_builder
import src.algorithms.training.flow_matching_trainer  # noqa: F401 — registers trainer
import src.algorithms.training.layout_flow_matching_trainer  # noqa: F401 — registers trainer
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
    parser.add_argument("--dataset_id", type=str, default=None,
                        help="Named dataset target (overrides train_dir/val_dir when set)")
    parser.add_argument("--train_dir", type=str, default="./data/raw/v18/train/",
                        help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="./data/raw/v18/val/",
                        help="Path to validation data")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Square resize target. Must be a positive multiple of 32.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Deterministic limit on training samples (debug only).")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Deterministic limit on validation samples (debug only).")
    parser.add_argument("--subset_strategy", type=str, default="first_n",
                        help="Dataset subsetting policy for debug runs.")

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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_every_n_epochs", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Run validation every N epochs. Set <= 0 to disable.")
    parser.add_argument("--sample_batch_size", type=int, default=4,
                        help="Batch size for sampling")
    parser.add_argument("--t_scale", type=float, default=1000,
                        help="Time scale for UNet")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay.")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam/AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam/AdamW beta2.")
    parser.add_argument("--scheduler_name", type=str, default="warmup_cosine",
                        choices=["none", "warmup_cosine", "constant_with_warmup"],
                        help="Learning-rate scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio over total optimizer steps.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Final LR ratio for cosine schedules.")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay.")
    parser.add_argument("--ema_start_step", type=int, default=100,
                        help="Start EMA updates after this many steps.")
    parser.add_argument("--mixed_precision", type=str, default="auto",
                        choices=["auto", "bf16", "fp16", "no"],
                        help="Mixed precision mode.")
    parser.add_argument("--path_mode", type=str, default="independent",
                        choices=["independent", "minibatch_ot", "conditional_ot"],
                        help="Flow-matching path coupling mode.")
    parser.add_argument("--path_solver", type=str, default="hungarian",
                        choices=["hungarian"],
                        help="Assignment solver for OT-based paths.")
    parser.add_argument("--layout_cost_resolution", type=int, default=16,
                        help="Low-resolution raster size for conditional OT layout cost.")
    parser.add_argument("--condition_weight", type=float, default=1.0,
                        help="Weight of conditioning cost in conditional OT.")

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

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, cuda:0, cuda:1, etc.")

    # Annotations (for curriculum learning)
    parser.add_argument("--annotations_path", type=str, default=None,
                        help="Path to COCO-format annotations.json (enables curriculum)")

    return parser


# Mapping from flat CLI argument names → dotted dataclass paths.
# Used by merge_config_and_cli to place CLI overrides in the right
# nested sub-config.
_FLAT_TO_NESTED = {
    # Data
    "dataset_id":          "data.dataset_id",
    "train_dir":           "data.train_dir",
    "val_dir":             "data.val_dir",
    "annotations_path":    "data.annotations_path",
    "image_size":          "data.image_size",
    "batch_size":          "data.batch_size",
    "num_workers":         "data.num_workers",
    "max_train_samples":   "data.max_train_samples",
    "max_val_samples":     "data.max_val_samples",
    "subset_strategy":     "data.subset_strategy",
    # Model
    "unet_config":         "model.unet_config",
    "vae_config":          "model.vae_config",
    "vae_weights":         "model.vae_weights",
    # Output
    "model_dir":           "output.model_dir",
    "resume":              "output.resume",
    # Training hyper-params
    "epochs":              "training.epochs",
    "lr":                  "training.lr",
    "t_scale":             "training.t_scale",
    "train_target":        "training.train_target",
    "save_every_n_epochs": "training.save_every_n_epochs",
    "eval_every":          "training.eval_every",
    "max_grad_norm":       "training.max_grad_norm",
    # Optimizer / scheduler / EMA / precision / path
    "weight_decay":        "optimizer.weight_decay",
    "beta1":               "optimizer.beta1",
    "beta2":               "optimizer.beta2",
    "scheduler_name":      "scheduler.name",
    "warmup_ratio":        "scheduler.warmup_ratio",
    "min_lr_ratio":        "scheduler.min_lr_ratio",
    "ema_decay":           "ema.decay",
    "ema_start_step":      "ema.start_step",
    "mixed_precision":     "precision.mixed_precision",
    "path_mode":           "path.mode",
    "path_solver":         "path.solver",
    "layout_cost_resolution": "path.layout_cost_resolution",
    "condition_weight":    "path.condition_weight",
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
    # Device
    "device":              "device",
}


@dataclass(frozen=True)
class ResolvedTrainingData:
    """Resolved dataset paths used by the FM training pipeline."""

    train_dir: str
    val_dir: str
    train_annotations_path: Optional[str]
    val_annotations_path: Optional[str]
    normalization_mode: str


def _resolve_training_data(cfg: FMTrainConfig) -> ResolvedTrainingData:
    """Resolve dataset directories, split annotations, and normalization mode."""
    train_dir = cfg.data.train_dir
    val_dir = cfg.data.val_dir
    train_annotations_path = cfg.data.annotations_path
    val_annotations_path = cfg.data.annotations_path
    normalization_mode = RAW_UINT16_PERCENTILE

    if cfg.data.dataset_id is not None:
        target = resolve_dataset_target(cfg.data.dataset_id)
        train_dir = str(target.split_dir("train"))
        val_dir = str(target.split_dir("val"))
        normalization_mode = target.normalization_mode

        if train_annotations_path is None:
            train_annotations_path = str(target.annotations_path("train"))
            val_annotations_path = str(target.annotations_path("val"))

    return ResolvedTrainingData(
        train_dir=train_dir,
        val_dir=val_dir,
        train_annotations_path=train_annotations_path,
        val_annotations_path=val_annotations_path,
        normalization_mode=normalization_mode,
    )


def _apply_subset(dataset, max_samples: Optional[int], strategy: str):
    """Apply a deterministic debug subset without disturbing sample order."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    if strategy != "first_n":
        raise ValueError(
            f"Unsupported subset_strategy={strategy!r}. Only 'first_n' is supported."
        )
    return Subset(dataset, list(range(int(max_samples))))


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
    resolved_data = _resolve_training_data(cfg)
    layout_enabled = bool(cfg.layout_conditioning.enabled)

    # Propagate total_epochs into curriculum config
    if cfg.curriculum.enabled:
        cfg.curriculum.total_epochs = total_epochs

    if layout_enabled:
        if resolved_data.train_annotations_path is None or resolved_data.val_annotations_path is None:
            raise ValueError(
                "Layout-conditioned FM requires split-specific COCO annotations."
            )

        train_base_dataset = AnnotationLayoutDataset(
            root_dir=resolved_data.train_dir,
            annotations_path=resolved_data.train_annotations_path,
            image_size=cfg.data.image_size,
            normalization_mode=resolved_data.normalization_mode,
            include_label_names=True,
        )
        eval_base_dataset = AnnotationLayoutDataset(
            root_dir=resolved_data.val_dir,
            annotations_path=resolved_data.val_annotations_path,
            image_size=cfg.data.image_size,
            normalization_mode=resolved_data.normalization_mode,
            include_label_names=True,
        )

        cfg.layout_conditioning.num_classes = train_base_dataset.num_categories
        cfg.layout_conditioning.category_id_to_name = dict(train_base_dataset.category_id_to_name)
        if cfg.trainer_name is None:
            cfg.trainer_name = "layout_fm"

        train_dataset = _apply_subset(
            train_base_dataset,
            cfg.data.max_train_samples,
            cfg.data.subset_strategy,
        )
        eval_dataset = _apply_subset(
            eval_base_dataset,
            cfg.data.max_val_samples,
            cfg.data.subset_strategy,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_layout_batch,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_layout_batch,
        )
        use_annotation_ds = True
    else:
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
            image_size=cfg.data.image_size,
            normalization_mode=resolved_data.normalization_mode,
        )
        train_transform = ScheduledAugment256(**aug_kwargs)
        eval_transform = ScheduledAugment256(**aug_kwargs)

        # ── Datasets / loaders ──
        use_annotation_ds = (
            resolved_data.train_annotations_path is not None
            and cfg.curriculum.enabled
        )

        if use_annotation_ds:
            train_dataset = AnnotationFMDataset(
                root_dir=resolved_data.train_dir,
                annotations_path=resolved_data.train_annotations_path,
                text_mode=False,
                curriculum=cfg.curriculum,
                transform=train_transform,
            )
            eval_dataset = AnnotationFMDataset(
                root_dir=resolved_data.val_dir,
                annotations_path=resolved_data.val_annotations_path,
                text_mode=False,
                curriculum=None,
                transform=eval_transform,
            )
        else:
            train_dataset = NPYImageDataset(root_dir=resolved_data.train_dir, transform=train_transform)
            eval_dataset = NPYImageDataset(root_dir=resolved_data.val_dir, transform=eval_transform)

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
    if cfg.output.resume is None and not use_annotation_ds:
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
