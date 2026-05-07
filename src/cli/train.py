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

from torch.utils.data import DataLoader

from src.core.configs.fm_config import FMTrainConfig
from src.core.configs.config_loader import merge_config_and_cli
from src.core.normalization import norm_to_display as from_norm_to_display
from src.core.data import collate_layout_batch
from src.core.data.datasets import AnnotationLayoutDataset
from src.core.data.training_data import (
    ResolvedTrainingData,
    apply_dataset_subset,
    build_non_layout_dataloaders,
    resolve_training_data,
)
from src.core.data.transforms import ScheduledHorizontalFlip, save_transform_examples
from src.core.diffusers_compat import disable_diffusers_optional_scipy
from src.core.registry import REGISTRIES

disable_diffusers_optional_scipy()

# Ensure default components are registered
import src.models.fm_unet  # noqa: F401 — registers model_builder
import src.algorithms.training.flow_matching_trainer  # noqa: F401 — registers trainer
import src.algorithms.training.layout_flow_matching_trainer  # noqa: F401 — registers trainer
import src.algorithms.inference.flow_matching_sampler  # noqa: F401 — registers sampler


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


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
    parser.add_argument("--vae_pretrained_model_name_or_path", type=str, default=None,
                        help="Optional diffusers model id/path used to load a pretrained VAE.")
    parser.add_argument("--vae_pretrained_subfolder", type=str, default="vae",
                        help="Subfolder containing the pretrained diffusers VAE.")
    parser.add_argument("--vae_revision", type=str, default=None,
                        help="Optional pretrained VAE revision.")
    parser.add_argument("--vae_variant", type=str, default=None,
                        help="Optional pretrained VAE variant.")

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

    # RegionDiff layout conditioning (opt-in; defaults preserve unconditional FM)
    parser.add_argument("--layout_conditioning_enabled", action="store_true",
                        help="Enable layout-conditioned FM data/model path.")
    parser.add_argument("--layout_conditioning_variant", type=str, default=None,
                        help="Layout conditioning variant, e.g. regiondiff_v1.")
    parser.add_argument("--active_region_resolutions", type=_csv_ints, default=None,
                        help="Comma-separated RegionDiff latent resolutions, e.g. 64,32,16.")
    parser.add_argument("--layout_token_dim", type=int, default=None)
    parser.add_argument("--bbox_fourier_dim", type=int, default=None)
    parser.add_argument("--same_class_position_slots", type=int, default=None)
    parser.add_argument("--use_background_token", type=_str2bool, default=None)
    parser.add_argument("--layout_train_mode", type=str, default=None)
    parser.add_argument("--partial_backbone_modules", nargs="+", default=None)
    parser.add_argument("--adapter_learning_rate", type=float, default=None)
    parser.add_argument("--backbone_learning_rate", type=float, default=None)
    parser.add_argument("--area_loss_enabled", type=_str2bool, default=None)
    parser.add_argument("--area_loss_alpha", type=float, default=None)
    parser.add_argument("--area_loss_background_weight", type=float, default=None)
    parser.add_argument("--area_loss_min_weight", type=float, default=None)
    parser.add_argument("--area_loss_max_weight", type=float, default=None)
    parser.add_argument("--area_x0_loss_weight", type=float, default=None)

    # RegionDiff attention distillation (opt-in; defaults preserve FM behavior)
    parser.add_argument("--distillation_enabled", type=_str2bool, default=None,
                        help="Enable RegionDiff attention-map distillation.")
    parser.add_argument("--distillation_teacher_checkpoint", type=str, default=None,
                        help="Path to SD1.5+RegionDiff teacher artifact/checkpoint.")
    parser.add_argument("--distillation_teacher_model_type", type=str, default=None,
                        choices=["sd15_regiondiff"],
                        help="Teacher model family.")
    parser.add_argument("--distillation_loss_type", type=str, default=None,
                        choices=["attention_kl", "attention_l2"],
                        help="Attention-map distillation loss.")
    parser.add_argument("--distillation_lambda_attn", type=float, default=None,
                        help="Weight for RegionDiff attention KD loss.")
    parser.add_argument("--distillation_warmup_epochs", type=int, default=None,
                        help="Epochs to wait before enabling attention KD.")
    parser.add_argument("--distillation_selected_categories", type=_csv_strings, default=None,
                        help="Comma-separated category names/ids to distill; empty means all.")
    parser.add_argument("--distillation_selected_region_layers", type=_csv_strings, default=None,
                        help="Comma-separated RegionDiff layer names/aliases/substrings.")
    parser.add_argument("--distillation_timestep_range", type=_csv_floats, default=None,
                        help="Comma-separated FM timestep range, e.g. 0.2,0.8.")
    parser.add_argument("--distillation_bbox_mask_only", type=_str2bool, default=None,
                        help="Restrict attention KD to object bbox regions.")

    # Augmentation schedule
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--ramp_frac", type=float, default=0.3)
    parser.add_argument("--p_crop_warmup", type=float, default=0.05)
    parser.add_argument("--p_crop_max", type=float, default=0.20)
    parser.add_argument("--p_crop_final", type=float, default=0.05)
    parser.add_argument("--p_rot_warmup", type=float, default=0.05)
    parser.add_argument("--p_rot_max", type=float, default=0.30)
    parser.add_argument("--p_rot_final", type=float, default=0.05)
    parser.add_argument("--p_hflip_warmup", type=float, default=0.0)
    parser.add_argument("--p_hflip_max", type=float, default=0.0)
    parser.add_argument("--p_hflip_final", type=float, default=0.0)

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
    "vae_pretrained_model_name_or_path": "model.vae_pretrained_model_name_or_path",
    "vae_pretrained_subfolder": "model.vae_pretrained_subfolder",
    "vae_revision":        "model.vae_revision",
    "vae_variant":         "model.vae_variant",
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
    "layout_conditioning_enabled": "layout_conditioning.enabled",
    "layout_conditioning_variant": "layout_conditioning.variant",
    "active_region_resolutions": "layout_conditioning.active_region_resolutions",
    "layout_token_dim": "layout_conditioning.layout_token_dim",
    "bbox_fourier_dim": "layout_conditioning.bbox_fourier_dim",
    "same_class_position_slots": "layout_conditioning.same_class_position_slots",
    "use_background_token": "layout_conditioning.use_background_token",
    "layout_train_mode": "layout_conditioning.train_mode",
    "partial_backbone_modules": "layout_conditioning.partial_backbone_modules",
    "adapter_learning_rate": "layout_conditioning.adapter_learning_rate",
    "backbone_learning_rate": "layout_conditioning.backbone_learning_rate",
    "area_loss_enabled": "layout_conditioning.area_loss_enabled",
    "area_loss_alpha": "layout_conditioning.area_loss_alpha",
    "area_loss_background_weight": "layout_conditioning.area_loss_background_weight",
    "area_loss_min_weight": "layout_conditioning.area_loss_min_weight",
    "area_loss_max_weight": "layout_conditioning.area_loss_max_weight",
    "area_x0_loss_weight": "layout_conditioning.area_x0_loss_weight",
    "distillation_enabled": "distillation.enabled",
    "distillation_teacher_checkpoint": "distillation.teacher_checkpoint",
    "distillation_teacher_model_type": "distillation.teacher_model_type",
    "distillation_loss_type": "distillation.loss_type",
    "distillation_lambda_attn": "distillation.lambda_attn",
    "distillation_warmup_epochs": "distillation.warmup_epochs",
    "distillation_selected_categories": "distillation.selected_categories",
    "distillation_selected_region_layers": "distillation.selected_region_layers",
    "distillation_timestep_range": "distillation.timestep_range",
    "distillation_bbox_mask_only": "distillation.bbox_mask_only",
    # Augmentation
    "warmup_frac":         "augment.warmup_frac",
    "ramp_frac":           "augment.ramp_frac",
    "p_crop_warmup":       "augment.p_crop_warmup",
    "p_crop_max":          "augment.p_crop_max",
    "p_crop_final":        "augment.p_crop_final",
    "p_rot_warmup":        "augment.p_rot_warmup",
    "p_rot_max":           "augment.p_rot_max",
    "p_rot_final":         "augment.p_rot_final",
    "p_hflip_warmup":      "augment.p_hflip_warmup",
    "p_hflip_max":         "augment.p_hflip_max",
    "p_hflip_final":       "augment.p_hflip_final",
    # Sampling
    "sample_batch_size":   "sampling.sample_batch_size",
    # Device
    "device":              "device",
}


def _resolve_training_data(cfg: FMTrainConfig) -> ResolvedTrainingData:
    """Resolve dataset directories, split annotations, and normalization mode."""
    return resolve_training_data(cfg.data)


def _apply_subset(dataset, max_samples: Optional[int], strategy: str):
    """Apply a deterministic debug subset without disturbing sample order."""
    return apply_dataset_subset(dataset, max_samples, strategy)


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
    layout_variant = str(getattr(cfg.layout_conditioning, "variant", "raster_v1"))

    # Propagate total_epochs into curriculum config
    if cfg.curriculum.enabled:
        cfg.curriculum.total_epochs = total_epochs

    if layout_enabled:
        if resolved_data.train_annotations_path is None or resolved_data.val_annotations_path is None:
            raise ValueError(
                "Layout-conditioned FM requires split-specific COCO annotations."
            )

        train_horizontal_flip = None
        if max(
            float(getattr(cfg.augment, "p_hflip_warmup", 0.0)),
            float(getattr(cfg.augment, "p_hflip_max", 0.0)),
            float(getattr(cfg.augment, "p_hflip_final", 0.0)),
        ) > 0.0:
            train_horizontal_flip = ScheduledHorizontalFlip(
                total_epochs=total_epochs,
                warmup_frac=cfg.augment.warmup_frac,
                ramp_frac=cfg.augment.ramp_frac,
                p_hflip_warmup=cfg.augment.p_hflip_warmup,
                p_hflip_max=cfg.augment.p_hflip_max,
                p_hflip_final=cfg.augment.p_hflip_final,
            )

        train_base_dataset = AnnotationLayoutDataset(
            root_dir=resolved_data.train_dir,
            annotations_path=resolved_data.train_annotations_path,
            image_size=cfg.data.image_size,
            normalization_mode=resolved_data.normalization_mode,
            include_label_names=True,
            horizontal_flip_schedule=train_horizontal_flip,
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
        if cfg.trainer_name is None and layout_variant != "regiondiff_v1":
            cfg.trainer_name = "layout_fm"
        elif cfg.trainer_name is None:
            cfg.trainer_name = "default_fm"

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
        non_layout = build_non_layout_dataloaders(
            data_config=cfg.data,
            augment_config=cfg.augment,
            curriculum_config=cfg.curriculum,
            total_epochs=total_epochs,
            resolved_data=resolved_data,
        )
        train_dataset = non_layout.train_dataset
        eval_dataset = non_layout.eval_dataset
        train_loader = non_layout.train_loader
        eval_loader = non_layout.eval_loader
        use_annotation_ds = non_layout.use_annotation_ds

    # ── Resolve trainer class through registry ──
    TrainerCls = REGISTRIES.trainer.get(cfg.trainer_name)
    trainer = TrainerCls.from_config(cfg, from_norm_to_display=from_norm_to_display)

    # ── Save transform examples for fresh runs ──
    if cfg.output.resume is None and not use_annotation_ds:
        save_transform_examples(
            getattr(non_layout, "train_base_dataset", train_dataset),
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
    import sys

    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(effective_argv)
    cfg = merge_config_and_cli(
        FMTrainConfig, args.config, parser, args,
        flat_to_nested=_FLAT_TO_NESTED,
        cli_argv=effective_argv,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
