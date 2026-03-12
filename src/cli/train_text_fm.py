"""CLI entrypoint for text-conditioned flow-matching training with CFG.

Usage::

    python -m src.cli.train_text_fm --config configs/fm/train/presets/text_cfg.yaml
    python -m src.cli.train_text_fm --config configs/fm/train/presets/text_cfg.yaml --epochs 10
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.core.configs.text_fm_config import TextFMTrainConfig
from src.core.configs.config_loader import merge_config_and_cli
from src.core.normalization import norm_to_display as from_norm_to_display
from src.core.data.datasets import TextImageDataset
from src.core.data.transforms import ScheduledAugment256
from src.core.registry import REGISTRIES

# Ensure components are registered
import src.models.fm_text_unet  # noqa: F401
import src.algorithms.training.text_fm_trainer  # noqa: F401
import src.algorithms.inference.cfg_flow_matching_sampler  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text-Conditioned FM Training with CFG")

    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file. CLI overrides config values.")

    # Data
    parser.add_argument("--train_dir", type=str, default="./data/raw/v18/train/")
    parser.add_argument("--val_dir", type=str, default="./data/raw/v18/val/")
    parser.add_argument("--text_annotations", type=str, default=None,
                        help="JSON mapping {stem: caption}")
    parser.add_argument("--fallback_text", type=str, default=None,
                        help="Fallback caption when no text found for an image")

    # Model
    parser.add_argument("--unet_config", type=str,
                        default="configs/models/fm/text_unet_config.json")
    parser.add_argument("--vae_config", type=str,
                        default="configs/models/fm/vae_config.json")
    parser.add_argument("--vae_weights", type=str, default=None)
    parser.add_argument("--pretrained_unet_path", type=str, default=None)

    # Conditioning
    parser.add_argument("--text_encoder", type=str,
                        default="openai/clip-vit-large-patch14")
    parser.add_argument("--max_text_length", type=int, default=77)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1,
                        help="Probability of dropping text conditioning (CFG)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--t_scale", type=float, default=1000.0)
    parser.add_argument("--train_target", type=str, default="v",
                        choices=["v", "x0"])
    parser.add_argument("--save_every_n_epochs", type=int, default=10)

    # Augmentation
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--ramp_frac", type=float, default=0.3)
    parser.add_argument("--p_crop_warmup", type=float, default=0.05)
    parser.add_argument("--p_crop_max", type=float, default=0.20)
    parser.add_argument("--p_crop_final", type=float, default=0.05)
    parser.add_argument("--p_rot_warmup", type=float, default=0.05)
    parser.add_argument("--p_rot_max", type=float, default=0.30)
    parser.add_argument("--p_rot_final", type=float, default=0.05)

    # Sampling
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=50)

    # Attention visualization
    parser.add_argument("--attn_vis_enabled", action="store_true",
                        help="Enable cross-attention heatmap visualization")
    parser.add_argument("--attn_vis_target_tokens", type=str, nargs="+",
                        default=["person", "people"],
                        help="Tokens to isolate in attention maps")
    parser.add_argument("--attn_vis_num_steps", type=int, default=8,
                        help="Number of timesteps to capture attention at")
    parser.add_argument("--attn_vis_layer_filter", type=str, default="all",
                        choices=["all", "up", "down", "mid"],
                        help="Which block group to capture")
    parser.add_argument("--attn_vis_head_reduction", type=str, default="mean",
                        choices=["mean", "none"],
                        help="How to aggregate across attention heads")
    parser.add_argument("--attn_vis_overlay", action="store_true", default=True,
                        help="Overlay heatmap on generated image")
    parser.add_argument("--attn_vis_colormap", type=str, default="jet")
    parser.add_argument("--attn_vis_per_layer", action="store_true",
                        help="Also log per-layer heatmaps")
    parser.add_argument("--attn_vis_guidance_scale", type=float, default=7.5)

    # Output
    parser.add_argument("--model_dir", type=str,
                        default="./artifacts/checkpoints/flow_matching/text_fm/")
    parser.add_argument("--resume", type=str, default=None)

    return parser


_FLAT_TO_NESTED = {
    "train_dir":           "data.train_dir",
    "val_dir":             "data.val_dir",
    "text_annotations":    "data.text_annotations",
    "fallback_text":       "data.fallback_text",
    "batch_size":          "data.batch_size",
    "num_workers":         "data.num_workers",
    "unet_config":         "model.unet_config",
    "vae_config":          "model.vae_config",
    "vae_weights":         "model.vae_weights",
    "pretrained_unet_path": "model.pretrained_unet_path",
    "text_encoder":        "conditioning.text_encoder",
    "max_text_length":     "conditioning.max_text_length",
    "cond_drop_prob":      "conditioning.cond_drop_prob",
    "epochs":              "training.epochs",
    "lr":                  "training.lr",
    "t_scale":             "training.t_scale",
    "train_target":        "training.train_target",
    "save_every_n_epochs": "training.save_every_n_epochs",
    "warmup_frac":         "augment.warmup_frac",
    "ramp_frac":           "augment.ramp_frac",
    "p_crop_warmup":       "augment.p_crop_warmup",
    "p_crop_max":          "augment.p_crop_max",
    "p_crop_final":        "augment.p_crop_final",
    "p_rot_warmup":        "augment.p_rot_warmup",
    "p_rot_max":           "augment.p_rot_max",
    "p_rot_final":         "augment.p_rot_final",
    "sample_batch_size":   "sampling.sample_batch_size",
    "sample_steps":        "sampling.sample_steps",
    "attn_vis_enabled":    "attention_vis.enabled",
    "attn_vis_target_tokens": "attention_vis.target_tokens",
    "attn_vis_num_steps":  "attention_vis.num_vis_steps",
    "attn_vis_layer_filter": "attention_vis.layer_filter",
    "attn_vis_head_reduction": "attention_vis.head_reduction",
    "attn_vis_overlay":    "attention_vis.overlay",
    "attn_vis_colormap":   "attention_vis.colormap",
    "attn_vis_per_layer":  "attention_vis.per_layer",
    "attn_vis_guidance_scale": "attention_vis.vis_guidance_scale",
    "model_dir":           "output.model_dir",
    "resume":              "output.resume",
}


# ═══════════════════════════════════════════════════════════════════════════
# Training pipeline
# ═══════════════════════════════════════════════════════════════════════════

def _text_collate_fn(batch):
    """Custom collate for TextImageDataset dicts."""
    images = torch.stack([b["pixel_values"] for b in batch])
    texts = [b["text"] for b in batch]
    return {"pixel_values": images, "text": texts}


def run_training(cfg: TextFMTrainConfig) -> None:
    """Execute text-conditioned FM training from a structured config."""
    total_epochs = cfg.training.epochs

    # Augmentation
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
    train_tf = ScheduledAugment256(**aug_kwargs)
    eval_tf = ScheduledAugment256(**aug_kwargs)

    # Datasets
    train_ds = TextImageDataset(
        root_dir=cfg.data.train_dir,
        text_annotations=cfg.data.text_annotations,
        fallback_text=cfg.data.fallback_text,
        transform=train_tf,
    )
    eval_ds = TextImageDataset(
        root_dir=cfg.data.val_dir,
        text_annotations=cfg.data.text_annotations,
        fallback_text=cfg.data.fallback_text,
        transform=eval_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=_text_collate_fn,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=_text_collate_fn,
    )

    # Trainer
    TrainerCls = REGISTRIES.trainer.get(cfg.trainer_name)
    trainer = TrainerCls.from_config(cfg, from_norm_to_display=from_norm_to_display)

    # Train
    trainer.train_from_config(cfg, train_loader, eval_loader)


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = merge_config_and_cli(
        TextFMTrainConfig, args.config, parser, args,
        flat_to_nested=_FLAT_TO_NESTED,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
