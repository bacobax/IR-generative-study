"""CLI entrypoint for full curriculum meta FM training.

Usage::

    python -m src.cli.train_meta_fm --config configs/fm/train/presets/meta_curriculum_cfg.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.core.configs.meta_fm_config import MetaFMTrainConfig
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.fm_config import CountFilterConfig
from src.core.data.annotation_dataset import AnnotationFMDataset
from src.core.data.transforms import ScheduledAugment256
from src.core.conditions import ConditionSplit, save_split
from src.core.registry import REGISTRIES

# Ensure components are registered
import src.models.moe_text_unet  # noqa: F401
import src.algorithms.training.meta_fm_trainer  # noqa: F401
import src.algorithms.inference.cfg_flow_matching_sampler  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meta FM curriculum training")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file. CLI overrides config values.")
    return parser


_FLAT_TO_NESTED: dict = {}


def _text_collate_fn(batch):
    images = torch.stack([b["pixel_values"] for b in batch])
    texts = [b["text"] for b in batch]
    return {"pixel_values": images, "text": texts}


def run_training(cfg: MetaFMTrainConfig) -> None:
    if cfg.data.annotations_path is None:
        raise ValueError(
            "Meta FM curriculum training requires data.annotations_path "
            "pointing to a COCO-format annotations.json."
        )

    # Build transforms
    total_epochs = cfg.phase_a.epochs + len(cfg.condition_split.incremental) * (
        cfg.phase_b.epochs + cfg.phase_c.epochs
    )

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

    if cfg.curriculum.enabled:
        cfg.curriculum.total_epochs = total_epochs

    # Explicit condition split (counts are person counts)
    split = ConditionSplit(
        base=list(cfg.condition_split.base),
        incremental=list(cfg.condition_split.incremental),
        test=list(cfg.condition_split.test),
    )
    split.validate()

    def _make_count_filter(allowed_counts):
        if not allowed_counts:
            return None
        return CountFilterConfig(
            seen_counts=list(allowed_counts),
            unseen_counts=None,
            max_crop_retries=cfg.count_filter.max_crop_retries,
        )

    def _build_dataset(allowed_counts):
        return AnnotationFMDataset(
            root_dir=cfg.data.train_dir,
            annotations_path=cfg.data.annotations_path,
            text_mode=True,
            curriculum=cfg.curriculum if cfg.curriculum.enabled else None,
            count_filter=_make_count_filter(allowed_counts),
            transform=train_tf,
        )

    base_ds = _build_dataset(split.base)
    base_loader = DataLoader(
        base_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=_text_collate_fn,
    )

    incremental_loaders = []
    for cond in split.incremental:
        cond_ds = _build_dataset([cond])
        loader = DataLoader(
            cond_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=_text_collate_fn,
        )
        incremental_loaders.append((cond, loader))

    # Build test dataset (for inspection only)
    _ = _build_dataset(split.test)

    # Save split for inspection
    split_path = os.path.join(cfg.output.model_dir, "condition_split.json")
    save_split(split_path, split)

    # Trainer
    TrainerCls = REGISTRIES.trainer.get(cfg.trainer_name)
    trainer = TrainerCls.from_config(cfg)

    router_weights_dir = os.path.join(cfg.output.model_dir, "analysis", "router_weights")
    eval_output_dir = cfg.evaluation.output_dir if cfg.evaluation.enabled else None

    trainer.train_curriculum(
        base_dataloader=base_loader,
        incremental_loaders=incremental_loaders,
        test_conditions=list(split.test),
        phase_a_epochs=cfg.phase_a.epochs,
        phase_b_epochs=cfg.phase_b.epochs,
        phase_c_epochs=cfg.phase_c.epochs,
        phase_a_lr=cfg.phase_a.lr,
        phase_b_lr=cfg.phase_b.lr,
        phase_c_lr=cfg.phase_c.lr,
        phase_c_unfreeze_policy=cfg.phase_c.unfreeze_unet_policy,
        phase_c_router_trainable=cfg.phase_c.router_trainable,
        phase_c_router_lr_scale=cfg.phase_c.router_lr_scale,
        phase_c_replay_every=cfg.phase_c.replay_every,
        log_router_weights=True,
        router_weights_dir=router_weights_dir,
        eval_output_dir=eval_output_dir,
        eval_steps=cfg.evaluation.steps,
        eval_guidance_scale=cfg.evaluation.guidance_scale,
        eval_samples_per_condition=cfg.evaluation.samples_per_condition,
    )


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = merge_config_and_cli(
        MetaFMTrainConfig, args.config, parser, args,
        flat_to_nested=_FLAT_TO_NESTED,
    )

    run_training(cfg)


if __name__ == "__main__":
    main()
