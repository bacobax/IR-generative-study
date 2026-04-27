"""Modular CLI entrypoint for unconditional latent Stable Diffusion training."""

from __future__ import annotations

import os
from typing import Optional

from src.core.diffusers_compat import disable_diffusers_optional_scipy

disable_diffusers_optional_scipy()

from src.core.configs.sd_uncond_config import build_parser, parse_args
from src.core.data import collate_layout_batch
from src.core.data.datasets import AnnotationLayoutDataset
from src.core.data.training_data import (
    apply_dataset_subset,
    build_non_layout_dataloaders,
    resolve_training_data,
)
from src.core.normalization import norm_to_display as from_norm_to_display
from src.core.registry import REGISTRIES
from src.core.data.transforms import ScheduledHorizontalFlip, save_transform_examples

# Ensure default components are registered.
import src.models.fm_unet  # noqa: F401
import src.algorithms.training.unconditional_sd_trainer  # noqa: F401
import src.algorithms.inference.unconditional_sd_sampler  # noqa: F401


def run_training(cfg) -> None:
    """Execute unconditional latent SD training from a structured config."""
    total_epochs = cfg.training.epochs
    resolved_data = resolve_training_data(cfg.data)
    layout_enabled = (
        bool(getattr(cfg.layout_conditioning, "enabled", False))
        and str(getattr(cfg.layout_conditioning, "variant", "")) == "regiondiff_v1"
    )

    if layout_enabled:
        if resolved_data.train_annotations_path is None or resolved_data.val_annotations_path is None:
            raise ValueError(
                "RegionDiff layout-conditioned unconditional SD requires split-specific COCO annotations."
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
        train_dataset = apply_dataset_subset(
            train_base_dataset,
            cfg.data.max_train_samples,
            cfg.data.subset_strategy,
        )
        eval_dataset = apply_dataset_subset(
            eval_base_dataset,
            cfg.data.max_val_samples,
            cfg.data.subset_strategy,
        )
        from torch.utils.data import DataLoader

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
        train_base_for_examples = train_base_dataset
    else:
        non_layout = build_non_layout_dataloaders(
            data_config=cfg.data,
            augment_config=cfg.augment,
            curriculum_config=type("_NullCurriculum", (), {"enabled": False})(),
            total_epochs=total_epochs,
            resolved_data=resolved_data,
        )
        train_loader = non_layout.train_loader
        eval_loader = non_layout.eval_loader
        use_annotation_ds = non_layout.use_annotation_ds
        train_base_for_examples = non_layout.train_base_dataset

    TrainerCls = REGISTRIES.trainer.get(cfg.trainer_name)
    trainer = TrainerCls.from_config(cfg, from_norm_to_display=from_norm_to_display)

    if cfg.output.resume is None and not use_annotation_ds:
        save_transform_examples(
            train_base_for_examples,
            os.path.join(cfg.output.model_dir, "transform_examples"),
        )

    trainer.train_from_config(cfg, train_loader, eval_loader)


def main(argv: Optional[list] = None) -> None:
    """Parse CLI flags and launch unconditional latent SD training."""
    cfg = parse_args(argv)
    run_training(cfg)


if __name__ == "__main__":
    main()
