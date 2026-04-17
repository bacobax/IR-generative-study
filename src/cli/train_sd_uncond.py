"""Modular CLI entrypoint for unconditional latent Stable Diffusion training."""

from __future__ import annotations

import os
from typing import Optional

from src.core.configs.sd_uncond_config import build_parser, parse_args
from src.core.data.training_data import build_non_layout_dataloaders, resolve_training_data
from src.core.normalization import norm_to_display as from_norm_to_display
from src.core.registry import REGISTRIES
from src.core.data.transforms import save_transform_examples

# Ensure default components are registered.
import src.models.fm_unet  # noqa: F401
import src.algorithms.training.unconditional_sd_trainer  # noqa: F401
import src.algorithms.inference.unconditional_sd_sampler  # noqa: F401


def run_training(cfg) -> None:
    """Execute unconditional latent SD training from a structured config."""
    total_epochs = cfg.training.epochs
    resolved_data = resolve_training_data(cfg.data)
    non_layout = build_non_layout_dataloaders(
        data_config=cfg.data,
        augment_config=cfg.augment,
        curriculum_config=type("_NullCurriculum", (), {"enabled": False})(),
        total_epochs=total_epochs,
        resolved_data=resolved_data,
    )

    TrainerCls = REGISTRIES.trainer.get(cfg.trainer_name)
    trainer = TrainerCls.from_config(cfg, from_norm_to_display=from_norm_to_display)

    if cfg.output.resume is None and not non_layout.use_annotation_ds:
        save_transform_examples(
            non_layout.train_base_dataset,
            os.path.join(cfg.output.model_dir, "transform_examples"),
        )

    trainer.train_from_config(cfg, non_layout.train_loader, non_layout.eval_loader)


def main(argv: Optional[list] = None) -> None:
    """Parse CLI flags and launch unconditional latent SD training."""
    cfg = parse_args(argv)
    run_training(cfg)


if __name__ == "__main__":
    main()
