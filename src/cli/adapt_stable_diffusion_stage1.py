"""Stage-1 Stable Diffusion 1.5 IR adaptation implementation.

This module is invoked by ``src.cli.adapt_stable_diffusion`` when
``--stage stage1`` is selected.

Usage::

    python -m src.cli.adapt_stable_diffusion --stage stage1 --config configs/sd/train/default.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

from src.core.diffusers_compat import disable_diffusers_optional_scipy

disable_diffusers_optional_scipy(lightweight_diffusers_imports=False)

from diffusers.utils import check_min_version

from src.algorithms.stable_diffusion.config import DEFAULT_PROMPT_TEXT, parse_args
from src.algorithms.stable_diffusion.data import create_dataloader
from src.algorithms.stable_diffusion.layout_data import (
    StableDiffusionLayoutDataset,
    collate_sd_layout_batch,
    resolve_layout_split,
)
from src.algorithms.stable_diffusion.models import (
    configure_trainable_components,
    get_canonical_output_dir,
    load_models,
)
from src.algorithms.stable_diffusion.training import Trainer
from src.algorithms.stable_diffusion.utils import setup_logging, save_model_card
from src.core.gpu_utils import get_least_used_cuda_gpu

# Require minimum diffusers version
check_min_version("0.37.0.dev0")


def main(argv=None):
    """Main SD IR adaptation training function."""
    # Parse arguments first (before accelerator init)
    print("Parsing arguments...")
    config = parse_args(argv)
    config.output_dir = get_canonical_output_dir(config)

    # Setup accelerator
    print("Initializing accelerator...")
    # Only select GPU for single-process training
    if "RANK" not in os.environ:
        device, smi_out = get_least_used_cuda_gpu(
            prefer="memory",
            min_free_mb=0,
            return_type="torch",
        )
        if device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
            print(f"Selected GPU: {device}\nGPU Info:\n{smi_out}")

    logging_dir = Path(config.output_dir, config.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Setup logging (must be after accelerator init)
    setup_logging(accelerator)
    logger = get_logger(__name__, log_level="INFO")
    logger.info(accelerator.state, main_process_only=False)

    # Set seed for reproducibility
    if config.seed is not None:
        set_seed(config.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)

        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name,
                exist_ok=True,
                token=config.hub_token,
            ).repo_id

    # Load models
    logger.info("Loading models...")
    models = load_models(config=config, device=accelerator.device)

    # Enable TF32 for faster training on Ampere GPUs
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Create dataloader
    logger.info("Creating dataloader...")
    with accelerator.main_process_first():
        if config.layout_conditioning_enabled:
            layout_split = resolve_layout_split(
                dataset_id=config.dataset_id or "flir_private_proxy_alignment_v18",
                split=config.train_split,
                root_dir=config.train_data_dir,
                annotations_path=config.layout_annotations_path,
            )
            layout_dataset = StableDiffusionLayoutDataset(
                root_dir=layout_split.root_dir,
                annotations_path=layout_split.annotations_path,
                tokenizer=models.tokenizer,
                resolution=config.resolution,
                normalization_mode=layout_split.normalization_mode,
                prompt_mode="constant" if config.resolved_training_prompt_text() else "class_list",
                constant_prompt=config.resolved_training_prompt_text() or DEFAULT_PROMPT_TEXT,
                thermal_scene_suffix="in thermal scene.",
                use_captions_if_available=False,
                max_samples=config.max_train_samples,
                subset_manifest=config.subset_manifest,
            )
            config.layout_category_id_to_name = dict(layout_dataset.category_id_to_name)
            from torch.utils.data import DataLoader

            train_dataloader = DataLoader(
                layout_dataset,
                shuffle=True,
                collate_fn=collate_sd_layout_batch,
                batch_size=config.train_batch_size,
                num_workers=config.dataloader_num_workers,
            )
            normalization_mode = layout_split.normalization_mode
        else:
            train_dataloader, normalization_mode = create_dataloader(
                dataset_id=config.dataset_id,
                dataset_name=config.dataset_name,
                dataset_config_name=config.dataset_config_name,
                train_data_dir=config.train_data_dir,
                train_split=config.train_split,
                cache_dir=config.cache_dir,
                tokenizer=models.tokenizer,
                resolution=config.resolution,
                center_crop=config.center_crop,
                random_flip=config.random_flip,
                interpolation_mode=config.image_interpolation_mode,
                image_column=config.image_column,
                caption_column=config.caption_column,
                batch_size=config.train_batch_size,
                num_workers=config.dataloader_num_workers,
                max_train_samples=config.max_train_samples,
                subset_manifest=config.subset_manifest,
                seed=config.seed,
                use_ir_preprocessing=config.use_ir_preprocessing,
                prompt_text=config.resolved_training_prompt_text(),
            )

    logger.info("Configuring Stage-1 adaptation baseline: %s", config.baseline_mode)
    adaptation_info = configure_trainable_components(models=models, config=config)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        config=config,
        models=models,
        train_dataloader=train_dataloader,
        normalization_mode=normalization_mode,
        adaptation_info=adaptation_info,
        accelerator=accelerator,
    )

    # Setup training
    trainer.setup()

    # Run training
    logger.info("Starting training...")
    trainer.train()

    # Push to hub if requested
    if config.push_to_hub and accelerator.is_main_process:
        logger.info("Pushing to hub...")
        save_model_card(
            repo_id=repo_id,
            images=None,
            base_model=config.pretrained_model_name_or_path,
            dataset_name=config.dataset_name,
            repo_folder=config.output_dir,
        )
        upload_folder(
            repo_id=repo_id,
            folder_path=config.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*", "checkpoint-*"],
        )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
