"""Modular CLI entrypoint for Stable Diffusion LoRA training.

This module is the **source of truth** for launching SD 1.5 LoRA training.
The root-level ``train_sd.py`` is a thin compatibility wrapper that forwards
to :func:`main` here.

Usage::

    python -m src.cli.train_sd --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 ...
    # or via the legacy wrapper:
    python train_sd.py ...
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

from diffusers.utils import check_min_version

from src.algorithms.stable_diffusion.config import parse_args
from src.algorithms.stable_diffusion.data import create_dataloader
from src.algorithms.stable_diffusion.models import (
    load_models,
    get_lora_config,
    setup_lora,
)
from src.algorithms.stable_diffusion.training import Trainer
from src.algorithms.stable_diffusion.utils import setup_logging, save_model_card
from src.core.gpu_utils import get_least_used_cuda_gpu

# Require minimum diffusers version
check_min_version("0.37.0.dev0")


def main():
    """Main SD LoRA training function."""
    # Parse arguments first (before accelerator init)
    print("Parsing arguments...")
    config = parse_args()

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
    models = load_models(
        pretrained_model_name_or_path=config.pretrained_model_name_or_path,
        revision=config.revision,
        variant=config.variant,
        device=accelerator.device,
        mixed_precision=config.mixed_precision,
    )

    # Setup LoRA
    logger.info("Setting up LoRA...")
    lora_config = get_lora_config(rank=config.rank, lora_alpha_scale=config.lora_alpha_scale)
    models.unet = setup_lora(
        unet=models.unet,
        lora_config=lora_config,
        mixed_precision=config.mixed_precision,
        gradient_checkpointing=config.gradient_checkpointing,
        enable_xformers=config.enable_xformers_memory_efficient_attention,
    )

    # Enable TF32 for faster training on Ampere GPUs
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Create dataloader
    logger.info("Creating dataloader...")
    with accelerator.main_process_first():
        train_dataloader = create_dataloader(
            dataset_name=config.dataset_name,
            dataset_config_name=config.dataset_config_name,
            train_data_dir=config.train_data_dir,
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
            seed=config.seed,
            accelerator=accelerator,
            use_ir_preprocessing=config.use_ir_preprocessing,
            generic_prompt=config.generic_prompt,
        )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        config=config,
        models=models,
        train_dataloader=train_dataloader,
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
