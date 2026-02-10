#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning script for Stable Diffusion 1.5 with LoRA.

This is the main entry point for training. The codebase is organized into
separate modules for better maintainability:

    - src/config.py: Argument parsing and configuration management
    - src/data.py: Dataset loading, preprocessing, and data augmentation
    - src/models.py: Model loading, LoRA configuration, and model utilities  
    - src/training.py: Training loop, validation, and checkpointing
    - src/utils.py: Helper functions and utilities

Example usage:
    accelerate launch train.py \\
        --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
        --train_data_dir="./my_dataset" \\
        --output_dir="./output" \\
        --resolution=512 \\
        --train_batch_size=4 \\
        --gradient_accumulation_steps=4 \\
        --num_train_epochs=100 \\
        --learning_rate=1e-4 \\
        --rank=4 \\
        --validation_prompt="A photo of a cat" \\
        --validation_epochs=10
"""

import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

from diffusers.utils import check_min_version

from sd_src.config import parse_args
from sd_src.data import create_dataloader
from sd_src.models import (
    load_models,
    get_lora_config,
    setup_lora,
)
from sd_src.training import Trainer
from sd_src.utils import setup_logging, save_model_card


# Require minimum diffusers version
check_min_version("0.37.0.dev0")


def main():
    """Main training function."""
    # Parse arguments first (before accelerator init)
    print("Parsing arguments...")
    config = parse_args()
    
    # Setup accelerator
    print("Initializing accelerator...")
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
    lora_config = get_lora_config(rank=config.rank)
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
            images=None,  # Could pass validation images here
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
