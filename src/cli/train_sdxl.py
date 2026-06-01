"""Canonical CLI for Stage-1 Stable Diffusion XL LoRA IR adaptation."""

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

from src.algorithms.stable_diffusion.utils import save_model_card, setup_logging
from src.algorithms.stable_diffusion_xl.config import parse_args
from src.algorithms.stable_diffusion_xl.data import create_dataloader
from src.algorithms.stable_diffusion_xl.models import (
    configure_trainable_components,
    get_canonical_output_dir,
    load_models,
)
from src.algorithms.stable_diffusion_xl.training import Trainer
from src.core.gpu_utils import get_least_used_cuda_gpu


check_min_version("0.37.0.dev0")


def main(argv=None) -> None:
    print("Parsing SDXL arguments...")
    config = parse_args(argv)
    config.output_dir = get_canonical_output_dir(config)

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
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.accelerator_mixed_precision(),
        log_with=config.report_to,
        project_config=ProjectConfiguration(
            project_dir=config.output_dir,
            logging_dir=logging_dir,
        ),
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if config.mixed_precision is None and accelerator.mixed_precision != "no":
        config.mixed_precision = accelerator.mixed_precision

    setup_logging(accelerator)
    logger = get_logger(__name__, log_level="INFO")
    logger.info(accelerator.state, main_process_only=False)

    if config.seed is not None:
        set_seed(config.seed)

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name,
                exist_ok=True,
                token=config.hub_token,
            ).repo_id
        else:
            repo_id = None
    else:
        repo_id = None

    logger.info("Loading SDXL models...")
    models = load_models(config=config, device=accelerator.device)

    logger.info("Creating SDXL dataloader...")
    with accelerator.main_process_first():
        train_dataloader, normalization_mode = create_dataloader(
            dataset_id=config.dataset_id,
            dataset_name=config.dataset_name,
            dataset_config_name=config.dataset_config_name,
            train_data_dir=config.train_data_dir,
            train_split=config.train_split,
            cache_dir=config.cache_dir,
            tokenizer_one=models.tokenizer,
            tokenizer_two=models.tokenizer_2,
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
            prompt_text=config.resolved_prompt_text(),
        )

    logger.info("Configuring SDXL adaptation baseline: %s", config.baseline_mode)
    adaptation_info = configure_trainable_components(models=models, config=config)

    trainer = Trainer(
        config=config,
        models=models,
        train_dataloader=train_dataloader,
        normalization_mode=normalization_mode,
        adaptation_info=adaptation_info,
        accelerator=accelerator,
    )
    trainer.setup()
    trainer.train()

    if config.push_to_hub and accelerator.is_main_process and repo_id is not None:
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
            commit_message="End of SDXL training",
            ignore_patterns=["step_*", "epoch_*", "checkpoint-*"],
        )

    logger.info("SDXL training complete!")


if __name__ == "__main__":
    main()
