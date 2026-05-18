"""RegionDiff Stage-2 Stable Diffusion 1.5 adaptation implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from src.core.diffusers_compat import disable_diffusers_optional_scipy

disable_diffusers_optional_scipy(lightweight_diffusers_imports=False)

from transformers import CLIPTokenizer

from diffusers.utils import check_min_version

from src.algorithms.stable_diffusion.layout_data import (
    build_fixed_validation_batch,
    create_layout_dataloaders,
)
from src.algorithms.stable_diffusion.layout_models import (
    configure_layout_trainability,
    load_layout_model_components,
)
from src.algorithms.stable_diffusion.layout_training import LayoutTrainer
from src.algorithms.stable_diffusion.utils import setup_logging
from src.core.configs.sd_layout_config import parse_args
from src.core.gpu_utils import get_least_used_cuda_gpu


check_min_version("0.37.0.dev0")


def main(argv: Optional[list[str]] = None) -> None:
    """Main entrypoint for SD layout stage-2 training."""
    config = parse_args(argv)

    if "RANK" not in os.environ:
        device, smi_out = get_least_used_cuda_gpu(
            prefer="memory",
            min_free_mb=0,
            return_type="torch",
        )
        if device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
            print(f"Selected GPU: {device}\nGPU Info:\n{smi_out}")

    logging_dir = Path(config.output.output_dir, config.output.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output.output_dir,
        logging_dir=logging_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=config.training.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    setup_logging(accelerator)
    logger = get_logger(__name__, log_level="INFO")
    logger.info(accelerator.state, main_process_only=False)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    if accelerator.is_main_process:
        os.makedirs(config.output.output_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(
        config.stage1.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=config.stage1.revision,
    )

    with accelerator.main_process_first():
        train_dataloader, _, train_dataset, val_dataset = create_layout_dataloaders(
            config,
            tokenizer=tokenizer,
        )
        validation_batch = build_fixed_validation_batch(
            val_dataset,
            max_examples=config.validation.num_validation_images,
        )

    logger.info("Loading stage-2 RegionDiff wrapper")
    with accelerator.main_process_first():
        models, init_info = load_layout_model_components(
            config=config,
            category_id_to_name=train_dataset.category_id_to_name,
            device=accelerator.device,
        )
    trainability_info = configure_layout_trainability(models=models, config=config)

    trainer = LayoutTrainer(
        config=config,
        models=models,
        train_dataloader=train_dataloader,
        validation_batch=validation_batch,
        init_info=init_info,
        trainability_info=trainability_info,
        accelerator=accelerator,
    )
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
