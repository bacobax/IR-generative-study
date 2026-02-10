#!/usr/bin/env python
# coding=utf-8
"""
Training module for Stable Diffusion LoRA fine-tuning.

Handles the training loop, validation, checkpointing, and optimization.
"""

import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from torchvision.utils import save_image

from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import convert_state_dict_to_diffusers, is_wandb_available

from .config import TrainingConfig
from .models import (
    ModelComponents,
    create_save_model_hook,
    create_load_model_hook,
    unwrap_model,
    get_trainable_params,
)


if is_wandb_available():
    import wandb


logger = get_logger(__name__, log_level="INFO")


class Trainer:
    """
    Trainer class for Stable Diffusion LoRA fine-tuning.
    
    Encapsulates the training loop, validation, and checkpointing logic.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        models: ModelComponents,
        train_dataloader,
        accelerator: Optional[Accelerator] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration.
            models: Model components.
            train_dataloader: Training dataloader.
            accelerator: Accelerator for distributed training.
        """
        self.config = config
        self.models = models
        self.train_dataloader = train_dataloader
        
        # Setup accelerator if not provided
        if accelerator is None:
            accelerator = self._create_accelerator()
        self.accelerator = accelerator
        
        # Training state
        self.global_step = 0
        self.first_epoch = 0
        self.optimizer = None
        self.lr_scheduler = None
    
    def _create_accelerator(self) -> Accelerator:
        """Create and configure the accelerator."""
        logging_dir = Path(self.config.output_dir, self.config.logging_dir)
        project_config = ProjectConfiguration(
            project_dir=self.config.output_dir,
            logging_dir=logging_dir,
        )
        
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with=self.config.report_to,
            project_config=project_config,
        )
        
        # Disable AMP for MPS
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        
        return accelerator
    
    def setup(self) -> None:
        """Setup training components (optimizer, scheduler, etc.)."""
        logger.info("Setting up training components...")
        
        # Set seed
        if self.config.seed is not None:
            set_seed(self.config.seed)
        
        # Create output directory
        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Enable TF32 if requested
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Scale learning rate if requested
        learning_rate = self.config.learning_rate
        if self.config.scale_lr:
            learning_rate = (
                learning_rate 
                * self.config.gradient_accumulation_steps 
                * self.config.train_batch_size 
                * self.accelerator.num_processes
            )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer(learning_rate)
        
        # Calculate training steps
        self._calculate_training_steps()
        
        # Setup learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Prepare with accelerator
        self._prepare_for_training()
        
        # Register hooks for checkpointing
        self._register_hooks()
        
        # Initialize trackers
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "text2image-fine-tune",
                config=vars(self.config),
            )
        
        logger.info("Training setup complete")
    
    def _create_optimizer(self, learning_rate: float):
        """Create the optimizer."""
        trainable_params = get_trainable_params(self.models.unet)
        
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes: pip install bitsandbytes"
                )
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW
        
        return optimizer_cls(
            trainable_params,
            lr=learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )
    
    def _calculate_training_steps(self) -> None:
        """Calculate the total number of training steps."""
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        
        if self.config.max_train_steps is None:
            self.config.max_train_steps = (
                self.config.num_train_epochs * num_update_steps_per_epoch
            )
        
        self.config.num_train_epochs = math.ceil(
            self.config.max_train_steps / num_update_steps_per_epoch
        )
        
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
    
    def _create_lr_scheduler(self):
        """Create the learning rate scheduler."""
        num_warmup_steps = (
            self.config.lr_warmup_steps * self.accelerator.num_processes
        )
        num_training_steps = (
            self.config.max_train_steps * self.accelerator.num_processes
        )
        
        return get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def _prepare_for_training(self) -> None:
        """Prepare models and dataloader with accelerator."""
        (
            self.models.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.models.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        
        # Recalculate steps after preparation
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        
        if self.config.max_train_steps is None:
            self.config.max_train_steps = (
                self.config.num_train_epochs * self.num_update_steps_per_epoch
            )
    
    def _register_hooks(self) -> None:
        """Register save/load hooks for checkpointing."""
        save_hook = create_save_model_hook(self.models.unet, self.accelerator)
        load_hook = create_load_model_hook(
            self.models.unet, 
            self.accelerator, 
            self.config.mixed_precision,
        )
        
        self.accelerator.register_save_state_pre_hook(save_hook)
        self.accelerator.register_load_state_pre_hook(load_hook)
    
    def resume_from_checkpoint(self) -> None:
        """Resume training from a checkpoint if specified."""
        if self.config.resume_from_checkpoint is None:
            return
        
        if self.config.resume_from_checkpoint != "latest":
            path = os.path.basename(self.config.resume_from_checkpoint)
        else:
            # Get most recent checkpoint
            dirs = os.listdir(self.config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            self.accelerator.print(
                f"Checkpoint '{self.config.resume_from_checkpoint}' not found. "
                "Starting new training run."
            )
            self.config.resume_from_checkpoint = None
            return
        
        self.accelerator.print(f"Resuming from checkpoint {path}")
        self.accelerator.load_state(os.path.join(self.config.output_dir, path))
        self.global_step = int(path.split("-")[1])
        self.first_epoch = self.global_step // self.num_update_steps_per_epoch
    
    def train(self) -> None:
        """Run the training loop."""
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
        logger.info(f"  Batch size per device = {self.config.train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.max_train_steps}")
        
        total_batch_size = (
            self.config.train_batch_size 
            * self.accelerator.num_processes 
            * self.config.gradient_accumulation_steps
        )
        logger.info(f"  Total train batch size = {total_batch_size}")
        
        # Resume from checkpoint
        self.resume_from_checkpoint()
        initial_global_step = self.global_step
        
        # Progress bar
        progress_bar = tqdm(
            range(0, self.config.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )
        
        for epoch in range(self.first_epoch, self.config.num_train_epochs):
            train_loss = self._train_epoch(epoch, progress_bar)
            
            # Run validation
            if self.accelerator.is_main_process:
                if (
                    self.config.validation_prompt is not None 
                    and epoch % self.config.validation_epochs == 0
                ):
                    self._run_validation(epoch)
            
            if self.global_step >= self.config.max_train_steps:
                break
        
        # Final save and validation
        self._finalize_training()
    
    def _train_epoch(self, epoch: int, progress_bar) -> float:
        """Run a single training epoch."""
        self.models.unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(self.train_dataloader):
            # Debug: save first batch samples
            if self.global_step == 0 and step == 0 and self.accelerator.is_main_process:
                self._debug_save_batch(batch)
            
            loss = self._train_step(batch)
            
            # Gather losses across processes
            avg_loss = self.accelerator.gather(
                loss.repeat(self.config.train_batch_size)
            ).mean()
            train_loss += avg_loss.item() / self.config.gradient_accumulation_steps
            
            # Update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.global_step += 1
                self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                train_loss = 0.0
                
                # Save checkpoint
                self._maybe_save_checkpoint()
            
            logs = {
                "step_loss": loss.detach().item(),
                "lr": self.lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            
            if self.global_step >= self.config.max_train_steps:
                break
        
        return train_loss
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step."""
        with self.accelerator.accumulate(self.models.unet):
            # Encode images to latent space
            latents = self.models.vae.encode(
                batch["pixel_values"].to(dtype=self.models.weight_dtype)
            ).latent_dist.sample()
            latents = latents * self.models.vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            if self.config.noise_offset:
                noise += self.config.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device,
                )
            
            # Sample timesteps
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                self.models.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            ).long()
            
            # Add noise (forward diffusion)
            noisy_latents = self.models.noise_scheduler.add_noise(
                latents, noise, timesteps
            )
            
            # Get text embeddings
            encoder_hidden_states = self.models.text_encoder(
                batch["input_ids"], return_dict=False
            )[0]
            
            # Get prediction target
            target = self._get_prediction_target(latents, noise, timesteps)
            
            # Forward pass
            model_pred = self.models.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                return_dict=False,
            )[0]
            
            # Compute loss
            loss = self._compute_loss(model_pred, target, timesteps)
            
            # Backward pass
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    get_trainable_params(self.models.unet),
                    self.config.max_grad_norm,
                )
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        return loss
    
    def _get_prediction_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get the target for loss computation based on prediction type."""
        # Set prediction type if specified
        if self.config.prediction_type is not None:
            self.models.noise_scheduler.register_to_config(
                prediction_type=self.config.prediction_type
            )
        
        prediction_type = self.models.noise_scheduler.config.prediction_type
        
        if prediction_type == "epsilon":
            return noise
        elif prediction_type == "v_prediction":
            return self.models.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    def _compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training loss."""
        if self.config.snr_gamma is None:
            return F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        # SNR-weighted loss
        snr = compute_snr(self.models.noise_scheduler, timesteps)
        mse_loss_weights = torch.stack(
            [snr, self.config.snr_gamma * torch.ones_like(timesteps)],
            dim=1,
        ).min(dim=1)[0]
        
        prediction_type = self.models.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        return loss.mean()
    
    def _maybe_save_checkpoint(self) -> None:
        """Save checkpoint if at the right step."""
        if self.global_step % self.config.checkpointing_steps != 0:
            return
        
        if not self.accelerator.is_main_process:
            return
        
        # Manage checkpoint limit
        if self.config.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.config.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            
            if len(checkpoints) >= self.config.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                removing = checkpoints[:num_to_remove]
                
                logger.info(f"Removing {len(removing)} old checkpoints")
                for checkpoint in removing:
                    checkpoint_path = os.path.join(self.config.output_dir, checkpoint)
                    shutil.rmtree(checkpoint_path)
        
        # Save checkpoint
        save_path = os.path.join(
            self.config.output_dir,
            f"checkpoint-{self.global_step}",
        )
        self.accelerator.save_state(save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    
    def _run_validation(self, epoch: int, is_final: bool = False) -> List:
        """Run validation and log images."""
        logger.info(f"Running validation with prompt: {self.config.validation_prompt}")
        
        pipeline = DiffusionPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            unet=unwrap_model(self.models.unet, self.accelerator),
            vae=self.models.vae,
            text_encoder=self.models.text_encoder,
            tokenizer=self.models.tokenizer,
            revision=self.config.revision,
            variant=self.config.variant,
            torch_dtype=self.models.weight_dtype,
        )
        
        images = log_validation(
            pipeline=pipeline,
            validation_prompt=self.config.validation_prompt,
            num_images=self.config.num_validation_images,
            device=self.accelerator.device,
            seed=self.config.seed,
            accelerator=self.accelerator,
            epoch=epoch,
            is_final=is_final,
        )
        
        del pipeline
        torch.cuda.empty_cache()
        
        return images
    
    def _debug_save_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Save first batch for debugging."""
        pv = batch["pixel_values"]
        print(f"pixel_values: {pv.shape} {pv.dtype} "
              f"min={float(pv.min()):.3f} max={float(pv.max()):.3f} "
              f"mean={float(pv.mean()):.3f} std={float(pv.std()):.3f}")
        
        os.makedirs("debug_samples", exist_ok=True)
        x = (pv[:4].cpu() + 1) / 2  # Unnormalize
        save_image(x, "debug_samples/batch0.png")
    
    def _finalize_training(self) -> None:
        """Finalize training: save final model and run final validation."""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            # Save final LoRA weights
            self.models.unet = self.models.unet.to(torch.float32)
            unwrapped_unet = unwrap_model(self.models.unet, self.accelerator)
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unwrapped_unet)
            )
            
            StableDiffusionPipeline.save_lora_weights(
                save_directory=self.config.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
            logger.info(f"Saved final LoRA weights to {self.config.output_dir}")
            
            # Final validation with trained weights
            if self.config.validation_prompt is not None:
                # Use the same approach as _run_validation: pass trained UNet directly
                pipeline = DiffusionPipeline.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    unet=unwrapped_unet,
                    vae=self.models.vae,
                    text_encoder=self.models.text_encoder,
                    tokenizer=self.models.tokenizer,
                    revision=self.config.revision,
                    variant=self.config.variant,
                    torch_dtype=self.models.weight_dtype,
                )
                
                log_validation(
                    pipeline=pipeline,
                    validation_prompt=self.config.validation_prompt,
                    num_images=self.config.num_validation_images,
                    device=self.accelerator.device,
                    seed=self.config.seed,
                    accelerator=self.accelerator,
                    epoch=self.config.num_train_epochs,
                    is_final=True,
                )
        
        self.accelerator.end_training()


def log_validation(
    pipeline,
    validation_prompt: str,
    num_images: int,
    device: torch.device,
    seed: Optional[int],
    accelerator: Accelerator,
    epoch: int,
    is_final: bool = False,
) -> List:
    """
    Run validation and log images.
    
    Args:
        pipeline: Diffusion pipeline.
        validation_prompt: Prompt for generation.
        num_images: Number of images to generate.
        device: Device to use.
        seed: Random seed.
        accelerator: Accelerator instance.
        epoch: Current epoch.
        is_final: Whether this is final validation.
    
    Returns:
        List of generated images.
    """
    logger.info(f"Generating {num_images} images with prompt: {validation_prompt}")
    
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    
    # Select autocast context
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device.type)
    
    images = []
    with autocast_ctx:
        for _ in range(num_images):
            image = pipeline(
                validation_prompt,
                num_inference_steps=30,
                generator=generator,
            ).images[0]
            images.append(image)
    
    # Log to trackers
    phase_name = "test" if is_final else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images_u8 = np.stack([np.asarray(img) for img in images])          # (N,H,W,3) uint8
            np_images_01 = np_images_u8.astype(np.float32) / 255.0               # (N,H,W,3) float [0,1]

            # 1) Generated in [0,1] domain (ToTensor domain)
            tracker.writer.add_images(
                f"{phase_name}/generated_rgb_01",
                np_images_01,
                epoch,
                dataformats="NHWC"
            )

            # 2) "Raw domain" via inverse fixed window
            A, B = 11667.0, 13944.0
            S = B - A

            x01 = np_images_01.mean(axis=-1)       # (N,H,W) [0,1] (assumes grayscale replicated RGB)
            raw = A + x01 * S                      # (N,H,W) [A,B]
        
            tracker.writer.add_scalar(f"{phase_name}/raw_min", float(raw.min()), epoch)
            tracker.writer.add_scalar(f"{phase_name}/raw_max", float(raw.max()), epoch)
            tracker.writer.add_scalar(f"{phase_name}/raw_mean", float(raw.mean()), epoch)

            # optional: visualize raw with the SAME window (maps back to [0,1])
            raw_vis01 = np.clip((raw - A) / S, 0.0, 1.0)
            raw_vis_rgb01 = np.repeat(raw_vis01[..., None], 3, axis=-1)

            tracker.writer.add_images(
                f"{phase_name}/raw_fixed_window_01",
                raw_vis_rgb01,
                epoch,
                dataformats="NHWC"
            )

            
        elif tracker.name == "wandb":
            tracker.log({
                phase_name: [
                    wandb.Image(image, caption=f"{i}: {validation_prompt}")
                    for i, image in enumerate(images)
                ]
            })
    
    return images
