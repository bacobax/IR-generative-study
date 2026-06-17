#!/usr/bin/env python
# coding=utf-8
"""Training loop for FLUX.1-dev QLoRA fine-tuning."""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import is_wandb_available
from torchvision.utils import save_image
from tqdm.auto import tqdm

from .config import TrainingConfig
from .models import (
    ModelComponents,
    build_stage1_manifest,
    create_load_model_hook,
    create_save_model_hook,
    get_trainable_models,
    get_trainable_params,
    save_stage1_manifest,
    unwrap_model,
)


if is_wandb_available():
    import wandb


logger = get_logger(__name__, log_level="INFO")
CHECKPOINT_METADATA_FILENAME = "training_state.json"


def _sanitize_tracker_value(value):
    if value is None:
        return "null"
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _sanitize_tracker_config(config: Dict[str, object]) -> Dict[str, object]:
    return {key: _sanitize_tracker_value(value) for key, value in config.items()}


def _get_sigmas(
    timesteps: torch.Tensor,
    scheduler_copy,
    *,
    n_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device,
) -> torch.Tensor:
    """Look up the flow-matching sigmas for the given timestep indices."""
    sigmas = scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler_copy.timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class Trainer:
    """Trainer for FLUX.1-dev QLoRA stage-1 adaptation."""

    def __init__(
        self,
        config: TrainingConfig,
        models: ModelComponents,
        train_dataloader,
        *,
        normalization_mode: str,
        adaptation_info: Dict[str, object],
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        accelerator: Optional[Accelerator] = None,
    ):
        self.config = config
        self.models = models
        self.train_dataloader = train_dataloader
        self.normalization_mode = normalization_mode
        self.adaptation_info = adaptation_info
        # Cached prompt embeddings (CPU; moved to device in _train_step).
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.text_ids = text_ids
        self.accelerator = accelerator or self._create_accelerator()
        self.global_step = 0
        self.first_epoch = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.manifest = None
        # Deep-copy of the scheduler used to look up sigmas without mutating
        # the training scheduler.  Created in setup() after model loading.
        self._scheduler_copy = None
        # Optional VAE kept alive when cache_latents=False.
        self._vae_config = None
        self._latents_cache: Optional[List] = None

    def _create_accelerator(self) -> Accelerator:
        logging_dir = Path(self.config.output_dir, self.config.logging_dir)
        project_config = ProjectConfiguration(
            project_dir=self.config.output_dir,
            logging_dir=logging_dir,
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.accelerator_mixed_precision(),
            log_with=self.config.report_to,
            project_config=project_config,
        )
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        return accelerator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if self.config.seed is not None:
            set_seed(self.config.seed)
        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            self.manifest = save_stage1_manifest(
                self.config.output_dir,
                build_stage1_manifest(
                    config=self.config,
                    normalization_mode=self.normalization_mode,
                    adaptation_info=self.adaptation_info,
                ),
            )
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Deep-copy the scheduler BEFORE prepare() so it is a plain Python
        # object that we can use device-independently in _get_sigmas.
        self._scheduler_copy = copy.deepcopy(self.models.noise_scheduler)

        learning_rate = self.config.learning_rate
        if self.config.scale_lr:
            learning_rate *= (
                self.config.gradient_accumulation_steps
                * self.config.train_batch_size
                * self.accelerator.num_processes
            )
        self.optimizer = self._create_optimizer(learning_rate)
        self._calculate_training_steps()
        self.lr_scheduler = self._create_lr_scheduler()
        self._prepare_for_training()
        self._register_hooks()

        if self.accelerator.is_main_process:
            tracker_config = vars(self.config).copy()
            tracker_config["normalization_mode"] = self.normalization_mode
            tracker_config["adaptation_info"] = self.adaptation_info
            self.accelerator.init_trackers(
                "flux-qlora-adaptation",
                config=_sanitize_tracker_config(tracker_config),
            )

        # Optional latent caching: pre-encode all images, then free the VAE.
        self._vae_config = self.models.vae.config
        if self.config.cache_latents and self.accelerator.is_main_process:
            logger.info("Caching latents (this frees the VAE from GPU memory)…")
            self._latents_cache = []
            for batch in tqdm(self.train_dataloader, desc="Caching latents"):
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].to(
                        self.accelerator.device, dtype=torch.float32
                    )
                    self._latents_cache.append(
                        self.models.vae.encode(pixel_values).latent_dist
                    )
            del self.models.vae
            free_memory()
            logger.info("Latent cache complete (%d batches). VAE freed.", len(self._latents_cache))

    def train(self) -> None:
        logger.info("***** Running FLUX QLoRA training *****")
        logger.info("  Num examples = %s", len(self.train_dataloader.dataset))
        logger.info("  Num Epochs = %s", self.config.num_train_epochs)
        logger.info("  Batch size per device = %s", self.config.train_batch_size)
        logger.info("  Total optimization steps = %s", self.config.max_train_steps)
        self.resume_from_checkpoint()
        progress_bar = tqdm(
            range(0, self.config.max_train_steps),
            initial=self.global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )
        for epoch in range(self.first_epoch, self.config.num_train_epochs):
            self._train_epoch(progress_bar)
            if (
                self.accelerator.is_main_process
                and self.config.validation_prompt is not None
                and epoch % self.config.validation_epochs == 0
            ):
                self._run_validation(epoch)
            self._maybe_save_checkpoint(epoch)
            if self.global_step >= self.config.max_train_steps:
                break
        self._finalize_training()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_optimizer(self, learning_rate: float):
        params = get_trainable_params(self.models)
        if not params:
            raise ValueError("No trainable parameters configured for FLUX QLoRA.")
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError("Please install bitsandbytes: pip install bitsandbytes") from exc
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW
        return optimizer_cls(
            params,
            lr=learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

    def _calculate_training_steps(self) -> None:
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        if self.config.max_train_steps is None:
            self.config.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        self.config.num_train_epochs = math.ceil(
            self.config.max_train_steps / num_update_steps_per_epoch
        )
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

    def _create_lr_scheduler(self):
        return get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.max_train_steps * self.accelerator.num_processes,
        )

    def _prepare_for_training(self) -> None:
        prepared = self.accelerator.prepare(
            self.models.transformer,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        self.models.transformer = prepared[0]
        self.optimizer = prepared[1]
        self.train_dataloader = prepared[2]
        self.lr_scheduler = prepared[3]
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )

    def _register_hooks(self) -> None:
        save_hook = create_save_model_hook(self.models, self.accelerator)
        load_hook = create_load_model_hook(self.models, self.accelerator, self.config.mixed_precision)
        self.accelerator.register_save_state_pre_hook(save_hook)
        self.accelerator.register_load_state_pre_hook(load_hook)

    def _checkpoint_metadata(self) -> Dict[str, object]:
        return {
            "global_step": self.global_step,
            "lr_scheduler": self.config.lr_scheduler,
            "lr_warmup_steps": self.config.lr_warmup_steps,
            "max_train_steps": self.config.max_train_steps,
            "checkpointing_epochs": self.config.checkpointing_epochs,
            "save_optimizer_state": self.config.save_optimizer_state,
        }

    def _write_checkpoint_metadata(self, checkpoint_dir: Path) -> None:
        (checkpoint_dir / CHECKPOINT_METADATA_FILENAME).write_text(
            json.dumps(self._checkpoint_metadata(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _read_checkpoint_metadata(self, checkpoint_dir: Path) -> Optional[Dict[str, object]]:
        metadata_path = checkpoint_dir / CHECKPOINT_METADATA_FILENAME
        if not metadata_path.exists():
            return None
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _prune_optimizer_state(self, checkpoint_dir: Path) -> None:
        for pattern in ("optimizer*.bin", "optimizer*.pt", "optimizer*.safetensors"):
            for path in checkpoint_dir.glob(pattern):
                if path.is_file():
                    path.unlink()

    def _validate_resume_constraints(self, checkpoint_dir: Path, *, step: int) -> None:
        if self.config.max_train_steps is not None and self.config.max_train_steps < step:
            raise ValueError(
                f"Cannot resume from checkpoint beyond max_train_steps: "
                f"checkpoint step={step}, max_train_steps={self.config.max_train_steps}."
            )
        metadata = self._read_checkpoint_metadata(checkpoint_dir)
        if metadata is None:
            if self.config.lr_scheduler == "constant" and self.config.lr_warmup_steps == 0:
                return
            raise ValueError(f"Checkpoint {checkpoint_dir} has no {CHECKPOINT_METADATA_FILENAME} metadata.")
        if metadata.get("lr_scheduler") != self.config.lr_scheduler or metadata.get("lr_warmup_steps") != self.config.lr_warmup_steps:
            raise ValueError("Resume config does not match the checkpointed LR schedule.")

    def resume_from_checkpoint(self) -> None:
        if self.config.resume_from_checkpoint is None:
            return
        if self.config.resume_from_checkpoint != "latest":
            checkpoint_dir = Path(self.config.resume_from_checkpoint)
            if not checkpoint_dir.is_absolute():
                checkpoint_dir = Path(self.config.output_dir) / checkpoint_dir
            path = checkpoint_dir.name
        else:
            dirs = [d for d in os.listdir(self.config.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
            checkpoint_dir = Path(self.config.output_dir) / path if path is not None else None
        if path is None:
            self.accelerator.print(
                f"Checkpoint '{self.config.resume_from_checkpoint}' not found. Starting new training run."
            )
            self.config.resume_from_checkpoint = None
            return
        assert checkpoint_dir is not None
        step = int(path.split("-")[1])
        self._validate_resume_constraints(checkpoint_dir, step=step)
        self.accelerator.print(f"Resuming from checkpoint {path}")
        self.accelerator.load_state(str(checkpoint_dir))
        self.global_step = step
        self.first_epoch = self.global_step // self.num_update_steps_per_epoch

    def _train_epoch(self, progress_bar) -> None:
        self.models.transformer.train()
        train_loss = 0.0
        for step, batch in enumerate(self.train_dataloader):
            if self.global_step == 0 and step == 0 and self.accelerator.is_main_process:
                self._debug_save_batch(batch)
            loss = self._train_step(batch, step=step)
            avg_loss = self.accelerator.gather(loss.repeat(self.config.train_batch_size)).mean()
            train_loss += avg_loss.item() / self.config.gradient_accumulation_steps
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.global_step += 1
                self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                train_loss = 0.0
            progress_bar.set_postfix(
                step_loss=loss.detach().item(), lr=self.lr_scheduler.get_last_lr()[0]
            )
            if self.global_step >= self.config.max_train_steps:
                break

    def _train_step(self, batch: Dict[str, torch.Tensor], *, step: int) -> torch.Tensor:
        """Single FLUX flow-matching training step.

        The math here follows the blog's training loop (flux_lora_quant_blogpost.py
        lines 456-521) which is the canonical FLUX diffusers training recipe.

        Key differences from SDXL:
          - Latents have 16 channels (vs 4); shift_factor must be applied.
          - Noisy input is formed as linear interpolation:
              noisy = (1-sigma)*latent + sigma*noise   (flow-matching)
            rather than DDPM's add_noise.
          - Latents are "packed" into a 1D token sequence for the MMDiT.
          - Timestep is passed as timestep/1000 (normalised to [0,1]).
          - Loss target is: noise - latent (velocity prediction).
        """
        with self.accelerator.accumulate(self.models.transformer):
            device = self.accelerator.device

            # ------ Get latents (cached or on-the-fly) ------
            if self._latents_cache is not None:
                # Cache is indexed modulo its length so an epoch that has
                # more batches than the cache still works (rare edge case).
                model_input = self._latents_cache[step % len(self._latents_cache)].sample()
            else:
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                with torch.no_grad():
                    model_input = self.models.vae.encode(pixel_values).latent_dist.sample()

            # Apply FLUX VAE normalisation: (x - shift_factor) * scaling_factor
            model_input = (model_input - self._vae_config.shift_factor) * self._vae_config.scaling_factor
            model_input = model_input.to(dtype=torch.float16, device=device)

            bsz = model_input.shape[0]

            # ------ Prepare positional image IDs for the packed tokens ------
            # FLUX packs (H/2) x (W/2) patches into a 1D sequence; each patch
            # gets a 3-element (batch, row, col) position id.
            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                bsz,
                model_input.shape[2] // 2,
                model_input.shape[3] // 2,
                device,
                torch.float16,
            )

            # ------ Sample a flow-matching timestep and compute sigma ------
            # compute_density_for_timestep_sampling draws u ~ Uniform(0,1) then
            # optionally warps it (logit-normal, mode-weighted, etc.).  Here
            # weighting_scheme="none" → plain uniform sampling.
            u = compute_density_for_timestep_sampling(
                weighting_scheme=self.config.weighting_scheme,
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self._scheduler_copy.config.num_train_timesteps).long()
            timesteps = self._scheduler_copy.timesteps[indices].to(device=device)

            # sigma is the noise fraction at this timestep (0 = clean, 1 = pure noise).
            sigmas = _get_sigmas(
                timesteps,
                self._scheduler_copy,
                n_dim=model_input.ndim,
                dtype=model_input.dtype,
                device=device,
            )

            # ------ Corrupt the latent ------
            noise = torch.randn_like(model_input)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            # ------ Pack latents into 1D token sequence ------
            packed_noisy_model_input = FluxPipeline._pack_latents(
                noisy_model_input,
                bsz,
                model_input.shape[1],
                model_input.shape[2],
                model_input.shape[3],
            )

            # ------ Broadcast the single cached prompt to the batch ------
            prompt_embeds = self.prompt_embeds.to(device=device, dtype=torch.float16).expand(bsz, -1, -1)
            pooled_prompt_embeds = self.pooled_prompt_embeds.to(device=device, dtype=torch.float16).expand(bsz, -1)
            txt_ids = self.text_ids.to(device=device, dtype=torch.float16)
            # text_ids shape from encode_prompt is (1, seq, 3); replicate to batch.
            if txt_ids.ndim == 3 and txt_ids.shape[0] == 1:
                txt_ids = txt_ids.expand(bsz, -1, -1)

            # ------ FLUX.1-dev requires a guidance scalar ------
            guidance = None
            transformer_unwrapped = unwrap_model(self.models.transformer, self.accelerator)
            if getattr(transformer_unwrapped.config, "guidance_embeds", False):
                guidance = torch.tensor(
                    [self.config.guidance_scale], device=device
                ).expand(bsz)

            # ------ Forward pass ------
            model_pred = self.models.transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            # ------ Unpack and compute flow-matching loss ------
            vae_scale_factor = 2 ** (len(self._vae_config.block_out_channels) - 1)
            model_pred = FluxPipeline._unpack_latents(
                model_pred,
                model_input.shape[2] * vae_scale_factor,
                model_input.shape[3] * vae_scale_factor,
                vae_scale_factor,
            )

            # Loss weight (none = uniform, sigma_sqrt / logit_normal / etc. supported).
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=self.config.weighting_scheme, sigmas=sigmas
            )

            # Velocity target for flow matching: noise - latent.
            target = noise - model_input

            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(bsz, -1),
                dim=1,
            ).mean()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    get_trainable_params(self.models), self.config.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        return loss

    def _save_checkpoint(self, *, epoch: Optional[int] = None, force: bool = False) -> None:
        if not self.accelerator.is_main_process:
            return
        if not force and epoch is not None:
            if (epoch + 1) % self.config.checkpointing_epochs != 0:
                return
        save_path = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        if save_path.exists():
            logger.info("Checkpoint already exists at %s; skipping save.", save_path)
            return
        if self.config.checkpoints_total_limit is not None:
            checkpoints = [d for d in os.listdir(self.config.output_dir) if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            if len(checkpoints) >= self.config.checkpoints_total_limit:
                for checkpoint in checkpoints[: len(checkpoints) - self.config.checkpoints_total_limit + 1]:
                    shutil.rmtree(os.path.join(self.config.output_dir, checkpoint))
        self.accelerator.save_state(str(save_path))
        if not self.config.save_optimizer_state:
            self._prune_optimizer_state(save_path)
        self._write_checkpoint_metadata(save_path)
        logger.info("Saved checkpoint to %s", save_path)

    def _maybe_save_checkpoint(self, epoch: int) -> None:
        self._save_checkpoint(epoch=epoch)

    def _run_validation(self, epoch: int, is_final: bool = False) -> List:
        """Run validation inference and log images to the tracker."""
        assert self.config.validation_prompt is not None
        logger.info("Running validation (epoch=%d)…", epoch)
        pipeline = FluxPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            transformer=unwrap_model(self.models.transformer, self.accelerator),
            torch_dtype=self.models.weight_dtype,
        )
        return log_validation(
            pipeline=pipeline,
            validation_prompt=self.config.validation_prompt,
            num_images=self.config.num_validation_images,
            num_inference_steps=self.config.validation_num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            height=self.config.resolution,
            width=self.config.resolution,
            device=self.accelerator.device,
            seed=self.config.seed,
            accelerator=self.accelerator,
            epoch=epoch,
            is_final=is_final,
        )

    def _debug_save_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        pv = batch["pixel_values"]
        print(
            f"pixel_values: {pv.shape} {pv.dtype} "
            f"min={float(pv.min()):.3f} max={float(pv.max()):.3f} "
            f"mean={float(pv.mean()):.3f} std={float(pv.std()):.3f}"
        )
        os.makedirs("artifacts/debug/debug_samples", exist_ok=True)
        save_image((pv[:4].cpu() + 1) / 2, "artifacts/debug/debug_samples/flux_batch0.png")

    def _finalize_training(self) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_checkpoint(force=True)
            self._finalize_lora_export()
            if self.config.validation_prompt is not None:
                self._run_validation(self.config.num_train_epochs, is_final=True)
        self.accelerator.end_training()

    def _finalize_lora_export(self) -> None:
        from peft.utils import get_peft_model_state_dict

        transformer = unwrap_model(self.models.transformer, self.accelerator).to(torch.float32)
        FluxPipeline.save_lora_weights(
            save_directory=self.config.output_dir,
            transformer_lora_layers=get_peft_model_state_dict(transformer),
            text_encoder_lora_layers=None,
            safe_serialization=True,
        )
        logger.info("Saved final FLUX LoRA weights to %s", self.config.output_dir)


def log_validation(
    pipeline,
    validation_prompt: str,
    num_images: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    device: torch.device,
    seed: Optional[int],
    accelerator: Accelerator,
    epoch: int,
    *,
    is_final: bool = False,
) -> List:
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    autocast_ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(device.type)
    images = []
    with autocast_ctx:
        for _ in range(num_images):
            image = pipeline(
                validation_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            images.append(image)

    phase_name = "test" if is_final else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images_u8 = np.stack([np.asarray(img) for img in images])
            np_images_01 = np_images_u8.astype(np.float32) / 255.0
            tracker.writer.add_images(
                f"{phase_name}/generated_rgb_01",
                np_images_01,
                epoch,
                dataformats="NHWC",
            )
        elif tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{index}: {validation_prompt}")
                        for index, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images
