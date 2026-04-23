#!/usr/bin/env python
# coding=utf-8
"""Training loop for Stage-1 Stable Diffusion IR adaptation."""

from __future__ import annotations

import math
import json
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft.utils import get_peft_model_state_dict
from torchvision.utils import save_image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import convert_state_dict_to_diffusers, is_wandb_available

from .config import TrainingConfig
from .models import (
    ModelComponents,
    REGIONDIFF_ADAPTER_WEIGHTS,
    build_stage1_manifest,
    UNET_EXPORT_DIRNAME,
    VAE_EXPORT_DIRNAME,
    TEXT_ENCODER_EXPORT_DIRNAME,
    _save_state_dict,
    create_load_model_hook,
    create_save_model_hook,
    get_trainable_models,
    get_trainable_params,
    save_stage1_manifest,
    trainable_component_names,
    unwrap_model,
)
from src.models.regiondiffusion import iter_regiondiff_adapter_parameters
from src.models.regiondiffusion_factory import regiondiff_state_dict, save_regiondiff_metadata


if is_wandb_available():
    import wandb


logger = get_logger(__name__, log_level="INFO")
CHECKPOINT_METADATA_FILENAME = "training_state.json"


def _sanitize_tracker_value(value):
    """Convert nested config values into TensorBoard hparams-safe scalars."""
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
    """TensorBoard hparams only accepts scalar-ish values, so stringify the rest."""
    return {key: _sanitize_tracker_value(value) for key, value in config.items()}


class Trainer:
    """Trainer for Stage-1 SD IR adaptation."""

    def __init__(
        self,
        config: TrainingConfig,
        models: ModelComponents,
        train_dataloader,
        *,
        normalization_mode: str,
        adaptation_info: Dict[str, object],
        accelerator: Optional[Accelerator] = None,
    ):
        self.config = config
        self.models = models
        self.train_dataloader = train_dataloader
        self.normalization_mode = normalization_mode
        self.adaptation_info = adaptation_info

        if accelerator is None:
            accelerator = self._create_accelerator()
        self.accelerator = accelerator

        self.global_step = 0
        self.first_epoch = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.manifest = None

    def _create_accelerator(self) -> Accelerator:
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
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        return accelerator

    def setup(self) -> None:
        logger.info("Setting up training components...")
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

        learning_rate = self.config.learning_rate
        if self.config.scale_lr:
            learning_rate = (
                learning_rate
                * self.config.gradient_accumulation_steps
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
            tracker_config["trainable_components"] = trainable_component_names(self.models)
            tracker_config["adaptation_info"] = self.adaptation_info
            self.accelerator.init_trackers(
                "sd-ir-adaptation",
                config=_sanitize_tracker_config(tracker_config),
            )

        logger.info("Training setup complete")

    def _create_optimizer(self, learning_rate: float):
        params = get_trainable_params(self.models)
        if not params:
            raise ValueError("No trainable parameters were configured for SD adaptation.")

        optimizer_params = params
        if self.config.layout_conditioning_enabled:
            region_adapter_ids = {
                id(param) for param in iter_regiondiff_adapter_parameters(self.models.unet)
            }
            region_params = [
                param
                for param in self.models.unet.parameters()
                if param.requires_grad and id(param) in region_adapter_ids
            ]
            other_params = [
                param
                for param in params
                if id(param) not in region_adapter_ids
            ]
            optimizer_params = []
            if region_params:
                optimizer_params.append(
                    {
                        "params": region_params,
                        "lr": self.config.adapter_learning_rate,
                        "name": "regiondiff_adapters",
                    }
                )
            if other_params:
                optimizer_params.append(
                    {
                        "params": other_params,
                        "lr": (
                            self.config.backbone_learning_rate
                            if self.config.baseline_mode != "sd_ir_lora"
                            else learning_rate
                        ),
                        "name": "backbone_or_lora",
                    }
                )

        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError("Please install bitsandbytes: pip install bitsandbytes") from exc
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            optimizer_params,
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
        num_warmup_steps = self.config.lr_warmup_steps * self.accelerator.num_processes
        num_training_steps = self.config.max_train_steps * self.accelerator.num_processes
        return get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _prepare_for_training(self) -> None:
        trainable_modules = get_trainable_models(self.models)
        to_prepare = [*trainable_modules, self.optimizer, self.train_dataloader, self.lr_scheduler]
        prepared = self.accelerator.prepare(*to_prepare)

        idx = 0
        if any(param.requires_grad for param in self.models.unet.parameters()):
            self.models.unet = prepared[idx]
            idx += 1
        if any(param.requires_grad for param in self.models.text_encoder.parameters()):
            self.models.text_encoder = prepared[idx]
            idx += 1
        if any(param.requires_grad for param in self.models.vae.parameters()):
            self.models.vae = prepared[idx]
            idx += 1

        self.optimizer = prepared[idx]
        self.train_dataloader = prepared[idx + 1]
        self.lr_scheduler = prepared[idx + 2]

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )

    def _register_hooks(self) -> None:
        if self.config.baseline_mode != "sd_ir_lora":
            return

        save_hook = create_save_model_hook(self.models.unet, self.accelerator)
        load_hook = create_load_model_hook(
            self.models.unet,
            self.accelerator,
            self.config.mixed_precision,
        )
        self.accelerator.register_save_state_pre_hook(save_hook)
        self.accelerator.register_load_state_pre_hook(load_hook)

    def _checkpoint_metadata(self) -> Dict[str, object]:
        return {
            "global_step": self.global_step,
            "lr_scheduler": self.config.lr_scheduler,
            "lr_warmup_steps": self.config.lr_warmup_steps,
            "max_train_steps": self.config.max_train_steps,
        }

    def _write_checkpoint_metadata(self, checkpoint_dir: Path) -> None:
        metadata_path = checkpoint_dir / CHECKPOINT_METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(self._checkpoint_metadata(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _read_checkpoint_metadata(self, checkpoint_dir: Path) -> Optional[Dict[str, object]]:
        metadata_path = checkpoint_dir / CHECKPOINT_METADATA_FILENAME
        if not metadata_path.exists():
            return None
        with open(metadata_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None

    def _validate_resume_constraints(self, checkpoint_dir: Path, *, step: int) -> None:
        if self.config.max_train_steps is not None and self.config.max_train_steps < step:
            raise ValueError(
                "Cannot resume from a checkpoint beyond max_train_steps: "
                f"checkpoint step={step}, max_train_steps={self.config.max_train_steps}."
            )

        metadata = self._read_checkpoint_metadata(checkpoint_dir)
        if metadata is None:
            if self.config.lr_scheduler == "constant" and self.config.lr_warmup_steps == 0:
                logger.warning(
                    "Checkpoint %s has no %s metadata; allowing resume because the current "
                    "schedule is constant with zero warmup.",
                    checkpoint_dir,
                    CHECKPOINT_METADATA_FILENAME,
                )
                return
            raise ValueError(
                f"Checkpoint {checkpoint_dir} has no {CHECKPOINT_METADATA_FILENAME} metadata. "
                "Resume is only allowed without metadata when lr_scheduler='constant' and "
                "lr_warmup_steps=0."
            )

        saved_scheduler = metadata.get("lr_scheduler")
        saved_warmup = metadata.get("lr_warmup_steps")
        saved_max_train_steps = metadata.get("max_train_steps")

        if saved_scheduler != self.config.lr_scheduler or saved_warmup != self.config.lr_warmup_steps:
            raise ValueError(
                "Resume config does not match the checkpointed LR schedule: "
                f"checkpoint has lr_scheduler={saved_scheduler!r}, "
                f"lr_warmup_steps={saved_warmup!r}; current config has "
                f"lr_scheduler={self.config.lr_scheduler!r}, "
                f"lr_warmup_steps={self.config.lr_warmup_steps!r}."
            )

        if (
            saved_max_train_steps is not None
            and self.config.max_train_steps is not None
            and saved_max_train_steps != self.config.max_train_steps
        ):
            if not (
                self.config.lr_scheduler == "constant"
                and self.config.lr_warmup_steps == 0
                and self.config.max_train_steps > saved_max_train_steps
            ):
                raise ValueError(
                    "Changing max_train_steps across resume is only supported for a constant "
                    "schedule with zero warmup when extending the run. "
                    f"Checkpoint max_train_steps={saved_max_train_steps}, "
                    f"current max_train_steps={self.config.max_train_steps}."
                )

    def resume_from_checkpoint(self) -> None:
        if self.config.resume_from_checkpoint is None:
            return

        if self.config.resume_from_checkpoint != "latest":
            checkpoint_dir = Path(self.config.resume_from_checkpoint)
            if not checkpoint_dir.is_absolute():
                checkpoint_dir = Path(self.config.output_dir) / checkpoint_dir
            path = checkpoint_dir.name
        else:
            dirs = os.listdir(self.config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
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

    def train(self) -> None:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %s", len(self.train_dataloader.dataset))
        logger.info("  Num Epochs = %s", self.config.num_train_epochs)
        logger.info("  Batch size per device = %s", self.config.train_batch_size)
        logger.info("  Gradient Accumulation steps = %s", self.config.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %s", self.config.max_train_steps)
        logger.info("  Baseline mode = %s", self.config.baseline_mode)
        logger.info("  Normalization mode = %s", self.normalization_mode)

        total_batch_size = (
            self.config.train_batch_size
            * self.accelerator.num_processes
            * self.config.gradient_accumulation_steps
        )
        logger.info("  Total train batch size = %s", total_batch_size)

        self.resume_from_checkpoint()
        progress_bar = tqdm(
            range(0, self.config.max_train_steps),
            initial=self.global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.first_epoch, self.config.num_train_epochs):
            train_loss = self._train_epoch(epoch, progress_bar)

            if (
                self.accelerator.is_main_process
                and self.config.validation_prompt is not None
                and epoch % self.config.validation_epochs == 0
            ):
                self._run_validation(epoch)

            if self.global_step >= self.config.max_train_steps:
                break

        self._finalize_training()

    def _train_epoch(self, epoch: int, progress_bar) -> float:
        self.models.unet.train()
        if any(param.requires_grad for param in self.models.text_encoder.parameters()):
            self.models.text_encoder.train()
        else:
            self.models.text_encoder.eval()

        if any(param.requires_grad for param in self.models.vae.parameters()):
            self.models.vae.train()
        else:
            self.models.vae.eval()

        train_loss = 0.0
        for step, batch in enumerate(self.train_dataloader):
            if self.global_step == 0 and step == 0 and self.accelerator.is_main_process:
                self._debug_save_batch(batch)

            loss = self._train_step(batch)

            avg_loss = self.accelerator.gather(loss.repeat(self.config.train_batch_size)).mean()
            train_loss += avg_loss.item() / self.config.gradient_accumulation_steps

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.global_step += 1
                self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                train_loss = 0.0
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
        accumulate_targets = get_trainable_models(self.models)
        accumulation_context = self.accelerator.accumulate(*accumulate_targets)

        with accumulation_context:
            pixel_values = batch["pixel_values"].to(device=self.accelerator.device)
            latents = self.models.vae.encode(
                pixel_values.to(dtype=next(self.models.vae.parameters()).dtype)
            ).latent_dist.sample()
            latents = latents * self.models.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            if self.config.noise_offset:
                noise += self.config.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device,
                )

            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                self.models.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            ).long()
            noisy_latents = self.models.noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = self.models.text_encoder(
                batch["input_ids"].to(device=self.accelerator.device),
                return_dict=False,
            )[0]

            target = self._get_prediction_target(latents, noise, timesteps)
            cross_attention_kwargs = None
            if self.config.layout_conditioning_enabled:
                cross_attention_kwargs = {
                    "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(device=latents.device, dtype=latents.dtype),
                    "labels": batch["labels"].to(device=latents.device),
                    "object_mask": batch["object_mask"].to(device=latents.device),
                }
            model_pred = self.models.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            loss = self._compute_loss(model_pred, target, timesteps)
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    get_trainable_params(self.models),
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
        if self.config.prediction_type is not None:
            self.models.noise_scheduler.register_to_config(
                prediction_type=self.config.prediction_type
            )

        prediction_type = self.models.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "v_prediction":
            return self.models.noise_scheduler.get_velocity(latents, noise, timesteps)
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    def _compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.snr_gamma is None:
            return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

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
        if self.global_step % self.config.checkpointing_steps != 0:
            return
        if not self.accelerator.is_main_process:
            return

        if self.config.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.config.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            if len(checkpoints) >= self.config.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                removing = checkpoints[:num_to_remove]
                logger.info("Removing %s old checkpoints", len(removing))
                for checkpoint in removing:
                    shutil.rmtree(os.path.join(self.config.output_dir, checkpoint))

        save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)
        self._write_checkpoint_metadata(Path(save_path))
        logger.info("Saved checkpoint to %s", save_path)

    def _run_validation(self, epoch: int, is_final: bool = False) -> List:
        logger.info("Running validation with prompt: %s", self.config.validation_prompt)
        pipeline_kwargs = dict(
            vae=unwrap_model(self.models.vae, self.accelerator)
            if any(param.requires_grad for param in self.models.vae.parameters())
            else self.models.vae,
            text_encoder=unwrap_model(self.models.text_encoder, self.accelerator)
            if any(param.requires_grad for param in self.models.text_encoder.parameters())
            else self.models.text_encoder,
            tokenizer=self.models.tokenizer,
            revision=self.config.revision,
            variant=self.config.variant,
            torch_dtype=self.models.weight_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if not self.config.layout_conditioning_enabled:
            pipeline_kwargs["unet"] = unwrap_model(self.models.unet, self.accelerator)
        pipeline = DiffusionPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            **pipeline_kwargs,
        )
        if self.config.layout_conditioning_enabled:
            pipeline.unet = unwrap_model(self.models.unet, self.accelerator)

        images = log_validation(
            pipeline=pipeline,
            validation_prompt=self.config.validation_prompt,
            num_images=self.config.num_validation_images,
            num_inference_steps=self.config.validation_num_inference_steps,
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
        pv = batch["pixel_values"]
        print(
            f"pixel_values: {pv.shape} {pv.dtype} "
            f"min={float(pv.min()):.3f} max={float(pv.max()):.3f} "
            f"mean={float(pv.mean()):.3f} std={float(pv.std()):.3f}"
        )
        os.makedirs("artifacts/debug/debug_samples", exist_ok=True)
        x = (pv[:4].cpu() + 1) / 2
        save_image(x, "artifacts/debug/debug_samples/batch0.png")

    def _finalize_training(self) -> None:
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            if self.config.baseline_mode == "sd_ir_lora":
                self._finalize_lora_export()
            else:
                self._finalize_unet_export()

            if self.config.validation_prompt is not None:
                pipeline_kwargs = dict(
                    vae=unwrap_model(self.models.vae, self.accelerator)
                    if any(param.requires_grad for param in self.models.vae.parameters())
                    else self.models.vae,
                    text_encoder=unwrap_model(self.models.text_encoder, self.accelerator)
                    if any(param.requires_grad for param in self.models.text_encoder.parameters())
                    else self.models.text_encoder,
                    tokenizer=self.models.tokenizer,
                    revision=self.config.revision,
                    variant=self.config.variant,
                    torch_dtype=self.models.weight_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                if not self.config.layout_conditioning_enabled:
                    pipeline_kwargs["unet"] = unwrap_model(self.models.unet, self.accelerator)
                pipeline = DiffusionPipeline.from_pretrained(
                    self.config.pretrained_model_name_or_path,
                    **pipeline_kwargs,
                )
                if self.config.layout_conditioning_enabled:
                    pipeline.unet = unwrap_model(self.models.unet, self.accelerator)
                log_validation(
                    pipeline=pipeline,
                    validation_prompt=self.config.validation_prompt,
                    num_images=self.config.num_validation_images,
                    num_inference_steps=self.config.validation_num_inference_steps,
                    device=self.accelerator.device,
                    seed=self.config.seed,
                    accelerator=self.accelerator,
                    epoch=self.config.num_train_epochs,
                    is_final=True,
                )

        self.accelerator.end_training()

    def _finalize_lora_export(self) -> None:
        self.models.unet = self.models.unet.to(torch.float32)
        unwrapped_unet = unwrap_model(self.models.unet, self.accelerator)
        lora_source = unwrapped_unet.base_unet if hasattr(unwrapped_unet, "base_unet") else unwrapped_unet
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(lora_source)
        )
        StableDiffusionPipeline.save_lora_weights(
            save_directory=self.config.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
        if self.config.layout_conditioning_enabled:
            save_regiondiff_metadata(
                unwrapped_unet,
                self.config.output_dir,
                extra={
                    "coexists_with": "sd_ir_lora",
                    "saved_weights": REGIONDIFF_ADAPTER_WEIGHTS,
                },
            )
            _save_state_dict(
                os.path.join(self.config.output_dir, REGIONDIFF_ADAPTER_WEIGHTS),
                {
                    key: value.detach().cpu().to(torch.float32)
                    for key, value in regiondiff_state_dict(
                        unwrapped_unet,
                        adapters_only=True,
                    ).items()
                },
            )
        logger.info("Saved final LoRA weights to %s", self.config.output_dir)

    def _finalize_unet_export(self) -> None:
        unwrapped_unet = unwrap_model(self.models.unet, self.accelerator).to(torch.float32)
        unet_dir = os.path.join(self.config.output_dir, UNET_EXPORT_DIRNAME)
        unet_to_save = unwrapped_unet.base_unet if self.config.layout_conditioning_enabled else unwrapped_unet
        unet_to_save.save_pretrained(unet_dir, safe_serialization=True)
        if self.config.layout_conditioning_enabled:
            save_regiondiff_metadata(
                unwrapped_unet,
                self.config.output_dir,
                extra={
                    "coexists_with": "sd_ir_unet",
                    "saved_weights": REGIONDIFF_ADAPTER_WEIGHTS,
                },
            )
            _save_state_dict(
                os.path.join(self.config.output_dir, REGIONDIFF_ADAPTER_WEIGHTS),
                {
                    key: value.detach().cpu().to(torch.float32)
                    for key, value in regiondiff_state_dict(
                        unwrapped_unet,
                        adapters_only=True,
                    ).items()
                },
            )
        logger.info("Saved adapted U-Net to %s", unet_dir)

        if any(param.requires_grad for param in self.models.text_encoder.parameters()):
            text_encoder = unwrap_model(self.models.text_encoder, self.accelerator).to(torch.float32)
            text_encoder.save_pretrained(os.path.join(self.config.output_dir, TEXT_ENCODER_EXPORT_DIRNAME))

        if any(param.requires_grad for param in self.models.vae.parameters()):
            vae = unwrap_model(self.models.vae, self.accelerator).to(torch.float32)
            vae.save_pretrained(os.path.join(self.config.output_dir, VAE_EXPORT_DIRNAME), safe_serialization=True)


def log_validation(
    pipeline,
    validation_prompt: str,
    num_images: int,
    num_inference_steps: int,
    device: torch.device,
    seed: Optional[int],
    accelerator: Accelerator,
    epoch: int,
    is_final: bool = False,
) -> List:
    logger.info("Generating %s images with prompt: %s", num_images, validation_prompt)

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device.type)

    images = []
    with autocast_ctx:
        for _ in range(num_images):
            image = pipeline(
                validation_prompt,
                num_inference_steps=num_inference_steps,
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
            tracker.writer.add_scalar(f"{phase_name}/generated_mean", float(np_images_01.mean()), epoch)
            tracker.writer.add_scalar(f"{phase_name}/generated_std", float(np_images_01.std()), epoch)
        elif tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{index}: {validation_prompt}")
                        for index, image in enumerate(images)
                    ]
                }
            )

    return images
