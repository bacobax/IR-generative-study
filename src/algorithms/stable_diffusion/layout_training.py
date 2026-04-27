"""Training loop for RegionDiff-style SD layout stage-2."""

from __future__ import annotations

import json
import math
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
from torchvision.utils import save_image
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available

from src.algorithms.stable_diffusion.layout_models import (
    SDLayoutModelComponents,
    build_optimizer_param_groups,
    create_stage2_load_model_hook,
    create_stage2_save_model_hook,
    save_stage2_layout_artifact,
)
from src.algorithms.stable_diffusion.models import unwrap_model
from src.core.configs.config_loader import dataclass_to_dict
from src.core.configs.sd_layout_config import SDLayoutTrainConfig
from src.core.training_utils import compute_snr
from src.core.visualization.layout_debug import draw_bbox_overlays, render_class_layout
from src.models.regiondiffusion import build_area_weight_map


if is_wandb_available():
    import wandb


logger = get_logger(__name__, log_level="INFO")
CHECKPOINT_METADATA_FILENAME = "training_state.json"


def _sanitize_tracker_value(value):
    if value is None:
        return "null"
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _sanitize_tracker_config(config: Dict[str, object]) -> Dict[str, object]:
    return {key: _sanitize_tracker_value(value) for key, value in config.items()}


class LayoutTrainer:
    """Accelerate-powered trainer for layout-conditioned Stable Diffusion stage-2."""

    def __init__(
        self,
        *,
        config: SDLayoutTrainConfig,
        models: SDLayoutModelComponents,
        train_dataloader,
        validation_batch: Optional[Dict[str, object]],
        init_info: Dict[str, object],
        trainability_info: Dict[str, object],
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        self.config = config
        self.models = models
        self.train_dataloader = train_dataloader
        self.validation_batch = validation_batch
        self.init_info = init_info
        self.trainability_info = trainability_info

        if accelerator is None:
            accelerator = self._create_accelerator()
        self.accelerator = accelerator

        self.global_step = 0
        self.first_epoch = 0
        self.optimizer = None
        self.lr_scheduler = None

    def _create_accelerator(self) -> Accelerator:
        logging_dir = Path(self.config.output.output_dir, self.config.output.logging_dir)
        project_config = ProjectConfiguration(
            project_dir=self.config.output.output_dir,
            logging_dir=logging_dir,
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            mixed_precision=self.config.training.mixed_precision,
            log_with=self.config.training.report_to,
            project_config=project_config,
        )
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        return accelerator

    def setup(self) -> None:
        if self.config.training.seed is not None:
            set_seed(self.config.training.seed)

        if self.accelerator.is_main_process:
            os.makedirs(self.config.output.output_dir, exist_ok=True)

        if self.config.training.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.optimizer = self._create_optimizer()
        self._calculate_training_steps()
        self.lr_scheduler = self._create_lr_scheduler()
        self._prepare_for_training()
        self._register_hooks()

        if self.accelerator.is_main_process:
            tracker_config = dataclass_to_dict(self.config)
            tracker_config["stage1_initialization"] = self.init_info
            tracker_config["trainability_info"] = self.trainability_info
            self.accelerator.init_trackers(
                "sd-layout-stage2",
                config=_sanitize_tracker_config(tracker_config),
            )

    def _create_optimizer(self):
        param_groups = build_optimizer_param_groups(
            models=self.models,
            config=self.config,
            accelerator_processes=self.accelerator.num_processes,
        )
        if self.config.training.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("Please install bitsandbytes: pip install bitsandbytes") from exc
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            param_groups,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            weight_decay=self.config.training.adam_weight_decay,
            eps=self.config.training.adam_epsilon,
        )

    def _calculate_training_steps(self) -> None:
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.training.gradient_accumulation_steps
        )
        if self.config.training.max_train_steps is None:
            self.config.training.max_train_steps = (
                self.config.training.num_train_epochs * num_update_steps_per_epoch
            )
        self.config.training.num_train_epochs = math.ceil(
            self.config.training.max_train_steps / num_update_steps_per_epoch
        )
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

    def _create_lr_scheduler(self):
        num_warmup_steps = self.config.training.lr_warmup_steps * self.accelerator.num_processes
        num_training_steps = self.config.training.max_train_steps * self.accelerator.num_processes
        return get_scheduler(
            self.config.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _prepare_for_training(self) -> None:
        prepared = self.accelerator.prepare(
            self.models.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        self.models.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = prepared
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.training.gradient_accumulation_steps
        )

    def _register_hooks(self) -> None:
        save_hook = create_stage2_save_model_hook(
            unwrap_model(self.models.unet, self.accelerator),
            self.accelerator,
        )
        load_hook = create_stage2_load_model_hook(
            unwrap_model(self.models.unet, self.accelerator),
            self.accelerator,
        )
        self.accelerator.register_save_state_pre_hook(save_hook)
        self.accelerator.register_load_state_pre_hook(load_hook)

    def _checkpoint_metadata(self) -> Dict[str, object]:
        return {
            "global_step": self.global_step,
            "lr_scheduler": self.config.training.lr_scheduler,
            "lr_warmup_steps": self.config.training.lr_warmup_steps,
            "max_train_steps": self.config.training.max_train_steps,
            "train_mode": self.config.training.train_mode,
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
        with open(metadata_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None

    def _validate_resume_constraints(self, checkpoint_dir: Path, *, step: int) -> None:
        if self.config.training.max_train_steps is not None and self.config.training.max_train_steps < step:
            raise ValueError(
                "Cannot resume from a checkpoint beyond max_train_steps: "
                f"checkpoint step={step}, max_train_steps={self.config.training.max_train_steps}."
            )

        metadata = self._read_checkpoint_metadata(checkpoint_dir)
        if metadata is None:
            if self.config.training.lr_scheduler == "constant" and self.config.training.lr_warmup_steps == 0:
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
        if (
            saved_scheduler != self.config.training.lr_scheduler
            or saved_warmup != self.config.training.lr_warmup_steps
        ):
            raise ValueError(
                "Resume config does not match the checkpointed LR schedule: "
                f"checkpoint has lr_scheduler={saved_scheduler!r}, "
                f"lr_warmup_steps={saved_warmup!r}; current config has "
                f"lr_scheduler={self.config.training.lr_scheduler!r}, "
                f"lr_warmup_steps={self.config.training.lr_warmup_steps!r}."
            )

    def resume_from_checkpoint(self) -> None:
        resume_from = self.config.training.resume_from_checkpoint
        if resume_from is None:
            return

        if resume_from != "latest":
            checkpoint_dir = Path(resume_from)
            if not checkpoint_dir.is_absolute() and not checkpoint_dir.exists():
                checkpoint_dir = Path(self.config.output.output_dir) / checkpoint_dir
            checkpoint_name = checkpoint_dir.name
        else:
            checkpoints = [
                entry
                for entry in os.listdir(self.config.output.output_dir)
                if entry.startswith("checkpoint-")
            ]
            checkpoints = sorted(checkpoints, key=lambda entry: int(entry.split("-")[1]))
            checkpoint_name = checkpoints[-1] if checkpoints else None
            checkpoint_dir = (
                Path(self.config.output.output_dir) / checkpoint_name
                if checkpoint_name is not None
                else None
            )

        if checkpoint_name is None or checkpoint_dir is None:
            self.accelerator.print(
                f"Checkpoint '{resume_from}' not found. Starting new training run."
            )
            self.config.training.resume_from_checkpoint = None
            return

        step = int(checkpoint_name.split("-")[1])
        self._validate_resume_constraints(checkpoint_dir, step=step)
        self.accelerator.print(f"Resuming from checkpoint {checkpoint_name}")
        self.accelerator.load_state(str(checkpoint_dir))
        self.global_step = step
        self.first_epoch = self.global_step // self.num_update_steps_per_epoch

    def train(self) -> None:
        logger.info("***** Running SD layout stage-2 training *****")
        logger.info("  Num examples = %s", len(self.train_dataloader.dataset))
        logger.info("  Num Epochs = %s", self.config.training.num_train_epochs)
        logger.info("  Batch size per device = %s", self.config.data.batch_size)
        logger.info(
            "  Gradient Accumulation steps = %s",
            self.config.training.gradient_accumulation_steps,
        )
        logger.info("  Total optimization steps = %s", self.config.training.max_train_steps)
        logger.info("  Stage-1 source = %s", self.init_info.get("resolved_stage1_checkpoint"))
        logger.info("  Prompt mode = %s", self.config.prompt.prompt_mode)
        logger.info("  Area loss enabled = %s", self.config.area_loss.enabled)
        logger.info(
            "  Active region resolutions = %s",
            self.config.region.active_region_resolutions,
        )

        self.resume_from_checkpoint()
        progress_bar = tqdm(
            range(0, self.config.training.max_train_steps),
            initial=self.global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.first_epoch, self.config.training.num_train_epochs):
            train_loss = self._train_epoch(epoch, progress_bar)

            if (
                self.accelerator.is_main_process
                and self.validation_batch is not None
                and epoch % self.config.validation.validation_epochs == 0
            ):
                self._run_validation(epoch)

            if self.global_step >= self.config.training.max_train_steps:
                break

        self._finalize_training()

    def _train_epoch(self, epoch: int, progress_bar) -> float:
        del epoch
        self.models.unet.train()
        train_loss = 0.0

        for step, batch in enumerate(self.train_dataloader):
            if self.global_step == 0 and step == 0 and self.accelerator.is_main_process:
                self._debug_save_batch(batch)

            loss = self._train_step(batch)
            avg_loss = self.accelerator.gather(loss.repeat(self.config.data.batch_size)).mean()
            train_loss += avg_loss.item() / self.config.training.gradient_accumulation_steps

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.global_step += 1
                self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                train_loss = 0.0
                self._maybe_save_checkpoint()

            logs = {
                "step_loss": loss.detach().item(),
                "adapter_lr": self.optimizer.param_groups[0]["lr"],
            }
            if len(self.optimizer.param_groups) > 1:
                logs["backbone_lr"] = self.optimizer.param_groups[1]["lr"]
            progress_bar.set_postfix(**logs)

            if self.global_step >= self.config.training.max_train_steps:
                break

        return train_loss

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with self.accelerator.accumulate(self.models.unet):
            pixel_values = batch["pixel_values"].to(device=self.accelerator.device)
            latents = self.models.vae.encode(
                pixel_values.to(dtype=next(self.models.vae.parameters()).dtype)
            ).latent_dist.sample()
            latents = latents * self.models.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            if self.config.training.noise_offset:
                noise += self.config.training.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device,
                )

            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                self.models.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=latents.device,
            ).long()
            noisy_latents = self.models.noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = self.models.text_encoder(
                batch["input_ids"].to(device=self.accelerator.device),
                return_dict=False,
            )[0]

            target = self._get_prediction_target(latents, noise, timesteps)
            model_pred = self.models.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                cross_attention_kwargs={
                    "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(device=latents.device, dtype=latents.dtype),
                    "labels": batch["labels"].to(device=latents.device),
                    "object_mask": batch["object_mask"].to(device=latents.device),
                },
                return_dict=False,
            )[0]

            loss = self._compute_loss(
                model_pred=model_pred,
                target=target,
                timesteps=timesteps,
                boxes_xyxy_norm=batch["boxes_xyxy_norm"].to(device=latents.device, dtype=latents.dtype),
                object_mask=batch["object_mask"].to(device=latents.device),
            )
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    [param for param in self.models.unet.parameters() if param.requires_grad],
                    self.config.training.max_grad_norm,
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
        if self.config.training.prediction_type is not None:
            self.models.noise_scheduler.register_to_config(
                prediction_type=self.config.training.prediction_type
            )

        prediction_type = self.models.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "v_prediction":
            return self.models.noise_scheduler.get_velocity(latents, noise, timesteps)
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    def _compute_loss(
        self,
        *,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        boxes_xyxy_norm: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")

        if self.config.area_loss.enabled:
            area_weights = build_area_weight_map(
                boxes_xyxy_norm=boxes_xyxy_norm,
                object_mask=object_mask,
                latent_height=loss.shape[-2],
                latent_width=loss.shape[-1],
                alpha=self.config.area_loss.alpha,
                background_weight=self.config.area_loss.background_weight,
                min_weight=self.config.area_loss.min_weight,
                max_weight=self.config.area_loss.max_weight,
            )
            loss = loss * area_weights

        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        if self.config.training.snr_gamma is None:
            return loss.mean()

        snr = compute_snr(self.models.noise_scheduler, timesteps)
        mse_loss_weights = torch.stack(
            [
                snr,
                self.config.training.snr_gamma * torch.ones_like(timesteps),
            ],
            dim=1,
        ).min(dim=1)[0]

        prediction_type = self.models.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        return (loss * mse_loss_weights).mean()

    def _maybe_save_checkpoint(self) -> None:
        if self.global_step % self.config.training.checkpointing_steps != 0:
            return
        if not self.accelerator.is_main_process:
            return

        if self.config.training.checkpoints_total_limit is not None:
            checkpoints = [
                entry
                for entry in os.listdir(self.config.output.output_dir)
                if entry.startswith("checkpoint-")
            ]
            checkpoints = sorted(checkpoints, key=lambda entry: int(entry.split("-")[1]))
            if len(checkpoints) >= self.config.training.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.config.training.checkpoints_total_limit + 1
                for checkpoint in checkpoints[:num_to_remove]:
                    shutil.rmtree(os.path.join(self.config.output.output_dir, checkpoint))

        save_path = os.path.join(self.config.output.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)
        self._write_checkpoint_metadata(Path(save_path))
        logger.info("Saved checkpoint to %s", save_path)

    def _run_validation(self, epoch: int, is_final: bool = False) -> List:
        if self.validation_batch is None:
            return []

        unet = unwrap_model(self.models.unet, self.accelerator)
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.stage1.pretrained_model_name_or_path,
            revision=self.config.stage1.revision,
            variant=self.config.stage1.variant,
            vae=self.models.vae,
            text_encoder=self.models.text_encoder,
            tokenizer=self.models.tokenizer,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=self.models.weight_dtype,
        )
        # The RegionDiff wrapper is a plain nn.Module, so diffusers rejects it if it is
        # passed through from_pretrained(..., unet=...). Swap it in after construction.
        pipeline.unet = unet

        images = log_layout_validation(
            pipeline=pipeline,
            validation_batch=self.validation_batch,
            num_images=min(
                self.config.validation.num_validation_images,
                len(self.validation_batch["prompt_text"]),
            ),
            num_inference_steps=self.config.validation.validation_num_inference_steps,
            guidance_scale=self.config.validation.guidance_scale,
            device=self.accelerator.device,
            seed=self.config.training.seed,
            accelerator=self.accelerator,
            epoch=epoch,
            image_size=self.config.data.resolution,
            is_final=is_final,
        )
        del pipeline
        torch.cuda.empty_cache()
        return images

    def _debug_save_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        pixel_values = batch["pixel_values"]
        print(
            f"pixel_values: {pixel_values.shape} {pixel_values.dtype} "
            f"min={float(pixel_values.min()):.3f} max={float(pixel_values.max()):.3f} "
            f"mean={float(pixel_values.mean()):.3f} std={float(pixel_values.std()):.3f}"
        )
        os.makedirs("artifacts/debug/debug_samples", exist_ok=True)
        save_image((pixel_values[:4].cpu() + 1) / 2, "artifacts/debug/debug_samples/layout_batch0.png")

    def _finalize_training(self) -> None:
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            save_stage2_layout_artifact(
                output_dir=self.config.output.output_dir,
                unet=unwrap_model(self.models.unet, self.accelerator).to(torch.float32),
                config=self.config,
                init_info=self.init_info,
                trainability_info=self.trainability_info,
            )

            if self.validation_batch is not None:
                self._run_validation(self.config.training.num_train_epochs, is_final=True)

        self.accelerator.end_training()


def log_layout_validation(
    *,
    pipeline,
    validation_batch: Dict[str, object],
    num_images: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
    seed: Optional[int],
    accelerator: Accelerator,
    epoch: int,
    image_size: int,
    is_final: bool = False,
) -> List:
    """Generate fixed-layout validation samples and log them to trackers."""
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device.type)

    images = []
    for index in range(int(num_images)):
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed + index)

        with autocast_ctx:
            image = pipeline(
                validation_batch["prompt_text"][index],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                cross_attention_kwargs={
                    "boxes_xyxy_norm": validation_batch["boxes_xyxy_norm"][index:index + 1].to(device=device),
                    "labels": validation_batch["labels"][index:index + 1].to(device=device),
                    "object_mask": validation_batch["object_mask"][index:index + 1].to(device=device),
                },
            ).images[0]
        images.append(image)

    phase_name = "test" if is_final else "validation"
    layout_canvas = render_class_layout(
        boxes_xyxy=validation_batch["boxes_xyxy"][:num_images],
        labels=validation_batch["labels"][:num_images],
        object_mask=validation_batch["object_mask"][:num_images],
        image_size=image_size,
    )
    np_images_u8 = np.stack([np.asarray(image) for image in images])
    np_images_01 = np_images_u8.astype(np.float32) / 255.0
    generated_tensor = torch.from_numpy(np_images_01).permute(0, 3, 1, 2)
    generated_overlay = draw_bbox_overlays(
        generated_tensor,
        boxes_xyxy=validation_batch["boxes_xyxy"][:num_images],
        object_mask=validation_batch["object_mask"][:num_images],
        labels=validation_batch["labels"][:num_images],
    )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_images(
                f"{phase_name}/generated_rgb_01",
                np_images_01,
                epoch,
                dataformats="NHWC",
            )
            tracker.writer.add_images(
                f"{phase_name}/layout_rgb_01",
                layout_canvas.permute(0, 2, 3, 1).numpy(),
                epoch,
                dataformats="NHWC",
            )
            tracker.writer.add_images(
                f"{phase_name}/generated_with_boxes_rgb_01",
                generated_overlay.permute(0, 2, 3, 1).numpy(),
                epoch,
                dataformats="NHWC",
            )
            tracker.writer.add_scalar(
                f"{phase_name}/generated_mean",
                float(np_images_01.mean()),
                epoch,
            )
            tracker.writer.add_scalar(
                f"{phase_name}/generated_std",
                float(np_images_01.std()),
                epoch,
            )
        elif tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=validation_batch["prompt_text"][index])
                        for index, image in enumerate(images)
                    ]
                }
            )

    return images
