"""Trainer for unconditional latent Stable Diffusion."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn.functional as F

from src.algorithms.inference.unconditional_sd_sampler import UnconditionalStableDiffusionSampler
from src.algorithms.training.flow_matching_trainer import (
    FlowMatchingTrainer,
    _resolve_unet_sample_size,
)
from src.core.diffusers_compat import import_diffusers_attr
from src.core.training_utils import compute_snr
from src.models.fm_unet import build_fm_unet_from_config, load_unet_config
from src.models.regiondiffusion_factory import (
    build_regiondiff_wrapper,
    configure_regiondiff_trainability,
    save_regiondiff_metadata,
)

if TYPE_CHECKING:
    from diffusers import DDPMScheduler


class UnconditionalStableDiffusionTrainer(FlowMatchingTrainer):
    """Config-driven unconditional latent diffusion trainer."""

    def __init__(
        self,
        unet,
        *,
        noise_scheduler: DDPMScheduler,
        diffusion_config,
        device: Optional[str] = None,
        model_dir: str = "./artifacts/checkpoints/stable_diffusion/uncond_runs/uncond_latent_sd15",
        from_norm_to_display=None,
        unet_config: Optional[Dict[str, Any]] = None,
        vae=None,
        vae_config: Optional[Dict[str, Any]] = None,
        layout_config=None,
        regiondiff_trainability_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            unet,
            device=device,
            t_scale=1.0,
            train_target="v",
            model_dir=model_dir,
            from_norm_to_display=from_norm_to_display,
            unet_config=unet_config,
            vae=vae,
            vae_config=vae_config,
            conditioner=None,
            layout_config=layout_config,
            regiondiff_trainability_info=regiondiff_trainability_info,
        )
        self.noise_scheduler = noise_scheduler
        self.diffusion_config = diffusion_config

    def _metric_prefix(self) -> str:
        return "sd_uncond"

    def _checkpoint_stem(self) -> str:
        return "unet_sd_uncond"

    def _progress_label(self) -> str:
        return "SDUncond"

    def _scheduler_dir(self) -> str:
        return os.path.join(self.model_dir, "SCHEDULER")

    def _save_additional_configs(self) -> None:
        os.makedirs(self._scheduler_dir(), exist_ok=True)
        self.noise_scheduler.save_pretrained(self._scheduler_dir())
        if self._uses_regiondiff_layout():
            save_regiondiff_metadata(
                self.unet,
                self.model_dir,
                extra={"trainability": self.regiondiff_trainability_info},
            )

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "num_train_timesteps": int(self.noise_scheduler.config.num_train_timesteps),
            "prediction_type": str(self.noise_scheduler.config.prediction_type),
            "beta_schedule": str(self.noise_scheduler.config.beta_schedule),
        }

    @classmethod
    def from_config(
        cls,
        config,
        *,
        from_norm_to_display=None,
    ) -> "UnconditionalStableDiffusionTrainer":
        from src.models.vae import build_vae_from_config, resolve_vae_config_from_model_config

        device = config.resolved_device() if hasattr(config, "resolved_device") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        vae_cfg = resolve_vae_config_from_model_config(config.model)
        if vae_cfg is None:
            raise ValueError("Unconditional latent SD requires a VAE config or pretrained VAE.")

        unet_cfg = dict(load_unet_config(config.model.unet_config))
        unet_cfg["sample_size"] = _resolve_unet_sample_size(config, vae_cfg)
        latent_channels = int(vae_cfg.get("latent_channels", unet_cfg.get("in_channels", 4)))
        unet_cfg["in_channels"] = latent_channels
        unet_cfg["out_channels"] = latent_channels

        unet = build_fm_unet_from_config(unet_cfg, device=device)
        vae = build_vae_from_config(vae_cfg, device=device)
        noise_scheduler = cls.build_noise_scheduler(config.diffusion)
        layout_config = getattr(config, "layout_conditioning", None)
        regiondiff_trainability_info = None
        if (
            layout_config is not None
            and bool(getattr(layout_config, "enabled", False))
            and str(getattr(layout_config, "variant", "")) == "regiondiff_v1"
        ):
            unet = build_regiondiff_wrapper(
                base_model=unet,
                region_config=layout_config,
                category_id_to_name=getattr(layout_config, "category_id_to_name", {}),
                num_classes=getattr(layout_config, "num_classes", None),
                backbone_kind="sd_uncond_unet2d",
                attachment_kind="attention",
            ).to(device)
            regiondiff_trainability_info = configure_regiondiff_trainability(
                wrapper=unet,
                train_mode=str(getattr(layout_config, "train_mode", "adapters_only")),
                partial_backbone_modules=getattr(layout_config, "partial_backbone_modules", []),
                mixed_precision=getattr(getattr(config, "precision", None), "mixed_precision", None),
            )

        return cls(
            unet,
            noise_scheduler=noise_scheduler,
            diffusion_config=config.diffusion,
            device=device,
            model_dir=config.output.model_dir,
            from_norm_to_display=from_norm_to_display,
            unet_config=unet_cfg,
            vae=vae,
            vae_config=vae_cfg,
            layout_config=layout_config,
            regiondiff_trainability_info=regiondiff_trainability_info,
        )

    @staticmethod
    def build_noise_scheduler(diffusion_config) -> DDPMScheduler:
        """Construct the DDPM scheduler used for training and sampling."""
        DDPMScheduler = import_diffusers_attr("diffusers", "DDPMScheduler")
        return DDPMScheduler(
            num_train_timesteps=int(diffusion_config.num_train_timesteps),
            beta_schedule=str(diffusion_config.beta_schedule),
            beta_start=float(diffusion_config.beta_start),
            beta_end=float(diffusion_config.beta_end),
            prediction_type=str(diffusion_config.prediction_type),
            clip_sample=False,
        )

    def train_from_config(
        self,
        config,
        dataloader,
        eval_dataloader=None,
    ) -> None:
        pretrained_vae_path = config.model.vae_weights
        if getattr(config.model, "vae_pretrained_model_name_or_path", None):
            pretrained_vae_path = None

        self.train(
            dataloader=dataloader,
            epochs=config.training.epochs,
            eval_dataloader=eval_dataloader,
            pretrained_vae_path=pretrained_vae_path,
            pretrained_unet_path=config.model.pretrained_unet_path,
            strict_load=config.training.strict_load,
            log_dir=config.output.resolved_log_dir(),
            sample_every=config.sampling.sample_every,
            sample_steps=config.sampling.sample_steps,
            sample_batch_size=config.sampling.sample_batch_size,
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            sample_shape=config.sampling.sample_shape,
            save_every_n_epochs=config.training.save_every_n_epochs,
            eval_every=config.training.eval_every,
            resume_from_checkpoint=config.output.resume,
            lr=config.resolved_lr(),
            optimizer_name=getattr(config.optimizer, "name", "adamw"),
            weight_decay=getattr(config.optimizer, "weight_decay", 0.01),
            beta1=getattr(config.optimizer, "beta1", 0.9),
            beta2=getattr(config.optimizer, "beta2", 0.999),
            scheduler_name=getattr(config.scheduler, "name", "warmup_cosine"),
            warmup_ratio=getattr(config.scheduler, "warmup_ratio", 0.05),
            min_lr_ratio=getattr(config.scheduler, "min_lr_ratio", 0.1),
            ema_enabled=getattr(config.ema, "enabled", True),
            ema_decay=getattr(config.ema, "decay", 0.999),
            ema_start_step=getattr(config.ema, "start_step", 100),
            mixed_precision=getattr(config.precision, "mixed_precision", "auto"),
            max_grad_norm=getattr(config.training, "max_grad_norm", 1.0),
            fixed_validation_examples=getattr(config.sampling, "fixed_validation_examples", 0),
            early_sanity_sample_epoch=getattr(config.sampling, "early_sanity_sample_epoch", 0),
            save_debug_images=getattr(config.sampling, "save_debug_images", False),
            debug_dir=config.output.resolved_debug_dir(),
            gradient_accumulation_steps=getattr(config.training, "gradient_accumulation_steps", 1),
        )

    def _make_sampler(self) -> UnconditionalStableDiffusionSampler:
        return UnconditionalStableDiffusionSampler.from_stable(
            self.unet,
            self.vae,
            self.noise_scheduler,
            device=self.device,
            from_norm_to_display=self.from_norm_to_display,
        )

    def _get_prediction_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "v_prediction":
            return self.noise_scheduler.get_velocity(latents, noise, timesteps)
        raise ValueError(f"Unknown prediction_type={prediction_type!r}")

    def _compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        snr_gamma = getattr(self.diffusion_config, "snr_gamma", None)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = self._apply_regiondiff_area_loss_weights(loss, cond_kwargs)
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        if snr_gamma is None:
            return loss.mean()

        snr = compute_snr(self.noise_scheduler, timesteps)
        mse_loss_weights = torch.stack(
            [snr, float(snr_gamma) * torch.ones_like(timesteps)],
            dim=1,
        ).min(dim=1)[0]

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        return (loss * mse_loss_weights).mean()

    def _predict_x0_from_model_pred(
        self,
        noisy_latents: torch.Tensor,
        model_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the clean latent implied by the diffusion prediction."""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        timesteps = timesteps.long()
        alpha_prod_t = alphas_cumprod[timesteps]
        beta_prod_t = 1.0 - alpha_prod_t
        while alpha_prod_t.ndim < noisy_latents.ndim:
            alpha_prod_t = alpha_prod_t[..., None]
            beta_prod_t = beta_prod_t[..., None]

        sqrt_alpha_prod = alpha_prod_t.sqrt()
        sqrt_beta_prod = beta_prod_t.clamp(min=0.0).sqrt()
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            return (noisy_latents - sqrt_beta_prod * model_pred) / sqrt_alpha_prod.clamp(min=1e-8)
        if prediction_type == "v_prediction":
            return sqrt_alpha_prod * noisy_latents - sqrt_beta_prod * model_pred
        if prediction_type == "sample":
            return model_pred
        raise ValueError(f"Unknown prediction_type={prediction_type!r}")

    def _compute_regiondiff_x0_area_loss(
        self,
        *,
        noisy_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        model_pred: torch.Tensor,
        timesteps: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Apply RegionDiff area weighting to clean-latent reconstruction."""
        if not self._uses_regiondiff_area_loss():
            return model_pred.new_zeros(())
        pred_x0 = self._predict_x0_from_model_pred(noisy_latents, model_pred, timesteps)
        loss = F.mse_loss(pred_x0.float(), clean_latents.float(), reduction="none")
        loss = self._apply_regiondiff_area_loss_weights(loss, cond_kwargs)
        return loss.mean()

    def diffusion_step(
        self,
        latents: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute one unconditional diffusion training loss in latent space."""
        if cond_kwargs is None:
            cond_kwargs = {}
        noise = torch.randn_like(latents)
        noise_offset = float(getattr(self.diffusion_config, "noise_offset", 0.0) or 0.0)
        if noise_offset:
            noise += noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1),
                device=latents.device,
            )

        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latents.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        target = self._get_prediction_target(latents, noise, timesteps)
        model_pred = self.unet(noisy_latents, timesteps, **cond_kwargs).sample
        loss = self._compute_loss(model_pred, target, timesteps, cond_kwargs)
        x0_loss_weight = float(getattr(self.layout_config, "area_x0_loss_weight", 1.0))
        if x0_loss_weight:
            loss = loss + x0_loss_weight * self._compute_regiondiff_x0_area_loss(
                noisy_latents=noisy_latents,
                clean_latents=latents,
                model_pred=model_pred,
                timesteps=timesteps,
                cond_kwargs=cond_kwargs,
            )
        return loss

    def _compute_batch_loss(
        self,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return self.diffusion_step(x_fm, cond_kwargs)


from src.core.registry import REGISTRIES  # noqa: E402


REGISTRIES.trainer.register("sd_uncond")(UnconditionalStableDiffusionTrainer)
