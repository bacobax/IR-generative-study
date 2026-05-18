"""Structured configuration for unconditional latent Stable Diffusion training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch

from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.fm_config import (
    AugmentConfig,
    DataConfig,
    EMAConfig,
    LayoutConditioningConfig,
    ModelConfig,
    OptimizerConfig,
    PrecisionConfig,
    SampleConfig,
    SchedulerConfig,
)
from src.core.data.dataset_targets import supported_dataset_ids
from src.core.paths import sd_uncond_runs_dir


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


@dataclass
class SDUncondTrainingConfig:
    """Core training hyper-parameters for unconditional latent SD."""

    epochs: int = 100
    lr: float = 1e-4
    save_every_n_epochs: int = 10
    eval_every: int = 1
    patience: Optional[int] = None
    min_delta: float = 0.0
    strict_load: bool = True
    max_grad_norm: float = 1.0


@dataclass
class DiffusionConfig:
    """Noise schedule and loss configuration."""

    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    prediction_type: str = "epsilon"
    noise_offset: float = 0.0
    snr_gamma: Optional[float] = None


@dataclass
class SDUncondOutputConfig:
    """Checkpoint, log, and debug directories."""

    model_dir: str = str(sd_uncond_runs_dir() / "uncond_latent_sd15")
    log_dir: Optional[str] = None
    debug_dir: Optional[str] = None
    resume: Optional[str] = None

    def resolved_log_dir(self) -> str:
        if self.log_dir is not None:
            return self.log_dir
        return f"{self.model_dir}/runs/stable_diffusion_uncond_logs/"

    def resolved_debug_dir(self) -> str:
        if self.debug_dir is not None:
            return self.debug_dir
        return f"{self.model_dir}/debug_samples/"


@dataclass
class SDUncondTrainConfig:
    """Complete unconditional latent SD training configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: SDUncondTrainingConfig = field(default_factory=SDUncondTrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    layout_conditioning: LayoutConditioningConfig = field(default_factory=LayoutConditioningConfig)
    sampling: SampleConfig = field(default_factory=SampleConfig)
    output: SDUncondOutputConfig = field(default_factory=SDUncondOutputConfig)
    trainer_name: Optional[str] = "sd_uncond"
    sampler_name: Optional[str] = "sd_uncond"
    device: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolved_lr(self) -> float:
        if self.optimizer.lr is not None:
            return float(self.optimizer.lr)
        return float(self.training.lr)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for unconditional latent SD training."""
    parser = argparse.ArgumentParser(
        description="Unconditional latent Stable Diffusion training"
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI flags override config values.")

    parser.add_argument("--dataset_id", type=str, default=None,
                        choices=sorted(supported_dataset_ids()),
                        help="Named dataset target (overrides train_dir/val_dir when set)")
    parser.add_argument("--train_dir", type=str, default="./data/raw/v18/train/",
                        help="Path to training data")
    parser.add_argument("--val_dir", type=str, default="./data/raw/v18/val/",
                        help="Path to validation data")
    parser.add_argument("--annotations_path", type=str, default=None,
                        help="Optional COCO annotations path for shared dataset resolution.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Square resize target. Must be a positive multiple of 32.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Deterministic limit on training samples (debug only).")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Deterministic limit on validation samples (debug only).")
    parser.add_argument("--subset_strategy", type=str, default="first_n",
                        help="Dataset subsetting policy for debug runs.")
    parser.add_argument("--subset_manifest", type=str, default=None,
                        help="Optional manifest selecting the training split subset.")

    parser.add_argument("--unet_config", type=str, default="configs/models/fm/stable_unet_config.json",
                        help="UNet config JSON")
    parser.add_argument("--vae_config", type=str, default="configs/models/fm/vae_config.json",
                        help="VAE config JSON")
    parser.add_argument("--vae_weights", type=str, default="./vae_best.pt",
                        help="Pretrained VAE weights")
    parser.add_argument("--vae_pretrained_model_name_or_path", type=str, default=None,
                        help="Optional diffusers model id/path used to load a pretrained VAE.")
    parser.add_argument("--vae_pretrained_subfolder", type=str, default="vae",
                        help="Subfolder containing the pretrained diffusers VAE.")
    parser.add_argument("--vae_revision", type=str, default=None,
                        help="Optional pretrained VAE revision.")
    parser.add_argument("--vae_variant", type=str, default=None,
                        help="Optional pretrained VAE variant.")
    parser.add_argument("--pretrained_unet_path", type=str, default=None,
                        help="Optional UNet checkpoint to initialize from.")

    parser.add_argument("--model_dir", type=str, default=str(sd_uncond_runs_dir() / "uncond_latent_sd15"),
                        help="Model output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_every_n_epochs", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Run validation every N epochs. Set <= 0 to disable.")
    parser.add_argument("--sample_every", type=int, default=1,
                        help="Log generated samples every N epochs. Set <= 0 to disable.")
    parser.add_argument("--sample_steps", type=int, default=50,
                        help="Number of diffusion steps for validation sampling.")
    parser.add_argument("--sample_batch_size", type=int, default=4,
                        help="Batch size for validation sampling")
    parser.add_argument("--early_sanity_sample_epoch", type=int, default=0,
                        help="Log an extra validation sample batch every N epochs. Set <= 0 to disable.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay.")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam/AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam/AdamW beta2.")
    parser.add_argument("--scheduler_name", type=str, default="warmup_cosine",
                        choices=["none", "warmup_cosine", "constant_with_warmup"],
                        help="Learning-rate scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio over total optimizer steps.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Final LR ratio for cosine schedules.")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay.")
    parser.add_argument("--ema_start_step", type=int, default=100,
                        help="Start EMA updates after this many steps.")
    parser.add_argument("--mixed_precision", type=str, default="auto",
                        choices=["auto", "bf16", "fp16", "no"],
                        help="Mixed precision mode.")

    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                        help="Number of training timesteps for the DDPM scheduler.")
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear",
                        help="DDPM beta schedule.")
    parser.add_argument("--beta_start", type=float, default=0.00085,
                        help="DDPM beta schedule start.")
    parser.add_argument("--beta_end", type=float, default=0.012,
                        help="DDPM beta schedule end.")
    parser.add_argument("--prediction_type", type=str, default="epsilon",
                        choices=["epsilon", "v_prediction"],
                        help="Prediction target used by the diffusion scheduler.")
    parser.add_argument("--noise_offset", type=float, default=0.0,
                        help="Optional noise offset added during training.")
    parser.add_argument("--snr_gamma", type=float, default=None,
                        help="Optional SNR loss reweighting gamma.")

    parser.add_argument("--layout_conditioning_enabled", action="store_true",
                        help="Enable RegionDiff layout conditioning.")
    parser.add_argument("--layout_conditioning_variant", type=str, default=None,
                        help="Layout conditioning variant, e.g. regiondiff_v1.")
    parser.add_argument("--active_region_resolutions", type=_csv_ints, default=None,
                        help="Comma-separated RegionDiff latent resolutions, e.g. 64,32,16.")
    parser.add_argument("--layout_token_dim", type=int, default=None)
    parser.add_argument("--bbox_fourier_dim", type=int, default=None)
    parser.add_argument("--same_class_position_slots", type=int, default=None)
    parser.add_argument("--use_background_token", type=_str2bool, default=None)
    parser.add_argument("--layout_train_mode", type=str, default=None)
    parser.add_argument("--partial_backbone_modules", nargs="+", default=None)
    parser.add_argument("--adapter_learning_rate", type=float, default=None)
    parser.add_argument("--backbone_learning_rate", type=float, default=None)
    parser.add_argument("--area_loss_enabled", type=_str2bool, default=None)
    parser.add_argument("--area_loss_alpha", type=float, default=None)
    parser.add_argument("--area_loss_background_weight", type=float, default=None)
    parser.add_argument("--area_loss_min_weight", type=float, default=None)
    parser.add_argument("--area_loss_max_weight", type=float, default=None)
    parser.add_argument("--area_x0_loss_weight", type=float, default=None)

    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--ramp_frac", type=float, default=0.3)
    parser.add_argument("--p_crop_warmup", type=float, default=0.05)
    parser.add_argument("--p_crop_max", type=float, default=0.20)
    parser.add_argument("--p_crop_final", type=float, default=0.05)
    parser.add_argument("--p_rot_warmup", type=float, default=0.05)
    parser.add_argument("--p_rot_max", type=float, default=0.30)
    parser.add_argument("--p_rot_final", type=float, default=0.05)
    parser.add_argument("--p_hflip_warmup", type=float, default=0.0)
    parser.add_argument("--p_hflip_max", type=float, default=0.0)
    parser.add_argument("--p_hflip_final", type=float, default=0.0)

    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, cuda:0, cuda:1, etc.")

    return parser


_FLAT_TO_NESTED = {
    "dataset_id": "data.dataset_id",
    "train_dir": "data.train_dir",
    "val_dir": "data.val_dir",
    "annotations_path": "data.annotations_path",
    "image_size": "data.image_size",
    "batch_size": "data.batch_size",
    "num_workers": "data.num_workers",
    "max_train_samples": "data.max_train_samples",
    "max_val_samples": "data.max_val_samples",
    "subset_strategy": "data.subset_strategy",
    "subset_manifest": "data.subset_manifest",
    "unet_config": "model.unet_config",
    "vae_config": "model.vae_config",
    "vae_weights": "model.vae_weights",
    "vae_pretrained_model_name_or_path": "model.vae_pretrained_model_name_or_path",
    "vae_pretrained_subfolder": "model.vae_pretrained_subfolder",
    "vae_revision": "model.vae_revision",
    "vae_variant": "model.vae_variant",
    "pretrained_unet_path": "model.pretrained_unet_path",
    "model_dir": "output.model_dir",
    "resume": "output.resume",
    "epochs": "training.epochs",
    "lr": "training.lr",
    "save_every_n_epochs": "training.save_every_n_epochs",
    "eval_every": "training.eval_every",
    "max_grad_norm": "training.max_grad_norm",
    "weight_decay": "optimizer.weight_decay",
    "beta1": "optimizer.beta1",
    "beta2": "optimizer.beta2",
    "scheduler_name": "scheduler.name",
    "warmup_ratio": "scheduler.warmup_ratio",
    "min_lr_ratio": "scheduler.min_lr_ratio",
    "ema_decay": "ema.decay",
    "ema_start_step": "ema.start_step",
    "mixed_precision": "precision.mixed_precision",
    "num_train_timesteps": "diffusion.num_train_timesteps",
    "beta_schedule": "diffusion.beta_schedule",
    "beta_start": "diffusion.beta_start",
    "beta_end": "diffusion.beta_end",
    "prediction_type": "diffusion.prediction_type",
    "noise_offset": "diffusion.noise_offset",
    "snr_gamma": "diffusion.snr_gamma",
    "layout_conditioning_enabled": "layout_conditioning.enabled",
    "layout_conditioning_variant": "layout_conditioning.variant",
    "active_region_resolutions": "layout_conditioning.active_region_resolutions",
    "layout_token_dim": "layout_conditioning.layout_token_dim",
    "bbox_fourier_dim": "layout_conditioning.bbox_fourier_dim",
    "same_class_position_slots": "layout_conditioning.same_class_position_slots",
    "use_background_token": "layout_conditioning.use_background_token",
    "layout_train_mode": "layout_conditioning.train_mode",
    "partial_backbone_modules": "layout_conditioning.partial_backbone_modules",
    "adapter_learning_rate": "layout_conditioning.adapter_learning_rate",
    "backbone_learning_rate": "layout_conditioning.backbone_learning_rate",
    "area_loss_enabled": "layout_conditioning.area_loss_enabled",
    "area_loss_alpha": "layout_conditioning.area_loss_alpha",
    "area_loss_background_weight": "layout_conditioning.area_loss_background_weight",
    "area_loss_min_weight": "layout_conditioning.area_loss_min_weight",
    "area_loss_max_weight": "layout_conditioning.area_loss_max_weight",
    "area_x0_loss_weight": "layout_conditioning.area_x0_loss_weight",
    "warmup_frac": "augment.warmup_frac",
    "ramp_frac": "augment.ramp_frac",
    "p_crop_warmup": "augment.p_crop_warmup",
    "p_crop_max": "augment.p_crop_max",
    "p_crop_final": "augment.p_crop_final",
    "p_rot_warmup": "augment.p_rot_warmup",
    "p_rot_max": "augment.p_rot_max",
    "p_rot_final": "augment.p_rot_final",
    "p_hflip_warmup": "augment.p_hflip_warmup",
    "p_hflip_max": "augment.p_hflip_max",
    "p_hflip_final": "augment.p_hflip_final",
    "sample_every": "sampling.sample_every",
    "sample_steps": "sampling.sample_steps",
    "sample_batch_size": "sampling.sample_batch_size",
    "early_sanity_sample_epoch": "sampling.early_sanity_sample_epoch",
    "device": "device",
}


def parse_args(argv: Optional[list] = None) -> SDUncondTrainConfig:
    """Parse CLI flags and return a structured unconditional-SD config."""
    import sys

    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(effective_argv)
    return merge_config_and_cli(
        SDUncondTrainConfig,
        args.config,
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
        cli_argv=effective_argv,
    )
