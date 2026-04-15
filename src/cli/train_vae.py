"""Modular CLI entrypoint for VAE training.

This module is the **source of truth** for VAE training on NPY datasets.
The root-level ``train_vae.py`` is a thin compatibility wrapper that
forwards to :func:`main` here.

Usage::

    python -m src.cli.train_vae --train-dir ./data/raw/v18/train/ --val-dir ./data/raw/v18/val/
    # or via the legacy wrapper:
    python train_vae.py --train-dir ./data/raw/v18/train/ --val-dir ./data/raw/v18/val/
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.core.normalization import (
    RAW_UINT16_PERCENTILE,
    UINT8_LINEAR,
    norm_to_display as from_norm_to_display,
    norm_to_uint16,
    resize_and_normalize,
)
from src.core.data.datasets import NPYImageDataset
from src.models.vae import (
    build_vae_from_config,
    load_diffusers_vae_config,
    load_vae_config,
    save_vae_config,
    save_vae_weights,
)
from src.core.gpu_utils import get_least_used_cuda_gpu
from src.core.configs.config_loader import apply_yaml_defaults
from src.core.training_utils import (
    EMAState,
    autocast_context,
    build_grad_scaler,
    build_scheduler,
    grad_norm,
    move_optimizer_state_to_device,
    resolve_precision_settings,
)


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE on NPY dataset.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI flags override config values.")
    parser.add_argument("--train-dir", type=str, default=None, help="Path to train .npy files.")
    parser.add_argument("--val-dir", type=str, default=None, help="Optional path to val .npy files.")
    parser.add_argument("--image-size", type=int, default=256, help="Square image size.")
    parser.add_argument(
        "--normalization-mode",
        type=str,
        default=RAW_UINT16_PERCENTILE,
        choices=[RAW_UINT16_PERCENTILE, UINT8_LINEAR],
        help="Input normalization applied after resize.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory",
                        help="Disable DataLoader pin_memory.")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--t-scale", type=int, default=1000, help="Pipeline t_scale (saved in config).")
    parser.add_argument("--model-dir", type=str, default="artifacts/checkpoints/vae/vae_runs/vae_fm_x8", help="VAE model dir.")
    parser.add_argument("--vae-json", type=str, default="configs/models/fm/vae_config.json", help="VAE config JSON.")
    parser.add_argument(
        "--vae-pretrained-model-name-or-path",
        type=str,
        default=None,
        help="Optional diffusers model id/path used to load a pretrained VAE.",
    )
    parser.add_argument(
        "--vae-pretrained-subfolder",
        type=str,
        default="vae",
        help="Subfolder containing the pretrained diffusers VAE.",
    )
    parser.add_argument("--vae-revision", type=str, default=None, help="Optional pretrained VAE revision.")
    parser.add_argument("--vae-variant", type=str, default=None, help="Optional pretrained VAE variant.")
    parser.add_argument("--log-dir", type=str, default="./artifacts/runs/main/autoencoder_kl", help="TensorBoard log dir.")
    parser.add_argument("--patience", type=int, default=4, help="Early-stopping patience.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Early-stopping min delta.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2.")
    parser.add_argument("--scheduler", type=str, default="warmup_cosine",
                        choices=["none", "warmup_cosine", "constant_with_warmup"],
                        help="Learning-rate scheduler.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Warmup ratio over total optimizer steps.")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                        help="Minimum LR ratio for cosine schedule tail.")
    parser.add_argument("--recon-loss", type=str, default="l1_mse",
                        choices=["l1_mse"],
                        help="Reconstruction loss family.")
    parser.add_argument("--recon-mse-weight", type=float, default=0.25,
                        help="Relative weight on reconstruction MSE.")
    parser.add_argument("--kl-start-weight", type=float, default=0.0,
                        help="Initial KL weight.")
    parser.add_argument("--kl-end-weight", type=float, default=1e-5,
                        help="Final KL weight after warmup.")
    parser.add_argument("--kl-warmup-ratio", type=float, default=0.1,
                        help="Fraction of training steps used to warm KL from start to end.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--mixed-precision", type=str, default="auto",
                        choices=["auto", "bf16", "fp16", "no"],
                        help="Mixed precision mode.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay.")
    parser.add_argument("--ema-start-step", type=int, default=100, help="EMA start step.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from VAE checkpoint.")
    parser.add_argument("--save-every-n-epochs", type=int, default=10,
                        help="Save plain VAE weights every N epochs.")
    parser.add_argument("--device", type=str, default=None,
                        help="Explicit device override, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--gpu-prefer", type=str, default="memory", help="GPU selection preference.")
    parser.add_argument("--min-free-mb", type=int, default=4096, help="Minimum free GPU memory.")
    # Two-pass parse: first grab --config, apply YAML defaults, then re-parse
    preliminary, _ = parser.parse_known_args()
    apply_yaml_defaults(parser, preliminary.config)
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resize_and_normalize(
    x: torch.Tensor,
    size: int,
    normalization_mode: str,
) -> torch.Tensor:
    return resize_and_normalize(
        x,
        image_size=size,
        normalization_mode=normalization_mode,
    )


def _resize_raw(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(
        x.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


class _VAEReconstructionDataset(Dataset):
    """Return both normalized inputs and raw resized targets for VAE training."""

    def __init__(self, root_dir: str, image_size: int, normalization_mode: str):
        self.base_dataset = NPYImageDataset(root_dir=root_dir, transform=None)
        self.image_size = int(image_size)
        self.normalization_mode = normalization_mode

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.base_dataset[idx]
        raw_resized = _resize_raw(x, self.image_size)
        normalized = _resize_and_normalize(x, self.image_size, self.normalization_mode)
        return {
            "normalized": normalized,
            "raw_resized": raw_resized,
        }


def _build_dataloader(
    root_dir: str,
    image_size: int,
    normalization_mode: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = _VAEReconstructionDataset(
        root_dir=root_dir,
        image_size=image_size,
        normalization_mode=normalization_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _resolve_vae_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.vae_pretrained_model_name_or_path:
        return load_diffusers_vae_config(
            args.vae_pretrained_model_name_or_path,
            subfolder=args.vae_pretrained_subfolder,
            revision=args.vae_revision,
            variant=args.vae_variant,
        )
    return load_vae_config(path=args.vae_json)


def _stretch_for_vis(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image min/max stretch to [0, 1] for TensorBoard."""
    x = from_norm_to_display(x).clamp(0, 1)
    flat = x.view(x.size(0), -1)
    mn = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    mx = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    return ((x - mn) / (mx - mn + eps)).clamp(0, 1)


def _compute_kl_loss(mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) for diagonal Gaussian posterior stats."""
    variance = sigma.float().pow(2).clamp(min=eps)
    mu = mu.float()
    return 0.5 * torch.mean(mu.pow(2) + variance - variance.log() - 1.0)


def _kl_weight_at_step(
    step: int,
    *,
    total_steps: int,
    start_weight: float,
    end_weight: float,
    warmup_ratio: float,
) -> float:
    warmup_steps = max(1, int(round(float(warmup_ratio) * max(1, total_steps))))
    if step >= warmup_steps:
        return float(end_weight)
    alpha = float(step) / float(max(1, warmup_steps))
    return float(start_weight) + alpha * (float(end_weight) - float(start_weight))


def _compute_psnr(
    x: torch.Tensor,
    recon: torch.Tensor,
    *,
    data_range: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    mse = torch.mean((x.float() - recon.float()).pow(2), dim=(1, 2, 3)).clamp(min=eps)
    return 10.0 * torch.log10((float(data_range) ** 2) / mse)


def _compute_ssim(
    x: torch.Tensor,
    recon: torch.Tensor,
    *,
    data_range: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cheap global SSIM for grayscale monitoring."""
    x = x.float()
    recon = recon.float()
    mu_x = x.mean(dim=(2, 3), keepdim=True)
    mu_y = recon.mean(dim=(2, 3), keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_y = ((recon - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_xy = ((x - mu_x) * (recon - mu_y)).mean(dim=(2, 3), keepdim=True)
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return ssim.mean(dim=(1, 2, 3))


def _unpack_vae_batch(
    batch: Any,
    device: torch.device | str,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, dict):
        normalized = batch.get("normalized")
        raw_resized = batch.get("raw_resized")
        if normalized is None:
            raise KeyError("Expected batch['normalized'] in VAE batch mapping.")
        return normalized.to(device), None if raw_resized is None else raw_resized.to(device)
    return batch.to(device), None


def _raw_metric_data_range(normalization_mode: str) -> Optional[float]:
    if normalization_mode != RAW_UINT16_PERCENTILE:
        return None
    bounds = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    raw_bounds = norm_to_uint16(bounds)
    return float((raw_bounds[1] - raw_bounds[0]).item())


def _compute_raw_reconstruction_metrics(
    raw_target: Optional[torch.Tensor],
    recon: torch.Tensor,
    *,
    normalization_mode: str,
) -> Optional[dict[str, torch.Tensor]]:
    raw_data_range = _raw_metric_data_range(normalization_mode)
    if raw_target is None or raw_data_range is None:
        return None

    recon_raw = norm_to_uint16(recon)
    return {
        "recon_raw": recon_raw,
        "l1": F.l1_loss(recon_raw, raw_target),
        "mse": F.mse_loss(recon_raw, raw_target),
        "psnr": _compute_psnr(raw_target, recon_raw, data_range=raw_data_range),
        "ssim": _compute_ssim(raw_target, recon_raw, data_range=raw_data_range),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_vae(
    vae,
    device,
    dataloader: DataLoader,
    epochs: int,
    eval_dataloader: Optional[DataLoader] = None,
    log_dir: str = "./artifacts/runs/main/autoencoder_kl",
    model_dir: str = "./artifacts/checkpoints/vae/vae_runs/vae_fm_x8",
    patience: Optional[int] = None,
    min_delta: float = 0.0,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    scheduler_name: str = "warmup_cosine",
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.1,
    recon_loss: str = "l1_mse",
    recon_mse_weight: float = 0.25,
    kl_start_weight: float = 0.0,
    kl_end_weight: float = 1e-5,
    kl_warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    mixed_precision: str = "auto",
    ema_decay: float = 0.999,
    ema_start_step: int = 100,
    resume: Optional[str] = None,
    save_every_n_epochs: int = 10,
    normalization_mode: str = RAW_UINT16_PERCENTILE,
):
    """Standalone VAE training loop (no old pipeline dependency)."""
    if eval_dataloader is None:
        raise ValueError("eval_dataloader must be provided.")
    if recon_loss != "l1_mse":
        raise ValueError(f"Unsupported recon_loss={recon_loss!r}.")

    vae_dir = os.path.join(model_dir, "VAE")
    os.makedirs(vae_dir, exist_ok=True)

    total_steps = max(1, epochs * len(dataloader))
    precision = resolve_precision_settings(device, mixed_precision)
    scaler = build_grad_scaler(precision)
    optimizer = AdamW(
        vae.parameters(),
        lr=lr,
        betas=(float(beta1), float(beta2)),
        weight_decay=float(weight_decay),
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )
    ema = EMAState(vae, decay=ema_decay) if ema_decay and ema_decay > 0.0 else None
    writer = SummaryWriter(log_dir)

    best_eval = float("inf")
    best_epoch = -1
    bad_epochs = 0
    global_step = 0
    start_epoch = 0

    if resume is not None:
        print(f"[VAE Resume] Loading checkpoint from {resume}")
        ckpt = torch.load(resume, map_location=device)
        if "vae_state" not in ckpt:
            vae.load_state_dict(ckpt)
            if ema is not None:
                ema = EMAState(vae, decay=ema_decay)
            print("[VAE Resume] Loaded weights only; optimizer/scheduler state reset.")
        else:
            vae.load_state_dict(ckpt["vae_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            move_optimizer_state_to_device(optimizer, device)
            if scheduler is not None and ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if scaler is not None and ckpt.get("scaler_state") is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            if ema is not None:
                if ckpt.get("ema_state") is not None:
                    ema.load_state_dict(ckpt["ema_state"], device=device)
                else:
                    ema = EMAState(vae, decay=ema_decay)
            best_eval = float(ckpt.get("best_eval", best_eval))
            best_epoch = int(ckpt.get("best_epoch", best_epoch))
            bad_epochs = int(ckpt.get("bad_epochs", bad_epochs))
            global_step = int(ckpt.get("global_step", 0))
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            print(
                f"[VAE Resume] epoch={start_epoch}, global_step={global_step}, "
                f"best_eval={best_eval:.6f}"
            )

    def _save_training_checkpoint(path: str, epoch_idx: int) -> None:
        ckpt = {
            "epoch": int(epoch_idx),
            "global_step": int(global_step),
            "vae_state": vae.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "scaler_state": None if scaler is None else scaler.state_dict(),
            "ema_state": None if ema is None else ema.state_dict(),
            "best_eval": float(best_eval),
            "best_epoch": int(best_epoch),
            "bad_epochs": int(bad_epochs),
        }
        torch.save(ckpt, path)

    if precision.requested != precision.mode:
        print(
            f"[VAE Precision] requested={precision.requested!r} -> using {precision.mode!r} "
            f"on device={precision.device_type}"
        )

    for epoch in range(start_epoch, epochs):
        vae.train()
        train_total = train_l1 = train_mse = train_kl = 0.0
        train_psnr = train_ssim = train_grad_norm = 0.0
        train_raw_l1 = train_raw_mse = train_raw_psnr = train_raw_ssim = 0.0
        n_train = 0
        n_train_raw = 0

        for batch in tqdm(dataloader, desc=f"VAE Train {epoch + 1}/{epochs}"):
            x, raw_resized = _unpack_vae_batch(batch, device)
            beta = _kl_weight_at_step(
                global_step,
                total_steps=total_steps,
                start_weight=kl_start_weight,
                end_weight=kl_end_weight,
                warmup_ratio=kl_warmup_ratio,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(precision):
                recon, mu, sigma = vae(x)
                recon_l1 = F.l1_loss(recon, x)
                recon_mse = F.mse_loss(recon, x)
                kl_loss = _compute_kl_loss(mu, sigma)
                loss = recon_l1 + float(recon_mse_weight) * recon_mse + float(beta) * kl_loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            if max_grad_norm and max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_grad_norm)
            batch_grad_norm = grad_norm(vae.parameters())

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
            if ema is not None and global_step >= int(ema_start_step):
                ema.update(vae)

            with torch.no_grad():
                psnr = _compute_psnr(x, recon, data_range=2.0)
                ssim = _compute_ssim(x, recon, data_range=2.0)
                raw_metrics = _compute_raw_reconstruction_metrics(
                    raw_resized,
                    recon,
                    normalization_mode=normalization_mode,
                )
                # Log one un-stretched sanity image early so TensorBoard shows the true display range.
                if global_step == 2:
                    sanity_vis = from_norm_to_display(x[:4]).clamp(0, 1)
                    writer.add_images("sanity/input_step_3_raw_display", sanity_vis, global_step)

            bs = x.size(0)
            train_total += loss.item() * bs
            train_l1 += recon_l1.item() * bs
            train_mse += recon_mse.item() * bs
            train_kl += kl_loss.item() * bs
            train_psnr += float(psnr.mean().item()) * bs
            train_ssim += float(ssim.mean().item()) * bs
            train_grad_norm += float(batch_grad_norm) * bs
            n_train += bs
            if raw_metrics is not None:
                train_raw_l1 += raw_metrics["l1"].item() * bs
                train_raw_mse += raw_metrics["mse"].item() * bs
                train_raw_psnr += float(raw_metrics["psnr"].mean().item()) * bs
                train_raw_ssim += float(raw_metrics["ssim"].mean().item()) * bs
                n_train_raw += bs
            writer.add_scalar("train/total_step", float(loss.item()), global_step)
            writer.add_scalar("train/kl_weight", float(beta), global_step)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
            global_step += 1

        train_total /= max(1, n_train)
        train_l1 /= max(1, n_train)
        train_mse /= max(1, n_train)
        train_kl /= max(1, n_train)
        train_psnr /= max(1, n_train)
        train_ssim /= max(1, n_train)
        train_grad_norm /= max(1, n_train)
        if n_train_raw > 0:
            train_raw_l1 /= n_train_raw
            train_raw_mse /= n_train_raw
            train_raw_psnr /= n_train_raw
            train_raw_ssim /= n_train_raw

        vae.eval()
        eval_total = eval_l1 = eval_mse = eval_kl = 0.0
        eval_psnr = eval_ssim = eval_mu_abs = eval_sigma_mean = 0.0
        eval_raw_l1 = eval_raw_mse = eval_raw_psnr = eval_raw_ssim = 0.0
        n_eval = 0
        n_eval_raw = 0

        eval_context = ema.average_parameters(vae) if ema is not None and global_step >= int(ema_start_step) else torch.no_grad()
        with eval_context:
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc=f"VAE Eval  {epoch + 1}/{epochs}"):
                    x, raw_resized = _unpack_vae_batch(batch, device)

                    with autocast_context(precision):
                        recon, mu, sigma = vae(x)
                        recon_l1 = F.l1_loss(recon, x)
                        recon_mse = F.mse_loss(recon, x)
                        kl_loss = _compute_kl_loss(mu, sigma)
                        loss = recon_l1 + float(recon_mse_weight) * recon_mse + float(kl_end_weight) * kl_loss

                    psnr = _compute_psnr(x, recon, data_range=2.0)
                    ssim = _compute_ssim(x, recon, data_range=2.0)
                    raw_metrics = _compute_raw_reconstruction_metrics(
                        raw_resized,
                        recon,
                        normalization_mode=normalization_mode,
                    )

                    bs = x.size(0)
                    eval_total += loss.item() * bs
                    eval_l1 += recon_l1.item() * bs
                    eval_mse += recon_mse.item() * bs
                    eval_kl += kl_loss.item() * bs
                    eval_psnr += float(psnr.mean().item()) * bs
                    eval_ssim += float(ssim.mean().item()) * bs
                    eval_mu_abs += float(mu.detach().abs().mean().item()) * bs
                    eval_sigma_mean += float(sigma.detach().mean().item()) * bs
                    n_eval += bs
                    if raw_metrics is not None:
                        eval_raw_l1 += raw_metrics["l1"].item() * bs
                        eval_raw_mse += raw_metrics["mse"].item() * bs
                        eval_raw_psnr += float(raw_metrics["psnr"].mean().item()) * bs
                        eval_raw_ssim += float(raw_metrics["ssim"].mean().item()) * bs
                        n_eval_raw += bs

        eval_total /= max(1, n_eval)
        eval_l1 /= max(1, n_eval)
        eval_mse /= max(1, n_eval)
        eval_kl /= max(1, n_eval)
        eval_psnr /= max(1, n_eval)
        eval_ssim /= max(1, n_eval)
        eval_mu_abs /= max(1, n_eval)
        eval_sigma_mean /= max(1, n_eval)
        if n_eval_raw > 0:
            eval_raw_l1 /= n_eval_raw
            eval_raw_mse /= n_eval_raw
            eval_raw_psnr /= n_eval_raw
            eval_raw_ssim /= n_eval_raw

        summary = (
            f"[Epoch {epoch + 1}/{epochs}] "
            f"train: total={train_total:.6f} l1={train_l1:.6f} mse={train_mse:.6f} kl={train_kl:.6f} | "
            f"eval: total={eval_total:.6f} l1={eval_l1:.6f} mse={eval_mse:.6f} kl={eval_kl:.6f}"
        )
        if n_train_raw > 0 or n_eval_raw > 0:
            summary += (
                f" | raw train: l1={train_raw_l1:.6f} mse={train_raw_mse:.6f}"
                f" | raw eval: l1={eval_raw_l1:.6f} mse={eval_raw_mse:.6f}"
            )
        print(summary)

        writer.add_scalar("train/total", train_total, epoch)
        writer.add_scalar("train/recon_l1", train_l1, epoch)
        writer.add_scalar("train/recon_mse", train_mse, epoch)
        writer.add_scalar("train/kl", train_kl, epoch)
        writer.add_scalar("train/psnr", train_psnr, epoch)
        writer.add_scalar("train/ssim", train_ssim, epoch)
        writer.add_scalar("train/grad_norm", train_grad_norm, epoch)
        writer.add_scalar("eval/total", eval_total, epoch)
        writer.add_scalar("eval/recon_l1", eval_l1, epoch)
        writer.add_scalar("eval/recon_mse", eval_mse, epoch)
        writer.add_scalar("eval/kl", eval_kl, epoch)
        writer.add_scalar("eval/psnr", eval_psnr, epoch)
        writer.add_scalar("eval/ssim", eval_ssim, epoch)
        writer.add_scalar("eval/mu_abs_mean", eval_mu_abs, epoch)
        writer.add_scalar("eval/sigma_mean", eval_sigma_mean, epoch)
        if n_train_raw > 0:
            writer.add_scalar("train/raw_recon_l1", train_raw_l1, epoch)
            writer.add_scalar("train/raw_recon_mse", train_raw_mse, epoch)
            writer.add_scalar("train/raw_psnr", train_raw_psnr, epoch)
            writer.add_scalar("train/raw_ssim", train_raw_ssim, epoch)
        if n_eval_raw > 0:
            writer.add_scalar("eval/raw_recon_l1", eval_raw_l1, epoch)
            writer.add_scalar("eval/raw_recon_mse", eval_raw_mse, epoch)
            writer.add_scalar("eval/raw_psnr", eval_raw_psnr, epoch)
            writer.add_scalar("eval/raw_ssim", eval_raw_ssim, epoch)

        if save_every_n_epochs and (epoch + 1) % int(save_every_n_epochs) == 0:
            save_vae_weights(vae, os.path.join(vae_dir, f"vae_epoch_{epoch + 1}.pt"))

        image_context = ema.average_parameters(vae) if ema is not None and global_step >= int(ema_start_step) else torch.no_grad()
        with image_context:
            with torch.no_grad():
                x_tr, _ = _unpack_vae_batch(next(iter(dataloader)), device)
                recon_tr, _, _ = vae(x_tr)
                x_tr_vis = _stretch_for_vis(x_tr[:4])
                recon_tr_vis = _stretch_for_vis(recon_tr[:4])
                writer.add_images("train/input", x_tr_vis, epoch)
                writer.add_images("train/reconstruction", recon_tr_vis, epoch)

                x_ev, _ = _unpack_vae_batch(next(iter(eval_dataloader)), device)
                recon_ev, _, _ = vae(x_ev)
                x_ev_vis = _stretch_for_vis(x_ev[:4])
                recon_ev_vis = _stretch_for_vis(recon_ev[:4])
                writer.add_images("eval/input", x_ev_vis, epoch)
                writer.add_images("eval/reconstruction", recon_ev_vis, epoch)

        improved = (best_eval - eval_total) > min_delta
        if improved:
            best_eval = eval_total
            best_epoch = epoch
            bad_epochs = 0
            if ema is not None and global_step >= int(ema_start_step):
                with ema.average_parameters(vae):
                    save_vae_weights(vae, os.path.join(vae_dir, "vae_best.pt"))
            else:
                save_vae_weights(vae, os.path.join(vae_dir, "vae_best.pt"))
            print(f"  New best eval_total={best_eval:.6f} at epoch {epoch + 1} -> saved VAE/vae_best.pt")
        elif patience is not None:
            bad_epochs += 1
            print(f"  No improvement (best={best_eval:.6f}), bad_epochs={bad_epochs}/{patience}")
            if bad_epochs >= patience:
                _save_training_checkpoint(os.path.join(vae_dir, "last.pt"), epoch_idx=epoch)
                print(f"Early stopping triggered. Best epoch: {best_epoch + 1} (eval_total={best_eval:.6f})")
                break
        elif not improved:
            print(f"  No improvement (best={best_eval:.6f})")

        _save_training_checkpoint(os.path.join(vae_dir, "last.pt"), epoch_idx=epoch)

    writer.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    train_loader = _build_dataloader(
        root_dir=args.train_dir,
        image_size=args.image_size,
        normalization_mode=args.normalization_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
    )
    eval_loader = None
    if args.val_dir:
        eval_loader = _build_dataloader(
            root_dir=args.val_dir,
            image_size=args.image_size,
            normalization_mode=args.normalization_mode,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=False,
        )

    if args.device:
        device = torch.device(args.device)
        smi_out = "manual device override"
    else:
        device, smi_out = get_least_used_cuda_gpu(
            prefer=args.gpu_prefer,
            min_free_mb=args.min_free_mb,
            return_type="torch",
        )
    print(f"Using device: {device}, GPU info:\n{smi_out}")

    # Build VAE from config JSON using src/models/vae.py
    vae_config = _resolve_vae_config(args)
    vae = build_vae_from_config(vae_config, device=device)

    # Save config to model_dir/VAE/
    vae_dir = os.path.join(args.model_dir, "VAE")
    os.makedirs(vae_dir, exist_ok=True)
    save_vae_config(vae_config, os.path.join(vae_dir, "config.json"))

    train_vae(
        vae=vae,
        device=device,
        dataloader=train_loader,
        epochs=args.epochs,
        eval_dataloader=eval_loader,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        scheduler_name=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        recon_loss=args.recon_loss,
        recon_mse_weight=args.recon_mse_weight,
        kl_start_weight=args.kl_start_weight,
        kl_end_weight=args.kl_end_weight,
        kl_warmup_ratio=args.kl_warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
        ema_decay=args.ema_decay,
        ema_start_step=args.ema_start_step,
        resume=args.resume,
        save_every_n_epochs=args.save_every_n_epochs,
        normalization_mode=args.normalization_mode,
    )


if __name__ == "__main__":
    main()
