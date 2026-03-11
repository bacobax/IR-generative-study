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
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.core.normalization import (
    raw_to_norm as to_sd_tensor_and_x,
    norm_to_display as from_norm_to_display,
)
from src.core.data.datasets import NPYImageDataset
from src.models.vae import (
    load_vae_config,
    build_vae_from_config,
    save_vae_config,
    save_vae_weights,
)
from src.core.gpu_utils import get_least_used_cuda_gpu
from src.core.configs.config_loader import apply_yaml_defaults


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
    parser.add_argument("--log-dir", type=str, default="./artifacts/runs/main/autoencoder_kl", help="TensorBoard log dir.")
    parser.add_argument("--patience", type=int, default=4, help="Early-stopping patience.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Early-stopping min delta.")
    parser.add_argument("--gpu-prefer", type=str, default="memory", help="GPU selection preference.")
    parser.add_argument("--min-free-mb", type=int, default=4096, help="Minimum free GPU memory.")
    # Two-pass parse: first grab --config, apply YAML defaults, then re-parse
    preliminary, _ = parser.parse_known_args()
    apply_yaml_defaults(parser, preliminary.config)
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resize_and_normalize(x: torch.Tensor, size: int) -> torch.Tensor:
    x = F.interpolate(
        x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False,
    ).squeeze(0)
    x = to_sd_tensor_and_x(x)
    return x


def _build_dataloader(
    root_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = NPYImageDataset(
        root_dir=root_dir,
        transform=lambda x: _resize_and_normalize(x, image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _stretch_for_vis(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image min/max stretch to [0, 1] for TensorBoard."""
    x = from_norm_to_display(x).clamp(0, 1)
    flat = x.view(x.size(0), -1)
    mn = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    mx = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    return ((x - mn) / (mx - mn + eps)).clamp(0, 1)


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
):
    """Standalone VAE training loop (no old pipeline dependency)."""
    if eval_dataloader is None:
        raise ValueError("eval_dataloader must be provided.")

    vae_dir = os.path.join(model_dir, "VAE")
    os.makedirs(vae_dir, exist_ok=True)

    optimizer = Adam(vae.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir)
    kl_weight = 1e-4
    max_grad_norm = 1.0
    logvar_clamp = (-30.0, 20.0)

    best_eval = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(epochs):
        vae.train()
        train_total = train_mse = train_mae = train_kl = 0.0
        n_train = 0

        for x in tqdm(dataloader, desc=f"VAE Train {epoch + 1}/{epochs}"):
            x = x.to(device)

            recon, mu, logvar = vae(x)
            recon_mse = F.mse_loss(recon, x)
            recon_mae = F.l1_loss(recon, x)

            logvar = logvar.clamp(*logvar_clamp)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_mse + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_grad_norm)
            optimizer.step()

            bs = x.size(0)
            train_total += loss.item() * bs
            train_mse += recon_mse.item() * bs
            train_mae += recon_mae.item() * bs
            train_kl += kl_loss.item() * bs
            n_train += bs

        train_total /= max(1, n_train)
        train_mse /= max(1, n_train)
        train_mae /= max(1, n_train)
        train_kl /= max(1, n_train)

        # ── Evaluation ──
        vae.eval()
        eval_total = eval_mse = eval_mae = eval_kl = 0.0
        n_eval = 0

        with torch.no_grad():
            for x in tqdm(eval_dataloader, desc=f"VAE Eval  {epoch + 1}/{epochs}"):
                x = x.to(device)

                recon, mu, logvar = vae(x)
                recon_mse = F.mse_loss(recon, x)
                recon_mae = F.l1_loss(recon, x)

                logvar = logvar.clamp(*logvar_clamp)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_mse + kl_weight * kl_loss

                bs = x.size(0)
                eval_total += loss.item() * bs
                eval_mse += recon_mse.item() * bs
                eval_mae += recon_mae.item() * bs
                eval_kl += kl_loss.item() * bs
                n_eval += bs

        eval_total /= max(1, n_eval)
        eval_mse /= max(1, n_eval)
        eval_mae /= max(1, n_eval)
        eval_kl /= max(1, n_eval)

        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"train: total={train_total:.6f} mse={train_mse:.6f} mae={train_mae:.6f} kl={train_kl:.6f} | "
            f"eval:  total={eval_total:.6f} mse={eval_mse:.6f} mae={eval_mae:.6f} kl={eval_kl:.6f}"
        )

        writer.add_scalar("train/total", train_total, epoch)
        writer.add_scalar("train/recon_mse", train_mse, epoch)
        writer.add_scalar("train/recon_mae", train_mae, epoch)
        writer.add_scalar("train/kl", train_kl, epoch)
        writer.add_scalar("eval/total", eval_total, epoch)
        writer.add_scalar("eval/recon_mse", eval_mse, epoch)
        writer.add_scalar("eval/recon_mae", eval_mae, epoch)
        writer.add_scalar("eval/kl", eval_kl, epoch)

        # Epoch checkpoint
        save_vae_weights(vae, os.path.join(vae_dir, f"vae_epoch_{epoch + 1}.pt"))

        # Image logging
        with torch.no_grad():
            x_tr = next(iter(dataloader)).to(device)
            recon_tr, _, _ = vae(x_tr)
            x_tr_vis = _stretch_for_vis(x_tr[:4])
            recon_tr_vis = _stretch_for_vis(recon_tr[:4])
            writer.add_images("train/input", x_tr_vis, epoch)
            writer.add_images("train/reconstruction", recon_tr_vis, epoch)

            x_ev = next(iter(eval_dataloader)).to(device)
            recon_ev, _, _ = vae(x_ev)
            x_ev_vis = _stretch_for_vis(x_ev[:4])
            recon_ev_vis = _stretch_for_vis(recon_ev[:4])
            writer.add_images("eval/input", x_ev_vis, epoch)
            writer.add_images("eval/reconstruction", recon_ev_vis, epoch)

        # Early stopping
        if patience is not None:
            improved = (best_eval - eval_total) > min_delta
            if improved:
                best_eval = eval_total
                best_epoch = epoch
                bad_epochs = 0
                save_vae_weights(vae, os.path.join(vae_dir, "vae_best.pt"))
                print(f"  New best eval_total={best_eval:.6f} at epoch {epoch + 1} -> saved VAE/vae_best.pt")
            else:
                bad_epochs += 1
                print(f"  No improvement (best={best_eval:.6f}), bad_epochs={bad_epochs}/{patience}")
                if bad_epochs >= patience:
                    print(f"Early stopping triggered. Best epoch: {best_epoch + 1} (eval_total={best_eval:.6f})")
                    break

    writer.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    train_loader = _build_dataloader(
        root_dir=args.train_dir,
        image_size=args.image_size,
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
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=False,
        )

    device, smi_out = get_least_used_cuda_gpu(
        prefer=args.gpu_prefer,
        min_free_mb=args.min_free_mb,
        return_type="torch",
    )
    print(f"Using device: {device}, GPU info:\n{smi_out}")

    # Build VAE from config JSON using src/models/vae.py
    vae_config = load_vae_config(path=args.vae_json)
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
        log_dir=f"{args.model_dir}/runs/autoencoder_kl",
        model_dir=args.model_dir,
        patience=args.patience,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()
