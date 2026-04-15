"""Smoke tests for the upgraded VAE training loop."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.cli.train_vae import (
    _build_dataloader,
    _compute_kl_loss,
    _compute_raw_reconstruction_metrics,
    _kl_weight_at_step,
    train_vae,
)
from src.core.normalization import RAW_UINT16_PERCENTILE, norm_to_uint16


class TinyVAE(nn.Module):
    """Minimal VAE-like module for fast CPU smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.decoder = nn.Conv2d(4, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        hidden = torch.tanh(self.encoder(x))
        mu = hidden[:, :2]
        sigma = F.softplus(hidden[:, 2:4]) + 1e-3
        recon = torch.tanh(self.decoder(torch.cat([mu, sigma], dim=1)))
        return recon, mu, sigma


def _toy_loader(batch_size: int = 2) -> DataLoader:
    samples = []
    base = torch.linspace(-1.0, 1.0, steps=16 * 16, dtype=torch.float32).reshape(1, 16, 16)
    for idx in range(4):
        samples.append(base.roll(shifts=idx, dims=-1))
    return DataLoader(samples, batch_size=batch_size, shuffle=False)


def _toy_dual_domain_loader(batch_size: int = 2) -> DataLoader:
    samples = []
    base = torch.linspace(-1.0, 1.0, steps=16 * 16, dtype=torch.float32).reshape(1, 16, 16)
    for idx in range(4):
        normalized = base.roll(shifts=idx, dims=-1)
        samples.append(
            {
                "normalized": normalized,
                "raw_resized": norm_to_uint16(normalized),
            }
        )
    return DataLoader(samples, batch_size=batch_size, shuffle=False)


def test_kl_helpers_are_well_behaved() -> None:
    mu = torch.zeros(2, 3, 4, 4)
    sigma = torch.ones(2, 3, 4, 4)
    assert float(_compute_kl_loss(mu, sigma).item()) == 0.0
    assert _kl_weight_at_step(0, total_steps=100, start_weight=0.0, end_weight=1e-5, warmup_ratio=0.1) == 0.0
    assert _kl_weight_at_step(10, total_steps=100, start_weight=0.0, end_weight=1e-5, warmup_ratio=0.1) == 1e-5


def test_train_vae_smoke_saves_resume_and_best_checkpoint() -> None:
    train_loader = _toy_loader()
    eval_loader = _toy_loader()
    vae = TinyVAE()

    with tempfile.TemporaryDirectory() as tmpdir:
        train_vae(
            vae=vae,
            device="cpu",
            dataloader=train_loader,
            epochs=1,
            eval_dataloader=eval_loader,
            log_dir=os.path.join(tmpdir, "tb"),
            model_dir=tmpdir,
            patience=2,
            lr=1e-3,
            scheduler_name="warmup_cosine",
            warmup_ratio=0.1,
            min_lr_ratio=0.1,
            recon_mse_weight=0.25,
            kl_start_weight=0.0,
            kl_end_weight=1e-5,
            kl_warmup_ratio=0.1,
            max_grad_norm=1.0,
            mixed_precision="bf16",
            ema_decay=0.99,
            ema_start_step=0,
            save_every_n_epochs=1,
        )

        vae_dir = os.path.join(tmpdir, "VAE")
        last_path = os.path.join(vae_dir, "last.pt")
        best_path = os.path.join(vae_dir, "vae_best.pt")
        epoch_path = os.path.join(vae_dir, "vae_epoch_1.pt")

        assert os.path.isfile(last_path)
        assert os.path.isfile(best_path)
        assert os.path.isfile(epoch_path)

        ckpt = torch.load(last_path, map_location="cpu")
        assert "optimizer_state" in ckpt
        assert "scheduler_state" in ckpt
        assert "ema_state" in ckpt
        assert ckpt["ema_state"] is not None

        resumed = TinyVAE()
        train_vae(
            vae=resumed,
            device="cpu",
            dataloader=train_loader,
            epochs=1,
            eval_dataloader=eval_loader,
            log_dir=os.path.join(tmpdir, "tb_resume"),
            model_dir=os.path.join(tmpdir, "resume"),
            resume=last_path,
            mixed_precision="no",
            ema_decay=0.99,
            ema_start_step=0,
        )


def test_build_dataloader_returns_normalized_and_raw_resized_batches() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        arr = np.linspace(11000, 14000, num=192 * 256, dtype=np.float32).reshape(1, 192, 256)
        np.save(os.path.join(tmpdir, "sample.npy"), arr)

        loader = _build_dataloader(
            root_dir=tmpdir,
            image_size=256,
            normalization_mode=RAW_UINT16_PERCENTILE,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )
        batch = next(iter(loader))

        assert set(batch.keys()) == {"normalized", "raw_resized"}
        assert batch["normalized"].shape == (1, 1, 256, 256)
        assert batch["raw_resized"].shape == (1, 1, 256, 256)


def test_raw_reconstruction_metrics_use_norm_to_uint16() -> None:
    recon = torch.zeros(2, 1, 4, 4)
    raw_target = norm_to_uint16(recon)

    metrics = _compute_raw_reconstruction_metrics(
        raw_target,
        recon,
        normalization_mode=RAW_UINT16_PERCENTILE,
    )

    assert metrics is not None
    assert torch.allclose(metrics["recon_raw"], raw_target)
    assert float(metrics["l1"].item()) == 0.0
    assert float(metrics["mse"].item()) == 0.0


def test_train_vae_supports_dual_domain_mapping_batches() -> None:
    train_loader = _toy_dual_domain_loader()
    eval_loader = _toy_dual_domain_loader()
    vae = TinyVAE()

    with tempfile.TemporaryDirectory() as tmpdir:
        train_vae(
            vae=vae,
            device="cpu",
            dataloader=train_loader,
            epochs=1,
            eval_dataloader=eval_loader,
            log_dir=os.path.join(tmpdir, "tb"),
            model_dir=tmpdir,
            patience=2,
            lr=1e-3,
            scheduler_name="warmup_cosine",
            warmup_ratio=0.1,
            min_lr_ratio=0.1,
            recon_mse_weight=0.25,
            kl_start_weight=0.0,
            kl_end_weight=1e-5,
            kl_warmup_ratio=0.1,
            max_grad_norm=1.0,
            mixed_precision="no",
            ema_decay=0.99,
            ema_start_step=0,
            save_every_n_epochs=1,
            normalization_mode=RAW_UINT16_PERCENTILE,
        )

        vae_dir = os.path.join(tmpdir, "VAE")
        assert os.path.isfile(os.path.join(vae_dir, "last.pt"))
        assert os.path.isfile(os.path.join(vae_dir, "vae_best.pt"))
