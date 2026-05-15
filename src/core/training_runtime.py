"""Conservative runtime setup helpers shared by trainer loops."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.core.training_utils import (
    EMAState,
    build_grad_scaler,
    build_scheduler,
    resolve_precision_settings,
)


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
    *,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> torch.optim.Optimizer:
    """Create the optimizer used by current training loops."""
    if str(optimizer_name).lower() != "adamw":
        raise ValueError(f"Unsupported optimizer_name={optimizer_name!r}. Only 'adamw' is implemented.")
    return AdamW(
        parameters,
        lr=float(lr),
        betas=(float(beta1), float(beta2)),
        weight_decay=float(weight_decay),
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    total_steps: int,
    warmup_ratio: float = 0.0,
    min_lr_ratio: float = 0.0,
):
    """Build the existing lightweight LR scheduler."""
    return build_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )


def setup_precision(
    device: str | torch.device,
    mixed_precision: Optional[str],
):
    """Resolve precision settings and create a GradScaler when needed."""
    precision = resolve_precision_settings(device, mixed_precision)
    return precision, build_grad_scaler(precision)


def build_ema(
    model: torch.nn.Module,
    *,
    enabled: bool = True,
    decay: float = 0.999,
) -> EMAState | None:
    """Create EMA state when enabled, preserving current trainer semantics."""
    return EMAState(model, decay=decay) if bool(enabled) and float(decay) > 0.0 else None


def set_epoch_for_dataloader(dl: Optional[DataLoader], epoch_idx: int) -> None:
    """Propagate epoch to nested datasets and transforms that expose set_epoch."""
    if dl is None:
        return
    current = getattr(dl, "dataset", None)
    while current is not None:
        if hasattr(current, "set_epoch"):
            current.set_epoch(epoch_idx)
        transform = getattr(current, "transform", None)
        if transform is not None and hasattr(transform, "set_epoch"):
            transform.set_epoch(epoch_idx)
        current = getattr(current, "dataset", None)
