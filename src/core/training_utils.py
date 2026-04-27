"""Shared training helpers for EMA, precision, and LR scheduling."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import torch


@dataclass(frozen=True)
class PrecisionSettings:
    """Resolved mixed-precision settings for a training run."""

    requested: str
    mode: str
    device_type: str
    dtype: Optional[torch.dtype]
    enabled: bool
    use_grad_scaler: bool


class NullSummaryWriter:
    """No-op TensorBoard writer used when tensorboard is not installed."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def __getattr__(self, name: str):
        if name.startswith("add_") or name in {"flush", "close"}:
            return self._noop
        raise AttributeError(name)

    def _noop(self, *args, **kwargs) -> None:
        del args, kwargs

    def close(self) -> None:
        pass


def build_summary_writer(log_dir: str):
    """Create a TensorBoard writer, falling back to no-op logging if unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        if exc.name != "tensorboard":
            raise
        print(
            "[TensorBoard] tensorboard is not installed; continuing with "
            "scalar/image logging disabled."
        )
        return NullSummaryWriter(log_dir)
    return SummaryWriter(log_dir)


def resolve_precision_settings(
    device: Union[str, torch.device],
    mixed_precision: Optional[str],
) -> PrecisionSettings:
    """Resolve a requested precision mode into concrete runtime settings."""
    requested = str(mixed_precision or "no").lower()
    device_type = torch.device(device).type
    mode = requested

    if requested == "auto":
        if device_type != "cuda":
            mode = "no"
        elif bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            mode = "bf16"
        else:
            mode = "fp16"

    if mode == "bf16":
        enabled = device_type == "cuda" and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        dtype = torch.bfloat16 if enabled else None
    elif mode == "fp16":
        enabled = device_type == "cuda"
        dtype = torch.float16 if enabled else None
    elif mode in {"no", "none", "fp32"}:
        enabled = False
        dtype = None
        mode = "no"
    else:
        raise ValueError(
            f"Unsupported mixed_precision={mixed_precision!r}. Expected one of "
            "'auto', 'bf16', 'fp16', or 'no'."
        )

    return PrecisionSettings(
        requested=requested,
        mode=mode,
        device_type=device_type,
        dtype=dtype,
        enabled=enabled,
        use_grad_scaler=(enabled and mode == "fp16" and device_type == "cuda"),
    )


def autocast_context(settings: PrecisionSettings):
    """Return an autocast context manager or a no-op context."""
    if not settings.enabled or settings.dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(
        device_type=settings.device_type,
        dtype=settings.dtype,
        enabled=True,
    )


def build_grad_scaler(settings: PrecisionSettings) -> Optional[torch.cuda.amp.GradScaler]:
    """Create a CUDA GradScaler when fp16 autocast is active."""
    if not settings.use_grad_scaler:
        return None
    return torch.cuda.amp.GradScaler(enabled=True)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: Union[str, torch.device]) -> None:
    """Move optimizer tensor state to the requested device after resume."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def resolve_warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    """Convert a warmup ratio to an integer warmup step count."""
    total_steps = max(1, int(total_steps))
    return max(0, min(total_steps, int(round(float(warmup_ratio) * total_steps))))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    total_steps: int,
    warmup_ratio: float = 0.0,
    min_lr_ratio: float = 0.0,
):
    """Create a lightweight warmup scheduler for optimizer updates."""
    name = str(scheduler_name or "none").lower()
    total_steps = max(1, int(total_steps))
    warmup_steps = resolve_warmup_steps(total_steps, warmup_ratio)
    min_lr_ratio = float(min_lr_ratio)

    if name in {"none", "disabled"}:
        return None

    def _warmup_scale(step_idx: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step_idx + 1) / float(max(1, warmup_steps)))

    def _cosine_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return _warmup_scale(step_idx)
        progress = float(step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    def _constant_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return _warmup_scale(step_idx)
        return 1.0

    if name == "warmup_cosine":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_lambda)
    if name == "constant_with_warmup":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_constant_lambda)

    raise ValueError(
        f"Unsupported scheduler={scheduler_name!r}. Expected 'none', "
        "'warmup_cosine', or 'constant_with_warmup'."
    )


def compute_snr(noise_scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    """Compute per-timestep signal-to-noise ratio without importing diffusers helpers."""
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    timesteps = timesteps.long()
    alpha = sqrt_alphas_cumprod[timesteps].float()
    sigma = sqrt_one_minus_alphas_cumprod[timesteps].float()
    while alpha.ndim < timesteps.ndim:
        alpha = alpha[..., None]
        sigma = sigma[..., None]
    return (alpha / sigma.clamp(min=1e-20)) ** 2


class EMAState:
    """Simple exponential moving average tracker for model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow_params = [
            param.detach().clone()
            for param in model.parameters()
            if param.requires_grad
        ]

    def update(self, model: torch.nn.Module) -> None:
        """Update the running EMA weights from the current model."""
        idx = 0
        with torch.no_grad():
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                self.shadow_params[idx].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
                idx += 1

    def copy_to(self, model: torch.nn.Module) -> None:
        """Copy EMA weights into the given model in-place."""
        idx = 0
        with torch.no_grad():
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                param.copy_(self.shadow_params[idx].to(device=param.device, dtype=param.dtype))
                idx += 1

    def state_dict(self) -> dict:
        """Serialize EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow_params": [param.detach().cpu() for param in self.shadow_params],
        }

    def load_state_dict(self, state_dict: dict, device: Union[str, torch.device]) -> None:
        """Restore EMA state from a checkpoint payload."""
        self.decay = float(state_dict.get("decay", self.decay))
        self.shadow_params = [
            tensor.to(device)
            for tensor in state_dict.get("shadow_params", [])
        ]

    @contextlib.contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """Temporarily swap model parameters with the EMA weights."""
        backup = [
            param.detach().clone()
            for param in model.parameters()
            if param.requires_grad
        ]
        self.copy_to(model)
        try:
            yield
        finally:
            idx = 0
            with torch.no_grad():
                for param in model.parameters():
                    if not param.requires_grad:
                        continue
                    param.copy_(backup[idx].to(device=param.device, dtype=param.dtype))
                    idx += 1


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Compute the global L2 norm of currently populated gradients."""
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return float(total ** 0.5)


def cast_training_params(
    model: Union[torch.nn.Module, Iterable[torch.nn.Module]],
    *,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Cast trainable parameters to ``dtype`` without importing diffusers pipelines."""
    if isinstance(model, torch.nn.Module):
        modules = [model]
    else:
        modules = list(model)

    for module in modules:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(
                "cast_training_params expects a torch.nn.Module or an iterable "
                f"of modules, got {type(module).__name__}."
            )
        for param in module.parameters():
            if param.requires_grad:
                param.data = param.data.to(dtype=dtype)
