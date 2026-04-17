"""Training logic for flow-matching models.

This module is the sole source of truth for flow-matching training.
It replaces the training methods formerly embedded in
``fm_src.pipelines.flow_matching_pipeline``.

Two modes are supported:
* **pixel-space** — UNet operates directly on normalised images.
* **latent-space** (stable) — images are first encoded by a VAE;
  the UNet operates in the VAE latent space.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler
from src.core.ot import match_target_batch
from src.core.training_utils import (
    EMAState,
    autocast_context,
    build_grad_scaler,
    build_scheduler,
    move_optimizer_state_to_device,
    resolve_precision_settings,
)
from src.models.fm_unet import save_unet_config, load_unet_config, build_fm_unet_from_config


# ---------------------------------------------------------------------------
# Default display helper (same as sampler)
# ---------------------------------------------------------------------------
def _default_from_norm_to_display(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


def _infer_vae_downsample_factor(vae_config: Dict[str, Any]) -> int:
    """Infer the VAE spatial downsample factor from its channel schedule."""
    num_channels = vae_config.get("num_channels")
    if isinstance(num_channels, (list, tuple)) and num_channels:
        return 2 ** max(0, len(num_channels) - 1)

    block_out_channels = vae_config.get("block_out_channels")
    if isinstance(block_out_channels, (list, tuple)) and block_out_channels:
        return 2 ** max(0, len(block_out_channels) - 1)

    down_block_types = vae_config.get("down_block_types")
    if isinstance(down_block_types, (list, tuple)) and down_block_types:
        return 2 ** max(0, len(down_block_types) - 1)

    raise ValueError(
        "VAE config must define a non-empty num_channels, block_out_channels, "
        "or down_block_types sequence to infer sample_size."
    )


def _resolve_unet_sample_size(config, vae_config: Dict[str, Any]) -> int | None:
    """Map the configured training image size to the latent UNet sample size."""
    data_cfg = getattr(config, "data", None)
    image_size = getattr(data_cfg, "image_size", None)
    if image_size is None:
        return None

    downsample_factor = _infer_vae_downsample_factor(vae_config)
    image_size = int(image_size)
    if image_size % downsample_factor != 0:
        raise ValueError(
            f"image_size={image_size} is not divisible by VAE downsample factor "
            f"{downsample_factor}"
        )
    return image_size // downsample_factor


# ═══════════════════════════════════════════════════════════════════════════
# FlowMatchingTrainer
# ═══════════════════════════════════════════════════════════════════════════

class FlowMatchingTrainer:
    """Self-contained trainer for flow-matching UNets.

    Parameters
    ----------
    unet : UNet2DModel
        The UNet to train.
    device : str or torch.device
    t_scale : float
        Time-scaling factor for UNet timestep input.
    train_target : ``"v"`` | ``"x0"``
    model_dir : str
        Root output directory. Weights are saved under ``model_dir/UNET/``.
    from_norm_to_display : callable, optional
        [-1,1] → [0,1] for TensorBoard visualisation.
    unet_config : dict, optional
        If provided, saved as ``model_dir/UNET/config.json``.
    vae : AutoencoderKL, optional
        If provided, training happens in latent space.
        The VAE is frozen automatically.
    vae_config : dict, optional
        Saved as ``model_dir/VAE/config.json`` when given.
    conditioner : BaseConditioner, optional
        Conditioning module.  When ``None``, no extra kwargs are fed
        to the UNet (unconditional training).
    """

    def __init__(
        self,
        unet: UNet2DModel,
        *,
        device: Optional[Union[str, torch.device]] = None,
        t_scale: float = 1.0,
        train_target: str = "v",
        model_dir: str = "./artifacts/checkpoints/legacy/pipeline_model",
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        unet_config: Optional[Dict[str, Any]] = None,
        vae=None,
        vae_config: Optional[Dict[str, Any]] = None,
        conditioner=None,
        path_mode: str = "independent",
        path_solver: str = "hungarian",
        layout_cost_resolution: int = 16,
        condition_weight: float = 1.0,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.unet = unet
        self.t_scale = float(t_scale)
        assert train_target in ("v", "x0"), f"train_target must be 'v' or 'x0', got '{train_target}'"
        self.train_target = train_target
        self.model_dir = model_dir
        self.from_norm_to_display = from_norm_to_display or _default_from_norm_to_display
        self.unet_config = unet_config
        self.vae = vae
        self.vae_config = vae_config
        self.conditioner = conditioner
        self.path_mode = str(path_mode)
        self.path_solver = str(path_solver)
        self.layout_cost_resolution = int(layout_cost_resolution)
        self.condition_weight = float(condition_weight)

        # Freeze VAE if present
        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

    def _metric_prefix(self) -> str:
        return "fm"

    def _checkpoint_stem(self) -> str:
        return "unet_fm"

    def _progress_label(self) -> str:
        return "FM"

    def _sample_tensorboard_tag(self) -> str:
        return f"{self._metric_prefix()}/generated"

    def _save_additional_configs(self) -> None:
        """Persist algorithm-specific config artifacts."""

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "t_scale": self.t_scale,
            "train_target": self.train_target,
        }

    def _best_weights_path(self) -> str:
        return os.path.join(self._unet_dir(), f"{self._checkpoint_stem()}_best.pt")

    def _epoch_weights_path(self, epoch_num: int) -> str:
        return os.path.join(self._unet_dir(), f"{self._checkpoint_stem()}_epoch_{epoch_num}.pt")

    def _epoch_checkpoint_path(self, epoch_num: int) -> str:
        return os.path.join(self._unet_dir(), f"{self._checkpoint_stem()}_epoch_{epoch_num}_ckpt.pt")

    # ------------------------------------------------------------------
    # Config-driven constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config,
        *,
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> "FlowMatchingTrainer":
        """Build a trainer from an :class:`FMTrainConfig`.

        Parameters
        ----------
        config : FMTrainConfig
            Structured configuration object.
        from_norm_to_display : callable, optional
            Override display normalisation if needed.
        """
        from src.models.vae import build_vae_from_config, resolve_vae_config_from_model_config

        device = config.resolved_device() if hasattr(config, "resolved_device") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        unet_cfg = load_unet_config(config.model.unet_config)
        vae_cfg = None
        vae = None

        if (
            getattr(config.model, "vae_config", None) is not None
            or getattr(config.model, "vae_pretrained_model_name_or_path", None) is not None
        ):
            vae_cfg = resolve_vae_config_from_model_config(config.model)
            resolved_sample_size = _resolve_unet_sample_size(config, vae_cfg)
        else:
            resolved_sample_size = int(getattr(config.data, "image_size", unet_cfg.get("sample_size")))

        if resolved_sample_size is not None:
            unet_cfg = dict(unet_cfg)
            unet_cfg["sample_size"] = resolved_sample_size

        unet = build_fm_unet_from_config(unet_cfg, device=device)
        if vae_cfg is not None:
            vae = build_vae_from_config(vae_cfg, device=device)

        return cls(
            unet,
            device=device,
            t_scale=config.training.t_scale,
            train_target=config.training.train_target,
            model_dir=config.output.model_dir,
            from_norm_to_display=from_norm_to_display,
            unet_config=unet_cfg,
            vae=vae,
            vae_config=vae_cfg,
            path_mode=getattr(config.path, "mode", "independent"),
            path_solver=getattr(config.path, "solver", "hungarian"),
            layout_cost_resolution=getattr(config.path, "layout_cost_resolution", 16),
            condition_weight=getattr(config.path, "condition_weight", 1.0),
        )

    def train_from_config(
        self,
        config,
        dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> None:
        """Launch training driven by an :class:`FMTrainConfig`.

        Extracts all keyword arguments from the config sub-objects and
        delegates to :meth:`train`.
        """
        self.train(
            dataloader=dataloader,
            epochs=config.training.epochs,
            eval_dataloader=eval_dataloader,
            pretrained_vae_path=config.model.vae_weights,
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
            eval_every=getattr(config.training, "eval_every", 1),
            resume_from_checkpoint=config.output.resume,
            lr=config.resolved_lr() if hasattr(config, "resolved_lr") else getattr(config.training, "lr", 1e-4),
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
        )

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def _unet_dir(self) -> str:
        return os.path.join(self.model_dir, "UNET")

    def _vae_dir(self) -> str:
        return os.path.join(self.model_dir, "VAE")

    def _ensure_dirs(self) -> None:
        os.makedirs(self._unet_dir(), exist_ok=True)
        if self.vae is not None:
            os.makedirs(self._vae_dir(), exist_ok=True)

    def _save_configs(self) -> None:
        if self.unet_config is not None:
            save_unet_config(self.unet_config, os.path.join(self._unet_dir(), "config.json"))
        if self.vae_config is not None:
            os.makedirs(self._vae_dir(), exist_ok=True)
            import json
            path = os.path.join(self._vae_dir(), "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.vae_config, f, indent=2, sort_keys=True)
        self._save_additional_configs()

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------
    def load_unet_weights(self, path: str, *, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "unet_state" in state:
            state = state["unet_state"]
        missing, unexpected = self.unet.load_state_dict(state, strict=strict)
        if (not strict) or missing or unexpected:
            print(f"[load_unet_weights] strict={strict}")
            if missing:
                print("  Missing keys:", missing)
            if unexpected:
                print("  Unexpected keys:", unexpected)

    def save_unet_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.unet.state_dict(), path)

    def load_vae_weights(self, path: str, *, strict: bool = True) -> None:
        assert self.vae is not None, "VAE not set."
        from src.models.vae import load_vae_weights as _load_vae_weights

        _load_vae_weights(
            self.vae,
            path,
            strict=strict,
            map_location=self.device,
        )

    # ------------------------------------------------------------------
    # Encode helper (pixel-space passthrough or VAE)
    # ------------------------------------------------------------------
    def encode_fm_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return x
        with torch.no_grad():
            z_mu, z_sigma = self.vae.encode(x)
            return self.vae.sampling(z_mu, z_sigma)

    def _prepare_batch(self, batch) -> tuple[torch.Tensor, Dict[str, Any]]:
        if torch.is_tensor(batch):
            return batch.to(self.device), {}
        if isinstance(batch, dict) and "pixel_values" in batch:
            return batch["pixel_values"].to(self.device), {}
        raise TypeError(
            "Expected a tensor batch or a dict containing 'pixel_values' for "
            f"{self.__class__.__name__}, got {type(batch)!r}."
        )

    def _compute_batch_loss(
        self,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return self.flow_matching_step(x_fm, cond_kwargs)

    # ------------------------------------------------------------------
    # Flow-matching loss
    # ------------------------------------------------------------------
    def _match_flow_targets(
        self,
        z0: torch.Tensor,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply the configured path coupling and return the matched targets."""
        del cond_kwargs
        if self.path_mode == "independent":
            return x_fm
        if self.path_mode in {"minibatch_ot", "conditional_ot"}:
            # The base trainer has no condition-dependent cost term. In that case
            # conditional OT reduces to plain minibatch OT instead of erroring.
            matched, _, _ = match_target_batch(
                z0,
                x_fm,
                solver=self.path_solver,
            )
            return matched
        raise ValueError(
            f"Unsupported path_mode={self.path_mode!r}. Expected 'independent', "
            "'minibatch_ot', or 'conditional_ot'."
        )

    def flow_matching_step(self, x_fm: torch.Tensor, cond_kwargs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Compute a single flow-matching loss on encoded input *x_fm*."""
        if cond_kwargs is None:
            cond_kwargs = {}
        B = x_fm.shape[0]
        z0 = torch.randn_like(x_fm)
        t = torch.rand(B, device=x_fm.device)
        t_expanded = t[:, None, None, None]
        x_target = self._match_flow_targets(z0, x_fm, cond_kwargs)

        zt = (1 - t_expanded) * z0 + t_expanded * x_target
        v_target = x_target - z0

        unet_out = self.unet(zt, t * self.t_scale, **cond_kwargs).sample

        if self.train_target == "x0":
            x0_pred = unet_out
            v_pred = (x0_pred - zt) / (1 - t_expanded).clamp(min=1e-5)
        else:
            v_pred = unet_out

        return F.mse_loss(v_pred, v_target)

    # ------------------------------------------------------------------
    # Build a sampler for sample-at-epoch
    # ------------------------------------------------------------------
    def _make_sampler(self) -> FlowMatchingSampler:
        if self.vae is not None:
            return FlowMatchingSampler.from_stable(
                self.unet,
                self.vae,
                device=self.device,
                t_scale=self.t_scale,
                train_target=self.train_target,
                from_norm_to_display=self.from_norm_to_display,
            )
        return FlowMatchingSampler(
            self.unet,
            device=self.device,
            t_scale=self.t_scale,
            train_target=self.train_target,
            from_norm_to_display=self.from_norm_to_display,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        *,
        pretrained_vae_path: Optional[str] = None,
        pretrained_unet_path: Optional[str] = None,
        strict_load: bool = True,
        log_dir: str = "./artifacts/runs/main/flow_matching",
        sample_every: int = 1,
        sample_steps: int = 50,
        sample_batch_size: int = 4,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        save_every_n_epochs: int = 1,
        eval_every: int = 1,
        resume_from_checkpoint: Optional[str] = None,
        lr: float = 1e-4,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        scheduler_name: str = "warmup_cosine",
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.1,
        ema_enabled: bool = True,
        ema_decay: float = 0.999,
        ema_start_step: int = 100,
        mixed_precision: str = "auto",
        max_grad_norm: float = 1.0,
    ) -> None:
        eval_every = int(eval_every)
        if patience is not None and eval_dataloader is None:
            raise ValueError("eval_dataloader must be provided when using patience early stopping.")
        if patience is not None and eval_every <= 0:
            raise ValueError("eval_every must be > 0 when using patience early stopping.")
        if str(optimizer_name).lower() != "adamw":
            raise ValueError(f"Unsupported optimizer_name={optimizer_name!r}. Only 'adamw' is implemented.")

        self._ensure_dirs()
        self._save_configs()

        # Pre-load weights
        if pretrained_vae_path is not None and self.vae is not None:
            self.load_vae_weights(pretrained_vae_path, strict=strict_load)
            # Re-freeze after loading
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

        if pretrained_unet_path is not None:
            self.load_unet_weights(pretrained_unet_path, strict=strict_load)

        total_steps = max(1, epochs * len(dataloader))
        precision = resolve_precision_settings(self.device, mixed_precision)
        scaler = build_grad_scaler(precision)
        optimizer = AdamW(
            self.unet.parameters(),
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
        ema = EMAState(self.unet, decay=ema_decay) if ema_enabled and ema_decay > 0.0 else None

        # Resume state
        global_step = 0
        best_eval = float("inf")
        best_epoch = -1
        bad_epochs = 0
        start_epoch = 0

        if resume_from_checkpoint is not None:
            print(f"[{self._progress_label()} Resume] Loading checkpoint from {resume_from_checkpoint}")
            ckpt = torch.load(resume_from_checkpoint, map_location=self.device)
            self.unet.load_state_dict(ckpt["unet_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            move_optimizer_state_to_device(optimizer, self.device)
            if scheduler is not None and ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if scaler is not None and ckpt.get("scaler_state") is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            if ema is not None:
                if ckpt.get("ema_state") is not None:
                    ema.load_state_dict(ckpt["ema_state"], device=self.device)
                else:
                    ema = EMAState(self.unet, decay=ema_decay)
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt["global_step"]
            best_eval = ckpt.get("best_eval", float("inf"))
            best_epoch = ckpt.get("best_epoch", -1)
            bad_epochs = ckpt.get("bad_epochs", 0)
            if "rng_state" in ckpt:
                rng_state = ckpt["rng_state"]
                if not torch.is_tensor(rng_state) or rng_state.dtype != torch.uint8:
                    rng_state = torch.tensor(rng_state, dtype=torch.uint8)
                if rng_state.device.type != "cpu":
                    rng_state = rng_state.cpu()
                torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt:
                cuda_states = []
                for s in ckpt["cuda_rng_state_all"]:
                    if not torch.is_tensor(s) or s.dtype != torch.uint8:
                        s = torch.tensor(s, dtype=torch.uint8)
                    if s.device.type != "cpu":
                        s = s.cpu()
                    cuda_states.append(s)
                torch.cuda.set_rng_state_all(cuda_states)
            print(
                f"[{self._progress_label()} Resume] Resuming from epoch {start_epoch}, "
                f"global_step={global_step}, best_eval={best_eval:.6f}"
            )

        writer = SummaryWriter(log_dir)

        def _save_checkpoint(path: str, epoch_idx: int) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            ckpt = {
                "epoch": epoch_idx,
                "global_step": global_step,
                "unet_state": self.unet.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                "scaler_state": None if scaler is None else scaler.state_dict(),
                "ema_state": None if ema is None else ema.state_dict(),
                "best_eval": best_eval,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "rng_state": torch.random.get_rng_state(),
            }
            ckpt.update(self._checkpoint_metadata())
            if torch.cuda.is_available():
                ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            torch.save(ckpt, path)

        def _set_epoch_for_dataloader(dl: Optional[DataLoader], epoch_idx: int) -> None:
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

        sampler_obj = self._make_sampler() if sample_every > 0 else None

        if precision.requested != precision.mode:
            print(
                f"[FM Precision] requested={precision.requested!r} -> using {precision.mode!r} "
                f"on device={precision.device_type}"
            )

        for epoch in range(start_epoch, epochs):
            _set_epoch_for_dataloader(dataloader, epoch)
            _set_epoch_for_dataloader(eval_dataloader, epoch)
            self.unet.train()
            total_loss = 0.0

            for batch in tqdm(dataloader, desc=f"{self._progress_label()} Epoch {epoch+1}/{epochs}"):
                x, cond_kw = self._prepare_batch(batch)
                with torch.no_grad():
                    x_fm = self.encode_fm_input(x)
                if self.conditioner is not None:
                    cond_kw.update(self.conditioner.prepare_for_training(x, self.device))
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(precision):
                    loss = self._compute_batch_loss(x_fm, cond_kw)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                if max_grad_norm and max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                if ema is not None and global_step >= int(ema_start_step):
                    ema.update(self.unet)

                total_loss += loss.item()
                writer.add_scalar(f"{self._metric_prefix()}/loss_step", loss.item(), global_step)
                writer.add_scalar(
                    f"{self._metric_prefix()}/lr",
                    float(optimizer.param_groups[0]["lr"]),
                    global_step,
                )
                global_step += 1

            avg_loss = total_loss / max(1, len(dataloader))
            print(f"[{self._progress_label()} Epoch {epoch+1}] loss: {avg_loss:.6f}")
            writer.add_scalar(f"{self._metric_prefix()}/loss_epoch", avg_loss, epoch)

            if (save_every_n_epochs is not None) and ((epoch + 1) % save_every_n_epochs == 0):
                self.save_unet_weights(self._epoch_weights_path(epoch + 1))
                _save_checkpoint(self._epoch_checkpoint_path(epoch + 1), epoch_idx=epoch)

            # Eval + early stopping + best save
            should_run_eval = (
                eval_dataloader is not None
                and eval_every > 0
                and (epoch + 1) % eval_every == 0
            )
            if should_run_eval:
                self.unet.eval()
                eval_loss = 0.0
                n_eval = 0

                ema_context = ema.average_parameters(self.unet) if ema is not None and global_step >= int(ema_start_step) else torch.no_grad()
                with ema_context:
                    with torch.no_grad():
                        for batch in tqdm(eval_dataloader, desc=f"{self._progress_label()} Eval  {epoch+1}/{epochs}"):
                            x, cond_kw = self._prepare_batch(batch)
                            x_fm = self.encode_fm_input(x)
                            if self.conditioner is not None:
                                cond_kw.update(self.conditioner.prepare_for_training(x, self.device))
                            with autocast_context(precision):
                                loss = self._compute_batch_loss(x_fm, cond_kw)

                            bs = x.size(0)
                            eval_loss += loss.item() * bs
                            n_eval += bs

                avg_eval_loss = eval_loss / max(1, n_eval)
                print(f"  [Eval loss: {avg_eval_loss:.6f}]")
                writer.add_scalar(f"{self._metric_prefix()}/eval_loss_epoch", avg_eval_loss, epoch)

                improved = (best_eval - avg_eval_loss) > min_delta
                if improved:
                    best_eval = avg_eval_loss
                    best_epoch = epoch
                    bad_epochs = 0
                    if ema is not None and global_step >= int(ema_start_step):
                        with ema.average_parameters(self.unet):
                            self.save_unet_weights(self._best_weights_path())
                    else:
                        self.save_unet_weights(self._best_weights_path())
                    print(
                        f"  ✅ New best eval_loss={best_eval:.6f} at epoch {epoch+1} "
                        f"-> saved UNET/{os.path.basename(self._best_weights_path())}"
                    )
                elif patience is not None:
                    bad_epochs += 1
                    print(f"  ⏳ No improvement (best={best_eval:.6f}), bad_epochs={bad_epochs}/{patience}")
                    if bad_epochs >= patience:
                        print(f"🛑 Early stopping triggered. Best epoch: {best_epoch+1} (eval_loss={best_eval:.6f})")
                        break

            # Sampling
            if sampler_obj is not None and (epoch + 1) % sample_every == 0:
                ema_context = ema.average_parameters(self.unet) if ema is not None and global_step >= int(ema_start_step) else torch.no_grad()
                with ema_context:
                    sampler_obj.log_samples_to_tensorboard(
                        writer=writer,
                        epoch=epoch,
                        steps=sample_steps,
                        batch_size=sample_batch_size,
                        tag=self._sample_tensorboard_tag(),
                        sample_shape=sample_shape,
                    )

        writer.close()


# ── registry ──────────────────────────────────────────────────────────────────
from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.trainer.register("default_fm", default=True)(FlowMatchingTrainer)
