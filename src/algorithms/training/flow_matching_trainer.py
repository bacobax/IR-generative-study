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
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler
from src.models.fm_unet import save_unet_config, load_unet_config, build_fm_unet_from_config


# ---------------------------------------------------------------------------
# Default display helper (same as sampler)
# ---------------------------------------------------------------------------
def _default_from_norm_to_display(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


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

        # Freeze VAE if present
        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

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
        from src.models.vae import load_vae_config, build_vae_from_config

        device = config.resolved_device() if hasattr(config, "resolved_device") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        unet_cfg = load_unet_config(config.model.unet_config)
        unet = build_fm_unet_from_config(unet_cfg, device=device)

        vae_cfg = load_vae_config(config.model.vae_config)
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
            resume_from_checkpoint=config.output.resume,
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

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------
    def load_unet_weights(self, path: str, *, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device)
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
        state = torch.load(path, map_location=self.device)
        missing, unexpected = self.vae.load_state_dict(state, strict=strict)
        if (not strict) or missing or unexpected:
            print(f"[load_vae_weights] strict={strict}")
            if missing:
                print("  Missing keys:", missing)
            if unexpected:
                print("  Unexpected keys:", unexpected)

    # ------------------------------------------------------------------
    # Encode helper (pixel-space passthrough or VAE)
    # ------------------------------------------------------------------
    def encode_fm_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return x
        with torch.no_grad():
            z_mu, z_sigma = self.vae.encode(x)
            return self.vae.sampling(z_mu, z_sigma)

    # ------------------------------------------------------------------
    # Flow-matching loss
    # ------------------------------------------------------------------
    def flow_matching_step(self, x_fm: torch.Tensor, cond_kwargs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Compute a single flow-matching loss on encoded input *x_fm*."""
        if cond_kwargs is None:
            cond_kwargs = {}
        B = x_fm.shape[0]
        z0 = torch.randn_like(x_fm)
        t = torch.rand(B, device=x_fm.device)
        t_expanded = t[:, None, None, None]

        zt = (1 - t_expanded) * z0 + t_expanded * x_fm
        v_target = x_fm - z0

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
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        if patience is not None and eval_dataloader is None:
            raise ValueError("eval_dataloader must be provided when using patience early stopping.")

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

        optimizer = Adam(self.unet.parameters(), lr=1e-4)

        # Resume state
        global_step = 0
        best_eval = float("inf")
        best_epoch = -1
        bad_epochs = 0
        start_epoch = 0

        if resume_from_checkpoint is not None:
            print(f"[Resume] Loading checkpoint from {resume_from_checkpoint}")
            ckpt = torch.load(resume_from_checkpoint, map_location=self.device)
            self.unet.load_state_dict(ckpt["unet_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
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
            print(f"[Resume] Resuming from epoch {start_epoch}, global_step={global_step}, best_eval={best_eval:.6f}")

        writer = SummaryWriter(log_dir)

        def _save_checkpoint(path: str, epoch_idx: int) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            ckpt = {
                "epoch": epoch_idx,
                "global_step": global_step,
                "unet_state": self.unet.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_eval": best_eval,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "t_scale": self.t_scale,
                "train_target": self.train_target,
                "rng_state": torch.random.get_rng_state(),
            }
            if torch.cuda.is_available():
                ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            torch.save(ckpt, path)

        def _set_epoch_for_dataloader(dl: Optional[DataLoader], epoch_idx: int) -> None:
            if dl is None:
                return
            ds = getattr(dl, "dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch_idx)
            if ds is not None and hasattr(ds, "transform") and hasattr(ds.transform, "set_epoch"):
                ds.transform.set_epoch(epoch_idx)

        sampler_obj = self._make_sampler() if sample_every > 0 else None

        for epoch in range(start_epoch, epochs):
            _set_epoch_for_dataloader(dataloader, epoch)
            _set_epoch_for_dataloader(eval_dataloader, epoch)
            self.unet.train()
            total_loss = 0.0

            for x in tqdm(dataloader, desc=f"FM Epoch {epoch+1}/{epochs}"):
                x = x.to(self.device)
                x_fm = self.encode_fm_input(x)
                cond_kw = self.conditioner.prepare_for_training(x, self.device) if self.conditioner is not None else {}
                loss = self.flow_matching_step(x_fm, cond_kw)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                writer.add_scalar("fm/loss_step", loss.item(), global_step)
                global_step += 1

            avg_loss = total_loss / max(1, len(dataloader))
            print(f"[FM Epoch {epoch+1}] loss: {avg_loss:.6f}")
            writer.add_scalar("fm/loss_epoch", avg_loss, epoch)

            if (save_every_n_epochs is not None) and ((epoch + 1) % save_every_n_epochs == 0):
                self.save_unet_weights(os.path.join(self._unet_dir(), f"unet_fm_epoch_{epoch+1}.pt"))
                _save_checkpoint(
                    os.path.join(self._unet_dir(), f"unet_fm_epoch_{epoch+1}_ckpt.pt"),
                    epoch_idx=epoch,
                )

            # Eval + early stopping + best save
            if patience is not None and eval_dataloader is not None:
                self.unet.eval()
                eval_loss = 0.0
                n_eval = 0

                with torch.no_grad():
                    for x in tqdm(eval_dataloader, desc=f"FM Eval  {epoch+1}/{epochs}"):
                        x = x.to(self.device)
                        x_fm = self.encode_fm_input(x)
                        cond_kw = self.conditioner.prepare_for_training(x, self.device) if self.conditioner is not None else {}
                        loss = self.flow_matching_step(x_fm, cond_kw)

                        bs = x.size(0)
                        eval_loss += loss.item() * bs
                        n_eval += bs

                avg_eval_loss = eval_loss / max(1, n_eval)
                print(f"  [Eval loss: {avg_eval_loss:.6f}]")
                writer.add_scalar("fm/eval_loss_epoch", avg_eval_loss, epoch)

                improved = (best_eval - avg_eval_loss) > min_delta
                if improved:
                    best_eval = avg_eval_loss
                    best_epoch = epoch
                    bad_epochs = 0
                    self.save_unet_weights(os.path.join(self._unet_dir(), "unet_fm_best.pt"))
                    print(f"  ✅ New best eval_loss={best_eval:.6f} at epoch {epoch+1} -> saved UNET/unet_fm_best.pt")
                else:
                    bad_epochs += 1
                    print(f"  ⏳ No improvement (best={best_eval:.6f}), bad_epochs={bad_epochs}/{patience}")
                    if bad_epochs >= patience:
                        print(f"🛑 Early stopping triggered. Best epoch: {best_epoch+1} (eval_loss={best_eval:.6f})")
                        break

            # Sampling
            if sampler_obj is not None and (epoch + 1) % sample_every == 0:
                sampler_obj.log_samples_to_tensorboard(
                    writer=writer,
                    epoch=epoch,
                    steps=sample_steps,
                    batch_size=sample_batch_size,
                    tag="fm/generated",
                    sample_shape=sample_shape,
                )

        writer.close()


# ── registry ──────────────────────────────────────────────────────────────────
from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.trainer.register("default_fm", default=True)(FlowMatchingTrainer)
