"""Training logic for ControlNet flow-matching models.

Implements a personalized trainer for ControlNet following the architecture
from `Adding Conditional Control to Text-to-Image Diffusion Models
<https://arxiv.org/abs/2302.05543>`_ (Zhang et al., 2023).

The trainer freezes the pre-trained UNet and VAE, and trains only the
ControlNet parameters using a flow-matching velocity-prediction loss.

This module replaces the deprecated ``ControlNetFlowMatchingPipeline``
that previously lived in ``fm_src.pipelines``.
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

from src.models.controlnet import (
    ControlNetModel,
    unet_forward_with_controlnet,
    save_controlnet_config,
)


# ---------------------------------------------------------------------------
# Default display helper
# ---------------------------------------------------------------------------
def _default_from_norm_to_display(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


# ═══════════════════════════════════════════════════════════════════════════
# ControlNetTrainer
# ═══════════════════════════════════════════════════════════════════════════

class ControlNetTrainer:
    """Self-contained trainer for ControlNet on top of a frozen FM UNet.

    Following the ControlNet paper and the official implementation:
    * The pre-trained UNet and VAE are **frozen** (eval mode, no gradients).
    * Only the ControlNet parameters are trained.
    * The training objective is flow-matching velocity-prediction MSE.
    * Zero convolutions ensure the ControlNet contributes nothing at the
      start of training (faithful to the paper's design).

    Parameters
    ----------
    unet : UNet2DModel
        Pre-trained UNet (frozen during training).
    controlnet : ControlNetModel
        ControlNet to train.
    device : str or torch.device
    t_scale : float
        Time-scaling factor for UNet timestep input.
    model_dir : str
        Root output directory. ControlNet weights are saved under
        ``model_dir/CONTROLNET/``.
    from_norm_to_display : callable, optional
        ``[-1, 1] -> [0, 1]`` for TensorBoard visualisation.
    vae : optional
        If provided, training happens in latent space.
        The VAE is frozen automatically.
    """

    def __init__(
        self,
        unet: UNet2DModel,
        controlnet: ControlNetModel,
        *,
        device: Optional[Union[str, torch.device]] = None,
        t_scale: float = 1000.0,
        model_dir: str = "./controlnet_runs/bbox_controlnet",
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        vae=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.unet = unet
        self.controlnet = controlnet
        self.t_scale = float(t_scale)
        self.model_dir = model_dir
        self.from_norm_to_display = from_norm_to_display or _default_from_norm_to_display
        self.vae = vae

        # Freeze UNet
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False

        # Freeze VAE if present
        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def _controlnet_dir(self) -> str:
        return os.path.join(self.model_dir, "CONTROLNET")

    def _ensure_dirs(self) -> None:
        os.makedirs(self._controlnet_dir(), exist_ok=True)

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------
    def save_controlnet_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.controlnet.state_dict(), path)

    def load_controlnet_weights(
        self, path: str, *, strict: bool = True
    ) -> None:
        state = torch.load(path, map_location=self.device)
        missing, unexpected = self.controlnet.load_state_dict(
            state, strict=strict
        )
        if (not strict) or missing or unexpected:
            print(f"[load_controlnet_weights] strict={strict}")
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

    def decode_fm_output(self, z: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return z
        with torch.no_grad():
            return self.vae.decode(z)

    # ------------------------------------------------------------------
    # Flow-matching loss with ControlNet
    # ------------------------------------------------------------------
    def controlnet_flow_matching_step(
        self,
        x_fm: torch.Tensor,
        conditioning_image: torch.Tensor,
        conditioning_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute the FM velocity-prediction MSE loss with ControlNet."""
        B = x_fm.shape[0]
        z0 = torch.randn_like(x_fm)
        t = torch.rand(B, device=x_fm.device)

        zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * x_fm
        v_target = x_fm - z0

        # ControlNet -> residuals
        cn_down, cn_mid = self.controlnet(
            zt, t * self.t_scale, conditioning_image
        )

        # Frozen UNet + residuals -> velocity prediction
        v_pred = unet_forward_with_controlnet(
            self.unet,
            zt,
            t * self.t_scale,
            cn_down,
            cn_mid,
            conditioning_scale=conditioning_scale,
        )

        return F.mse_loss(v_pred, v_target)

    # ------------------------------------------------------------------
    # Sampling with ControlNet (for TensorBoard visualisation)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_euler_with_controlnet(
        self,
        conditioning_image: torch.Tensor,
        steps: int = 50,
        batch_size: int = 4,
        conditioning_scale: float = 1.0,
    ) -> torch.Tensor:
        """Euler ODE sampling with ControlNet conditioning."""
        self.unet.eval()
        self.controlnet.eval()

        # Derive sample shape from UNet config.
        in_ch = self.unet.config.in_channels
        sample_size = self.unet.config.sample_size
        if isinstance(sample_size, (list, tuple)):
            h, w = sample_size
        else:
            h = w = sample_size
        shape = (in_ch, h, w)

        z = torch.randn(batch_size, *shape, device=self.device)
        cond = conditioning_image.to(self.device)
        if cond.shape[0] != batch_size:
            cond = cond.expand(batch_size, -1, -1, -1)

        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=self.device)

            cn_down, cn_mid = self.controlnet(z, t * self.t_scale, cond)
            v = unet_forward_with_controlnet(
                self.unet,
                z,
                t * self.t_scale,
                cn_down,
                cn_mid,
                conditioning_scale=conditioning_scale,
            )
            z = z + v / steps

        return z

    @torch.no_grad()
    def _log_controlnet_samples(
        self,
        writer: SummaryWriter,
        epoch: int,
        conditioning_images: torch.Tensor,
        steps: int = 50,
        conditioning_scale: float = 1.0,
        tag: str = "controlnet/generated",
    ) -> None:
        """Generate and log ControlNet-conditioned samples to TensorBoard."""
        self.controlnet.eval()

        B = conditioning_images.shape[0]
        z = self.sample_euler_with_controlnet(
            conditioning_images,
            steps=steps,
            batch_size=B,
            conditioning_scale=conditioning_scale,
        )
        x_gen = self.decode_fm_output(z)
        x_vis = self.from_norm_to_display(x_gen).clamp(0, 1)

        writer.add_images(tag, x_vis, epoch)
        writer.add_images(f"{tag}_cond", conditioning_images, epoch)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        *,
        lr: float = 1e-4,
        conditioning_scale: float = 1.0,
        log_dir: str = "./artifacts/runs/main/controlnet",
        sample_every: int = 1,
        sample_steps: int = 50,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        save_every_n_epochs: int = 10,
        resume_from_checkpoint: Optional[str] = None,
    ) -> None:
        """Train the ControlNet while keeping VAE and UNet frozen.

        The dataloader must yield dicts with keys:
          - ``pixel_values``               (B, C, H, W)
          - ``conditioning_pixel_values``   (B, 1, H_img, W_img)
        """
        if patience is not None and eval_dataloader is None:
            raise ValueError(
                "eval_dataloader must be provided when using patience."
            )

        self._ensure_dirs()
        cn_dir = self._controlnet_dir()

        optimizer = Adam(self.controlnet.parameters(), lr=lr)

        # Resume state
        global_step = 0
        best_eval = float("inf")
        best_epoch = -1
        bad_epochs = 0
        start_epoch = 0

        if resume_from_checkpoint is not None:
            print(
                f"[Resume] Loading ControlNet checkpoint: "
                f"{resume_from_checkpoint}"
            )
            ckpt = torch.load(
                resume_from_checkpoint, map_location=self.device
            )
            self.controlnet.load_state_dict(ckpt["controlnet_state"])
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
            print(
                f"[Resume] epoch={start_epoch}, step={global_step}, "
                f"best_eval={best_eval:.6f}"
            )

        writer = SummaryWriter(log_dir)

        def _save_checkpoint(path: str, epoch_idx: int) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            ckpt = {
                "epoch": epoch_idx,
                "global_step": global_step,
                "controlnet_state": self.controlnet.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_eval": best_eval,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "t_scale": self.t_scale,
                "conditioning_scale": conditioning_scale,
            }
            torch.save(ckpt, path)

        # Grab a small fixed batch of conditioning images for sampling.
        sample_cond = None
        if sample_every > 0:
            sample_dl = eval_dataloader or dataloader
            sample_batch = next(iter(sample_dl))
            sample_cond = sample_batch["conditioning_pixel_values"][:4].to(
                self.device
            )

        for epoch in range(start_epoch, epochs):
            self.controlnet.train()
            total_loss = 0.0

            for batch in tqdm(
                dataloader, desc=f"CN Epoch {epoch + 1}/{epochs}"
            ):
                x = batch["pixel_values"].to(self.device)
                cond = batch["conditioning_pixel_values"].to(self.device)

                x_fm = self.encode_fm_input(x)
                loss = self.controlnet_flow_matching_step(
                    x_fm, cond, conditioning_scale
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                writer.add_scalar(
                    "controlnet/loss_step", loss.item(), global_step
                )
                global_step += 1

            avg_loss = total_loss / max(1, len(dataloader))
            print(f"[CN Epoch {epoch + 1}] loss: {avg_loss:.6f}")
            writer.add_scalar("controlnet/loss_epoch", avg_loss, epoch)

            # --- checkpoint ---
            if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                self.save_controlnet_weights(
                    os.path.join(
                        cn_dir, f"controlnet_epoch_{epoch + 1}.pt"
                    )
                )
                _save_checkpoint(
                    os.path.join(
                        cn_dir, f"controlnet_epoch_{epoch + 1}_ckpt.pt"
                    ),
                    epoch_idx=epoch,
                )

            # --- eval + early stopping ---
            if patience is not None and eval_dataloader is not None:
                self.controlnet.eval()
                eval_loss = 0.0
                n_eval = 0

                with torch.no_grad():
                    for batch in tqdm(
                        eval_dataloader,
                        desc=f"CN Eval  {epoch + 1}/{epochs}",
                    ):
                        x = batch["pixel_values"].to(self.device)
                        cond = batch["conditioning_pixel_values"].to(
                            self.device
                        )

                        x_fm = self.encode_fm_input(x)
                        loss = self.controlnet_flow_matching_step(
                            x_fm, cond, conditioning_scale
                        )

                        bs = x.size(0)
                        eval_loss += loss.item() * bs
                        n_eval += bs

                avg_eval = eval_loss / max(1, n_eval)
                print(f"  [Eval loss: {avg_eval:.6f}]")
                writer.add_scalar(
                    "controlnet/eval_loss_epoch", avg_eval, epoch
                )

                improved = (best_eval - avg_eval) > min_delta
                if improved:
                    best_eval = avg_eval
                    best_epoch = epoch
                    bad_epochs = 0
                    self.save_controlnet_weights(
                        os.path.join(cn_dir, "controlnet_best.pt")
                    )
                    print(
                        f"  New best eval={best_eval:.6f} "
                        f"at epoch {epoch + 1}"
                    )
                else:
                    bad_epochs += 1
                    print(
                        f"  No improvement (best={best_eval:.6f}), "
                        f"bad_epochs={bad_epochs}/{patience}"
                    )
                    if bad_epochs >= patience:
                        print(
                            f"Early stopping. "
                            f"Best epoch: {best_epoch + 1}"
                        )
                        break

            # --- sampling ---
            if sample_cond is not None and (epoch + 1) % sample_every == 0:
                self._log_controlnet_samples(
                    writer=writer,
                    epoch=epoch,
                    conditioning_images=sample_cond,
                    steps=sample_steps,
                    conditioning_scale=conditioning_scale,
                )

        writer.close()


# ── registry ──────────────────────────────────────────────────────────────

from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.trainer.register("controlnet")(ControlNetTrainer)
