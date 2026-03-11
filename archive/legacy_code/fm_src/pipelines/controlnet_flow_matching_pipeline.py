"""
ControlNet extension for StableFlowMatchingPipeline.

Stage 2: Train a ControlNet on top of a frozen, pretrained UNet
using bounding-box spatial conditioning.

This module adds:
  - ControlNetModel        – mirrors the UNet encoder + zero-conv outputs
  - unet_forward_with_controlnet – custom UNet forward with residual injection
  - ControlNetFlowMatchingPipeline(StableFlowMatchingPipeline)

Nothing in the base pipeline or its training logic is modified.
"""

import copy
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffusers import UNet2DModel

from fm_src.pipelines.flow_matching_pipeline import StableFlowMatchingPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize all parameters of *module* and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ---------------------------------------------------------------------------
# ControlNet model
# ---------------------------------------------------------------------------


class ControlNetModel(nn.Module):
    """
    ControlNet that mirrors the encoder half of a ``UNet2DModel``.

    * A small conditioning encoder converts a spatial conditioning image
      (e.g. a bounding-box mask at image resolution) into features at the
      latent resolution and adds them to the ``conv_in`` output.
    * The encoder (down-blocks + mid-block) is deep-copied from the
      pre-trained UNet so that training starts from a good initialisation.
    * Zero convolutions on every skip-connection output guarantee that the
      ControlNet contributes *nothing* at the start of training – the
      augmented UNet therefore behaves identically to the base UNet.
    """

    def __init__(
        self,
        unet: UNet2DModel,
        conditioning_channels: int = 1,
        conditioning_downscale_factor: int = 4,
    ):
        super().__init__()

        block_out_ch0 = unet.config.block_out_channels[0]

        # --- conditioning encoder -------------------------------------------
        # Downsamples from image resolution to latent resolution.
        # Default factor = 4  (256×256 → 64×64 for VAE with 2 downsamples).
        cond_layers = [
            nn.Conv2d(conditioning_channels, 16, 3, stride=1, padding=1),
            nn.SiLU(),
        ]
        ch_in = 16
        n_stride2 = 0
        factor = conditioning_downscale_factor
        while factor > 1:
            ch_out = min(ch_in * 2, 96)
            cond_layers += [
                nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                nn.SiLU(),
            ]
            ch_in = ch_out
            factor //= 2
            n_stride2 += 1
        # Final projection to match conv_in output channels
        cond_layers.append(nn.Conv2d(ch_in, block_out_ch0, 3, stride=1, padding=1))
        self.controlnet_cond_embedding = nn.Sequential(*cond_layers)

        # --- encoder copied from UNet ---------------------------------------
        self.time_proj = copy.deepcopy(unet.time_proj)
        self.time_embedding = copy.deepcopy(unet.time_embedding)
        self.conv_in = copy.deepcopy(unet.conv_in)
        self.down_blocks = copy.deepcopy(unet.down_blocks)
        self.mid_block = copy.deepcopy(unet.mid_block)

        # --- zero convolutions ----------------------------------------------
        down_channels, mid_ch = self._compute_residual_channels(unet.config)

        self.controlnet_down_blocks = nn.ModuleList(
            [zero_module(nn.Conv2d(ch, ch, 1)) for ch in down_channels]
        )
        self.controlnet_mid_block = zero_module(nn.Conv2d(mid_ch, mid_ch, 1))

    # --------------------------------------------------------------------- #

    @staticmethod
    def _compute_residual_channels(config) -> Tuple[list, int]:
        """Return the channel count for every skip-connection tensor and
        the mid-block output channel count."""
        block_out_channels = list(config.block_out_channels)
        layers_per_block = getattr(config, "layers_per_block", 2)
        n_blocks = len(block_out_channels)

        # First entry comes from conv_in
        channels = [block_out_channels[0]]
        for i, ch in enumerate(block_out_channels):
            for _ in range(layers_per_block):
                channels.append(ch)
            # All blocks except the last one have a down-sampler
            if i < n_blocks - 1:
                channels.append(ch)

        mid_ch = block_out_channels[-1]
        return channels, mid_ch

    # --------------------------------------------------------------------- #

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        conditioning_image: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Parameters
        ----------
        sample : (B, C_latent, H_lat, W_lat)
            Noisy latent tensor.
        timestep : (B,)
            Scaled timesteps.
        conditioning_image : (B, cond_channels, H_img, W_img)
            Spatial conditioning (bbox mask at image resolution).

        Returns
        -------
        (down_block_residuals, mid_block_residual)
            Residuals to be injected into the frozen UNet.
        """
        # -- time embedding ---------------------------------------------------
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # -- pre-process + conditioning ---------------------------------------
        sample = self.conv_in(sample)
        conditioning = self.controlnet_cond_embedding(conditioning_image)
        sample = sample + conditioning

        # -- encoder ----------------------------------------------------------
        down_block_res_samples: Tuple[torch.Tensor, ...] = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb
            )
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb)

        # -- zero convolutions ------------------------------------------------
        controlnet_down = tuple(
            zc(s)
            for zc, s in zip(self.controlnet_down_blocks, down_block_res_samples)
        )
        controlnet_mid = self.controlnet_mid_block(sample)

        return controlnet_down, controlnet_mid


# ---------------------------------------------------------------------------
# Custom UNet forward with ControlNet residuals
# ---------------------------------------------------------------------------


def unet_forward_with_controlnet(
    unet: UNet2DModel,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    down_block_additional_residuals: Tuple[torch.Tensor, ...],
    mid_block_additional_residual: torch.Tensor,
    conditioning_scale: float = 1.0,
) -> torch.Tensor:
    """
    Run the frozen ``UNet2DModel`` forward pass while injecting ControlNet
    residuals into the skip connections and mid-block output.

    This replicates the logic of ``UNet2DModel.forward`` but adds
    *conditioning_scale × residual* to each skip tensor before the decoder
    consumes it.
    """
    # -- time embedding -------------------------------------------------------
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(
            [timesteps], dtype=torch.long, device=sample.device
        )
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = unet.time_proj(timesteps)
    t_emb = t_emb.to(dtype=sample.dtype)
    emb = unet.time_embedding(t_emb)

    # -- encoder --------------------------------------------------------------
    sample = unet.conv_in(sample)

    down_block_res_samples: Tuple[torch.Tensor, ...] = (sample,)
    for downsample_block in unet.down_blocks:
        sample, res_samples = downsample_block(
            hidden_states=sample, temb=emb
        )
        down_block_res_samples += res_samples

    sample = unet.mid_block(sample, emb)

    # -- inject ControlNet residuals ------------------------------------------
    sample = sample + mid_block_additional_residual * conditioning_scale

    down_block_res_samples = tuple(
        s + r * conditioning_scale
        for s, r in zip(down_block_res_samples, down_block_additional_residuals)
    )

    # -- decoder --------------------------------------------------------------
    for upsample_block in unet.up_blocks:
        n_res = len(upsample_block.resnets)
        res_samples = down_block_res_samples[-n_res:]
        down_block_res_samples = down_block_res_samples[:-n_res]
        sample = upsample_block(sample, res_samples, emb)

    # -- post-process ---------------------------------------------------------
    sample = unet.conv_norm_out(sample)
    sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)

    return sample


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ControlNetFlowMatchingPipeline(StableFlowMatchingPipeline):
    """
    Stage-2 pipeline that trains a **ControlNet** on top of a frozen,
    pre-trained UNet (from ``StableFlowMatchingPipeline``).

    Save structure::

        [model_dir]/
          VAE/            ← loaded from stage 1, frozen
          UNET/           ← loaded from stage 1, frozen
          CONTROLNET/
            config.json
            controlnet_epoch_*.pt
            controlnet_best.pt
    """

    def __init__(
        self,
        device: Optional[str] = None,
        t_scale: Optional[float] = None,
        model_dir: str = "./pipeline_model",
        sample_shape: Optional[Tuple[int, int, int]] = None,
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            device=device,
            t_scale=t_scale,
            model_dir=model_dir,
            sample_shape=sample_shape,
            from_norm_to_display=from_norm_to_display,
        )
        self.controlnet_config: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------

    def _controlnet_dir(self) -> str:
        return os.path.join(self.model_dir, "CONTROLNET")

    def _ensure_dirs(self) -> None:
        super()._ensure_dirs()
        os.makedirs(self._controlnet_dir(), exist_ok=True)

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    def build_controlnet(
        self,
        conditioning_channels: int = 1,
        conditioning_downscale_factor: int = 4,
        *,
        save_config: bool = True,
    ):
        """Initialise a ControlNet from the current (frozen) UNet."""
        assert hasattr(self, "unet"), (
            "UNet must be built and loaded before building the ControlNet."
        )
        self._ensure_dirs()

        self.controlnet = ControlNetModel(
            self.unet,
            conditioning_channels=conditioning_channels,
            conditioning_downscale_factor=conditioning_downscale_factor,
        ).to(self.device)

        self.controlnet_config = {
            "conditioning_channels": conditioning_channels,
            "conditioning_downscale_factor": conditioning_downscale_factor,
        }

        if save_config:
            self._save_json(
                os.path.join(self._controlnet_dir(), "config.json"),
                self.controlnet_config,
            )

        return self

    # -----------------------------------------------------------------
    # Freeze helper
    # -----------------------------------------------------------------

    def freeze_unet(self):
        """Set UNet to eval mode and disable all gradients."""
        assert hasattr(self, "unet"), "UNet not set."
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False
        return self

    # -----------------------------------------------------------------
    # Save / load
    # -----------------------------------------------------------------

    def save_controlnet_weights(self, path: str):
        assert hasattr(self, "controlnet"), "ControlNet not set."
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.controlnet.state_dict(), path)
        return self

    def load_controlnet_weights(
        self,
        path: str,
        *,
        strict: bool = True,
        map_location: str | None = None,
    ):
        assert hasattr(self, "controlnet"), (
            "ControlNet not set. Build it first with .build_controlnet(...)."
        )
        map_location = map_location or self.device
        state = torch.load(path, map_location=map_location)
        missing, unexpected = self.controlnet.load_state_dict(state, strict=strict)

        if (not strict) or missing or unexpected:
            print(f"[load_controlnet_weights] strict={strict}")
            if missing:
                print("  Missing keys:", missing)
            if unexpected:
                print("  Unexpected keys:", unexpected)

        return self

    def save_configs(self):
        super().save_configs()
        if self.controlnet_config is not None:
            self._save_json(
                os.path.join(self._controlnet_dir(), "config.json"),
                self.controlnet_config,
            )
        return self

    def load_controlnet_from_folder(
        self,
        pipeline_folder: str,
        *,
        controlnet_weights: str = "controlnet_best.pt",
        strict: bool = True,
        map_location: str | None = None,
    ):
        """
        Load ControlNet weights from a pipeline folder.
        Assumes VAE and UNet are already loaded.
        """
        cn_dir = os.path.join(pipeline_folder, "CONTROLNET")

        cfg_path = os.path.join(cn_dir, "config.json")
        if os.path.isfile(cfg_path):
            self.controlnet_config = self._load_json(cfg_path)
            cond_ch = self.controlnet_config.get("conditioning_channels", 1)
            cond_ds = self.controlnet_config.get("conditioning_downscale_factor", 4)
        else:
            cond_ch = 1
            cond_ds = 4

        if not hasattr(self, "controlnet"):
            self.build_controlnet(
                conditioning_channels=cond_ch,
                conditioning_downscale_factor=cond_ds,
                save_config=False,
            )

        w_path = os.path.join(cn_dir, controlnet_weights)
        if not os.path.isfile(w_path):
            # Try latest epoch
            w_path_alt = self._pick_latest_by_prefix(
                cn_dir, "controlnet_epoch_"
            )
            if w_path_alt is None:
                raise FileNotFoundError(
                    f"No ControlNet weights found in {cn_dir}"
                )
            w_path = w_path_alt

        self.load_controlnet_weights(
            w_path, strict=strict, map_location=map_location
        )
        return self

    # -----------------------------------------------------------------
    # ControlNet flow-matching step
    # -----------------------------------------------------------------

    def _controlnet_flow_matching_step(
        self,
        x_fm: torch.Tensor,
        conditioning_image: torch.Tensor,
        conditioning_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute the FM velocity-prediction MSE loss with ControlNet."""
        assert hasattr(self, "unet") and hasattr(self, "controlnet")

        B = x_fm.shape[0]
        z0 = torch.randn_like(x_fm)
        t = torch.rand(B, device=x_fm.device)

        zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * x_fm
        v_target = x_fm - z0

        # ControlNet  →  residuals
        cn_down, cn_mid = self.controlnet(
            zt, t * self.t_scale, conditioning_image
        )

        # Frozen UNet + residuals  →  velocity prediction
        v_pred = unet_forward_with_controlnet(
            self.unet,
            zt,
            t * self.t_scale,
            cn_down,
            cn_mid,
            conditioning_scale=conditioning_scale,
        )

        return F.mse_loss(v_pred, v_target)

    # -----------------------------------------------------------------
    # Sampling with ControlNet
    # -----------------------------------------------------------------

    @torch.no_grad()
    def sample_euler_with_controlnet(
        self,
        conditioning_image: torch.Tensor,
        steps: int = 50,
        batch_size: int = 4,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        conditioning_scale: float = 1.0,
    ) -> torch.Tensor:
        """Euler ODE sampling with ControlNet conditioning."""
        assert hasattr(self, "unet") and hasattr(self, "controlnet")
        self.unet.eval()
        self.controlnet.eval()

        shape = sample_shape or self._get_unet_sample_shape()
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
    def log_controlnet_samples_to_tensorboard(
        self,
        writer: SummaryWriter,
        epoch: int,
        conditioning_images: torch.Tensor,
        steps: int = 50,
        conditioning_scale: float = 1.0,
        tag: str = "controlnet/generated",
    ):
        """Generate and log ControlNet-conditioned samples."""
        assert hasattr(self, "vae") and hasattr(self, "unet") and hasattr(
            self, "controlnet"
        )
        self.vae.eval()
        self.unet.eval()
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
        # Log the conditioning masks alongside
        writer.add_images(f"{tag}_cond", conditioning_images, epoch)

    # -----------------------------------------------------------------
    # Training: ControlNet flow matching  (stage 2)
    # -----------------------------------------------------------------

    def train_controlnet_flow_matching(
        self,
        dataloader,
        epochs: int,
        eval_dataloader=None,
        lr: float = 1e-4,
        conditioning_scale: float = 1.0,
        log_dir: str = "./runs/controlnet_fm",
        sample_every_epoch: bool = True,
        sample_steps: int = 50,
        patience: int | None = None,
        min_delta: float = 0.0,
        save_every_n_epochs: int = 1,
        resume_from_checkpoint: str | None = None,
    ):
        """
        Train the ControlNet while keeping VAE and UNet frozen.

        The dataloader must yield dicts with keys:
          - ``pixel_values``               (B, C, H, W)
          - ``conditioning_pixel_values``   (B, 1, H_img, W_img)
        """
        assert hasattr(self, "vae"), "VAE not set."
        assert hasattr(self, "unet"), "UNet not set."
        assert hasattr(self, "controlnet"), (
            "ControlNet not set. Build it first with .build_controlnet(...)."
        )
        if patience is not None and eval_dataloader is None:
            raise ValueError(
                "eval_dataloader must be provided when using patience."
            )

        self._ensure_dirs()
        self.save_configs()

        # Freeze base models
        self.freeze_vae()
        self.freeze_unet()

        # Only ControlNet parameters are trainable
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
        cn_dir = self._controlnet_dir()

        def _save_checkpoint(path: str, epoch_idx: int) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
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

        # Grab a small fixed batch of conditioning images for sampling
        sample_cond = None
        if sample_every_epoch:
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
                loss = self._controlnet_flow_matching_step(
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
                        loss = self._controlnet_flow_matching_step(
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
                        f"  ✅ New best eval={best_eval:.6f} "
                        f"at epoch {epoch + 1}"
                    )
                else:
                    bad_epochs += 1
                    print(
                        f"  ⏳ No improvement (best={best_eval:.6f}), "
                        f"bad_epochs={bad_epochs}/{patience}"
                    )
                    if bad_epochs >= patience:
                        print(
                            f"🛑 Early stopping. "
                            f"Best epoch: {best_epoch + 1}"
                        )
                        break

            # --- sampling ---
            if sample_every_epoch and sample_cond is not None:
                self.log_controlnet_samples_to_tensorboard(
                    writer=writer,
                    epoch=epoch,
                    conditioning_images=sample_cond,
                    steps=sample_steps,
                    conditioning_scale=conditioning_scale,
                )
                # Unconditional baseline for comparison
                self.log_fm_samples_to_tensorboard(
                    writer=writer,
                    epoch=epoch,
                    steps=sample_steps,
                    batch_size=4,
                    tag="controlnet/baseline_uncond",
                )

        writer.close()
        return self
