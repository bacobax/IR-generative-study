import os
import json
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# You already use these outside; now we instantiate inside.
from diffusers import UNet2DModel
from generative.networks.nets import AutoencoderKL

from src.models.fm_unet import (
    load_unet_config,
    build_fm_unet_from_config,
    save_unet_config,
)


# Default normalization helpers (simple full-range [-1,1] <-> [0,1])
def _default_from_norm_to_display(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for display / TensorBoard."""
    return (x + 1) / 2


class FlowMatchingPipeline:
    """
    .. deprecated::
        This monolithic pipeline is **deprecated**.
        Use ``src.algorithms.training.flow_matching_trainer.FlowMatchingTrainer``
        for training and ``src.algorithms.inference.flow_matching_sampler.FlowMatchingSampler``
        for inference instead.

    Base flow-matching pipeline that operates directly in pixel space
    (no VAE). The UNet is trained to predict the velocity field from
    noisy interpolations between z0 and x (here, x is the pixel tensor).

    Save structure:
        [model_dir]/
          UNET/
            config.json
            unet_fm_epoch_*.pt
            unet_fm_best.pt
    """

    def __init__(
        self,
        device: Optional[str] = None,
        t_scale: Optional[float] = None,
        model_dir: str = "./pipeline_model",
        sample_shape: Optional[Tuple[int, int, int]] = None,
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        train_target: str = "v",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # scaling factor for time input to UNet
        self.t_scale = 1.0 if t_scale is None else float(t_scale)

        # Root folder for all saves
        self.model_dir = model_dir

        # Default sampling shape (C, H, W). If None, derived from UNet config.
        self.sample_shape = sample_shape

        # Normalization: [-1,1] -> [0,1] for TensorBoard display.
        # Override this to match whatever normalization the training script uses.
        self.from_norm_to_display = from_norm_to_display or _default_from_norm_to_display

        # Prediction target: "v" (velocity) or "x0" (clean sample)
        assert train_target in ("v", "x0"), f"train_target must be 'v' or 'x0', got '{train_target}'"
        self.train_target = train_target

        # Remember config if provided
        self.unet_config: Optional[Dict[str, Any]] = None

    # -------------------------
    # Paths / JSON helpers
    # -------------------------
    def _unet_dir(self) -> str:
        return os.path.join(self.model_dir, "UNET")

    def _ensure_dirs(self) -> None:
        os.makedirs(self._unet_dir(), exist_ok=True)

    @staticmethod
    def _save_json(path: str, obj: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------
    # Model builders (UNet only)
    # -------------------------
    def build_unet(self, config: Dict[str, Any], *, save_config: bool = True):
        """
        Instantiate UNet2DModel inside pipeline.
        """
        self._ensure_dirs()
        self.unet_config = dict(config)

        self.unet = build_fm_unet_from_config(config, device=self.device)

        if save_config:
            save_unet_config(self.unet_config, os.path.join(self._unet_dir(), "config.json"))

        return self

    def build_from_configs(
        self,
        *,
        unet_config: Optional[Dict[str, Any]] = None,
        unet_json: Optional[str] = None,
        combined_json: Optional[str] = None,
        save_configs: bool = True,
    ):
        """
        Convenience:
        - pass dict, or
        - load from JSON, or
        - load from a single JSON containing {"UNET": {...}}.
        Priority:
        - explicit dicts override JSON
        - combined_json fills missing ones
        """
        self._ensure_dirs()

        try:
            cfg = load_unet_config(
                path=unet_json,
                config_dict=unet_config,
                combined_json=combined_json,
            )
        except ValueError:
            cfg = None

        if cfg is not None:
            self.build_unet(cfg, save_config=save_configs)

        return self

    def save_configs(self):
        """
        Explicitly save configs (useful if you built models then changed model_dir).
        """
        self._ensure_dirs()
        if self.unet_config is not None:
            save_unet_config(self.unet_config, os.path.join(self._unet_dir(), "config.json"))
        return self

    # -------------------------
    # Backward-compat add_*
    # -------------------------
    def add_unet(self, unet):
        self.unet = unet.to(self.device)
        return self

    # -------------------------
    # Loading / saving helpers (UPDATED paths)
    # -------------------------
    def load_unet_weights(self, path: str, *, strict: bool = True, map_location: str | None = None):
        assert hasattr(self, "unet"), "UNet not set. Build it first with .build_unet(...) or .build_from_configs(...)."
        map_location = map_location or self.device
        state = torch.load(path, map_location=map_location)
        missing, unexpected = self.unet.load_state_dict(state, strict=strict)

        if (not strict) or missing or unexpected:
            print(f"[load_unet_weights] strict={strict}")
            if missing:
                print("  Missing keys:", missing)
            if unexpected:
                print("  Unexpected keys:", unexpected)

        return self

    def load_pretrained(
        self,
        *,
        unet_path: str | None = None,
        strict: bool = True,
        map_location: str | None = None,
        set_eval: bool = False,
    ):
        if unet_path is not None:
            self.load_unet_weights(unet_path, strict=strict, map_location=map_location)

        if set_eval and hasattr(self, "unet"):
            self.unet.eval()
        return self

    def save_unet_weights(self, path: str):
        assert hasattr(self, "unet"), "UNet not set."
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.unet.state_dict(), path)
        return self

    # -------------------------
    # Flow Matching (pixel space)
    # -------------------------
    def _get_unet_sample_shape(self) -> Tuple[int, int, int]:
        if self.sample_shape is not None:
            return self.sample_shape

        if not hasattr(self, "unet") or not hasattr(self.unet, "config"):
            raise ValueError("Sample shape not set and UNet config is not available.")

        in_channels = self.unet.config.in_channels
        sample_size = self.unet.config.sample_size
        if isinstance(sample_size, Sequence):
            h, w = sample_size
        else:
            h = w = sample_size
        return (in_channels, h, w)

    def encode_fm_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Base pipeline encodes directly in pixel space.
        Override in subclasses for latent-space training.
        """
        return x

    def decode_fm_output(self, z: torch.Tensor) -> torch.Tensor:
        """
        Base pipeline returns pixel space output.
        Override in subclasses for latent-space sampling.
        """
        return z

    def _flow_matching_step(self, x_fm: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "unet"), "UNet not set. Build it first."

        B = x_fm.shape[0]
        z0 = torch.randn_like(x_fm)
        t = torch.rand(B, device=x_fm.device)
        t_expanded = t[:, None, None, None]

        zt = (1 - t_expanded) * z0 + t_expanded * x_fm
        v_target = x_fm - z0

        unet_out = self.unet(zt, t * self.t_scale).sample

        if self.train_target == "x0":
            # UNet predicts x0; reconstruct velocity for loss
            x0_pred = unet_out
            v_pred = (x0_pred - zt) / (1 - t_expanded).clamp(min=1e-5)
        else:
            # UNet directly predicts velocity
            v_pred = unet_out

        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample_euler(
        self,
        steps: int = 50,
        batch_size: int = 4,
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        assert hasattr(self, "unet"), "UNet not set."
        self.unet.eval()

        shape = sample_shape or self._get_unet_sample_shape()
        z = torch.randn(batch_size, *shape, device=self.device)
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t = torch.full((batch_size,), t_val, device=self.device)
            unet_out = self.unet(z, t * self.t_scale).sample

            if self.train_target == "x0":
                # UNet predicts x0; derive velocity
                t_expanded = t[:, None, None, None]
                v = (unet_out - z) / (1 - t_expanded).clamp(min=1e-5)
            else:
                v = unet_out

            z = z + v * dt
        return z

    @torch.no_grad()
    def log_fm_samples_to_tensorboard(
        self,
        writer: SummaryWriter,
        epoch: int,
        steps: int = 50,
        batch_size: int = 4,
        tag: str = "fm_samples",
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ):
        assert hasattr(self, "unet"), "UNet not set."
        self.unet.eval()

        z = self.sample_euler(steps=steps, batch_size=batch_size, sample_shape=sample_shape)
        x_gen = self.decode_fm_output(z)
        x_vis = self.from_norm_to_display(x_gen).clamp(0, 1)
        writer.add_images(tag, x_vis, epoch)

    def train_flow_matching(
        self,
        dataloader,
        epochs,
        eval_dataloader=None,
        pretrained_unet_path: str | None = None,
        strict_load: bool = True,
        log_dir: str = "./runs/flow_matching",
        sample_every_epoch: bool = True,
        sample_steps: int = 50,
        sample_batch_size: int = 4,
        patience: int | None = None,
        min_delta: float = 0.0,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        save_every_n_epochs: int = 1,
        resume_from_checkpoint: str | None = None,
    ):
        assert hasattr(self, "unet"), "UNet not set. Build it first."
        if patience is not None and eval_dataloader is None:
            raise ValueError("eval_dataloader must be provided when using patience early stopping.")

        self._ensure_dirs()
        self.save_configs()

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
                for state in ckpt["cuda_rng_state_all"]:
                    if not torch.is_tensor(state) or state.dtype != torch.uint8:
                        state = torch.tensor(state, dtype=torch.uint8)
                    if state.device.type != "cpu":
                        state = state.cpu()
                    cuda_states.append(state)
                torch.cuda.set_rng_state_all(cuda_states)
            print(f"[Resume] Resuming from epoch {start_epoch}, global_step={global_step}, best_eval={best_eval:.6f}")

        writer = SummaryWriter(log_dir)

        def _save_checkpoint(path: str, epoch_idx: int) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
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

        def _set_epoch_for_dataloader(dl, epoch_idx: int) -> None:
            if dl is None:
                return
            ds = getattr(dl, "dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch_idx)
            if ds is not None and hasattr(ds, "transform") and hasattr(ds.transform, "set_epoch"):
                ds.transform.set_epoch(epoch_idx)

        for epoch in range(start_epoch, epochs):
            _set_epoch_for_dataloader(dataloader, epoch)
            _set_epoch_for_dataloader(eval_dataloader, epoch)
            self.unet.train()
            total_loss = 0.0

            for x in tqdm(dataloader, desc=f"FM Epoch {epoch+1}/{epochs}"):
                x = x.to(self.device)

                x_fm = self.encode_fm_input(x)
                loss = self._flow_matching_step(x_fm)

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
                # ---- Save epoch checkpoint to model_dir/UNET/ ----
                self.save_unet_weights(os.path.join(self._unet_dir(), f"unet_fm_epoch_{epoch+1}.pt"))
                _save_checkpoint(
                    os.path.join(self._unet_dir(), f"unet_fm_epoch_{epoch+1}_ckpt.pt"),
                    epoch_idx=epoch,
                )

            # Eval + early stopping + best save to model_dir/UNET/
            if patience is not None and eval_dataloader is not None:
                self.unet.eval()
                eval_loss = 0.0
                n_eval = 0

                with torch.no_grad():
                    for x in tqdm(eval_dataloader, desc=f"FM Eval  {epoch+1}/{epochs}"):
                        x = x.to(self.device)
                        x_fm = self.encode_fm_input(x)
                        loss = self._flow_matching_step(x_fm)

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
            if sample_every_epoch:
                self.log_fm_samples_to_tensorboard(
                    writer=writer,
                    epoch=epoch,
                    steps=sample_steps,
                    batch_size=sample_batch_size,
                    tag="fm/generated",
                    sample_shape=sample_shape,
                )

        writer.close()
        return self

    # -------------------------
    # Restore from folder (UNet only)
    # -------------------------
    def load_from_pipeline_folder(
        self,
        pipeline_folder: str,
        *,
        unet_weights: str = "unet_fm_best.pt",
        unet_config_name: str = "config.json",
        strict: bool = True,
        map_location: Optional[str] = None,
        set_eval: bool = True,
    ):
        self.model_dir = pipeline_folder
        map_location = map_location or self.device

        unet_dir = self._unet_dir()
        unet_cfg_path = os.path.join(unet_dir, unet_config_name)

        if not os.path.isfile(unet_cfg_path):
            raise FileNotFoundError(f"Missing UNET config: {unet_cfg_path}")

        unet_cfg = self._load_json(unet_cfg_path)
        self.build_unet(unet_cfg, save_config=False)

        unet_w_path = os.path.join(unet_dir, unet_weights)
        if not os.path.isfile(unet_w_path):
            raise FileNotFoundError(f"Missing UNET weights: {unet_w_path}")

        self.load_unet_weights(unet_w_path, strict=strict, map_location=map_location)

        if set_eval:
            self.unet.eval()
        return self

    # -------------------------
    # OPTIONAL: auto-pick "latest" epoch checkpoint if best not present
    # -------------------------
    @staticmethod
    def _pick_latest_by_prefix(folder: str, prefix: str, suffix: str = ".pt") -> Optional[str]:
        if not os.path.isdir(folder):
            return None
        best_i = None
        best_path = None
        for fn in os.listdir(folder):
            if not (fn.startswith(prefix) and fn.endswith(suffix)):
                continue
            mid = fn[len(prefix) : -len(suffix)]
            try:
                i = int(mid)
            except ValueError:
                continue
            if best_i is None or i > best_i:
                best_i = i
                best_path = os.path.join(folder, fn)
        return best_path

    def load_from_pipeline_folder_auto(
        self,
        pipeline_folder: str,
        *,
        strict: bool = True,
        map_location: Optional[str] = None,
        set_eval: bool = True,
    ):
        self.model_dir = pipeline_folder
        map_location = map_location or self.device

        unet_dir = self._unet_dir()
        unet_cfg_path = os.path.join(unet_dir, "config.json")
        if not os.path.isfile(unet_cfg_path):
            raise FileNotFoundError(f"Missing UNET config: {unet_cfg_path}")

        unet_cfg = self._load_json(unet_cfg_path)
        self.build_unet(unet_cfg, save_config=False)

        unet_best = os.path.join(unet_dir, "unet_fm_best.pt")
        unet_w_path = unet_best if os.path.isfile(unet_best) else self._pick_latest_by_prefix(unet_dir, "unet_fm_epoch_")

        if unet_w_path is None or not os.path.isfile(unet_w_path):
            raise FileNotFoundError(f"No UNET weights found in {unet_dir} (expected unet_fm_best.pt or unet_fm_epoch_*.pt)")

        self.load_unet_weights(unet_w_path, strict=strict, map_location=map_location)

        if set_eval:
            self.unet.eval()

        return self


class StableFlowMatchingPipeline(FlowMatchingPipeline):
    """
    .. deprecated::
        This monolithic pipeline is **deprecated**.
        Use ``src.algorithms.training.flow_matching_trainer.FlowMatchingTrainer``
        for training and ``src.algorithms.inference.flow_matching_sampler.FlowMatchingSampler``
        for inference instead.

    Stable flow-matching pipeline that operates in latent space using a VAE.

    Save structure:
        [model_dir]/
          VAE/
            config.json
            vae_epoch_*.pt
            vae_best.pt
          UNET/
            config.json
            unet_fm_epoch_*.pt
            unet_fm_best.pt
    """

    def __init__(
        self,
        device: Optional[str] = None,
        t_scale: Optional[float] = None,
        model_dir: str = "./pipeline_model",
        sample_shape: Optional[Tuple[int, int, int]] = None,
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        train_target: str = "v",
    ):
        super().__init__(device=device, t_scale=t_scale, model_dir=model_dir, sample_shape=sample_shape, from_norm_to_display=from_norm_to_display, train_target=train_target)
        self.vae_config: Optional[Dict[str, Any]] = None

    def _vae_dir(self) -> str:
        return os.path.join(self.model_dir, "VAE")

    def _ensure_dirs(self) -> None:
        os.makedirs(self._vae_dir(), exist_ok=True)
        os.makedirs(self._unet_dir(), exist_ok=True)

    # -------------------------
    # Model builders (VAE + UNet)
    # -------------------------
    def build_vae(self, config: Dict[str, Any], *, save_config: bool = True):
        self._ensure_dirs()
        self.vae_config = dict(config)

        self.vae = AutoencoderKL(**config).to(self.device)

        if save_config:
            self._save_json(os.path.join(self._vae_dir(), "config.json"), self.vae_config)

        return self

    def build_from_configs(
        self,
        *,
        vae_config: Optional[Dict[str, Any]] = None,
        unet_config: Optional[Dict[str, Any]] = None,
        vae_json: Optional[str] = None,
        unet_json: Optional[str] = None,
        combined_json: Optional[str] = None,
        save_configs: bool = True,
    ):
        self._ensure_dirs()

        combined = None
        if combined_json is not None:
            combined = self._load_json(combined_json)

        if vae_config is None:
            if vae_json is not None:
                vae_config = self._load_json(vae_json)
            elif combined is not None and "VAE" in combined:
                vae_config = combined["VAE"]

        # Delegate UNet config resolution to the centralised helper
        try:
            unet_cfg = load_unet_config(
                path=unet_json,
                config_dict=unet_config,
                combined_json=combined_json,
            )
        except (ValueError, KeyError):
            unet_cfg = None

        if vae_config is not None:
            self.build_vae(vae_config, save_config=save_configs)
        if unet_cfg is not None:
            super().build_unet(unet_cfg, save_config=save_configs)

        return self

    def save_configs(self):
        self._ensure_dirs()
        if self.vae_config is not None:
            self._save_json(os.path.join(self._vae_dir(), "config.json"), self.vae_config)
        if self.unet_config is not None:
            save_unet_config(self.unet_config, os.path.join(self._unet_dir(), "config.json"))
        return self

    def add_vae(self, vae):
        self.vae = vae.to(self.device)
        return self

    # -------------------------
    # Loading / saving helpers (VAE)
    # -------------------------
    def load_vae_weights(self, path: str, *, strict: bool = True, map_location: str | None = None):
        assert hasattr(self, "vae"), "VAE not set. Build it first with .build_vae(...) or .build_from_configs(...)."
        map_location = map_location or self.device
        state = torch.load(path, map_location=map_location)
        missing, unexpected = self.vae.load_state_dict(state, strict=strict)

        if (not strict) or missing or unexpected:
            print(f"[load_vae_weights] strict={strict}")
            if missing:
                print("  Missing keys:", missing)
            if unexpected:
                print("  Unexpected keys:", unexpected)

        return self

    def load_pretrained(
        self,
        *,
        vae_path: str | None = None,
        unet_path: str | None = None,
        strict: bool = True,
        map_location: str | None = None,
        set_eval: bool = False,
    ):
        if vae_path is not None:
            self.load_vae_weights(vae_path, strict=strict, map_location=map_location)
        if unet_path is not None:
            self.load_unet_weights(unet_path, strict=strict, map_location=map_location)

        if set_eval:
            if hasattr(self, "vae"):
                self.vae.eval()
            if hasattr(self, "unet"):
                self.unet.eval()
        return self

    def save_vae_weights(self, path: None | str = None):
        assert hasattr(self, "vae"), "VAE not set."

        if path is None:
            path = os.path.join(self._vae_dir(), "vae_weights.pt")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.vae.state_dict(), path)
        return self

    # -------------------------
    # Training: VAE
    # -------------------------
    def train_vae(
        self,
        dataloader,
        epochs,
        eval_dataloader=None,
        log_dir: str = "./runs/autoencoder_kl",
        patience: int | None = None,
        min_delta: float = 0.0,
    ):
        assert hasattr(self, "vae"), "VAE not set. Build it first."
        if eval_dataloader is None:
            raise ValueError("eval_dataloader must be provided.")

        self._ensure_dirs()
        self.save_configs()

        optimizer = Adam(self.vae.parameters(), lr=1e-4)
        writer = SummaryWriter(log_dir)
        kl_weight = 1e-4
        max_grad_norm = 1.0
        logvar_clamp = (-30.0, 20.0)

        def _stretch_for_vis(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            # Per-image min/max stretch to improve contrast for TensorBoard only.
            x = self.from_norm_to_display(x).clamp(0, 1)
            flat = x.view(x.size(0), -1)
            mn = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            mx = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            return ((x - mn) / (mx - mn + eps)).clamp(0, 1)

        best_eval = float("inf")
        best_epoch = -1
        bad_epochs = 0

        for epoch in range(epochs):
            self.vae.train()

            train_total = train_mse = train_mae = train_kl = 0.0
            n_train = 0

            for x in tqdm(dataloader, desc=f"VAE Train {epoch+1}/{epochs}"):
                x = x.to(self.device)

                recon, mu, logvar = self.vae(x)
                recon_mse = F.mse_loss(recon, x)
                recon_mae = F.l1_loss(recon, x)

                logvar = logvar.clamp(*logvar_clamp)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_mse + kl_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_grad_norm)
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

            self.vae.eval()
            eval_total = eval_mse = eval_mae = eval_kl = 0.0
            n_eval = 0

            with torch.no_grad():
                for x in tqdm(eval_dataloader, desc=f"VAE Eval  {epoch+1}/{epochs}"):
                    x = x.to(self.device)

                    recon, mu, logvar = self.vae(x)
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
                f"[Epoch {epoch+1}/{epochs}] "
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

            # ---- Save epoch checkpoint to model_dir/VAE/ ----
            self.save_vae_weights(os.path.join(self._vae_dir(), f"vae_epoch_{epoch+1}.pt"))

            # Image logging
            with torch.no_grad():
                x_tr = next(iter(dataloader)).to(self.device)
                recon_tr, _, _ = self.vae(x_tr)
                x_tr_vis = _stretch_for_vis(x_tr[:4])
                recon_tr_vis = _stretch_for_vis(recon_tr[:4])
                writer.add_images("train/input", x_tr_vis, epoch)
                writer.add_images("train/reconstruction", recon_tr_vis, epoch)

                x_ev = next(iter(eval_dataloader)).to(self.device)
                recon_ev, _, _ = self.vae(x_ev)
                x_ev_vis = _stretch_for_vis(x_ev[:4])
                recon_ev_vis = _stretch_for_vis(recon_ev[:4])
                writer.add_images("eval/input", x_ev_vis, epoch)
                writer.add_images("eval/reconstruction", recon_ev_vis, epoch)

            # Early stopping + best save to model_dir/VAE/
            if patience is not None:
                improved = (best_eval - eval_total) > min_delta
                if improved:
                    best_eval = eval_total
                    best_epoch = epoch
                    bad_epochs = 0
                    self.save_vae_weights(os.path.join(self._vae_dir(), "vae_best.pt"))
                    print(f"  ✅ New best eval_total={best_eval:.6f} at epoch {epoch+1} -> saved VAE/vae_best.pt")
                else:
                    bad_epochs += 1
                    print(f"  ⏳ No improvement (best={best_eval:.6f}), bad_epochs={bad_epochs}/{patience}")
                    if bad_epochs >= patience:
                        print(f"🛑 Early stopping triggered. Best epoch: {best_epoch+1} (eval_total={best_eval:.6f})")
                        break

        writer.close()
        return self

    # -------------------------
    # Latent encoding/decoding overrides
    # -------------------------
    def freeze_vae(self):
        assert hasattr(self, "vae"), "VAE not set."
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        return self

    def encode_fm_input(self, x: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "vae"), "VAE not set."
        with torch.no_grad():
            z_mu, z_sigma = self.vae.encode(x)
            z = self.vae.sampling(z_mu, z_sigma)
        return z

    def decode_fm_output(self, z: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "vae"), "VAE not set."
        return self.vae.decode(z)

    @torch.no_grad()
    def sample_latents_euler(
        self,
        steps: int = 50,
        batch_size: int = 4,
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        return super().sample_euler(steps=steps, batch_size=batch_size, sample_shape=sample_shape)

    @torch.no_grad()
    def log_fm_samples_to_tensorboard(
        self,
        writer: SummaryWriter,
        epoch: int,
        steps: int = 50,
        batch_size: int = 4,
        tag: str = "fm_samples",
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ):
        assert hasattr(self, "vae"), "VAE not set."
        self.vae.eval()
        return super().log_fm_samples_to_tensorboard(
            writer=writer,
            epoch=epoch,
            steps=steps,
            batch_size=batch_size,
            tag=tag,
            sample_shape=sample_shape,
        )

    def log_fm_samples_to_tensorboard_guided(
        self,
        writer: SummaryWriter,
        epoch: int,
        guidance=None,              # Optional[ScorePredictorGuidance]
        guidance_scale: float = 1.0,
        steps: int = 50,
        batch_size: int = 4,
        tag_unguided: str = "fm_samples/unguided",
        tag_guided: str = "fm_samples/guided",
        tag_suffix: str = "",
        sample_shape: Optional[Tuple[int, int, int]] = None,
        log_score_scalars: bool = True,
    ):
        """Log both unguided and guided FM samples side-by-side to TensorBoard.

        Generates one unguided batch and one guided batch and logs them under
        separate tags.  Optionally logs mean predicted surprise / GMM scores
        as scalars so you can verify the guidance effect directionally.

        Parameters
        ----------
        writer : SummaryWriter
        epoch : int
        guidance : :class:`ScorePredictorGuidance`, optional
            If ``None``, only the unguided samples are logged.
        guidance_scale : float
        steps, batch_size, sample_shape : standard sampling args
        tag_unguided, tag_guided : str
            TensorBoard image tags.
        tag_suffix : str
            Optional suffix appended to both tags (useful for multi-run comparison).
        log_score_scalars : bool
            Log mean_surprise / mean_gmm scalars when *guidance* is provided.
        """
        assert hasattr(self, "vae"),  "VAE not set."
        assert hasattr(self, "unet"), "UNet not set."
        self.vae.eval()
        self.unet.eval()

        t_ug = (tag_unguided + tag_suffix) if tag_suffix else tag_unguided
        t_g  = (tag_guided   + tag_suffix) if tag_suffix else tag_guided

        # -- Unguided samples --
        z_plain = self.sample_euler_guided(
            steps=steps,
            batch_size=batch_size,
            sample_shape=sample_shape,
            guidance=None,
            guidance_scale=0.0,
        )
        with torch.no_grad():
            x_plain = self.decode_fm_output(z_plain)
        x_plain_vis = self.from_norm_to_display(x_plain).clamp(0, 1)
        writer.add_images(t_ug, x_plain_vis, epoch)

        if guidance is not None:
            # -- Guided samples --
            z_guided = self.sample_euler_guided(
                steps=steps,
                batch_size=batch_size,
                sample_shape=sample_shape,
                guidance=guidance,
                guidance_scale=guidance_scale,
            )
            with torch.no_grad():
                x_guided = self.decode_fm_output(z_guided)
            x_guided_vis = self.from_norm_to_display(x_guided).clamp(0, 1)
            writer.add_images(t_g, x_guided_vis, epoch)

            if log_score_scalars:
                scores_plain  = guidance.log_scores(z_plain)
                scores_guided = guidance.log_scores(z_guided)
                writer.add_scalars(
                    f"guided_scores{tag_suffix}/mean_surprise",
                    {"unguided": scores_plain["mean_surprise"],
                     "guided":   scores_guided["mean_surprise"]},
                    epoch,
                )
                writer.add_scalars(
                    f"guided_scores{tag_suffix}/mean_gmm",
                    {"unguided": scores_plain["mean_gmm"],
                     "guided":   scores_guided["mean_gmm"]},
                    epoch,
                )
                print(
                    f"[log_guided] epoch={epoch}  "
                    f"unguided: surprise={scores_plain['mean_surprise']:.4f}  "
                    f"gmm={scores_plain['mean_gmm']:.4f}  |  "
                    f"guided:   surprise={scores_guided['mean_surprise']:.4f}  "
                    f"gmm={scores_guided['mean_gmm']:.4f}"
                )

    # -------------------------
    # Training: Flow Matching (latent space)
    # -------------------------
    def train_flow_matching(
        self,
        dataloader,
        epochs,
        eval_dataloader=None,
        pretrained_vae_path: str | None = None,
        pretrained_unet_path: str | None = None,
        strict_load: bool = True,
        log_dir: str = "./runs/flow_matching",
        sample_every_epoch: bool = True,
        sample_steps: int = 50,
        sample_batch_size: int = 4,
        patience: int | None = None,
        min_delta: float = 0.0,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        save_every_n_epochs: int = 1,
        resume_from_checkpoint: str | None = None,
    ):
        assert hasattr(self, "vae"), "VAE not set. Build it first."
        assert hasattr(self, "unet"), "UNet not set. Build it first."

        if pretrained_vae_path is not None:
            self.load_vae_weights(pretrained_vae_path, strict=strict_load)
        if pretrained_unet_path is not None:
            self.load_unet_weights(pretrained_unet_path, strict=strict_load)

        self.freeze_vae()

        return super().train_flow_matching(
            dataloader=dataloader,
            epochs=epochs,
            eval_dataloader=eval_dataloader,
            pretrained_unet_path=None,
            strict_load=strict_load,
            log_dir=log_dir,
            sample_every_epoch=sample_every_epoch,
            sample_steps=sample_steps,
            sample_batch_size=sample_batch_size,
            patience=patience,
            min_delta=min_delta,
            sample_shape=sample_shape,
            save_every_n_epochs=save_every_n_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    # =========================================================================
    # Guided sampling methods (non-destructive additions; default sample_euler
    # is inherited unchanged from FlowMatchingPipeline).
    # =========================================================================

    # ------------------------------------------------------------------
    # 1. Euler + predictor guidance
    # ------------------------------------------------------------------

    def sample_euler_guided(
        self,
        steps: int = 50,
        batch_size: int = 4,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        guidance=None,       # Optional[ScorePredictorGuidance]
        guidance_scale: float = 1.0,
        return_logs: bool = False,
    ):
        """Euler ODE sampler with optional energy-based guidance.

        When *guidance* is ``None`` or *guidance_scale* == 0 the result is
        identical to :meth:`sample_euler` (the base Euler sampler).

        The UNet forward is always executed inside ``torch.no_grad()``.
        Gradients for the guidance signal are computed separately via
        :meth:`ScorePredictorGuidance.guidance_grad`, which creates a fresh
        leaf tensor internally and does not interfere with the velocity graph.

        Parameters
        ----------
        steps : int
        batch_size : int
        sample_shape : (C, H, W), optional
        guidance : :class:`ScorePredictorGuidance`, optional
        guidance_scale : float
            Global multiplier applied *on top of* the λ(t) schedule baked
            into the guidance object.  Set to 0 to disable at call-site.
        return_logs : bool
            If ``True``, also return a list of per-step dicts containing
            ``t``, ``grad_norm`` (mean over batch), and predicted scores.

        Returns
        -------
        z : (B, C, H, W) final latent
        logs : list of dicts  (only when *return_logs* is ``True``)
        """
        assert hasattr(self, "unet"), "UNet not set."
        assert hasattr(self, "vae"),  "VAE not set."
        self.unet.eval()
        self.vae.eval()

        shape = sample_shape or self._get_unet_sample_shape()
        z = torch.randn(batch_size, *shape, device=self.device)
        dt = 1.0 / steps
        logs = []

        for i in range(steps):
            t_val = i / steps
            t = torch.full((batch_size,), t_val, device=self.device)

            # -- UNet velocity (no grad needed here) --
            with torch.no_grad():
                unet_out = self.unet(z, t * self.t_scale).sample
                if self.train_target == "x0":
                    t_exp = t[:, None, None, None]
                    v = (unet_out - z) / (1.0 - t_exp).clamp(min=1e-5)
                else:
                    v = unet_out

            # -- Guidance gradient --
            if guidance is not None and guidance_scale > 0.0:
                g = guidance.guidance_grad(z, t=t_val, pipeline=self, velocity=v)
                guided_v = v + guidance_scale * g

                if return_logs:
                    grad_norm = g.view(batch_size, -1).norm(dim=1).mean().item()
                    step_log: dict = {"step": i, "t": t_val, "grad_norm": grad_norm}
                    with torch.no_grad():
                        scores = guidance.log_scores(z)
                    step_log.update(scores)
                    logs.append(step_log)
            else:
                guided_v = v

            z = (z + guided_v * dt).detach()

        if return_logs:
            return z, logs
        return z

    # ------------------------------------------------------------------
    # 2. Rejection / reranking over N candidates
    # ------------------------------------------------------------------

    def sample_euler_with_candidates(
        self,
        steps: int = 50,
        n_candidates: int = 8,
        keep_top_k: int = 1,
        batch_size: int = 4,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        guidance=None,       # Optional[ScorePredictorGuidance]
        guidance_scale: float = 0.0,
        return_all_scores: bool = False,
    ):
        """Generate *n_candidates* independent trajectories and keep the best.

        No guidance is applied to the ODE by default (``guidance_scale=0``);
        selection is done purely by the predictor energy at the end.

        Parameters
        ----------
        steps, batch_size, sample_shape :
            As in :meth:`sample_euler_guided`.
        n_candidates : int
            Number of candidate trajectories to generate per request.
        keep_top_k : int
            Number of best candidates (by *lowest* energy) to return.
        guidance : :class:`ScorePredictorGuidance`, optional
            If given, its energy function is used for scoring.  If ``None``
            and guidance_scale > 0, raises ValueError.
        guidance_scale : float
            Guidance scale during trajectory sampling (0 = unguided).
        return_all_scores : bool
            Also return the energy tensor for all candidates.

        Returns
        -------
        z_best : (keep_top_k * batch_size, C, H, W) or (batch_size, C, H, W)
            Selected latents (sorted ascending by energy).
        scores : tensor, optional  (when *return_all_scores* is ``True``)
        """
        if guidance_scale > 0.0 and guidance is None:
            raise ValueError("guidance must be provided when guidance_scale > 0.")

        shape = sample_shape or self._get_unet_sample_shape()
        all_z = []

        for _ in range(n_candidates):
            z_c = self.sample_euler_guided(
                steps=steps,
                batch_size=batch_size,
                sample_shape=shape,
                guidance=guidance,
                guidance_scale=guidance_scale,
                return_logs=False,
            )
            all_z.append(z_c)

        # Stack: (n_candidates, B, C, H, W) → (n_candidates * B, C, H, W)
        all_z_cat = torch.cat(all_z, dim=0)   # (n_candidates * B, C, H, W)

        if guidance is not None:
            with torch.no_grad():
                E = guidance.energy(all_z_cat)   # (n_candidates * B,)
        else:
            # No scoring possible – return first keep_top_k candidates.
            if return_all_scores:
                return all_z_cat[: keep_top_k * batch_size], None
            return all_z_cat[: keep_top_k * batch_size]

        # For each position in [0, B), pick the candidate with lowest energy.
        # Reshape: (n_candidates, B)
        E_mat = E.view(n_candidates, batch_size)
        z_mat = all_z_cat.view(n_candidates, batch_size, *shape)

        # Rank candidates per sample position.
        order = E_mat.argsort(dim=0)           # ascending energy = better sample

        kept_indices = order[:keep_top_k]      # (keep_top_k, B)
        # Gather selected latents: result shape (keep_top_k, B, C, H, W)
        z_best_list = []
        for k in range(keep_top_k):
            idx = kept_indices[k]              # (B,)
            z_best_list.append(
                z_mat[idx, torch.arange(batch_size)]   # (B, C, H, W)
            )
        z_best = torch.cat(z_best_list, dim=0)  # (keep_top_k * B, C, H, W)

        if return_all_scores:
            return z_best, E
        return z_best

    # ------------------------------------------------------------------
    # 3. Beam / branching sampling
    # ------------------------------------------------------------------

    def sample_euler_beam(
        self,
        steps: int = 50,
        batch_size: int = 4,
        beam_size: int = 4,
        branch_factor: int = 2,
        sigma_perturb: float = 0.05,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        guidance=None,       # Optional[ScorePredictorGuidance]
        guidance_scale: float = 0.0,
        return_all_beams: bool = False,
    ):
        """Beam sampling: maintain *beam_size* parallel latent trajectories.

        At each step each beam candidate is expanded into *branch_factor*
        perturbed copies, all are advanced one Euler step, the predictor
        scores them, and only the *beam_size* lowest-energy candidates survive.

        Memory note: the total number of latents processed per step is
        ``batch_size × beam_size × branch_factor``.  Keep these small.

        Parameters
        ----------
        steps, batch_size, sample_shape :
            Standard args.
        beam_size : int
            Number of surviving trajectories per sample position.
        branch_factor : int
            Number of (noise-perturbed) children expanded from each beam candidate.
        sigma_perturb : float
            Std-dev of Gaussian noise added to branch.  Set to 0 to disable branching.
        guidance : :class:`ScorePredictorGuidance`, optional
            Used for scoring and optionally velocity guidance.
        guidance_scale : float
            Guidance scale for the velocity field (0 = use predictor only for pruning).
        return_all_beams : bool
            If ``True``, return all surviving beams; if ``False``, return the
            single best (lowest-energy) beam per sample position.

        Returns
        -------
        z_out : (batch_size, C, H, W)  or  (beam_size * batch_size, C, H, W)
        """
        assert hasattr(self, "unet"), "UNet not set."
        assert hasattr(self, "vae"),  "VAE not set."
        self.unet.eval()
        self.vae.eval()

        shape = sample_shape or self._get_unet_sample_shape()
        C, H, W = shape
        dt = 1.0 / steps

        # Initialise beams: (beam_size, B, C, H, W)
        z_beams = torch.randn(beam_size, batch_size, C, H, W, device=self.device)

        for i in range(steps):
            t_val = i / steps
            t_scalar = torch.full((batch_size,), t_val, device=self.device)

            expanded_beams = []

            for b in range(beam_size):
                z_b = z_beams[b]                               # (B, C, H, W)

                for _ in range(branch_factor):
                    if sigma_perturb > 0.0:
                        z_branch = z_b + sigma_perturb * torch.randn_like(z_b)
                    else:
                        z_branch = z_b.clone()

                    with torch.no_grad():
                        unet_out = self.unet(z_branch, t_scalar * self.t_scale).sample
                        if self.train_target == "x0":
                            t_exp = t_scalar[:, None, None, None]
                            v = (unet_out - z_branch) / (1.0 - t_exp).clamp(min=1e-5)
                        else:
                            v = unet_out

                    if guidance is not None and guidance_scale > 0.0:
                        g = guidance.guidance_grad(z_branch, t=t_val, pipeline=self, velocity=v)
                        v = v + guidance_scale * g

                    z_next = (z_branch + v * dt).detach()
                    expanded_beams.append(z_next)

            # expanded_beams: list of (B, C, H, W), length = beam_size * branch_factor
            z_expanded = torch.stack(expanded_beams, dim=0)  # (K, B, C, H, W)
            K = z_expanded.shape[0]

            if guidance is not None:
                # Score all expanded candidates and prune to beam_size.
                z_flat = z_expanded.view(K * batch_size, C, H, W)
                with torch.no_grad():
                    E_flat = guidance.energy(z_flat)           # (K * B,)
                E_mat = E_flat.view(K, batch_size)             # (K, B)
                order = E_mat.argsort(dim=0)                   # ascending energy

                z_mat = z_expanded                             # (K, B, C, H, W)
                new_beams = []
                for b_new in range(min(beam_size, K)):
                    idx = order[b_new]                         # (B,)
                    new_beams.append(z_mat[idx, torch.arange(batch_size)])
                z_beams = torch.stack(new_beams, dim=0)        # (beam_size, B, C, H, W)
            else:
                # No scoring – just keep first beam_size candidates.
                z_beams = z_expanded[:beam_size]

        if return_all_beams:
            # (beam_size * B, C, H, W)
            return z_beams.view(beam_size * batch_size, C, H, W)

        # Return best beam per sample position (index 0 = lowest energy after pruning).
        if guidance is not None:
            z_flat = z_beams.view(beam_size * batch_size, C, H, W)
            with torch.no_grad():
                E_final = guidance.energy(z_flat).view(beam_size, batch_size)
            best_idx = E_final.argmin(dim=0)               # (B,)
            z_best = z_beams[best_idx, torch.arange(batch_size)]
        else:
            z_best = z_beams[0]

        return z_best

    # ------------------------------------------------------------------
    # 4. Post-sampling latent refinement
    # ------------------------------------------------------------------

    def refine_latents_energy(
        self,
        z: torch.Tensor,
        guidance,       # ScorePredictorGuidance
        num_steps: Optional[int] = None,
        step_size: Optional[float] = None,
        clamp_latent_norm: Optional[float] = None,
    ) -> torch.Tensor:
        """Refine latents via gradient descent / ascent on energy.

        Runs ``num_steps`` iterations of::

            z ← z + step_size · guidance_grad(z)

        The guidance object's ``sign`` field determines direction
        (``"minimize"`` → descend energy; ``"maximize"`` → ascend).

        Parameters
        ----------
        z : (B, C, H, W)
            Initial latents (e.g. output of :meth:`sample_euler_guided`).
        guidance : :class:`ScorePredictorGuidance`
        num_steps : int, optional
            Overrides ``guidance.config.num_refine_steps``.
        step_size : float, optional
            Overrides ``guidance.config.refine_step_size``.
        clamp_latent_norm : float, optional
            After each step, clamp per-sample latent ℓ₂ norm to this value.
            Helps prevent divergence.  ``None`` = no clamping.

        Returns
        -------
        z : (B, C, H, W) refined latent
        """
        cfg = guidance.config
        n_steps   = num_steps  if num_steps  is not None else cfg.num_refine_steps
        step_sz   = step_size  if step_size  is not None else cfg.refine_step_size
        B = z.shape[0]

        z = z.detach()
        for _ in range(n_steps):
            g = guidance.guidance_grad(z, t=1.0, pipeline=self)
            z = (z + step_sz * g).detach()

            if clamp_latent_norm is not None:
                norms = z.view(B, -1).norm(dim=1).view(B, 1, 1, 1).clamp(min=1e-8)
                scale = torch.where(
                    norms > clamp_latent_norm,
                    torch.full_like(norms, clamp_latent_norm) / norms,
                    torch.ones_like(norms),
                )
                z = (z * scale).detach()

        return z

    # -------------------------
    # Restore from folder (VAE + UNet)
    # -------------------------
    def load_from_pipeline_folder(
        self,
        pipeline_folder: str,
        *,
        vae_weights: str = "vae_best.pt",
        unet_weights: str = "unet_fm_best.pt",
        vae_config_name: str = "config.json",
        unet_config_name: str = "config.json",
        strict: bool = True,
        map_location: Optional[str] = None,
        set_eval: bool = True,
    ):
        self.model_dir = pipeline_folder
        map_location = map_location or self.device

        vae_dir = self._vae_dir()
        unet_dir = self._unet_dir()

        vae_cfg_path = os.path.join(vae_dir, vae_config_name)
        unet_cfg_path = os.path.join(unet_dir, unet_config_name)

        if not os.path.isfile(vae_cfg_path):
            raise FileNotFoundError(f"Missing VAE config: {vae_cfg_path}")
        if not os.path.isfile(unet_cfg_path):
            raise FileNotFoundError(f"Missing UNET config: {unet_cfg_path}")

        vae_cfg = self._load_json(vae_cfg_path)
        unet_cfg = self._load_json(unet_cfg_path)

        self.build_vae(vae_cfg, save_config=False)
        self.build_unet(unet_cfg, save_config=False)

        vae_w_path = os.path.join(vae_dir, vae_weights)
        unet_w_path = os.path.join(unet_dir, unet_weights)

        if not os.path.isfile(vae_w_path):
            raise FileNotFoundError(f"Missing VAE weights: {vae_w_path}")
        if not os.path.isfile(unet_w_path):
            raise FileNotFoundError(f"Missing UNET weights: {unet_w_path}")

        self.load_vae_weights(vae_w_path, strict=strict, map_location=map_location)
        self.load_unet_weights(unet_w_path, strict=strict, map_location=map_location)

        if set_eval:
            self.vae.eval()
            self.unet.eval()

        return self

    def load_from_pipeline_folder_auto(
        self,
        pipeline_folder: str,
        *,
        strict: bool = True,
        map_location: Optional[str] = None,
        set_eval: bool = True,
    ):
        self.model_dir = pipeline_folder
        map_location = map_location or self.device

        vae_dir = self._vae_dir()
        unet_dir = self._unet_dir()

        vae_cfg_path = os.path.join(vae_dir, "config.json")
        unet_cfg_path = os.path.join(unet_dir, "config.json")
        if not os.path.isfile(vae_cfg_path):
            raise FileNotFoundError(f"Missing VAE config: {vae_cfg_path}")
        if not os.path.isfile(unet_cfg_path):
            raise FileNotFoundError(f"Missing UNET config: {unet_cfg_path}")

        vae_cfg = self._load_json(vae_cfg_path)
        unet_cfg = self._load_json(unet_cfg_path)

        self.build_vae(vae_cfg, save_config=False)
        self.build_unet(unet_cfg, save_config=False)

        vae_best = os.path.join(vae_dir, "vae_best.pt")
        unet_best = os.path.join(unet_dir, "unet_fm_best.pt")

        vae_w_path = vae_best if os.path.isfile(vae_best) else self._pick_latest_by_prefix(vae_dir, "vae_epoch_")
        unet_w_path = unet_best if os.path.isfile(unet_best) else self._pick_latest_by_prefix(unet_dir, "unet_fm_epoch_")

        if vae_w_path is None or not os.path.isfile(vae_w_path):
            raise FileNotFoundError(f"No VAE weights found in {vae_dir} (expected vae_best.pt or vae_epoch_*.pt)")
        if unet_w_path is None or not os.path.isfile(unet_w_path):
            raise FileNotFoundError(f"No UNET weights found in {unet_dir} (expected unet_fm_best.pt or unet_fm_epoch_*.pt)")

        self.load_vae_weights(vae_w_path, strict=strict, map_location=map_location)
        self.load_unet_weights(unet_w_path, strict=strict, map_location=map_location)

        if set_eval:
            self.vae.eval()
            self.unet.eval()

        return self


# -------------------------
# Example usage
# -------------------------
"""
# Pixel-space (no VAE)
unet_cfg = dict(
    sample_size=64,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
)

pipe = FlowMatchingPipeline(device="cuda", t_scale=1000, model_dir="./my_pixel_run") \
    .build_from_configs(unet_config=unet_cfg)

# Stable (VAE + UNet)
vae_cfg = dict(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(32, 64, 128),
    num_res_blocks=2,
    attention_levels=(False, False, False),
    latent_channels=4,
)

unet_cfg = dict(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
)

stable_pipe = StableFlowMatchingPipeline(device="cuda", t_scale=1000, model_dir="./my_run_001") \
    .build_from_configs(vae_config=vae_cfg, unet_config=unet_cfg) \
    .load_pretrained(
        vae_path="./my_run_001/VAE/vae_best.pt",
        unet_path="./my_run_001/UNET/unet_fm_best.pt",
        set_eval=True,
    )
"""

# JSON options:
"""
# (A) separate json files
stable_pipe.build_from_configs(vae_json="vae_config.json", unet_json="unet_config.json")

# (B) single json file:
# {
#   "VAE": {...},
#   "UNET": {...}
# }
stable_pipe.build_from_configs(combined_json="model_config.json")
"""