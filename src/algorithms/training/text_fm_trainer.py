"""Trainer for text-conditioned flow-matching with CFG conditioning dropout.

Inherits from :class:`FlowMatchingTrainer` and overrides the training
loop to handle dict-style batches containing both images and text.
The text conditioner's :attr:`cond_drop_prob` controls the probability
of replacing text embeddings with null embeddings so the same network
learns both conditional and unconditional velocity prediction.

Everything that does not need changing (FM loss computation, weight I/O,
checkpoint logic, VAE encoding) is inherited from the parent.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.algorithms.training.flow_matching_trainer import FlowMatchingTrainer
from src.conditioning.text_conditioner import TextConditioner
from src.models.fm_text_unet import load_text_unet_config, build_text_fm_unet, save_text_unet_config


def _default_from_norm_to_display(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


class TextFMTrainer(FlowMatchingTrainer):
    """Flow-matching trainer for text-conditioned models with CFG.

    Parameters
    ----------
    unet : UNet2DConditionModel
        Text-conditioned UNet with cross-attention layers.
    conditioner : TextConditioner
        CLIP text conditioner (handles tokenisation, encoding, and
        conditioning dropout).
    device, t_scale, train_target, model_dir, from_norm_to_display,
    unet_config, vae, vae_config :
        Forwarded to :class:`FlowMatchingTrainer`.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        *,
        conditioner: TextConditioner,
        device: Optional[Union[str, torch.device]] = None,
        t_scale: float = 1.0,
        train_target: str = "v",
        model_dir: str = "./artifacts/checkpoints/flow_matching/text_fm/",
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        unet_config: Optional[Dict[str, Any]] = None,
        vae=None,
        vae_config: Optional[Dict[str, Any]] = None,
    ):
        # Parent constructor — conditioner passed through
        super().__init__(
            unet,
            device=device,
            t_scale=t_scale,
            train_target=train_target,
            model_dir=model_dir,
            from_norm_to_display=from_norm_to_display,
            unet_config=unet_config,
            vae=vae,
            vae_config=vae_config,
            conditioner=conditioner,
        )
        # Move conditioner to the training device
        self.conditioner: TextConditioner
        self.conditioner.to(self.device)
        self._attn_vis_config = None  # set by train_from_config if enabled

    # ------------------------------------------------------------------
    # Config-driven constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config,
        *,
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> "TextFMTrainer":
        """Build a TextFMTrainer from a :class:`TextFMTrainConfig`."""
        from src.models.vae import load_vae_config, build_vae_from_config

        device = config.resolved_device() if hasattr(config, "resolved_device") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        unet_cfg = load_text_unet_config(config.model.unet_config)
        unet = build_text_fm_unet(unet_cfg, device=device)

        vae, vae_cfg = None, None
        if config.model.vae_config:
            vae_cfg = load_vae_config(config.model.vae_config)
            vae = build_vae_from_config(vae_cfg, device=device)

        conditioner = TextConditioner(
            encoder_name=config.conditioning.text_encoder,
            max_length=config.conditioning.max_text_length,
            cond_drop_prob=config.conditioning.cond_drop_prob,
            device=device,
        )

        return cls(
            unet,
            conditioner=conditioner,
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
        """Launch training driven by a :class:`TextFMTrainConfig`."""
        # Store attention visualization config if present
        if hasattr(config, "attention_vis"):
            self._attn_vis_config = config.attention_vis
        # Store count filter info for metadata persistence
        self._count_filter_config = getattr(config, "count_filter", None)
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
            lr=config.training.lr,
        )

    # ------------------------------------------------------------------
    # Override: save text-unet config (different format)
    # ------------------------------------------------------------------
    def _save_configs(self) -> None:
        if self.unet_config is not None:
            save_text_unet_config(
                self.unet_config,
                os.path.join(self._unet_dir(), "config.json"),
            )
        if self.vae_config is not None:
            import json
            os.makedirs(self._vae_dir(), exist_ok=True)
            path = os.path.join(self._vae_dir(), "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.vae_config, f, indent=2, sort_keys=True)
        # Save conditioner config for reproducible loading at inference
        import json
        cond_path = os.path.join(self.model_dir, "conditioner.json")
        os.makedirs(os.path.dirname(cond_path) or ".", exist_ok=True)
        with open(cond_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "encoder_name": self.conditioner.encoder_name,
                    "max_length": self.conditioner.max_length,
                    "embedding_dim": self.conditioner.embedding_dim,
                },
                f,
                indent=2,
            )
        # Persist count filter metadata
        cf = getattr(self, "_count_filter_config", None)
        if cf is not None:
            seen = getattr(cf, "seen_counts", None)
            unseen = getattr(cf, "unseen_counts", None)
            if seen is not None or unseen is not None:
                cf_path = os.path.join(self.model_dir, "count_filter.json")
                with open(cf_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "seen_counts": list(seen) if seen is not None else None,
                            "unseen_counts": list(unseen) if unseen is not None else None,
                            "max_crop_retries": getattr(cf, "max_crop_retries", 5),
                        },
                        f,
                        indent=2,
                    )

    # ------------------------------------------------------------------
    # Override: build a sampler that passes null text for TensorBoard vis
    # ------------------------------------------------------------------
    def _make_sampler(self):
        from src.algorithms.inference.cfg_flow_matching_sampler import (
            CFGFlowMatchingSampler,
        )

        if self.vae is not None:
            return CFGFlowMatchingSampler.from_stable(
                self.unet,
                self.vae,
                conditioner=self.conditioner,
                device=self.device,
                t_scale=self.t_scale,
                train_target=self.train_target,
                from_norm_to_display=self.from_norm_to_display,
            )
        return CFGFlowMatchingSampler(
            self.unet,
            conditioner=self.conditioner,
            device=self.device,
            t_scale=self.t_scale,
            train_target=self.train_target,
            from_norm_to_display=self.from_norm_to_display,
        )

    # ------------------------------------------------------------------
    # Override: main training loop to handle dict batches
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
        log_dir: str = "./artifacts/runs/main/text_fm",
        sample_every: int = 1,
        sample_steps: int = 50,
        sample_batch_size: int = 4,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        save_every_n_epochs: int = 1,
        resume_from_checkpoint: Optional[str] = None,
        lr: float = 1e-4,
    ) -> None:
        if patience is not None and eval_dataloader is None:
            raise ValueError("eval_dataloader required when using patience.")

        self._ensure_dirs()
        self._save_configs()

        # Pre-load weights ------------------------------------------------
        if pretrained_vae_path is not None and self.vae is not None:
            self.load_vae_weights(pretrained_vae_path, strict=strict_load)
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

        if pretrained_unet_path is not None:
            self.load_unet_weights(pretrained_unet_path, strict=strict_load)

        optimizer = Adam(self.unet.parameters(), lr=lr)

        # Resume state -----------------------------------------------------
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
            print(f"[Resume] epoch {start_epoch}, step={global_step}, best_eval={best_eval:.6f}")

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
            # Embed count filter metadata in checkpoint
            cf = getattr(self, "_count_filter_config", None)
            if cf is not None:
                seen = getattr(cf, "seen_counts", None)
                unseen = getattr(cf, "unseen_counts", None)
                if seen is not None or unseen is not None:
                    ckpt["count_filter"] = {
                        "seen_counts": list(seen) if seen is not None else None,
                        "unseen_counts": list(unseen) if unseen is not None else None,
                    }
            if torch.cuda.is_available():
                ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            torch.save(ckpt, path)

        def _set_epoch(dl: Optional[DataLoader], epoch_idx: int) -> None:
            if dl is None:
                return
            ds = getattr(dl, "dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch_idx)
            if ds is not None and hasattr(ds, "transform") and hasattr(ds.transform, "set_epoch"):
                ds.transform.set_epoch(epoch_idx)

        sampler_obj = self._make_sampler() if sample_every > 0 else None

        # ── Main loop ─────────────────────────────────────────────────────
        last_prompts: list = []  # collect prompts from training for vis

        for epoch in range(start_epoch, epochs):
            _set_epoch(dataloader, epoch)
            _set_epoch(eval_dataloader, epoch)
            self.unet.train()
            total_loss = 0.0
            n_cond = 0
            n_drop = 0
            idx=0
            for batch in tqdm(dataloader, desc=f"TextFM Epoch {epoch+1}/{epochs}"):
                # batch is a dict: {"pixel_values": Tensor, "text": list[str]}
                images = batch["pixel_values"].to(self.device)
                x_fm = self.encode_fm_input(images)
                if idx == 0:
                    print(f"[Debug] texts: {batch['text'][:5]} ...")
                cond_kw = self.conditioner.prepare_for_training(batch, self.device)

                loss = self.flow_matching_step(x_fm, cond_kw)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                writer.add_scalar("text_fm/loss_step", loss.item(), global_step)
                global_step += 1

                # Keep prompts from this batch for TensorBoard vis
                last_prompts = batch["text"]
                idx+=1

            avg_loss = total_loss / max(1, len(dataloader))
            print(f"[TextFM Epoch {epoch+1}] loss: {avg_loss:.6f}")
            writer.add_scalar("text_fm/loss_epoch", avg_loss, epoch)

            # Checkpoint
            if (save_every_n_epochs is not None) and ((epoch + 1) % save_every_n_epochs == 0):
                self.save_unet_weights(
                    os.path.join(self._unet_dir(), f"unet_fm_epoch_{epoch+1}.pt"),
                )
                _save_checkpoint(
                    os.path.join(self._unet_dir(), f"unet_fm_epoch_{epoch+1}_ckpt.pt"),
                    epoch_idx=epoch,
                )

            # Eval + early stopping
            if patience is not None and eval_dataloader is not None:
                self.unet.eval()
                eval_loss = 0.0
                n_eval = 0
                with torch.no_grad():
                    for batch in tqdm(eval_dataloader, desc=f"TextFM Eval {epoch+1}/{epochs}"):
                        images = batch["pixel_values"].to(self.device)
                        x_fm = self.encode_fm_input(images)
                        cond_kw = self.conditioner.prepare_for_training(batch, self.device)
                        loss = self.flow_matching_step(x_fm, cond_kw)
                        bs = images.size(0)
                        eval_loss += loss.item() * bs
                        n_eval += bs

                avg_eval = eval_loss / max(1, n_eval)
                print(f"  [Eval loss: {avg_eval:.6f}]")
                writer.add_scalar("text_fm/eval_loss_epoch", avg_eval, epoch)

                improved = (best_eval - avg_eval) > min_delta
                if improved:
                    best_eval = avg_eval
                    best_epoch = epoch
                    bad_epochs = 0
                    self.save_unet_weights(
                        os.path.join(self._unet_dir(), "unet_fm_best.pt"),
                    )
                    print(f"  New best eval={best_eval:.6f} at epoch {epoch+1}")
                else:
                    bad_epochs += 1
                    print(f"  No improvement (best={best_eval:.6f}), bad={bad_epochs}/{patience}")
                    if bad_epochs >= patience:
                        print(f"Early stopping. Best epoch: {best_epoch+1}")
                        break

            # Sample visualisation with CFG using real training prompts
            if sampler_obj is not None and (epoch + 1) % sample_every == 0:
                vis_prompts = last_prompts[:sample_batch_size]
                if not vis_prompts:
                    vis_prompts = [""] * sample_batch_size
                elif len(vis_prompts) < sample_batch_size:
                    vis_prompts = vis_prompts + vis_prompts[:sample_batch_size - len(vis_prompts)]
                sampler_obj.log_samples_to_tensorboard_cfg(
                    writer=writer,
                    epoch=epoch,
                    prompts=vis_prompts,
                    steps=sample_steps,
                    guidance_scale=7.5,
                    tag="text_fm/cfg_generated",
                    sample_shape=sample_shape,
                )

                # Cross-attention heatmap visualization
                if self._attn_vis_config is not None and self._attn_vis_config.enabled:
                    from src.analysis.cross_attention_maps import AttentionExtractionConfig
                    avc = self._attn_vis_config
                    attn_cfg = AttentionExtractionConfig(
                        target_tokens=avc.target_tokens,
                        num_vis_steps=avc.num_vis_steps,
                        layer_filter=avc.layer_filter,
                        head_reduction=avc.head_reduction,
                        overlay=avc.overlay,
                        colormap=avc.colormap,
                    )
                    sampler_obj.log_attention_heatmaps_to_tensorboard(
                        writer=writer,
                        epoch=epoch,
                        prompts=vis_prompts,
                        steps=sample_steps,
                        guidance_scale=avc.vis_guidance_scale,
                        attn_config=attn_cfg,
                        tag_prefix="cross_attn",
                        per_layer=avc.per_layer,
                        sample_shape=sample_shape,
                    )

        writer.close()


# ── registry ──────────────────────────────────────────────────────────────
from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.trainer.register("text_fm_cfg")(TextFMTrainer)
