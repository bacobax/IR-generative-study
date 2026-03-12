"""Meta-learning trainer for a single incremental condition episode.

Implements three phases:
  A. Base training on seen conditions
  B. Router-only adaptation on one new condition
  C. Adapter (and optional UNet parts) refinement with replay
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Optional, Union

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.algorithms.training.text_fm_trainer import TextFMTrainer
from src.core.registry import REGISTRIES
from src.models.moe_text_unet import TextMOEUNet


class MetaFMTrainer(TextFMTrainer):
    """Meta-learning trainer for one incremental condition episode.

    Phases:
      - Phase A: base training
      - Phase B: router-only adaptation on new condition
      - Phase C: adapter refinement with replay
    """

    # ------------------------------------------------------------------
    # Config-driven constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config,
        *,
        from_norm_to_display=None,
    ) -> "MetaFMTrainer":
        from src.models.vae import load_vae_config, build_vae_from_config
        from src.models.fm_text_unet import load_text_unet_config, build_text_fm_unet
        from src.conditioning.text_conditioner import TextConditioner

        device = "cuda" if torch.cuda.is_available() else "cpu"

        unet_cfg = load_text_unet_config(config.model.unet_config)
        if config.model.model_builder_name:
            builder = REGISTRIES.model_builder[config.model.model_builder_name]
            unet = builder(unet_cfg, device=device)
        else:
            unet = build_text_fm_unet(unet_cfg, device=device)

        vae, vae_cfg = None, None
        if config.model.vae_config:
            vae_cfg = load_vae_config(config.model.vae_config)
            vae = build_vae_from_config(vae_cfg, device=device)

        return_pooled = getattr(config.conditioning, "return_pooled", False)
        if config.model.model_builder_name == "text_moe_unet":
            return_pooled = True

        conditioner = TextConditioner(
            encoder_name=config.conditioning.text_encoder,
            max_length=config.conditioning.max_text_length,
            cond_drop_prob=config.conditioning.cond_drop_prob,
            return_pooled=return_pooled,
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

    # ------------------------------------------------------------------
    # Freezing utilities
    # ------------------------------------------------------------------
    def _set_trainable(self, module: torch.nn.Module, trainable: bool) -> None:
        for p in module.parameters():
            p.requires_grad = trainable

    def _moe_unet(self) -> TextMOEUNet:
        if not isinstance(self.unet, TextMOEUNet):
            raise TypeError("MetaFMTrainer requires TextMOEUNet")
        return self.unet

    def _freeze_all(self) -> None:
        self._set_trainable(self.unet, False)

    def _set_router_trainable(self, trainable: bool) -> None:
        self._set_trainable(self._moe_unet().router, trainable)

    def _set_adapter_trainable(self, trainable: bool) -> None:
        self._set_trainable(self._moe_unet().mid_adapter, trainable)

    def _set_unet_parts_trainable(self, policy: str) -> None:
        """Unfreeze selected UNet parts (policy: none|all|mid|up)."""
        unet = self._moe_unet().unet
        if policy == "none":
            return
        if policy == "all":
            self._set_trainable(unet, True)
            return
        if policy == "mid":
            self._set_trainable(unet.mid_block, True)
            return
        if policy == "up":
            self._set_trainable(unet.up_blocks, True)
            return
        raise ValueError(f"Unknown unfreeze policy: {policy!r}")

    # ------------------------------------------------------------------
    # Phase training helpers
    # ------------------------------------------------------------------
    def _loss_from_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        images = batch["pixel_values"].to(self.device)
        x_fm = self.encode_fm_input(images)
        cond_kw = self.conditioner.prepare_for_training(batch, self.device)
        return self.flow_matching_step(x_fm, cond_kw)

    def _build_optimizer(
        self,
        lr: float,
        *,
        router_lr_scale: Optional[float] = None,
    ) -> torch.optim.Optimizer:
        if router_lr_scale is None or router_lr_scale == 1.0:
            params = [p for p in self.unet.parameters() if p.requires_grad]
            return Adam(params, lr=lr)

        moe = self._moe_unet()
        router_params = [p for p in moe.router.parameters() if p.requires_grad]
        router_param_ids = {id(p) for p in router_params}
        other_params = [
            p for p in self.unet.parameters()
            if p.requires_grad and id(p) not in router_param_ids
        ]
        return Adam(
            [
                {"params": other_params, "lr": lr},
                {"params": router_params, "lr": lr * router_lr_scale},
            ]
        )

    def _train_phase(
        self,
        dataloader: DataLoader,
        *,
        epochs: int,
        lr: float,
        phase_name: str,
        replay_dataloader: Optional[DataLoader] = None,
        replay_every: int = 1,
        router_lr_scale: Optional[float] = None,
    ) -> None:
        optimizer = self._build_optimizer(lr, router_lr_scale=router_lr_scale)
        replay_iter = None
        if replay_dataloader is not None:
            replay_iter = itertools.cycle(replay_dataloader)

        for epoch in range(epochs):
            self.unet.train()
            total_loss = 0.0
            for step, batch in enumerate(tqdm(dataloader, desc=f"{phase_name} {epoch+1}/{epochs}")):
                loss = self._loss_from_batch(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if replay_iter is not None and replay_every > 0 and (step + 1) % replay_every == 0:
                    replay_batch = next(replay_iter)
                    replay_loss = self._loss_from_batch(replay_batch)
                    optimizer.zero_grad()
                    replay_loss.backward()
                    optimizer.step()

            avg = total_loss / max(1, len(dataloader))
            print(f"[{phase_name} epoch {epoch+1}] loss: {avg:.6f}")

    # ------------------------------------------------------------------
    # Public API: single-episode meta training
    # ------------------------------------------------------------------
    def train_single_episode(
        self,
        *,
        base_dataloader: DataLoader,
        new_dataloader: DataLoader,
        phase_a_epochs: int,
        phase_b_epochs: int,
        phase_c_epochs: int,
        phase_a_lr: float,
        phase_b_lr: float,
        phase_c_lr: float,
        phase_c_unfreeze_policy: str = "none",
        phase_c_router_trainable: bool = True,
        phase_c_router_lr_scale: float = 1.0,
        phase_c_replay_every: int = 1,
    ) -> None:
        """Run a single incremental episode (base -> router -> refine)."""
        # Phase A: base training
        self._set_trainable(self.unet, True)
        self._train_phase(
            base_dataloader,
            epochs=phase_a_epochs,
            lr=phase_a_lr,
            phase_name="Phase A (base)",
        )

        # Phase B: router-only on new condition
        self._freeze_all()
        self._set_router_trainable(True)
        self._train_phase(
            new_dataloader,
            epochs=phase_b_epochs,
            lr=phase_b_lr,
            phase_name="Phase B (router-only)",
        )

        # Phase C: adapters (+ optional UNet parts) with replay
        self._freeze_all()
        self._set_adapter_trainable(True)
        self._set_unet_parts_trainable(phase_c_unfreeze_policy)
        if phase_c_router_trainable:
            self._set_router_trainable(True)

        self._train_phase(
            new_dataloader,
            epochs=phase_c_epochs,
            lr=phase_c_lr,
            phase_name="Phase C (refine+replay)",
            replay_dataloader=base_dataloader,
            replay_every=phase_c_replay_every,
            router_lr_scale=phase_c_router_lr_scale if phase_c_router_trainable else None,
        )

    def train_from_config(
        self,
        config,
        *,
        base_dataloader: DataLoader,
        new_dataloader: DataLoader,
    ) -> None:
        """Run a single episode using MetaFMTrainConfig fields."""
        self.train_single_episode(
            base_dataloader=base_dataloader,
            new_dataloader=new_dataloader,
            phase_a_epochs=config.phase_a.epochs,
            phase_b_epochs=config.phase_b.epochs,
            phase_c_epochs=config.phase_c.epochs,
            phase_a_lr=config.phase_a.lr,
            phase_b_lr=config.phase_b.lr,
            phase_c_lr=config.phase_c.lr,
            phase_c_unfreeze_policy=config.phase_c.unfreeze_unet_policy,
            phase_c_router_trainable=config.phase_c.router_trainable,
            phase_c_router_lr_scale=config.phase_c.router_lr_scale,
            phase_c_replay_every=config.phase_c.replay_every,
        )


# ── registry ──────────────────────────────────────────────────────────────
REGISTRIES.trainer.register("meta_fm")(MetaFMTrainer)
