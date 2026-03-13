"""Meta-learning trainer for a single incremental condition episode.

Implements three phases:
  A. Base training on seen conditions
  B. Router-only adaptation on one new condition
  C. Adapter (and optional UNet parts) refinement with replay
"""

from __future__ import annotations

import itertools
import json
import os
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.algorithms.training.text_fm_trainer import TextFMTrainer
from src.core.data.annotations import caption_from_count
from src.core.normalization import fm_output_to_uint16, uint16_to_png_uint8
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
    def __init__(
        self,
        unet,
        *,
        conditioner,
        device: Optional[Union[str, torch.device]] = None,
        t_scale: float = 1.0,
        train_target: str = "v",
        model_dir: str = "./artifacts/checkpoints/flow_matching/meta_fm/",
        from_norm_to_display=None,
        unet_config: Optional[Dict[str, Any]] = None,
        vae=None,
        vae_config: Optional[Dict[str, Any]] = None,
        router_sparsity_weight: float = 0.0,
        router_smoothness_weight: float = 0.0,
        log_dir: Optional[str] = None,
        log_every_steps: int = 10,
    ) -> None:
        super().__init__(
            unet,
            conditioner=conditioner,
            device=device,
            t_scale=t_scale,
            train_target=train_target,
            model_dir=model_dir,
            from_norm_to_display=from_norm_to_display,
            unet_config=unet_config,
            vae=vae,
            vae_config=vae_config,
        )
        self.router_sparsity_weight = router_sparsity_weight
        self.router_smoothness_weight = router_smoothness_weight
        self._meta_log_path = os.path.join(self.model_dir, "meta_fm_log.jsonl")
        self._tb_log_dir = log_dir
        self._tb_writer: Optional[SummaryWriter] = None
        self._global_step = 0
        self._log_every_steps = max(1, int(log_every_steps))

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
            router_sparsity_weight=config.router_reg.sparsity_weight,
            router_smoothness_weight=config.router_reg.smoothness_weight,
            log_dir=config.output.resolved_log_dir(),
        )

    # ------------------------------------------------------------------
    # TensorBoard logging helpers
    # ------------------------------------------------------------------
    def _ensure_writer(self) -> Optional[SummaryWriter]:
        if self._tb_writer is None and self._tb_log_dir is not None:
            os.makedirs(self._tb_log_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(self._tb_log_dir)
        return self._tb_writer

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        writer = self._ensure_writer()
        if writer is not None:
            writer.add_scalar(tag, value, step)

    def _log_hist(self, tag: str, values: torch.Tensor, step: int) -> None:
        writer = self._ensure_writer()
        if writer is not None:
            writer.add_histogram(tag, values, step)

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.unet.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total += param_norm.item() ** 2
        return math.sqrt(total)

    def _router_stats(self, weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        eps = 1e-8
        mean = weights.mean(dim=0)
        std = weights.std(dim=0, unbiased=False)
        entropy = -(weights * (weights + eps).log()).sum(dim=1).mean()
        max_w = weights.max(dim=1).values.mean()
        top1 = weights.max(dim=1).values
        sparsity = 1.0 - weights.pow(2).sum(dim=1).mean()
        return {
            "mean": mean,
            "std": std,
            "entropy": entropy,
            "max": max_w,
            "top1": top1.mean(),
            "sparsity": sparsity,
            "batch_var": weights.var(dim=0, unbiased=False),
        }

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
    def _loss_components(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        images = batch["pixel_values"].to(self.device)
        x_fm = self.encode_fm_input(images)
        cond_kw = self.conditioner.prepare_for_training(batch, self.device)
        fm_loss = self.flow_matching_step(x_fm, cond_kw)

        pooled = cond_kw.get("pooled_text_embeds")
        weights = None
        sparsity = torch.tensor(0.0, device=fm_loss.device)
        smooth = torch.tensor(0.0, device=fm_loss.device)

        if pooled is not None:
            weights = self._moe_unet().compute_router_weights(pooled)
            if self.router_sparsity_weight > 0:
                sparsity = 1.0 - weights.pow(2).sum(dim=1).mean()
            if self.router_smoothness_weight > 0 and weights.shape[0] > 1:
                smooth = (weights[1:] - weights[:-1]).pow(2).mean()

        total = fm_loss
        if self.router_sparsity_weight > 0:
            total = total + self.router_sparsity_weight * sparsity
        if self.router_smoothness_weight > 0:
            total = total + self.router_smoothness_weight * smooth

        return {
            "total": total,
            "fm": fm_loss,
            "sparsity": sparsity,
            "smooth": smooth,
            "weights": weights,
        }

    def _loss_from_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self._loss_components(batch)["total"]

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
        phase_tag: str,
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
            total_fm = 0.0
            total_sparsity = 0.0
            total_smooth = 0.0
            total_replay = 0.0
            replay_steps = 0
            for step, batch in enumerate(tqdm(dataloader, desc=f"{phase_name} {epoch+1}/{epochs}")):
                losses = self._loss_components(batch)
                loss = losses["total"]
                optimizer.zero_grad()
                loss.backward()
                grad_norm = self._grad_norm()
                optimizer.step()

                total_loss += loss.item()
                total_fm += losses["fm"].item()
                total_sparsity += losses["sparsity"].item()
                total_smooth += losses["smooth"].item()

                if (step + 1) % self._log_every_steps == 0:
                    self._log_scalar(f"{phase_tag}/loss/total_step", loss.item(), self._global_step)
                    self._log_scalar(f"{phase_tag}/loss/fm_step", losses["fm"].item(), self._global_step)
                    if self.router_sparsity_weight > 0:
                        self._log_scalar(f"{phase_tag}/loss/sparsity_step", losses["sparsity"].item(), self._global_step)
                    if self.router_smoothness_weight > 0:
                        self._log_scalar(f"{phase_tag}/loss/smooth_step", losses["smooth"].item(), self._global_step)
                    self._log_scalar(f"{phase_tag}/grad_norm", grad_norm, self._global_step)
                    for gi, group in enumerate(optimizer.param_groups):
                        self._log_scalar(f"{phase_tag}/lr/group_{gi}", group.get("lr", 0.0), self._global_step)

                    weights = losses.get("weights")
                    if weights is not None:
                        stats = self._router_stats(weights.detach())
                        self._log_scalar(f"{phase_tag}/router/entropy", stats["entropy"].item(), self._global_step)
                        self._log_scalar(f"{phase_tag}/router/max_weight", stats["max"].item(), self._global_step)
                        self._log_scalar(f"{phase_tag}/router/top1_mean", stats["top1"].item(), self._global_step)
                        self._log_scalar(f"{phase_tag}/router/sparsity", stats["sparsity"].item(), self._global_step)
                        self._log_hist(f"{phase_tag}/router/weights", weights.detach().cpu(), self._global_step)
                        for k in range(stats["mean"].numel()):
                            self._log_scalar(
                                f"{phase_tag}/router/mean_expert_{k}",
                                stats["mean"][k].item(),
                                self._global_step,
                            )
                            self._log_scalar(
                                f"{phase_tag}/router/std_expert_{k}",
                                stats["std"][k].item(),
                                self._global_step,
                            )
                            self._log_scalar(
                                f"{phase_tag}/router/batch_var_expert_{k}",
                                stats["batch_var"][k].item(),
                                self._global_step,
                            )

                self._global_step += 1

                if replay_iter is not None and replay_every > 0 and (step + 1) % replay_every == 0:
                    replay_batch = next(replay_iter)
                    replay_losses = self._loss_components(replay_batch)
                    replay_loss = replay_losses["total"]
                    optimizer.zero_grad()
                    replay_loss.backward()
                    replay_grad_norm = self._grad_norm()
                    optimizer.step()
                    replay_steps += 1
                    total_replay += replay_loss.item()

                    if (step + 1) % self._log_every_steps == 0:
                        self._log_scalar(f"{phase_tag}/replay/total_step", replay_loss.item(), self._global_step)
                        self._log_scalar(f"{phase_tag}/replay/fm_step", replay_losses["fm"].item(), self._global_step)
                        self._log_scalar(f"{phase_tag}/replay/grad_norm", replay_grad_norm, self._global_step)

            avg = total_loss / max(1, len(dataloader))
            avg_fm = total_fm / max(1, len(dataloader))
            avg_sparsity = total_sparsity / max(1, len(dataloader))
            avg_smooth = total_smooth / max(1, len(dataloader))
            avg_replay = total_replay / max(1, replay_steps) if replay_steps else 0.0
            replay_ratio = replay_steps / max(1, (len(dataloader) + replay_steps))
            print(f"[{phase_name} epoch {epoch+1}] loss: {avg:.6f}")

            self._log_scalar(f"{phase_tag}/loss/total_epoch", avg, epoch)
            self._log_scalar(f"{phase_tag}/loss/fm_epoch", avg_fm, epoch)
            if self.router_sparsity_weight > 0:
                self._log_scalar(f"{phase_tag}/loss/sparsity_epoch", avg_sparsity, epoch)
            if self.router_smoothness_weight > 0:
                self._log_scalar(f"{phase_tag}/loss/smooth_epoch", avg_smooth, epoch)
            if replay_steps:
                self._log_scalar(f"{phase_tag}/replay/total_epoch", avg_replay, epoch)
                self._log_scalar(f"{phase_tag}/replay/ratio", replay_ratio, epoch)

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
        self._ensure_writer()
        # Phase A: base training
        self._set_trainable(self.unet, True)
        trainable = self._trainable_summary()
        self._log_scalar("phase_a/trainable/router", float(trainable["router"]), self._global_step)
        self._log_scalar("phase_a/trainable/adapters", float(trainable["adapters"]), self._global_step)
        self._log_scalar("phase_a/trainable/unet", float(trainable["unet"]), self._global_step)
        self._train_phase(
            base_dataloader,
            epochs=phase_a_epochs,
            lr=phase_a_lr,
            phase_name="Phase A (base)",
            phase_tag="phase_a",
        )

        # Phase B: router-only on new condition
        self._freeze_all()
        self._set_router_trainable(True)
        trainable = self._trainable_summary()
        self._log_scalar("phase_b/trainable/router", float(trainable["router"]), self._global_step)
        self._log_scalar("phase_b/trainable/adapters", float(trainable["adapters"]), self._global_step)
        self._log_scalar("phase_b/trainable/unet", float(trainable["unet"]), self._global_step)
        self._train_phase(
            new_dataloader,
            epochs=phase_b_epochs,
            lr=phase_b_lr,
            phase_name="Phase B (router-only)",
            phase_tag="phase_b",
        )

        # Phase C: adapters (+ optional UNet parts) with replay
        self._freeze_all()
        self._set_adapter_trainable(True)
        self._set_unet_parts_trainable(phase_c_unfreeze_policy)
        if phase_c_router_trainable:
            self._set_router_trainable(True)
        trainable = self._trainable_summary()
        self._log_scalar("phase_c/trainable/router", float(trainable["router"]), self._global_step)
        self._log_scalar("phase_c/trainable/adapters", float(trainable["adapters"]), self._global_step)
        self._log_scalar("phase_c/trainable/unet", float(trainable["unet"]), self._global_step)

        self._train_phase(
            new_dataloader,
            epochs=phase_c_epochs,
            lr=phase_c_lr,
            phase_name="Phase C (refine+replay)",
            phase_tag="phase_c",
            replay_dataloader=base_dataloader,
            replay_every=phase_c_replay_every,
            router_lr_scale=phase_c_router_lr_scale if phase_c_router_trainable else None,
        )

        if self._tb_writer is not None:
            self._tb_writer.close()

    # ------------------------------------------------------------------
    # Full curriculum: base -> incremental -> final eval
    # ------------------------------------------------------------------
    def _log_stage(self, stage: str, condition: Optional[int] = None) -> None:
        label = f"{stage}" if condition is None else f"{stage} (cond={condition})"
        print(f"\n[MetaFM] === {label} ===")

        trainable = self._trainable_summary()
        self._append_log_event({
            "event": "stage_start",
            "stage": stage,
            "condition": condition,
            "trainable": trainable,
        })

        phase_tag = "stage" if condition is None else f"stage/cond_{condition}"
        self._log_scalar(f"{phase_tag}/trainable/router", float(trainable["router"]), self._global_step)
        self._log_scalar(f"{phase_tag}/trainable/adapters", float(trainable["adapters"]), self._global_step)
        self._log_scalar(f"{phase_tag}/trainable/unet", float(trainable["unet"]), self._global_step)

    def _append_log_event(self, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self._meta_log_path) or ".", exist_ok=True)
        with open(self._meta_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _trainable_summary(self) -> Dict[str, bool]:
        moe = self._moe_unet()

        def _any_trainable(module: torch.nn.Module) -> bool:
            return any(p.requires_grad for p in module.parameters())

        return {
            "router": _any_trainable(moe.router),
            "adapters": _any_trainable(moe.mid_adapter),
            "unet": _any_trainable(moe.unet),
        }

    def _save_stage_checkpoint(self, tag: str) -> None:
        filename = f"unet_{tag}.pt"
        path = os.path.join(self._unet_dir(), filename)
        self.save_unet_weights(path)

    def _save_router_weights(
        self,
        *,
        conditions: List[int],
        output_path: str,
    ) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        prompts = [caption_from_count(c) for c in conditions]
        _, pooled = self.conditioner.encode_text_with_pooler(prompts, self.device)
        weights = self._moe_unet().compute_router_weights(pooled).detach().cpu().tolist()
        payload = {
            "conditions": conditions,
            "prompts": prompts,
            "weights": weights,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        writer = self._ensure_writer()
        if writer is not None:
            w = torch.tensor(weights)
            writer.add_histogram("router/snapshot_weights", w, self._global_step)
            for i, cond in enumerate(conditions):
                writer.add_histogram(f"router/cond_{cond}/weights", w[i], self._global_step)

    @torch.no_grad()
    def evaluate_conditions(
        self,
        *,
        conditions: List[int],
        output_dir: str,
        steps: int = 50,
        guidance_scale: float = 7.5,
        samples_per_condition: int = 4,
    ) -> None:
        if not conditions:
            print("[MetaFM] No test conditions provided; skipping evaluation.")
            return

        os.makedirs(output_dir, exist_ok=True)
        sampler = self._make_sampler()
        writer = self._ensure_writer()

        for cond in conditions:
            prompt = caption_from_count(cond)
            cond_dir = os.path.join(output_dir, f"cond_{cond}")
            os.makedirs(cond_dir, exist_ok=True)

            prompts = [prompt] * samples_per_condition
            z = sampler.sample_euler_cfg(
                prompts,
                steps=steps,
                guidance_scale=guidance_scale,
            )
            x_gen = sampler.decode(z)

            if writer is not None:
                disp = self.from_norm_to_display(x_gen)
                writer.add_images(
                    f"eval/cond_{cond}/samples",
                    disp,
                    self._global_step,
                )
                _, pooled = self.conditioner.encode_text_with_pooler(prompts, self.device)
                weights = self._moe_unet().compute_router_weights(pooled)
                stats = self._router_stats(weights)
                self._log_scalar(f"eval/cond_{cond}/router/entropy", stats["entropy"].item(), self._global_step)
                self._log_scalar(f"eval/cond_{cond}/router/max_weight", stats["max"].item(), self._global_step)

            for i in range(samples_per_condition):
                raw_uint16 = fm_output_to_uint16(x_gen[i])
                npy_path = os.path.join(cond_dir, f"sample_{i:03d}.npy")
                png_path = os.path.join(cond_dir, f"sample_{i:03d}.png")
                np.save(npy_path, raw_uint16)
                png = uint16_to_png_uint8(raw_uint16)
                from PIL import Image
                Image.fromarray(png, mode="L").save(png_path)

    def train_curriculum(
        self,
        *,
        base_dataloader: DataLoader,
        incremental_loaders: List[Tuple[int, DataLoader]],
        test_conditions: List[int],
        phase_a_epochs: int,
        phase_b_epochs: int,
        phase_c_epochs: int,
        phase_a_lr: float,
        phase_b_lr: float,
        phase_c_lr: float,
        phase_c_unfreeze_policy: str,
        phase_c_router_trainable: bool,
        phase_c_router_lr_scale: float,
        phase_c_replay_every: int,
        log_router_weights: bool = True,
        router_weights_dir: Optional[str] = None,
        eval_output_dir: Optional[str] = None,
        eval_steps: int = 50,
        eval_guidance_scale: float = 7.5,
        eval_samples_per_condition: int = 4,
    ) -> None:
        self._ensure_writer()
        # Stage 1: base training
        self._log_stage("Stage 1: base training")
        self._set_trainable(self.unet, True)
        self._append_log_event({
            "event": "phase_start",
            "phase": "A",
            "condition": None,
            "trainable": self._trainable_summary(),
        })
        trainable = self._trainable_summary()
        self._log_scalar("phase_a/trainable/router", float(trainable["router"]), self._global_step)
        self._log_scalar("phase_a/trainable/adapters", float(trainable["adapters"]), self._global_step)
        self._log_scalar("phase_a/trainable/unet", float(trainable["unet"]), self._global_step)
        self._train_phase(
            base_dataloader,
            epochs=phase_a_epochs,
            lr=phase_a_lr,
            phase_name="Stage 1 (base)",
            phase_tag="phase_a",
        )
        self._save_stage_checkpoint("stage_base")

        if log_router_weights and router_weights_dir is not None:
            self._save_router_weights(
                conditions=[c for c, _ in incremental_loaders] + test_conditions,
                output_path=os.path.join(router_weights_dir, "router_weights_after_base.json"),
            )

        # Stage 2: incremental conditions
        for cond, loader in incremental_loaders:
            self._log_stage("Stage 2: incremental", condition=cond)

            # Phase B: router-only
            self._freeze_all()
            self._set_router_trainable(True)
            self._append_log_event({
                "event": "phase_start",
                "phase": "B",
                "condition": cond,
                "trainable": self._trainable_summary(),
            })
            trainable = self._trainable_summary()
            self._log_scalar(f"phase_b/cond_{cond}/trainable/router", float(trainable["router"]), self._global_step)
            self._log_scalar(f"phase_b/cond_{cond}/trainable/adapters", float(trainable["adapters"]), self._global_step)
            self._log_scalar(f"phase_b/cond_{cond}/trainable/unet", float(trainable["unet"]), self._global_step)
            self._train_phase(
                loader,
                epochs=phase_b_epochs,
                lr=phase_b_lr,
                phase_name=f"Phase B (router-only, cond={cond})",
                phase_tag=f"phase_b/cond_{cond}",
            )
            self._save_stage_checkpoint(f"cond_{cond}_router")

            # Phase C: refine with replay
            self._freeze_all()
            self._set_adapter_trainable(True)
            self._set_unet_parts_trainable(phase_c_unfreeze_policy)
            if phase_c_router_trainable:
                self._set_router_trainable(True)
            self._append_log_event({
                "event": "phase_start",
                "phase": "C",
                "condition": cond,
                "trainable": self._trainable_summary(),
            })
            trainable = self._trainable_summary()
            self._log_scalar(f"phase_c/cond_{cond}/trainable/router", float(trainable["router"]), self._global_step)
            self._log_scalar(f"phase_c/cond_{cond}/trainable/adapters", float(trainable["adapters"]), self._global_step)
            self._log_scalar(f"phase_c/cond_{cond}/trainable/unet", float(trainable["unet"]), self._global_step)

            self._train_phase(
                loader,
                epochs=phase_c_epochs,
                lr=phase_c_lr,
                phase_name=f"Phase C (refine, cond={cond})",
                phase_tag=f"phase_c/cond_{cond}",
                replay_dataloader=base_dataloader,
                replay_every=phase_c_replay_every,
                router_lr_scale=phase_c_router_lr_scale if phase_c_router_trainable else None,
            )
            self._save_stage_checkpoint(f"cond_{cond}_refine")

            if log_router_weights and router_weights_dir is not None:
                self._save_router_weights(
                    conditions=[c for c, _ in incremental_loaders] + test_conditions,
                    output_path=os.path.join(router_weights_dir, f"router_weights_after_cond_{cond}.json"),
                )

        # Stage 3: final evaluation on unseen conditions
        self._log_stage("Stage 3: final evaluation")
        self._append_log_event({
            "event": "evaluation_start",
            "conditions": test_conditions,
        })
        if eval_output_dir is not None:
            self.evaluate_conditions(
                conditions=test_conditions,
                output_dir=eval_output_dir,
                steps=eval_steps,
                guidance_scale=eval_guidance_scale,
                samples_per_condition=eval_samples_per_condition,
            )

        if self._tb_writer is not None:
            self._tb_writer.close()

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
