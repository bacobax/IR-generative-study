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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.conditioning.expert_subset_policy import ExpertSubsetPolicy
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
    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        """Convert config scalar values to float with a safe default."""
        if value is None:
            return default
        return float(value)

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
        router_balance_weight: float = 0.0,
        subset_policy: Optional[ExpertSubsetPolicy] = None,
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
        self.router_sparsity_weight = self._coerce_float(router_sparsity_weight)
        self.router_smoothness_weight = self._coerce_float(router_smoothness_weight)
        self.router_balance_weight = self._coerce_float(router_balance_weight)
        self._meta_log_path = os.path.join(self.model_dir, "meta_fm_log.jsonl")
        self._tb_log_dir = log_dir
        self._tb_writer: Optional[SummaryWriter] = None
        self._global_step = 0
        self._log_every_steps = max(1, int(log_every_steps))
        if subset_policy is None:
            subset_policy = ExpertSubsetPolicy(
                num_experts=self._moe_unet().num_experts,
                enabled=False,
            )
        self.subset_policy = subset_policy

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

        device = config.resolved_device() if hasattr(config, "resolved_device") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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

        if not isinstance(unet, TextMOEUNet):
            raise TypeError("MetaFMTrainer requires a TextMOEUNet model")

        subset_policy = cls._resume_subset_policy(
            resume=getattr(config.output, "resume", None),
            model_dir=config.output.model_dir,
            checkpoint_dir=config.checkpoint.resolved_dir(config.output.model_dir),
            latest_filename=config.checkpoint.latest_filename,
            num_experts=unet.num_experts,
        )
        if subset_policy is None:
            subset_policy = ExpertSubsetPolicy.from_config(
                config.subset_policy,
                num_experts=unet.num_experts,
            )
            subset_policy.validate_training_conditions(
                base_conditions=getattr(config.condition_split, "base", None),
                incremental_conditions=getattr(config.condition_split, "incremental", None),
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
            router_balance_weight=config.router_reg.balance_weight,
            subset_policy=subset_policy,
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

    def _condition_ids_from_batch(self, batch: Dict[str, Any]) -> List[Any]:
        condition_ids = batch.get("condition_id")
        if condition_ids is None:
            raise ValueError("MetaFMTrainer batches must include 'condition_id'")
        if torch.is_tensor(condition_ids):
            if condition_ids.ndim == 0:
                return [condition_ids.item()]
            return condition_ids.detach().cpu().tolist()
        return list(condition_ids)

    @staticmethod
    def _coerce_condition_key(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        return value

    @classmethod
    def _subset_policy_payload_from_policy(
        cls,
        policy: ExpertSubsetPolicy,
    ) -> Dict[str, Any]:
        return {
            "enabled": bool(policy.enabled),
            "configured_subsets": {
                key: list(value) for key, value in policy.configured_subsets.items()
            },
            "incremental_must_use_base_experts": bool(
                policy.incremental_must_use_base_experts
            ),
            "unseen_policy": policy.unseen_policy,
            "top_k": int(policy.top_k),
            "threshold": policy.threshold,
            "min_experts": int(policy.min_experts),
            "empty_fallback": policy.empty_fallback,
        }

    @classmethod
    def _subset_policy_from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        num_experts: int,
    ) -> ExpertSubsetPolicy:
        configured = payload.get("configured_subsets", {}) or {}
        normalized_configured = {
            cls._coerce_condition_key(key): list(value)
            for key, value in configured.items()
        }
        return ExpertSubsetPolicy(
            num_experts=int(payload.get("num_experts", num_experts)),
            enabled=bool(payload.get("enabled", False)),
            configured_subsets=normalized_configured,
            incremental_must_use_base_experts=bool(
                payload.get("incremental_must_use_base_experts", True)
            ),
            unseen_policy=str(payload.get("unseen_policy", "router_topk")),
            top_k=int(payload.get("top_k", 2)),
            threshold=payload.get("threshold"),
            min_experts=int(payload.get("min_experts", 1)),
            empty_fallback=str(payload.get("empty_fallback", "top1")),
        )

    def _resolve_routing(
        self,
        pooled_text_embeds: Optional[torch.Tensor],
        *,
        condition_ids: Optional[List[Any]],
        require_configured: bool,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if pooled_text_embeds is None:
            return None

        moe = self._moe_unet()
        raw_logits = moe.compute_router_logits(pooled_text_embeds)
        raw_weights = moe.router.weights_from_logits(raw_logits)
        expert_mask = self.subset_policy.build_masks(
            condition_ids or [None] * raw_weights.shape[0],
            raw_weights=raw_weights,
            device=raw_weights.device,
            require_configured=require_configured,
        )
        masked_weights = moe.normalize_masked_weights(raw_weights, expert_mask)
        return {
            "raw_logits": raw_logits,
            "raw_weights": raw_weights,
            "expert_mask": expert_mask,
            "masked_weights": masked_weights,
        }

    def _log_router_distribution(
        self,
        *,
        phase_tag: str,
        prefix: str,
        weights: Optional[torch.Tensor],
        usage: Optional[torch.Tensor],
    ) -> None:
        if weights is None:
            return
        stats = self._router_stats(weights.detach())
        self._log_scalar(f"{phase_tag}/router/{prefix}_entropy", stats["entropy"].item(), self._global_step)
        self._log_scalar(f"{phase_tag}/router/{prefix}_max_weight", stats["max"].item(), self._global_step)
        self._log_scalar(f"{phase_tag}/router/{prefix}_top1_mean", stats["top1"].item(), self._global_step)
        self._log_scalar(f"{phase_tag}/router/{prefix}_sparsity", stats["sparsity"].item(), self._global_step)
        self._log_hist(f"{phase_tag}/router/{prefix}_weights", weights.detach().cpu(), self._global_step)
        for k in range(stats["mean"].numel()):
            self._log_scalar(
                f"{phase_tag}/router/{prefix}_mean_expert_{k}",
                stats["mean"][k].item(),
                self._global_step,
            )
            self._log_scalar(
                f"{phase_tag}/router/{prefix}_std_expert_{k}",
                stats["std"][k].item(),
                self._global_step,
            )
            self._log_scalar(
                f"{phase_tag}/router/{prefix}_batch_var_expert_{k}",
                stats["batch_var"][k].item(),
                self._global_step,
            )
        if usage is not None:
            for k in range(usage.numel()):
                self._log_scalar(
                    f"{phase_tag}/router/{prefix}_usage_expert_{k}",
                    usage[k].item(),
                    self._global_step,
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

    def _set_correction_trainable(self, trainable: bool) -> None:
        self._set_trainable(self._moe_unet().gated_correction, trainable)

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

    def _apply_phase_trainability(self, phase_config: Any) -> Dict[str, bool]:
        """Apply explicit YAML-driven trainability for one phase."""
        self._freeze_all()
        self._moe_unet().set_lambda_corr(float(getattr(phase_config, "lambda_corr", 1.0)))
        self._set_correction_trainable(bool(getattr(phase_config, "mlp_trainable", False)))
        self._set_router_trainable(bool(getattr(phase_config, "router_trainable", False)))
        self._set_adapter_trainable(bool(getattr(phase_config, "moe_trainable", False)))
        if bool(getattr(phase_config, "unet_trainable", False)):
            self._set_unet_parts_trainable(getattr(phase_config, "unfreeze_unet_policy", "all"))
        return self._trainable_summary()

    def _checkpoint_dir(self, checkpoint_dir: Optional[str] = None) -> str:
        return checkpoint_dir or os.path.join(self.model_dir, "meta_checkpoints")

    @staticmethod
    def _resolve_resume_path_static(
        resume: Optional[str],
        *,
        checkpoint_dir: str,
        latest_filename: str,
    ) -> Optional[str]:
        if resume is None:
            return None
        if resume == "latest":
            path = os.path.join(checkpoint_dir, latest_filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"No latest checkpoint found at {path}")
            return path
        if not os.path.isfile(resume):
            raise FileNotFoundError(f"Checkpoint not found: {resume}")
        return resume

    @classmethod
    def _resume_subset_policy(
        cls,
        *,
        resume: Optional[str],
        model_dir: str,
        checkpoint_dir: str,
        latest_filename: str,
        num_experts: int,
    ) -> Optional[ExpertSubsetPolicy]:
        resolved_resume = cls._resolve_resume_path_static(
            resume,
            checkpoint_dir=checkpoint_dir,
            latest_filename=latest_filename,
        )
        if resolved_resume is None:
            return None
        candidate_paths: List[str] = []
        ckpt_dir = os.path.dirname(resolved_resume)
        candidate_paths.append(os.path.join(ckpt_dir, "subset_policy.json"))
        candidate_paths.append(
            os.path.join(os.path.dirname(ckpt_dir), "subset_policy.json")
        )
        candidate_paths.append(os.path.join(model_dir, "subset_policy.json"))

        seen_paths = set()
        for path in candidate_paths:
            if path in seen_paths or not os.path.isfile(path):
                continue
            seen_paths.add(path)
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return cls._subset_policy_from_payload(payload, num_experts=num_experts)
        checkpoint = torch.load(resolved_resume, map_location="cpu")
        payload = checkpoint.get("subset_policy")
        if payload is not None:
            return cls._subset_policy_from_payload(payload, num_experts=num_experts)
        return None

    def _latest_checkpoint_path(
        self,
        *,
        checkpoint_dir: Optional[str] = None,
        latest_filename: str = "meta_fm_latest.pt",
    ) -> str:
        return os.path.join(self._checkpoint_dir(checkpoint_dir), latest_filename)

    def _move_optimizer_state_to_device(self, optimizer: torch.optim.Optimizer) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _restore_rng_state(self, checkpoint: Dict[str, Any]) -> None:
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            if not torch.is_tensor(rng_state) or rng_state.dtype != torch.uint8:
                rng_state = torch.tensor(rng_state, dtype=torch.uint8)
            torch.random.set_rng_state(rng_state.cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            cuda_states = []
            for state in checkpoint["cuda_rng_state_all"]:
                if not torch.is_tensor(state) or state.dtype != torch.uint8:
                    state = torch.tensor(state, dtype=torch.uint8)
                cuda_states.append(state.cpu())
            torch.cuda.set_rng_state_all(cuda_states)

    def _save_training_checkpoint(
        self,
        *,
        checkpoint_dir: str,
        latest_filename: str,
        save_latest: bool,
        phase: str,
        epoch_in_phase: int,
        incremental_index: int,
        condition: Optional[int],
        optimizer: torch.optim.Optimizer,
        lr: float,
        router_lr_scale: Optional[float],
        replay_every: int,
        scheduler_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        os.makedirs(checkpoint_dir, exist_ok=True)
        cond_label = "base" if condition is None else f"cond_{condition}"
        filename = (
            f"meta_fm_{phase}_{cond_label}_epoch_{epoch_in_phase:04d}.pt"
        )
        path = os.path.join(checkpoint_dir, filename)
        payload = {
            "epoch_in_phase": epoch_in_phase,
            "phase": phase,
            "incremental_index": incremental_index,
            "condition": condition,
            "global_step": self._global_step,
            "unet_state": self.unet.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler_state,
            "rng_state": torch.random.get_rng_state(),
            "t_scale": self.t_scale,
            "train_target": self.train_target,
            "phase_lr": lr,
            "router_lr_scale": router_lr_scale,
            "replay_every": replay_every,
            "trainable": self._trainable_summary(),
            "subset_policy": {
                **self._subset_policy_payload_from_policy(self.subset_policy),
                "num_experts": int(self.subset_policy.num_experts),
            },
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)
        if save_latest:
            torch.save(payload, os.path.join(checkpoint_dir, latest_filename))
        return path

    def _resolve_resume_path(
        self,
        resume: Optional[str],
        *,
        checkpoint_dir: Optional[str] = None,
        latest_filename: str = "meta_fm_latest.pt",
    ) -> Optional[str]:
        return self._resolve_resume_path_static(
            resume,
            checkpoint_dir=self._checkpoint_dir(checkpoint_dir),
            latest_filename=latest_filename,
        )

    def _load_training_checkpoint(
        self,
        resume: Optional[str],
        *,
        checkpoint_dir: Optional[str] = None,
        latest_filename: str = "meta_fm_latest.pt",
    ) -> Optional[Dict[str, Any]]:
        path = self._resolve_resume_path(
            resume,
            checkpoint_dir=checkpoint_dir,
            latest_filename=latest_filename,
        )
        if path is None:
            return None
        print(f"[MetaFM][Resume] Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(checkpoint["unet_state"])
        self._global_step = int(checkpoint.get("global_step", 0))
        self._restore_rng_state(checkpoint)
        subset_policy_payload = checkpoint.get("subset_policy")
        if subset_policy_payload is not None:
            self.subset_policy = self._subset_policy_from_payload(
                subset_policy_payload,
                num_experts=self._moe_unet().num_experts,
            )
        else:
            restored_subset_policy = self._resume_subset_policy(
                resume=path,
                model_dir=self.model_dir,
                checkpoint_dir=self._checkpoint_dir(checkpoint_dir),
                latest_filename=latest_filename,
                num_experts=self._moe_unet().num_experts,
            )
            if restored_subset_policy is not None:
                self.subset_policy = restored_subset_policy
        checkpoint["_resolved_path"] = path
        return checkpoint

    def _phase_total_epochs(self, phase: str, totals: Dict[str, int]) -> int:
        return int(totals.get(phase, 0))

    def _advance_resume_cursor(
        self,
        *,
        phase: str,
        epoch_in_phase: int,
        incremental_index: int,
        incremental_loaders: List[Tuple[int, DataLoader]],
        phase_epochs: Dict[str, int],
    ) -> Dict[str, Any]:
        current_phase = phase
        current_epoch = int(epoch_in_phase)
        current_index = int(incremental_index)

        while True:
            total_epochs = self._phase_total_epochs(current_phase, phase_epochs)
            if current_epoch < total_epochs:
                break
            current_epoch = 0
            if current_phase == "phase_a":
                if not incremental_loaders:
                    current_phase = "evaluation"
                else:
                    current_phase = "phase_b"
                    current_index = 0
            elif current_phase == "phase_b":
                current_phase = "phase_c"
            elif current_phase == "phase_c":
                current_index += 1
                if current_index >= len(incremental_loaders):
                    current_phase = "evaluation"
                else:
                    current_phase = "phase_b"
            elif current_phase in {"evaluation", "completed"}:
                current_phase = "completed"
                break
            else:
                raise ValueError(f"Unknown resume phase: {current_phase!r}")

        condition = None
        if current_phase in {"phase_b", "phase_c"} and 0 <= current_index < len(incremental_loaders):
            condition = incremental_loaders[current_index][0]
        return {
            "phase": current_phase,
            "epoch_in_phase": current_epoch,
            "incremental_index": current_index,
            "condition": condition,
        }

    # ------------------------------------------------------------------
    # Phase training helpers
    # ------------------------------------------------------------------
    def _loss_components(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        images = batch["pixel_values"].to(self.device)
        x_fm = self.encode_fm_input(images)
        cond_kw = self.conditioner.prepare_for_training(batch, self.device)
        pooled = cond_kw.get("pooled_text_embeds")
        condition_ids = self._condition_ids_from_batch(batch)
        routing = self._resolve_routing(
            pooled,
            condition_ids=condition_ids,
            require_configured=True,
        )
        if routing is not None:
            cond_kw = dict(cond_kw)
            cond_kw["router_weights"] = routing["masked_weights"]

        fm_loss = self.flow_matching_step(x_fm, cond_kw)

        raw_weights = None
        masked_weights = None
        expert_mask = None
        raw_usage = None
        masked_usage = None
        sparsity = torch.tensor(0.0, device=fm_loss.device)
        smooth = torch.tensor(0.0, device=fm_loss.device)
        balance = torch.tensor(0.0, device=fm_loss.device)

        if routing is not None:
            raw_weights = routing["raw_weights"]
            masked_weights = routing["masked_weights"]
            expert_mask = routing["expert_mask"]
            raw_usage = raw_weights.mean(dim=0)
            masked_usage = masked_weights.mean(dim=0)
            if self.router_sparsity_weight > 0:
                sparsity = 1.0 - raw_weights.pow(2).sum(dim=1).mean()
            if self.router_smoothness_weight > 0 and raw_weights.shape[0] > 1:
                smooth = (raw_weights[1:] - raw_weights[:-1]).pow(2).mean()
            if self.router_balance_weight > 0:
                balance = raw_weights.shape[1] * raw_usage.pow(2).sum()

        total = fm_loss
        if self.router_sparsity_weight > 0:
            total = total + self.router_sparsity_weight * sparsity
        if self.router_smoothness_weight > 0:
            total = total + self.router_smoothness_weight * smooth
        if self.router_balance_weight > 0:
            total = total + self.router_balance_weight * balance

        return {
            "total": total,
            "fm": fm_loss,
            "sparsity": sparsity,
            "smooth": smooth,
            "balance": balance,
            "usage": raw_usage,
            "masked_usage": masked_usage,
            "weights": raw_weights,
            "raw_weights": raw_weights,
            "masked_weights": masked_weights,
            "expert_mask": expert_mask,
        }

    def _loss_from_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self._loss_components(batch)["total"]

    def _build_optimizer(
        self,
        lr: float,
        *,
        router_lr_scale: Optional[float] = None,
    ) -> torch.optim.Optimizer:
        def _require_params(params: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
            if not params:
                raise ValueError(
                    "No trainable parameters are enabled for this phase. "
                    "Check the YAML phase module trainability settings."
                )
            return params

        if router_lr_scale is None or router_lr_scale == 1.0:
            params = _require_params([p for p in self.unet.parameters() if p.requires_grad])
            return Adam(params, lr=lr)

        moe = self._moe_unet()
        router_params = [p for p in moe.router.parameters() if p.requires_grad]
        router_param_ids = {id(p) for p in router_params}
        other_params = [
            p for p in self.unet.parameters()
            if p.requires_grad and id(p) not in router_param_ids
        ]
        _require_params(router_params + other_params)
        return Adam(
            [
                {"params": other_params, "lr": lr},
                {"params": router_params, "lr": lr * router_lr_scale},
            ]
        )

    def _save_subset_policy_metadata(self) -> None:
        path = os.path.join(self.model_dir, "subset_policy.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = self._subset_policy_payload_from_policy(self.subset_policy)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

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
        start_epoch: int = 0,
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        on_epoch_end: Optional[Callable[[int], None]] = None,
        checkpoint_callback: Optional[
            Callable[[int, torch.optim.Optimizer, Optional[Dict[str, Any]]], None]
        ] = None,
    ) -> None:
        optimizer = self._build_optimizer(lr, router_lr_scale=router_lr_scale)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_device(optimizer)
        replay_iter = None
        if replay_dataloader is not None:
            replay_iter = itertools.cycle(replay_dataloader)

        for epoch in range(start_epoch, epochs):
            self.unet.train()
            total_loss = 0.0
            total_fm = 0.0
            total_sparsity = 0.0
            total_smooth = 0.0
            total_balance = 0.0
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
                total_balance += losses["balance"].item()

                if (step + 1) % self._log_every_steps == 0:
                    self._log_scalar(f"{phase_tag}/loss/total_step", loss.item(), self._global_step)
                    self._log_scalar(f"{phase_tag}/loss/fm_step", losses["fm"].item(), self._global_step)
                    if self.router_sparsity_weight > 0:
                        self._log_scalar(f"{phase_tag}/loss/sparsity_step", losses["sparsity"].item(), self._global_step)
                    if self.router_smoothness_weight > 0:
                        self._log_scalar(f"{phase_tag}/loss/smooth_step", losses["smooth"].item(), self._global_step)
                    if self.router_balance_weight > 0:
                        self._log_scalar(f"{phase_tag}/loss/balance_step", losses["balance"].item(), self._global_step)
                    self._log_scalar(f"{phase_tag}/grad_norm", grad_norm, self._global_step)
                    for gi, group in enumerate(optimizer.param_groups):
                        self._log_scalar(f"{phase_tag}/lr/group_{gi}", group.get("lr", 0.0), self._global_step)

                    self._log_router_distribution(
                        phase_tag=phase_tag,
                        prefix="raw",
                        weights=losses.get("raw_weights"),
                        usage=losses.get("usage"),
                    )
                    self._log_router_distribution(
                        phase_tag=phase_tag,
                        prefix="masked",
                        weights=losses.get("masked_weights"),
                        usage=losses.get("masked_usage"),
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
            avg_balance = total_balance / max(1, len(dataloader))
            avg_replay = total_replay / max(1, replay_steps) if replay_steps else 0.0
            replay_ratio = replay_steps / max(1, (len(dataloader) + replay_steps))
            print(f"[{phase_name} epoch {epoch+1}] loss: {avg:.6f}")

            self._log_scalar(f"{phase_tag}/loss/total_epoch", avg, epoch)
            self._log_scalar(f"{phase_tag}/loss/fm_epoch", avg_fm, epoch)
            if self.router_sparsity_weight > 0:
                self._log_scalar(f"{phase_tag}/loss/sparsity_epoch", avg_sparsity, epoch)
            if self.router_smoothness_weight > 0:
                self._log_scalar(f"{phase_tag}/loss/smooth_epoch", avg_smooth, epoch)
            if self.router_balance_weight > 0:
                self._log_scalar(f"{phase_tag}/loss/balance_epoch", avg_balance, epoch)
            if replay_steps:
                self._log_scalar(f"{phase_tag}/replay/total_epoch", avg_replay, epoch)
                self._log_scalar(f"{phase_tag}/replay/ratio", replay_ratio, epoch)

            if on_epoch_end is not None:
                on_epoch_end(epoch)
            if checkpoint_callback is not None:
                checkpoint_callback(epoch + 1, optimizer, scheduler_state)

    def _maybe_load_pretrained_components(
        self,
        *,
        pretrained_vae_path: Optional[str] = None,
        strict_load: bool = True,
    ) -> None:
        """Load frozen pretrained components needed before meta training begins."""
        self._ensure_dirs()
        self._save_configs()
        self._save_subset_policy_metadata()

        if pretrained_vae_path is not None and self.vae is not None:
            self.load_vae_weights(pretrained_vae_path, strict=strict_load)
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

    def _make_sampler(self):
        from src.algorithms.inference.cfg_flow_matching_sampler import (
            CFGFlowMatchingSampler,
        )

        if self.vae is not None:
            return CFGFlowMatchingSampler.from_stable(
                self.unet,
                self.vae,
                conditioner=self.conditioner,
                subset_policy=self.subset_policy,
                device=self.device,
                t_scale=self.t_scale,
                train_target=self.train_target,
                from_norm_to_display=self.from_norm_to_display,
            )
        return CFGFlowMatchingSampler(
            self.unet,
            conditioner=self.conditioner,
            subset_policy=self.subset_policy,
            device=self.device,
            t_scale=self.t_scale,
            train_target=self.train_target,
            from_norm_to_display=self.from_norm_to_display,
        )

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
        phase_a_config: Optional[Any] = None,
        phase_b_config: Optional[Any] = None,
        phase_c_config: Optional[Any] = None,
        phase_c_unfreeze_policy: str = "none",
        phase_c_router_trainable: bool = True,
        phase_c_router_lr_scale: float = 1.0,
        phase_c_replay_every: int = 1,
        pretrained_vae_path: Optional[str] = None,
        strict_load: bool = True,
    ) -> None:
        """Run a single incremental episode (base -> router -> refine)."""
        self._ensure_writer()
        self._maybe_load_pretrained_components(
            pretrained_vae_path=pretrained_vae_path,
            strict_load=strict_load,
        )
        if phase_a_config is None:
            phase_a_config = type(
                "PhaseAConfig",
                (),
                {
                    "lambda_corr": 1.0,
                    "mlp_trainable": True,
                    "router_trainable": True,
                    "moe_trainable": True,
                    "unet_trainable": True,
                    "unfreeze_unet_policy": "all",
                },
            )()
        if phase_b_config is None:
            phase_b_config = type(
                "PhaseBConfig",
                (),
                {
                    "lambda_corr": 1.0,
                    "mlp_trainable": True,
                    "router_trainable": True,
                    "moe_trainable": False,
                    "unet_trainable": False,
                    "unfreeze_unet_policy": "none",
                },
            )()
        if phase_c_config is None:
            phase_c_config = type(
                "PhaseCConfig",
                (),
                {
                    "lambda_corr": 1.0,
                    "mlp_trainable": True,
                    "router_trainable": phase_c_router_trainable,
                    "moe_trainable": True,
                    "unet_trainable": phase_c_unfreeze_policy != "none",
                    "unfreeze_unet_policy": phase_c_unfreeze_policy,
                },
            )()
        # Phase A: base training
        self._apply_phase_trainability(phase_a_config)
        trainable = self._trainable_summary()
        self._log_scalar("phase_a/trainable/router", float(trainable["router"]), self._global_step)
        self._log_scalar("phase_a/trainable/moe", float(trainable["moe"]), self._global_step)
        self._log_scalar("phase_a/trainable/mlp", float(trainable["mlp"]), self._global_step)
        self._log_scalar("phase_a/trainable/unet", float(trainable["unet"]), self._global_step)
        self._train_phase(
            base_dataloader,
            epochs=phase_a_epochs,
            lr=phase_a_lr,
            phase_name="Phase A (base)",
            phase_tag="phase_a",
        )

        # Phase B: router-only on new condition
        self._apply_phase_trainability(phase_b_config)
        trainable = self._trainable_summary()
        self._log_scalar("phase_b/trainable/router", float(trainable["router"]), self._global_step)
        self._log_scalar("phase_b/trainable/moe", float(trainable["moe"]), self._global_step)
        self._log_scalar("phase_b/trainable/mlp", float(trainable["mlp"]), self._global_step)
        self._log_scalar("phase_b/trainable/unet", float(trainable["unet"]), self._global_step)
        self._train_phase(
            new_dataloader,
            epochs=phase_b_epochs,
            lr=phase_b_lr,
            phase_name="Phase B (router-only)",
            phase_tag="phase_b",
        )

        # Phase C: adapters (+ optional UNet parts) with replay
        self._apply_phase_trainability(phase_c_config)
        trainable = self._trainable_summary()
        self._log_scalar("phase_c/trainable/router", float(trainable["router"]), self._global_step)
        self._log_scalar("phase_c/trainable/moe", float(trainable["moe"]), self._global_step)
        self._log_scalar("phase_c/trainable/mlp", float(trainable["mlp"]), self._global_step)
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
        self._log_scalar(f"{phase_tag}/trainable/correction", float(trainable["correction"]), self._global_step)
        self._log_scalar(f"{phase_tag}/trainable/unet", float(trainable["unet"]), self._global_step)

    def _append_log_event(self, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self._meta_log_path) or ".", exist_ok=True)
        with open(self._meta_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _trainable_summary(self) -> Dict[str, bool]:
        moe = self._moe_unet()

        def _any_trainable(module: torch.nn.Module) -> bool:
            return any(p.requires_grad for p in module.parameters())

        router = _any_trainable(moe.router)
        moe_trainable = _any_trainable(moe.mid_adapter)
        mlp = _any_trainable(moe.gated_correction)
        unet = _any_trainable(moe.unet)
        return {
            "router": router,
            "moe": moe_trainable,
            "mlp": mlp,
            "unet": unet,
            "adapters": moe_trainable,
            "correction": mlp,
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
        routing = self._resolve_routing(
            pooled,
            condition_ids=conditions,
            require_configured=False,
        )
        if routing is None:
            raise RuntimeError("Expected pooled text embeddings for router snapshot logging")
        payload = {
            "condition_ids": conditions,
            "prompts": prompts,
            "raw_weights": routing["raw_weights"].detach().cpu().tolist(),
            "expert_mask": routing["expert_mask"].detach().cpu().tolist(),
            "masked_weights": routing["masked_weights"].detach().cpu().tolist(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        writer = self._ensure_writer()
        if writer is not None:
            raw_w = routing["raw_weights"].detach().cpu()
            masked_w = routing["masked_weights"].detach().cpu()
            writer.add_histogram("router/snapshot_raw_weights", raw_w, self._global_step)
            writer.add_histogram("router/snapshot_masked_weights", masked_w, self._global_step)
            for i, cond in enumerate(conditions):
                writer.add_histogram(f"router/cond_{cond}/raw_weights", raw_w[i], self._global_step)
                writer.add_histogram(f"router/cond_{cond}/masked_weights", masked_w[i], self._global_step)

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
                condition_ids=[cond] * samples_per_condition,
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
                routing = self._resolve_routing(
                    pooled,
                    condition_ids=[cond] * samples_per_condition,
                    require_configured=False,
                )
                if routing is not None:
                    self._log_router_distribution(
                        phase_tag=f"eval/cond_{cond}",
                        prefix="raw",
                        weights=routing["raw_weights"],
                        usage=routing["raw_weights"].mean(dim=0),
                    )
                    self._log_router_distribution(
                        phase_tag=f"eval/cond_{cond}",
                        prefix="masked",
                        weights=routing["masked_weights"],
                        usage=routing["masked_weights"].mean(dim=0),
                    )

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
        base_conditions: Optional[List[int]] = None,
        test_conditions: List[int],
        phase_a_epochs: int,
        phase_b_epochs: int,
        phase_c_epochs: int,
        phase_a_lr: float,
        phase_b_lr: float,
        phase_c_lr: float,
        phase_a_config: Optional[Any] = None,
        phase_b_config: Optional[Any] = None,
        phase_c_config: Optional[Any] = None,
        phase_c_unfreeze_policy: str,
        phase_c_router_trainable: bool,
        phase_c_router_lr_scale: float,
        phase_c_replay_every: int,
        resume_from_checkpoint: Optional[str] = None,
        checkpoint_enabled: bool = True,
        checkpoint_every_epochs: int = 1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_latest_filename: str = "meta_fm_latest.pt",
        checkpoint_save_latest: bool = True,
        pretrained_vae_path: Optional[str] = None,
        strict_load: bool = True,
        log_router_weights: bool = True,
        router_weights_dir: Optional[str] = None,
        eval_output_dir: Optional[str] = None,
        eval_steps: int = 50,
        eval_guidance_scale: float = 7.5,
        eval_samples_per_condition: int = 4,
        sampling_enabled: bool = False,
        sampling_phase_a_every: int = 0,
        sampling_output_dir: Optional[str] = None,
        sampling_steps: int = 50,
        sampling_guidance_scale: float = 7.5,
        sampling_samples_per_condition: int = 4,
    ) -> None:
        self._ensure_writer()
        self._maybe_load_pretrained_components(
            pretrained_vae_path=pretrained_vae_path,
            strict_load=strict_load,
        )
        base_conditions = list(base_conditions or [])
        checkpoint_root = self._checkpoint_dir(checkpoint_dir)

        if phase_a_config is None:
            phase_a_config = type(
                "PhaseAConfig",
                (),
                {
                    "lambda_corr": 1.0,
                    "mlp_trainable": True,
                    "router_trainable": True,
                    "moe_trainable": True,
                    "unet_trainable": True,
                    "unfreeze_unet_policy": "all",
                },
            )()
        if phase_b_config is None:
            phase_b_config = type(
                "PhaseBConfig",
                (),
                {
                    "lambda_corr": 1.0,
                    "mlp_trainable": True,
                    "router_trainable": True,
                    "moe_trainable": False,
                    "unet_trainable": False,
                    "unfreeze_unet_policy": "none",
                },
            )()
        if phase_c_config is None:
            phase_c_config = type(
                "PhaseCConfig",
                (),
                {
                    "lambda_corr": 1.0,
                    "mlp_trainable": True,
                    "router_trainable": phase_c_router_trainable,
                    "moe_trainable": True,
                    "unet_trainable": phase_c_unfreeze_policy != "none",
                    "unfreeze_unet_policy": phase_c_unfreeze_policy,
                },
            )()

        phase_epochs = {
            "phase_a": phase_a_epochs,
            "phase_b": phase_b_epochs,
            "phase_c": phase_c_epochs,
        }
        resume_checkpoint = self._load_training_checkpoint(
            resume_from_checkpoint,
            checkpoint_dir=checkpoint_root,
            latest_filename=checkpoint_latest_filename,
        )
        resume_cursor = {
            "phase": "phase_a",
            "epoch_in_phase": 0,
            "incremental_index": -1,
            "condition": None,
        }
        if resume_checkpoint is not None:
            resume_cursor = self._advance_resume_cursor(
                phase=resume_checkpoint.get("phase", "phase_a"),
                epoch_in_phase=int(resume_checkpoint.get("epoch_in_phase", 0)),
                incremental_index=int(resume_checkpoint.get("incremental_index", -1)),
                incremental_loaders=incremental_loaders,
                phase_epochs=phase_epochs,
            )
            print(
                "[MetaFM][Resume] "
                f"phase={resume_cursor['phase']} "
                f"epoch={resume_cursor['epoch_in_phase']} "
                f"incremental_index={resume_cursor['incremental_index']}"
            )

        def _run_sanity_sampling(stage_tag: str, conditions: List[int]) -> None:
            if not sampling_enabled or sampling_output_dir is None or not conditions:
                return
            out_dir = os.path.join(sampling_output_dir, stage_tag)
            self.evaluate_conditions(
                conditions=conditions,
                output_dir=out_dir,
                steps=sampling_steps,
                guidance_scale=sampling_guidance_scale,
                samples_per_condition=sampling_samples_per_condition,
            )

        def _phase_a_epoch_hook(epoch_idx: int) -> None:
            if sampling_phase_a_every <= 0:
                return
            if (epoch_idx + 1) % sampling_phase_a_every != 0:
                return
            _run_sanity_sampling(
                stage_tag=f"phase_a_epoch_{epoch_idx + 1:04d}",
                conditions=base_conditions,
            )

        def _log_phase_state(phase_tag: str, condition: Optional[int] = None) -> None:
            trainable = self._trainable_summary()
            self._log_scalar(f"{phase_tag}/trainable/router", float(trainable["router"]), self._global_step)
            self._log_scalar(f"{phase_tag}/trainable/moe", float(trainable["moe"]), self._global_step)
            self._log_scalar(f"{phase_tag}/trainable/mlp", float(trainable["mlp"]), self._global_step)
            self._log_scalar(f"{phase_tag}/trainable/unet", float(trainable["unet"]), self._global_step)
            self._append_log_event({
                "event": "phase_start",
                "phase": phase_tag,
                "condition": condition,
                "trainable": trainable,
            })

        def _checkpoint_cb(
            *,
            phase: str,
            incremental_index: int,
            condition: Optional[int],
            lr: float,
            router_lr_scale: Optional[float],
            replay_every: int,
        ) -> Callable[[int, torch.optim.Optimizer, Optional[Dict[str, Any]]], None]:
            def _save(epoch_in_phase: int, optimizer: torch.optim.Optimizer, scheduler_state: Optional[Dict[str, Any]]) -> None:
                if not checkpoint_enabled or checkpoint_every_epochs <= 0:
                    return
                if epoch_in_phase % checkpoint_every_epochs != 0:
                    return
                self._save_training_checkpoint(
                    checkpoint_dir=checkpoint_root,
                    latest_filename=checkpoint_latest_filename,
                    save_latest=checkpoint_save_latest,
                    phase=phase,
                    epoch_in_phase=epoch_in_phase,
                    incremental_index=incremental_index,
                    condition=condition,
                    optimizer=optimizer,
                    scheduler_state=scheduler_state,
                    lr=lr,
                    router_lr_scale=router_lr_scale,
                    replay_every=replay_every,
                )

            return _save

        # Stage 1: base training
        if resume_cursor["phase"] == "phase_a":
            self._log_stage("Stage 1: base training")
            self._apply_phase_trainability(phase_a_config)
            _log_phase_state("phase_a", None)
            self._train_phase(
                base_dataloader,
                epochs=phase_a_epochs,
                lr=phase_a_lr,
                phase_name="Stage 1 (base)",
                phase_tag="phase_a",
                start_epoch=resume_cursor["epoch_in_phase"],
                optimizer_state=resume_checkpoint.get("optimizer_state") if resume_checkpoint is not None else None,
                scheduler_state=resume_checkpoint.get("scheduler_state") if resume_checkpoint is not None else None,
                on_epoch_end=_phase_a_epoch_hook,
                checkpoint_callback=_checkpoint_cb(
                    phase="phase_a",
                    incremental_index=-1,
                    condition=None,
                    lr=phase_a_lr,
                    router_lr_scale=None,
                    replay_every=0,
                ),
            )
            self._save_stage_checkpoint("stage_base")
            _run_sanity_sampling("after_phase_a", base_conditions)

            if log_router_weights and router_weights_dir is not None:
                self._save_router_weights(
                    conditions=[c for c, _ in incremental_loaders] + test_conditions,
                    output_path=os.path.join(router_weights_dir, "router_weights_after_base.json"),
                )
            resume_checkpoint = None

        # Stage 2: incremental conditions
        start_incremental_index = 0
        if resume_cursor["phase"] in {"phase_b", "phase_c"}:
            start_incremental_index = max(0, resume_cursor["incremental_index"])
        elif resume_cursor["phase"] in {"evaluation", "completed"}:
            start_incremental_index = len(incremental_loaders)

        for incremental_index in range(start_incremental_index, len(incremental_loaders)):
            cond, loader = incremental_loaders[incremental_index]
            self._log_stage("Stage 2: incremental", condition=cond)

            if (
                resume_cursor["phase"] == "phase_b"
                and incremental_index == resume_cursor["incremental_index"]
            ) or resume_checkpoint is None:
                self._apply_phase_trainability(phase_b_config)
                _log_phase_state(f"phase_b/cond_{cond}", cond)
                phase_b_optimizer_state = None
                phase_b_scheduler_state = None
                phase_b_start_epoch = 0
                if (
                    resume_checkpoint is not None
                    and resume_cursor["phase"] == "phase_b"
                    and incremental_index == resume_cursor["incremental_index"]
                ):
                    phase_b_optimizer_state = resume_checkpoint.get("optimizer_state")
                    phase_b_scheduler_state = resume_checkpoint.get("scheduler_state")
                    phase_b_start_epoch = resume_cursor["epoch_in_phase"]
                self._train_phase(
                    loader,
                    epochs=phase_b_epochs,
                    lr=phase_b_lr,
                    phase_name=f"Phase B (router-only, cond={cond})",
                    phase_tag=f"phase_b/cond_{cond}",
                    start_epoch=phase_b_start_epoch,
                    optimizer_state=phase_b_optimizer_state,
                    scheduler_state=phase_b_scheduler_state,
                    checkpoint_callback=_checkpoint_cb(
                        phase="phase_b",
                        incremental_index=incremental_index,
                        condition=cond,
                        lr=phase_b_lr,
                        router_lr_scale=None,
                        replay_every=0,
                    ),
                )
                self._save_stage_checkpoint(f"cond_{cond}_router")
                _run_sanity_sampling(f"after_phase_b_cond_{cond}", [cond])
                resume_checkpoint = None

            self._apply_phase_trainability(phase_c_config)
            _log_phase_state(f"phase_c/cond_{cond}", cond)
            phase_c_optimizer_state = None
            phase_c_scheduler_state = None
            phase_c_start_epoch = 0
            if (
                resume_checkpoint is not None
                and resume_cursor["phase"] == "phase_c"
                and incremental_index == resume_cursor["incremental_index"]
            ):
                phase_c_optimizer_state = resume_checkpoint.get("optimizer_state")
                phase_c_scheduler_state = resume_checkpoint.get("scheduler_state")
                phase_c_start_epoch = resume_cursor["epoch_in_phase"]

            self._train_phase(
                loader,
                epochs=phase_c_epochs,
                lr=phase_c_lr,
                phase_name=f"Phase C (refine, cond={cond})",
                phase_tag=f"phase_c/cond_{cond}",
                replay_dataloader=base_dataloader,
                replay_every=phase_c_replay_every,
                router_lr_scale=phase_c_router_lr_scale if phase_c_config.router_trainable else None,
                start_epoch=phase_c_start_epoch,
                optimizer_state=phase_c_optimizer_state,
                scheduler_state=phase_c_scheduler_state,
                checkpoint_callback=_checkpoint_cb(
                    phase="phase_c",
                    incremental_index=incremental_index,
                    condition=cond,
                    lr=phase_c_lr,
                    router_lr_scale=phase_c_router_lr_scale if phase_c_config.router_trainable else None,
                    replay_every=phase_c_replay_every,
                ),
            )
            self._save_stage_checkpoint(f"cond_{cond}_refine")
            _run_sanity_sampling(f"after_phase_c_cond_{cond}", [cond])

            if log_router_weights and router_weights_dir is not None:
                self._save_router_weights(
                    conditions=[c for c, _ in incremental_loaders] + test_conditions,
                    output_path=os.path.join(router_weights_dir, f"router_weights_after_cond_{cond}.json"),
                )
            resume_checkpoint = None

        # Stage 3: final evaluation on unseen conditions
        if resume_cursor["phase"] != "completed":
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
            if checkpoint_enabled:
                final_optimizer = Adam(self.unet.parameters(), lr=phase_c_lr)
                self._save_training_checkpoint(
                    checkpoint_dir=checkpoint_root,
                    latest_filename=checkpoint_latest_filename,
                    save_latest=checkpoint_save_latest,
                    phase="completed",
                    epoch_in_phase=0,
                    incremental_index=len(incremental_loaders),
                    condition=None,
                    optimizer=final_optimizer,
                    scheduler_state=None,
                    lr=phase_c_lr,
                    router_lr_scale=None,
                    replay_every=0,
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
            phase_a_config=config.phase_a,
            phase_b_config=config.phase_b,
            phase_c_config=config.phase_c,
            phase_c_unfreeze_policy=config.phase_c.unfreeze_unet_policy,
            phase_c_router_trainable=config.phase_c.router_trainable,
            phase_c_router_lr_scale=config.phase_c.router_lr_scale,
            phase_c_replay_every=config.phase_c.replay_every,
            pretrained_vae_path=config.model.vae_weights,
            strict_load=config.training.strict_load,
        )


# ── registry ──────────────────────────────────────────────────────────────
REGISTRIES.trainer.register("meta_fm")(MetaFMTrainer)
