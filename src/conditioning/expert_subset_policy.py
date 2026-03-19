"""Subset policy for constrained MoE expert routing.

This module owns the mapping from condition IDs to allowed experts and the
dynamic unseen-condition fallback rules used at evaluation / inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


_VALID_UNSEEN_POLICIES = {"router_topk", "router_threshold", "full"}
_VALID_EMPTY_FALLBACKS = {"top1", "full"}


def _normalize_condition_id(condition_id: Any) -> Any:
    """Convert tensor-like scalar IDs into hashable Python scalars."""
    if torch.is_tensor(condition_id):
        if condition_id.numel() != 1:
            raise ValueError("condition_id tensors must be scalar")
        return condition_id.detach().cpu().item()
    return condition_id


@dataclass
class ExpertSubsetPolicy:
    """Resolve configured and dynamic expert subsets for routing."""

    num_experts: int
    enabled: bool = False
    configured_subsets: Dict[Any, Tuple[int, ...]] = field(default_factory=dict)
    incremental_must_use_base_experts: bool = True
    unseen_policy: str = "router_topk"
    top_k: int = 2
    threshold: Optional[float] = None
    min_experts: int = 1
    empty_fallback: str = "top1"

    def __post_init__(self) -> None:
        self.configured_subsets = {
            _normalize_condition_id(k): tuple(int(idx) for idx in v)
            for k, v in self.configured_subsets.items()
        }
        self._validate_self()

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        num_experts: int,
    ) -> "ExpertSubsetPolicy":
        configured = getattr(config, "configured_subsets", {}) or {}
        return cls(
            num_experts=int(num_experts),
            enabled=bool(getattr(config, "enabled", False)),
            configured_subsets=configured,
            incremental_must_use_base_experts=bool(
                getattr(config, "incremental_must_use_base_experts", True)
            ),
            unseen_policy=str(getattr(config, "unseen_policy", "router_topk")),
            top_k=int(getattr(config, "top_k", 2)),
            threshold=getattr(config, "threshold", None),
            min_experts=int(getattr(config, "min_experts", 1)),
            empty_fallback=str(getattr(config, "empty_fallback", "top1")),
        )

    def _validate_self(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be >= 1")
        if self.unseen_policy not in _VALID_UNSEEN_POLICIES:
            raise ValueError(
                f"Unknown unseen_policy {self.unseen_policy!r}; expected one of "
                f"{sorted(_VALID_UNSEEN_POLICIES)}"
            )
        if self.empty_fallback not in _VALID_EMPTY_FALLBACKS:
            raise ValueError(
                f"Unknown empty_fallback {self.empty_fallback!r}; expected one of "
                f"{sorted(_VALID_EMPTY_FALLBACKS)}"
            )
        if self.top_k <= 0:
            raise ValueError("top_k must be >= 1")
        if self.min_experts <= 0:
            raise ValueError("min_experts must be >= 1")
        if self.unseen_policy == "router_threshold" and self.threshold is None:
            raise ValueError("threshold must be set when unseen_policy='router_threshold'")

        for condition_id, subset in self.configured_subsets.items():
            if not subset:
                raise ValueError(f"Configured subset for condition {condition_id!r} is empty")
            for expert_idx in subset:
                if expert_idx < 0 or expert_idx >= self.num_experts:
                    raise ValueError(
                        f"Condition {condition_id!r} references invalid expert index "
                        f"{expert_idx}; expected [0, {self.num_experts - 1}]"
                    )

    def validate_training_conditions(
        self,
        *,
        base_conditions: Optional[Iterable[Any]] = None,
        incremental_conditions: Optional[Iterable[Any]] = None,
    ) -> None:
        """Validate configured subsets against the curriculum split."""
        if not self.enabled:
            return

        base_ids = [_normalize_condition_id(c) for c in (base_conditions or [])]
        incremental_ids = [_normalize_condition_id(c) for c in (incremental_conditions or [])]

        missing = [
            cond for cond in base_ids + incremental_ids
            if cond not in self.configured_subsets
        ]
        if missing:
            raise ValueError(
                "subset_policy.enabled=True requires configured_subsets for all "
                f"base/incremental training conditions; missing={missing}"
            )

        if self.incremental_must_use_base_experts and incremental_ids:
            base_experts = set()
            for cond in base_ids:
                base_experts.update(self.configured_subsets.get(cond, ()))
            for cond in incremental_ids:
                subset = set(self.configured_subsets[cond])
                unknown = sorted(subset - base_experts)
                if unknown:
                    raise ValueError(
                        f"Incremental condition {cond!r} uses experts {unknown} that do not "
                        "appear in any base subset"
                    )

    def has_configured_subset(self, condition_id: Any) -> bool:
        return _normalize_condition_id(condition_id) in self.configured_subsets

    def get_configured_subset(self, condition_id: Any) -> Optional[Tuple[int, ...]]:
        return self.configured_subsets.get(_normalize_condition_id(condition_id))

    def full_mask(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return torch.ones(batch_size, self.num_experts, dtype=torch.bool, device=device)

    def build_masks(
        self,
        condition_ids: Sequence[Any],
        *,
        raw_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        require_configured: bool = False,
    ) -> torch.Tensor:
        """Build a boolean expert mask per sample."""
        batch_size = len(condition_ids)
        if batch_size == 0:
            return self.full_mask(0, device=device)
        if raw_weights is not None and raw_weights.shape[0] != batch_size:
            raise ValueError(
                "condition_ids and raw_weights must have matching batch dimensions"
            )

        if not self.enabled:
            return self.full_mask(batch_size, device=device)

        if device is None and raw_weights is not None:
            device = raw_weights.device
        masks = self.full_mask(batch_size, device=device)

        for i, condition_id in enumerate(condition_ids):
            subset = self.get_configured_subset(condition_id)
            if subset is not None:
                masks[i] = False
                masks[i, list(subset)] = True
                continue

            if require_configured:
                raise ValueError(
                    f"Missing configured subset for condition_id={_normalize_condition_id(condition_id)!r}"
                )
            masks[i] = self._build_dynamic_mask(
                raw_weights=raw_weights,
                row_index=i,
                device=masks.device,
            )

        return masks

    def _build_dynamic_mask(
        self,
        *,
        raw_weights: Optional[torch.Tensor],
        row_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.unseen_policy == "full":
            return self.full_mask(1, device=device)[0]
        if raw_weights is None:
            raise ValueError(
                "raw_weights are required to build a dynamic unseen subset"
            )
        if raw_weights.ndim != 2 or raw_weights.shape[1] != self.num_experts:
            raise ValueError(
                f"raw_weights must have shape (B, {self.num_experts}), got {tuple(raw_weights.shape)}"
            )

        row = raw_weights[row_index]
        mask = torch.zeros(self.num_experts, dtype=torch.bool, device=device)

        if self.unseen_policy == "router_topk":
            k = min(self.num_experts, max(self.min_experts, self.top_k))
            top_idx = torch.topk(row, k=k, dim=-1).indices
            mask[top_idx] = True
            return self._apply_empty_fallback(mask, row)

        # unseen_policy == "router_threshold"
        threshold = float(self.threshold if self.threshold is not None else 0.0)
        mask = row >= threshold
        if int(mask.sum().item()) < self.min_experts:
            top_idx = torch.topk(row, k=min(self.num_experts, self.min_experts), dim=-1).indices
            mask[top_idx] = True
        return self._apply_empty_fallback(mask, row)

    def _apply_empty_fallback(
        self,
        mask: torch.Tensor,
        raw_row: torch.Tensor,
    ) -> torch.Tensor:
        if mask.any():
            return mask
        if self.empty_fallback == "full":
            return torch.ones_like(mask, dtype=torch.bool)
        top_idx = torch.argmax(raw_row).view(1)
        fallback = torch.zeros_like(mask, dtype=torch.bool)
        fallback[top_idx] = True
        return fallback
