"""Task-level flow-matching state sampling and loss helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from src.core.ot import match_target_batch


AreaLossFn = Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]


@dataclass
class FlowMatchingTask:
    """Core FM target construction, path matching, and loss computation."""

    train_target: str = "v"
    path_mode: str = "independent"
    path_solver: str = "hungarian"
    area_loss_fn: AreaLossFn | None = None

    def __post_init__(self) -> None:
        if self.train_target not in {"v", "x0"}:
            raise ValueError(f"train_target must be 'v' or 'x0', got {self.train_target!r}")
        self.path_mode = str(self.path_mode)
        self.path_solver = str(self.path_solver)

    @staticmethod
    def permute_conditioning_kwargs(
        cond_kwargs: Dict[str, Any],
        permutation: Optional[torch.Tensor],
        batch_size: int,
    ) -> Dict[str, Any]:
        """Keep batch-aligned conditioning tensors and lists paired with matched targets."""
        if permutation is None:
            return cond_kwargs

        aligned: Dict[str, Any] = {}
        cpu_permutation = None
        for key, value in cond_kwargs.items():
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(batch_size):
                aligned[key] = value.index_select(0, permutation.to(value.device))
            elif isinstance(value, list) and len(value) == int(batch_size):
                if cpu_permutation is None:
                    cpu_permutation = permutation.detach().cpu().tolist()
                aligned[key] = [value[int(index)] for index in cpu_permutation]
            else:
                aligned[key] = value
        return aligned

    def match_targets_with_permutation(
        self,
        z0: torch.Tensor,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the configured path coupling and return targets plus permutation."""
        del cond_kwargs
        if self.path_mode == "independent":
            return x_fm, None
        if self.path_mode in {"minibatch_ot", "conditional_ot"}:
            matched, permutation, _ = match_target_batch(
                z0,
                x_fm,
                solver=self.path_solver,
            )
            return matched, permutation
        raise ValueError(
            f"Unsupported path_mode={self.path_mode!r}. Expected 'independent', "
            "'minibatch_ot', or 'conditional_ot'."
        )

    def match_targets(
        self,
        z0: torch.Tensor,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply the configured path coupling and return matched targets."""
        matched, _ = self.match_targets_with_permutation(z0, x_fm, cond_kwargs)
        return matched

    def sample_state(
        self,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sample FM path variables and keep conditioning aligned to matched targets."""
        if cond_kwargs is None:
            cond_kwargs = {}
        batch_size = x_fm.shape[0]
        z0 = torch.randn_like(x_fm)
        t = torch.rand(batch_size, device=x_fm.device)
        t_expanded = t[:, None, None, None]
        x_target, target_permutation = self.match_targets_with_permutation(z0, x_fm, cond_kwargs)
        aligned_cond = self.permute_conditioning_kwargs(cond_kwargs, target_permutation, batch_size)

        zt = (1 - t_expanded) * z0 + t_expanded * x_target
        v_target = x_target - z0
        return {
            "z0": z0,
            "t": t,
            "t_expanded": t_expanded,
            "x_target": x_target,
            "target_permutation": target_permutation,
            "cond_kwargs": aligned_cond,
            "zt": zt,
            "v_target": v_target,
        }

    def loss_from_prediction(
        self,
        prediction: torch.Tensor,
        state: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute the current FM MSE objective for v or x0 prediction."""
        t_expanded = state["t_expanded"]
        if self.train_target == "x0":
            x0_pred = prediction
            v_pred = (x0_pred - state["zt"]) / (1 - t_expanded).clamp(min=1e-5)
        else:
            v_pred = prediction

        loss = F.mse_loss(v_pred.float(), state["v_target"].float(), reduction="none")
        if self.area_loss_fn is not None:
            loss = self.area_loss_fn(loss, state["cond_kwargs"])
        return loss.mean()
