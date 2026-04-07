"""Mini-batch optimal transport helpers for flow-matching training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def pairwise_mean_squared_cost(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise mean squared cost matrix between two batches."""
    if source.shape[0] != target.shape[0]:
        raise ValueError(
            f"Mini-batch OT expects equal batch sizes, got {source.shape[0]} and {target.shape[0]}."
        )
    source_flat = source.detach().float().flatten(1)
    target_flat = target.detach().float().flatten(1)
    return (source_flat[:, None, :] - target_flat[None, :, :]).pow(2).mean(dim=-1)


def build_layout_descriptor(
    *,
    boxes_xyxy_norm: torch.Tensor,
    labels: torch.Tensor,
    object_mask: torch.Tensor,
    num_classes: int,
    resolution: int = 16,
) -> torch.Tensor:
    """Rasterize low-resolution class occupancy and counts for layout OT."""
    batch_size = int(boxes_xyxy_norm.shape[0])
    num_classes = int(max(1, num_classes))
    resolution = int(max(1, resolution))
    dtype = boxes_xyxy_norm.dtype
    device = boxes_xyxy_norm.device

    occupancy = torch.zeros(
        batch_size,
        num_classes,
        resolution,
        resolution,
        device=device,
        dtype=dtype,
    )
    counts = torch.zeros(batch_size, num_classes, device=device, dtype=dtype)

    for batch_idx in range(batch_size):
        for obj_idx in range(int(boxes_xyxy_norm.shape[1])):
            if not bool(object_mask[batch_idx, obj_idx]):
                continue

            class_idx = int(labels[batch_idx, obj_idx].item())
            if class_idx < 0 or class_idx >= num_classes:
                continue

            x1, y1, x2, y2 = boxes_xyxy_norm[batch_idx, obj_idx]
            ix1 = max(0, min(resolution - 1, int(torch.floor(x1 * resolution).item())))
            iy1 = max(0, min(resolution - 1, int(torch.floor(y1 * resolution).item())))
            ix2 = max(ix1 + 1, min(resolution, int(torch.ceil(x2 * resolution).item())))
            iy2 = max(iy1 + 1, min(resolution, int(torch.ceil(y2 * resolution).item())))

            occupancy[batch_idx, class_idx, iy1:iy2, ix1:ix2] = 1.0
            counts[batch_idx, class_idx] += 1.0

    descriptor = torch.cat(
        [occupancy.flatten(1), counts],
        dim=1,
    )
    return descriptor


def solve_assignment(cost_matrix: torch.Tensor, solver: str = "hungarian") -> torch.Tensor:
    """Solve a square assignment problem and return the target permutation."""
    solver_name = str(solver or "hungarian").lower()
    if solver_name != "hungarian":
        raise ValueError(
            f"Unsupported OT solver={solver!r}. Only 'hungarian' is implemented."
        )

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for minibatch OT. Install scipy>=1.10 to use "
            "path.mode='minibatch_ot' or 'conditional_ot'."
        ) from exc

    cpu_cost = cost_matrix.detach().cpu().float().numpy()
    row_ind, col_ind = linear_sum_assignment(cpu_cost)
    permutation = torch.empty(cost_matrix.shape[0], dtype=torch.long)
    permutation[torch.as_tensor(row_ind, dtype=torch.long)] = torch.as_tensor(col_ind, dtype=torch.long)
    return permutation.to(cost_matrix.device)


def match_target_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    solver: str = "hungarian",
    extra_cost: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder a target batch to the minimum-cost OT assignment."""
    cost_matrix = pairwise_mean_squared_cost(source, target)
    if extra_cost is not None:
        extra_cost = extra_cost.to(device=cost_matrix.device, dtype=cost_matrix.dtype)
        if extra_cost.shape != cost_matrix.shape:
            raise ValueError(
                f"extra_cost shape must match {tuple(cost_matrix.shape)}, got {tuple(extra_cost.shape)}"
            )
        cost_matrix = cost_matrix + extra_cost

    permutation = solve_assignment(cost_matrix, solver=solver)
    matched = target.index_select(0, permutation)
    return matched, permutation, cost_matrix
