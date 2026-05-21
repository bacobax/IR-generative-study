"""DomainStudio/CDC-style Intra-LPIPS diversity metric."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from src.core.normalization import UINT8_LINEAR
from src.evaluation.feature_extractors import load_image_rgb


@dataclass(frozen=True)
class IntraLPIPSResult:
    """JSON-friendly Intra-LPIPS result and diagnostics."""

    value: float
    backbone: str
    num_real: int
    num_generated: int
    num_assigned_clusters: int
    num_singleton_clusters: int

    def to_dict(self) -> dict[str, object]:
        return {
            "value": float(self.value),
            "backbone": self.backbone,
            "num_real": int(self.num_real),
            "num_generated": int(self.num_generated),
            "num_assigned_clusters": int(self.num_assigned_clusters),
            "num_singleton_clusters": int(self.num_singleton_clusters),
        }


def _image_to_lpips_tensor(image: Image.Image, *, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device=device, dtype=torch.float32)


def _load_lpips_batch(
    paths: Sequence[str | Path],
    *,
    normalization_mode: str,
    device: torch.device,
) -> torch.Tensor:
    tensors = [
        _image_to_lpips_tensor(
            load_image_rgb(path, normalization_mode=normalization_mode),
            device=device,
        )
        for path in paths
    ]
    if not tensors:
        raise ValueError("Cannot build an LPIPS batch from an empty path list.")
    return torch.cat(tensors, dim=0)


@torch.no_grad()
def _pairwise_lpips(
    model,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    for start in range(0, left.shape[0], batch_size):
        left_chunk = left[start : start + batch_size]
        row_chunks = []
        for right_start in range(0, right.shape[0], batch_size):
            right_chunk = right[right_start : right_start + batch_size]
            expanded_left = left_chunk[:, None].expand(-1, right_chunk.shape[0], -1, -1, -1)
            expanded_right = right_chunk[None].expand(left_chunk.shape[0], -1, -1, -1, -1)
            flat_left = expanded_left.reshape(-1, *left_chunk.shape[1:])
            flat_right = expanded_right.reshape(-1, *right_chunk.shape[1:])
            distances = model(flat_left, flat_right).reshape(left_chunk.shape[0], right_chunk.shape[0])
            row_chunks.append(distances.detach().cpu())
        rows.append(torch.cat(row_chunks, dim=1))
    return torch.cat(rows, dim=0)


def compute_intra_lpips(
    *,
    real_paths: Sequence[str | Path],
    generated_paths: Sequence[str | Path],
    backbone: str = "alex",
    device: str | torch.device = "cpu",
    batch_size: int = 16,
    real_normalization_mode: str = UINT8_LINEAR,
    generated_normalization_mode: str = UINT8_LINEAR,
) -> IntraLPIPSResult:
    """Compute DomainStudio/CDC-style Intra-LPIPS.

    Generated images are first assigned to the nearest real training image by
    LPIPS distance.  The metric is then the mean, over non-empty nearest-real
    clusters, of the average pairwise LPIPS distance inside that cluster.
    Singleton clusters contribute 0.0.
    """

    if not real_paths:
        raise ValueError("Intra-LPIPS requires at least one real image.")
    if not generated_paths:
        raise ValueError("Intra-LPIPS requires at least one generated image.")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"LPIPS batch_size must be positive, got {batch_size}.")

    try:
        import lpips
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The final Intra-LPIPS metric requires the 'lpips' package. "
            "Install it with the project requirements or disable compute_intra_lpips."
        ) from exc

    active_device = torch.device(device)
    model = lpips.LPIPS(net=str(backbone)).to(active_device).eval()

    real = _load_lpips_batch(
        real_paths,
        normalization_mode=real_normalization_mode,
        device=active_device,
    )
    generated = _load_lpips_batch(
        generated_paths,
        normalization_mode=generated_normalization_mode,
        device=active_device,
    )

    nearest_real_indices = []
    for gen_start in range(0, generated.shape[0], batch_size):
        gen_chunk = generated[gen_start : gen_start + batch_size]
        best_dist = torch.full((gen_chunk.shape[0],), float("inf"))
        best_idx = torch.zeros((gen_chunk.shape[0],), dtype=torch.long)
        for real_start in range(0, real.shape[0], batch_size):
            real_chunk = real[real_start : real_start + batch_size]
            distances = _pairwise_lpips(model, gen_chunk, real_chunk, batch_size=batch_size)
            values, indices = distances.min(dim=1)
            update = values < best_dist
            best_dist[update] = values[update]
            best_idx[update] = indices[update] + real_start
        nearest_real_indices.extend(int(idx) for idx in best_idx.tolist())

    clusters: dict[int, list[int]] = {}
    for gen_idx, real_idx in enumerate(nearest_real_indices):
        clusters.setdefault(real_idx, []).append(gen_idx)

    cluster_values = []
    singleton_count = 0
    for member_indices in clusters.values():
        if len(member_indices) < 2:
            singleton_count += 1
            cluster_values.append(0.0)
            continue
        cluster = generated[member_indices]
        distances = _pairwise_lpips(model, cluster, cluster, batch_size=batch_size)
        n = distances.shape[0]
        upper = torch.triu_indices(n, n, offset=1)
        cluster_values.append(float(distances[upper[0], upper[1]].mean().item()))

    return IntraLPIPSResult(
        value=float(np.mean(cluster_values)) if cluster_values else 0.0,
        backbone=str(backbone),
        num_real=len(real_paths),
        num_generated=len(generated_paths),
        num_assigned_clusters=len(clusters),
        num_singleton_clusters=singleton_count,
    )

