"""Maximum mean discrepancy helpers for generative evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_float64_features(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got shape {arr.shape}.")
    if arr.shape[0] < 2:
        raise ValueError("At least two feature rows are required.")
    return arr


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    dists = x_norm + y_norm - 2.0 * np.matmul(x, y.T)
    return np.maximum(dists, 0.0)


def _rbf_kernel_from_dists(sq_dists: np.ndarray, bandwidth: float) -> np.ndarray:
    if bandwidth <= 0.0:
        raise ValueError(f"RBF bandwidth must be positive, got {bandwidth}.")
    return np.exp(-sq_dists / (2.0 * float(bandwidth) ** 2))


def compute_rbf_mmd(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    *,
    bandwidths: Iterable[float],
) -> float:
    """Compute RBF-kernel MMD averaged over the requested bandwidths."""
    x = _as_float64_features(real_features)
    y = _as_float64_features(generated_features)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "Feature dimensions must match: "
            f"real={x.shape[1]}, generated={y.shape[1]}."
        )

    bw_values = [float(value) for value in bandwidths]
    if not bw_values:
        raise ValueError("At least one RBF bandwidth is required.")

    xx_dists = _pairwise_sq_dists(x, x)
    yy_dists = _pairwise_sq_dists(y, y)
    xy_dists = _pairwise_sq_dists(x, y)
    estimates = []
    for bandwidth in bw_values:
        kxx = _rbf_kernel_from_dists(xx_dists, bandwidth)
        kyy = _rbf_kernel_from_dists(yy_dists, bandwidth)
        kxy = _rbf_kernel_from_dists(xy_dists, bandwidth)

        xx_term = float(np.mean(kxx))
        yy_term = float(np.mean(kyy))
        xy_term = 2.0 * float(np.mean(kxy))
        estimates.append(xx_term + yy_term - xy_term)

    return float(np.mean(estimates))
