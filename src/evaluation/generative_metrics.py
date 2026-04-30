"""FID, KID, and summary helpers for generative image evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def _as_float64_features(features: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} features must be 2D, got shape {arr.shape}.")
    if arr.shape[0] < 2:
        raise ValueError(f"{name} features need at least two rows.")
    return arr


def _covariance(features: np.ndarray) -> np.ndarray:
    cov = np.cov(features, rowvar=False)
    cov = np.atleast_2d(cov).astype(np.float64, copy=False)
    return cov


def _sqrtm_with_optional_scipy(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root with a NumPy fallback for SciPy-light environments."""
    try:
        from scipy import linalg
    except Exception:
        return _sqrtm_via_eigendecomposition(matrix)

    try:
        result = linalg.sqrtm(matrix, disp=False)
    except TypeError:
        result = linalg.sqrtm(matrix)
    if isinstance(result, tuple):
        return np.asarray(result[0])
    return np.asarray(result)


def _sqrtm_via_eigendecomposition(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=np.float64))
    values = values.astype(np.complex128, copy=False)
    sqrt_values = np.sqrt(values)
    return (vectors * sqrt_values) @ np.linalg.pinv(vectors)


def compute_fid(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet distance between two feature distributions."""
    real = _as_float64_features(real_features, name="real")
    generated = _as_float64_features(generated_features, name="generated")
    if real.shape[1] != generated.shape[1]:
        raise ValueError(
            "Feature dimensions must match: "
            f"real={real.shape[1]}, generated={generated.shape[1]}."
        )

    mu_real = np.mean(real, axis=0)
    mu_generated = np.mean(generated, axis=0)
    sigma_real = _covariance(real)
    sigma_generated = _covariance(generated)

    offset = mu_real - mu_generated
    covmean = _sqrtm_with_optional_scipy(sigma_real @ sigma_generated)
    if not np.isfinite(covmean).all():
        eye = np.eye(sigma_real.shape[0], dtype=np.float64)
        covmean = _sqrtm_with_optional_scipy(
            (sigma_real + eps * eye) @ (sigma_generated + eps * eye)
        )

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = (
        float(offset.dot(offset))
        + float(np.trace(sigma_real))
        + float(np.trace(sigma_generated))
        - 2.0 * float(np.trace(covmean))
    )
    return float(max(fid, 0.0))


def _polynomial_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dim = float(x.shape[1])
    return (np.matmul(x, y.T) / dim + 1.0) ** 3


def _unbiased_mmd2_from_kernel(kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray) -> float:
    n = kxx.shape[0]
    m = kyy.shape[0]
    xx = (np.sum(kxx) - np.trace(kxx)) / (n * (n - 1))
    yy = (np.sum(kyy) - np.trace(kyy)) / (m * (m - 1))
    xy = float(np.mean(kxy))
    return float(xx + yy - 2.0 * xy)


def compute_kid(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    *,
    subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 42,
) -> float:
    """Compute KID as deterministic subset-averaged polynomial MMD."""
    real = _as_float64_features(real_features, name="real")
    generated = _as_float64_features(generated_features, name="generated")
    if real.shape[1] != generated.shape[1]:
        raise ValueError(
            "Feature dimensions must match: "
            f"real={real.shape[1]}, generated={generated.shape[1]}."
        )

    subsets = int(subsets)
    subset_size = int(subset_size)
    if subsets <= 0:
        raise ValueError("KID subsets must be positive.")
    if subset_size < 2:
        raise ValueError("KID subset_size must be at least 2.")

    active_size = min(subset_size, real.shape[0], generated.shape[0])
    if active_size < 2:
        raise ValueError("KID requires at least two samples per set.")

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(subsets):
        real_idx = rng.choice(real.shape[0], size=active_size, replace=False)
        gen_idx = rng.choice(generated.shape[0], size=active_size, replace=False)
        x = real[real_idx]
        y = generated[gen_idx]
        estimates.append(
            _unbiased_mmd2_from_kernel(
                _polynomial_kernel(x, x),
                _polynomial_kernel(y, y),
                _polynomial_kernel(x, y),
            )
        )
    return float(np.mean(estimates))


def metrics_row_to_jsonable(row: Mapping[str, object]) -> Dict[str, object]:
    """Convert numpy scalars in one metric row to plain Python values."""
    converted: Dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            converted[key] = value.item()
        else:
            converted[key] = value
    return converted


def sort_metric_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    ranking_keys: Iterable[str],
) -> List[Dict[str, object]]:
    """Sort metric rows by lower-is-better ranking keys."""
    keys = list(ranking_keys)
    if not keys:
        raise ValueError("At least one ranking key is required.")

    def sort_key(row: Mapping[str, object]) -> tuple:
        return tuple(float(row[key]) for key in keys)

    return [metrics_row_to_jsonable(row) for row in sorted(rows, key=sort_key)]
