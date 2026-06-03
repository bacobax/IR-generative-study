"""Feature extraction/cache compatibility helpers for checkpoint selection."""

from __future__ import annotations

from scripts.select_best_checkpoint_and_compute_metrics import _features_for_paths
from src.evaluation.feature_extractors import build_feature_extractor, extract_features

__all__ = [
    "_features_for_paths",
    "build_feature_extractor",
    "extract_features",
]
