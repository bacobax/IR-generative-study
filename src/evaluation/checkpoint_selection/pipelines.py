"""Pipeline entry points for legacy and publication checkpoint selection."""

from __future__ import annotations

from typing import Any, Mapping

from scripts.select_best_checkpoint_and_compute_metrics import (
    run_clean_fid_publication_one,
    run_one,
)


def run_legacy_staged_pipeline(
    run_entry: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    cleanup_checkpoints: bool = False,
) -> dict[str, Any]:
    """Run the backward-compatible staged KID/FID checkpoint-selection pipeline."""
    return run_one(run_entry, config, cleanup_checkpoints=cleanup_checkpoints)


def run_clean_fid_publication_pipeline(
    run_entry: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    cleanup_checkpoints: bool = False,
) -> dict[str, Any]:
    """Run the publication-style Clean-FID/fallback checkpoint-selection pipeline."""
    return run_clean_fid_publication_one(
        run_entry,
        config,
        cleanup_checkpoints=cleanup_checkpoints,
    )


__all__ = [
    "run_clean_fid_publication_pipeline",
    "run_legacy_staged_pipeline",
]
