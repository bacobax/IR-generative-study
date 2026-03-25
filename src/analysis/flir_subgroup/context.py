"""Cached analysis context for the subgroup analysis app."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

from src.analysis.flir_subgroup.analysis import PhaseAnalysisBundle, build_phase_bundle
from src.analysis.flir_subgroup.constants import ANALYSIS_SPLITS, POSITION_MODE
from src.analysis.flir_subgroup.datasets import (
    DEFAULT_DATASET_REGISTRY,
    DatasetConfig,
    default_dataset_id,
    resolve_dataset_config,
)
from src.analysis.flir_subgroup.data import inspect_dataset_root, load_dataset_tables


@dataclass
class FlirSubgroupAnalysisContext:
    """All memoized data needed by the backend API."""

    dataset_id: str
    dataset_label: str
    dataset_description: str
    data_root: Path
    analysis_splits: tuple[str, ...]
    dataset_layout_df: pd.DataFrame
    root_metadata_df: pd.DataFrame
    image_table: pd.DataFrame
    instance_table_raw: pd.DataFrame
    category_table: pd.DataFrame
    phases: Dict[str, PhaseAnalysisBundle]

    @property
    def dataset_summary(self) -> dict:
        """Return chart-friendly dataset summary metadata."""

        return {
            "dataset_id": self.dataset_id,
            "label": self.dataset_label,
            "description": self.dataset_description,
            "data_root": str(self.data_root),
            "analysis_splits": list(self.analysis_splits),
            "available_splits": self.dataset_layout_df["split"].tolist(),
            "n_images": int(self.image_table["image_key"].nunique()),
            "n_annotations": int(len(self.instance_table_raw)),
            "n_classes": int(self.instance_table_raw["class_label"].nunique()),
            "classes": sorted(self.instance_table_raw["class_label"].dropna().unique().tolist()),
            "n_missing_image_files": int((~self.image_table["image_exists"]).sum()),
        }

    def get_phase_bundle(self, phase: str) -> PhaseAnalysisBundle:
        """Return a phase bundle by name."""

        try:
            return self.phases[phase]
        except KeyError as exc:
            raise ValueError(f"Unknown phase: {phase}") from exc


def build_analysis_context(
    dataset_id: str | None = None,
    data_root: Path | None = None,
    *,
    analysis_splits: Sequence[str] = ANALYSIS_SPLITS,
    dataset_registry: Mapping[str, DatasetConfig] | None = None,
) -> FlirSubgroupAnalysisContext:
    """Build a full analysis context from a dataset root."""

    resolved_dataset_id, resolved_data_root, dataset_label, dataset_description = _resolve_context_target(
        dataset_id=dataset_id,
        data_root=data_root,
        dataset_registry=dataset_registry,
    )
    dataset_layout_df, root_metadata_df = inspect_dataset_root(resolved_data_root)
    image_table, instance_table_raw, category_table = load_dataset_tables(dataset_layout_df, analysis_splits)

    phase1_bundle = build_phase_bundle("phase1", image_table, instance_table_raw, include_position=False)
    phase2_bundle = build_phase_bundle(
        "phase2",
        image_table,
        instance_table_raw,
        include_position=True,
        position_mode=POSITION_MODE,
    )

    return FlirSubgroupAnalysisContext(
        dataset_id=resolved_dataset_id,
        dataset_label=dataset_label,
        dataset_description=dataset_description,
        data_root=resolved_data_root,
        analysis_splits=tuple(analysis_splits),
        dataset_layout_df=dataset_layout_df,
        root_metadata_df=root_metadata_df,
        image_table=image_table,
        instance_table_raw=instance_table_raw,
        category_table=category_table,
        phases={"phase1": phase1_bundle, "phase2": phase2_bundle},
    )


@lru_cache(maxsize=8)
def _build_cached_context(
    dataset_id: str | None,
    data_root_str: str | None,
    analysis_splits: tuple[str, ...],
) -> FlirSubgroupAnalysisContext:
    data_root = Path(data_root_str) if data_root_str is not None else None
    return build_analysis_context(dataset_id=dataset_id, data_root=data_root, analysis_splits=analysis_splits)


def get_analysis_context(
    dataset_id: str | None = None,
    data_root: str | Path | None = None,
    *,
    analysis_splits: Sequence[str] = ANALYSIS_SPLITS,
) -> FlirSubgroupAnalysisContext:
    """Return a cached analysis context."""

    data_root_str: Optional[str]
    if data_root is None:
        data_root_str = None
    else:
        data_root_str = str(Path(data_root).resolve())
    return _build_cached_context(dataset_id, data_root_str, tuple(analysis_splits))


def clear_analysis_context_cache() -> None:
    """Clear cached contexts. Mostly useful for tests."""

    _build_cached_context.cache_clear()


def _resolve_context_target(
    *,
    dataset_id: str | None,
    data_root: Path | None,
    dataset_registry: Mapping[str, DatasetConfig] | None,
) -> tuple[str, Path, str, str]:
    """Resolve dataset metadata for a context build."""

    if data_root is not None:
        custom_id = dataset_id or "custom"
        return (
            custom_id,
            data_root.resolve(),
            "Custom dataset",
            "Dataset root provided directly to the analysis context.",
        )

    active_registry = dataset_registry or DEFAULT_DATASET_REGISTRY
    resolved_dataset = resolve_dataset_config(dataset_id or default_dataset_id(registry=active_registry), registry=active_registry)
    return (
        resolved_dataset.dataset_id,
        resolved_dataset.root.resolve(),
        resolved_dataset.label,
        resolved_dataset.description,
    )
