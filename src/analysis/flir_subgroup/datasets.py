"""Dataset registry for the subgroup analysis app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

from src.analysis.flir_subgroup.constants import DEFAULT_DATASET_ID
from src.core.paths import flir_root, v18_root


@dataclass(frozen=True)
class DatasetConfig:
    """Resolved configuration for one supported analysis dataset."""

    dataset_id: str
    label: str
    description: str
    root: Path
    is_default: bool = False

    def to_metadata(self) -> dict:
        """Return JSON-friendly metadata for dataset selection."""

        return {
            "dataset_id": self.dataset_id,
            "label": self.label,
            "description": self.description,
            "data_root": str(self.root),
            "is_default": self.is_default,
        }


def build_default_dataset_registry() -> Dict[str, DatasetConfig]:
    """Return the supported dataset registry."""

    registry = {
        "flir_private_proxy_alignment_v18": DatasetConfig(
            dataset_id="flir_private_proxy_alignment_v18",
            label="FLIR private proxy alignment v18",
            description="Multi-class FLIR proxy dataset used by the original notebook analysis.",
            root=flir_root(),
            is_default=True,
        ),
        "v18": DatasetConfig(
            dataset_id="v18",
            label="v18",
            description="Single-class dataset with the same FLIR-style structure and only the person category.",
            root=v18_root(),
            is_default=False,
        ),
    }
    return registry


DEFAULT_DATASET_REGISTRY = build_default_dataset_registry()


def resolve_dataset_config(
    dataset_id: str,
    *,
    registry: Mapping[str, DatasetConfig] | None = None,
) -> DatasetConfig:
    """Resolve one dataset id from the configured registry."""

    active_registry = registry or DEFAULT_DATASET_REGISTRY
    try:
        return active_registry[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset id: {dataset_id}") from exc


def list_dataset_metadata(
    *,
    registry: Mapping[str, DatasetConfig] | None = None,
) -> list[dict]:
    """Return chart-friendly dataset metadata records."""

    active_registry = registry or DEFAULT_DATASET_REGISTRY
    return [active_registry[key].to_metadata() for key in sorted(active_registry.keys(), key=lambda key: _dataset_sort_key(key, active_registry))]


def _dataset_sort_key(dataset_id: str, registry: Mapping[str, DatasetConfig]) -> tuple[int, str]:
    """Sort default dataset first, then alphabetically."""

    config = registry.get(dataset_id)
    return (0 if config and config.is_default else 1, dataset_id)


def dataset_ids(*, registry: Mapping[str, DatasetConfig] | None = None) -> Iterable[str]:
    """Return the available dataset ids."""

    active_registry = registry or DEFAULT_DATASET_REGISTRY
    return active_registry.keys()


def default_dataset_id(*, registry: Mapping[str, DatasetConfig] | None = None) -> str:
    """Return the configured default dataset id."""

    active_registry = registry or DEFAULT_DATASET_REGISTRY
    for dataset_id in active_registry:
        if active_registry[dataset_id].is_default:
            return dataset_id
    return DEFAULT_DATASET_ID
