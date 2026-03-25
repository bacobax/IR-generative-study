"""Named dataset targets for config-driven FM data selection.

This module keeps dataset identity separate from raw path strings so the FM
training CLI can select first-class datasets by name while reusing the existing
folder layout and loader plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

from src.core.normalization import RAW_UINT16_PERCENTILE, UINT8_LINEAR
from src.core.paths import flir_root, v18_root


@dataclass(frozen=True)
class DatasetTarget:
    """Resolved on-disk dataset target used by the FM pipeline."""

    dataset_id: str
    root: Path
    normalization_mode: str

    def split_dir(self, split: str) -> Path:
        """Return the directory for one dataset split."""
        return self.root / split

    def annotations_path(self, split: str) -> Path:
        """Return the canonical COCO annotations file for one split."""
        return self.split_dir(split) / "annotations.json"


def build_default_dataset_targets() -> Dict[str, DatasetTarget]:
    """Return the supported named dataset targets."""
    return {
        "v18": DatasetTarget(
            dataset_id="v18",
            root=v18_root(),
            normalization_mode=RAW_UINT16_PERCENTILE,
        ),
        "flir_private_proxy_alignment_v18": DatasetTarget(
            dataset_id="flir_private_proxy_alignment_v18",
            root=flir_root(),
            normalization_mode=UINT8_LINEAR,
        ),
    }


DEFAULT_DATASET_TARGETS = build_default_dataset_targets()


def resolve_dataset_target(
    dataset_id: str,
    *,
    registry: Mapping[str, DatasetTarget] | None = None,
) -> DatasetTarget:
    """Resolve one named dataset target."""
    active_registry = registry or DEFAULT_DATASET_TARGETS
    try:
        return active_registry[dataset_id]
    except KeyError as exc:
        available = ", ".join(sorted(active_registry))
        raise ValueError(
            f"Unknown dataset_id={dataset_id!r}. Available: {available}"
        ) from exc


def supported_dataset_ids(
    *,
    registry: Mapping[str, DatasetTarget] | None = None,
) -> Iterable[str]:
    """Return the supported dataset ids."""
    active_registry = registry or DEFAULT_DATASET_TARGETS
    return active_registry.keys()
