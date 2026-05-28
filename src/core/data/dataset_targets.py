"""Named dataset targets for config-driven FM data selection.

This module keeps dataset identity separate from raw path strings so the FM
training CLI can select first-class datasets by name while reusing the existing
folder layout and loader plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

from src.core.data.adapters import RepoDatasetAdapter
from src.core.normalization import (
    RAW_UINT16_PERCENTILE,
    SENTINEL2_REFLECTANCE,
    UINT8_LINEAR,
)
from src.core.paths import bigearthnet_s2_b08_5x5_stride3_root, flir_root, v18_root
from src.core.registry import REGISTRIES


@dataclass(frozen=True)
class DatasetTarget:
    """Resolved on-disk dataset target used by the FM pipeline."""

    dataset_id: str
    root: Path
    normalization_mode: str
    adapter: Any | None = None

    def split_dir(self, split: str) -> Path:
        """Return the directory for one dataset split."""
        if self.adapter is not None:
            return self.adapter.split_dir(split)
        return self.root / split

    def annotations_path(self, split: str) -> Path:
        """Return the canonical COCO annotations file for one split."""
        if self.adapter is not None:
            return self.adapter.annotations_path(split)
        return self.split_dir(split) / "annotations.json"

    def manifest_path(self, split: str) -> Path | None:
        """Return a split manifest path when the target has one."""
        if self.adapter is not None and hasattr(self.adapter, "manifest_path"):
            return self.adapter.manifest_path(split)
        return None

    def has_coco_annotations(self) -> bool:
        """Return whether this target provides COCO layout annotations."""
        if self.adapter is not None and hasattr(self.adapter, "has_coco_annotations"):
            return bool(self.adapter.has_coco_annotations())
        return True

    def category_metadata(self, split: str) -> dict[int, str]:
        """Return split category metadata when annotations are available."""
        if self.adapter is not None:
            return self.adapter.category_metadata(split)
        annotations_path = self.annotations_path(split)
        if not annotations_path.is_file():
            return {}
        from src.core.data.annotations import build_category_id_to_name, load_coco_annotations

        return build_category_id_to_name(load_coco_annotations(annotations_path))

    def collate_fn_for_task(self, task: str | None) -> Callable[..., Any] | None:
        """Return the task-specific collate function, when one is defined."""
        if self.adapter is not None:
            return self.adapter.collate_fn_for_task(task)
        return None

    def build_request(self, *, split: str = "train", **options):
        """Build a dataset adapter request for this target."""
        return target_to_dataset_build_request(self, split=split, **options)


def _build_repo_dataset_target(
    *,
    dataset_id: str,
    root: Path,
    normalization_mode: str,
) -> DatasetTarget:
    adapter = RepoDatasetAdapter(
        dataset_id=dataset_id,
        root=root,
        normalization_mode=normalization_mode,
    )
    return DatasetTarget(
        dataset_id=dataset_id,
        root=root,
        normalization_mode=normalization_mode,
        adapter=adapter,
    )


class BigEarthNetS2B08DatasetAdapter:
    """Adapter for manifest-backed BigEarthNet Sentinel-2 B08 TIFF mosaics."""

    dataset_id = "bigearthnet_s2_b08_5x5_stride3"
    normalization_mode = SENTINEL2_REFLECTANCE

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def _canonical_split(split: str) -> str:
        normalized = str(split or "train")
        if normalized == "val":
            return "validation"
        return normalized

    def split_dir(self, split: str) -> Path:
        return self.root / "images" / self._canonical_split(split)

    def annotations_path(self, split: str) -> Path:
        return self.split_dir(split) / "annotations.json"

    def manifest_path(self, split: str) -> Path:
        return self.root / "manifests" / f"{self._canonical_split(split)}.jsonl"

    def category_metadata(self, split: str) -> dict[int, str]:
        del split
        return {}

    def collate_fn_for_task(self, task: str | None) -> Callable[..., Any] | None:
        del task
        return None

    def has_coco_annotations(self) -> bool:
        return False

    def build(self, request):
        from src.core.data.adapters import DatasetBundle

        split = str(request.split or "train")
        split_dir = self.split_dir(split)
        manifest_path = self.manifest_path(split)
        metadata = dict(request.metadata or {})
        metadata.update(
            {
                "dataset_id": self.dataset_id,
                "root": str(self.root),
                "split": split,
                "resolved_split": self._canonical_split(split),
                "split_dir": str(split_dir),
                "annotations_path": str(self.annotations_path(split)),
                "manifest_path": str(manifest_path),
                "normalization_mode": self.normalization_mode,
                "category_id_to_name": {},
            }
        )
        return DatasetBundle(
            collate_fn=None,
            normalization_mode=self.normalization_mode,
            adapter_name=self.dataset_id,
            metadata=metadata,
        )


def build_default_dataset_targets() -> Dict[str, DatasetTarget]:
    """Return the supported named dataset targets."""
    bigearthnet_adapter = BigEarthNetS2B08DatasetAdapter(
        bigearthnet_s2_b08_5x5_stride3_root()
    )
    return {
        "v18": _build_repo_dataset_target(
            dataset_id="v18",
            root=v18_root(),
            normalization_mode=RAW_UINT16_PERCENTILE,
        ),
        "flir_private_proxy_alignment_v18": _build_repo_dataset_target(
            dataset_id="flir_private_proxy_alignment_v18",
            root=flir_root(),
            normalization_mode=UINT8_LINEAR,
        ),
        "bigearthnet_s2_b08_5x5_stride3": DatasetTarget(
            dataset_id="bigearthnet_s2_b08_5x5_stride3",
            root=bigearthnet_adapter.root,
            normalization_mode=SENTINEL2_REFLECTANCE,
            adapter=bigearthnet_adapter,
        ),
    }


DEFAULT_DATASET_TARGETS = build_default_dataset_targets()


def _register_dataset_adapters(registry: Mapping[str, DatasetTarget]) -> None:
    for dataset_id, target in registry.items():
        if target.adapter is not None and dataset_id not in REGISTRIES.dataset_adapter:
            REGISTRIES.dataset_adapter.register(dataset_id)(target.adapter)


_register_dataset_adapters(DEFAULT_DATASET_TARGETS)


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


def target_to_dataset_build_request(
    target: DatasetTarget,
    *,
    split: str = "train",
    **options,
):
    """Build a dataset adapter request from a resolved dataset target."""
    from src.core.data.adapters import DatasetBuildRequest

    return DatasetBuildRequest(
        name=target.dataset_id,
        dataset_id=target.dataset_id,
        split=split,
        root_dir=target.split_dir(split),
        annotations_path=target.annotations_path(split),
        options={"normalization_mode": target.normalization_mode, **options},
        metadata={
            "dataset_id": target.dataset_id,
            "root": str(target.root),
            "split": split,
            "split_dir": str(target.split_dir(split)),
            "annotations_path": str(target.annotations_path(split)),
            "manifest_path": (
                str(target.manifest_path(split))
                if target.manifest_path(split) is not None
                else None
            ),
            "normalization_mode": target.normalization_mode,
            "category_id_to_name": target.category_metadata(split),
        },
    )
