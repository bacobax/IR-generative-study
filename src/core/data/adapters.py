"""Dataset adapter contracts and canonicalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from src.core.data.schema import CanonicalBatch, CanonicalSample, SampleKeys
from src.core.registry import REGISTRIES


METADATA_ALIAS_KEYS = (
    "image_id",
    "file_name",
    "label_names",
    "prompt_text",
    "caption_text",
    "split",
    "dataset_id",
    "width",
    "height",
    "original_width",
    "original_height",
)


@dataclass(frozen=True)
class DatasetBuildRequest:
    """Inputs needed to build a dataset or dataloader bundle."""

    name: str | None = None
    dataset_id: str | None = None
    split: str = "train"
    root_dir: str | Path | None = None
    annotations_path: str | Path | None = None
    tokenizer: Any = None
    transforms: Any = None
    batch_size: int | None = None
    num_workers: int = 0
    config: Any = None
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetBundle:
    """Built dataset/dataloader pair plus provenance."""

    dataset: Any = None
    dataloader: Any = None
    collate_fn: Callable[..., Any] | None = None
    normalization_mode: str | None = None
    adapter_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Protocol for dataset builders."""

    def build(self, request: DatasetBuildRequest) -> DatasetBundle:
        """Build a dataset bundle from a structured request."""


def _metadata_with_aliases(sample: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(sample.get(SampleKeys.METADATA, {}) or {})
    for key in METADATA_ALIAS_KEYS:
        if key in sample and key not in metadata:
            metadata[key] = sample[key]
    return metadata


def normalize_sample(sample: Any) -> dict[str, Any]:
    """Return a canonical dict while preserving existing top-level keys."""
    if not isinstance(sample, Mapping):
        return {SampleKeys.PIXEL_VALUES: sample, SampleKeys.METADATA: {}}
    normalized = dict(sample)
    normalized[SampleKeys.METADATA] = _metadata_with_aliases(normalized)
    if SampleKeys.TEXT not in normalized and "prompt_text" in normalized:
        normalized[SampleKeys.TEXT] = normalized["prompt_text"]
    return normalized


def normalize_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical batch dict while preserving existing top-level keys."""
    normalized = dict(batch)
    normalized[SampleKeys.METADATA] = _metadata_with_aliases(normalized)
    if SampleKeys.TEXT not in normalized and "prompt_text" in normalized:
        normalized[SampleKeys.TEXT] = normalized["prompt_text"]
    return normalized


def canonical_sample_from_mapping(sample: Any) -> CanonicalSample:
    """Convert a legacy sample mapping or image tensor into ``CanonicalSample``."""
    data = normalize_sample(sample)
    if SampleKeys.PIXEL_VALUES not in data:
        raise KeyError("CanonicalSample requires 'pixel_values'")
    return CanonicalSample(
        pixel_values=data[SampleKeys.PIXEL_VALUES],
        text=data.get(SampleKeys.TEXT),
        input_ids=data.get(SampleKeys.INPUT_IDS),
        attention_mask=data.get(SampleKeys.ATTENTION_MASK),
        boxes_xyxy=data.get(SampleKeys.BOXES_XYXY),
        boxes_xyxy_norm=data.get(SampleKeys.BOXES_XYXY_NORM),
        labels=data.get(SampleKeys.LABELS),
        label_names=data.get(SampleKeys.LABEL_NAMES),
        object_mask=data.get(SampleKeys.OBJECT_MASK),
        n_objects=data.get(SampleKeys.N_OBJECTS),
        metadata=data.get(SampleKeys.METADATA, {}),
    )


def canonical_batch_from_mapping(batch: Mapping[str, Any]) -> CanonicalBatch:
    """Convert a legacy batch mapping into ``CanonicalBatch``."""
    data = normalize_batch(batch)
    if SampleKeys.PIXEL_VALUES not in data:
        raise KeyError("CanonicalBatch requires 'pixel_values'")
    return CanonicalBatch(
        pixel_values=data[SampleKeys.PIXEL_VALUES],
        text=data.get(SampleKeys.TEXT),
        input_ids=data.get(SampleKeys.INPUT_IDS),
        attention_mask=data.get(SampleKeys.ATTENTION_MASK),
        boxes_xyxy=data.get(SampleKeys.BOXES_XYXY),
        boxes_xyxy_norm=data.get(SampleKeys.BOXES_XYXY_NORM),
        labels=data.get(SampleKeys.LABELS),
        label_names=data.get(SampleKeys.LABEL_NAMES),
        object_mask=data.get(SampleKeys.OBJECT_MASK),
        n_objects=data.get(SampleKeys.N_OBJECTS),
        metadata=data.get(SampleKeys.METADATA, {}),
    )


def _register_once(registry, name: str, value: Any) -> None:
    if name not in registry:
        registry.register(name)(value)


class RepoDatasetTargetAdapter:
    """Resolve a repo-native dataset target without building a trainer dataset."""

    def build(self, request: DatasetBuildRequest) -> DatasetBundle:
        from src.core.data.dataset_targets import resolve_dataset_target

        dataset_id = request.dataset_id or request.name
        if not dataset_id:
            raise ValueError("repo_dataset_target adapter requires dataset_id or name")
        target = resolve_dataset_target(str(dataset_id))
        metadata = dict(request.metadata or {})
        metadata.update(
            {
                "dataset_id": target.dataset_id,
                "root": str(target.root),
                "split": request.split,
                "split_dir": str(target.split_dir(request.split)),
                "annotations_path": str(target.annotations_path(request.split)),
            }
        )
        return DatasetBundle(
            dataset=target,
            normalization_mode=target.normalization_mode,
            adapter_name="repo_dataset_target",
            metadata=metadata,
        )


class AnnotationLayoutDatasetAdapter:
    """Build the existing annotation layout dataset lazily."""

    def build(self, request: DatasetBuildRequest) -> DatasetBundle:
        from src.core.data.datasets import AnnotationLayoutDataset

        options = dict(request.options or {})
        root_dir = request.root_dir or options.pop("root_dir", None)
        annotations_path = request.annotations_path or options.pop("annotations_path", None)
        if root_dir is None or annotations_path is None:
            if not request.dataset_id:
                raise ValueError(
                    "annotation_layout adapter requires root_dir/annotations_path "
                    "or dataset_id"
                )
            target_bundle = RepoDatasetTargetAdapter().build(request)
            root_dir = target_bundle.metadata["split_dir"]
            annotations_path = target_bundle.metadata["annotations_path"]
            normalization_mode = target_bundle.normalization_mode
        else:
            normalization_mode = options.pop("normalization_mode", None)
        dataset = AnnotationLayoutDataset(
            root_dir=str(root_dir),
            annotations_path=str(annotations_path),
            image_size=int(options.pop("image_size", 256)),
            normalization_mode=normalization_mode or options.pop("normalization_mode", "raw_uint16_percentile"),
            include_label_names=bool(options.pop("include_label_names", True)),
            horizontal_flip_schedule=options.pop("horizontal_flip_schedule", None),
        )
        return DatasetBundle(
            dataset=dataset,
            normalization_mode=getattr(dataset, "normalization_mode", normalization_mode),
            adapter_name="annotation_layout",
            metadata=dict(request.metadata or {}),
        )


class TextImageDatasetAdapter:
    """Build the existing local text-image dataset lazily."""

    def build(self, request: DatasetBuildRequest) -> DatasetBundle:
        from src.core.data.datasets import TextImageDataset

        options = dict(request.options or {})
        root_dir = request.root_dir or options.pop("root_dir", None)
        if root_dir is None:
            raise ValueError("text_image adapter requires root_dir")
        dataset = TextImageDataset(
            root_dir=str(root_dir),
            text_annotations=options.pop("text_annotations", None),
            fallback_text=options.pop("fallback_text", None),
            transform=request.transforms or options.pop("transform", None),
        )
        return DatasetBundle(dataset=dataset, adapter_name="text_image", metadata=dict(request.metadata or {}))


class HFTextImageDatasetAdapter:
    """Build an existing Hugging Face/local text-image dataset through current helpers."""

    def build(self, request: DatasetBuildRequest) -> DatasetBundle:
        from src.algorithms.stable_diffusion.data import load_training_dataset

        options = dict(request.options or {})
        dataset, image_column, caption_column = load_training_dataset(
            dataset_name=request.name or options.get("dataset_name"),
            dataset_config_name=options.get("dataset_config_name"),
            train_data_dir=str(request.root_dir) if request.root_dir is not None else options.get("train_data_dir"),
            cache_dir=options.get("cache_dir"),
            image_column=options.get("image_column", "image"),
            caption_column=options.get("caption_column", "text"),
        )
        metadata = dict(request.metadata or {})
        metadata.update({"image_column": image_column, "caption_column": caption_column})
        return DatasetBundle(dataset=dataset, adapter_name="hf_text_image", metadata=metadata)


_register_once(REGISTRIES.dataset_adapter, "repo_dataset_target", RepoDatasetTargetAdapter())
_register_once(REGISTRIES.dataset_adapter, "annotation_layout", AnnotationLayoutDatasetAdapter())
_register_once(REGISTRIES.dataset_adapter, "text_image", TextImageDatasetAdapter())
_register_once(REGISTRIES.dataset_adapter, "hf_text_image", HFTextImageDatasetAdapter())
