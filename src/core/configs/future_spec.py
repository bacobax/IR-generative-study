"""Experimental future-facing config shape.

These dataclasses are not wired into training yet. They provide a strict,
versioned shape for new presets while legacy YAML continues through the
existing permissive config path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelSpec:
    kind: str
    family: str
    base_model: Optional[str] = None
    components: dict[str, Any] = field(default_factory=dict)
    adapters: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DatasetSpec:
    kind: str
    dataset_id: Optional[str] = None
    image_size: Optional[int] = None
    train_split: Optional[str] = None
    val_split: Optional[str] = None
    annotations: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformSpec:
    kind: str
    normalization: Optional[str] = None
    horizontal_flip: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    kind: str
    target: Optional[str] = None
    t_scale: Optional[float] = None
    conditioning: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeSpec:
    kind: str
    device: Optional[str] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    mixed_precision: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactSpec:
    kind: str
    output_dir: str
    manifest_name: str = "artifact_manifest.json"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentalConfigSpec:
    kind: str
    schema_version: int
    model: ModelSpec
    dataset: DatasetSpec
    transform: TransformSpec
    task: TaskSpec
    runtime: RuntimeSpec
    artifact: ArtifactSpec


__all__ = [
    "ArtifactSpec",
    "DatasetSpec",
    "ExperimentalConfigSpec",
    "ModelSpec",
    "RuntimeSpec",
    "TaskSpec",
    "TransformSpec",
]
