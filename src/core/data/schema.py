"""Canonical sample and batch schema for repo data pipelines.

The canonical wire format deliberately matches the dict keys already consumed
by current trainers. These dataclasses document the shape without forcing
callers to stop using plain dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class SampleKeys:
    PIXEL_VALUES = "pixel_values"
    TEXT = "text"
    INPUT_IDS = "input_ids"
    ATTENTION_MASK = "attention_mask"
    BOXES_XYXY = "boxes_xyxy"
    BOXES_XYXY_NORM = "boxes_xyxy_norm"
    LABELS = "labels"
    LABEL_NAMES = "label_names"
    OBJECT_MASK = "object_mask"
    N_OBJECTS = "n_objects"
    METADATA = "metadata"


@dataclass(frozen=True)
class TextFields:
    text: str | None = None
    prompt_text: str | None = None
    caption_text: str | None = None


@dataclass(frozen=True)
class TokenizerFields:
    input_ids: Any = None
    attention_mask: Any = None


@dataclass(frozen=True)
class LayoutFields:
    boxes_xyxy: Any = None
    boxes_xyxy_norm: Any = None
    labels: Any = None
    label_names: Any = None
    object_mask: Any = None
    n_objects: Any = None


@dataclass(frozen=True)
class CanonicalSample:
    pixel_values: Any
    text: str | None = None
    input_ids: Any = None
    attention_mask: Any = None
    boxes_xyxy: Any = None
    boxes_xyxy_norm: Any = None
    labels: Any = None
    label_names: Any = None
    object_mask: Any = None
    n_objects: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            SampleKeys.PIXEL_VALUES: self.pixel_values,
            SampleKeys.METADATA: dict(self.metadata),
        }
        for key in (
            SampleKeys.TEXT,
            SampleKeys.INPUT_IDS,
            SampleKeys.ATTENTION_MASK,
            SampleKeys.BOXES_XYXY,
            SampleKeys.BOXES_XYXY_NORM,
            SampleKeys.LABELS,
            SampleKeys.LABEL_NAMES,
            SampleKeys.OBJECT_MASK,
            SampleKeys.N_OBJECTS,
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass(frozen=True)
class CanonicalBatch:
    pixel_values: Any
    text: Any = None
    input_ids: Any = None
    attention_mask: Any = None
    boxes_xyxy: Any = None
    boxes_xyxy_norm: Any = None
    labels: Any = None
    label_names: Any = None
    object_mask: Any = None
    n_objects: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            SampleKeys.PIXEL_VALUES: self.pixel_values,
            SampleKeys.METADATA: dict(self.metadata),
        }
        for key in (
            SampleKeys.TEXT,
            SampleKeys.INPUT_IDS,
            SampleKeys.ATTENTION_MASK,
            SampleKeys.BOXES_XYXY,
            SampleKeys.BOXES_XYXY_NORM,
            SampleKeys.LABELS,
            SampleKeys.LABEL_NAMES,
            SampleKeys.OBJECT_MASK,
            SampleKeys.N_OBJECTS,
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data
