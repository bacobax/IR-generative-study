"""Small model adapter contracts.

These interfaces are intentionally additive. They describe how future model
construction can be routed without changing the current trainers, CLIs, or
``REGISTRIES.model_builder`` behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelBuildRequest:
    """Inputs needed to build or wrap a model component."""

    name: str | None = None
    config: Mapping[str, Any] | None = None
    device: Any = None
    dtype: Any = None
    components: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelBundle:
    """Built model plus optional named components and provenance."""

    primary: Any
    components: Mapping[str, Any] = field(default_factory=dict)
    adapter_name: str | None = None
    model_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def component(self, name: str, default: Any = None) -> Any:
        """Return a named component from the bundle."""
        return self.components.get(name, default)


@dataclass(frozen=True)
class ArtifactLoadRequest:
    """Inputs needed to load a model or artifact from disk."""

    path: str | Path
    kind: str | None = None
    device: Any = None
    strict: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for model builders or wrappers."""

    def build(self, request: ModelBuildRequest) -> ModelBundle:
        """Build a model bundle from a structured request."""


class ModelAdapterBase:
    """Concrete defaults for optional adapter behavior."""

    def load_artifact(self, request: ArtifactLoadRequest) -> ModelBundle:
        raise NotImplementedError(f"{self.__class__.__name__} does not load artifacts")

    def save_artifact(self, bundle: ModelBundle, path: str | Path, **kwargs: Any) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not save artifacts")

    def trainable_modules(self, bundle: ModelBundle) -> Mapping[str, Any]:
        return {
            name: module
            for name, module in bundle.components.items()
            if hasattr(module, "parameters")
            and any(getattr(param, "requires_grad", False) for param in module.parameters())
        }
