"""Task adapter contracts for future trainer orchestration."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TaskAdapter(Protocol):
    """Minimal contract for task-level orchestration."""

    def build_model(self, config: Any, **kwargs: Any) -> Any:
        """Build the model or model bundle for a task."""

    def build_dataloaders(self, config: Any, **kwargs: Any) -> Any:
        """Build dataloaders or a dataset bundle for a task."""

    def build_trainer(self, config: Any, model: Any, dataloaders: Any, **kwargs: Any) -> Any:
        """Build the trainer object for a task."""
