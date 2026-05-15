"""Artifact loader contracts and default lightweight loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from src.core.registry import REGISTRIES
from src.models.adapters.base import ArtifactLoadRequest


@dataclass(frozen=True)
class ArtifactLoadResult:
    """Structured artifact load result."""

    kind: str
    path: Path
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ArtifactLoader(Protocol):
    """Protocol for loading a named artifact family."""

    def load(self, request: ArtifactLoadRequest) -> ArtifactLoadResult:
        """Load or resolve an artifact from disk."""


def _register_once(registry, name: str, value: Any) -> None:
    if name not in registry:
        registry.register(name)(value)


class FMUnetCheckpointLoader:
    """Load current FM UNET checkpoint payloads."""

    def load(self, request: ArtifactLoadRequest) -> ArtifactLoadResult:
        import torch

        path = Path(request.path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing FM UNET checkpoint: {path}")
        state = torch.load(path, map_location=request.device or "cpu")
        metadata: dict[str, Any] = {"raw_type": type(state).__name__}
        if isinstance(state, dict):
            metadata["raw_keys"] = sorted(str(key) for key in state.keys())
            if "unet_state" in state:
                return ArtifactLoadResult(
                    kind="fm_unet_checkpoint",
                    path=path,
                    payload=state["unet_state"],
                    metadata={**metadata, "payload_key": "unet_state"},
                )
        return ArtifactLoadResult(
            kind="fm_unet_checkpoint",
            path=path,
            payload=state,
            metadata=metadata,
        )


class SDStage1PipelineLoader:
    """Load existing Stage-1 Stable Diffusion artifacts."""

    def load(self, request: ArtifactLoadRequest) -> ArtifactLoadResult:
        path = Path(request.path)
        if not path.exists():
            raise FileNotFoundError(f"Missing SD stage-1 artifact: {path}")
        from src.algorithms.stable_diffusion.models import load_stage1_pipeline

        pipeline, manifest = load_stage1_pipeline(
            stage1_dir=str(path),
            base_model=request.options.get("base_model"),
            torch_dtype=request.options.get("torch_dtype"),
        )
        return ArtifactLoadResult(
            kind="sd_stage1_pipeline",
            path=path,
            payload=pipeline,
            metadata={"manifest": manifest},
        )


class RegionDiffGeneratorArtifactLoader:
    """Resolve and cheaply validate RegionDiff generator artifacts."""

    def load(self, request: ArtifactLoadRequest) -> ArtifactLoadResult:
        path = Path(request.path)
        if not path.exists():
            raise FileNotFoundError(f"Missing RegionDiff generator artifact: {path}")
        from src.algorithms.inference.regiondiff.backend_loaders import (
            validate_generator_checkpoint_readability,
        )

        ok, detail = validate_generator_checkpoint_readability(path)
        if not ok:
            raise ValueError(detail)
        return ArtifactLoadResult(
            kind="regiondiff_generator",
            path=path,
            payload=detail,
            metadata={"readability_detail": detail},
        )


_register_once(REGISTRIES.artifact_loader, "fm_unet_checkpoint", FMUnetCheckpointLoader())
_register_once(REGISTRIES.artifact_loader, "sd_stage1_pipeline", SDStage1PipelineLoader())
_register_once(REGISTRIES.artifact_loader, "regiondiff_generator", RegionDiffGeneratorArtifactLoader())


__all__ = [
    "ArtifactLoadResult",
    "ArtifactLoader",
    "FMUnetCheckpointLoader",
    "RegionDiffGeneratorArtifactLoader",
    "SDStage1PipelineLoader",
]
