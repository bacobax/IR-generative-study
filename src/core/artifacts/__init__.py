"""Artifact loading extension interfaces."""

from src.core.artifacts.loaders import (
    ArtifactLoader,
    ArtifactLoadResult,
    FMUnetCheckpointLoader,
    RegionDiffGeneratorArtifactLoader,
    SDStage1PipelineLoader,
)

__all__ = [
    "ArtifactLoader",
    "ArtifactLoadResult",
    "FMUnetCheckpointLoader",
    "RegionDiffGeneratorArtifactLoader",
    "SDStage1PipelineLoader",
]
