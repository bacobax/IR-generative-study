"""Artifact loading extension interfaces."""

from src.core.artifacts.manifest import (
    ARTIFACT_MANIFEST_NAME,
    ArtifactManifest,
    read_artifact_manifest,
    validate_manifest_compatibility,
    write_artifact_manifest,
)
from src.core.artifacts.loaders import (
    ArtifactLoader,
    ArtifactLoadResult,
    FMUnetCheckpointLoader,
    RegionDiffGeneratorArtifactLoader,
    SDStage1PipelineLoader,
)

__all__ = [
    "ARTIFACT_MANIFEST_NAME",
    "ArtifactManifest",
    "ArtifactLoader",
    "ArtifactLoadResult",
    "FMUnetCheckpointLoader",
    "RegionDiffGeneratorArtifactLoader",
    "SDStage1PipelineLoader",
    "read_artifact_manifest",
    "validate_manifest_compatibility",
    "write_artifact_manifest",
]
