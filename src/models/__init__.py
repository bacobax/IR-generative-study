"""Model definitions and extension interfaces."""

from src.models.adapters import (
    ArtifactLoadRequest,
    BuiltFMModelAdapter,
    DiffusersModelAdapter,
    FMModelAdapter,
    ModelAdapter,
    ModelAdapterBase,
    ModelBuildRequest,
    ModelBundle,
)

__all__ = [
    "ArtifactLoadRequest",
    "BuiltFMModelAdapter",
    "DiffusersModelAdapter",
    "FMModelAdapter",
    "ModelAdapter",
    "ModelAdapterBase",
    "ModelBuildRequest",
    "ModelBundle",
]
