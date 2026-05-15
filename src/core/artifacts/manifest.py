"""Versioned artifact manifest helpers.

The manifest is intentionally additive: existing artifacts may not have one,
and readers should treat that as a legacy-compatible artifact rather than a
hard failure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class ArtifactManifest:
    """JSON-serializable provenance for a trainable artifact bundle."""

    schema_version: int = 1
    model_kind: str = ""
    model_family: str = ""
    base_model: str | None = None
    components: dict[str, Any] = field(default_factory=dict)
    adapters: list[dict[str, Any]] = field(default_factory=list)
    task: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)
    checkpoints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe plain dictionary."""
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactManifest":
        """Build a manifest from a JSON mapping."""
        if not isinstance(data, Mapping):
            raise TypeError(f"Artifact manifest must be a mapping, got {type(data).__name__}")
        known = {
            "schema_version",
            "model_kind",
            "model_family",
            "base_model",
            "components",
            "adapters",
            "task",
            "dataset",
            "normalization",
            "checkpoints",
            "metadata",
        }
        payload = {key: data[key] for key in known if key in data}
        return cls(**payload)


def write_artifact_manifest(output_dir: str | Path, manifest: ArtifactManifest) -> Path:
    """Write *manifest* to ``output_dir/artifact_manifest.json``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / ARTIFACT_MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def read_artifact_manifest(path_or_dir: str | Path) -> ArtifactManifest | None:
    """Read an artifact manifest from a file or directory.

    Missing manifests return ``None`` so legacy artifacts remain loadable.
    """
    path = Path(path_or_dir)
    manifest_path = path / ARTIFACT_MANIFEST_NAME if path.is_dir() else path
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ArtifactManifest.from_dict(data)


def validate_manifest_compatibility(
    manifest: ArtifactManifest | None,
    *,
    expected_task: str | None = None,
    expected_model_family: str | None = None,
) -> None:
    """Validate a manifest against expected task and model family.

    ``None`` means no manifest was present, which is accepted as a legacy
    artifact for backward compatibility.
    """
    if manifest is None:
        return

    if expected_task is not None:
        actual_task = manifest.task.get("kind") or manifest.task.get("name")
        if actual_task != expected_task:
            raise ValueError(
                "Artifact task mismatch: "
                f"expected {expected_task!r}, got {actual_task!r}."
            )

    if expected_model_family is not None and manifest.model_family != expected_model_family:
        raise ValueError(
            "Artifact model family mismatch: "
            f"expected {expected_model_family!r}, got {manifest.model_family!r}."
        )


__all__ = [
    "ARTIFACT_MANIFEST_NAME",
    "ArtifactManifest",
    "read_artifact_manifest",
    "validate_manifest_compatibility",
    "write_artifact_manifest",
]
