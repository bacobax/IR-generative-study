from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ArtifactManifest,
    read_artifact_manifest,
    validate_manifest_compatibility,
    write_artifact_manifest,
)


def _manifest() -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=1,
        model_kind="native_fm_unet",
        model_family="flow_matching",
        base_model="runwayml/stable-diffusion-v1-5",
        components={"unet": {"config": "UNET/config.json"}},
        adapters=[{"kind": "layout_conditioning", "variant": "stay_v2"}],
        task={"kind": "flow_matching", "target": "v"},
        dataset={"dataset_id": "flir_private_proxy_alignment_v18"},
        normalization={"mode": "flir_thermal_minus_one_to_one"},
        checkpoints={"best": "UNET/unet_fm_best.pt"},
        metadata={"note": "roundtrip"},
    )


def test_artifact_manifest_json_roundtrip(tmp_path: Path) -> None:
    manifest = _manifest()

    path = write_artifact_manifest(tmp_path, manifest)
    loaded = read_artifact_manifest(tmp_path)

    assert path == tmp_path / ARTIFACT_MANIFEST_NAME
    assert loaded == manifest
    assert json.loads(path.read_text(encoding="utf-8")) == manifest.to_dict()


def test_missing_legacy_manifest_returns_none(tmp_path: Path) -> None:
    assert read_artifact_manifest(tmp_path) is None
    validate_manifest_compatibility(
        None,
        expected_task="flow_matching",
        expected_model_family="flow_matching",
    )


def test_manifest_compatibility_accepts_matching_manifest() -> None:
    validate_manifest_compatibility(
        _manifest(),
        expected_task="flow_matching",
        expected_model_family="flow_matching",
    )


def test_manifest_compatibility_rejects_task_mismatch() -> None:
    with pytest.raises(ValueError, match="task mismatch"):
        validate_manifest_compatibility(_manifest(), expected_task="stable_diffusion")


def test_manifest_compatibility_rejects_model_family_mismatch() -> None:
    with pytest.raises(ValueError, match="model family mismatch"):
        validate_manifest_compatibility(_manifest(), expected_model_family="stable_diffusion")
