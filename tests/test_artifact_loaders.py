from pathlib import Path

import pytest
import torch

from src.core.artifacts.loaders import FMUnetCheckpointLoader
from src.core.registry import REGISTRIES
from src.models.adapters import ArtifactLoadRequest


def test_artifact_loader_registry_resolves_known_loaders() -> None:
    import src.core.artifacts.loaders  # noqa: F401

    for name in ("fm_unet_checkpoint", "sd_stage1_pipeline", "sdxl_stage1_pipeline", "regiondiff_generator"):
        assert name in REGISTRIES.artifact_loader


def test_fm_checkpoint_loader_extracts_unet_state(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "unet.pt"
    checkpoint.write_bytes(b"placeholder")
    payload = {"unet_state": {"weight": torch.ones(1)}, "epoch": 3}

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)

    result = FMUnetCheckpointLoader().load(ArtifactLoadRequest(path=checkpoint))

    assert result.kind == "fm_unet_checkpoint"
    assert result.payload == payload["unet_state"]
    assert result.metadata["payload_key"] == "unet_state"
    assert "unet_state" in result.metadata["raw_keys"]


def test_fm_checkpoint_loader_accepts_raw_state_dict(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "raw.pt"
    checkpoint.write_bytes(b"placeholder")
    payload = {"weight": torch.ones(1)}

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)

    result = FMUnetCheckpointLoader().load(ArtifactLoadRequest(path=checkpoint))

    assert result.payload == payload
    assert "payload_key" not in result.metadata


def test_fm_checkpoint_loader_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing FM UNET checkpoint"):
        FMUnetCheckpointLoader().load(ArtifactLoadRequest(path=tmp_path / "missing.pt"))
