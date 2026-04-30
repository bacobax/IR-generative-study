"""Tests for checkpoint quality comparison generation helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.standalone import generate_checkpoint_quality_comparison as cmp


def _save_state(path: Path, value: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": torch.tensor([value])}, path)


def test_resolve_run_dirs_accepts_unet_and_parent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    unet_dir = run_dir / "UNET"
    unet_dir.mkdir(parents=True)

    from_unet = cmp.resolve_run_dirs(unet_dir)
    from_parent = cmp.resolve_run_dirs(run_dir)

    assert from_unet.pipeline_dir == run_dir
    assert from_unet.unet_dir == unet_dir
    assert from_parent.pipeline_dir == run_dir
    assert from_parent.unet_dir == unet_dir


def test_resolve_checkpoint_pair_skips_corrupt_last_and_uses_latest_epoch(tmp_path: Path) -> None:
    unet_dir = tmp_path / "UNET"
    unet_dir.mkdir()
    _save_state(unet_dir / "unet_fm_best.pt", 1.0)
    (unet_dir / "unet_last_ckpt.pt").write_text("not a torch checkpoint", encoding="utf-8")
    _save_state(unet_dir / "unet_fm_epoch_30.pt", 30.0)
    _save_state(unet_dir / "unet_fm_epoch_60.pt", 60.0)
    _save_state(unet_dir / "unet_fm_epoch_60_ckpt.pt", 600.0)

    pair = cmp.resolve_checkpoint_pair(unet_dir)

    assert pair["best"].path.name == "unet_fm_best.pt"
    assert pair["latest"].path.name == "unet_fm_epoch_60.pt"
    assert pair["latest"].epoch == 60
    assert pair["latest"].source == "latest_epoch_weights"


def test_resolve_checkpoint_pair_falls_back_to_epoch_when_best_missing(tmp_path: Path) -> None:
    unet_dir = tmp_path / "UNET"
    unet_dir.mkdir()
    (unet_dir / "unet_last_ckpt.pt").write_text("partial checkpoint", encoding="utf-8")
    _save_state(unet_dir / "unet_fm_epoch_120.pt", 120.0)

    pair = cmp.resolve_checkpoint_pair(unet_dir)

    assert pair["best"].path.name == "unet_fm_epoch_120.pt"
    assert pair["best"].source == "best_fallback_latest_epoch_weights"
    assert pair["latest"].path.name == "unet_fm_epoch_120.pt"


def test_resolve_checkpoint_pair_finds_sd_best(tmp_path: Path) -> None:
    unet_dir = tmp_path / "UNET"
    _save_state(unet_dir / "unet_sd_uncond_best.pt", 1.0)
    _save_state(unet_dir / "unet_sd_uncond_epoch_9.pt", 9.0)

    pair = cmp.resolve_checkpoint_pair(unet_dir)

    assert pair["best"].path.name == "unet_sd_uncond_best.pt"
    assert pair["latest"].path.name == "unet_sd_uncond_epoch_9.pt"


def test_export_conditional_comparison_split_writes_annotations_and_provenance(tmp_path: Path) -> None:
    sample = {
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "boxes_xyxy": torch.tensor([[1.0, 2.0, 5.0, 8.0], [10.0, 11.0, 13.0, 15.0]]),
        "image_id": "real-1",
        "file_name": "real-1.npy",
        "n_objects": 2,
    }
    checkpoint = cmp.CheckpointChoice(
        role="best",
        path=tmp_path / "UNET" / "unet_fm_best.pt",
        epoch=None,
        source="best_name",
    )

    summary = cmp.export_conditional_comparison_split(
        output_dir=tmp_path / "out",
        records=[sample],
        generated_images=[torch.zeros(1, 16, 16)],
        categories=[{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        checkpoint=checkpoint,
        model_family="fm",
        layout_variant="stay_v2",
        split="train",
        dataset_id="dummy",
        steps=5,
        seed=7,
        overwrite=False,
    )

    assert summary["n_generated_samples"] == 1
    assert summary["n_annotations"] == 2
    assert (tmp_path / "out" / "images" / "sample_000001.npy").is_file()
    assert (tmp_path / "out" / "previews" / "sample_000001.png").is_file()
    payload = json.loads((tmp_path / "out" / "annotations.json").read_text(encoding="utf-8"))
    assert payload["annotations"][0]["source_file_name"] == "real-1.npy"
    provenance = (tmp_path / "out" / "metadata" / "provenance.jsonl").read_text(encoding="utf-8")
    assert "unet_fm_best.pt" in provenance


def test_layout_meta_fallback_uses_checkpoint_class_embedding_size() -> None:
    meta = cmp._layout_meta_from_preset(
        {"layout_conditioning": {"variant": "stay_v2", "class_embed_dim": 48}},
        {"in_channels": 4},
        {1: "person", 15: "car"},
        checkpoint_state={
            "object_encoder.class_embedding.weight": torch.zeros(80, 48),
        },
    )

    assert meta["num_classes"] == 80


def test_sparse_run_unet_config_sample_size_matches_training_vae_factor() -> None:
    adjusted = cmp._apply_training_sample_size(
        {"sample_size": 128, "in_channels": 4, "out_channels": 4},
        {"data": {"image_size": 512}},
        {"block_out_channels": [128, 256, 512, 512]},
    )

    assert adjusted["sample_size"] == 64


def test_cli_smoke_with_mocked_unconditional_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    unet_dir = run_dir / "UNET"
    _save_state(unet_dir / "unet_fm_best.pt", 1.0)
    _save_state(unet_dir / "unet_fm_epoch_2.pt", 2.0)
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_id: dummy",
                "model:",
                "  unet_config: dummy.json",
                "training:",
                "  t_scale: 1000.0",
                "  train_target: v",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakeSampler:
        device = "cpu"

    def _fake_build_fm_sampler(**_kwargs):
        return _FakeSampler()

    def _fake_sample_unconditional(_sampler, **kwargs):
        return [torch.zeros(1, 8, 8) for _ in range(int(kwargs["n_samples"]))]

    monkeypatch.setattr(cmp, "_build_fm_sampler", _fake_build_fm_sampler)
    monkeypatch.setattr(cmp, "_sample_unconditional", _fake_sample_unconditional)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_checkpoint_quality_comparison.py",
            "--weights_dir",
            str(unet_dir),
            "--preset_path",
            str(preset_path),
            "--output_dir",
            str(tmp_path / "out"),
            "--max_samples",
            "2",
            "--batch_size",
            "2",
            "--device",
            "cpu",
        ],
    )

    cmp.main()

    assert (tmp_path / "out" / "best" / "images" / "sample_000001.npy").is_file()
    assert (tmp_path / "out" / "latest" / "images" / "sample_000002.npy").is_file()
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoints"]["latest"]["checkpoint_path"].endswith("unet_fm_epoch_2.pt")
