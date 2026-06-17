"""Fast CPU tests for FLUX.1-dev QLoRA stage-1 integration.

No model downloads, no GPU required.  Mirrors the structure of test_sdxl_stage1.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.algorithms.flux.config import (
    DEFAULT_FLUX_LORA_TARGET_MODULES,
    DEFAULT_FLUX_MODEL,
    parse_args,
)
from src.algorithms.flux.data import create_dataloader
from src.algorithms.flux.models import (
    build_stage1_manifest,
    get_lora_config,
    load_stage1_manifest,
    save_stage1_manifest,
)


# ─────────────────────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────────────────────

def test_flux_config_defaults(tmp_path: Path) -> None:
    cfg = parse_args(
        [
            "--dataset_id",
            "flir_private_proxy_alignment_v18",
            "--output_dir",
            str(tmp_path / "run"),
        ]
    )
    assert cfg.pretrained_model_name_or_path == DEFAULT_FLUX_MODEL
    assert cfg.resolution == 512
    assert cfg.rank == 8
    assert cfg.lora_target_modules == DEFAULT_FLUX_LORA_TARGET_MODULES
    assert cfg.quantize_4bit is True
    assert cfg.cache_latents is True
    assert cfg.use_8bit_adam is True
    assert cfg.guidance_scale == 1.0
    assert cfg.weighting_scheme == "none"


def test_flux_config_yaml_unknown_key_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "dataset_id: flir_private_proxy_alignment_v18\nnot_a_flux_key: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown keys"):
        parse_args(["--config", str(config_path)])


def test_flux_config_yaml_overrides_default(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "dataset_id: flir_private_proxy_alignment_v18\nrank: 4\nresolution: 256\n",
        encoding="utf-8",
    )
    cfg = parse_args(["--config", str(config_path), "--output_dir", str(tmp_path / "run")])
    assert cfg.rank == 4
    assert cfg.resolution == 256


def test_flux_config_cli_overrides_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "dataset_id: flir_private_proxy_alignment_v18\nrank: 4\n",
        encoding="utf-8",
    )
    cfg = parse_args(
        ["--config", str(config_path), "--rank", "16", "--output_dir", str(tmp_path / "run")]
    )
    assert cfg.rank == 16


def test_flux_config_invalid_dataset_id(tmp_path: Path) -> None:
    # dataset_id is a choices= argument; argparse raises SystemExit on invalid choice.
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset_id",
                "not_a_real_dataset",
                "--output_dir",
                str(tmp_path / "run"),
            ]
        )


def test_flux_config_requires_data_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Need either"):
        parse_args(["--output_dir", str(tmp_path / "run")])


def test_flux_config_subset_manifest_rejected_for_hf_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="subset_manifest is only supported"):
        parse_args(
            [
                "--dataset_name",
                "some/hf-dataset",
                "--subset_manifest",
                "train_100.json",
                "--output_dir",
                str(tmp_path / "run"),
            ]
        )


def test_flux_config_invalid_rank(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--rank must be positive"):
        parse_args(
            [
                "--dataset_id",
                "flir_private_proxy_alignment_v18",
                "--rank",
                "0",
                "--output_dir",
                str(tmp_path / "run"),
            ]
        )


def test_flux_config_invalid_weighting_scheme(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset_id",
                "flir_private_proxy_alignment_v18",
                "--weighting_scheme",
                "bogus",
                "--output_dir",
                str(tmp_path / "run"),
            ]
        )


# ─────────────────────────────────────────────────────────────
# LoRA config helper
# ─────────────────────────────────────────────────────────────

def test_get_lora_config_attributes() -> None:
    from peft import LoraConfig

    cfg = get_lora_config(rank=8, lora_alpha_scale=1.0, target_modules=["to_q", "to_k"])
    assert isinstance(cfg, LoraConfig)
    assert cfg.r == 8
    assert cfg.lora_alpha == 8.0
    assert set(cfg.target_modules) == {"to_q", "to_k"}
    assert cfg.init_lora_weights == "gaussian"


def test_get_lora_config_alpha_scale() -> None:
    from peft import LoraConfig

    cfg = get_lora_config(rank=4, lora_alpha_scale=0.5, target_modules=["to_q"])
    assert cfg.lora_alpha == 2.0


# ─────────────────────────────────────────────────────────────
# Manifest round-trip
# ─────────────────────────────────────────────────────────────

def _make_minimal_config(tmp_path: Path):
    return parse_args(
        [
            "--dataset_id",
            "flir_private_proxy_alignment_v18",
            "--output_dir",
            str(tmp_path / "run"),
            "--max_train_steps",
            "10",
        ]
    )


def test_stage1_manifest_round_trip(tmp_path: Path) -> None:
    output_dir = str(tmp_path / "run")
    cfg = _make_minimal_config(tmp_path)
    adaptation_info = {"lora_active": True, "model_family": "flux"}
    manifest = build_stage1_manifest(
        config=cfg,
        normalization_mode="raw_uint16_percentile",
        adaptation_info=adaptation_info,
    )
    saved_path = save_stage1_manifest(output_dir, manifest)
    assert Path(saved_path).is_file()

    loaded = load_stage1_manifest(output_dir)
    assert loaded["model_family"] == "flux"
    assert loaded["training_mode"] == "qlora"
    assert loaded["rank"] == 8
    assert loaded["quantize_4bit"] is True
    assert loaded["dataset_id"] == "flir_private_proxy_alignment_v18"
    assert loaded["normalization_mode"] == "raw_uint16_percentile"
    assert loaded["adaptation_info"]["lora_active"] is True


def test_stage1_manifest_writes_artifact_json(tmp_path: Path) -> None:
    output_dir = str(tmp_path / "run2")
    cfg = _make_minimal_config(tmp_path)
    manifest = build_stage1_manifest(
        config=cfg,
        normalization_mode="uint8_linear",
        adaptation_info={},
    )
    save_stage1_manifest(output_dir, manifest)
    artifact_path = Path(output_dir) / "artifact_manifest.json"
    assert artifact_path.is_file()
    data = json.loads(artifact_path.read_text())
    assert data["model_kind"] == "flux_stage1_lora"
    assert data["model_family"] == "flux"


def test_load_stage1_manifest_raises_if_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_stage1_manifest(tmp_path / "does_not_exist")


# ─────────────────────────────────────────────────────────────
# Dataloader
# ─────────────────────────────────────────────────────────────

def test_flux_dataloader_pixel_values_shape(tmp_path: Path) -> None:
    """Dataloader should produce CHW pixel_values in [-1, 1] from a tiny IR npy."""
    np.save(tmp_path / "sample.npy", np.arange(16, dtype=np.uint16).reshape(4, 4))
    dataloader, normalization_mode = create_dataloader(
        dataset_id=None,
        dataset_name=None,
        dataset_config_name=None,
        train_data_dir=str(tmp_path),
        train_split="train",
        cache_dir=None,
        resolution=8,
        center_crop=False,
        random_flip=False,
        interpolation_mode="nearest",
        image_column="image",
        caption_column="text",
        batch_size=1,
        use_ir_preprocessing=True,
    )
    batch = next(iter(dataloader))
    assert normalization_mode
    # FLUX dataset emits only pixel_values (no tokenized input_ids).
    assert set(batch.keys()) == {"pixel_values"}
    assert batch["pixel_values"].shape == (1, 3, 8, 8)
    pv = batch["pixel_values"]
    assert pv.min() >= -1.01 and pv.max() <= 1.01, "pixel_values should be in [-1, 1]"


def test_flux_dataloader_no_prompt_in_batch(tmp_path: Path) -> None:
    """The FLUX dataloader must not include prompt/text fields — prompt is global."""
    np.save(tmp_path / "img.npy", np.zeros((8, 8), dtype=np.uint16))
    dataloader, _ = create_dataloader(
        dataset_id=None,
        dataset_name=None,
        dataset_config_name=None,
        train_data_dir=str(tmp_path),
        train_split="train",
        cache_dir=None,
        resolution=8,
        center_crop=False,
        random_flip=False,
        interpolation_mode="nearest",
        image_column="image",
        caption_column="text",
        batch_size=1,
        use_ir_preprocessing=True,
    )
    batch = next(iter(dataloader))
    assert "prompt_embeds" not in batch
    assert "input_ids_one" not in batch
    assert "text" not in batch


def test_flux_dataloader_max_train_samples(tmp_path: Path) -> None:
    """max_train_samples must limit the dataset size."""
    for i in range(5):
        np.save(tmp_path / f"img{i}.npy", np.zeros((4, 4), dtype=np.uint16))
    dataloader, _ = create_dataloader(
        dataset_id=None,
        dataset_name=None,
        dataset_config_name=None,
        train_data_dir=str(tmp_path),
        train_split="train",
        cache_dir=None,
        resolution=4,
        center_crop=False,
        random_flip=False,
        interpolation_mode="nearest",
        image_column="image",
        caption_column="text",
        batch_size=1,
        max_train_samples=2,
        seed=0,
        use_ir_preprocessing=True,
    )
    assert len(dataloader.dataset) == 2
