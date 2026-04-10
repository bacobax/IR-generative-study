"""Unit tests for Stage-1 Stable Diffusion IR adaptation baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DConditionModel

from src.algorithms.stable_diffusion.config import parse_args
from src.algorithms.stable_diffusion.data import (
    TextImageDataset,
    ir_npy_to_normalized_rgb,
    resolve_training_data_source,
)
from src.algorithms.stable_diffusion.models import ModelComponents, configure_trainable_components
from src.core.normalization import RAW_UINT16_PERCENTILE, UINT8_LINEAR


class _TokenizerOutput:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class _MockTokenizer:
    model_max_length = 8

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        batch = torch.ones(len(texts), self.model_max_length, dtype=torch.long)
        return _TokenizerOutput(batch)


def _tiny_unet() -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=8,
    )


def _build_model_components() -> ModelComponents:
    return ModelComponents(
        unet=_tiny_unet(),
        vae=torch.nn.Conv2d(1, 1, kernel_size=1),
        text_encoder=torch.nn.Linear(8, 8),
        tokenizer=_MockTokenizer(),
        noise_scheduler=None,
        weight_dtype=torch.float32,
    )


def test_sd_config_parses_new_baseline_options():
    cfg = parse_args(
        [
            "--baseline_mode",
            "sd_ir_unet",
            "--dataset_id",
            "flir_private_proxy_alignment_v18",
            "--prompt_text",
            "thermal image",
            "--unet_train_mode",
            "partial",
            "--unet_trainable_modules",
            "mid_block",
            "up_blocks",
            "--output_dir",
            "/tmp/sd_ir_test",
        ]
    )
    assert cfg.baseline_mode == "sd_ir_unet"
    assert cfg.dataset_id == "flir_private_proxy_alignment_v18"
    assert cfg.prompt_text == "thermal image"
    assert cfg.unet_train_mode == "partial"
    assert cfg.unet_trainable_modules == ["mid_block", "up_blocks"]


def test_dataset_resolution_uses_repo_normalization_modes():
    flir = resolve_training_data_source(
        dataset_id="flir_private_proxy_alignment_v18",
        dataset_name=None,
        dataset_config_name=None,
        train_data_dir=None,
        train_split="train",
    )
    v18 = resolve_training_data_source(
        dataset_id="v18",
        dataset_name=None,
        dataset_config_name=None,
        train_data_dir=None,
        train_split="train",
    )

    assert flir.normalization_mode == UINT8_LINEAR
    assert v18.normalization_mode == RAW_UINT16_PERCENTILE
    assert flir.train_data_dir.endswith("data/raw/flir_private_proxy_alignment_v18/train")
    assert v18.train_data_dir.endswith("data/raw/v18/train")


def test_ir_preprocessing_respects_v18_and_flir_normalization():
    v18 = np.array([[11667, 13944]], dtype=np.uint16)
    flir = np.array([[0, 255]], dtype=np.uint8)

    v18_img = ir_npy_to_normalized_rgb(v18, normalization_mode=RAW_UINT16_PERCENTILE)
    flir_img = ir_npy_to_normalized_rgb(flir, normalization_mode=UINT8_LINEAR)

    v18_arr = np.asarray(v18_img)
    flir_arr = np.asarray(flir_img)

    assert v18_arr.shape == (1, 2, 3)
    assert flir_arr.shape == (1, 2, 3)
    assert int(v18_arr[0, 0, 0]) == 0
    assert int(v18_arr[0, 1, 0]) == 255
    assert int(flir_arr[0, 0, 0]) == 0
    assert int(flir_arr[0, 1, 0]) == 255


def test_constant_prompt_dataset_works_without_metadata(tmp_path: Path):
    image_path = tmp_path / "sample.npy"
    np.save(image_path, np.zeros((8, 8), dtype=np.uint8))

    from datasets import Dataset

    ds = Dataset.from_dict({"image": [str(image_path)], "text": [""]})
    dataset = TextImageDataset(
        dataset=ds,
        tokenizer=_MockTokenizer(),
        image_transforms=torch.nn.Identity(),
        image_column="image",
        caption_column="text",
        image_preprocessor=lambda path: torch.zeros(3, 8, 8),
        prompt_text="thermal image",
    )

    item = dataset[0]
    assert item["pixel_values"].shape == (3, 8, 8)
    assert item["input_ids"].shape == (8,)


def test_lora_mode_trains_only_adapter_params():
    cfg = parse_args(
        [
            "--baseline_mode",
            "sd_ir_lora",
            "--dataset_id",
            "v18",
            "--output_dir",
            "/tmp/sd_ir_test_lora",
        ]
    )
    models = _build_model_components()
    info = configure_trainable_components(models=models, config=cfg)

    assert info["lora_active"] is True
    assert any(models.unet.peft_config)
    assert count_trainable(models.text_encoder) == 0
    assert count_trainable(models.vae) == 0
    assert count_trainable(models.unet) > 0


def test_unet_full_mode_unfreezes_all_unet_params():
    cfg = parse_args(
        [
            "--baseline_mode",
            "sd_ir_unet",
            "--dataset_id",
            "v18",
            "--unet_train_mode",
            "full",
            "--output_dir",
            "/tmp/sd_ir_test_unet",
        ]
    )
    models = _build_model_components()
    configure_trainable_components(models=models, config=cfg)

    assert all(param.requires_grad for param in models.unet.parameters())
    assert not any(param.requires_grad for param in models.text_encoder.parameters())
    assert not any(param.requires_grad for param in models.vae.parameters())


def test_unet_partial_mode_only_trains_selected_prefixes():
    cfg = parse_args(
        [
            "--baseline_mode",
            "sd_ir_unet",
            "--dataset_id",
            "v18",
            "--unet_train_mode",
            "partial",
            "--unet_trainable_modules",
            "mid_block",
            "--output_dir",
            "/tmp/sd_ir_test_partial",
        ]
    )
    models = _build_model_components()
    info = configure_trainable_components(models=models, config=cfg)

    trainable_names = info["trainable_parameter_names"]["unet"]
    assert trainable_names
    assert all(name.startswith("mid_block") for name in trainable_names)
    assert not any(param.requires_grad for param in models.text_encoder.parameters())
    assert not any(param.requires_grad for param in models.vae.parameters())


def test_unet_partial_mode_fails_on_no_match():
    cfg = parse_args(
        [
            "--baseline_mode",
            "sd_ir_unet",
            "--dataset_id",
            "v18",
            "--unet_train_mode",
            "partial",
            "--unet_trainable_modules",
            "does_not_exist",
            "--output_dir",
            "/tmp/sd_ir_test_partial_fail",
        ]
    )
    models = _build_model_components()

    try:
        configure_trainable_components(models=models, config=cfg)
    except ValueError as exc:
        assert "zero trainable parameters" in str(exc)
    else:
        raise AssertionError("Expected partial U-Net config to fail when no modules match.")


def count_trainable(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)
