"""Unit tests for Stage-1 Stable Diffusion IR adaptation baselines."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import UNet2DConditionModel

from src.algorithms.stable_diffusion.config import (
    DEFAULT_NUM_TRAIN_EPOCHS,
    LEGACY_GENERIC_PROMPT,
    parse_args,
)
from src.algorithms.stable_diffusion.data import (
    TextImageDataset,
    create_dataloader,
    ir_npy_to_normalized_rgb,
    resolve_training_data_source,
)
from src.algorithms.stable_diffusion.models import (
    ModelComponents,
    configure_trainable_components,
    normalize_lora_state_dict_keys,
)
import src.algorithms.stable_diffusion.training as sd_training
from src.algorithms.stable_diffusion.training import Trainer, log_validation
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


class _TensorboardWriter:
    def __init__(self):
        self.images = []
        self.scalars = []

    def add_images(self, tag, images, step, dataformats):
        self.images.append((tag, images.shape, step, dataformats))

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), step))


class _Tracker:
    def __init__(self):
        self.name = "tensorboard"
        self.writer = _TensorboardWriter()


class _FakeValidationAccelerator:
    def __init__(self):
        self.trackers = [_Tracker()]


class _FakeResumeAccelerator:
    def __init__(self):
        self.loaded_path = None
        self.messages = []

    def print(self, message):
        self.messages.append(message)

    def load_state(self, path):
        self.loaded_path = path


class _FakeCheckpointAccelerator:
    def __init__(self):
        self.is_main_process = True
        self.saved_paths = []

    def save_state(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.saved_paths.append(path)


class _FakePipeline:
    def __init__(self):
        self.calls = []
        self.progress_bar_disabled = None
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def set_progress_bar_config(self, *, disable):
        self.progress_bar_disabled = disable

    def __call__(self, prompt, *, num_inference_steps, generator):
        self.calls.append(
            {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "seed": generator.initial_seed(),
            }
        )
        return type("PipelineOutput", (), {"images": [np.zeros((4, 4, 3), dtype=np.uint8)]})()


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


def test_sd_config_yaml_parses_training_and_validation_steps(tmp_path: Path):
    config_path = tmp_path / "sd_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: flir_private_proxy_alignment_v18",
                f"output_dir: {tmp_path / 'sd_run'}",
                "max_train_steps: 1234",
                "validation_num_inference_steps: 17",
            ]
        ),
        encoding="utf-8",
    )

    cfg = parse_args(["--config", str(config_path)])

    assert cfg.max_train_steps == 1234
    assert cfg.validation_num_inference_steps == 17


def test_sd_stage1_defaults_to_epoch_driven_training():
    cfg = parse_args(
        [
            "--dataset_id",
            "flir_private_proxy_alignment_v18",
            "--output_dir",
            "/tmp/sd_ir_epochs",
        ]
    )

    assert cfg.num_train_epochs == DEFAULT_NUM_TRAIN_EPOCHS == 80
    assert cfg.max_train_steps is None


def test_flir_stage1_presets_are_epoch_driven_by_default():
    preset_paths = [
        "configs/sd/train/presets/flir_unet_full_stage1.yaml",
        "configs/sd/train/presets/flir_unet_partial_stage1.yaml",
        "configs/sd/train/presets/flir_lora_stage1_r8.yaml",
        "configs/sd/train/presets/flir_lora_stage1_r16.yaml",
        "configs/sd/train/presets/flir_lora_stage1_r32.yaml",
        "configs/sd/train/presets/flir_lora_stage1_r64.yaml",
        "configs/sd/train/presets/flir_lora_stage1_r128.yaml",
    ]

    for preset_path in preset_paths:
        cfg = parse_args(["--config", preset_path])
        assert cfg.num_train_epochs > 0
        assert cfg.max_train_steps is None
        assert cfg.resume_from_checkpoint is None
        assert cfg.checkpointing_epochs > 0


def test_yaml_can_choose_stage1_epoch_count(tmp_path: Path):
    config_path = tmp_path / "sd_epochs.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: flir_private_proxy_alignment_v18",
                f"output_dir: {tmp_path / 'sd_run'}",
                "num_train_epochs: 12",
            ]
        ),
        encoding="utf-8",
    )

    cfg = parse_args(["--config", str(config_path)])

    assert cfg.num_train_epochs == 12
    assert cfg.max_train_steps is None


def test_sd_config_generic_prompt_overrides_default_prompt_text():
    cfg = parse_args(
        [
            "--baseline_mode",
            "sd_ir_lora",
            "--dataset_id",
            "v18",
            "--generic_prompt",
            "--output_dir",
            "/tmp/sd_ir_generic_prompt",
        ]
    )

    assert cfg.prompt_text == "thermal image"
    assert cfg.resolved_prompt_text() == LEGACY_GENERIC_PROMPT


def test_sd_config_yaml_rejects_unknown_keys(tmp_path: Path):
    config_path = tmp_path / "sd_bad.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: flir_private_proxy_alignment_v18",
                f"output_dir: {tmp_path / 'sd_run'}",
                "validation_num_inference_stepz: 17",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown keys in SD config"):
        parse_args(["--config", str(config_path)])


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


def test_local_dataloader_does_not_require_huggingface_datasets(tmp_path: Path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    np.save(images_dir / "sample.npy", np.zeros((8, 8), dtype=np.uint8))

    real_import = __import__

    def import_without_datasets(name, *args, **kwargs):
        if name == "datasets" or name.startswith("datasets."):
            raise ModuleNotFoundError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_datasets)

    dataloader, normalization_mode = create_dataloader(
        dataset_id=None,
        dataset_name=None,
        dataset_config_name=None,
        train_data_dir=str(tmp_path),
        train_split="train",
        cache_dir=None,
        tokenizer=_MockTokenizer(),
        resolution=8,
        center_crop=False,
        random_flip=False,
        interpolation_mode="nearest",
        image_column="image",
        caption_column="text",
        batch_size=1,
        num_workers=0,
        max_train_samples=1,
        seed=7,
        prompt_text="thermal image",
    )

    batch = next(iter(dataloader))
    assert normalization_mode == RAW_UINT16_PERCENTILE
    assert batch["pixel_values"].shape == (1, 3, 8, 8)
    assert batch["input_ids"].shape == (1, 8)


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


def test_flir_lora_r64_preset_reaches_lora_and_checkpoint_consumers(tmp_path: Path, monkeypatch):
    cfg = parse_args(["--config", "configs/sd/train/presets/flir_lora_stage1_r64.yaml"])
    cfg.output_dir = str(tmp_path / "flir_lora_r64")

    models = _build_model_components()
    configure_trainable_components(models=models, config=cfg)

    peft_cfg = models.unet.peft_config["default"]
    assert peft_cfg.r == 64
    assert float(peft_cfg.lora_alpha) == 64.0
    assert sorted(peft_cfg.target_modules) == sorted(cfg.lora_target_modules)

    monkeypatch.setattr(sd_training, "logger", logging.getLogger("test_sd_checkpointing"))
    accelerator = _FakeCheckpointAccelerator()
    trainer = Trainer(
        config=cfg,
        models=models,
        train_dataloader=[],
        normalization_mode="uint8_linear",
        adaptation_info={},
        accelerator=accelerator,
    )

    cfg.checkpointing_epochs = 2
    trainer.global_step = 17
    trainer._maybe_save_checkpoint(epoch=0)
    assert accelerator.saved_paths == []

    trainer._maybe_save_checkpoint(epoch=1)

    expected_dir = Path(cfg.output_dir) / "checkpoint-17"
    assert accelerator.saved_paths == [str(expected_dir)]

    metadata = json.loads((expected_dir / "training_state.json").read_text(encoding="utf-8"))
    assert metadata["global_step"] == 17
    assert metadata["lr_warmup_steps"] == cfg.lr_warmup_steps
    assert metadata["lr_scheduler"] == cfg.lr_scheduler


def test_yaml_lora_overrides_reach_target_modules_and_alpha_scale(tmp_path: Path):
    config_path = tmp_path / "sd_lora.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_id: v18",
                f"output_dir: {tmp_path / 'sd_run'}",
                "rank: 16",
                "lora_alpha_scale: 0.5",
                "lora_target_modules:",
                "  - to_q",
                "  - to_v",
                "checkpointing_epochs: 3",
            ]
        ),
        encoding="utf-8",
    )

    cfg = parse_args(["--config", str(config_path)])
    models = _build_model_components()
    configure_trainable_components(models=models, config=cfg)

    peft_cfg = models.unet.peft_config["default"]
    assert peft_cfg.r == 16
    assert float(peft_cfg.lora_alpha) == 8.0
    assert sorted(peft_cfg.target_modules) == ["to_q", "to_v"]
    assert cfg.checkpointing_epochs == 3


def test_lora_state_dict_normalization_preserves_checkpoint_loader_keys():
    state = {
        "unet.block.to_q.lora.down.weight": torch.zeros(2, 3),
        "unet.block.to_q.lora.up.weight": torch.zeros(3, 2),
        "unet.block.proj_in.lora_A.weight": torch.zeros(2, 3, 1, 1),
    }

    normalized = normalize_lora_state_dict_keys(state)

    assert "unet.block.to_q.lora_A.weight" in normalized
    assert "unet.block.to_q.lora_B.weight" in normalized
    assert "unet.block.proj_in.lora_A.weight" in normalized
    assert "unet.block.to_q.lora.down.weight" not in normalized


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


def test_log_validation_uses_configured_inference_steps(monkeypatch):
    monkeypatch.setattr(sd_training, "logger", logging.getLogger("test_sd_validation"))
    pipeline = _FakePipeline()
    accelerator = _FakeValidationAccelerator()

    images = log_validation(
        pipeline=pipeline,
        validation_prompt="thermal image",
        num_images=3,
        num_inference_steps=17,
        device=torch.device("cpu"),
        seed=123,
        accelerator=accelerator,
        epoch=2,
    )

    assert len(images) == 3
    assert pipeline.progress_bar_disabled is True
    assert [call["num_inference_steps"] for call in pipeline.calls] == [17, 17, 17]


def test_resume_allows_constant_schedule_extension_without_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sd_training, "logger", logging.getLogger("test_sd_resume"))
    checkpoint_dir = tmp_path / "checkpoint-12000"
    checkpoint_dir.mkdir()

    accelerator = _FakeResumeAccelerator()
    trainer = Trainer(
        config=parse_args(
            [
                "--baseline_mode",
                "sd_ir_unet",
                "--dataset_id",
                "flir_private_proxy_alignment_v18",
                "--output_dir",
                str(tmp_path),
                "--resume_from_checkpoint",
                "latest",
                "--max_train_steps",
                "20000",
                "--lr_scheduler",
                "constant",
                "--lr_warmup_steps",
                "0",
            ]
        ),
        models=None,
        train_dataloader=None,
        normalization_mode="uint8_linear",
        adaptation_info={},
        accelerator=accelerator,
    )
    trainer.num_update_steps_per_epoch = 100

    trainer.resume_from_checkpoint()

    assert accelerator.loaded_path == str(checkpoint_dir)
    assert trainer.global_step == 12000
    assert trainer.first_epoch == 120


def test_resume_rejects_extending_non_constant_schedule(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint-12000"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "training_state.json").write_text(
        json.dumps(
            {
                "global_step": 12000,
                "lr_scheduler": "linear",
                "lr_warmup_steps": 0,
                "max_train_steps": 12000,
            }
        ),
        encoding="utf-8",
    )

    accelerator = _FakeResumeAccelerator()
    trainer = Trainer(
        config=parse_args(
            [
                "--baseline_mode",
                "sd_ir_unet",
                "--dataset_id",
                "flir_private_proxy_alignment_v18",
                "--output_dir",
                str(tmp_path),
                "--resume_from_checkpoint",
                "latest",
                "--max_train_steps",
                "20000",
                "--lr_scheduler",
                "linear",
                "--lr_warmup_steps",
                "0",
            ]
        ),
        models=None,
        train_dataloader=None,
        normalization_mode="uint8_linear",
        adaptation_info={},
        accelerator=accelerator,
    )
    trainer.num_update_steps_per_epoch = 100

    with pytest.raises(ValueError, match="Changing max_train_steps across resume is only supported"):
        trainer.resume_from_checkpoint()


def count_trainable(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)
