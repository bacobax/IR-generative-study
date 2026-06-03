"""Fast tests for Stage-1 Stable Diffusion XL LoRA integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.algorithms.stable_diffusion_xl.config import (
    DEFAULT_SDXL_LORA_TARGET_MODULES,
    DEFAULT_SDXL_MODEL,
    DEFAULT_TEXT_ENCODER_LORA_TARGET_MODULES,
    parse_args,
)
from src.algorithms.stable_diffusion_xl.data import create_dataloader
from src.algorithms.stable_diffusion_xl.models import (
    build_stage1_manifest,
    build_time_ids,
    encode_prompt,
    get_lora_config,
    load_sdxl_stage1_pipeline,
    save_stage1_manifest,
)
from src.algorithms.stable_diffusion_xl.training import log_validation


class _TokenizerOutput:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class _MockTokenizer:
    model_max_length = 6

    def __init__(self, value: int):
        self.value = value

    def __call__(self, text, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        return _TokenizerOutput(
            torch.full((len(texts), self.model_max_length), self.value, dtype=torch.long)
        )


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self, hidden_dim: int, pooled_dim: int, offset: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooled_dim = pooled_dim
        self.offset = offset
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, output_hidden_states=True, return_dict=False):
        batch, seq = input_ids.shape
        hidden = torch.ones(batch, seq, self.hidden_dim) * self.offset
        hidden_states = (
            hidden * 0.25,
            hidden * 0.5,
            hidden,
        )
        pooled = torch.ones(batch, self.pooled_dim) * (self.offset + 10)
        return (pooled, hidden_states)


class _FakeTensorboardWriter:
    def __init__(self) -> None:
        self.images = {}
        self.scalars = {}

    def add_images(self, tag, images, step, dataformats):
        self.images[tag] = {
            "images": images,
            "step": step,
            "dataformats": dataformats,
        }

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = {
            "value": value,
            "step": step,
        }


class _FakeValidationAccelerator:
    def __init__(self) -> None:
        self.writer = _FakeTensorboardWriter()
        self.trackers = [SimpleNamespace(name="tensorboard", writer=self.writer)]


class _FakeValidationVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
        self.config = SimpleNamespace(
            scaling_factor=2.0,
            latents_mean=None,
            latents_std=None,
        )
        self.decode_dtypes = []

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def decode(self, latents, return_dict=False):
        self.decode_dtypes.append(latents.dtype)
        image = torch.full(
            (latents.shape[0], 3, 2, 2),
            0.5,
            device=latents.device,
            dtype=latents.dtype,
        )
        return (image,)


class _FakeImageProcessor:
    def postprocess(self, image, output_type="pil"):
        assert output_type == "pil"
        arrays = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        arrays = (np.clip(arrays, 0.0, 1.0) * 255.0).astype(np.uint8)
        return [Image.fromarray(array) for array in arrays]


class _FakeSDXLValidationPipeline:
    def __init__(self) -> None:
        self.vae = _FakeValidationVAE()
        self.image_processor = _FakeImageProcessor()
        self.watermark = None
        self.calls = []
        self.progress_bar_disabled = False
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        self.vae.to(device=self.device)
        return self

    def set_progress_bar_config(self, *, disable):
        self.progress_bar_disabled = disable

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        latents = torch.zeros(1, 4, 2, 2, device=self.device, dtype=torch.float16)
        return SimpleNamespace(images=latents)


def test_sdxl_config_defaults_and_validation(tmp_path: Path) -> None:
    cfg = parse_args(
        [
            "--dataset_id",
            "flir_private_proxy_alignment_v18",
            "--output_dir",
            str(tmp_path / "run"),
        ]
    )

    assert cfg.pretrained_model_name_or_path == DEFAULT_SDXL_MODEL
    assert cfg.baseline_mode == "sdxl_ir_lora"
    assert cfg.resolution == 1024
    assert cfg.lora_target_modules == DEFAULT_SDXL_LORA_TARGET_MODULES
    assert cfg.text_encoder_lora_enabled is False


def test_sdxl_config_yaml_unknown_key_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "dataset_id: v18\nnot_a_real_key: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown keys"):
        parse_args(["--config", str(config_path)])


def test_sdxl_config_rejects_invalid_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset_id",
                "v18",
                "--baseline_mode",
                "sd_ir_lora",
                "--output_dir",
                str(tmp_path / "run"),
            ]
        )


def test_sdxl_dataloader_collate_keys_and_shapes(tmp_path: Path) -> None:
    np.save(tmp_path / "sample.npy", np.arange(16, dtype=np.uint16).reshape(4, 4))
    dataloader, normalization_mode = create_dataloader(
        dataset_id=None,
        dataset_name=None,
        dataset_config_name=None,
        train_data_dir=str(tmp_path),
        train_split="train",
        cache_dir=None,
        tokenizer_one=_MockTokenizer(1),
        tokenizer_two=_MockTokenizer(2),
        resolution=8,
        center_crop=False,
        random_flip=False,
        interpolation_mode="nearest",
        image_column="image",
        caption_column="text",
        batch_size=1,
        use_ir_preprocessing=True,
        prompt_text="thermal image",
    )

    batch = next(iter(dataloader))

    assert normalization_mode
    assert batch["pixel_values"].shape == (1, 3, 8, 8)
    assert batch["input_ids_one"].shape == (1, 6)
    assert batch["input_ids_two"].shape == (1, 6)
    assert batch["original_sizes"] == [(4, 4)]
    assert batch["crop_top_lefts"] == [(0, 0)]
    assert batch["target_sizes"] == [(8, 8)]


def test_sdxl_prompt_encoding_concatenates_and_preserves_pooled() -> None:
    prompt_embeds, pooled = encode_prompt(
        text_encoder=_FakeTextEncoder(hidden_dim=3, pooled_dim=5, offset=1.0),
        text_encoder_2=_FakeTextEncoder(hidden_dim=4, pooled_dim=7, offset=2.0),
        input_ids_one=torch.ones(2, 6, dtype=torch.long),
        input_ids_two=torch.ones(2, 6, dtype=torch.long),
    )

    assert prompt_embeds.shape == (2, 6, 7)
    assert pooled.shape == (2, 7)
    assert torch.allclose(prompt_embeds[..., :3], torch.full((2, 6, 3), 0.5))
    assert torch.allclose(prompt_embeds[..., 3:], torch.full((2, 6, 4), 1.0))
    assert torch.allclose(pooled, torch.full((2, 7), 12.0))


def test_sdxl_time_ids_construction() -> None:
    time_ids = build_time_ids(
        original_sizes=[(480, 640), (1024, 1024)],
        crop_top_lefts=[(0, 0), (12, 34)],
        target_sizes=[(512, 512), (768, 768)],
        dtype=torch.float32,
    )

    assert time_ids.tolist() == [
        [480.0, 640.0, 0.0, 0.0, 512.0, 512.0],
        [1024.0, 1024.0, 12.0, 34.0, 768.0, 768.0],
    ]


def test_sdxl_lora_target_module_config() -> None:
    pytest.importorskip("peft")

    unet_cfg = get_lora_config(
        rank=8,
        lora_alpha_scale=1.0,
        target_modules=DEFAULT_SDXL_LORA_TARGET_MODULES,
    )
    text_cfg = get_lora_config(
        rank=4,
        lora_alpha_scale=0.5,
        target_modules=DEFAULT_TEXT_ENCODER_LORA_TARGET_MODULES,
    )

    assert unet_cfg.r == 8
    assert unet_cfg.lora_alpha == 8
    assert set(unet_cfg.target_modules) == set(DEFAULT_SDXL_LORA_TARGET_MODULES)
    assert text_cfg.r == 4
    assert text_cfg.lora_alpha == 2
    assert set(text_cfg.target_modules) == set(DEFAULT_TEXT_ENCODER_LORA_TARGET_MODULES)


def test_sdxl_stage1_manifest_and_pipeline_loader(monkeypatch, tmp_path: Path) -> None:
    cfg = parse_args(
        [
            "--dataset_id",
            "v18",
            "--resolution",
            "512",
            "--rank",
            "8",
            "--output_dir",
            str(tmp_path),
        ]
    )
    manifest = build_stage1_manifest(
        config=cfg,
        normalization_mode="raw_uint16_percentile",
        adaptation_info={"ok": True},
    )
    save_stage1_manifest(str(tmp_path), manifest)
    (tmp_path / "pytorch_lora_weights.safetensors").write_bytes(b"placeholder")

    class _FakePipeline:
        def __init__(self):
            self.loaded = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pipe = cls()
            pipe.args = args
            pipe.kwargs = kwargs
            return pipe

        def load_lora_weights(self, path):
            self.loaded = path

    monkeypatch.setattr(
        "src.algorithms.stable_diffusion_xl.models.StableDiffusionXLPipeline",
        _FakePipeline,
    )

    pipe, loaded = load_sdxl_stage1_pipeline(stage1_dir=tmp_path, torch_dtype=torch.float32)

    assert loaded["model_family"] == "sdxl"
    assert loaded["resolution"] == 512
    assert pipe.loaded == str(tmp_path)


def test_sdxl_validation_decodes_latents_in_fp32_for_tensorboard() -> None:
    pipeline = _FakeSDXLValidationPipeline()
    accelerator = _FakeValidationAccelerator()

    images = log_validation(
        pipeline=pipeline,
        validation_prompt="thermal image",
        num_images=2,
        num_inference_steps=17,
        device=torch.device("cpu"),
        seed=123,
        accelerator=accelerator,
        epoch=5,
        height=512,
        width=512,
    )

    assert len(images) == 2
    assert pipeline.progress_bar_disabled is True
    assert [call["num_inference_steps"] for call in pipeline.calls] == [17, 17]
    assert [call["output_type"] for call in pipeline.calls] == ["latent", "latent"]
    assert pipeline.vae.decode_dtypes == [torch.float32, torch.float32]
    assert pipeline.vae.dtype == torch.float16

    logged = accelerator.writer.images["validation/generated_rgb_01"]
    assert logged["step"] == 5
    assert logged["dataformats"] == "NHWC"
    assert logged["images"].shape == (2, 2, 2, 3)
    assert float(logged["images"].mean()) > 0.0
    assert "validation/generated_mean" in accelerator.writer.scalars
    assert "validation/generated_std" in accelerator.writer.scalars


def test_generate_cli_accepts_sdxl_and_routes(monkeypatch, tmp_path: Path) -> None:
    import src.cli.generate as generate

    calls = []

    def _fake_generate_sdxl(args, entries):
        calls.append((args.mode, args.stage1_dir, len(entries)))

    monkeypatch.setattr(generate, "generate_sdxl", _fake_generate_sdxl)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--mode",
            "sdxl",
            "--stage1_dir",
            str(tmp_path / "stage1"),
            "--metadata",
            str(tmp_path / "missing.jsonl"),
            "--max_samples",
            "2",
            "--output_dir",
            str(tmp_path / "out"),
        ],
    )

    generate.main()

    assert calls == [("sdxl", str(tmp_path / "stage1"), 2)]
