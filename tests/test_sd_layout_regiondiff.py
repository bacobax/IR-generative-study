"""Tests for RegionDiff-style SD layout stage-2 components."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler, UNet2DConditionModel

from src.algorithms.stable_diffusion.layout_data import (
    StableDiffusionLayoutDataset,
    build_layout_prompt,
)
from src.algorithms.stable_diffusion.layout_training import LayoutTrainer, log_layout_validation
from src.algorithms.stable_diffusion.layout_models import (
    FUSED_UNET_EXPORT_DIRNAME,
    FUSED_UNET_METADATA_NAME,
    LEGACY_ACCELERATE_MODEL_WEIGHTS,
    SDLayoutModelComponents,
    configure_layout_trainability,
    create_stage2_load_model_hook,
    create_stage2_save_model_hook,
    load_stage1_pipeline_for_stage2,
    load_stage2_layout_pipeline,
    save_stage2_layout_artifact,
)
from src.core.configs.sd_layout_config import SDLayoutTrainConfig
from src.core.normalization import UINT8_LINEAR
from src.models.regiondiffusion import (
    LayoutTokenizer,
    RegionDiffusionUNetWrapper,
    build_area_weight_map,
    build_region_token_mask,
)


class _TokenizerOutput:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class _MockTokenizer:
    model_max_length = 8

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        input_ids = torch.zeros(len(texts), self.model_max_length, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, item in enumerate(texts):
            token_count = min(max(len(str(item).split()), 1), self.model_max_length)
            input_ids[index, :token_count] = torch.arange(1, token_count + 1)
            attention_mask[index, :token_count] = 1
        return _TokenizerOutput(input_ids=input_ids, attention_mask=attention_mask)


class _MockTextEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask=None, return_dict=False):
        hidden = self.embedding(input_ids)
        if return_dict:
            return SimpleNamespace(last_hidden_state=hidden)
        return (hidden,)


class _FakePipeline:
    def __init__(self, unet, tokenizer, text_encoder):
        self.unet = unet
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = None
        self.progress_bar_disabled = None

    def to(self, device):
        self.device = device
        return self

    def set_progress_bar_config(self, *, disable):
        self.progress_bar_disabled = disable

    def __call__(self, prompt, **kwargs):
        del prompt, kwargs
        return SimpleNamespace(images=[np.zeros((16, 16, 3), dtype=np.uint8)])


class _FakeLoRAPipeline:
    def __init__(self):
        self.unet = _tiny_unet()
        with torch.no_grad():
            self.unet.conv_in.bias.zero_()
        self.fused = False
        self.unloaded = False
        self.device = None

    def fuse_lora(self):
        self.fused = True
        with torch.no_grad():
            self.unet.conv_in.bias.add_(1.0)

    def unload_lora_weights(self):
        self.unloaded = True

    def to(self, device):
        self.device = device
        return self


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
        self.device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model):
        return model


class _FakeSaveLoadAccelerator:
    is_main_process = True

    @staticmethod
    def unwrap_model(model):
        return model


class _FakeResumeAccelerator:
    def __init__(self):
        self.loaded_path = None
        self.messages = []

    def print(self, message):
        self.messages.append(message)

    def load_state(self, path):
        self.loaded_path = path


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


def _wrapped_unet() -> RegionDiffusionUNetWrapper:
    return RegionDiffusionUNetWrapper(
        base_unet=_tiny_unet(),
        class_text_features=torch.randn(4, 32),
        category_id_to_name={0: "person", 1: "car", 2: "bicycle", 3: "dog"},
        layout_token_dim=32,
        bbox_fourier_dim=4,
        same_class_position_slots=8,
        use_background_token=True,
        active_region_resolutions=[16],
    )


def _make_layout_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root_dir = tmp_path / "train"
    root_dir.mkdir(parents=True, exist_ok=True)
    np.save(root_dir / "annotated.npy", np.random.randint(0, 255, size=(50, 100), dtype=np.uint8))
    np.save(root_dir / "empty.npy", np.random.randint(0, 255, size=(50, 100), dtype=np.uint8))
    with open(root_dir / "captions.json", "w", encoding="utf-8") as handle:
        json.dump({"annotated": "caption wins", "empty": "no objects caption"}, handle)

    annotations = {
        "images": [
            {"id": "annotated-id", "file_name": "annotated.npy", "width": 100, "height": 50},
            {"id": "empty-id", "file_name": "empty.npy", "width": 100, "height": 50},
        ],
        "annotations": [
            {"id": 1, "image_id": "annotated-id", "category_id": 0, "bbox": [10, 5, 20, 10]},
            {"id": 2, "image_id": "annotated-id", "category_id": 1, "bbox": [50, 10, 25, 20]},
        ],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "car"},
        ],
    }
    annotations_path = root_dir / "annotations.json"
    with open(annotations_path, "w", encoding="utf-8") as handle:
        json.dump(annotations, handle)
    return root_dir, annotations_path


def _write_stage1_manifest(stage1_dir: Path, baseline_mode: str = "sd_ir_lora") -> None:
    stage1_dir.mkdir(parents=True, exist_ok=True)
    (stage1_dir / "stage1_manifest.json").write_text(
        json.dumps(
            {
                "baseline_mode": baseline_mode,
                "pretrained_model_name_or_path": "tiny-sd",
                "revision": None,
                "variant": None,
                "dataset_id": "flir_private_proxy_alignment_v18",
                "train_split": "train",
                "adaptation_info": {
                    "baseline_mode": baseline_mode,
                    "lora_active": baseline_mode == "sd_ir_lora",
                },
            }
        ),
        encoding="utf-8",
    )


def _stage2_config_for_stage1(stage1_dir: Path, checkpoint: str | None) -> SDLayoutTrainConfig:
    cfg = SDLayoutTrainConfig()
    cfg.stage1.pretrained_model_name_or_path = "tiny-sd"
    cfg.stage1.stage1_dir = str(stage1_dir)
    cfg.stage1.stage1_checkpoint = checkpoint
    cfg.training.mixed_precision = "no"
    return cfg


def test_lora_stage1_materializes_fused_unet_checkpoint_before_stage2_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stage1_dir = tmp_path / "lora_stage1"
    checkpoint_dir = stage1_dir / "checkpoint-5"
    checkpoint_dir.mkdir(parents=True)
    _write_stage1_manifest(stage1_dir, baseline_mode="sd_ir_lora")
    (checkpoint_dir / "pytorch_lora_weights.safetensors").write_bytes(b"lora")
    (checkpoint_dir / "training_state.json").write_text(
        json.dumps({"global_step": 5, "lr_scheduler": "constant"}),
        encoding="utf-8",
    )

    calls = {"load_lora": 0}
    fake_pipeline = _FakeLoRAPipeline()

    def _fake_load_lora(pipeline, path):
        assert pipeline is fake_pipeline
        assert Path(path) == checkpoint_dir
        calls["load_lora"] += 1

    monkeypatch.setattr(
        "src.algorithms.stable_diffusion.layout_models.StableDiffusionPipeline.from_pretrained",
        lambda *args, **kwargs: fake_pipeline,
    )
    monkeypatch.setattr(
        "src.algorithms.stable_diffusion.layout_models.load_lora_weights_compat",
        _fake_load_lora,
    )

    pipeline, init_info = load_stage1_pipeline_for_stage2(
        config=_stage2_config_for_stage1(stage1_dir, "checkpoint-5"),
        device=torch.device("cpu"),
    )

    fused_checkpoint = stage1_dir / FUSED_UNET_EXPORT_DIRNAME / "checkpoint-5"
    assert calls["load_lora"] == 1
    assert fake_pipeline.fused is True
    assert fake_pipeline.unloaded is True
    assert (fused_checkpoint / LEGACY_ACCELERATE_MODEL_WEIGHTS).is_file()
    assert (fused_checkpoint / FUSED_UNET_METADATA_NAME).is_file()
    assert (stage1_dir / FUSED_UNET_EXPORT_DIRNAME / "stage1_manifest.json").is_file()
    assert init_info["source_lora_checkpoint"] == str(checkpoint_dir)
    assert init_info["materialized_unet_checkpoint"] == str(fused_checkpoint)
    assert init_info["resolved_stage1_checkpoint"] == str(fused_checkpoint)
    assert init_info["source_kind"] == "materialized_lora_unet_checkpoint"
    assert init_info["materialized_unet_reused"] is False
    assert torch.allclose(pipeline.unet.conv_in.bias, torch.ones_like(pipeline.unet.conv_in.bias))

    training_state = json.loads((fused_checkpoint / "training_state.json").read_text())
    assert training_state["global_step"] == 5
    assert training_state["materialized_from_lora"] is True
    assert training_state["contains_optimizer_state"] is False


def test_lora_stage1_reuses_current_fused_checkpoint(monkeypatch, tmp_path: Path) -> None:
    stage1_dir = tmp_path / "lora_stage1"
    checkpoint_dir = stage1_dir / "checkpoint-5"
    checkpoint_dir.mkdir(parents=True)
    _write_stage1_manifest(stage1_dir, baseline_mode="sd_ir_lora")
    (checkpoint_dir / "pytorch_lora_weights.safetensors").write_bytes(b"lora")

    calls = {"load_lora": 0}

    def _fake_load_lora(_pipeline, _path):
        calls["load_lora"] += 1

    monkeypatch.setattr(
        "src.algorithms.stable_diffusion.layout_models.StableDiffusionPipeline.from_pretrained",
        lambda *args, **kwargs: _FakeLoRAPipeline(),
    )
    monkeypatch.setattr(
        "src.algorithms.stable_diffusion.layout_models.load_lora_weights_compat",
        _fake_load_lora,
    )

    cfg = _stage2_config_for_stage1(stage1_dir, "checkpoint-5")
    _, first_info = load_stage1_pipeline_for_stage2(config=cfg)
    _, second_info = load_stage1_pipeline_for_stage2(config=cfg)

    assert calls["load_lora"] == 1
    assert first_info["materialized_unet_reused"] is False
    assert second_info["materialized_unet_reused"] is True
    assert first_info["materialized_unet_checkpoint"] == second_info["materialized_unet_checkpoint"]


def test_unet_stage1_checkpoint_does_not_materialize_lora(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stage1_dir = tmp_path / "unet_stage1"
    checkpoint_dir = stage1_dir / "checkpoint-5"
    checkpoint_dir.mkdir(parents=True)
    _write_stage1_manifest(stage1_dir, baseline_mode="sd_ir_unet")

    from safetensors.torch import save_file

    expected_unet = _tiny_unet()
    with torch.no_grad():
        expected_unet.conv_in.bias.fill_(2.0)
    save_file(
        {key: value.detach().cpu() for key, value in expected_unet.state_dict().items()},
        str(checkpoint_dir / LEGACY_ACCELERATE_MODEL_WEIGHTS),
    )

    fake_pipeline = _FakeLoRAPipeline()

    def _unexpected_materialize(*args, **kwargs):
        del args, kwargs
        raise AssertionError("LoRA materialization should not run for sd_ir_unet checkpoints")

    monkeypatch.setattr(
        "src.algorithms.stable_diffusion.layout_models.StableDiffusionPipeline.from_pretrained",
        lambda *args, **kwargs: fake_pipeline,
    )
    monkeypatch.setattr(
        "src.algorithms.stable_diffusion.layout_models.materialize_lora_as_unet_checkpoint",
        _unexpected_materialize,
    )

    pipeline, init_info = load_stage1_pipeline_for_stage2(
        config=_stage2_config_for_stage1(stage1_dir, "checkpoint-5")
    )

    assert init_info["resolved_stage1_checkpoint"] == str(checkpoint_dir)
    assert "materialized_unet_checkpoint" not in init_info
    assert torch.allclose(pipeline.unet.conv_in.bias, torch.full_like(pipeline.unet.conv_in.bias, 2.0))


def test_sd_layout_dataset_applies_square_pad_resize_geometry(tmp_path: Path) -> None:
    root_dir, annotations_path = _make_layout_dataset(tmp_path)
    dataset = StableDiffusionLayoutDataset(
        root_dir=str(root_dir),
        annotations_path=str(annotations_path),
        tokenizer=_MockTokenizer(),
        resolution=200,
        normalization_mode=UINT8_LINEAR,
        prompt_mode="class_list",
        constant_prompt="thermal image",
        thermal_scene_suffix="in thermal scene.",
        use_captions_if_available=False,
    )

    sample = dataset[0]
    assert sample["pixel_values"].shape == (3, 200, 200)
    assert sample["boxes_xyxy"].tolist() == [
        [20.0, 60.0, 60.0, 80.0],
        [100.0, 70.0, 150.0, 110.0],
    ]
    assert sample["prompt_text"] == "An image of person and car in thermal scene."


def test_build_layout_prompt_modes_and_caption_fallback() -> None:
    constant = build_layout_prompt(
        label_names=["person"],
        prompt_mode="constant",
        constant_prompt="thermal image",
        thermal_scene_suffix="in thermal scene.",
        caption=None,
        use_captions_if_available=False,
    )
    class_list = build_layout_prompt(
        label_names=["person", "person", "car"],
        prompt_mode="class_list",
        constant_prompt="thermal image",
        thermal_scene_suffix="in thermal scene.",
        caption=None,
        use_captions_if_available=False,
    )
    captioned = build_layout_prompt(
        label_names=["person"],
        prompt_mode="class_list",
        constant_prompt="thermal image",
        thermal_scene_suffix="in thermal scene.",
        caption="caption wins",
        use_captions_if_available=True,
    )

    assert constant == "thermal image"
    assert class_list == "An image of person and car in thermal scene."
    assert captioned == "caption wins"


def test_layout_tokenizer_includes_background_and_same_class_positioning() -> None:
    tokenizer = LayoutTokenizer(
        class_text_features=torch.ones(2, 4),
        layout_token_dim=4,
        bbox_fourier_dim=2,
        same_class_position_slots=4,
        use_background_token=True,
    )
    with torch.no_grad():
        tokenizer.class_projection.weight.zero_()
        tokenizer.class_projection.bias.zero_()
        tokenizer.bbox_projection[0].weight.zero_()
        tokenizer.bbox_projection[0].bias.zero_()
        tokenizer.bbox_projection[2].weight.zero_()
        tokenizer.bbox_projection[2].bias.zero_()
        tokenizer.fusion[0].weight.zero_()
        tokenizer.fusion[0].bias.zero_()
        tokenizer.fusion[2].weight.zero_()
        tokenizer.fusion[2].bias.zero_()
        tokenizer.position_embedding.weight.zero_()
        tokenizer.position_embedding.weight[1].fill_(1.0)

    tokens = tokenizer(
        boxes_xyxy_norm=torch.tensor([[[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.4, 0.4]]]),
        labels=torch.tensor([[0, 0]]),
        object_mask=torch.tensor([[True, True]]),
    )

    assert tokens.shape == (1, 3, 4)
    assert torch.allclose(tokens[0, 0], torch.zeros(4))
    assert torch.allclose(tokens[0, 1], torch.zeros(4))
    assert torch.allclose(tokens[0, 2], tokenizer.background_token[0, 0])


def test_region_token_mask_overlap_and_background() -> None:
    mask = build_region_token_mask(
        boxes_xyxy_norm=torch.tensor([[[0.0, 0.0, 0.75, 0.75], [0.5, 0.5, 1.0, 1.0]]]),
        object_mask=torch.tensor([[True, True]]),
        resolution=4,
        use_background_token=True,
    )

    overlap_index = 2 * 4 + 2
    background_index = 0 * 4 + 3
    assert mask.shape == (1, 16, 3)
    assert mask[0, overlap_index].tolist() == [True, True, False]
    assert mask[0, background_index].tolist() == [False, False, True]


def test_regiondiff_wrapper_smoke_forward() -> None:
    wrapper = _wrapped_unet()
    sample = torch.randn(1, 4, 16, 16)
    encoder_hidden_states = torch.randn(1, 8, 32)

    output = wrapper(
        sample,
        torch.tensor([10]),
        encoder_hidden_states,
        cross_attention_kwargs={
            "boxes_xyxy_norm": torch.tensor([[[0.1, 0.1, 0.4, 0.4]]]),
            "labels": torch.tensor([[0]]),
            "object_mask": torch.tensor([[True]]),
        },
        return_dict=False,
    )[0]

    assert output.shape == sample.shape
    assert wrapper.num_region_blocks > 0


def test_configure_layout_trainability_modes() -> None:
    models = SDLayoutModelComponents(
        unet=_wrapped_unet(),
        vae=torch.nn.Conv2d(1, 1, kernel_size=1),
        text_encoder=_MockTextEncoder(),
        tokenizer=_MockTokenizer(),
        noise_scheduler=DDPMScheduler(),
        weight_dtype=torch.float32,
    )
    cfg = SDLayoutTrainConfig()
    info = configure_layout_trainability(models=models, config=cfg)
    assert info["adapter_parameter_count"] > 0
    assert info["backbone_parameter_count"] == 0

    models = SDLayoutModelComponents(
        unet=_wrapped_unet(),
        vae=torch.nn.Conv2d(1, 1, kernel_size=1),
        text_encoder=_MockTextEncoder(),
        tokenizer=_MockTokenizer(),
        noise_scheduler=DDPMScheduler(),
        weight_dtype=torch.float32,
    )
    cfg = SDLayoutTrainConfig()
    cfg.training.train_mode = "adapters_plus_partial_unet"
    cfg.training.partial_unet_modules = ["mid_block"]
    info = configure_layout_trainability(models=models, config=cfg)
    assert info["backbone_parameter_count"] > 0
    assert all(name.startswith("base_unet.mid_block") for name in info["trainable_parameter_groups"]["backbone"])


def test_configure_layout_trainability_keeps_trainable_params_fp32_in_fp16_mode() -> None:
    models = SDLayoutModelComponents(
        unet=_wrapped_unet().half(),
        vae=torch.nn.Conv2d(1, 1, kernel_size=1),
        text_encoder=_MockTextEncoder(),
        tokenizer=_MockTokenizer(),
        noise_scheduler=DDPMScheduler(),
        weight_dtype=torch.float16,
    )
    cfg = SDLayoutTrainConfig()
    cfg.training.mixed_precision = "fp16"

    configure_layout_trainability(models=models, config=cfg)

    trainable_dtypes = {
        param.dtype
        for param in models.unet.parameters()
        if param.requires_grad
    }
    frozen_dtypes = {
        param.dtype
        for param in models.unet.parameters()
        if not param.requires_grad
    }

    assert trainable_dtypes == {torch.float32}
    assert torch.float16 in frozen_dtypes


def test_area_loss_weights_small_objects_higher_than_large_and_background() -> None:
    weights = build_area_weight_map(
        boxes_xyxy_norm=torch.tensor(
            [[[0.1, 0.1, 0.2, 0.2], [0.4, 0.4, 0.9, 0.9]]],
            dtype=torch.float32,
        ),
        object_mask=torch.tensor([[True, True]]),
        latent_height=16,
        latent_width=16,
        alpha=1.0,
        background_weight=0.5,
        min_weight=0.5,
        max_weight=4.0,
    )

    small_value = float(weights[0, 0, 2, 2])
    large_value = float(weights[0, 0, 8, 8])
    background_value = float(weights[0, 0, 0, 15])
    assert small_value > large_value > background_value
    assert pytest.approx(float(weights.mean()), rel=1e-5) == 1.0


def test_stage2_artifact_save_and_load_round_trip(monkeypatch, tmp_path: Path) -> None:
    wrapper = _wrapped_unet()
    config = SDLayoutTrainConfig()
    config.output.output_dir = str(tmp_path)

    save_stage2_layout_artifact(
        output_dir=str(tmp_path),
        unet=wrapper,
        config=config,
        init_info={"resolved_stage1_checkpoint": "checkpoint-123"},
        trainability_info={"train_mode": "adapters_only"},
    )

    tokenizer = _MockTokenizer()
    text_encoder = _MockTextEncoder()

    def _fake_from_pretrained(*args, **kwargs):
        return _FakePipeline(
            unet=_tiny_unet(),
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    monkeypatch.setattr(
        "src.algorithms.stable_diffusion.layout_models.StableDiffusionPipeline.from_pretrained",
        _fake_from_pretrained,
    )

    pipeline, manifest = load_stage2_layout_pipeline(stage2_dir=str(tmp_path))
    assert manifest["model_type"] == "regiondiff_sd_layout"

    output = pipeline.unet(
        torch.randn(1, 4, 16, 16),
        torch.tensor([5]),
        torch.randn(1, 8, 32),
        cross_attention_kwargs={
            "boxes_xyxy_norm": torch.tensor([[[0.1, 0.1, 0.4, 0.4]]]),
            "labels": torch.tensor([[0]]),
            "object_mask": torch.tensor([[True]]),
        },
        return_dict=False,
    )[0]
    assert output.shape == (1, 4, 16, 16)


def test_log_layout_validation_writes_generated_box_overlay() -> None:
    pipeline = _FakePipeline(
        unet=_tiny_unet(),
        tokenizer=_MockTokenizer(),
        text_encoder=_MockTextEncoder(),
    )
    accelerator = _FakeValidationAccelerator()

    validation_batch = {
        "prompt_text": ["person in thermal scene."],
        "boxes_xyxy_norm": torch.tensor([[[0.1, 0.1, 0.5, 0.5]]], dtype=torch.float32),
        "boxes_xyxy": torch.tensor([[[2.0, 2.0, 8.0, 8.0]]], dtype=torch.float32),
        "labels": torch.tensor([[0]], dtype=torch.long),
        "object_mask": torch.tensor([[True]]),
    }

    images = log_layout_validation(
        pipeline=pipeline,
        validation_batch=validation_batch,
        num_images=1,
        num_inference_steps=5,
        guidance_scale=7.5,
        device=torch.device("cpu"),
        seed=123,
        accelerator=accelerator,
        epoch=2,
        image_size=16,
        is_final=False,
    )

    assert len(images) == 1
    logged_tags = [tag for tag, *_ in accelerator.trackers[0].writer.images]
    assert "validation/generated_rgb_01" in logged_tags
    assert "validation/layout_rgb_01" in logged_tags
    assert "validation/generated_with_boxes_rgb_01" in logged_tags


def test_layout_trainer_validation_builds_pipeline_before_swapping_wrapped_unet(
    monkeypatch,
) -> None:
    wrapper = _wrapped_unet()
    accelerator = _FakeValidationAccelerator()
    captured = {}

    def _fake_from_pretrained(*args, **kwargs):
        assert "unet" not in kwargs
        pipeline = _FakePipeline(
            unet=_tiny_unet(),
            tokenizer=_MockTokenizer(),
            text_encoder=_MockTextEncoder(),
        )
        captured["pipeline"] = pipeline
        return pipeline

    monkeypatch.setattr(
        "src.algorithms.stable_diffusion.layout_training.StableDiffusionPipeline.from_pretrained",
        _fake_from_pretrained,
    )

    validation_batch = {
        "prompt_text": ["person in thermal scene."],
        "boxes_xyxy_norm": torch.tensor([[[0.1, 0.1, 0.5, 0.5]]], dtype=torch.float32),
        "boxes_xyxy": torch.tensor([[[2.0, 2.0, 8.0, 8.0]]], dtype=torch.float32),
        "labels": torch.tensor([[0]], dtype=torch.long),
        "object_mask": torch.tensor([[True]]),
    }

    trainer = LayoutTrainer(
        config=SDLayoutTrainConfig(),
        models=SimpleNamespace(
            unet=wrapper,
            vae=object(),
            text_encoder=object(),
            tokenizer=object(),
            weight_dtype=torch.float32,
        ),
        train_dataloader=None,
        validation_batch=validation_batch,
        init_info={},
        trainability_info={},
        accelerator=accelerator,
    )

    trainer._run_validation(epoch=0)

    assert captured["pipeline"].unet is wrapper


def test_stage2_save_and_load_hooks_round_trip(tmp_path: Path) -> None:
    accelerator = _FakeSaveLoadAccelerator()
    source = _wrapped_unet()
    target = _wrapped_unet()
    for parameter in target.parameters():
        with torch.no_grad():
            parameter.zero_()

    save_hook = create_stage2_save_model_hook(source, accelerator)
    weights = [source.state_dict()]
    save_hook([source], weights, str(tmp_path))

    assert weights == []

    load_hook = create_stage2_load_model_hook(target, accelerator)
    load_hook([target], str(tmp_path))

    for source_param, target_param in zip(source.parameters(), target.parameters()):
        assert torch.allclose(source_param, target_param)


def test_stage2_load_hook_accepts_legacy_accelerate_model_filename(tmp_path: Path) -> None:
    accelerator = _FakeSaveLoadAccelerator()
    source = _wrapped_unet()
    target = _wrapped_unet()
    for parameter in target.parameters():
        with torch.no_grad():
            parameter.zero_()

    from safetensors.torch import save_file

    legacy_path = tmp_path / "model.safetensors"
    save_file(
        {key: value.detach().cpu() for key, value in source.state_dict().items()},
        str(legacy_path),
    )

    load_hook = create_stage2_load_model_hook(target, accelerator)
    load_hook([target], str(tmp_path))

    for source_param, target_param in zip(source.parameters(), target.parameters()):
        assert torch.allclose(source_param, target_param)


def test_layout_trainer_resume_accepts_existing_repo_relative_checkpoint_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "layout_run"
    output_dir.mkdir()
    checkpoint_dir = output_dir / "checkpoint-1000"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "training_state.json").write_text(
        json.dumps(
            {
                "global_step": 1000,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 500,
                "max_train_steps": 2000,
                "train_mode": "adapters_only",
            }
        ),
        encoding="utf-8",
    )

    cfg = SDLayoutTrainConfig()
    cfg.output.output_dir = str(output_dir)
    cfg.training.resume_from_checkpoint = str(checkpoint_dir)
    cfg.training.max_train_steps = 2000

    accelerator = _FakeResumeAccelerator()
    trainer = LayoutTrainer(
        config=cfg,
        models=None,
        train_dataloader=None,
        validation_batch=None,
        init_info={},
        trainability_info={},
        accelerator=accelerator,
    )
    trainer.num_update_steps_per_epoch = 100

    trainer.resume_from_checkpoint()

    assert accelerator.loaded_path == str(checkpoint_dir)
    assert trainer.global_step == 1000
    assert trainer.first_epoch == 10
