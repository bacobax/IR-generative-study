"""Tests for RegionDiff attention-map distillation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers import UNet2DModel

from src.algorithms.training.regiondiff_attention_distillation import (
    AttentionMapRecord,
    RegionDiffAttentionRecorder,
    _resolve_stage2_teacher_source,
    compute_region_attention_distillation_loss,
    load_regiondiff_attention_teacher,
)
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.fm_config import DistillationConfig, FMTrainConfig, LayoutConditioningConfig
from src.models.regiondiffusion_factory import build_regiondiff_wrapper


def _tiny_unet2d() -> UNet2DModel:
    return UNet2DModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        down_block_types=("AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        attention_head_dim=8,
        norm_num_groups=8,
    )


def _region_config() -> LayoutConditioningConfig:
    return LayoutConditioningConfig(
        enabled=True,
        variant="regiondiff_v1",
        num_classes=3,
        category_id_to_name={0: "person", 1: "car", 2: "truck"},
        active_region_resolutions=[16],
        layout_token_dim=32,
        bbox_fourier_dim=4,
        same_class_position_slots=8,
        use_background_token=True,
    )


def _layout_kwargs(batch_size: int = 1):
    return {
        "boxes_xyxy_norm": torch.tensor(
            [[[0.1, 0.1, 0.5, 0.5], [0.55, 0.55, 0.9, 0.9]]] * batch_size,
            dtype=torch.float32,
        ),
        "labels": torch.tensor([[0, 1]] * batch_size, dtype=torch.long),
        "object_mask": torch.ones(batch_size, 2, dtype=torch.bool),
    }


def _record(name: str, attention: torch.Tensor, resolution: int) -> AttentionMapRecord:
    return AttentionMapRecord(
        attention=attention,
        layer_name=name,
        alias=f"layer_{resolution}",
        resolution=resolution,
    )


def test_distillation_config_defaults_and_validation() -> None:
    cfg = FMTrainConfig()

    assert cfg.distillation.enabled is False
    assert cfg.distillation.loss_type == "attention_kl"
    assert cfg.distillation.timestep_range == (0.2, 0.8)

    with pytest.raises(ValueError, match="loss_type"):
        DistillationConfig(loss_type="bad")
    with pytest.raises(ValueError, match="lambda_attn"):
        DistillationConfig(lambda_attn=-0.1)
    with pytest.raises(ValueError, match="timestep_range"):
        DistillationConfig(timestep_range=(0.9, 0.2))


def test_existing_regiondiff_baseline_config_remains_kd_disabled() -> None:
    from src.cli.train_flow_matching import _FLAT_TO_NESTED, build_parser

    preset = "configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_ot_b64_hflip.yaml"
    parser = build_parser()
    args = parser.parse_args(["--config", preset])
    cfg = merge_config_and_cli(
        FMTrainConfig,
        preset,
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
        cli_argv=["--config", preset],
    )

    assert cfg.layout_conditioning.enabled is True
    assert cfg.layout_conditioning.variant == "regiondiff_v1"
    assert cfg.distillation.enabled is False


def test_attention_recorder_captures_only_regiondiff_adapters_and_clears() -> None:
    region_cfg = _region_config()
    wrapper = build_regiondiff_wrapper(
        base_model=_tiny_unet2d(),
        region_config=region_cfg,
        category_id_to_name=region_cfg.category_id_to_name,
        backbone_kind="fm_unet2d",
        attachment_kind="attention",
    )
    sample = torch.randn(1, 4, 16, 16)

    recorder = RegionDiffAttentionRecorder(wrapper)
    with recorder:
        wrapper(sample, torch.tensor([10]), **_layout_kwargs()).sample

    assert recorder.records
    assert all("region_adapter" in key for key in recorder.records)
    assert all(record.attention.ndim == 3 for record in recorder.records.values())
    assert any(record.attention.requires_grad for record in recorder.records.values())

    recorder.clear()
    assert recorder.records == {}


def test_attention_distillation_loss_matching_resize_and_selected_categories() -> None:
    torch.manual_seed(7)
    teacher = {
        "teacher": _record("teacher", torch.rand(1, 16, 3) + 0.1, resolution=4),
    }
    student_attention = (torch.rand(1, 4, 3) + 0.1).requires_grad_(True)
    student = {
        "student": _record("student", student_attention, resolution=2),
    }
    layout = _layout_kwargs()
    cfg = DistillationConfig(
        loss_type="attention_l2",
        selected_categories=["person"],
        timestep_range=(0.2, 0.8),
    )

    loss, diagnostics = compute_region_attention_distillation_loss(
        teacher_attention_maps=teacher,
        student_attention_maps=student,
        boxes_xyxy_norm=layout["boxes_xyxy_norm"],
        labels=layout["labels"],
        object_mask=layout["object_mask"],
        timesteps=torch.tensor([0.5]),
        distillation_config=cfg,
        category_id_to_name={0: "person", 1: "car"},
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert student_attention.grad is not None
    assert diagnostics["matched_layers"] == 1
    assert diagnostics["selected_instances"] == 1
    assert diagnostics["skipped_layers_shape"] == 0


def test_attention_distillation_loss_skips_missing_and_incompatible_layers() -> None:
    layout = _layout_kwargs()
    cfg = DistillationConfig(loss_type="attention_kl")
    student = {
        "student": _record("student", torch.rand(1, 4, 3, requires_grad=True) + 0.1, resolution=2),
    }

    loss, diagnostics = compute_region_attention_distillation_loss(
        teacher_attention_maps={},
        student_attention_maps=student,
        boxes_xyxy_norm=layout["boxes_xyxy_norm"],
        labels=layout["labels"],
        object_mask=layout["object_mask"],
        timesteps=torch.tensor([0.5]),
        distillation_config=cfg,
        category_id_to_name={0: "person", 1: "car"},
    )
    assert torch.allclose(loss, torch.zeros_like(loss))
    assert diagnostics["matched_layers"] == 0
    assert diagnostics["skipped_layers_missing"] == 1

    bad_teacher = {
        "teacher": _record("teacher", torch.rand(1, 4, 1) + 0.1, resolution=2),
    }
    loss, diagnostics = compute_region_attention_distillation_loss(
        teacher_attention_maps=bad_teacher,
        student_attention_maps=student,
        boxes_xyxy_norm=layout["boxes_xyxy_norm"],
        labels=layout["labels"],
        object_mask=layout["object_mask"],
        timesteps=torch.tensor([0.5]),
        distillation_config=cfg,
        category_id_to_name={0: "person", 1: "car"},
    )
    assert torch.allclose(loss, torch.zeros_like(loss))
    assert diagnostics["matched_layers"] == 0
    assert diagnostics["skipped_layers_shape"] == 1


def test_attention_distillation_loss_handles_no_selected_instances() -> None:
    layout = _layout_kwargs()
    cfg = DistillationConfig(selected_categories=["truck"])
    teacher = {
        "teacher": _record("teacher", torch.rand(1, 4, 3) + 0.1, resolution=2),
    }
    student = {
        "student": _record("student", torch.rand(1, 4, 3, requires_grad=True) + 0.1, resolution=2),
    }

    loss, diagnostics = compute_region_attention_distillation_loss(
        teacher_attention_maps=teacher,
        student_attention_maps=student,
        boxes_xyxy_norm=layout["boxes_xyxy_norm"],
        labels=layout["labels"],
        object_mask=layout["object_mask"],
        timesteps=torch.tensor([0.5]),
        distillation_config=cfg,
        category_id_to_name={0: "person", 1: "car"},
    )

    assert torch.allclose(loss, torch.zeros_like(loss))
    assert diagnostics["selected_instances"] == 0


def test_teacher_loader_freezes_pipeline_modules(monkeypatch, tmp_path) -> None:
    from src.algorithms.stable_diffusion import layout_models

    (tmp_path / "stage2_layout_manifest.json").write_text("{}", encoding="utf-8")

    class _FakePipeline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unet = torch.nn.Linear(2, 2)
            self.unet.category_id_to_name = {0: "person"}
            self.vae = torch.nn.Linear(2, 2)
            self.text_encoder = torch.nn.Linear(2, 2)
            self.scheduler = SimpleNamespace(
                config=SimpleNamespace(num_train_timesteps=1000)
            )

        def to(self, device):
            self.unet.to(device)
            self.vae.to(device)
            self.text_encoder.to(device)
            return self

    fake_pipeline = _FakePipeline()

    def _fake_load_stage2_layout_pipeline(*, stage2_dir, torch_dtype=None, base_model=None):
        del stage2_dir, torch_dtype, base_model
        return fake_pipeline, {"prompt_mode": "constant"}

    monkeypatch.setattr(
        layout_models,
        "load_stage2_layout_pipeline",
        _fake_load_stage2_layout_pipeline,
    )

    teacher = load_regiondiff_attention_teacher(str(tmp_path), device="cpu")

    assert teacher.unet is fake_pipeline.unet
    for module in (fake_pipeline.unet, fake_pipeline.vae, fake_pipeline.text_encoder):
        assert module.training is False
        assert all(not parameter.requires_grad for parameter in module.parameters())


def test_teacher_source_resolves_latest_ongoing_stage2_checkpoint(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "flir_sd15_regiondiff_stage2_from_lora_r8_fm_comparable"
    ckpt_1000 = run_dir / "checkpoint-1000"
    ckpt_2000 = run_dir / "checkpoint-2000"
    ckpt_1000.mkdir(parents=True)
    ckpt_2000.mkdir()
    (ckpt_1000 / "regiondiff_unet_checkpoint.safetensors").write_bytes(b"old")
    (ckpt_2000 / "regiondiff_unet_checkpoint.safetensors").write_bytes(b"new")

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(
        "src.algorithms.training.regiondiff_attention_distillation._infer_stage2_config_path",
        lambda path: config_path if path == run_dir else None,
    )

    source = _resolve_stage2_teacher_source(run_dir)

    assert source.artifact_dir == run_dir
    assert source.weights == ckpt_2000 / "regiondiff_unet_checkpoint.safetensors"
    assert source.config_path == config_path
