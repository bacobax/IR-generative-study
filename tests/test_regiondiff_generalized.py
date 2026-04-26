"""Tests for generalized RegionDiff layout conditioning."""

from __future__ import annotations

import torch
from diffusers import DDPMScheduler, UNet2DModel

from src.algorithms.training.flow_matching_trainer import FlowMatchingTrainer
from src.algorithms.training.unconditional_sd_trainer import UnconditionalStableDiffusionTrainer
from src.core.configs.fm_config import LayoutConditioningConfig
from src.core.configs.sd_uncond_config import parse_args as parse_sd_uncond_args
from src.models.regiondiffusion import (
    RegionDiffAttentionBlock,
    RegionDiffusionModelWrapper,
    iter_regiondiff_adapter_parameters,
)
from src.models.regiondiffusion_factory import (
    build_identity_class_features,
    build_regiondiff_wrapper,
    configure_regiondiff_trainability,
)


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
        category_id_to_name={0: "person", 1: "car", 2: "dog"},
        active_region_resolutions=[16],
        layout_token_dim=32,
        bbox_fourier_dim=4,
        same_class_position_slots=8,
        use_background_token=True,
    )


def _layout_kwargs(batch_size: int = 1):
    return {
        "boxes_xyxy_norm": torch.tensor(
            [[[0.1, 0.1, 0.5, 0.5]]] * batch_size,
            dtype=torch.float32,
        ),
        "labels": torch.zeros(batch_size, 1, dtype=torch.long),
        "object_mask": torch.ones(batch_size, 1, dtype=torch.bool),
    }


def _layout_batch(batch_size: int = 1):
    boxes = torch.tensor([[[2.0, 2.0, 10.0, 10.0]]] * batch_size, dtype=torch.float32)
    return {
        "pixel_values": torch.zeros(batch_size, 1, 16, 16),
        "boxes_xyxy": boxes,
        "boxes_xyxy_norm": boxes / 16.0,
        "labels": torch.zeros(batch_size, 1, dtype=torch.long),
        "object_mask": torch.ones(batch_size, 1, dtype=torch.bool),
    }


class _RecordingWriter:
    def __init__(self) -> None:
        self.tags: list[str] = []
        self.shapes: dict[str, torch.Size] = {}

    def add_images(self, tag, images, step, *args, **kwargs) -> None:
        tag = str(tag)
        self.tags.append(tag)
        self.shapes[tag] = images.shape
        assert images.shape[0] == 1


class _FakeRegionDiffSampler:
    def __init__(self) -> None:
        self.saw_layout_batch = False

    def sample_euler_layout(self, batch, *, steps, sample_shape):
        self.saw_layout_batch = "boxes_xyxy_norm" in batch
        return torch.zeros(batch["pixel_values"].shape[0], 1, 16, 16)

    def decode(self, latents):
        return latents


def test_unet2d_regiondiff_wrapper_smoke_forward() -> None:
    wrapper = build_regiondiff_wrapper(
        base_model=_tiny_unet2d(),
        region_config=_region_config(),
        category_id_to_name={0: "person", 1: "car", 2: "dog"},
        backbone_kind="fm_unet2d",
        attachment_kind="attention",
    )

    assert isinstance(wrapper, RegionDiffusionModelWrapper)
    assert wrapper.num_region_blocks > 0
    assert any(isinstance(module, RegionDiffAttentionBlock) for module in wrapper.modules())

    sample = torch.randn(1, 4, 16, 16)
    output = wrapper(sample, torch.tensor([10]), **_layout_kwargs()).sample
    assert output.shape == sample.shape


def test_regiondiff_strict_load_accepts_plain_base_checkpoint(tmp_path) -> None:
    base_state = _tiny_unet2d().state_dict()
    checkpoint_path = tmp_path / "unet.pt"
    torch.save(base_state, checkpoint_path)

    wrapper = build_regiondiff_wrapper(
        base_model=_tiny_unet2d(),
        region_config=_region_config(),
        category_id_to_name={0: "person", 1: "car", 2: "dog"},
        backbone_kind="fm_unet2d",
        attachment_kind="attention",
    )
    trainer = FlowMatchingTrainer(wrapper, device="cpu")

    trainer.load_unet_weights(str(checkpoint_path), strict=True)

    assert torch.allclose(
        wrapper.base_model.conv_in.weight,
        base_state["conv_in.weight"],
    )


def test_regiondiff_trainability_adapters_only_freezes_unet2d_backbone() -> None:
    wrapper = build_regiondiff_wrapper(
        base_model=_tiny_unet2d(),
        region_config=_region_config(),
        category_id_to_name={0: "person", 1: "car", 2: "dog"},
        backbone_kind="sd_uncond_unet2d",
        attachment_kind="attention",
    )
    info = configure_regiondiff_trainability(
        wrapper=wrapper,
        train_mode="adapters_only",
        partial_backbone_modules=[],
    )

    adapter_ids = {id(param) for param in iter_regiondiff_adapter_parameters(wrapper)}
    assert info["adapter_parameter_count"] > 0
    assert info["backbone_parameter_count"] == 0
    assert all(
        (not param.requires_grad) or id(param) in adapter_ids
        for param in wrapper.parameters()
    )


def test_regiondiff_trainability_partial_backbone_unfreezes_prefix() -> None:
    wrapper = build_regiondiff_wrapper(
        base_model=_tiny_unet2d(),
        region_config=_region_config(),
        category_id_to_name={0: "person", 1: "car", 2: "dog"},
        backbone_kind="fm_unet2d",
        attachment_kind="attention",
    )
    info = configure_regiondiff_trainability(
        wrapper=wrapper,
        train_mode="adapters_plus_partial_unet",
        partial_backbone_modules=["mid_block"],
    )

    assert info["adapter_parameter_count"] > 0
    assert info["backbone_parameter_count"] > 0
    assert any(name.startswith("base_model.mid_block") for name in info["trainable_parameter_groups"]["backbone"])


def test_unconditional_sd_regiondiff_forward_pass() -> None:
    config = _region_config()
    unet = build_regiondiff_wrapper(
        base_model=_tiny_unet2d(),
        region_config=config,
        category_id_to_name=config.category_id_to_name,
        backbone_kind="sd_uncond_unet2d",
        attachment_kind="attention",
    )
    trainer = UnconditionalStableDiffusionTrainer(
        unet,
        noise_scheduler=DDPMScheduler(num_train_timesteps=10),
        diffusion_config=type(
            "_Diffusion",
            (),
            {"noise_offset": 0.0, "snr_gamma": None},
        )(),
        device="cpu",
        layout_config=config,
    )
    loss = trainer.diffusion_step(torch.randn(1, 4, 16, 16), _layout_kwargs())
    assert loss.ndim == 0


def test_sd_uncond_regiondiff_config_parses_preset_fields() -> None:
    cfg = parse_sd_uncond_args(
        [
            "--config",
            "configs/sd_uncond/train/presets/regiondiff.yaml",
            "--epochs",
            "1",
        ]
    )

    assert cfg.layout_conditioning.enabled is True
    assert cfg.layout_conditioning.variant == "regiondiff_v1"
    assert cfg.layout_conditioning.active_region_resolutions == [64, 32, 16]


def test_regiondiff_validation_logging_adds_generated_bbox_overlay() -> None:
    writer = _RecordingWriter()
    sampler = _FakeRegionDiffSampler()
    trainer = FlowMatchingTrainer(
        _tiny_unet2d(),
        device="cpu",
        layout_config=_region_config(),
    )

    trainer._log_regiondiff_validation_samples(
        writer,
        sampler=sampler,
        fixed_batch=_layout_batch(),
        epoch=0,
        steps=2,
        sample_shape=None,
        max_logged_images=1,
        save_debug_images=False,
        debug_dir="/tmp/unused-regiondiff-debug",
    )

    assert sampler.saw_layout_batch is True
    assert "fm/generated" in writer.tags
    assert "fm/generated_boxes" in writer.tags
    assert "fm/generated_layout" in writer.tags
    assert writer.shapes["fm/generated_boxes"][1] == 3


def test_identity_class_features_are_deterministic() -> None:
    features = build_identity_class_features({0: "person", 2: "dog"}, min_dim=4)
    assert features.shape == (3, 4)
    assert torch.allclose(features[0, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(features[2, :3], torch.tensor([0.0, 0.0, 1.0]))
