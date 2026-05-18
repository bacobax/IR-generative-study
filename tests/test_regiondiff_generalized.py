"""Tests for generalized RegionDiff layout conditioning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
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
        self.images: dict[str, torch.Tensor] = {}
        self.scalars: list[tuple[str, float, int]] = []

    def add_images(self, tag, images, step, *args, **kwargs) -> None:
        tag = str(tag)
        self.tags.append(tag)
        self.shapes[tag] = images.shape
        self.images[tag] = images.detach().cpu().clone()
        assert images.shape[0] == 1

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs) -> None:
        self.scalars.append((str(tag), float(scalar_value), int(global_step or 0)))

    def close(self) -> None:
        pass


class _FakeRegionDiffSampler:
    def __init__(self) -> None:
        self.saw_layout_batch = False

    def sample_euler_layout(self, batch, *, steps, sample_shape):
        self.saw_layout_batch = "boxes_xyxy_norm" in batch
        return torch.zeros(batch["pixel_values"].shape[0], 1, 16, 16)

    def decode(self, latents):
        return latents


class _FakeSDRegionDiffSampler:
    def __init__(self) -> None:
        self.saw_layout_batch = False

    def sample_layout(self, batch, *, steps, sample_shape):
        self.saw_layout_batch = "boxes_xyxy_norm" in batch
        return torch.zeros(batch["pixel_values"].shape[0], 1, 16, 16)

    def decode(self, latents):
        return latents


class _ZeroUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=4, sample_size=16)

    def forward(self, sample, timestep, **kwargs):
        del timestep, kwargs
        return SimpleNamespace(sample=torch.zeros_like(sample))


class _TrainableZeroUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(in_channels=1, sample_size=16)

    def forward(self, sample, timestep, **kwargs):
        del timestep, kwargs
        return SimpleNamespace(sample=torch.zeros_like(sample) + self.weight * 0.0)


class _EarlySampleTrainer(FlowMatchingTrainer):
    def __init__(self, *args, sampler, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._test_sampler = sampler

    def _make_sampler(self):
        return self._test_sampler

    def _save_configs(self) -> None:
        pass

    def _compute_batch_loss(self, x_fm, cond_kwargs=None):
        del cond_kwargs
        return self.unet.weight.square() + x_fm.sum() * 0.0


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


def test_unet2d_regiondiff_spatial_adapter_conditions_full_resolution() -> None:
    config = _region_config()
    config.active_region_resolutions = []
    config.use_spatial_adapter = True
    wrapper = build_regiondiff_wrapper(
        base_model=_tiny_unet2d(),
        region_config=config,
        category_id_to_name=config.category_id_to_name,
        backbone_kind="sd_uncond_unet2d",
        attachment_kind="attention",
    )

    assert wrapper.layout_input_adapter is not None
    with torch.no_grad():
        torch.nn.init.normal_(wrapper.layout_input_adapter.proj[-1].weight, std=0.05)

    sample = torch.randn(1, 4, 16, 16)
    labels = torch.zeros(1, 1, dtype=torch.long)
    object_mask = torch.ones(1, 1, dtype=torch.bool)
    left_box = torch.tensor([[[0.0, 0.0, 0.25, 0.25]]], dtype=torch.float32)
    right_box = torch.tensor([[[0.75, 0.75, 1.0, 1.0]]], dtype=torch.float32)

    with torch.no_grad():
        left = wrapper(
            sample,
            torch.tensor([10]),
            boxes_xyxy_norm=left_box,
            labels=labels,
            object_mask=object_mask,
        ).sample
        right = wrapper(
            sample,
            torch.tensor([10]),
            boxes_xyxy_norm=right_box,
            labels=labels,
            object_mask=object_mask,
        ).sample

    assert not torch.allclose(left, right)


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


def test_sd_uncond_regiondiff_strict_load_accepts_plain_base_checkpoint(tmp_path) -> None:
    base_state = _tiny_unet2d().state_dict()
    checkpoint_path = tmp_path / "unet_sd.pt"
    torch.save(base_state, checkpoint_path)

    wrapper = build_regiondiff_wrapper(
        base_model=_tiny_unet2d(),
        region_config=_region_config(),
        category_id_to_name={0: "person", 1: "car", 2: "dog"},
        backbone_kind="sd_uncond_unet2d",
        attachment_kind="attention",
    )
    trainer = UnconditionalStableDiffusionTrainer(
        wrapper,
        noise_scheduler=DDPMScheduler(num_train_timesteps=10),
        diffusion_config=type(
            "_Diffusion",
            (),
            {"noise_offset": 0.0, "snr_gamma": None},
        )(),
        device="cpu",
        layout_config=_region_config(),
    )

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


def test_regiondiff_fm_target_ot_permutation_keeps_layouts_with_targets() -> None:
    config = _region_config()
    wrapper = build_regiondiff_wrapper(
        base_model=_tiny_unet2d(),
        region_config=config,
        category_id_to_name=config.category_id_to_name,
        backbone_kind="fm_unet2d",
        attachment_kind="attention",
    )
    trainer = FlowMatchingTrainer(
        wrapper,
        device="cpu",
        layout_config=config,
        path_mode="minibatch_ot",
    )
    cond_kwargs = _layout_kwargs(batch_size=2)
    cond_kwargs["labels"] = torch.tensor([[0], [1]], dtype=torch.long)
    z0 = torch.stack([torch.ones(4, 16, 16), torch.zeros(4, 16, 16)], dim=0)
    x_fm = torch.stack([torch.zeros(4, 16, 16), torch.ones(4, 16, 16)], dim=0)

    _, permutation = trainer._match_flow_targets_with_permutation(z0, x_fm, cond_kwargs)
    aligned = trainer._permute_conditioning_kwargs(cond_kwargs, permutation, batch_size=2)

    assert torch.equal(permutation, torch.tensor([1, 0]))
    assert torch.equal(aligned["labels"], cond_kwargs["labels"].index_select(0, permutation))


def test_regiondiff_area_loss_weights_object_pixels_above_background() -> None:
    config = _region_config()
    config.area_loss_enabled = True
    trainer = FlowMatchingTrainer(
        _ZeroUNet(),
        device="cpu",
        layout_config=config,
    )
    loss = torch.ones(1, 4, 4, 4)
    weighted = trainer._apply_regiondiff_area_loss_weights(loss, _layout_kwargs())

    object_value = float(weighted[0, 0, 1, 1])
    background_value = float(weighted[0, 0, 3, 3])
    assert object_value > background_value


def test_regiondiff_fm_area_loss_sees_layouts_after_ot_permutation() -> None:
    config = _region_config()
    config.area_loss_enabled = True
    trainer = FlowMatchingTrainer(
        _ZeroUNet(),
        device="cpu",
        layout_config=config,
        path_mode="minibatch_ot",
    )
    cond_kwargs = _layout_kwargs(batch_size=2)
    cond_kwargs["labels"] = torch.tensor([[0], [1]], dtype=torch.long)
    permutation = torch.tensor([1, 0])
    recorded: dict[str, torch.Tensor] = {}

    def _fixed_match(z0, x_fm, cond_kwargs):
        del z0, cond_kwargs
        return x_fm.index_select(0, permutation), permutation

    def _record_loss_weights(loss, cond_kwargs):
        recorded["labels"] = cond_kwargs["labels"].detach().clone()
        return loss

    trainer._match_flow_targets_with_permutation = _fixed_match
    trainer._apply_regiondiff_area_loss_weights = _record_loss_weights

    loss = trainer.flow_matching_step(torch.randn(2, 4, 16, 16), cond_kwargs)

    assert torch.isfinite(loss)
    assert torch.equal(recorded["labels"], torch.tensor([[1], [0]], dtype=torch.long))


def test_fm_resume_rejects_sd_uncond_checkpoint_metadata(tmp_path) -> None:
    trainer = FlowMatchingTrainer(_tiny_unet2d(), device="cpu")
    checkpoint = {
        "unet_state": {},
        "num_train_timesteps": 1000,
        "beta_schedule": "scaled_linear",
        "prediction_type": "epsilon",
    }
    path = tmp_path / "unet_fm_epoch_1_ckpt.pt"

    with pytest.raises(ValueError, match="family mismatch"):
        trainer._validate_resume_checkpoint(checkpoint, str(path))


def test_sd_uncond_resume_rejects_fm_checkpoint_metadata(tmp_path) -> None:
    trainer = UnconditionalStableDiffusionTrainer(
        _tiny_unet2d(),
        noise_scheduler=DDPMScheduler(num_train_timesteps=10),
        diffusion_config=type(
            "_Diffusion",
            (),
            {"noise_offset": 0.0, "snr_gamma": None},
        )(),
        device="cpu",
    )
    checkpoint = {
        "unet_state": {},
        "t_scale": 1000.0,
        "train_target": "v",
    }
    path = tmp_path / "unet_sd_uncond_epoch_1_ckpt.pt"

    with pytest.raises(ValueError, match="family mismatch"):
        trainer._validate_resume_checkpoint(checkpoint, str(path))


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


def test_sd_epsilon_prediction_recovers_clean_latents_for_x0_area_loss() -> None:
    trainer = UnconditionalStableDiffusionTrainer(
        _tiny_unet2d(),
        noise_scheduler=DDPMScheduler(num_train_timesteps=10),
        diffusion_config=type(
            "_Diffusion",
            (),
            {"noise_offset": 0.0, "snr_gamma": None},
        )(),
        device="cpu",
    )
    clean_latents = torch.randn(2, 4, 16, 16)
    noise = torch.randn_like(clean_latents)
    timesteps = torch.tensor([2, 7], dtype=torch.long)
    noisy_latents = trainer.noise_scheduler.add_noise(clean_latents, noise, timesteps)

    pred_x0 = trainer._predict_x0_from_model_pred(noisy_latents, noise, timesteps)

    assert torch.allclose(pred_x0, clean_latents, atol=1e-5)


def test_sd_regiondiff_diffusion_step_adds_x0_area_loss() -> None:
    config = _region_config()
    config.area_loss_enabled = True
    config.area_x0_loss_weight = 0.25
    trainer = UnconditionalStableDiffusionTrainer(
        _ZeroUNet(),
        noise_scheduler=DDPMScheduler(num_train_timesteps=10),
        diffusion_config=type(
            "_Diffusion",
            (),
            {"noise_offset": 0.0, "snr_gamma": None},
        )(),
        device="cpu",
        layout_config=config,
    )

    trainer._compute_loss = lambda *args, **kwargs: torch.tensor(2.0)
    trainer._compute_regiondiff_x0_area_loss = lambda **kwargs: torch.tensor(4.0)

    loss = trainer.diffusion_step(torch.randn(1, 4, 16, 16), _layout_kwargs())

    assert torch.allclose(loss, torch.tensor(3.0))


def test_unconditional_sd_regiondiff_spatial_adapter_gets_first_step_gradients() -> None:
    config = _region_config()
    config.active_region_resolutions = []
    config.use_spatial_adapter = True
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
    loss.backward()

    grad = unet.layout_input_adapter.proj[-1].weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0


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


def test_regiondiff_fm_ot_flir_preset_is_fixed_quality_first() -> None:
    from src.cli.train_flow_matching import _FLAT_TO_NESTED, build_parser
    from src.core.configs.config_loader import merge_config_and_cli
    from src.core.configs.fm_config import FMTrainConfig

    preset_path = "configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_ot_b64_hflip.yaml"
    parser = build_parser()
    args = parser.parse_args(["--config", preset_path])
    cfg = merge_config_and_cli(
        FMTrainConfig,
        preset_path,
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
    )

    assert cfg.output.resume is None
    assert "flow_matching" in cfg.model.pretrained_unet_path
    assert cfg.layout_conditioning.active_region_resolutions == [64, 32, 16]
    assert cfg.layout_conditioning.use_spatial_adapter is False
    assert cfg.layout_conditioning.train_mode == "adapters_plus_partial_backbone"
    assert cfg.layout_conditioning.area_loss_enabled is True
    assert cfg.output.model_dir.endswith("_fixed")
    assert cfg.output.log_dir.endswith("_fixed")
    assert cfg.output.debug_dir.endswith("_fixed")


def test_regiondiff_sd_uncond_flir_preset_is_fixed_quality_first() -> None:
    cfg = parse_sd_uncond_args(
        [
            "--config",
            "configs/sd_uncond/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_b64_hflip.yaml",
        ]
    )

    assert cfg.output.resume is None
    assert "stable_diffusion/uncond_runs/uncond_latent" in cfg.model.pretrained_unet_path
    assert cfg.data.batch_size == 32
    assert cfg.layout_conditioning.active_region_resolutions == [64, 32, 16]
    assert cfg.layout_conditioning.use_spatial_adapter is False
    assert cfg.layout_conditioning.train_mode == "adapters_plus_partial_backbone"
    assert cfg.layout_conditioning.area_loss_enabled is True
    assert cfg.layout_conditioning.area_x0_loss_weight == 1.0
    assert cfg.sampling.sample_every == 50
    assert cfg.sampling.sample_batch_size == 2
    assert cfg.sampling.fixed_validation_examples == 2
    assert cfg.sampling.early_sanity_sample_epoch == 3
    assert cfg.sampling.save_debug_images is False
    assert cfg.output.model_dir.endswith("_fixed_x0")
    assert cfg.output.log_dir.endswith("_fixed_x0")
    assert cfg.output.debug_dir.endswith("_fixed_x0")


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
    assert "fm/generated_clean" in writer.tags
    assert "fm/generated_boxes" in writer.tags
    assert "fm/generated_layout" in writer.tags
    assert torch.allclose(writer.images["fm/generated"], writer.images["fm/generated_boxes"])
    assert not torch.allclose(writer.images["fm/generated"], writer.images["fm/generated_clean"])
    assert writer.shapes["fm/generated_boxes"][1] == 3


def test_regiondiff_early_sanity_sample_logs_bbox_overlay_on_configured_cadence(tmp_path, monkeypatch) -> None:
    import src.algorithms.training.flow_matching_trainer as fm_trainer_module

    writer = _RecordingWriter()
    sampler = _FakeRegionDiffSampler()
    monkeypatch.setattr(fm_trainer_module, "build_summary_writer", lambda _log_dir: writer)

    trainer = _EarlySampleTrainer(
        _TrainableZeroUNet(),
        sampler=sampler,
        device="cpu",
        model_dir=str(tmp_path / "model"),
        layout_config=_region_config(),
    )

    sample = {
        "pixel_values": torch.zeros(1, 16, 16),
        "boxes_xyxy": torch.tensor([[2.0, 2.0, 10.0, 10.0]]),
        "boxes_xyxy_norm": torch.tensor([[0.125, 0.125, 0.625, 0.625]]),
        "labels": torch.zeros(1, dtype=torch.long),
        "object_mask": torch.ones(1, dtype=torch.bool),
    }

    def _collate(samples):
        return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}

    dataloader = torch.utils.data.DataLoader([sample], batch_size=1, collate_fn=_collate)
    trainer.train(
        dataloader=dataloader,
        epochs=6,
        log_dir=str(tmp_path / "logs"),
        sample_every=50,
        sample_steps=2,
        sample_batch_size=1,
        fixed_validation_examples=1,
        early_sanity_sample_epoch=3,
        save_every_n_epochs=100,
        eval_every=0,
        ema_enabled=False,
        scheduler_name="none",
        debug_dir=str(tmp_path / "debug"),
    )

    assert sampler.saw_layout_batch is True
    assert writer.tags.count("fm/generated_early_sanity") == 2
    assert writer.tags.count("fm/generated_early_sanity_clean") == 2
    assert writer.tags.count("fm/generated_early_sanity_boxes") == 2
    assert writer.tags.count("fm/generated_early_sanity_layout") == 2
    assert "fm/generated" not in writer.tags


def test_sd_uncond_regiondiff_validation_logging_uses_layout_sampler_path() -> None:
    writer = _RecordingWriter()
    sampler = _FakeSDRegionDiffSampler()
    trainer = UnconditionalStableDiffusionTrainer(
        _tiny_unet2d(),
        noise_scheduler=DDPMScheduler(num_train_timesteps=10),
        diffusion_config=type(
            "_Diffusion",
            (),
            {"noise_offset": 0.0, "snr_gamma": None},
        )(),
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
    assert "sd_uncond/generated" in writer.tags
    assert "sd_uncond/generated_clean" in writer.tags
    assert "sd_uncond/generated_boxes" in writer.tags
    assert torch.allclose(writer.images["sd_uncond/generated"], writer.images["sd_uncond/generated_boxes"])
    assert not torch.allclose(writer.images["sd_uncond/generated"], writer.images["sd_uncond/generated_clean"])


def test_identity_class_features_are_deterministic() -> None:
    features = build_identity_class_features({0: "person", 2: "dog"}, min_dim=4)
    assert features.shape == (3, 4)
    assert torch.allclose(features[0, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(features[2, :3], torch.tensor([0.0, 0.0, 1.0]))
