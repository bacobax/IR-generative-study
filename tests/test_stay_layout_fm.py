"""Tests for the STAY-inspired layout-conditioned flow-matching path."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.algorithms.inference.layout_flow_matching_sampler import LayoutFlowMatchingSampler
from src.algorithms.training.layout_flow_matching_trainer import LayoutFMTrainer
from src.cli.train_flow_matching import _FLAT_TO_NESTED, build_parser
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.fm_config import DataConfig, FMTrainConfig, LayoutConditioningConfig, ModelConfig, OutputConfig
from src.core.normalization import RAW_UINT16_PERCENTILE
from src.core.data.layout_batching import collate_layout_batch
from src.core.visualization.layout_debug import draw_mask_overlays, render_mask_composite
from src.models.stay_layout_conditioned_unet import (
    EANormAdapter,
    LayoutMapAssembler,
    MaskedObjectContextBlock,
    build_stay_layout_conditioned_pixel_unet,
)


def _small_stay_layout_unet():
    config = {
        "sample_size": 16,
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": [32, 64],
        "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
        "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
        "norm_num_groups": 16,
    }
    return build_stay_layout_conditioned_pixel_unet(
        config,
        image_in_channels=1,
        num_classes=4,
        class_embed_dim=16,
        bbox_embed_dim=16,
        object_embed_dim=32,
        use_style_latent=True,
        style_latent_dim=8,
        style_seed=1234,
        mask_resolution=8,
        mask_hidden_channels=16,
        mask_threshold=0.5,
        edge_dilation=1,
        injection_mode="ea_norm",
        use_masked_context=True,
        mask_overlap_loss_weight=0.05,
        mask_sharpness_loss_weight=0.01,
        mask_activation_loss_weight=0.01,
        category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
        device="cpu",
    )


def _small_stay_layout_latent_unet():
    config = {
        "sample_size": 8,
        "in_channels": 4,
        "out_channels": 4,
        "layers_per_block": 1,
        "block_out_channels": [32, 64],
        "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
        "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
        "norm_num_groups": 16,
    }
    return build_stay_layout_conditioned_pixel_unet(
        config,
        image_in_channels=4,
        num_classes=4,
        class_embed_dim=16,
        bbox_embed_dim=16,
        object_embed_dim=32,
        use_style_latent=True,
        style_latent_dim=8,
        style_seed=1234,
        mask_resolution=8,
        mask_hidden_channels=16,
        mask_threshold=0.5,
        edge_dilation=1,
        injection_mode="ea_norm",
        use_masked_context=True,
        mask_overlap_loss_weight=0.05,
        mask_sharpness_loss_weight=0.01,
        mask_activation_loss_weight=0.01,
        category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
        device="cpu",
    )


class _FakeLatentVAE:
    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def encode(self, x: torch.Tensor):
        latents = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False)
        latents = latents.repeat(1, 4, 1, 1)
        sigma = torch.ones_like(latents) * 0.5
        return latents, sigma

    def sampling(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = z[:, :1]
        return F.interpolate(decoded, size=(16, 16), mode="bilinear", align_corners=False)


def _make_sample(idx: int, *, n_objects: int) -> dict:
    image = torch.linspace(-1.0, 1.0, steps=16 * 16, dtype=torch.float32).reshape(1, 16, 16)
    image = image.roll(shifts=idx, dims=-1)

    if n_objects == 0:
        boxes = torch.zeros(0, 4, dtype=torch.float32)
        labels = torch.zeros(0, dtype=torch.long)
        label_names: list[str] = []
    elif n_objects == 1:
        boxes = torch.tensor([[2.0, 2.0, 8.0, 10.0]], dtype=torch.float32)
        labels = torch.tensor([0], dtype=torch.long)
        label_names = ["person"]
    else:
        boxes = torch.tensor(
            [[1.0, 1.0, 12.0, 12.0], [6.0, 6.0, 10.0, 10.0]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 2], dtype=torch.long)
        label_names = ["person", "sign"]

    return {
        "pixel_values": image,
        "boxes_xyxy": boxes,
        "labels": labels,
        "image_id": f"image-{idx}",
        "file_name": f"image_{idx:03d}.npy",
        "n_objects": int(n_objects),
        "boxes_xyxy_original": boxes.clone(),
        "label_names": label_names,
    }


def test_stay_v2_conditioning_shapes_and_style_are_explicit() -> None:
    unet = _small_stay_layout_unet()
    batch = collate_layout_batch([
        _make_sample(0, n_objects=2),
        _make_sample(1, n_objects=1),
    ])

    style_a = unet.sample_style_noise(
        batch_size=2,
        max_objects=2,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )
    style_b = unet.sample_style_noise(
        batch_size=2,
        max_objects=2,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )
    assert torch.allclose(style_a, style_b)

    cond_a = unet.build_conditioning(
        boxes_xyxy_norm=batch["boxes_xyxy_norm"],
        labels=batch["labels"],
        object_mask=batch["object_mask"],
        spatial_size=(16, 16),
        style_noise=style_a,
    )
    cond_b = unet.build_conditioning(
        boxes_xyxy_norm=batch["boxes_xyxy_norm"],
        labels=batch["labels"],
        object_mask=batch["object_mask"],
        spatial_size=(16, 16),
        style_noise=style_b,
    )
    cond_zero = unet.build_conditioning(
        boxes_xyxy_norm=batch["boxes_xyxy_norm"],
        labels=batch["labels"],
        object_mask=batch["object_mask"],
        spatial_size=(16, 16),
        style_noise=torch.zeros_like(style_a),
    )

    assert cond_a["object_embeddings"].shape == (2, 2, 32)
    assert cond_a["style_latents"].shape == (2, 2, 8)
    assert cond_a["local_masks"].shape == (2, 2, 8, 8)
    assert cond_a["soft_masks_full"].shape == (2, 2, 16, 16)
    assert torch.allclose(cond_a["style_latents"], style_a)
    assert torch.allclose(cond_a["object_embeddings"], cond_b["object_embeddings"])
    assert not torch.allclose(
        cond_a["object_embeddings"][batch["object_mask"]],
        cond_zero["object_embeddings"][batch["object_mask"]],
    )


def test_stay_v2_empty_layout_returns_zero_maps_and_losses() -> None:
    unet = _small_stay_layout_unet()
    empty = collate_layout_batch([
        _make_sample(0, n_objects=0),
        _make_sample(1, n_objects=0),
    ])
    cond = unet.build_conditioning(
        boxes_xyxy_norm=empty["boxes_xyxy_norm"],
        labels=empty["labels"],
        object_mask=empty["object_mask"],
        spatial_size=(16, 16),
    )

    assert cond["object_embeddings"].shape == (2, 0, 32)
    assert cond["soft_masks_full"].shape == (2, 0, 16, 16)
    assert float(cond["semantic_map"].abs().sum().item()) == 0.0
    assert float(cond["edge_map"].abs().sum().item()) == 0.0
    assert float(cond["aux_loss"].item()) == 0.0
    assert float(cond["mask_overlap_loss"].item()) == 0.0
    assert float(cond["mask_sharpness_loss"].item()) == 0.0
    assert float(cond["mask_activation_loss"].item()) == 0.0


def test_stay_v2_supports_latent_resolution_inputs() -> None:
    unet = _small_stay_layout_latent_unet()
    batch = collate_layout_batch([
        _make_sample(0, n_objects=2),
        _make_sample(1, n_objects=1),
    ])
    sample = torch.randn(2, 4, 8, 8)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)

    out = unet(
        sample,
        t,
        boxes_xyxy_norm=batch["boxes_xyxy_norm"],
        labels=batch["labels"],
        object_mask=batch["object_mask"],
    )

    assert out.sample.shape == (2, 4, 8, 8)


def test_stay_v2_overlap_owner_prefers_smaller_object() -> None:
    assembler = LayoutMapAssembler(
        num_classes=4,
        object_embed_dim=4,
        mask_threshold=0.5,
        edge_dilation=1,
    )
    maps = assembler(
        object_embeddings=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        local_masks=torch.ones(1, 2, 4, 4),
        boxes_xyxy_norm=torch.tensor([[[0.0625, 0.0625, 0.8750, 0.8750], [0.3750, 0.3750, 0.6875, 0.6875]]]),
        labels=torch.tensor([[0, 1]], dtype=torch.long),
        object_mask=torch.tensor([[True, True]]),
        spatial_size=(16, 16),
    )

    center_owner = int(maps["non_overlap_masks_full"][0, :, 8, 8].argmax().item())
    outer_owner = int(maps["non_overlap_masks_full"][0, :, 2, 2].argmax().item())

    assert center_owner == 1
    assert outer_owner == 0
    assert float(maps["overlap_map"][0, 0, 8, 8].item()) == 1.0


def test_stay_v2_adapters_preserve_shapes_and_gradients() -> None:
    ea_norm = EANormAdapter(feature_channels=32, condition_channels=9)
    masked_context = MaskedObjectContextBlock(feature_channels=32, object_embed_dim=4)

    hidden_states = torch.randn(2, 32, 8, 8, requires_grad=True)
    conditioning = torch.randn(2, 9, 8, 8)
    object_embeddings = torch.randn(2, 2, 4, requires_grad=True)
    soft_masks_full = torch.rand(2, 2, 16, 16)
    object_mask = torch.tensor([[True, True], [True, False]])

    modulated = ea_norm(hidden_states, conditioning)
    updated, context = masked_context(
        modulated,
        object_embeddings=object_embeddings,
        soft_masks_full=soft_masks_full,
        object_mask=object_mask,
    )
    loss = updated.mean() + context.mean()
    loss.backward()

    assert updated.shape == hidden_states.shape
    assert context.shape == hidden_states.shape
    assert hidden_states.grad is not None
    assert object_embeddings.grad is not None
    assert ea_norm.to_gamma_beta[-1].weight.grad is not None
    assert masked_context.residual[-1].weight.grad is not None


def test_stay_v2_visual_helpers_render_mask_outputs() -> None:
    unet = _small_stay_layout_unet()
    batch = collate_layout_batch([
        _make_sample(0, n_objects=2),
        _make_sample(1, n_objects=1),
    ])
    cond = unet.build_conditioning(
        boxes_xyxy_norm=batch["boxes_xyxy_norm"],
        labels=batch["labels"],
        object_mask=batch["object_mask"],
        spatial_size=(16, 16),
        style_noise=torch.zeros(2, 2, 8),
    )
    display = ((batch["pixel_values"] + 1.0) / 2.0).clamp(0.0, 1.0)
    mask_composite = render_mask_composite(
        cond["soft_masks_full"],
        object_mask=batch["object_mask"],
        labels=batch["labels"],
    )
    mask_overlay = draw_mask_overlays(
        display,
        masks=cond["hard_masks_full"],
        object_mask=batch["object_mask"],
        labels=batch["labels"],
    )

    assert mask_composite.shape == (2, 3, 16, 16)
    assert mask_overlay.shape == (2, 3, 16, 16)


def test_stay_conditional_ot_keeps_condition_order_when_layout_cost_dominates() -> None:
    unet = _small_stay_layout_unet()
    trainer = LayoutFMTrainer(
        unet,
        layout_config=LayoutConditioningConfig(
            enabled=True,
            variant="stay_v2",
            num_classes=4,
            category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
            class_embed_dim=16,
            bbox_embed_dim=16,
            object_embed_dim=32,
            use_style_latent=True,
            style_latent_dim=8,
            style_seed=99,
            mask_resolution=8,
            mask_hidden_channels=16,
            mask_threshold=0.5,
            edge_dilation=1,
            injection_mode="ea_norm",
            use_masked_context=True,
            log_internal_maps=True,
        ),
        device="cpu",
        t_scale=1.0,
        model_dir="/tmp/stay_layout_fm_test",
        path_mode="conditional_ot",
        condition_weight=100.0,
    )
    batch = collate_layout_batch([
        _make_sample(0, n_objects=1),
        _make_sample(1, n_objects=2),
    ])
    cond_kw = trainer.prepare_conditioning_kwargs(batch, device="cpu")
    z0 = torch.stack([torch.ones(1, 16, 16), torch.zeros(1, 16, 16)], dim=0)
    x_fm = torch.stack([torch.zeros(1, 16, 16), torch.ones(1, 16, 16)], dim=0)

    matched = trainer._match_flow_targets(z0, x_fm, cond_kw)

    assert torch.allclose(matched[0], x_fm[0])
    assert torch.allclose(matched[1], x_fm[1])


def test_stay_conditional_ot_permutation_keeps_layouts_with_targets() -> None:
    unet = _small_stay_layout_unet()
    trainer = LayoutFMTrainer(
        unet,
        layout_config=LayoutConditioningConfig(
            enabled=True,
            variant="stay_v2",
            num_classes=4,
            category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
        ),
        device="cpu",
        t_scale=1.0,
        model_dir="/tmp/stay_layout_fm_test",
        path_mode="conditional_ot",
        condition_weight=0.0,
    )
    batch = collate_layout_batch([
        _make_sample(0, n_objects=1),
        _make_sample(1, n_objects=2),
    ])
    cond_kw = trainer.prepare_conditioning_kwargs(batch, device="cpu")
    z0 = torch.stack([torch.ones(1, 16, 16), torch.zeros(1, 16, 16)], dim=0)
    x_fm = torch.stack([torch.zeros(1, 16, 16), torch.ones(1, 16, 16)], dim=0)

    _, permutation = trainer._match_flow_targets_with_permutation(z0, x_fm, cond_kw)
    aligned = trainer._permute_conditioning_kwargs(cond_kw, permutation, batch_size=2)

    assert torch.equal(permutation, torch.tensor([1, 0]))
    assert torch.equal(aligned["object_mask"], cond_kw["object_mask"].index_select(0, permutation))
    assert torch.allclose(aligned["boxes_xyxy_norm"], cond_kw["boxes_xyxy_norm"].index_select(0, permutation))


def test_stay_v2_training_scales_unet_timesteps_from_config_t_scale() -> None:
    batch = collate_layout_batch([
        _make_sample(0, n_objects=1),
        _make_sample(1, n_objects=2),
    ])
    cond_kw = {
        "boxes_xyxy_norm": batch["boxes_xyxy_norm"],
        "labels": batch["labels"],
        "object_mask": batch["object_mask"],
        "style_noise": torch.zeros(2, 2, 8),
    }
    seen_timesteps: dict[float, torch.Tensor] = {}

    for t_scale in (100.0, 500.0):
        unet = _small_stay_layout_unet()
        recorded: list[torch.Tensor] = []

        def recording_forward(sample, timestep, **kwargs):
            recorded.append(timestep.detach().cpu())
            return SimpleNamespace(sample=torch.zeros_like(sample))

        unet.forward = recording_forward
        trainer = LayoutFMTrainer(
            unet,
            layout_config=LayoutConditioningConfig(
                enabled=True,
                variant="stay_v2",
                num_classes=4,
                category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
            ),
            device="cpu",
            t_scale=t_scale,
            model_dir="/tmp/stay_layout_fm_test",
        )

        torch.manual_seed(123)
        loss = trainer.flow_matching_step(batch["pixel_values"], cond_kw)

        assert torch.isfinite(loss)
        assert len(recorded) == 1
        seen_timesteps[t_scale] = recorded[0]

    assert torch.allclose(seen_timesteps[500.0], seen_timesteps[100.0] * 5.0)


def test_stay_v2_sampler_scales_unet_timesteps_from_config_t_scale() -> None:
    batch = collate_layout_batch([
        _make_sample(0, n_objects=1),
        _make_sample(1, n_objects=2),
    ])
    batch["style_noise"] = torch.zeros(2, 2, 8)
    seen_timesteps: dict[float, torch.Tensor] = {}

    for t_scale in (100.0, 500.0):
        unet = _small_stay_layout_unet()
        recorded: list[torch.Tensor] = []

        def recording_forward(sample, timestep, **kwargs):
            recorded.append(timestep.detach().cpu())
            return SimpleNamespace(sample=torch.zeros_like(sample))

        unet.forward = recording_forward
        sampler = LayoutFlowMatchingSampler(
            unet,
            device="cpu",
            t_scale=t_scale,
            sample_shape=(1, 16, 16),
        )

        z = sampler.sample_euler_layout(batch, steps=2)

        assert z.shape == (2, 1, 16, 16)
        assert len(recorded) == 2
        seen_timesteps[t_scale] = torch.stack(recorded)

    expected_100 = torch.tensor([[0.0, 0.0], [50.0, 50.0]])
    expected_500 = torch.tensor([[0.0, 0.0], [250.0, 250.0]])
    assert torch.allclose(seen_timesteps[100.0], expected_100)
    assert torch.allclose(seen_timesteps[500.0], expected_500)


def test_stay_v2_trainer_smoke_loop_and_config_loading() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--config",
        "configs/fm/train/presets/stay_layout_pixel_flir_v2_tiny.yaml",
    ])
    cfg = merge_config_and_cli(
        FMTrainConfig,
        args.config,
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
    )
    assert cfg.layout_conditioning.enabled is True
    assert cfg.layout_conditioning.variant == "stay_v2"
    assert cfg.layout_conditioning.use_style_latent is True
    assert cfg.data.max_train_samples == 8
    assert cfg.trainer_name == "layout_fm"

    unet = _small_stay_layout_unet()
    samples = [_make_sample(0, n_objects=2), _make_sample(1, n_objects=1)]
    loader = DataLoader(samples, batch_size=2, shuffle=False, collate_fn=collate_layout_batch)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = LayoutFMTrainer(
            unet,
            layout_config=LayoutConditioningConfig(
                enabled=True,
                variant="stay_v2",
                num_classes=4,
                category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
                class_embed_dim=16,
                bbox_embed_dim=16,
                object_embed_dim=32,
                use_style_latent=True,
                style_latent_dim=8,
                style_seed=99,
                mask_resolution=8,
                mask_hidden_channels=16,
                mask_threshold=0.5,
                edge_dilation=1,
                injection_mode="ea_norm",
                use_masked_context=True,
                log_internal_maps=True,
            ),
            device="cpu",
            t_scale=1.0,
            model_dir=tmpdir,
            unet_config=unet.base_unet_config,
        )

        fixed_a = trainer._attach_fixed_style_noise(collate_layout_batch(samples))
        fixed_b = trainer._attach_fixed_style_noise(collate_layout_batch(samples))
        assert "style_noise" in fixed_a
        assert torch.allclose(fixed_a["style_noise"], fixed_b["style_noise"])

        cond_kw = trainer.prepare_conditioning_kwargs(fixed_a, device="cpu")
        loss = trainer.flow_matching_step(fixed_a["pixel_values"], cond_kw)
        loss.backward()
        assert trainer.unet.object_encoder.class_embedding.weight.grad is not None
        assert trainer.unet.mask_predictor.seed_projection.weight.grad is not None
        assert trainer.unet.ea_norm_adapters["stem"].to_gamma_beta[-1].weight.grad is not None

        trainer.train(
            dataloader=loader,
            epochs=1,
            eval_dataloader=loader,
            log_dir=os.path.join(tmpdir, "tb"),
            debug_dir=os.path.join(tmpdir, "debug"),
            lr=1e-4,
            sample_every=0,
            sample_steps=2,
            save_every_n_epochs=1,
            scalar_every_steps=1,
            image_every_steps=1,
            max_logged_images=2,
            fixed_validation_examples=2,
            sample_every_steps=1,
            save_debug_images=True,
            log_internal_maps=True,
        )

        metadata_path = os.path.join(tmpdir, "layout_conditioning.json")
        assert os.path.isfile(os.path.join(tmpdir, "UNET", "config.json"))
        assert os.path.isfile(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["variant"] == "stay_v2"
        assert any(name.startswith("events.out.tfevents") for name in os.listdir(os.path.join(tmpdir, "tb")))
        debug_root = os.path.join(tmpdir, "debug")
        assert os.path.isdir(debug_root)
        step_dirs = [os.path.join(debug_root, name) for name in os.listdir(debug_root)]
        assert any(os.path.isdir(path) for path in step_dirs)
        saved_debug_files = []
        for path in step_dirs:
            if os.path.isdir(path):
                saved_debug_files.extend(os.listdir(path))
        assert any(name.startswith("generated_masks") for name in saved_debug_files)


def test_stay_v2_latent_scalar_logging_uses_latent_spatial_size() -> None:
    unet = _small_stay_layout_latent_unet()
    recorded_spatial_sizes: list[tuple[int, int]] = []
    original_build_conditioning = unet.build_conditioning

    def recording_build_conditioning(*args, **kwargs):
        recorded_spatial_sizes.append(tuple(kwargs["spatial_size"]))
        return original_build_conditioning(*args, **kwargs)

    unet.build_conditioning = recording_build_conditioning
    samples = [_make_sample(0, n_objects=2)]
    loader = DataLoader(samples, batch_size=1, shuffle=False, collate_fn=collate_layout_batch)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = LayoutFMTrainer(
            unet,
            layout_config=LayoutConditioningConfig(
                enabled=True,
                variant="stay_v2",
                num_classes=4,
                category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
                class_embed_dim=16,
                bbox_embed_dim=16,
                object_embed_dim=32,
                use_style_latent=True,
                style_latent_dim=8,
                mask_resolution=8,
                mask_hidden_channels=16,
                mask_threshold=0.5,
                edge_dilation=1,
                injection_mode="ea_norm",
                use_masked_context=True,
                log_internal_maps=True,
            ),
            device="cpu",
            t_scale=1.0,
            model_dir=tmpdir,
            unet_config=unet.base_unet_config,
            vae=_FakeLatentVAE(),
        )
        trainer.train(
            dataloader=loader,
            epochs=1,
            log_dir=os.path.join(tmpdir, "tb"),
            debug_dir=os.path.join(tmpdir, "debug"),
            lr=1e-4,
            sample_every=0,
            save_every_n_epochs=0,
            scalar_every_steps=1,
            image_every_steps=0,
            fixed_validation_examples=0,
            sample_every_steps=0,
            ema_enabled=False,
        )

    assert recorded_spatial_sizes
    assert set(recorded_spatial_sizes) == {(8, 8)}


def test_stay_v2_latent_from_config_builds_frozen_vae_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        unet_json = os.path.join(tmpdir, "tiny_latent_unet.json")
        vae_json = os.path.join(tmpdir, "tiny_vae.json")
        with open(unet_json, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "sample_size": 16,
                        "in_channels": 4,
                        "out_channels": 4,
                        "layers_per_block": 1,
                        "block_out_channels": [32, 64],
                    "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
                    "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
                    "norm_num_groups": 16,
                },
                handle,
            )
        with open(vae_json, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "attention_levels": [False, False],
                        "in_channels": 1,
                        "latent_channels": 4,
                        "num_channels": [16, 32],
                        "num_res_blocks": 1,
                        "norm_num_groups": 8,
                        "out_channels": 1,
                        "spatial_dims": 2,
                    },
                    handle,
                )

        cfg = FMTrainConfig(
            data=DataConfig(image_size=32),
            model=ModelConfig(
                unet_config=unet_json,
                vae_config=vae_json,
                vae_weights=None,
            ),
            layout_conditioning=LayoutConditioningConfig(
                enabled=True,
                variant="stay_v2",
                num_classes=4,
                category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
                class_embed_dim=16,
                bbox_embed_dim=16,
                object_embed_dim=32,
                use_style_latent=True,
                style_latent_dim=8,
                style_seed=99,
                mask_resolution=8,
                mask_hidden_channels=16,
                mask_threshold=0.5,
                edge_dilation=1,
                injection_mode="ea_norm",
                use_masked_context=True,
            ),
            output=OutputConfig(model_dir=tmpdir),
            trainer_name="layout_fm",
            device="cpu",
        )

        trainer = LayoutFMTrainer.from_config(cfg)
        assert trainer.vae is not None
        assert all(not param.requires_grad for param in trainer.vae.parameters())
        assert trainer.unet.config.sample_size == 16


def test_stay_v2_latent_from_config_supports_pretrained_sd15_vae(monkeypatch) -> None:
    fake_vae_cfg = {
        "_backend": "diffusers_autoencoder_kl",
        "latent_channels": 4,
        "block_out_channels": [128, 256, 512, 512],
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "pretrained_subfolder": "vae",
    }

    class _FakeSDVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def encode(self, x):
            return x, torch.ones_like(x)

        def sampling(self, mu, sigma):
            return mu

        def decode(self, z):
            return z

    monkeypatch.setattr(
        "src.models.vae.load_diffusers_vae_config",
        lambda *args, **kwargs: dict(fake_vae_cfg),
    )
    monkeypatch.setattr(
        "src.models.vae.build_vae_from_config",
        lambda config, device=None: _FakeSDVAE(),
    )

    cfg = FMTrainConfig(
        data=DataConfig(image_size=32),
        model=ModelConfig(
            unet_config="configs/models/fm/stable_unet_config.json",
            vae_config=None,
            vae_weights=None,
            vae_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            vae_pretrained_subfolder="vae",
        ),
        layout_conditioning=LayoutConditioningConfig(
            enabled=True,
            variant="stay_v2",
            num_classes=4,
            category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
            class_embed_dim=16,
            bbox_embed_dim=16,
            object_embed_dim=32,
            use_style_latent=True,
            style_latent_dim=8,
            style_seed=99,
            mask_resolution=8,
            mask_hidden_channels=16,
            mask_threshold=0.5,
            edge_dilation=1,
            injection_mode="ea_norm",
            use_masked_context=True,
        ),
        output=OutputConfig(model_dir="/tmp/stay_layout_sd15_vae"),
        trainer_name="layout_fm",
        device="cpu",
    )

    trainer = LayoutFMTrainer.from_config(cfg)
    assert trainer.vae is not None
    assert trainer.unet.config.sample_size == 4


def test_stay_v18_latent_preset_resolves_named_dataset_percentile_path() -> None:
    from src.cli.train_flow_matching import _resolve_training_data

    preset_path = Path("configs/fm/train/presets/stay_layout_latent_v18_x4_512.yaml")
    parser = build_parser()
    args = parser.parse_args(["--config", str(preset_path)])
    cfg = merge_config_and_cli(
        FMTrainConfig,
        str(preset_path),
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
    )

    resolved = _resolve_training_data(cfg)

    assert cfg.data.dataset_id == "v18"
    assert cfg.layout_conditioning.enabled is True
    assert cfg.layout_conditioning.variant == "stay_v2"
    assert resolved.normalization_mode == RAW_UINT16_PERCENTILE
    assert resolved.train_dir.endswith("data/raw/v18/train")
    assert resolved.val_dir.endswith("data/raw/v18/val")
    assert resolved.train_annotations_path.endswith("data/raw/v18/train/annotations.json")
    assert resolved.val_annotations_path.endswith("data/raw/v18/val/annotations.json")


def test_stay_v18_sd15ft_preset_points_to_saved_vae_artifact() -> None:
    from src.algorithms.training.flow_matching_trainer import _resolve_unet_sample_size
    from src.models.vae import resolve_vae_config_from_model_config

    preset_path = Path("configs/fm/train/presets/stay_layout_latent_v18_sd15ft_x8_256.yaml")
    parser = build_parser()
    args = parser.parse_args(["--config", str(preset_path)])
    cfg = merge_config_and_cli(
        FMTrainConfig,
        str(preset_path),
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
    )

    vae_cfg = resolve_vae_config_from_model_config(cfg.model)
    sample_size = _resolve_unet_sample_size(cfg, vae_cfg)

    assert cfg.data.dataset_id == "v18"
    assert cfg.model.vae_config.endswith("artifacts/checkpoints/vae/vae_runs/v18_sd15_vae_x8_256/VAE/config.json")
    assert cfg.model.vae_weights.endswith("artifacts/checkpoints/vae/vae_runs/v18_sd15_vae_x8_256/VAE/vae_best.pt")
    assert vae_cfg["_backend"] == "diffusers_autoencoder_kl"
    assert vae_cfg["pretrained_model_name_or_path"] == "runwayml/stable-diffusion-v1-5"
    assert int(vae_cfg["latent_channels"]) == 4
    assert sample_size == 32


def test_stay_flir_sd15_t_scale_sweep_presets_are_isolated_batch8_runs() -> None:
    parser = build_parser()
    expected = {
        "configs/fm/train/presets/stay_layout_latent_flir_sd15_512_t100_b8.yaml": 100.0,
        "configs/fm/train/presets/stay_layout_latent_flir_sd15_512_t500_b8.yaml": 500.0,
    }

    for preset_path, t_scale in expected.items():
        args = parser.parse_args(["--config", preset_path])
        cfg = merge_config_and_cli(
            FMTrainConfig,
            preset_path,
            parser,
            args,
            flat_to_nested=_FLAT_TO_NESTED,
        )

        suffix = f"t{int(t_scale)}_b8"
        assert cfg.data.batch_size == 8
        assert cfg.training.t_scale == t_scale
        assert cfg.layout_conditioning.enabled is True
        assert cfg.layout_conditioning.variant == "stay_v2"
        assert cfg.trainer_name == "layout_fm"
        assert cfg.output.resume is None
        assert suffix in cfg.output.model_dir
        assert suffix in cfg.output.log_dir
        assert suffix in cfg.output.debug_dir
