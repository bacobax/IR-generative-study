"""Tests for the STAY-inspired layout-conditioned flow-matching path."""

from __future__ import annotations

import json
import os
import tempfile

import torch
from torch.utils.data import DataLoader

from src.algorithms.training.layout_flow_matching_trainer import LayoutFMTrainer
from src.cli.train import _FLAT_TO_NESTED, build_parser
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.fm_config import DataConfig, FMTrainConfig, LayoutConditioningConfig, ModelConfig, OutputConfig
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
