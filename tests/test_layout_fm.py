"""Tests for layout-conditioned pixel-space flow matching."""

from __future__ import annotations

import os
import tempfile

import torch
from torch.utils.data import DataLoader

from src.algorithms.inference.layout_flow_matching_sampler import LayoutFlowMatchingSampler
from src.algorithms.training.layout_flow_matching_trainer import LayoutFMTrainer
from src.cli.train import _FLAT_TO_NESTED, build_parser
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.fm_config import FMTrainConfig, LayoutConditioningConfig
from src.core.data.layout_batching import collate_layout_batch
from src.core.visualization.layout_debug import (
    draw_bbox_overlays,
    make_side_by_side_panel,
    render_class_layout,
    save_image_batch,
)
from src.models.layout_conditioned_unet import build_layout_conditioned_pixel_unet


def _small_layout_unet():
    config = {
        "sample_size": 16,
        "in_channels": 5,
        "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": [32, 64],
        "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
        "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
        "norm_num_groups": 16,
    }
    return build_layout_conditioned_pixel_unet(
        config,
        image_in_channels=1,
        num_classes=4,
        class_embed_dim=16,
        bbox_embed_dim=16,
        spatial_channels=4,
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
            [[1.0, 1.0, 6.0, 6.0], [8.0, 4.0, 14.0, 12.0]],
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


def test_layout_collate_pads_and_normalizes() -> None:
    batch = collate_layout_batch([
        _make_sample(0, n_objects=2),
        _make_sample(1, n_objects=0),
    ])

    assert batch["pixel_values"].shape == (2, 1, 16, 16)
    assert batch["boxes_xyxy"].shape == (2, 2, 4)
    assert batch["boxes_xyxy_norm"].shape == (2, 2, 4)
    assert batch["labels"].shape == (2, 2)
    assert batch["object_mask"].shape == (2, 2)
    assert batch["object_mask"][0].tolist() == [True, True]
    assert batch["object_mask"][1].tolist() == [False, False]
    assert torch.allclose(
        batch["boxes_xyxy_norm"][0, 0],
        torch.tensor([1.0 / 16.0, 1.0 / 16.0, 6.0 / 16.0, 6.0 / 16.0]),
    )


def test_layout_conditioned_unet_uses_conditioning() -> None:
    unet = _small_layout_unet()
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([0.2, 0.7], dtype=torch.float32)

    batch = collate_layout_batch([
        _make_sample(0, n_objects=2),
        _make_sample(1, n_objects=1),
    ])
    empty = collate_layout_batch([
        _make_sample(0, n_objects=0),
        _make_sample(1, n_objects=0),
    ])

    out_cond = unet(
        x,
        t,
        boxes_xyxy_norm=batch["boxes_xyxy_norm"],
        labels=batch["labels"],
        object_mask=batch["object_mask"],
        return_layout_debug=True,
    )
    out_empty = unet(
        x,
        t,
        boxes_xyxy_norm=empty["boxes_xyxy_norm"],
        labels=empty["labels"],
        object_mask=empty["object_mask"],
    )

    assert out_cond.sample.shape == (2, 1, 16, 16)
    assert out_cond.conditioning_maps is not None
    assert out_cond.objectness_map is not None
    assert out_cond.conditioning_maps.shape == (2, 4, 16, 16)
    assert out_cond.objectness_map.shape == (2, 1, 16, 16)
    assert not torch.allclose(out_cond.sample, out_empty.sample)


def test_layout_trainer_backprops_into_layout_encoder() -> None:
    unet = _small_layout_unet()
    trainer = LayoutFMTrainer(
        unet,
        layout_config=LayoutConditioningConfig(enabled=True, num_classes=4),
        device="cpu",
        t_scale=1.0,
        model_dir="/tmp/layout_fm_test",
    )
    batch = collate_layout_batch([
        _make_sample(0, n_objects=2),
        _make_sample(1, n_objects=1),
    ])
    x_fm = batch["pixel_values"]
    cond_kw = trainer.prepare_conditioning_kwargs(batch, device="cpu")

    loss = trainer.flow_matching_step(x_fm, cond_kw)
    loss.backward()

    grad = trainer.unet.class_embedding.weight.grad
    assert grad is not None
    assert float(grad.abs().sum().item()) > 0.0


def test_layout_sampler_and_visual_panels_save() -> None:
    unet = _small_layout_unet()
    sampler = LayoutFlowMatchingSampler(unet, device="cpu", t_scale=1.0)
    batch = collate_layout_batch([
        _make_sample(0, n_objects=2),
        _make_sample(1, n_objects=1),
    ])

    z = sampler.sample_euler_layout(batch, steps=3, sample_shape=(1, 16, 16))
    assert z.shape == (2, 1, 16, 16)

    generated = ((z.clamp(-1.0, 1.0) + 1.0) / 2.0).detach().cpu()
    overlay = draw_bbox_overlays(
        generated,
        boxes_xyxy=batch["boxes_xyxy"],
        labels=batch["labels"],
        object_mask=batch["object_mask"],
    )
    class_layout = render_class_layout(
        boxes_xyxy=batch["boxes_xyxy"],
        labels=batch["labels"],
        object_mask=batch["object_mask"],
        image_size=16,
    )
    panel = make_side_by_side_panel([class_layout, generated, overlay])
    assert panel.shape[0] == 2

    with tempfile.TemporaryDirectory() as tmpdir:
        saved = save_image_batch(panel, output_dir=tmpdir, prefix="panel")
        assert len(saved) == 2
        assert all(os.path.isfile(path) for path in saved)


def test_layout_trainer_smoke_loop_and_config_loading() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--config",
        "configs/fm/train/presets/stay_layout_pixel_flir_tiny.yaml",
    ])
    cfg = merge_config_and_cli(
        FMTrainConfig,
        args.config,
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
    )
    assert cfg.layout_conditioning.enabled is True
    assert cfg.data.max_train_samples == 8
    assert cfg.training.lr == 1e-4
    assert cfg.trainer_name == "layout_fm"

    unet = _small_layout_unet()
    samples = [_make_sample(0, n_objects=2), _make_sample(1, n_objects=1)]
    loader = DataLoader(samples, batch_size=2, shuffle=False, collate_fn=collate_layout_batch)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = LayoutFMTrainer(
            unet,
            layout_config=LayoutConditioningConfig(
                enabled=True,
                num_classes=4,
                category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
            ),
            device="cpu",
            t_scale=1.0,
            model_dir=tmpdir,
            unet_config=unet.base_unet_config,
        )
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

        assert os.path.isfile(os.path.join(tmpdir, "UNET", "config.json"))
        assert os.path.isfile(os.path.join(tmpdir, "layout_conditioning.json"))
        assert any(name.startswith("events.out.tfevents") for name in os.listdir(os.path.join(tmpdir, "tb")))
        debug_root = os.path.join(tmpdir, "debug")
        assert os.path.isdir(debug_root)
        assert any(os.path.isdir(os.path.join(debug_root, name)) for name in os.listdir(debug_root))


def test_layout_trainer_epoch_image_logging_uses_sample_every() -> None:
    class RecordingLayoutFMTrainer(LayoutFMTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.training_log_steps: list[int] = []
            self.validation_log_steps: list[int] = []

        def _log_training_visuals(self, writer, *, batch, cond_kw, global_step, max_logged_images, log_internal_maps):
            self.training_log_steps.append(int(global_step))

        def _log_fixed_validation_samples(
            self,
            writer,
            *,
            sampler,
            fixed_batch,
            global_step,
            steps,
            sample_shape,
            max_logged_images,
            save_debug_images,
            debug_dir,
            log_internal_maps,
        ):
            self.validation_log_steps.append(int(global_step))

    unet = _small_layout_unet()
    samples = [_make_sample(0, n_objects=2), _make_sample(1, n_objects=1)]
    loader = DataLoader(samples, batch_size=1, shuffle=False, collate_fn=collate_layout_batch)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = RecordingLayoutFMTrainer(
            unet,
            layout_config=LayoutConditioningConfig(
                enabled=True,
                num_classes=4,
                category_id_to_name={0: "person", 1: "car", 2: "sign", 3: "light"},
            ),
            device="cpu",
            t_scale=1.0,
            model_dir=tmpdir,
            unet_config=unet.base_unet_config,
        )
        trainer.train(
            dataloader=loader,
            epochs=3,
            eval_dataloader=loader,
            log_dir=os.path.join(tmpdir, "tb"),
            debug_dir=os.path.join(tmpdir, "debug"),
            lr=1e-4,
            sample_every=2,
            sample_steps=2,
            save_every_n_epochs=0,
            scalar_every_steps=0,
            image_every_steps=1,
            max_logged_images=2,
            fixed_validation_examples=1,
            sample_every_steps=1,
            save_debug_images=False,
            log_internal_maps=False,
        )

        assert trainer.training_log_steps == [4]
        assert trainer.validation_log_steps == [4]
