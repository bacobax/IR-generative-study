"""Tests for the SSDLite detector backend."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.models.ssdlite import (
    SSDLiteConfig,
    SSDLiteDetector,
    build_ssdlite_model,
    generate_ssdlite_anchors,
)
from src.algorithms.training.ssdlite_detector import (
    SSDLiteLoss,
    assign_ssdlite_targets,
    decode_ssdlite_offsets,
    encode_ssdlite_offsets,
    load_ssdlite_checkpoint,
    _save_ssdlite_checkpoint,
    collect_ssdlite_predictions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _small_config(nc: int = 1) -> SSDLiteConfig:
    """Minimal SSDLiteConfig for fast CPU tests."""
    return SSDLiteConfig(
        nc=nc,
        input_channels=3,
        n_feature_maps=3,
        anchor_min_sizes=(0.07, 0.15, 0.33),
        anchor_max_sizes=(0.15, 0.33, 0.60),
        anchor_aspect_ratios=(2.0,),
        conf_threshold=0.25,
        nms_iou_threshold=0.45,
    )


class _FakeCfg:
    """Minimal cfg stub for SSDLiteLoss init."""

    def __init__(self) -> None:
        self.box_weight = 5.0
        self.class_weight = 1.0
        self.neg_pos_ratio = 3.0


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

def test_ssdlite_instantiation() -> None:
    config = _small_config()
    model = SSDLiteDetector(config)
    assert isinstance(model, torch.nn.Module)


def test_ssdlite_forward_shapes() -> None:
    config = _small_config(nc=1)
    model = SSDLiteDetector(config)
    model.eval()
    x = torch.zeros(2, 3, 256, 256)
    with torch.no_grad():
        cls_logits, bbox_pred, anchors = model(x)

    n_anchors = anchors.shape[0]
    assert cls_logits.shape == (2, n_anchors, 1), f"cls_logits shape: {cls_logits.shape}"
    assert bbox_pred.shape == (2, n_anchors, 4), f"bbox_pred shape: {bbox_pred.shape}"
    assert anchors.shape[1] == 4

    # For 3 FMs at 32×32, 16×16, 8×8 with 4 anchors/cell:
    expected = (32 * 32 + 16 * 16 + 8 * 8) * config.anchors_per_cell
    assert n_anchors == expected, f"expected {expected} anchors, got {n_anchors}"


def test_ssdlite_forward_training_mode() -> None:
    config = _small_config(nc=2)
    model = SSDLiteDetector(config)
    model.train()
    x = torch.randn(1, 3, 256, 256)
    cls_logits, bbox_pred, anchors = model(x)
    assert cls_logits.shape[2] == 2
    # Ensure gradients can flow
    cls_logits.sum().backward()


# ---------------------------------------------------------------------------
# Anchor generation tests
# ---------------------------------------------------------------------------

def test_generate_anchors_shape() -> None:
    config = _small_config()
    fm_sizes = [(32, 32), (16, 16), (8, 8)]
    anchors = generate_ssdlite_anchors(config, fm_sizes)
    expected = sum(h * w for h, w in fm_sizes) * config.anchors_per_cell
    assert anchors.shape == (expected, 4)


def test_generate_anchors_range() -> None:
    config = _small_config()
    fm_sizes = [(32, 32), (16, 16), (8, 8)]
    anchors = generate_ssdlite_anchors(config, fm_sizes)
    # cx/cy should be in (0, 1), sizes clamped to [0, 1]
    assert (anchors >= 0.0).all()
    assert (anchors <= 1.0).all()
    # Centers in (0, 1) strictly
    assert (anchors[:, 0] > 0).all() and (anchors[:, 0] < 1).all()
    assert (anchors[:, 1] > 0).all() and (anchors[:, 1] < 1).all()


# ---------------------------------------------------------------------------
# Box encoding roundtrip
# ---------------------------------------------------------------------------

def test_encode_decode_roundtrip() -> None:
    torch.manual_seed(0)
    gt_boxes = torch.rand(10, 4) * 0.5 + 0.1      # small cx/cy/w/h
    gt_boxes[:, 2:] = gt_boxes[:, 2:].clamp(0.05, 0.4)  # keep w/h positive
    anchors = torch.rand(10, 4) * 0.5 + 0.1
    anchors[:, 2:] = anchors[:, 2:].clamp(0.05, 0.4)

    encoded = encode_ssdlite_offsets(gt_boxes, anchors)
    decoded = decode_ssdlite_offsets(encoded, anchors)
    assert torch.allclose(decoded, gt_boxes.clamp(0.0, 1.0), atol=1e-5), \
        f"max error: {(decoded - gt_boxes.clamp(0,1)).abs().max():.2e}"


# ---------------------------------------------------------------------------
# Target assignment tests
# ---------------------------------------------------------------------------

def test_assign_targets_no_gt() -> None:
    anchors = torch.rand(100, 4)
    anchors[:, 2:] = 0.1
    gt_boxes = torch.zeros(0, 4)
    gt_classes = torch.zeros(0, dtype=torch.long)
    cls_tgt, loc_tgt, pos_mask = assign_ssdlite_targets(
        anchors, gt_boxes, gt_classes, nc=1
    )
    assert not pos_mask.any()
    assert (cls_tgt == -2).all()


def test_assign_targets_one_gt() -> None:
    # Place a GT box at center; nearest anchors should be positive
    n_anchors = 5376  # typical count for 3 FMs
    anchors = generate_ssdlite_anchors(_small_config(), [(32, 32), (16, 16), (8, 8)])

    gt_boxes = torch.tensor([[0.5, 0.5, 0.15, 0.30]])   # center, person-like aspect
    gt_classes = torch.tensor([0])
    cls_tgt, loc_tgt, pos_mask = assign_ssdlite_targets(
        anchors, gt_boxes, gt_classes,
        iou_pos_threshold=0.5,
        iou_neg_threshold=0.4,
        nc=1,
    )
    # At least one positive (the forced match)
    assert pos_mask.any()
    # Positive anchors have correct class
    assert (cls_tgt[pos_mask] == 0).all()


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

def test_ssdlite_loss_finite() -> None:
    config = _small_config(nc=1)
    model = SSDLiteDetector(config)
    criterion = SSDLiteLoss(_FakeCfg())

    x = torch.randn(2, 3, 256, 256)
    cls_logits, bbox_pred, anchors = model(x)

    gt_boxes = [
        torch.tensor([[0.5, 0.5, 0.2, 0.4]]),
        torch.tensor([[0.3, 0.3, 0.1, 0.2]]),
    ]
    gt_classes = [torch.tensor([0]), torch.tensor([0])]

    loss, parts = criterion(
        cls_logits, bbox_pred, anchors,
        boxes_xywh=gt_boxes, class_ids=gt_classes,
    )
    assert math.isfinite(float(loss.item())), f"Loss is not finite: {loss}"
    assert "loc_loss" in parts
    assert "conf_loss" in parts
    assert "n_pos" in parts


def test_ssdlite_loss_no_gt() -> None:
    """Loss should be finite even when all images have no GT boxes."""
    config = _small_config(nc=1)
    model = SSDLiteDetector(config)
    criterion = SSDLiteLoss(_FakeCfg())

    x = torch.randn(2, 3, 256, 256)
    cls_logits, bbox_pred, anchors = model(x)
    gt_boxes = [torch.zeros(0, 4), torch.zeros(0, 4)]
    gt_classes = [torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)]

    loss, _ = criterion(
        cls_logits, bbox_pred, anchors,
        boxes_xywh=gt_boxes, class_ids=gt_classes,
    )
    assert math.isfinite(float(loss.item()))


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip() -> None:
    from dataclasses import dataclass, field
    from src.algorithms.training.simple_yolo_detector import YoloSplitInfo

    config = _small_config(nc=1)
    model = SSDLiteDetector(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    split_info = YoloSplitInfo(
        dataset_yaml="dummy.yaml",
        split="train",
        image_dir="/tmp",
        label_dir="/tmp",
        names={0: "person"},
        nc=1,
    )

    @dataclass
    class _Cfg:
        @dataclass
        class _Data:
            batch_size: int = 2
            workers: int = 0
            image_size: int = 256
            cache_images: bool = False
            dataset_yaml: str = ""
            balanced_dataset_yaml: str = ""
            unbalanced_dataset_yaml: str = ""
            full_train_dataset_yaml: str = ""
            test_dataset_yaml: str = None

        @dataclass
        class _Model:
            backend: str = "ssdlite"
            weights: str = ""
            task: str = "detect"

            @dataclass
            class _SSDLite:
                input_channels: int = 3
                n_feature_maps: int = 3
                anchor_min_sizes: list = field(default_factory=lambda: [0.07, 0.15, 0.33])
                anchor_max_sizes: list = field(default_factory=lambda: [0.15, 0.33, 0.60])
                anchor_aspect_ratios: list = field(default_factory=lambda: [2.0])
                conf_threshold: float = 0.25
                nms_iou_threshold: float = 0.45
                iou_pos_threshold: float = 0.5
                iou_neg_threshold: float = 0.4

            @dataclass
            class _Simple:
                input_channels: int = 3
                base_channels: int = 16
                width_multiplier: float = 1.0
                channel_multipliers: list = field(default_factory=lambda: [1,2,4,8])
                blocks_per_stage: list = field(default_factory=lambda: [1,1,1,1])
                output_stride: int = 16
                boxes_per_cell: int = 2
                activation: str = "silu"
                dropout: float = 0.0

            ssdlite: _SSDLite = field(default_factory=_SSDLite)
            simple: _Simple = field(default_factory=_Simple)

        @dataclass
        class _Training:
            epochs: int = 2
            lr0: float = 1e-3
            optimizer: str = "AdamW"
            momentum: float = 0.9
            weight_decay: float = 1e-4
            seed: int = 0
            deterministic: bool = False
            patience: int = 5
            cos_lr: bool = False
            mixed_precision: str = "no"
            grad_clip_norm: float = 10.0
            val_interval: int = 1
            freeze_backbone_epochs: int = 0
            freeze_backbone_layers: Any = None
            backbone_lr_multiplier: float = 1.0
            backbone_param_prefixes: list = field(default_factory=list)
            tensorboard_image_interval: int = 1
            tensorboard_max_images: int = 2
            tensorboard_prediction_conf: float = 0.25

        @dataclass
        class _Loss:
            box_weight: float = 5.0
            giou_weight: float = 2.0
            objectness_weight: float = 1.0
            no_object_weight: float = 2.0
            class_weight: float = 1.0
            neg_pos_ratio: float = 3.0

        @dataclass
        class _Baseline:
            mode: str = "none"
            use_weighted_sampler: bool = False
            rarity_alpha: float = 1.0
            rarity_eps: float = 1e-6
            image_score_top_k: int = 3
            normalize_weights: bool = True
            clip_weight_min: float = None
            clip_weight_max: float = 10.0
            sampler_replacement: bool = True
            targeted_aug_probability: float = 0.5
            targeted_aug_rarity_quantile: float = 0.8
            translate_fraction: float = 0.05
            scale_min: float = 0.9
            scale_max: float = 1.1
            crop_scale_min: float = 0.85
            crop_scale_max: float = 1.0
            crop_min_rare_box_area_retained: float = 0.5
            crop_max_attempts: int = 10
            allow_horizontal_flip: bool = True
            seed: int = 7

        data: _Data = field(default_factory=_Data)
        model: _Model = field(default_factory=_Model)
        training: _Training = field(default_factory=_Training)
        loss: _Loss = field(default_factory=_Loss)
        baseline: _Baseline = field(default_factory=_Baseline)

        @dataclass
        class _Output:
            experiment_name: str = "test"
            runs_root: str = "/tmp"
            checkpoints_root: str = "/tmp"
            analysis_root: str = "/tmp"

        output: _Output = field(default_factory=_Output)

    cfg = _Cfg()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = Path(f.name)

    try:
        _save_ssdlite_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            split_info=split_info,
            epoch=1,
            metrics={"map50": 0.5},
        )
        loaded_model, payload = load_ssdlite_checkpoint(path, map_location="cpu")
        assert payload["format"] == "ssdlite_detector_v1"
        assert payload["nc"] == 1
        assert payload["epoch"] == 1
        assert isinstance(loaded_model, SSDLiteDetector)
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Config YAML loading
# ---------------------------------------------------------------------------

def test_base_ssdlite_yaml_loads() -> None:
    """_base_ssdlite.yaml resolves and model can be built from it."""
    from src.core.configs.config_loader import load_yaml
    from src.core.configs.yolo_experiment_config import YOLOExperimentConfig
    from src.core.configs.config_loader import merge_config_and_cli

    yaml_path = "configs/yolo/exp_v18_ssdlite/_base_ssdlite.yaml"
    # Just validate the YAML keys against the dataclass schema
    from src.cli.train_yolo import _validate_config_yaml_keys
    _validate_config_yaml_keys(yaml_path)


def test_existing_yolo_config_unchanged() -> None:
    """_base_v4.yaml still resolves to simple_torch backend."""
    from src.cli.train_yolo import _validate_config_yaml_keys
    from src.core.configs.config_loader import load_yaml

    data = _validate_config_yaml_keys("configs/yolo/exp_v18_simple_yolo_tiny/_base_v4.yaml")
    assert data.get("model", {}).get("backend") == "simple_torch"
