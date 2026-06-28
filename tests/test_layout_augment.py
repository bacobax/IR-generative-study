"""Tests for bbox-aware LayoutScheduledAugment + dropout passthrough.

Run with::

    python -m pytest tests/test_layout_augment.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.data.transforms import (  # noqa: E402
    LayoutScheduledAugment,
    _hflip_boxes,
    _rotate_boxes_90,
    _ramped_value,
    horizontal_flip,
    rotate_90,
)


def _bright_bbox(img: torch.Tensor) -> tuple[float, float, float, float]:
    """Tight (x1,y1,x2,y2) bounding box of the bright (>0.5) region."""
    ys, xs = torch.where(img[0] > 0.5)
    return (xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1)


def _make_image_with_box(size: int = 16):
    img = torch.zeros(1, size, size)
    img[0, 10:14, 2:6] = 1.0  # rows 10..13, cols 2..5
    box = torch.tensor([[2.0, 10.0, 6.0, 14.0]])  # x1,y1,x2,y2
    return img, box


# ── geometric correctness: transformed box matches the moved bright region ──
def test_hflip_box_matches_image():
    img, box = _make_image_with_box()
    fimg = horizontal_flip(img)
    fbox = _hflip_boxes(box, size=16)
    assert _bright_bbox(fimg) == tuple(fbox[0].tolist())


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rot90_box_matches_image(k):
    img, box = _make_image_with_box()
    rimg = rotate_90(img, k)
    rbox = _rotate_boxes_90(box, k=k, size=16)
    assert _bright_bbox(rimg) == tuple(rbox[0].tolist())


def test_rot90_four_times_identity():
    _, box = _make_image_with_box()
    b = box
    for _ in range(4):
        b = _rotate_boxes_90(b, k=1, size=16)
    assert torch.allclose(b, box)


def test_hflip_twice_identity():
    _, box = _make_image_with_box()
    assert torch.allclose(_hflip_boxes(_hflip_boxes(box, size=16), size=16), box)


def test_crop_drops_out_of_view_box_and_filters_labels():
    size = 16
    img = torch.zeros(1, size, size)
    box = torch.tensor([[0.0, 0.0, 2.0, 2.0], [12.0, 12.0, 16.0, 16.0]])
    labels = torch.tensor([0, 0])
    aug = LayoutScheduledAugment(total_epochs=10, p_crop_max=1.0, crop_min_scale=0.5,
                                 crop_min_visibility=0.5, p_hflip_max=0.0, p_rot_max=0.0,
                                 p_photometric_max=0.0)
    aug.set_epoch(5)
    # Force a deterministic crop of the bottom-right quadrant via the internal op.
    torch.manual_seed(0)
    out_img, out_boxes, out_labels = aug._apply_crop(img, box, labels, size=size)
    assert out_boxes.shape[0] == out_labels.shape[0]
    assert out_boxes.shape[0] <= box.shape[0]


def test_call_returns_labels_when_provided():
    img, box = _make_image_with_box()
    labels = torch.tensor([0])
    aug = LayoutScheduledAugment(total_epochs=10, p_hflip_max=1.0, p_crop_max=0.0,
                                 p_rot_max=0.0, p_photometric_max=0.0)
    aug.set_epoch(5)
    out = aug(img, box, labels)
    assert len(out) == 3
    img2, box2, labels2 = out
    assert labels2.shape == labels.shape


# ── photometric: image changes, boxes untouched, output in range ──
def test_photometric_changes_image_keeps_boxes():
    img = torch.rand(1, 16, 16) * 2 - 1  # [-1,1]
    box = torch.tensor([[2.0, 2.0, 6.0, 6.0]])
    aug = LayoutScheduledAugment(total_epochs=10, p_hflip_max=0.0, p_crop_max=0.0,
                                 p_rot_max=0.0, p_photometric_max=1.0,
                                 photometric_noise_std=0.05)
    aug.set_epoch(5)
    torch.manual_seed(1)
    out_img, out_box, _ = aug(img, box, torch.tensor([0]))
    assert not torch.allclose(out_img, img)
    assert torch.allclose(out_box, box)
    assert float(out_img.min()) >= -1.0 - 1e-5 and float(out_img.max()) <= 1.0 + 1e-5


# ── schedule values match warmup/ramp/decay formula ──
def test_schedule_warmup_ramp_decay():
    aug = LayoutScheduledAugment(total_epochs=100, warmup_frac=0.05, ramp_frac=0.15,
                                 p_crop_warmup=0.0, p_crop_max=0.3, p_crop_final=0.3)
    aug.set_epoch(0)
    assert aug.current_probs()["p_crop"] == pytest.approx(0.0)
    aug.set_epoch(20)  # past ramp_end (=20) → decay region at max (final==max)
    assert aug.current_probs()["p_crop"] == pytest.approx(0.3, abs=1e-6)


def test_ramped_value_midpoint():
    v = _ramped_value(epoch=10, total_epochs=100, warmup_frac=0.0, ramp_frac=0.2,
                      v_warmup=0.0, v_max=1.0, v_final=1.0)
    assert v == pytest.approx(0.5, abs=1e-6)  # halfway through a 0..20 ramp


# ── dropout passthrough into the reg UNet config ──
def test_reg_unet_has_dropout():
    from src.models.fm_unet import build_fm_unet_from_config, load_unet_config

    cfg = load_unet_config("configs/models/fm/stable_unet_3L_small_v18_256_reg.json")
    unet = build_fm_unet_from_config(cfg)
    dropout_ps = {round(float(m.p), 3) for m in unet.modules() if isinstance(m, torch.nn.Dropout)}
    assert 0.1 in dropout_ps


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
