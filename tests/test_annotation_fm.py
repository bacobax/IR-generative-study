"""Tests for annotation-driven FM data pipeline with curriculum learning.

Run with::

    python -m pytest tests/test_annotation_fm.py -v
    # or standalone:
    python tests/test_annotation_fm.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None


# ── Fixture: fake COCO annotations + .npy images ─────────────────────────

def _make_test_data(tmpdir: str, n_images: int = 6):
    """Create a minimal COCO annotation file and matching .npy images."""
    img_dir = os.path.join(tmpdir, "images")
    os.makedirs(img_dir, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1

    for i in range(n_images):
        fname = f"img_{i:04d}.npy"
        arr = np.random.randint(0, 65535, (1, 128, 128), dtype=np.uint16)
        np.save(os.path.join(img_dir, fname), arr)

        images.append({
            "id": i + 1,
            "file_name": fname,
            "width": 128,
            "height": 128,
        })

        # Add 0..3 person bboxes per image
        n_people = i % 4
        for j in range(n_people):
            x = 10 + j * 20
            y = 10 + j * 15
            annotations.append({
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [x, y, 15, 20],
            })
            ann_id += 1

    annot_path = os.path.join(tmpdir, "annotations.json")
    with open(annot_path, "w") as f:
        json.dump({"images": images, "annotations": annotations}, f)

    return img_dir, annot_path


# ═══════════════════════════════════════════════════════════════════════════
# Tests: annotation utilities
# ═══════════════════════════════════════════════════════════════════════════

def test_coco_loading_and_indexing():
    from src.core.data.annotations import load_coco_annotations, index_annotations

    with tempfile.TemporaryDirectory() as tmpdir:
        _, annot_path = _make_test_data(tmpdir)
        coco = load_coco_annotations(annot_path)
        images_by_id, anns_by_id, fname_to_id = index_annotations(coco)

        assert len(images_by_id) == 6
        assert fname_to_id["img_0000.npy"] == 1
        # img_0000 has 0 people, img_0001 has 1, img_0002 has 2, img_0003 has 3
        assert len(anns_by_id.get(1, [])) == 0
        assert len(anns_by_id.get(2, [])) == 1
        assert len(anns_by_id.get(3, [])) == 2
        assert len(anns_by_id.get(4, [])) == 3


def test_bbox_helpers():
    from src.core.data.annotations import (
        coco_bbox_to_xyxy, box_intersects, box_inside,
        clip_box_to_image,
    )

    assert coco_bbox_to_xyxy([10, 20, 30, 40]) == (10, 20, 40, 60)

    assert box_intersects((0, 0, 10, 10), (5, 5, 15, 15)) is True
    assert box_intersects((0, 0, 10, 10), (20, 20, 30, 30)) is False

    assert box_inside((2, 2, 8, 8), (0, 0, 10, 10)) is True
    assert box_inside((0, 0, 10, 10), (2, 2, 8, 8)) is False

    clipped = clip_box_to_image((-5, -5, 300, 300), 128, 128)
    assert clipped[0] >= 0 and clipped[1] >= 0
    assert clipped[2] <= 128 and clipped[3] <= 128


def test_caption_from_count():
    from src.core.data.annotations import caption_from_count

    c0 = caption_from_count(0)
    assert "person" not in c0 and "people" not in c0
    assert "overhead infrared" in c0

    c1 = caption_from_count(1)
    assert "1 person" in c1

    c5 = caption_from_count(5)
    assert "5 people" in c5


def test_crop_expansion():
    from src.core.data.annotations import expand_crop_to_include_intersecting_boxes

    # Crop intersects a box that isn't fully inside -> should expand
    bboxes = [(50, 50, 80, 80)]
    crop = (40, 40, 70, 70)  # intersects but doesn't contain (50,50,80,80)
    result = expand_crop_to_include_intersecting_boxes(crop, bboxes, 128, 128)
    x1, y1, x2, y2 = result
    # After expansion, the box should be fully inside
    assert x1 <= 50 and y1 <= 50 and x2 >= 80 and y2 >= 80


def test_sample_person_crop_no_partial():
    from src.core.data.annotations import sample_person_crop, box_inside

    bboxes = [
        (10.0, 10.0, 30.0, 30.0),
        (50.0, 50.0, 70.0, 70.0),
        (90.0, 90.0, 110.0, 110.0),
    ]
    image_shape = (128, 128)

    for _ in range(50):
        cx1, cy1, cx2, cy2 = sample_person_crop(
            image_shape, bboxes,
            margin_scale_range=(1.5, 2.5),
            center_jitter_frac=0.1,
        )
        crop = (float(cx1), float(cy1), float(cx2), float(cy2))
        from src.core.data.annotations import box_intersects
        for box in bboxes:
            if box_intersects(box, crop):
                assert box_inside(box, crop), (
                    f"Box {box} intersects crop {crop} but is not fully inside"
                )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: AnnotationFMDataset
# ═══════════════════════════════════════════════════════════════════════════

def test_annotation_dataset_unconditional():
    from src.core.data.annotation_dataset import AnnotationFMDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        ds = AnnotationFMDataset(
            root_dir=img_dir,
            annotations_path=annot_path,
            text_mode=False,
        )
        assert len(ds) == 6
        item = ds[0]
        # Should return a tensor (not a dict)
        import torch
        assert isinstance(item, torch.Tensor)
        assert item.shape[-1] == 256 and item.shape[-2] == 256


def test_annotation_dataset_text_mode():
    from src.core.data.annotation_dataset import AnnotationFMDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        ds = AnnotationFMDataset(
            root_dir=img_dir,
            annotations_path=annot_path,
            text_mode=True,
        )
        item = ds[0]
        assert isinstance(item, dict)
        assert "pixel_values" in item and "text" in item
        assert isinstance(item["text"], str)
        assert "overhead infrared" in item["text"]

        # img_0001 has 1 person
        item1 = ds[1]
        assert "1 person" in item1["text"]

        # img_0002 has 2 people
        item2 = ds[2]
        assert "2 people" in item2["text"]


def test_annotation_dataset_curriculum():
    """Curriculum enabled with high crop probability -> crops should happen."""
    from dataclasses import dataclass

    @dataclass
    class FakeCurriculum:
        enabled: bool = True
        crop_prob_start: float = 1.0
        crop_prob_end: float = 1.0
        schedule: str = "fixed"
        margin_min: float = 1.2
        margin_max: float = 2.0
        center_jitter: float = 0.1
        force_square: bool = False
        total_epochs: int = 10

    from src.core.data.annotation_dataset import AnnotationFMDataset
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        ds = AnnotationFMDataset(
            root_dir=img_dir,
            annotations_path=annot_path,
            text_mode=True,
            curriculum=FakeCurriculum(),
        )
        # img_0003 has 3 people - with crop_prob=1.0 it should always crop
        item = ds[3]
        assert isinstance(item, dict)
        assert "text" in item
        # Caption may differ from full-image count due to crop


def test_curriculum_disabled_no_crop():
    """When curriculum is disabled, no crop should be applied."""
    from dataclasses import dataclass

    @dataclass
    class FakeCurriculum:
        enabled: bool = False

    from src.core.data.annotation_dataset import AnnotationFMDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        ds = AnnotationFMDataset(
            root_dir=img_dir,
            annotations_path=annot_path,
            text_mode=True,
            curriculum=FakeCurriculum(),
        )
        # With curriculum disabled, img_0002 should have full-image caption
        item = ds[2]  # 2 people
        assert "2 people" in item["text"]


def test_unconditional_with_curriculum():
    """Unconditional FM + curriculum = crops applied but no captions."""
    from dataclasses import dataclass

    @dataclass
    class FakeCurriculum:
        enabled: bool = True
        crop_prob_start: float = 1.0
        crop_prob_end: float = 1.0
        schedule: str = "fixed"
        margin_min: float = 1.2
        margin_max: float = 2.0
        center_jitter: float = 0.1
        force_square: bool = False
        total_epochs: int = 10

    from src.core.data.annotation_dataset import AnnotationFMDataset
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        ds = AnnotationFMDataset(
            root_dir=img_dir,
            annotations_path=annot_path,
            text_mode=False,
            curriculum=FakeCurriculum(),
        )
        item = ds[3]  # 3 people, curriculum on
        assert isinstance(item, torch.Tensor)  # no dict, no text


def test_curriculum_schedule():
    """Linear schedule should interpolate crop probability."""
    from dataclasses import dataclass

    @dataclass
    class FakeCurriculum:
        enabled: bool = True
        crop_prob_start: float = 0.0
        crop_prob_end: float = 1.0
        schedule: str = "linear"
        margin_min: float = 1.2
        margin_max: float = 2.0
        center_jitter: float = 0.1
        force_square: bool = False
        total_epochs: int = 10

    from src.core.data.annotation_dataset import AnnotationFMDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        ds = AnnotationFMDataset(
            root_dir=img_dir,
            annotations_path=annot_path,
            text_mode=False,
            curriculum=FakeCurriculum(),
        )
        # Epoch 0: prob should be 0.0
        ds.set_epoch(0)
        assert abs(ds._current_crop_prob() - 0.0) < 1e-6

        # Epoch 9: prob should be 1.0
        ds.set_epoch(9)
        assert abs(ds._current_crop_prob() - 1.0) < 1e-6

        # Epoch 5: prob should be ~0.555
        ds.set_epoch(5)
        p = ds._current_crop_prob()
        assert 0.4 < p < 0.7


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Count filtering
# ═══════════════════════════════════════════════════════════════════════════

def test_count_filter_seen_counts():
    """Only images with person counts in seen_counts should remain."""
    from dataclasses import dataclass

    @dataclass
    class FakeCountFilter:
        seen_counts: list = None
        unseen_counts: list = None
        max_crop_retries: int = 5

    from src.core.data.annotation_dataset import AnnotationFMDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        # _make_test_data: 6 images, counts = [0, 1, 2, 3, 0, 1]
        # Keep only count 0 and 1
        cf = FakeCountFilter(seen_counts=[0, 1])
        ds = AnnotationFMDataset(
            root_dir=img_dir,
            annotations_path=annot_path,
            text_mode=True,
            count_filter=cf,
        )
        # Images with count 0: img_0000, img_0004 ; count 1: img_0001, img_0005
        assert len(ds) == 4
        for i in range(len(ds)):
            item = ds[i]
            text = item["text"]
            # Should never contain "2 people" or "3 people"
            assert "2 people" not in text
            assert "3 people" not in text


def test_count_filter_unseen_counts():
    """Images with unseen counts should be excluded."""
    from dataclasses import dataclass

    @dataclass
    class FakeCountFilter:
        seen_counts: list = None
        unseen_counts: list = None
        max_crop_retries: int = 5

    from src.core.data.annotation_dataset import AnnotationFMDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        # Exclude count 2 and 3 -> keep 0 and 1
        cf = FakeCountFilter(unseen_counts=[2, 3])
        ds = AnnotationFMDataset(
            root_dir=img_dir,
            annotations_path=annot_path,
            text_mode=True,
            count_filter=cf,
        )
        assert len(ds) == 4


def test_count_filter_both_raises():
    """Setting both seen_counts and unseen_counts should raise."""
    from dataclasses import dataclass

    @dataclass
    class FakeCountFilter:
        seen_counts: list = None
        unseen_counts: list = None
        max_crop_retries: int = 5

    from src.core.data.annotation_dataset import AnnotationFMDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        cf = FakeCountFilter(seen_counts=[0, 1], unseen_counts=[2])
        try:
            ds = AnnotationFMDataset(
                root_dir=img_dir,
                annotations_path=annot_path,
                text_mode=True,
                count_filter=cf,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not both" in str(e)


def test_count_filter_crop_retry():
    """Crop that produces an excluded count should be retried."""
    from dataclasses import dataclass

    @dataclass
    class FakeCurriculum:
        enabled: bool = True
        crop_prob_start: float = 1.0
        crop_prob_end: float = 1.0
        schedule: str = "fixed"
        margin_min: float = 1.2
        margin_max: float = 2.0
        center_jitter: float = 0.1
        force_square: bool = False
        total_epochs: int = 10

    @dataclass
    class FakeCountFilter:
        seen_counts: list = None
        unseen_counts: list = None
        max_crop_retries: int = 10

    from src.core.data.annotation_dataset import AnnotationFMDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        # Keep only count 3 — img_0003 has 3 people full image
        cf = FakeCountFilter(seen_counts=[3])
        ds = AnnotationFMDataset(
            root_dir=img_dir,
            annotations_path=annot_path,
            text_mode=True,
            curriculum=FakeCurriculum(),
            count_filter=cf,
        )
        # Only img_0003 should survive
        assert len(ds) == 1
        # With crop_prob=1.0, crops happen but must produce count=3 or fall back
        item = ds[0]
        assert isinstance(item, dict)
        assert "text" in item
        # Caption must reflect allowed count (3 people) or fallback full image (3)
        assert "3 people" in item["text"]


def test_count_filter_none_no_filtering():
    """No count filter -> all images kept."""
    from src.core.data.annotation_dataset import AnnotationFMDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir, annot_path = _make_test_data(tmpdir)
        ds = AnnotationFMDataset(
            root_dir=img_dir,
            annotations_path=annot_path,
            text_mode=True,
            count_filter=None,
        )
        assert len(ds) == 6


def test_count_filter_config_dataclass():
    """CountFilterConfig default values should be correct."""
    from src.core.configs.fm_config import CountFilterConfig

    cf = CountFilterConfig()
    assert cf.seen_counts is None
    assert cf.unseen_counts is None
    assert cf.max_crop_retries == 5

    cf2 = CountFilterConfig(seen_counts=[0, 1, 2])
    assert cf2.seen_counts == [0, 1, 2]


# ═══════════════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running annotation FM tests...")
    test_coco_loading_and_indexing()
    print("  ✓ COCO loading and indexing")
    test_bbox_helpers()
    print("  ✓ Bbox helpers")
    test_caption_from_count()
    print("  ✓ Caption from count")
    test_crop_expansion()
    print("  ✓ Crop expansion")
    test_sample_person_crop_no_partial()
    print("  ✓ Person crop no partial")
    test_annotation_dataset_unconditional()
    print("  ✓ Unconditional dataset")
    test_annotation_dataset_text_mode()
    print("  ✓ Text-mode dataset")
    test_annotation_dataset_curriculum()
    print("  ✓ Curriculum dataset")
    test_curriculum_disabled_no_crop()
    print("  ✓ Curriculum disabled")
    test_unconditional_with_curriculum()
    print("  ✓ Unconditional + curriculum")
    test_curriculum_schedule()
    print("  ✓ Curriculum schedule")
    test_count_filter_seen_counts()
    print("  ✓ Count filter: seen_counts")
    test_count_filter_unseen_counts()
    print("  ✓ Count filter: unseen_counts")
    test_count_filter_both_raises()
    print("  ✓ Count filter: both raises")
    test_count_filter_crop_retry()
    print("  ✓ Count filter: crop retry")
    test_count_filter_none_no_filtering()
    print("  ✓ Count filter: none = no filtering")
    test_count_filter_config_dataclass()
    print("  ✓ Count filter: config dataclass")
    print("\nAll tests passed!")
