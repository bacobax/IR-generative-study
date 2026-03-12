"""Annotation loading, crop sampling, and caption generation utilities.

Provides reusable functions for:
- Loading and indexing COCO-style annotations
- Bbox format conversion and geometry helpers
- Person-centered crop sampling with iterative expansion
- Dynamic caption generation from bounding-box counts
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# COCO annotation loading
# ═══════════════════════════════════════════════════════════════════════════

def load_coco_annotations(path: str | Path) -> dict:
    """Load a COCO-format annotations JSON and return the raw dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_annotations(coco: dict) -> Tuple[
    Dict[int, dict],
    Dict[int, List[dict]],
    Dict[str, int],
]:
    """Build efficient lookup structures from COCO annotations.

    Returns
    -------
    images_by_id : dict[int, dict]
        Mapping from image id to image info dict.
    anns_by_image_id : dict[int, list[dict]]
        Mapping from image id to list of annotation dicts.
    filename_to_image_id : dict[str, int]
        Mapping from ``file_name`` to image id.
    """
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image_id: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image_id[ann["image_id"]].append(ann)
    filename_to_image_id = {img["file_name"]: img["id"] for img in coco["images"]}
    return images_by_id, dict(anns_by_image_id), filename_to_image_id


# ═══════════════════════════════════════════════════════════════════════════
# Bbox geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def coco_bbox_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert COCO ``[x, y, w, h]`` to ``(x1, y1, x2, y2)``."""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def clip_box_to_image(
    box: Tuple[float, float, float, float], width: int, height: int,
) -> Tuple[float, float, float, float]:
    """Clip a box to image boundaries."""
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return x1, y1, x2, y2


def box_intersects(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> bool:
    """Return True if two xyxy boxes overlap."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return (min(ax2, bx2) > max(ax1, bx1)) and (min(ay2, by2) > max(ay1, by1))


def box_inside(
    inner: Tuple[float, float, float, float],
    outer: Tuple[float, float, float, float],
) -> bool:
    """Return True if *inner* is fully contained in *outer*."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2


# ═══════════════════════════════════════════════════════════════════════════
# Crop expansion
# ═══════════════════════════════════════════════════════════════════════════

def expand_crop_to_include_intersecting_boxes(
    crop_xyxy: Tuple[float, float, float, float],
    bboxes_xyxy: List[Tuple[float, float, float, float]],
    image_w: int,
    image_h: int,
    max_iter: int = 10,
) -> Tuple[int, int, int, int]:
    """Expand crop until every intersecting box is fully contained.

    If a person box intersects the crop boundary, the crop is enlarged
    to fully include that box.  This is repeated until convergence or
    *max_iter* is reached.
    """
    x1, y1, x2, y2 = crop_xyxy

    for _ in range(max_iter):
        changed = False
        for bx1, by1, bx2, by2 in bboxes_xyxy:
            box = (bx1, by1, bx2, by2)
            crop = (x1, y1, x2, y2)
            if box_intersects(box, crop) and not box_inside(box, crop):
                x1 = min(x1, bx1)
                y1 = min(y1, by1)
                x2 = max(x2, bx2)
                y2 = max(y2, by2)
                changed = True
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(image_w), x2)
        y2 = min(float(image_h), y2)
        if not changed:
            break

    return int(x1), int(y1), int(x2), int(y2)


# ═══════════════════════════════════════════════════════════════════════════
# Person-centered crop sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_person_crop(
    image_shape: Tuple[int, int],
    bboxes_xyxy: List[Tuple[float, float, float, float]],
    *,
    margin_scale_range: Tuple[float, float] = (1.2, 2.0),
    center_jitter_frac: float = 0.15,
    force_square: bool = False,
) -> Tuple[int, int, int, int]:
    """Sample a crop around one random person, expanding to include any
    intersecting people so that no visible person is partially cut off.

    Parameters
    ----------
    image_shape : (H, W)
    bboxes_xyxy : list of (x1, y1, x2, y2) person boxes
    margin_scale_range : (min_scale, max_scale) around the target box
    center_jitter_frac : fractional jitter applied to crop centre
    force_square : if True, crop is forced to be square

    Returns
    -------
    (x1, y1, x2, y2) : int pixel coordinates of the final crop
    """
    H, W = image_shape

    # Pick one target person
    tx1, ty1, tx2, ty2 = random.choice(bboxes_xyxy)
    bw = tx2 - tx1
    bh = ty2 - ty1
    cx = 0.5 * (tx1 + tx2)
    cy = 0.5 * (ty1 + ty2)

    scale = random.uniform(*margin_scale_range)
    crop_w = bw * scale
    crop_h = bh * scale

    if force_square:
        side = max(crop_w, crop_h)
        crop_w = crop_h = side

    jx = random.uniform(-center_jitter_frac, center_jitter_frac) * crop_w
    jy = random.uniform(-center_jitter_frac, center_jitter_frac) * crop_h
    cx += jx
    cy += jy

    x1 = cx - crop_w / 2
    y1 = cy - crop_h / 2
    x2 = cx + crop_w / 2
    y2 = cy + crop_h / 2

    x1, y1, x2, y2 = clip_box_to_image((x1, y1, x2, y2), W, H)

    # Expand iteratively to fully include any intersecting person
    x1, y1, x2, y2 = expand_crop_to_include_intersecting_boxes(
        (x1, y1, x2, y2), bboxes_xyxy, image_w=W, image_h=H,
    )
    return x1, y1, x2, y2


# ═══════════════════════════════════════════════════════════════════════════
# Counting people in a crop
# ═══════════════════════════════════════════════════════════════════════════

def count_fully_contained(
    bboxes_xyxy: List[Tuple[float, float, float, float]],
    crop_xyxy: Tuple[int, int, int, int],
) -> int:
    """Count bounding boxes fully contained inside a crop region."""
    cx1, cy1, cx2, cy2 = crop_xyxy
    crop = (float(cx1), float(cy1), float(cx2), float(cy2))
    return sum(1 for box in bboxes_xyxy if box_inside(box, crop))


def count_people_for_image(
    anns_by_image_id: Dict[int, List[dict]],
    image_id: int,
) -> int:
    """Count person annotations for a given image id."""
    return len(anns_by_image_id.get(image_id, []))


def get_bboxes_xyxy_for_image(
    anns_by_image_id: Dict[int, List[dict]],
    image_id: int,
) -> List[Tuple[float, float, float, float]]:
    """Return all person bboxes in xyxy format for an image."""
    anns = anns_by_image_id.get(image_id, [])
    return [coco_bbox_to_xyxy(ann["bbox"]) for ann in anns]


# ═══════════════════════════════════════════════════════════════════════════
# Caption generation
# ═══════════════════════════════════════════════════════════════════════════

_BASE_CAPTION = "overhead infrared surveillance image, circular field of view"


def caption_from_count(n_people: int) -> str:
    """Generate a text caption from the number of visible people.

    Rules:
        0 -> base caption only
        1 -> base caption + ", 1 person"
        n -> base caption + ", n people"
    """
    if n_people <= 0:
        return _BASE_CAPTION
    if n_people == 1:
        return f"{_BASE_CAPTION}, 1 person"
    return f"{_BASE_CAPTION}, {n_people} people"
