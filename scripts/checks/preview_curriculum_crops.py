"""Visual debug helper: preview curriculum crops and their captions.

Usage::

    python scripts/checks/preview_curriculum_crops.py \\
        --annotations data/raw/v18/train/annotations.json \\
        --image_dir data/raw/v18/train/ \\
        --n_samples 8 \\
        --out_dir artifacts/debug/curriculum_crops
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from src.core.data.annotations import (
    caption_from_count,
    coco_bbox_to_xyxy,
    count_fully_contained,
    expand_crop_to_include_intersecting_boxes,
    index_annotations,
    load_coco_annotations,
    sample_person_crop,
)


def normalize_for_display(arr: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Preview curriculum crops")
    parser.add_argument("--annotations", required=True, help="Path to annotations.json")
    parser.add_argument("--image_dir", required=True, help="Directory with .npy images")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of crops to preview")
    parser.add_argument("--out_dir", default="artifacts/debug/curriculum_crops")
    parser.add_argument("--margin_min", type=float, default=1.2)
    parser.add_argument("--margin_max", type=float, default=2.0)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument("--force_square", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    coco = load_coco_annotations(args.annotations)
    images_by_id, anns_by_image_id, fname_to_imgid = index_annotations(coco)

    # Filter images with at least 1 person
    images_with_people = [
        img for img in coco["images"]
        if len(anns_by_image_id.get(img["id"], [])) > 0
    ]

    if not images_with_people:
        print("No annotated images with people found.")
        return

    try:
        from PIL import Image, ImageDraw, ImageFont
        has_pil = True
    except ImportError:
        has_pil = False
        print("PIL not available; saving .npy only")

    for i in range(args.n_samples):
        img_info = random.choice(images_with_people)
        img_id = img_info["id"]
        fname = img_info["file_name"]
        npy_path = os.path.join(args.image_dir, fname)

        if not os.path.exists(npy_path):
            continue

        arr = np.load(npy_path)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        H, W = arr.shape[:2]

        anns = anns_by_image_id.get(img_id, [])
        bboxes_xyxy = [coco_bbox_to_xyxy(a["bbox"]) for a in anns]

        # Full image caption
        full_caption = caption_from_count(len(bboxes_xyxy))

        # Sample crop
        cx1, cy1, cx2, cy2 = sample_person_crop(
            (H, W), bboxes_xyxy,
            margin_scale_range=(args.margin_min, args.margin_max),
            center_jitter_frac=args.jitter,
            force_square=args.force_square,
        )
        crop_arr = arr[cy1:cy2, cx1:cx2]
        n_in_crop = count_fully_contained(bboxes_xyxy, (cx1, cy1, cx2, cy2))
        crop_caption = caption_from_count(n_in_crop)

        # Save
        stem = Path(fname).stem
        base = os.path.join(args.out_dir, f"{i:03d}_{stem}")

        if has_pil:
            # Full image with boxes
            disp = normalize_for_display(arr)
            full_img = Image.fromarray((disp * 255).astype(np.uint8))
            draw = ImageDraw.Draw(full_img)
            for bx1, by1, bx2, by2 in bboxes_xyxy:
                draw.rectangle([bx1, by1, bx2, by2], outline=255, width=2)
            draw.rectangle([cx1, cy1, cx2, cy2], outline=180, width=2)
            full_img.save(f"{base}_full.png")

            # Crop
            crop_disp = normalize_for_display(crop_arr)
            crop_img = Image.fromarray((crop_disp * 255).astype(np.uint8))
            crop_img.save(f"{base}_crop.png")

        # Metadata
        with open(f"{base}_info.txt", "w") as f:
            f.write(f"image: {fname}\n")
            f.write(f"full_people: {len(bboxes_xyxy)}\n")
            f.write(f"full_caption: {full_caption}\n")
            f.write(f"crop: ({cx1}, {cy1}, {cx2}, {cy2})\n")
            f.write(f"crop_people: {n_in_crop}\n")
            f.write(f"crop_caption: {crop_caption}\n")

        print(f"[{i+1}/{args.n_samples}] {fname}: full={len(bboxes_xyxy)}p -> crop={n_in_crop}p")
        print(f"  full: {full_caption}")
        print(f"  crop: {crop_caption}")

    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
