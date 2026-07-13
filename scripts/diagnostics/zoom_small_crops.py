"""Zoomed small-bin crop sheets (kept vs discarded) from the existing filter manifest.

No GPU/YOLO — reads per_instance_manifest.jsonl (has is_positive + bbox_xyxy) and crops
directly from the generated .npy images. Bigger tiles than the montage so tiny boxes are
legible enough to judge whether a person was actually rendered.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.algorithms.inference.yolo_adherence_filter import _npy_to_uint8_rgb

D = Path("artifacts/generated/yolo/exp_b/generated_candidates/small_v4_fmbalaug/fm_balanced_aug")
OUT = Path("artifacts/analysis/yolo/exp_v18_simple_yolo_tiny/small_v4_fmbalaug/discarded_crops")


def crop_tile(row: dict, tile: int = 180, ctx: float = 1.0) -> Image.Image:
    img = _npy_to_uint8_rgb(np.load(D / "images" / row["generated_file_name"]))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = row["bbox_xyxy"]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    m = ctx * max(bw, bh)
    cx1, cy1 = int(max(0, x1 - m)), int(max(0, y1 - m))
    cx2, cy2 = int(min(w, x2 + m)), int(min(h, y2 + m))
    im = Image.fromarray(np.ascontiguousarray(img[cy1:cy2, cx1:cx2]), "RGB")
    ImageDraw.Draw(im).rectangle([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], outline=(255, 0, 0), width=1)
    return im.resize((tile, tile), Image.NEAREST)


def sheet(rows: list, path: Path, cols: int = 6, tile: int = 180) -> None:
    n = len(rows)
    rws = math.ceil(n / cols)
    pad = 3
    canvas = Image.new("RGB", (cols * (tile + pad) + pad, rws * (tile + pad) + pad), (30, 30, 30))
    for i, r in enumerate(rows):
        rr, cc = divmod(i, cols)
        canvas.paste(crop_tile(r, tile), (pad + cc * (tile + pad), pad + rr * (tile + pad)))
    canvas.save(path)
    print("wrote", path)


def main() -> None:
    rows = [json.loads(l) for l in open(D / "filter_audit" / "per_instance_manifest.jsonl")]
    small = [r for r in rows if r["size_bin"] == "small"]
    disc = [r for r in small if not r["is_positive"]][:36]
    kept = [r for r in small if r["is_positive"]][:36]
    print(f"small: {len(small)} total, {len(disc)} discarded shown, {len(kept)} kept shown")
    OUT.mkdir(parents=True, exist_ok=True)
    sheet(disc, OUT / "zoom_discarded_small.png")
    sheet(kept, OUT / "zoom_kept_small.png")


if __name__ == "__main__":
    main()
