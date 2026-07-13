"""YOLO-detection adherence filter for Experiment B synthetic images.

Instead of the foreground/background crop classifier, this filter runs the trained
YOLO detector on every generated image and keeps a GT box only when a YOLO
prediction matches it at IoU >= ``iou_thr``.  Unmatched GT boxes are marked
negative (``is_positive=False``) so the downstream Experiment B machinery drops
them from the synthetic layout.  The output ``instance_rows`` / ``image_rows`` /
``stats`` mirror the schema produced by
``rare_layout_dataset_tools.audit_generated_layout_dataset`` so the rest of the
pipeline (annotation filtering, image classification, discard summary) is reused
unchanged.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.algorithms.inference.rare_layout_dataset_tools import (
    _load_generated_dataset,
    load_json,
    size_bin_name,
    summarize_instance_statistics,
)


def _iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU between every predicted box and every GT box (both xyxy, pixel)."""

    if pred.size == 0 or gt.size == 0:
        return np.zeros((pred.shape[0], gt.shape[0]), dtype=np.float64)
    px1, py1, px2, py2 = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    gx1, gy1, gx2, gy2 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    pa = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
    ga = np.clip(gx2 - gx1, 0, None) * np.clip(gy2 - gy1, 0, None)
    union = pa + ga - inter
    return inter / np.clip(union, 1e-9, None)


def _npy_to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    """Convert a generated ``.npy`` image to an HxWx3 uint8 array for YOLO.

    Mirrors ``yolo_experiment_b._array_to_png_uint8`` (percentile min-max stretch,
    grayscale mean) so the detector sees the same pixels that are written into the
    merged training set, then stacks to 3 channels.
    """

    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3:
        arr = arr.astype(np.float32).mean(axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.dtype == np.uint16 or float(np.nanmax(arr)) > 255.0:
            low = float(np.nanpercentile(arr, 1.0))
            high = float(np.nanpercentile(arr, 99.0))
        else:
            low = float(np.nanmin(arr))
            high = float(np.nanmax(arr))
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            arr = np.zeros(arr.shape, dtype=np.uint8)
        else:
            arr = np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(arr[..., None], 3, axis=-1)


def _match_gt_indices(pred: np.ndarray, gt: np.ndarray, *, iou_thr: float) -> Dict[int, float]:
    """Greedy IoU matching → {gt_index: matched_iou} for GT boxes matched >= thr."""

    n_pred, n_gt = pred.shape[0], gt.shape[0]
    if n_gt == 0 or n_pred == 0:
        return {}
    iou = _iou_matrix(pred, gt)
    pairs = sorted(
        ((iou[i, j], i, j) for i in range(n_pred) for j in range(n_gt)),
        key=lambda t: -t[0],
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    matched: Dict[int, float] = {}
    for value, i, j in pairs:
        if value < iou_thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched[j] = float(value)
    return matched


def _crop_gt_box(
    image_uint8: np.ndarray,
    bbox_xyxy: Sequence[float],
    *,
    context_frac: float,
    tile: int,
) -> "Any":
    """Crop a GT box (with context margin, GT rectangle drawn) → PIL tile.

    ``image_uint8`` is the HxWx3 uint8 array the detector saw, so the crop shows
    exactly the pixels the YOLO filter judged.  A context margin proportional to
    the box size is added so a tiny box is viewable in its surroundings, and the
    original GT box is drawn in red before resizing to ``tile``x``tile``.
    """

    from PIL import Image, ImageDraw

    h_img, w_img = image_uint8.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    margin = context_frac * max(bw, bh)
    cx1 = int(max(0, math.floor(x1 - margin)))
    cy1 = int(max(0, math.floor(y1 - margin)))
    cx2 = int(min(w_img, math.ceil(x2 + margin)))
    cy2 = int(min(h_img, math.ceil(y2 + margin)))
    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = 0, 0, w_img, h_img
    crop = image_uint8[cy1:cy2, cx1:cx2]
    img = Image.fromarray(np.ascontiguousarray(crop), mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
        outline=(255, 0, 0),
        width=1,
    )
    return img.resize((tile, tile), Image.NEAREST)


def _save_crop_montages(
    buckets: Dict[Tuple[str, str], List[Tuple["Any", float]]],
    *,
    crops_dir: Path,
    tile: int,
    cols: int = 10,
) -> List[str]:
    """Arrange collected crop tiles into one montage PNG per (status, size_bin)."""

    from PIL import Image, ImageDraw

    crops_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    pad = 2
    label_h = 12
    cell = tile + pad
    for (status, size_bin), tiles in sorted(buckets.items()):
        if not tiles:
            continue
        n = len(tiles)
        rows = math.ceil(n / cols)
        sheet = Image.new(
            "RGB", (cols * cell + pad, rows * (cell + label_h) + pad), (30, 30, 30)
        )
        draw = ImageDraw.Draw(sheet)
        for idx, (img, iou) in enumerate(tiles):
            r, c = divmod(idx, cols)
            x = pad + c * cell
            y = pad + r * (cell + label_h)
            sheet.paste(img, (x, y))
            draw.text((x + 1, y + tile + 1), f"{iou:.2f}", fill=(200, 200, 200))
        out = crops_dir / f"{status}_{size_bin}.png"
        sheet.save(out)
        written.append(str(out))
    return written


def audit_generated_candidates_yolo(
    *,
    generated_dataset_dir: str | Path,
    yolo_weights: str | Path,
    device: str,
    iou_thr: float,
    conf: float,
    batch_size: int = 16,
    crops_dir: str | Path | None = None,
    crop_context_frac: float = 0.6,
    crop_tile: int = 96,
    max_crops_per_bin: int = 80,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Audit generated candidates with the trained YOLO detector.

    A GT annotation is positive iff a YOLO prediction (class-agnostic, matching the
    single-class v18 setting) overlaps it at IoU >= ``iou_thr``.
    """

    from ultralytics import YOLO

    dataset_dir = Path(generated_dataset_dir)
    _annotations, images_by_id, anns_by_image_id, categories, images_dir = _load_generated_dataset(dataset_dir)

    metadata_summary_path = dataset_dir / "metadata" / "summary.json"
    metadata_summary = load_json(metadata_summary_path) if metadata_summary_path.is_file() else {}
    size_thresholds = np.asarray(
        metadata_summary.get("size_bin_thresholds", [0.002, 0.01]), dtype=np.float32
    )

    model = YOLO(str(yolo_weights))

    instance_rows: List[Dict[str, Any]] = []
    image_rows: List[Dict[str, Any]] = []

    save_crops = crops_dir is not None
    crop_buckets: Dict[Tuple[str, str], List[Tuple[Any, float]]] = defaultdict(list)

    image_items = list(images_by_id.items())
    for start in range(0, len(image_items), max(1, int(batch_size))):
        chunk = image_items[start:start + max(1, int(batch_size))]
        images = [
            _npy_to_uint8_rgb(np.load(images_dir / str(info["file_name"])))
            for _, info in chunk
        ]
        results = model.predict(source=images, conf=conf, device=device, verbose=False)

        for (image_id, image_info), result, image_uint8 in zip(chunk, results, images):
            width = int(image_info["width"])
            height = int(image_info["height"])
            pred = (
                result.boxes.xyxy.detach().cpu().numpy()
                if result.boxes is not None
                else np.zeros((0, 4), dtype=np.float64)
            )
            anns = list(anns_by_image_id.get(image_id, []))
            gt_xyxy = np.array(
                [
                    [
                        float(a["bbox"][0]),
                        float(a["bbox"][1]),
                        float(a["bbox"][0]) + float(a["bbox"][2]),
                        float(a["bbox"][1]) + float(a["bbox"][3]),
                    ]
                    for a in anns
                ],
                dtype=np.float64,
            ).reshape(-1, 4)

            matched = _match_gt_indices(pred, gt_xyxy, iou_thr=float(iou_thr))

            n_positive = 0
            for gt_index, ann in enumerate(anns):
                category_id = int(ann["category_id"])
                category_name = categories.get(category_id, str(category_id))
                x, y, w, h = [float(v) for v in ann["bbox"]]
                area_ratio = (w * h) / max(1.0, float(width * height))
                is_positive = gt_index in matched
                n_positive += int(is_positive)
                box_size_bin = size_bin_name(area_ratio, size_thresholds)
                if save_crops:
                    status = "kept" if is_positive else "discarded"
                    bucket = crop_buckets[(status, box_size_bin)]
                    if len(bucket) < int(max_crops_per_bin):
                        bucket.append(
                            (
                                _crop_gt_box(
                                    image_uint8,
                                    gt_xyxy[gt_index],
                                    context_frac=float(crop_context_frac),
                                    tile=int(crop_tile),
                                ),
                                float(matched.get(gt_index, 0.0)),
                            )
                        )
                instance_rows.append(
                    {
                        "annotation_id": int(ann["id"]),
                        "generated_image_id": image_id,
                        "generated_file_name": image_info["file_name"],
                        "source_image_id": ann.get("source_image_id"),
                        "source_file_name": ann.get("source_file_name"),
                        "category_id": category_id,
                        "category_name": category_name,
                        "expected_category_id": category_id,
                        "expected_category_name": category_name,
                        "predicted_category_id": category_id if is_positive else None,
                        "predicted_category_name": category_name if is_positive else "background",
                        "bbox_xywh": [x, y, w, h],
                        "bbox_xyxy": gt_xyxy[gt_index].tolist(),
                        "size_bin": box_size_bin,
                        "normalized_area_ratio": float(area_ratio),
                        "matched_iou": float(matched.get(gt_index, 0.0)),
                        "iou_threshold": float(iou_thr),
                        "is_positive": bool(is_positive),
                        "is_background_prediction": bool(not is_positive),
                        "is_class_match": bool(is_positive),
                        "passes_expected_class_threshold": bool(is_positive),
                    }
                )

            valid_fraction = float(n_positive / max(1, len(anns))) if anns else 0.0
            image_rows.append(
                {
                    "generated_image_id": image_id,
                    "generated_file_name": image_info["file_name"],
                    "source_image_id": anns[0].get("source_image_id") if anns else None,
                    "source_file_name": anns[0].get("source_file_name") if anns else None,
                    "n_instances": len(anns),
                    "n_positive_instances": n_positive,
                    "n_negative_instances": len(anns) - n_positive,
                    "n_predicted_boxes": int(pred.shape[0]),
                    "valid_fraction": valid_fraction,
                }
            )

    stats = summarize_instance_statistics(instance_rows)
    stats["image_level"] = {
        "total_image_count": len(image_rows),
        "mean_valid_fraction": float(np.mean([r["valid_fraction"] for r in image_rows])) if image_rows else 0.0,
        "median_valid_fraction": float(np.median([r["valid_fraction"] for r in image_rows])) if image_rows else 0.0,
        "iou_threshold": float(iou_thr),
        "yolo_conf": float(conf),
    }
    stats["generation_metadata"] = {
        "dataset_dir": str(dataset_dir),
        "size_bin_thresholds": [float(v) for v in size_thresholds.tolist()],
        "dataset_id": str(metadata_summary.get("dataset_id", "flir_private_proxy_alignment_v18")),
        "source_split": str(metadata_summary.get("split", "train")),
        "filter_kind": "yolo",
    }
    if save_crops:
        written = _save_crop_montages(crop_buckets, crops_dir=Path(crops_dir), tile=int(crop_tile))
        stats["crop_montages"] = {
            "crops_dir": str(crops_dir),
            "max_crops_per_bin": int(max_crops_per_bin),
            "bucket_counts": {f"{s}_{b}": len(v) for (s, b), v in sorted(crop_buckets.items())},
            "written": written,
        }
    return instance_rows, image_rows, stats
