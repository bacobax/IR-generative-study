"""RegionDiff synthetic generation dataset IO helpers."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from src.algorithms.training.yolo_experiment_b import YOLOTrainSample
from src.core.data.layout_batching import collate_layout_batch
from src.core.normalization import RAW_UINT16_PERCENTILE, raw_array_to_png_uint8
from src.core.paths import repo_root


STAGE2_LAYOUT_MANIFEST_NAME = "stage2_layout_manifest.json"
STAGE2_REGIONDIFF_CONFIG_NAME = "regiondiff_config.json"
STAGE2_UNET_WEIGHTS_NAME = "regiondiff_unet.safetensors"
STAGE2_CHECKPOINT_UNET_WEIGHTS_NAME = "regiondiff_unet_checkpoint.safetensors"


DEFAULT_CONFIG_PATH = "configs/yolo/exp_b/synthetic_generation/default.yaml"
DEFAULT_YOLO_DATASET_YAML = "data/derived/yolo-test-ds/full_train.yaml"
DEFAULT_OUTPUT_ROOT = "artifacts/generated/yolo/exp_b/precomputed_candidates"
DEFAULT_FILTER_RUN_DIR = (
    "artifacts/checkpoints/multiclass_foreground_background_filter/"
    "runs/multiclass_fgbg_20260415_210440"
)

def _repo_path(path_like: str | Path | None) -> Path | None:
    if path_like is None or str(path_like) == "":
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _normalise_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {idx: str(value) for idx, value in enumerate(raw_names)}
    raise TypeError("YOLO dataset names must be a mapping or list.")


def _image_size_from_array(array: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return int(arr.shape[1]), int(arr.shape[0])
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:
        return int(arr.shape[2]), int(arr.shape[1])
    if arr.ndim == 3:
        return int(arr.shape[1]), int(arr.shape[0])
    raise ValueError(f"Unsupported generated image shape: {tuple(arr.shape)}")


def _array_to_png_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3:
        arr = arr.astype(np.float32).mean(axis=-1)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return raw_array_to_png_uint8(arr, normalization_mode=RAW_UINT16_PERCENTILE)
    arr = arr.astype(np.float32)
    low = float(np.nanpercentile(arr, 1.0))
    high = float(np.nanpercentile(arr, 99.0))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    return np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _save_preview_png(array: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_array_to_png_uint8(array), mode="L").convert("RGB").save(path)
    return path


def _should_save_limited_artifact(image_id: int, max_images: int | None) -> bool:
    if max_images is None:
        return True
    return int(image_id) <= max(0, int(max_images))


def _draw_layout_overlay(
    *,
    array: np.ndarray,
    sample: YOLOTrainSample,
    names: Mapping[int, str],
) -> Image.Image:
    canvas = Image.fromarray(_array_to_png_uint8(array), mode="L").convert("RGB")
    draw = ImageDraw.Draw(canvas)
    image_w, image_h = canvas.size
    for box in sample.boxes:
        x, y, x2, y2 = box.xyxy_abs(image_w=image_w, image_h=image_h)
        category_id = int(box.class_id)
        color = _category_color(category_id)
        draw.rectangle((x, y, x2, y2), outline=color, width=2)
        label = str(names.get(category_id, category_id))
        tx = max(0, int(x))
        ty = max(0, int(y) - 12)
        draw.rectangle((tx, ty, tx + min(160, 7 * len(label) + 4), ty + 11), fill=(0, 0, 0))
        draw.text((tx + 2, ty), label, fill=color)
    return canvas


def _save_layout_overlay_png(
    *,
    output_dir: str | Path,
    image_id: int,
    array: np.ndarray,
    sample: YOLOTrainSample,
    names: Mapping[int, str],
    output_dir_name: str = "layout_overlays",
) -> Path:
    output_path = Path(output_dir) / output_dir_name / f"sample_{int(image_id):06d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _draw_layout_overlay(array=array, sample=sample, names=names).save(output_path)
    return output_path


def _size_bin_thresholds_from_samples(samples: Sequence[YOLOTrainSample]) -> list[float]:
    areas = [float(box.width * box.height) for sample in samples for box in sample.boxes]
    if not areas:
        return [0.002, 0.01]
    return [float(value) for value in np.quantile(np.asarray(areas, dtype=np.float32), [1 / 3, 2 / 3]).tolist()]


def _sample_to_layout_dict(sample: YOLOTrainSample, *, image_size: int, names: dict[int, str]) -> dict[str, Any]:
    boxes = [box.xyxy_abs(image_w=image_size, image_h=image_size) for box in sample.boxes]
    labels = [box.class_id for box in sample.boxes]
    return {
        "pixel_values": torch.zeros(3, image_size, image_size, dtype=torch.float32),
        "boxes_xyxy": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
        "n_objects": len(labels),
        "file_name": sample.image_path.name,
        "image_id": sample.index,
        "label_names": [names.get(int(label), str(int(label))) for label in labels],
    }


def build_layout_batches(
    samples: Sequence[YOLOTrainSample],
    *,
    image_size: int,
    names: dict[int, str],
    batch_size: int,
) -> list[dict[str, Any]]:
    layout_samples = [_sample_to_layout_dict(sample, image_size=image_size, names=names) for sample in samples]
    return [
        collate_layout_batch(layout_samples[start : start + max(1, int(batch_size))])
        for start in range(0, len(layout_samples), max(1, int(batch_size)))
    ]


def _generated_categories(dataset_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = _normalise_names(dataset_payload["names"])
    return [{"id": class_id, "name": name} for class_id, name in sorted(names.items())]


def _candidate_coco_and_provenance(
    *,
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: dict[str, Any],
    image_size: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, sample in enumerate(source_samples, start=1):
        file_name = f"sample_{image_id:06d}.npy"
        preview_name = f"sample_{image_id:06d}.png"
        image_w = image_h = int(image_size or 0)
        coco_images.append({"id": image_id, "file_name": file_name, "width": image_w, "height": image_h})
        if image_size is not None:
            for object_index, box in enumerate(sample.boxes):
                bbox = box.coco_bbox_abs(image_w=image_w, image_h=image_h)
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(box.class_id),
                        "bbox": bbox,
                        "area": float(bbox[2] * bbox[3]),
                        "iscrowd": 0,
                        "source_image_id": sample.index,
                        "source_file_name": sample.image_path.name,
                        "object_index": int(object_index),
                    }
                )
                annotation_id += 1
        provenance_rows.append(
            {
                "generated_image_id": image_id,
                "generated_file_name": file_name,
                "generated_preview_file_name": preview_name,
                "source_index": sample.index,
                "source_image_path": str(sample.image_path),
                "source_label_path": str(sample.label_path),
                "n_objects": len(sample.boxes),
                "labels": [box.to_line() for box in sample.boxes],
            }
        )
    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": _generated_categories(dataset_payload),
    }, provenance_rows


def initialize_generated_candidate_dataset(
    *,
    output_dir: str | Path,
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: dict[str, Any],
    generator_kind: str,
    generator_config: Mapping[str, Any] | None = None,
    image_size: int | None = None,
) -> Path:
    """Initialize candidate metadata before streaming generated arrays to disk."""

    output = Path(output_dir)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir(parents=True, exist_ok=True)
    metadata_dir = output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    coco_payload, provenance_rows = _candidate_coco_and_provenance(
        source_samples=source_samples,
        dataset_payload=dataset_payload,
        image_size=image_size,
    )
    _write_json(output / "annotations_unfiltered.json", coco_payload)
    _write_json(output / "annotations.json", coco_payload)
    _write_jsonl(metadata_dir / "provenance.jsonl", provenance_rows)
    _write_json(
        metadata_dir / "summary.json",
        {
            "dataset_id": "yolo_full_train",
            "split": "train",
            "selection_mode": "full_train_1_to_1",
            "generator_kind": generator_kind,
            "generator_config": dict(generator_config or {}),
            "n_requested_samples": len(source_samples),
            "n_generated_samples": len(source_samples),
            "n_annotations_unfiltered": len(coco_payload["annotations"]),
            "n_annotations": len(coco_payload["annotations"]),
            "source_dataset_yaml": str(dataset_payload.get("_yaml_path", "")),
            "size_bin_thresholds": _size_bin_thresholds_from_samples(source_samples),
            "images_dir": "images",
            "previews_dir": "previews",
            "annotations_path": "annotations.json",
            "unfiltered_annotations_path": "annotations_unfiltered.json",
            "samples": provenance_rows,
            "streamed_to_disk": True,
        },
    )
    return output


def _save_generated_array(
    output_dir: str | Path,
    *,
    image_id: int,
    array: np.ndarray,
    max_preview_images: int | None = None,
    overlay_sample: YOLOTrainSample | None = None,
    overlay_names: Mapping[int, str] | None = None,
    max_layout_overlay_images: int | None = None,
) -> None:
    output = Path(output_dir)
    file_name = f"sample_{int(image_id):06d}.npy"
    preview_name = f"sample_{int(image_id):06d}.png"
    generated_array = np.asarray(array)
    image_path = output / "images" / file_name
    tmp_path = image_path.with_name(f".{image_path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, generated_array)
        tmp_path.replace(image_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    if _should_save_limited_artifact(int(image_id), max_preview_images):
        _save_preview_png(generated_array, output / "previews" / preview_name)
    if (
        overlay_sample is not None
        and overlay_names is not None
        and _should_save_limited_artifact(int(image_id), max_layout_overlay_images)
    ):
        _save_layout_overlay_png(
            output_dir=output,
            image_id=int(image_id),
            array=generated_array,
            sample=overlay_sample,
            names=overlay_names,
        )


def _generated_sample_exists(output_dir: str | Path, *, image_id: int) -> bool:
    output = Path(output_dir)
    image_path = output / "images" / f"sample_{int(image_id):06d}.npy"
    if not image_path.is_file():
        return False
    try:
        array = np.load(image_path, mmap_mode="r")
        _ = array.shape
        mmap_handle = getattr(array, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
    except Exception:
        image_path.unlink(missing_ok=True)
        return False
    return True


def export_generated_candidate_dataset(
    *,
    output_dir: str | Path,
    source_samples: Sequence[YOLOTrainSample],
    generated_arrays: Sequence[np.ndarray],
    dataset_payload: dict[str, Any],
    generator_kind: str,
    generator_config: Mapping[str, Any] | None = None,
    max_preview_images: int | None = None,
    max_layout_overlay_images: int | None = None,
) -> Path:
    """Write generated arrays in the candidate format consumed by Experiment B."""

    output = Path(output_dir)
    images_dir = output / "images"
    previews_dir = output / "previews"
    metadata_dir = output / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    annotation_id = 1

    names = _normalise_names(dataset_payload["names"])
    for image_id, (sample, generated_array) in enumerate(zip(source_samples, generated_arrays), start=1):
        file_name = f"sample_{image_id:06d}.npy"
        preview_name = f"sample_{image_id:06d}.png"
        generated_array = np.asarray(generated_array)
        np.save(images_dir / file_name, generated_array)
        if _should_save_limited_artifact(image_id, max_preview_images):
            _save_preview_png(generated_array, previews_dir / preview_name)
        if _should_save_limited_artifact(image_id, max_layout_overlay_images):
            _save_layout_overlay_png(
                output_dir=output,
                image_id=image_id,
                array=generated_array,
                sample=sample,
                names=names,
            )
        image_w, image_h = _image_size_from_array(generated_array)
        coco_images.append({"id": image_id, "file_name": file_name, "width": image_w, "height": image_h})
        for object_index, box in enumerate(sample.boxes):
            bbox = box.coco_bbox_abs(image_w=image_w, image_h=image_h)
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(box.class_id),
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "source_image_id": sample.index,
                    "source_file_name": sample.image_path.name,
                    "object_index": int(object_index),
                }
            )
            annotation_id += 1
        provenance_rows.append(
            {
                "generated_image_id": image_id,
                "generated_file_name": file_name,
                "generated_preview_file_name": preview_name,
                "source_index": sample.index,
                "source_image_path": str(sample.image_path),
                "source_label_path": str(sample.label_path),
                "n_objects": len(sample.boxes),
                "labels": [box.to_line() for box in sample.boxes],
            }
        )

    coco_payload = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": _generated_categories(dataset_payload),
    }
    _write_json(output / "annotations_unfiltered.json", coco_payload)
    _write_json(output / "annotations.json", coco_payload)
    _write_jsonl(metadata_dir / "provenance.jsonl", provenance_rows)
    _write_json(
        metadata_dir / "summary.json",
        {
            "dataset_id": "yolo_full_train",
            "split": "train",
            "selection_mode": "full_train_1_to_1",
            "generator_kind": generator_kind,
            "generator_config": dict(generator_config or {}),
            "n_requested_samples": len(source_samples),
            "n_generated_samples": len(generated_arrays),
            "n_annotations_unfiltered": len(coco_annotations),
            "n_annotations": len(coco_annotations),
            "source_dataset_yaml": str(dataset_payload.get("_yaml_path", "")),
            "size_bin_thresholds": _size_bin_thresholds_from_samples(source_samples),
            "images_dir": "images",
            "previews_dir": "previews",
            "annotations_path": "annotations.json",
            "unfiltered_annotations_path": "annotations_unfiltered.json",
            "samples": provenance_rows,
        },
    )
    return output

def _category_color(category_id: int) -> tuple[int, int, int]:
    palette = [
        (24, 165, 88),
        (58, 120, 194),
        (229, 126, 49),
        (190, 77, 179),
        (223, 196, 55),
        (61, 174, 194),
        (215, 73, 91),
        (134, 94, 201),
    ]
    return palette[int(category_id) % len(palette)]

__all__ = [name for name in globals() if not name.startswith("__")]
