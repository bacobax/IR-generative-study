"""Production synthetic dataset generation for YOLO Experiment B.

The historical smoke entrypoint now drives a full precomputed synthetic
dataset workflow: one generated image per real YOLO full-train image, per-box
multiclass filter auditing, annotation-level filtering, sanity overlays, and
optional distribution metrics.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler
from src.algorithms.inference.layout_flow_matching_sampler import LayoutFlowMatchingSampler
from src.algorithms.inference.unconditional_sd_sampler import UnconditionalStableDiffusionSampler
from src.algorithms.inference.rare_layout_dataset_tools import (
    audit_generated_layout_dataset,
    export_audit_results,
    load_filter_from_run_or_checkpoint,
    sample_layout_batch,
)
from src.core.diffusers_compat import import_diffusers_attr
from src.algorithms.training.yolo_experiment_b import YOLOTrainSample, load_full_train_samples
from src.core.data.layout_batching import collate_layout_batch
from src.core.normalization import RAW_UINT16_PERCENTILE, raw_array_to_png_uint8
from src.core.paths import repo_root
from src.evaluation.feature_extractors import build_feature_extractor, extract_features
from src.evaluation.generative_metrics import compute_fid, compute_kid
from src.evaluation.mmd import compute_rbf_mmd
from src.models.fm_unet import build_fm_unet_from_config, load_unet_config
from src.models.regiondiffusion_factory import build_regiondiff_wrapper
from src.models.stay_layout_conditioned_unet import build_stay_layout_conditioned_unet
from src.models.vae import build_vae_from_config, freeze_vae, load_diffusers_vae_config, load_vae_config, load_vae_weights


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
    np.save(output / "images" / file_name, generated_array)
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
    return (output / "images" / f"sample_{int(image_id):06d}.npy").is_file()


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


def _extract_unet_state(checkpoint_path: Path, *, device: str | torch.device) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError(
                f"Cannot load safetensors checkpoint {checkpoint_path}; install safetensors."
            ) from exc
        return safe_load_file(str(checkpoint_path), device=str(device))

    try:
        state = torch.load(checkpoint_path, map_location=device)
    except RuntimeError as exc:
        message = str(exc)
        if "PytorchStreamReader" in message or "failed finding central directory" in message:
            raise RuntimeError(
                f"Checkpoint is not readable by torch.load: {checkpoint_path}. "
                "It looks like a truncated or incomplete PyTorch zip checkpoint. "
                "Regenerate/resync this checkpoint, choose another checkpoint_path, "
                "or run a different generator with --generators."
            ) from exc
        raise
    if isinstance(state, dict):
        for key in ("unet_state", "model_state", "state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint_path}")
    return state


def validate_generator_checkpoint_readability(
    checkpoint_path: str | Path,
) -> tuple[bool, str]:
    """Cheaply detect corrupt PyTorch zip checkpoints before model construction."""

    path = _repo_path(checkpoint_path)
    if path is None or not path.exists():
        return False, f"missing checkpoint: {checkpoint_path}"
    if path.is_dir():
        final_weights = path / STAGE2_UNET_WEIGHTS_NAME
        checkpoint_weights = path / STAGE2_CHECKPOINT_UNET_WEIGHTS_NAME
        parent_manifest = path.parent / STAGE2_LAYOUT_MANIFEST_NAME
        if final_weights.is_file() and (path / STAGE2_LAYOUT_MANIFEST_NAME).is_file():
            return True, str(path)
        if checkpoint_weights.is_file() and parent_manifest.is_file():
            return True, str(path)
        return False, f"directory is not a recognized generator checkpoint/artifact: {path}"
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError as exc:
        return False, f"cannot read checkpoint header: {path} ({exc})"
    if magic.startswith(b"PK") and not zipfile.is_zipfile(path):
        return (
            False,
            f"corrupt PyTorch zip checkpoint: {path} "
            "(zip central directory is missing; the file is likely truncated/incomplete)",
        )
    return True, str(path)


def _load_stage2_layout_pipeline(*args, **kwargs):
    from src.algorithms.stable_diffusion.layout_models import load_stage2_layout_pipeline

    return load_stage2_layout_pipeline(*args, **kwargs)


def _infer_stay_num_classes(
    *,
    state: Mapping[str, Any],
    dataset_names: Mapping[int, str],
) -> int:
    dataset_num_classes = max((int(key) for key in dataset_names), default=-1) + 1
    checkpoint_num_classes = 0
    class_embedding = state.get("object_encoder.class_embedding.weight")
    if isinstance(class_embedding, torch.Tensor) and class_embedding.ndim >= 2:
        checkpoint_num_classes = int(class_embedding.shape[0])
    return max(1, dataset_num_classes, checkpoint_num_classes)


def _infer_regiondiff_num_classes(
    *,
    state: Mapping[str, Any],
    dataset_names: Mapping[int, str],
) -> int:
    dataset_num_classes = max((int(key) for key in dataset_names), default=-1) + 1
    checkpoint_num_classes = 0
    class_features = state.get("layout_tokenizer.class_text_features")
    if isinstance(class_features, torch.Tensor) and class_features.ndim == 2:
        checkpoint_num_classes = int(class_features.shape[0])
    return max(1, dataset_num_classes, checkpoint_num_classes)


def _normalise_category_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").strip().lower().split())


def _category_names_from_coco(path: str | Path, *, num_classes: int) -> dict[int, str]:
    resolved = _repo_path(path)
    if resolved is None or not resolved.is_file():
        return {}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    categories = payload.get("categories", [])
    if not isinstance(categories, list) or len(categories) != int(num_classes):
        return {}
    try:
        ordered = sorted(categories, key=lambda row: int(row["id"]))
        return {idx: str(row["name"]) for idx, row in enumerate(ordered)}
    except (KeyError, TypeError, ValueError):
        return {}


def _default_checkpoint_category_names(num_classes: int) -> dict[int, str]:
    for path in (
        "data/raw/flir/images_thermal_train/coco.json",
        "data/raw/flir/images_thermal_val/coco.json",
        "data/raw/flir/video_thermal_test/coco.json",
        "data/tmp/flir_full_multiclass_v18_smoke/train/annotations.json",
    ):
        names = _category_names_from_coco(path, num_classes=num_classes)
        if names:
            return names
    return {}


def _expand_category_names(
    names: Mapping[int, str],
    *,
    num_classes: int,
) -> dict[int, str]:
    return {
        idx: str(names.get(idx, f"class {idx}"))
        for idx in range(max(1, int(num_classes)))
    }


def _regiondiff_checkpoint_category_names(
    generator_cfg: Mapping[str, Any],
    *,
    dataset_names: Mapping[int, str],
    num_classes: int,
) -> dict[int, str]:
    raw_names = generator_cfg.get("checkpoint_category_id_to_name")
    if raw_names is not None:
        return _expand_category_names(_normalise_names(raw_names), num_classes=num_classes)

    raw_path = generator_cfg.get("checkpoint_categories_path")
    if raw_path:
        names = _category_names_from_coco(str(raw_path), num_classes=num_classes)
        if names:
            return _expand_category_names(names, num_classes=num_classes)

    if int(num_classes) > max((int(key) for key in dataset_names), default=-1) + 1:
        names = _default_checkpoint_category_names(int(num_classes))
        if names:
            return _expand_category_names(names, num_classes=num_classes)

    return _expand_category_names(dataset_names, num_classes=num_classes)


def _coerce_label_id_map(raw_map: Any) -> dict[int, int]:
    if raw_map in (None, "", {}):
        return {}
    if isinstance(raw_map, Mapping):
        return {int(key): int(value) for key, value in raw_map.items()}
    if isinstance(raw_map, Sequence) and not isinstance(raw_map, (str, bytes)):
        return {idx: int(value) for idx, value in enumerate(raw_map)}
    raise TypeError("Label id map must be a mapping or sequence.")


def _regiondiff_label_id_map(
    generator_cfg: Mapping[str, Any],
    *,
    dataset_names: Mapping[int, str],
    checkpoint_names: Mapping[int, str],
) -> dict[int, int]:
    explicit_map = generator_cfg.get("dataset_label_to_checkpoint_label")
    if explicit_map is None:
        explicit_map = generator_cfg.get("label_id_map")
    if explicit_map is not None:
        return _coerce_label_id_map(explicit_map)

    checkpoint_by_name = {
        _normalise_category_name(name): int(category_id)
        for category_id, name in checkpoint_names.items()
    }
    inferred: dict[int, int] = {}
    for dataset_id, dataset_name in dataset_names.items():
        checkpoint_id = checkpoint_by_name.get(_normalise_category_name(dataset_name))
        if checkpoint_id is None:
            return {}
        inferred[int(dataset_id)] = int(checkpoint_id)
    if all(int(source) == int(target) for source, target in inferred.items()):
        return {}
    return inferred


def _remap_layout_batch_labels(
    batch: Mapping[str, Any],
    label_id_map: Mapping[int, int],
) -> dict[str, Any]:
    if not label_id_map:
        return dict(batch)
    remapped = dict(batch)
    labels = remapped["labels"]
    object_mask = remapped.get("object_mask")
    if object_mask is not None:
        active_labels = labels[object_mask].detach().cpu().tolist()
    else:
        active_labels = labels.detach().cpu().flatten().tolist()
    missing = sorted(
        {int(label) for label in active_labels}
        - {int(key) for key in label_id_map}
    )
    if missing:
        raise ValueError(
            "Missing RegionDiff checkpoint label mapping for dataset class id(s): "
            f"{missing}."
        )
    mapped_labels = labels.clone()
    for source_id, target_id in label_id_map.items():
        mapped_labels[labels == int(source_id)] = int(target_id)
    remapped["labels"] = mapped_labels
    return remapped


def _resolve_vae_config_from_preset(preset: Mapping[str, Any]) -> dict[str, Any] | None:
    model_cfg = dict(preset.get("model", {}))
    pretrained_name = model_cfg.get("vae_pretrained_model_name_or_path")
    if pretrained_name:
        return load_diffusers_vae_config(
            str(pretrained_name),
            subfolder=model_cfg.get("vae_pretrained_subfolder", "vae"),
            revision=model_cfg.get("vae_revision"),
            variant=model_cfg.get("vae_variant"),
        )
    if model_cfg.get("vae_config"):
        return load_vae_config(str(_repo_path(model_cfg["vae_config"])))
    return None


def _infer_vae_downsample_factor(vae_config: Mapping[str, Any]) -> int:
    for key in ("num_channels", "block_out_channels", "down_block_types"):
        values = vae_config.get(key)
        if isinstance(values, (list, tuple)) and values:
            return 2 ** max(0, len(values) - 1)
    raise ValueError(
        "VAE config must define a non-empty num_channels, block_out_channels, "
        "or down_block_types sequence to infer latent sample_size."
    )


def _apply_training_sample_size(
    unet_cfg: Mapping[str, Any],
    preset: Mapping[str, Any],
    vae_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Mirror training-time latent UNet sample-size resolution."""

    resolved = dict(unet_cfg)
    if vae_cfg is None:
        return resolved
    image_size = dict(preset.get("data", {})).get("image_size")
    if image_size is None:
        return resolved
    image_size = int(image_size)
    downsample_factor = _infer_vae_downsample_factor(vae_cfg)
    if image_size % downsample_factor != 0:
        raise ValueError(
            f"image_size={image_size} is not divisible by VAE downsample factor "
            f"{downsample_factor}"
        )
    latent_size = image_size // downsample_factor
    resolved["sample_size"] = latent_size
    return resolved


def _load_effective_unet_config(
    *,
    checkpoint_path: Path,
    preset: Mapping[str, Any],
    vae_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    saved_config_path = checkpoint_path.parent / "config.json"
    if saved_config_path.is_file():
        unet_cfg = load_unet_config(str(saved_config_path))
    else:
        model_cfg = dict(preset.get("model", {}))
        unet_config = model_cfg.get("unet_config")
        if not unet_config:
            raise FileNotFoundError(
                f"No saved UNET config at {saved_config_path} and no model.unet_config in preset."
            )
        unet_cfg = load_unet_config(str(_repo_path(unet_config)))
    return _apply_training_sample_size(unet_cfg, preset, vae_cfg)


def _build_vae_from_preset(preset: Mapping[str, Any], *, device: str | torch.device) -> torch.nn.Module:
    model_cfg = dict(preset.get("model", {}))
    vae_cfg = _resolve_vae_config_from_preset(preset)
    if vae_cfg is None:
        raise ValueError("Generator preset must define either a VAE config or a pretrained diffusers VAE.")

    vae = build_vae_from_config(vae_cfg, device=device)
    vae_weights = model_cfg.get("vae_weights")
    if vae_weights:
        load_vae_weights(vae, str(_repo_path(vae_weights)), map_location=device)
    freeze_vae(vae)
    return vae


def _load_stay_sampler(
    *,
    generator_cfg: Mapping[str, Any],
    dataset_payload: Mapping[str, Any],
    device: str | torch.device,
) -> tuple[LayoutFlowMatchingSampler, int]:
    checkpoint_path = _repo_path(generator_cfg["checkpoint_path"])
    preset_path = _repo_path(generator_cfg["preset_path"])
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing STAY checkpoint: {generator_cfg['checkpoint_path']}")
    if preset_path is None or not preset_path.is_file():
        raise FileNotFoundError(f"Missing STAY preset: {generator_cfg['preset_path']}")

    preset = _load_yaml(preset_path)
    layout_cfg = dict(preset.get("layout_conditioning", {}))
    vae_cfg = _resolve_vae_config_from_preset(preset)
    names = _normalise_names(dataset_payload["names"])
    unet_state = _extract_unet_state(checkpoint_path, device=device)
    num_classes = _infer_stay_num_classes(state=unet_state, dataset_names=names)
    unet_cfg = _load_effective_unet_config(
        checkpoint_path=checkpoint_path,
        preset=preset,
        vae_cfg=vae_cfg,
    )
    image_in_channels = int(unet_cfg.get("in_channels", 4))
    unet = build_stay_layout_conditioned_unet(
        unet_cfg,
        image_in_channels=image_in_channels,
        num_classes=num_classes,
        class_embed_dim=int(layout_cfg.get("class_embed_dim", 48)),
        bbox_embed_dim=int(layout_cfg.get("bbox_embed_dim", 48)),
        object_embed_dim=int(layout_cfg.get("object_embed_dim", 64)),
        use_style_latent=bool(layout_cfg.get("use_style_latent", True)),
        style_latent_dim=int(layout_cfg.get("style_latent_dim", 16)),
        style_seed=int(layout_cfg.get("style_seed", 1234)),
        mask_resolution=int(layout_cfg.get("mask_resolution", 16)),
        mask_hidden_channels=int(layout_cfg.get("mask_hidden_channels", 32)),
        mask_threshold=float(layout_cfg.get("mask_threshold", 0.5)),
        edge_dilation=int(layout_cfg.get("edge_dilation", 1)),
        injection_mode=str(layout_cfg.get("injection_mode", "ea_norm")),
        use_masked_context=bool(layout_cfg.get("use_masked_context", True)),
        mask_overlap_loss_weight=float(layout_cfg.get("mask_overlap_loss_weight", 0.0)),
        mask_sharpness_loss_weight=float(layout_cfg.get("mask_sharpness_loss_weight", 0.0)),
        mask_activation_loss_weight=float(layout_cfg.get("mask_activation_loss_weight", 0.0)),
        category_id_to_name=names,
        device=str(device),
    )
    unet.load_state_dict(unet_state, strict=True)
    unet.eval()
    sampler = LayoutFlowMatchingSampler.from_stable(
        unet,
        _build_vae_from_preset(preset, device=device),
        device=device,
        t_scale=float(preset.get("training", {}).get("t_scale", 1000.0)),
        train_target=str(preset.get("training", {}).get("train_target", "v")),
    )
    return sampler, int(preset.get("data", {}).get("image_size", 512))


def _load_regiondiff_sampler(
    *,
    generator_cfg: Mapping[str, Any],
    dataset_payload: Mapping[str, Any],
    device: str | torch.device,
) -> tuple[FlowMatchingSampler, int, dict[int, int]]:
    checkpoint_path = _repo_path(generator_cfg["checkpoint_path"])
    preset_path = _repo_path(generator_cfg["preset_path"])
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing RegionDiff checkpoint: {generator_cfg['checkpoint_path']}")
    if preset_path is None or not preset_path.is_file():
        raise FileNotFoundError(f"Missing RegionDiff preset: {generator_cfg['preset_path']}")

    preset = _load_yaml(preset_path)
    region_cfg = dict(preset.get("layout_conditioning", {}))
    vae_cfg = _resolve_vae_config_from_preset(preset)
    names = _normalise_names(dataset_payload["names"])
    unet_state = _extract_unet_state(checkpoint_path, device=device)
    num_classes = _infer_regiondiff_num_classes(state=unet_state, dataset_names=names)
    checkpoint_names = _regiondiff_checkpoint_category_names(
        generator_cfg,
        dataset_names=names,
        num_classes=num_classes,
    )
    label_id_map = _regiondiff_label_id_map(
        generator_cfg,
        dataset_names=names,
        checkpoint_names=checkpoint_names,
    )
    base_unet = build_fm_unet_from_config(
        _load_effective_unet_config(
            checkpoint_path=checkpoint_path,
            preset=preset,
            vae_cfg=vae_cfg,
        ),
        device=str(device),
    )
    unet = build_regiondiff_wrapper(
        base_model=base_unet,
        region_config=region_cfg,
        category_id_to_name=checkpoint_names,
        backbone_kind="fm_unet2d",
        attachment_kind=str(region_cfg.get("attachment_kind", "attention")),
    ).to(device)
    unet.load_state_dict(unet_state, strict=True)
    unet.eval()
    sampler = FlowMatchingSampler.from_stable(
        unet,
        _build_vae_from_preset(preset, device=device),
        device=device,
        t_scale=float(preset.get("training", {}).get("t_scale", 1000.0)),
        train_target=str(preset.get("training", {}).get("train_target", "v")),
    )
    return sampler, int(preset.get("data", {}).get("image_size", 512)), label_id_map


def _build_sd_uncond_noise_scheduler(preset: Mapping[str, Any]):
    diffusion_cfg = dict(preset.get("diffusion", {}))
    DDPMScheduler = import_diffusers_attr("diffusers", "DDPMScheduler")
    return DDPMScheduler(
        num_train_timesteps=int(diffusion_cfg.get("num_train_timesteps", 1000)),
        beta_schedule=str(diffusion_cfg.get("beta_schedule", "scaled_linear")),
        beta_start=float(diffusion_cfg.get("beta_start", 0.00085)),
        beta_end=float(diffusion_cfg.get("beta_end", 0.012)),
        prediction_type=str(diffusion_cfg.get("prediction_type", "epsilon")),
        clip_sample=False,
    )


def _load_regiondiff_sd_sampler(
    *,
    generator_cfg: Mapping[str, Any],
    dataset_payload: Mapping[str, Any],
    device: str | torch.device,
) -> tuple[UnconditionalStableDiffusionSampler, int, dict[int, int]]:
    checkpoint_path = _repo_path(generator_cfg["checkpoint_path"])
    preset_path = _repo_path(generator_cfg["preset_path"])
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing RegionDiff SD checkpoint: {generator_cfg['checkpoint_path']}"
        )
    if preset_path is None or not preset_path.is_file():
        raise FileNotFoundError(f"Missing RegionDiff SD preset: {generator_cfg['preset_path']}")

    preset = _load_yaml(preset_path)
    region_cfg = dict(preset.get("layout_conditioning", {}))
    if not bool(region_cfg.get("enabled", False)):
        raise ValueError("RegionDiff SD preset must enable layout_conditioning.")
    if str(region_cfg.get("variant", "")) != "regiondiff_v1":
        raise ValueError(
            "RegionDiff SD backend expects layout_conditioning.variant='regiondiff_v1'."
        )

    vae_cfg = _resolve_vae_config_from_preset(preset)
    names = _normalise_names(dataset_payload["names"])
    unet_state = _extract_unet_state(checkpoint_path, device=device)
    num_classes = _infer_regiondiff_num_classes(state=unet_state, dataset_names=names)
    checkpoint_names = _regiondiff_checkpoint_category_names(
        generator_cfg,
        dataset_names=names,
        num_classes=num_classes,
    )
    label_id_map = _regiondiff_label_id_map(
        generator_cfg,
        dataset_names=names,
        checkpoint_names=checkpoint_names,
    )

    base_unet = build_fm_unet_from_config(
        _load_effective_unet_config(
            checkpoint_path=checkpoint_path,
            preset=preset,
            vae_cfg=vae_cfg,
        ),
        device=str(device),
    )
    unet = build_regiondiff_wrapper(
        base_model=base_unet,
        region_config=region_cfg,
        category_id_to_name=checkpoint_names,
        num_classes=num_classes,
        backbone_kind="sd_uncond_unet2d",
        attachment_kind=str(region_cfg.get("attachment_kind", "attention")),
    ).to(device)
    unet.load_state_dict(unet_state, strict=True)
    unet.eval()

    sampler = UnconditionalStableDiffusionSampler.from_stable(
        unet,
        _build_vae_from_preset(preset, device=device),
        _build_sd_uncond_noise_scheduler(preset),
        device=device,
    )
    return sampler, int(preset.get("data", {}).get("image_size", 512)), label_id_map


def _resolve_regiondiff_sd_layout_artifact(
    generator_cfg: Mapping[str, Any],
) -> tuple[Path, Path]:
    stage2_dir = _repo_path(generator_cfg.get("stage2_dir"))
    checkpoint_path = _repo_path(generator_cfg.get("checkpoint_path"))

    if checkpoint_path is not None:
        if checkpoint_path.is_dir():
            if (checkpoint_path / STAGE2_UNET_WEIGHTS_NAME).is_file():
                stage2_dir = checkpoint_path
                checkpoint_path = checkpoint_path / STAGE2_UNET_WEIGHTS_NAME
            elif (checkpoint_path / STAGE2_CHECKPOINT_UNET_WEIGHTS_NAME).is_file():
                stage2_dir = checkpoint_path.parent
                checkpoint_path = checkpoint_path / STAGE2_CHECKPOINT_UNET_WEIGHTS_NAME
        elif checkpoint_path.name == STAGE2_UNET_WEIGHTS_NAME:
            stage2_dir = checkpoint_path.parent
        elif checkpoint_path.name == STAGE2_CHECKPOINT_UNET_WEIGHTS_NAME:
            candidate_stage2_dir = checkpoint_path.parent
            if not (candidate_stage2_dir / STAGE2_LAYOUT_MANIFEST_NAME).is_file():
                candidate_stage2_dir = checkpoint_path.parent.parent
            stage2_dir = candidate_stage2_dir

    if stage2_dir is None:
        raise ValueError(
            "RegionDiff SD-layout generator must define stage2_dir or checkpoint_path."
        )
    if not stage2_dir.is_dir():
        raise FileNotFoundError(f"Missing RegionDiff SD-layout artifact directory: {stage2_dir}")
    if not (stage2_dir / STAGE2_LAYOUT_MANIFEST_NAME).is_file():
        raise FileNotFoundError(
            f"Missing {STAGE2_LAYOUT_MANIFEST_NAME} under RegionDiff SD-layout artifact: {stage2_dir}"
        )
    if checkpoint_path is None:
        checkpoint_path = stage2_dir / STAGE2_UNET_WEIGHTS_NAME
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing RegionDiff SD-layout weights: {checkpoint_path}")
    return stage2_dir, checkpoint_path


def _torch_dtype_from_precision(precision: Any, *, device: str | torch.device) -> torch.dtype | None:
    value = str(precision or "").strip().lower()
    if value in {"", "auto"}:
        value = "fp16" if str(device).startswith("cuda") else "fp32"
    if value in {"fp16", "float16", "half"}:
        return torch.float16 if str(device).startswith("cuda") else torch.float32
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("precision must be one of: auto, fp16, bf16, fp32.")


def _stage2_manifest_prompt_config(
    manifest: Mapping[str, Any],
    generator_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_cfg = dict(generator_cfg.get("prompt", {}) or {})
    return {
        "prompt_mode": str(
            generator_cfg.get(
                "prompt_mode",
                prompt_cfg.get("prompt_mode", manifest.get("prompt_mode", "class_list")),
            )
        ),
        "constant_prompt": str(
            generator_cfg.get(
                "constant_prompt",
                prompt_cfg.get("constant_prompt", manifest.get("constant_prompt", "thermal image")),
            )
        ),
        "thermal_scene_suffix": str(
            generator_cfg.get(
                "thermal_scene_suffix",
                prompt_cfg.get("thermal_scene_suffix", manifest.get("thermal_scene_suffix", "in thermal scene.")),
            )
        ),
        "use_captions_if_available": bool(
            generator_cfg.get(
                "use_captions_if_available",
                prompt_cfg.get("use_captions_if_available", manifest.get("use_captions_if_available", False)),
            )
        ),
    }


def _build_regiondiff_sd_layout_prompts(
    *,
    batch: Mapping[str, Any],
    manifest: Mapping[str, Any],
    generator_cfg: Mapping[str, Any],
) -> list[str]:
    from src.algorithms.stable_diffusion.layout_data import build_layout_prompt

    prompt_cfg = _stage2_manifest_prompt_config(manifest, generator_cfg)
    return [
        build_layout_prompt(
            label_names=label_names,
            prompt_mode=str(prompt_cfg["prompt_mode"]),
            constant_prompt=str(prompt_cfg["constant_prompt"]),
            thermal_scene_suffix=str(prompt_cfg["thermal_scene_suffix"]),
            caption=None,
            use_captions_if_available=bool(prompt_cfg["use_captions_if_available"]),
        )
        for label_names in batch.get("label_names", [])
    ]


def _pipeline_images_to_arrays(images: Sequence[Any]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for image in images:
        rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        if rgb.ndim == 3:
            gray = rgb.astype(np.float32).mean(axis=-1)
        else:
            gray = rgb.astype(np.float32)
        if gray.dtype != np.uint8:
            if float(np.nanmax(gray)) <= 1.0:
                gray = gray * 255.0
            gray = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
        arrays.append(gray)
    return arrays


def _load_regiondiff_sd_layout_pipeline(
    *,
    generator_cfg: Mapping[str, Any],
    dataset_payload: Mapping[str, Any],
    device: str | torch.device,
):
    stage2_dir, checkpoint_path = _resolve_regiondiff_sd_layout_artifact(generator_cfg)
    dtype = _torch_dtype_from_precision(generator_cfg.get("precision", "auto"), device=device)
    pipeline, manifest = _load_stage2_layout_pipeline(
        stage2_dir=str(stage2_dir),
        torch_dtype=dtype,
        base_model=generator_cfg.get("base_model"),
        device=device,
    )
    default_weights = stage2_dir / STAGE2_UNET_WEIGHTS_NAME
    if checkpoint_path.resolve() != default_weights.resolve():
        state = _extract_unet_state(checkpoint_path, device="cpu")
        missing, unexpected = pipeline.unet.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "RegionDiff SD-layout checkpoint did not load cleanly. "
                f"Missing keys={missing[:5]}, unexpected keys={unexpected[:5]}"
            )

    if bool(generator_cfg.get("enable_vae_slicing", manifest.get("enable_vae_slicing", True))):
        vae = getattr(pipeline, "vae", None)
        if vae is not None and hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=bool(generator_cfg.get("disable_progress_bar", True)))
    pipeline = pipeline.to(device)

    names = _normalise_names(dataset_payload["names"])
    checkpoint_names = {
        int(key): str(value)
        for key, value in getattr(pipeline.unet, "category_id_to_name", {}).items()
    }
    if not checkpoint_names:
        region_config_path = stage2_dir / STAGE2_REGIONDIFF_CONFIG_NAME
        if region_config_path.is_file():
            region_config = json.loads(region_config_path.read_text(encoding="utf-8"))
            checkpoint_names = {
                int(key): str(value)
                for key, value in dict(region_config.get("category_id_to_name", {})).items()
            }
    if not checkpoint_names:
        checkpoint_names = names

    label_id_map = _regiondiff_label_id_map(
        generator_cfg,
        dataset_names=names,
        checkpoint_names=checkpoint_names,
    )
    image_size = int(generator_cfg.get("image_size", manifest.get("resolution", 512)))
    return pipeline, dict(manifest), image_size, label_id_map


def generate_stay_fm_arrays(
    *,
    generator_cfg: Mapping[str, Any],
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: Mapping[str, Any],
    device: str,
    seed: int,
) -> list[np.ndarray]:
    sampler, image_size = _load_stay_sampler(generator_cfg=generator_cfg, dataset_payload=dataset_payload, device=device)
    names = _normalise_names(dataset_payload["names"])
    layout_batches = build_layout_batches(
        source_samples,
        image_size=image_size,
        names=names,
        batch_size=int(generator_cfg.get("batch_size", 8)),
    )
    arrays: list[np.ndarray] = []
    start_idx = 0
    for batch in tqdm(layout_batches, desc=f"{generator_cfg.get('name', 'stay_fm')} generation"):
        generated = sample_layout_batch(
            sampler,
            batch,
            steps=int(generator_cfg.get("steps", 50)),
            seed=int(seed) + start_idx,
        )
        arrays.extend(image.detach().cpu().to(torch.float32).numpy() for image in generated)
        start_idx += int(batch["pixel_values"].shape[0])
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return arrays


def generate_stay_fm_dataset(
    *,
    output_dir: str | Path,
    generator_cfg: Mapping[str, Any],
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: dict[str, Any],
    device: str,
    seed: int,
    image_ids: Sequence[int] | None = None,
    initialize: bool = True,
    resume: bool = False,
    max_preview_images: int | None = None,
    max_layout_overlay_images: int | None = None,
) -> int:
    sampler, image_size = _load_stay_sampler(generator_cfg=generator_cfg, dataset_payload=dataset_payload, device=device)
    if initialize:
        initialize_generated_candidate_dataset(
            output_dir=output_dir,
            source_samples=source_samples,
            dataset_payload=dataset_payload,
            generator_kind=str(generator_cfg.get("backend", "stay_fm")),
            generator_config=generator_cfg,
            image_size=image_size,
        )
    names = _normalise_names(dataset_payload["names"])
    ids = list(image_ids) if image_ids is not None else list(range(1, len(source_samples) + 1))
    batch_size = max(1, int(generator_cfg.get("batch_size", 8)))
    generated_count = 0
    for start_idx in tqdm(range(0, len(source_samples), batch_size), desc=f"{generator_cfg.get('name', 'stay_fm')} generation"):
        chunk_samples = source_samples[start_idx : start_idx + batch_size]
        chunk_ids = ids[start_idx : start_idx + batch_size]
        pending = [
            (sample, image_id)
            for sample, image_id in zip(chunk_samples, chunk_ids)
            if not (resume and _generated_sample_exists(output_dir, image_id=int(image_id)))
        ]
        generated_count += len(chunk_samples) - len(pending)
        if not pending:
            continue
        batch = collate_layout_batch([
            _sample_to_layout_dict(sample, image_size=image_size, names=names)
            for sample, _image_id in pending
        ])
        generated = sample_layout_batch(
            sampler,
            batch,
            steps=int(generator_cfg.get("steps", 50)),
            seed=int(seed) + start_idx,
        )
        for (sample, image_id), image in zip(pending, generated):
            _save_generated_array(
                output_dir,
                image_id=int(image_id),
                array=image.detach().cpu().to(torch.float32).numpy(),
                max_preview_images=max_preview_images,
                overlay_sample=sample,
                overlay_names=names,
                max_layout_overlay_images=max_layout_overlay_images,
            )
            generated_count += 1
        del generated
        del batch
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return generated_count


def generate_regiondiff_fm_arrays(
    *,
    generator_cfg: Mapping[str, Any],
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: Mapping[str, Any],
    device: str,
    seed: int,
) -> list[np.ndarray]:
    sampler, image_size, label_id_map = _load_regiondiff_sampler(
        generator_cfg=generator_cfg,
        dataset_payload=dataset_payload,
        device=device,
    )
    names = _normalise_names(dataset_payload["names"])
    layout_batches = build_layout_batches(
        source_samples,
        image_size=image_size,
        names=names,
        batch_size=int(generator_cfg.get("batch_size", 8)),
    )
    arrays: list[np.ndarray] = []
    start_idx = 0
    for batch in tqdm(layout_batches, desc=f"{generator_cfg.get('name', 'regiondiff_fm')} generation"):
        batch = _remap_layout_batch_labels(batch, label_id_map)
        z = sampler.sample_euler_layout(
            batch,
            steps=int(generator_cfg.get("steps", 50)),
            seed=int(seed) + start_idx,
        )
        decoded = sampler.decode(z).detach().cpu()
        arrays.extend(image.to(torch.float32).numpy() for image in decoded)
        start_idx += int(batch["pixel_values"].shape[0])
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return arrays


def generate_regiondiff_fm_dataset(
    *,
    output_dir: str | Path,
    generator_cfg: Mapping[str, Any],
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: dict[str, Any],
    device: str,
    seed: int,
    image_ids: Sequence[int] | None = None,
    initialize: bool = True,
    resume: bool = False,
    max_preview_images: int | None = None,
    max_layout_overlay_images: int | None = None,
) -> int:
    sampler, image_size, label_id_map = _load_regiondiff_sampler(
        generator_cfg=generator_cfg,
        dataset_payload=dataset_payload,
        device=device,
    )
    if initialize:
        initialize_generated_candidate_dataset(
            output_dir=output_dir,
            source_samples=source_samples,
            dataset_payload=dataset_payload,
            generator_kind=str(generator_cfg.get("backend", "regiondiff_fm")),
            generator_config=generator_cfg,
            image_size=image_size,
        )
    names = _normalise_names(dataset_payload["names"])
    ids = list(image_ids) if image_ids is not None else list(range(1, len(source_samples) + 1))
    batch_size = max(1, int(generator_cfg.get("batch_size", 8)))
    generated_count = 0
    for start_idx in tqdm(range(0, len(source_samples), batch_size), desc=f"{generator_cfg.get('name', 'regiondiff_fm')} generation"):
        chunk_samples = source_samples[start_idx : start_idx + batch_size]
        chunk_ids = ids[start_idx : start_idx + batch_size]
        pending = [
            (sample, image_id)
            for sample, image_id in zip(chunk_samples, chunk_ids)
            if not (resume and _generated_sample_exists(output_dir, image_id=int(image_id)))
        ]
        generated_count += len(chunk_samples) - len(pending)
        if not pending:
            continue
        batch = collate_layout_batch([
            _sample_to_layout_dict(sample, image_size=image_size, names=names)
            for sample, _image_id in pending
        ])
        batch = _remap_layout_batch_labels(batch, label_id_map)
        z = sampler.sample_euler_layout(
            batch,
            steps=int(generator_cfg.get("steps", 50)),
            seed=int(seed) + start_idx,
        )
        decoded = sampler.decode(z).detach().cpu()
        for (sample, image_id), image in zip(pending, decoded):
            _save_generated_array(
                output_dir,
                image_id=int(image_id),
                array=image.to(torch.float32).numpy(),
                max_preview_images=max_preview_images,
                overlay_sample=sample,
                overlay_names=names,
                max_layout_overlay_images=max_layout_overlay_images,
            )
            generated_count += 1
        del decoded
        del z
        del batch
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return generated_count


def generate_regiondiff_sd_arrays(
    *,
    generator_cfg: Mapping[str, Any],
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: Mapping[str, Any],
    device: str,
    seed: int,
) -> list[np.ndarray]:
    sampler, image_size, label_id_map = _load_regiondiff_sd_sampler(
        generator_cfg=generator_cfg,
        dataset_payload=dataset_payload,
        device=device,
    )
    names = _normalise_names(dataset_payload["names"])
    layout_batches = build_layout_batches(
        source_samples,
        image_size=image_size,
        names=names,
        batch_size=int(generator_cfg.get("batch_size", 8)),
    )
    arrays: list[np.ndarray] = []
    start_idx = 0
    for batch in tqdm(
        layout_batches,
        desc=f"{generator_cfg.get('name', 'regiondiff_sd')} generation",
    ):
        batch = _remap_layout_batch_labels(batch, label_id_map)
        z = sampler.sample_layout(
            batch,
            steps=int(generator_cfg.get("steps", 50)),
            seed=int(seed) + start_idx,
        )
        decoded = sampler.decode(z).detach().cpu()
        arrays.extend(image.to(torch.float32).numpy() for image in decoded)
        start_idx += int(batch["pixel_values"].shape[0])
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return arrays


def generate_regiondiff_sd_dataset(
    *,
    output_dir: str | Path,
    generator_cfg: Mapping[str, Any],
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: dict[str, Any],
    device: str,
    seed: int,
    image_ids: Sequence[int] | None = None,
    initialize: bool = True,
    resume: bool = False,
    max_preview_images: int | None = None,
    max_layout_overlay_images: int | None = None,
) -> int:
    sampler, image_size, label_id_map = _load_regiondiff_sd_sampler(
        generator_cfg=generator_cfg,
        dataset_payload=dataset_payload,
        device=device,
    )
    if initialize:
        initialize_generated_candidate_dataset(
            output_dir=output_dir,
            source_samples=source_samples,
            dataset_payload=dataset_payload,
            generator_kind=str(generator_cfg.get("backend", "regiondiff_sd")),
            generator_config=generator_cfg,
            image_size=image_size,
        )
    names = _normalise_names(dataset_payload["names"])
    ids = list(image_ids) if image_ids is not None else list(range(1, len(source_samples) + 1))
    batch_size = max(1, int(generator_cfg.get("batch_size", 8)))
    generated_count = 0
    for start_idx in tqdm(
        range(0, len(source_samples), batch_size),
        desc=f"{generator_cfg.get('name', 'regiondiff_sd')} generation",
    ):
        chunk_samples = source_samples[start_idx : start_idx + batch_size]
        chunk_ids = ids[start_idx : start_idx + batch_size]
        pending = [
            (sample, image_id)
            for sample, image_id in zip(chunk_samples, chunk_ids)
            if not (resume and _generated_sample_exists(output_dir, image_id=int(image_id)))
        ]
        generated_count += len(chunk_samples) - len(pending)
        if not pending:
            continue
        batch = collate_layout_batch([
            _sample_to_layout_dict(sample, image_size=image_size, names=names)
            for sample, _image_id in pending
        ])
        batch = _remap_layout_batch_labels(batch, label_id_map)
        z = sampler.sample_layout(
            batch,
            steps=int(generator_cfg.get("steps", 50)),
            seed=int(seed) + start_idx,
        )
        decoded = sampler.decode(z).detach().cpu()
        for (sample, image_id), image in zip(pending, decoded):
            _save_generated_array(
                output_dir,
                image_id=int(image_id),
                array=image.to(torch.float32).numpy(),
                max_preview_images=max_preview_images,
                overlay_sample=sample,
                overlay_names=names,
                max_layout_overlay_images=max_layout_overlay_images,
            )
            generated_count += 1
        del decoded
        del z
        del batch
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return generated_count


def generate_regiondiff_sd_layout_arrays(
    *,
    generator_cfg: Mapping[str, Any],
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: Mapping[str, Any],
    device: str,
    seed: int,
) -> list[np.ndarray]:
    pipeline, manifest, image_size, label_id_map = _load_regiondiff_sd_layout_pipeline(
        generator_cfg=generator_cfg,
        dataset_payload=dataset_payload,
        device=device,
    )
    names = _normalise_names(dataset_payload["names"])
    layout_batches = build_layout_batches(
        source_samples,
        image_size=image_size,
        names=names,
        batch_size=int(generator_cfg.get("batch_size", 1)),
    )
    arrays: list[np.ndarray] = []
    guidance_scale = float(generator_cfg.get("guidance_scale", manifest.get("guidance_scale", 7.5)))
    steps = int(generator_cfg.get("steps", generator_cfg.get("num_inference_steps", 30)))
    for start_idx, batch in zip(
        range(0, len(source_samples), max(1, int(generator_cfg.get("batch_size", 1)))),
        tqdm(layout_batches, desc=f"{generator_cfg.get('name', 'regiondiff_sd_layout')} generation"),
    ):
        batch = _remap_layout_batch_labels(batch, label_id_map)
        prompts = _build_regiondiff_sd_layout_prompts(
            batch=batch,
            manifest=manifest,
            generator_cfg=generator_cfg,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + int(start_idx))
        result = pipeline(
            prompts,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=image_size,
            width=image_size,
            cross_attention_kwargs={
                "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(device),
                "labels": batch["labels"].to(device),
                "object_mask": batch["object_mask"].to(device),
            },
        )
        arrays.extend(_pipeline_images_to_arrays(result.images))
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return arrays


def generate_regiondiff_sd_layout_dataset(
    *,
    output_dir: str | Path,
    generator_cfg: Mapping[str, Any],
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: dict[str, Any],
    device: str,
    seed: int,
    image_ids: Sequence[int] | None = None,
    initialize: bool = True,
    resume: bool = False,
    max_preview_images: int | None = None,
    max_layout_overlay_images: int | None = None,
) -> int:
    pipeline, manifest, image_size, label_id_map = _load_regiondiff_sd_layout_pipeline(
        generator_cfg=generator_cfg,
        dataset_payload=dataset_payload,
        device=device,
    )
    if initialize:
        initialize_generated_candidate_dataset(
            output_dir=output_dir,
            source_samples=source_samples,
            dataset_payload=dataset_payload,
            generator_kind=str(generator_cfg.get("backend", "regiondiff_sd_layout")),
            generator_config=generator_cfg,
            image_size=image_size,
        )
    names = _normalise_names(dataset_payload["names"])
    ids = list(image_ids) if image_ids is not None else list(range(1, len(source_samples) + 1))
    batch_size = max(1, int(generator_cfg.get("batch_size", 1)))
    guidance_scale = float(generator_cfg.get("guidance_scale", manifest.get("guidance_scale", 7.5)))
    steps = int(generator_cfg.get("steps", generator_cfg.get("num_inference_steps", 30)))
    generated_count = 0
    for start_idx in tqdm(
        range(0, len(source_samples), batch_size),
        desc=f"{generator_cfg.get('name', 'regiondiff_sd_layout')} generation",
    ):
        chunk_samples = source_samples[start_idx : start_idx + batch_size]
        chunk_ids = ids[start_idx : start_idx + batch_size]
        pending = [
            (sample, image_id)
            for sample, image_id in zip(chunk_samples, chunk_ids)
            if not (resume and _generated_sample_exists(output_dir, image_id=int(image_id)))
        ]
        generated_count += len(chunk_samples) - len(pending)
        if not pending:
            continue
        batch = collate_layout_batch([
            _sample_to_layout_dict(sample, image_size=image_size, names=names)
            for sample, _image_id in pending
        ])
        batch = _remap_layout_batch_labels(batch, label_id_map)
        prompts = _build_regiondiff_sd_layout_prompts(
            batch=batch,
            manifest=manifest,
            generator_cfg=generator_cfg,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + int(start_idx))
        result = pipeline(
            prompts,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=image_size,
            width=image_size,
            cross_attention_kwargs={
                "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(device),
                "labels": batch["labels"].to(device),
                "object_mask": batch["object_mask"].to(device),
            },
        )
        for (sample, image_id), array in zip(pending, _pipeline_images_to_arrays(result.images)):
            _save_generated_array(
                output_dir,
                image_id=int(image_id),
                array=array,
                max_preview_images=max_preview_images,
                overlay_sample=sample,
                overlay_names=names,
                max_layout_overlay_images=max_layout_overlay_images,
            )
            generated_count += 1
        del result
        del batch
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return generated_count


GENERATOR_BACKENDS: dict[str, Callable[..., list[np.ndarray]]] = {
    "stay_fm": generate_stay_fm_arrays,
    "regiondiff_fm": generate_regiondiff_fm_arrays,
    "regiondiff_sd": generate_regiondiff_sd_arrays,
    "regiondiff_sd_layout": generate_regiondiff_sd_layout_arrays,
}

STREAMING_GENERATOR_BACKENDS: dict[str, Callable[..., int]] = {
    "stay_fm": generate_stay_fm_dataset,
    "regiondiff_fm": generate_regiondiff_fm_dataset,
    "regiondiff_sd": generate_regiondiff_sd_dataset,
    "regiondiff_sd_layout": generate_regiondiff_sd_layout_dataset,
}


def write_filtered_annotations_from_audit(
    *,
    dataset_dir: str | Path,
    instance_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep all generated images, but remove annotations marked invalid."""

    dataset_dir = Path(dataset_dir)
    unfiltered_path = dataset_dir / "annotations_unfiltered.json"
    annotations_path = dataset_dir / "annotations.json"
    if not unfiltered_path.is_file():
        shutil.copy2(annotations_path, unfiltered_path)
    payload = json.loads(unfiltered_path.read_text(encoding="utf-8"))
    valid_annotation_ids = {
        int(row["annotation_id"])
        for row in instance_rows
        if bool(row.get("is_positive", False))
    }
    filtered_annotations = [
        ann for ann in payload.get("annotations", [])
        if int(ann.get("id", -1)) in valid_annotation_ids
    ]
    filtered_payload = {
        "images": list(payload.get("images", [])),
        "annotations": filtered_annotations,
        "categories": list(payload.get("categories", [])),
    }
    _write_json(annotations_path, filtered_payload)

    total = len(payload.get("annotations", []))
    summary = {
        "n_images": len(filtered_payload["images"]),
        "n_annotations_unfiltered": int(total),
        "n_annotations": int(len(filtered_annotations)),
        "n_invalid_annotations_removed": int(total - len(filtered_annotations)),
    }
    _write_json(dataset_dir / "metadata" / "filtered_annotation_summary.json", summary)

    metadata_path = dataset_dir / "metadata" / "summary.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(summary)
        metadata["annotations_path"] = "annotations.json"
        metadata["unfiltered_annotations_path"] = "annotations_unfiltered.json"
        _write_json(metadata_path, metadata)
    return summary


def _load_filter(config: Mapping[str, Any], *, device: str) -> tuple[torch.nn.Module, dict[str, Any], Any, int, float, Path | None]:
    filter_cfg = dict(config.get("filter", {}))
    run_dir = filter_cfg.get("run_dir") or filter_cfg.get("filter_run_dir") or DEFAULT_FILTER_RUN_DIR
    checkpoint_path = filter_cfg.get("checkpoint_path")
    model, summary, threshold, input_size, context_ratio, resolved_run_dir = load_filter_from_run_or_checkpoint(
        device=device,
        run_dir=run_dir if checkpoint_path in (None, "") else None,
        checkpoint_path=checkpoint_path or None,
        threshold_override=None,
    )
    if str(summary.get("classifier_mode", "")) != "multiclass":
        raise ValueError("Production synthetic filtering requires a multiclass foreground/background filter.")
    return model, summary, threshold, input_size, context_ratio, resolved_run_dir


def _audit_and_filter_dataset(
    *,
    dataset_dir: str | Path,
    config: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    model, summary, threshold, input_size, context_ratio, resolved_run_dir = _load_filter(config, device=device)
    audit_summary, _instance_rows, _image_rows = _audit_dataset_with_loaded_filter(
        dataset_dir=dataset_dir,
        config=config,
        device=device,
        model=model,
        summary=summary,
        threshold=threshold,
        input_size=input_size,
        context_ratio=context_ratio,
        resolved_run_dir=resolved_run_dir,
        write_filtered_annotations=True,
    )
    return audit_summary


def _audit_dataset_with_loaded_filter(
    *,
    dataset_dir: str | Path,
    config: Mapping[str, Any],
    device: str,
    model: torch.nn.Module,
    summary: Mapping[str, Any],
    threshold: Any,
    input_size: int,
    context_ratio: float,
    resolved_run_dir: Path | None,
    write_filtered_annotations: bool,
    export_results: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    filter_cfg = dict(config.get("filter", {}))
    instance_rows, image_rows, stats = audit_generated_layout_dataset(
        dataset_dir=dataset_dir,
        filter_model=model,
        threshold=threshold,
        filter_input_size=input_size,
        context_ratio=context_ratio,
        device=device,
        crop_batch_size=int(filter_cfg.get("batch_size", 64)),
        show_progress=True,
        classifier_summary=summary,
    )
    audit_dir = Path(dataset_dir) / "filter_audit"
    if export_results:
        _export_loaded_filter_audit_results(
            audit_dir=audit_dir,
            dataset_dir=dataset_dir,
            resolved_run_dir=resolved_run_dir,
            summary=summary,
            threshold=threshold,
            input_size=input_size,
            context_ratio=context_ratio,
            stats=stats,
            instance_rows=instance_rows,
            image_rows=image_rows,
        )
    filtered_summary = (
        write_filtered_annotations_from_audit(dataset_dir=dataset_dir, instance_rows=instance_rows)
        if write_filtered_annotations
        else {"enabled": False}
    )
    return {
        "audit_dir": str(audit_dir),
        "filter_run_dir": None if resolved_run_dir is None else str(resolved_run_dir),
        "filter_stats": stats,
        "filtered_annotation_summary": filtered_summary,
    }, instance_rows, image_rows


def _export_loaded_filter_audit_results(
    *,
    audit_dir: Path,
    dataset_dir: str | Path,
    resolved_run_dir: Path | None,
    summary: Mapping[str, Any],
    threshold: Any,
    input_size: int,
    context_ratio: float,
    stats: Mapping[str, Any],
    instance_rows: Sequence[dict[str, Any]],
    image_rows: Sequence[dict[str, Any]],
) -> None:
    export_audit_results(
        output_dir=audit_dir,
        summary={
            "generated_dataset_dir": str(Path(dataset_dir)),
            "filter_run_dir": None if resolved_run_dir is None else str(resolved_run_dir),
            "checkpoint_summary": summary,
            "threshold": dict(threshold) if isinstance(threshold, dict) else float(threshold),
            "input_size": int(input_size),
            "context_ratio": float(context_ratio),
            "max_discarded_valid_threshold": 1.0,
            "score_alpha": 1.0,
            "score_beta": 1.0,
            "stats": dict(stats),
            "min_valid_object_fraction_sweep": "",
        },
        instance_rows=instance_rows,
        image_rows=image_rows,
    )


def render_sanity_check_images(
    *,
    dataset_dir: str | Path,
    max_images: int = 24,
) -> list[str]:
    """Render generated images with valid/invalid boxes and classifier scores."""

    dataset_dir = Path(dataset_dir)
    audit_path = dataset_dir / "filter_audit" / "per_instance_manifest.jsonl"
    if not audit_path.is_file():
        return []
    rows_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows_by_image[int(row["generated_image_id"])].append(row)

    output_dir = dataset_dir / "sanity_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for image_id in sorted(rows_by_image)[: max(0, int(max_images))]:
        rows = rows_by_image[image_id]
        image_path = dataset_dir / "images" / str(rows[0]["generated_file_name"])
        canvas = Image.fromarray(_array_to_png_uint8(np.load(image_path)), mode="L").convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for row in rows:
            color = "#18a558" if bool(row.get("is_positive", False)) else "#d1495b"
            x1, y1, x2, y2 = [float(value) for value in row["bbox_xyxy"]]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            text = (
                f"{row.get('expected_category_name')} -> {row.get('predicted_category_name')} "
                f"p={float(row.get('expected_class_probability', row.get('probability', 0.0))):.2f}"
            )
            tx = max(0, int(x1))
            ty = max(0, int(y1) - 12)
            draw.rectangle((tx, ty, tx + min(280, 7 * len(text)), ty + 11), fill=(0, 0, 0))
            draw.text((tx + 2, ty), text, fill=color)
        output_path = output_dir / f"sample_{image_id:06d}.png"
        canvas.save(output_path)
        saved_paths.append(str(output_path))
    _write_json(output_dir / "summary.json", {"paths": saved_paths, "n_images": len(saved_paths)})
    return saved_paths


def render_layout_overlay_previews(
    *,
    dataset_dir: str | Path,
    max_images: int = 24,
    annotations_filename: str = "annotations_unfiltered.json",
    output_dir_name: str = "layout_overlays",
) -> list[str]:
    """Render generated images with the source layout boxes used for conditioning."""

    dataset_dir = Path(dataset_dir)
    annotations_path = dataset_dir / annotations_filename
    if not annotations_path.is_file():
        return []
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in payload.get("categories", [])
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    output_dir = dataset_dir / str(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    selected_images = sorted(
        payload.get("images", []),
        key=lambda item: int(item.get("id", 0)),
    )[: max(0, int(max_images))]
    for image_info in selected_images:
        image_id = int(image_info["id"])
        image_path = dataset_dir / "images" / str(image_info["file_name"])
        if not image_path.is_file():
            continue
        canvas = Image.fromarray(_array_to_png_uint8(np.load(image_path)), mode="L").convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for annotation in annotations_by_image.get(image_id, []):
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            x2 = x + max(0.0, width)
            y2 = y + max(0.0, height)
            category_id = int(annotation["category_id"])
            color = _category_color(category_id)
            draw.rectangle((x, y, x2, y2), outline=color, width=2)
            label = categories.get(category_id, str(category_id))
            tx = max(0, int(x))
            ty = max(0, int(y) - 12)
            draw.rectangle((tx, ty, tx + min(160, 7 * len(label) + 4), ty + 11), fill=(0, 0, 0))
            draw.text((tx + 2, ty), label, fill=color)
        output_path = output_dir / f"sample_{image_id:06d}.png"
        canvas.save(output_path)
        saved_paths.append(str(output_path))
    _write_json(
        output_dir / "summary.json",
        {
            "paths": saved_paths,
            "n_images": len(saved_paths),
            "n_annotations": len(payload.get("annotations", [])),
            "annotations_path": annotations_filename,
            "output_dir": str(output_dir),
        },
    )
    return saved_paths


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


def render_filter_crop_contact_sheets(
    *,
    dataset_dir: str | Path,
    max_crops_per_sheet: int = 24,
) -> list[str]:
    """Save quick valid/invalid bbox crop sheets from the filter audit manifest."""

    dataset_dir = Path(dataset_dir)
    audit_path = dataset_dir / "filter_audit" / "per_instance_manifest.jsonl"
    if not audit_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    saved: list[str] = []
    output_dir = dataset_dir / "sanity_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, want_positive in (("valid", True), ("invalid", False)):
        selected = [row for row in rows if bool(row.get("is_positive", False)) is want_positive][: max(0, int(max_crops_per_sheet))]
        if not selected:
            continue
        thumbs: list[Image.Image] = []
        for row in selected:
            image_path = dataset_dir / "images" / str(row["generated_file_name"])
            image = _array_to_png_uint8(np.load(image_path))
            x1, y1, x2, y2 = [int(round(float(v))) for v in row["bbox_xyxy"]]
            crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                crop = np.zeros((8, 8), dtype=np.uint8)
            tile = Image.fromarray(crop, mode="L").convert("RGB").resize((96, 96))
            draw = ImageDraw.Draw(tile)
            text = (
                f"{row.get('expected_category_name')}->{row.get('predicted_category_name')}\n"
                f"p={float(row.get('expected_class_probability', row.get('probability', 0.0))):.2f}"
            )
            draw.rectangle((0, 72, 96, 96), fill=(0, 0, 0))
            draw.text((2, 74), text, fill=(24, 165, 88) if want_positive else (209, 73, 91))
            thumbs.append(tile)
        cols = min(6, len(thumbs))
        rows_n = int(math.ceil(len(thumbs) / cols))
        sheet = Image.new("RGB", (cols * 96, rows_n * 96), color=(20, 20, 20))
        for idx, tile in enumerate(thumbs):
            sheet.paste(tile, ((idx % cols) * 96, (idx // cols) * 96))
        output_path = output_dir / f"{label}_bbox_crops.png"
        sheet.save(output_path)
        saved.append(str(output_path))
    return saved


def compute_metric_summary_from_features(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    *,
    kid_subsets: int,
    kid_subset_size: int,
    kid_seed: int,
    mmd_bandwidths: Sequence[float],
) -> dict[str, Any]:
    if min(len(real_features), len(generated_features)) < 2:
        return {
            "skipped": True,
            "reason": "At least two real and two generated samples are required.",
            "fid": None,
            "kid": None,
            "mmd": None,
        }
    active_kid_subset = min(int(kid_subset_size), int(real_features.shape[0]), int(generated_features.shape[0]))
    return {
        "skipped": False,
        "fid": compute_fid(real_features, generated_features),
        "kid": compute_kid(
            real_features,
            generated_features,
            subsets=int(kid_subsets),
            subset_size=active_kid_subset,
            seed=int(kid_seed),
        ),
        "mmd": compute_rbf_mmd(real_features, generated_features, bandwidths=mmd_bandwidths),
        "kid_subset_size": int(active_kid_subset),
    }


def compute_distribution_metrics(
    *,
    dataset_dir: str | Path,
    source_samples: Sequence[YOLOTrainSample],
    config: Mapping[str, Any],
    device: str,
    seed: int,
) -> dict[str, Any]:
    metrics_cfg = dict(config.get("metrics", {}))
    if not bool(metrics_cfg.get("enabled", True)):
        return {"enabled": False}
    dataset_dir = Path(dataset_dir)
    metrics_dir = dataset_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    real_paths = [sample.image_path for sample in source_samples]
    generated_paths = sorted((dataset_dir / "images").glob("sample_*.npy"))
    if min(len(real_paths), len(generated_paths)) < 2:
        summary = {
            "enabled": True,
            "skipped": True,
            "reason": "At least two real and two generated images are required.",
            "num_real": len(real_paths),
            "num_generated": len(generated_paths),
        }
        _write_json(metrics_dir / "summary.json", summary)
        _write_csv(metrics_dir / "summary.csv", [summary])
        return summary

    extractor_name = str(metrics_cfg.get("feature_extractor", "inception"))
    extractor = build_feature_extractor(extractor_name, metrics_cfg, device)
    batch_size = int(metrics_cfg.get("batch_size", 16))
    force = bool(metrics_cfg.get("force", False))
    real_features = extract_features(
        real_paths,
        extractor,
        batch_size=batch_size,
        cache_path=metrics_dir / f"real_{extractor_name}_features.npz",
        force=force,
        metadata={"source": "real"},
    )
    generated_features = extract_features(
        generated_paths,
        extractor,
        batch_size=batch_size,
        cache_path=metrics_dir / f"generated_{extractor_name}_features.npz",
        force=force,
        metadata={"source": "generated"},
    )
    summary = {
        "enabled": True,
        "feature_extractor": extractor_name,
        "num_real": int(real_features.shape[0]),
        "num_generated": int(generated_features.shape[0]),
        **compute_metric_summary_from_features(
            real_features,
            generated_features,
            kid_subsets=int(metrics_cfg.get("kid", {}).get("subsets", 100)),
            kid_subset_size=int(metrics_cfg.get("kid", {}).get("subset_size", 1000)),
            kid_seed=int(metrics_cfg.get("kid", {}).get("seed", seed)),
            mmd_bandwidths=metrics_cfg.get("mmd", {}).get("bandwidths", [0.1, 1.0, 10.0]),
        ),
    }
    _write_json(metrics_dir / "summary.json", summary)
    _write_csv(metrics_dir / "summary.csv", [summary])
    return summary


def load_generation_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    path = _repo_path(config_path)
    if path is None or not path.is_file():
        return {}
    return _load_yaml(path)


def _select_generators(config: Mapping[str, Any], names: Sequence[str] | None = None) -> list[dict[str, Any]]:
    generators = [dict(item) for item in config.get("generators", [])]
    if not generators:
        raise ValueError("Synthetic generation config must define at least one generator.")
    if names:
        wanted = {str(name) for name in names}
        generators = [item for item in generators if str(item.get("name")) in wanted]
        missing = wanted.difference(str(item.get("name")) for item in generators)
        if missing:
            raise ValueError(f"Unknown generator name(s): {sorted(missing)}")
    return generators


def _retry_config(config: Mapping[str, Any]) -> dict[str, Any]:
    retry_cfg = dict(config.get("retry", {}))
    threshold = float(retry_cfg.get("invalid_instance_ratio_threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("retry.invalid_instance_ratio_threshold must be in [0, 1].")
    max_tries = int(retry_cfg.get("max_tries", 1))
    if max_tries < 1:
        raise ValueError("retry.max_tries must be >= 1.")
    return {
        "enabled": bool(retry_cfg.get("enabled", max_tries > 1)),
        "invalid_instance_ratio_threshold": threshold,
        "max_tries": max_tries,
        "seed_stride": int(retry_cfg.get("seed_stride", 1_000_000)),
    }


def _invalid_ratio_from_image_row(row: Mapping[str, Any]) -> float:
    return float(int(row.get("n_negative_instances", 0) or 0) / max(1, int(row.get("n_instances", 0) or 0)))


def _retry_generation_with_filter(
    *,
    output_dir: Path,
    source_samples: Sequence[YOLOTrainSample],
    initial_arrays: Sequence[np.ndarray],
    dataset_payload: dict[str, Any],
    generator_kind: str,
    generator_config: Mapping[str, Any],
    backend: Callable[..., list[np.ndarray]],
    active_config: Mapping[str, Any],
    device: str,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, Any], dict[str, Any]]:
    retry_cfg = _retry_config(active_config)
    arrays = [np.asarray(array) for array in initial_arrays]
    model, summary, threshold, input_size, context_ratio, resolved_run_dir = _load_filter(active_config, device=device)
    max_tries = int(retry_cfg["max_tries"]) if bool(retry_cfg["enabled"]) else 1
    invalid_threshold = float(retry_cfg["invalid_instance_ratio_threshold"])
    seed_stride = int(retry_cfg["seed_stride"])
    retry_rows: list[dict[str, Any]] = []
    final_audit: dict[str, Any] = {"enabled": False}
    final_instance_rows: list[dict[str, Any]] = []
    final_image_rows: list[dict[str, Any]] = []
    max_preview_images = int(active_config.get("sanity", {}).get("max_images", 24))
    max_layout_overlays = int(
        active_config.get("sanity", {}).get(
            "max_layout_overlays",
            active_config.get("sanity", {}).get("max_images", 24),
        )
    )

    for attempt_idx in range(1, max_tries + 1):
        export_generated_candidate_dataset(
            output_dir=output_dir,
            source_samples=source_samples,
            generated_arrays=arrays,
            dataset_payload=dataset_payload,
            generator_kind=generator_kind,
            generator_config=generator_config,
            max_preview_images=max_preview_images,
            max_layout_overlay_images=max_layout_overlays,
        )
        final_audit, final_instance_rows, final_image_rows = _audit_dataset_with_loaded_filter(
            dataset_dir=output_dir,
            config=active_config,
            device=device,
            model=model,
            summary=summary,
            threshold=threshold,
            input_size=input_size,
            context_ratio=context_ratio,
            resolved_run_dir=resolved_run_dir,
            write_filtered_annotations=False,
            export_results=False,
        )
        failed_image_ids: list[int] = []
        for row in final_image_rows:
            image_id = int(row["generated_image_id"])
            invalid_ratio = _invalid_ratio_from_image_row(row)
            should_retry = bool(invalid_ratio > invalid_threshold and attempt_idx < max_tries)
            if should_retry:
                failed_image_ids.append(image_id)
            retry_rows.append(
                {
                    "generated_image_id": image_id,
                    "generated_file_name": row.get("generated_file_name"),
                    "attempt": attempt_idx,
                    "n_instances": int(row.get("n_instances", 0) or 0),
                    "n_invalid_instances": int(row.get("n_negative_instances", 0) or 0),
                    "invalid_instance_ratio": invalid_ratio,
                    "retry_threshold": invalid_threshold,
                    "will_retry": should_retry,
                    "accepted": bool(invalid_ratio <= invalid_threshold or attempt_idx >= max_tries),
                    "exhausted_max_tries": bool(invalid_ratio > invalid_threshold and attempt_idx >= max_tries),
                }
            )
        if not failed_image_ids:
            break

        failed_samples = [source_samples[image_id - 1] for image_id in failed_image_ids]
        retry_seed = int(seed) + seed_stride * attempt_idx
        regenerated_arrays = backend(
            generator_cfg=generator_config,
            source_samples=failed_samples,
            dataset_payload=dataset_payload,
            device=device,
            seed=retry_seed,
        )
        if len(regenerated_arrays) != len(failed_image_ids):
            raise RuntimeError(
                f"Retry generated {len(regenerated_arrays)} images for {len(failed_image_ids)} failed source images."
            )
        for image_id, regenerated_array in zip(failed_image_ids, regenerated_arrays):
            arrays[image_id - 1] = np.asarray(regenerated_array)

    filtered_summary = write_filtered_annotations_from_audit(
        dataset_dir=output_dir,
        instance_rows=final_instance_rows,
    )
    _export_loaded_filter_audit_results(
        audit_dir=output_dir / "filter_audit",
        dataset_dir=output_dir,
        resolved_run_dir=resolved_run_dir,
        summary=summary,
        threshold=threshold,
        input_size=input_size,
        context_ratio=context_ratio,
        stats=final_audit.get("filter_stats", {}),
        instance_rows=final_instance_rows,
        image_rows=final_image_rows,
    )
    final_audit["filtered_annotation_summary"] = filtered_summary

    final_by_image: dict[int, dict[str, Any]] = {}
    for row in retry_rows:
        final_by_image[int(row["generated_image_id"])] = row
    retry_summary = {
        "enabled": bool(retry_cfg["enabled"]),
        "max_tries": max_tries,
        "invalid_instance_ratio_threshold": invalid_threshold,
        "n_images": len(source_samples),
        "n_attempt_rows": len(retry_rows),
        "n_retried_images": len({int(row["generated_image_id"]) for row in retry_rows if bool(row.get("will_retry", False))}),
        "n_exhausted_images": sum(int(bool(row.get("exhausted_max_tries", False))) for row in final_by_image.values()),
        "per_image_manifest_path": "metadata/retry_manifest.jsonl",
    }
    _write_jsonl(output_dir / "metadata" / "retry_manifest.jsonl", retry_rows)
    _write_json(output_dir / "metadata" / "retry_summary.json", retry_summary)
    return arrays, final_audit, retry_summary


def _retry_streamed_generation_with_filter(
    *,
    output_dir: Path,
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: dict[str, Any],
    generator_config: Mapping[str, Any],
    streaming_backend: Callable[..., int],
    active_config: Mapping[str, Any],
    device: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    retry_cfg = _retry_config(active_config)
    model, summary, threshold, input_size, context_ratio, resolved_run_dir = _load_filter(active_config, device=device)
    max_tries = int(retry_cfg["max_tries"]) if bool(retry_cfg["enabled"]) else 1
    invalid_threshold = float(retry_cfg["invalid_instance_ratio_threshold"])
    seed_stride = int(retry_cfg["seed_stride"])
    retry_rows: list[dict[str, Any]] = []
    final_audit: dict[str, Any] = {"enabled": False}
    final_instance_rows: list[dict[str, Any]] = []
    final_image_rows: list[dict[str, Any]] = []

    for attempt_idx in range(1, max_tries + 1):
        final_audit, final_instance_rows, final_image_rows = _audit_dataset_with_loaded_filter(
            dataset_dir=output_dir,
            config=active_config,
            device=device,
            model=model,
            summary=summary,
            threshold=threshold,
            input_size=input_size,
            context_ratio=context_ratio,
            resolved_run_dir=resolved_run_dir,
            write_filtered_annotations=False,
            export_results=False,
        )
        failed_image_ids: list[int] = []
        for row in final_image_rows:
            image_id = int(row["generated_image_id"])
            invalid_ratio = _invalid_ratio_from_image_row(row)
            should_retry = bool(invalid_ratio > invalid_threshold and attempt_idx < max_tries)
            if should_retry:
                failed_image_ids.append(image_id)
            retry_rows.append(
                {
                    "generated_image_id": image_id,
                    "generated_file_name": row.get("generated_file_name"),
                    "attempt": attempt_idx,
                    "n_instances": int(row.get("n_instances", 0) or 0),
                    "n_invalid_instances": int(row.get("n_negative_instances", 0) or 0),
                    "invalid_instance_ratio": invalid_ratio,
                    "retry_threshold": invalid_threshold,
                    "will_retry": should_retry,
                    "accepted": bool(invalid_ratio <= invalid_threshold or attempt_idx >= max_tries),
                    "exhausted_max_tries": bool(invalid_ratio > invalid_threshold and attempt_idx >= max_tries),
                }
            )
        if not failed_image_ids:
            break

        failed_samples = [source_samples[image_id - 1] for image_id in failed_image_ids]
        retry_seed = int(seed) + seed_stride * attempt_idx
        regenerated_count = streaming_backend(
            output_dir=output_dir,
            generator_cfg=generator_config,
            source_samples=failed_samples,
            dataset_payload=dataset_payload,
            device=device,
            seed=retry_seed,
            image_ids=failed_image_ids,
            initialize=False,
            resume=False,
            max_preview_images=int(active_config.get("sanity", {}).get("max_images", 24)),
            max_layout_overlay_images=int(
                active_config.get("sanity", {}).get(
                    "max_layout_overlays",
                    active_config.get("sanity", {}).get("max_images", 24),
                )
            ),
        )
        if int(regenerated_count) != len(failed_image_ids):
            raise RuntimeError(
                f"Retry generated {regenerated_count} images for {len(failed_image_ids)} failed source images."
            )

    filtered_summary = write_filtered_annotations_from_audit(
        dataset_dir=output_dir,
        instance_rows=final_instance_rows,
    )
    _export_loaded_filter_audit_results(
        audit_dir=output_dir / "filter_audit",
        dataset_dir=output_dir,
        resolved_run_dir=resolved_run_dir,
        summary=summary,
        threshold=threshold,
        input_size=input_size,
        context_ratio=context_ratio,
        stats=final_audit.get("filter_stats", {}),
        instance_rows=final_instance_rows,
        image_rows=final_image_rows,
    )
    final_audit["filtered_annotation_summary"] = filtered_summary

    final_by_image: dict[int, dict[str, Any]] = {}
    for row in retry_rows:
        final_by_image[int(row["generated_image_id"])] = row
    retry_summary = {
        "enabled": bool(retry_cfg["enabled"]),
        "max_tries": max_tries,
        "invalid_instance_ratio_threshold": invalid_threshold,
        "n_images": len(source_samples),
        "n_attempt_rows": len(retry_rows),
        "n_retried_images": len({int(row["generated_image_id"]) for row in retry_rows if bool(row.get("will_retry", False))}),
        "n_exhausted_images": sum(int(bool(row.get("exhausted_max_tries", False))) for row in final_by_image.values()),
        "per_image_manifest_path": "metadata/retry_manifest.jsonl",
    }
    _write_jsonl(output_dir / "metadata" / "retry_manifest.jsonl", retry_rows)
    _write_json(output_dir / "metadata" / "retry_summary.json", retry_summary)
    return final_audit, retry_summary


def generate_production_synthetic_datasets(
    *,
    config: Mapping[str, Any],
    yolo_dataset_yaml: str | Path | None = None,
    output_root: str | Path | None = None,
    max_samples: int | None = None,
    generator_names: Sequence[str] | None = None,
    device: str | None = None,
    skip_filter: bool = False,
    skip_metrics: bool = False,
    metrics_only: bool = False,
) -> dict[str, Any]:
    active_config: dict[str, Any] = dict(config)
    resume = bool(active_config.get("resume", False))
    active_device = device or str(active_config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_yaml = yolo_dataset_yaml or active_config.get("yolo_dataset_yaml") or DEFAULT_YOLO_DATASET_YAML
    root = Path(output_root or active_config.get("output_root") or DEFAULT_OUTPUT_ROOT)
    if not root.is_absolute():
        root = repo_root() / root
    seed = int(active_config.get("seed", 7))
    np.random.seed(seed)
    torch.manual_seed(seed)

    source_samples, dataset_payload = load_full_train_samples(dataset_yaml)
    if max_samples is None:
        raw_max = active_config.get("max_samples")
        max_samples = None if raw_max in (None, "", "null") else int(raw_max)
    if max_samples is not None:
        source_samples = source_samples[: max(0, int(max_samples))]
    if not source_samples:
        raise ValueError("No source samples selected for synthetic generation.")

    generators = _select_generators(active_config, generator_names)
    results: list[dict[str, Any]] = []
    for generator_cfg in generators:
        name = str(generator_cfg.get("name") or generator_cfg.get("backend"))
        backend_name = str(generator_cfg.get("backend", ""))
        backend = GENERATOR_BACKENDS.get(backend_name)
        streaming_backend = STREAMING_GENERATOR_BACKENDS.get(backend_name)
        if backend is None:
            raise ValueError(f"Unsupported generator backend={backend_name!r}.")
        output_dir = root / name
        if metrics_only:
            if not output_dir.exists():
                raise FileNotFoundError(f"Cannot compute metrics for missing generated dataset: {output_dir}")
            generated_paths = sorted((output_dir / "images").glob("sample_*.npy"))
            metrics_config = dict(active_config)
            if skip_metrics:
                metrics_config["metrics"] = {**dict(metrics_config.get("metrics", {})), "enabled": False}
            metrics_summary = compute_distribution_metrics(
                dataset_dir=output_dir,
                source_samples=source_samples,
                config=metrics_config,
                device=active_device,
                seed=seed,
            )
            result = {
                "name": name,
                "backend": backend_name,
                "output_dir": str(output_dir),
                "annotations_path": str(output_dir / "annotations.json"),
                "unfiltered_annotations_path": str(output_dir / "annotations_unfiltered.json"),
                "n_source_images": len(source_samples),
                "n_generated_images": len(generated_paths),
                "audit": {"enabled": False, "skipped": True, "reason": "metrics_only"},
                "retry": {"enabled": False, "skipped": True, "reason": "metrics_only"},
                "layout_overlay_paths": [],
                "filtered_layout_overlay_paths": [],
                "sanity_check_paths": [],
                "metrics": metrics_summary,
            }
            _write_json(output_dir / "metadata" / "production_summary.json", result)
            results.append(result)
            continue
        if bool(active_config.get("overwrite", True)) and not resume and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        active_seed = seed + int(generator_cfg.get("seed_offset", 0))
        max_preview_images = int(active_config.get("sanity", {}).get("max_images", 24))
        max_layout_overlays = int(
            active_config.get("sanity", {}).get(
                "max_layout_overlays",
                active_config.get("sanity", {}).get("max_images", 24),
            )
        )
        if streaming_backend is not None:
            n_generated = streaming_backend(
                output_dir=output_dir,
                generator_cfg=generator_cfg,
                source_samples=source_samples,
                dataset_payload=dataset_payload,
                device=active_device,
                seed=active_seed,
                initialize=True,
                resume=resume,
                max_preview_images=max_preview_images,
                max_layout_overlay_images=max_layout_overlays,
            )
            if int(n_generated) != len(source_samples):
                raise RuntimeError(f"{name} generated {n_generated} images for {len(source_samples)} source images.")
        else:
            arrays = backend(
                generator_cfg=generator_cfg,
                source_samples=source_samples,
                dataset_payload=dataset_payload,
                device=active_device,
                seed=active_seed,
            )
            if len(arrays) != len(source_samples):
                raise RuntimeError(f"{name} generated {len(arrays)} images for {len(source_samples)} source images.")
            export_generated_candidate_dataset(
                output_dir=output_dir,
                source_samples=source_samples,
                generated_arrays=arrays,
                dataset_payload=dataset_payload,
                generator_kind=backend_name,
                generator_config=generator_cfg,
                max_preview_images=max_preview_images,
                max_layout_overlay_images=max_layout_overlays,
            )
            n_generated = len(arrays)
        audit_summary: dict[str, Any] = {"enabled": False}
        retry_summary: dict[str, Any] = {"enabled": False}
        if not skip_filter and bool(active_config.get("filter", {}).get("enabled", True)):
            if streaming_backend is not None:
                audit_summary, retry_summary = _retry_streamed_generation_with_filter(
                    output_dir=output_dir,
                    source_samples=source_samples,
                    dataset_payload=dataset_payload,
                    generator_config=generator_cfg,
                    streaming_backend=streaming_backend,
                    active_config=active_config,
                    device=active_device,
                    seed=active_seed,
                )
            else:
                arrays, audit_summary, retry_summary = _retry_generation_with_filter(
                    output_dir=output_dir,
                    source_samples=source_samples,
                    initial_arrays=arrays,
                    dataset_payload=dataset_payload,
                    generator_kind=backend_name,
                    generator_config=generator_cfg,
                    backend=backend,
                    active_config=active_config,
                    device=active_device,
                    seed=active_seed,
                )
                n_generated = len(arrays)
        layout_overlay_paths = render_layout_overlay_previews(
            dataset_dir=output_dir,
            max_images=max_layout_overlays,
            annotations_filename="annotations_unfiltered.json",
            output_dir_name="layout_overlays",
        )
        filtered_layout_overlay_paths = render_layout_overlay_previews(
            dataset_dir=output_dir,
            max_images=max_layout_overlays,
            annotations_filename="annotations.json",
            output_dir_name="filtered_layout_overlays",
        )
        sanity_paths = render_sanity_check_images(
            dataset_dir=output_dir,
            max_images=int(active_config.get("sanity", {}).get("max_images", 24)),
        )
        sanity_paths.extend(
            render_filter_crop_contact_sheets(
                dataset_dir=output_dir,
                max_crops_per_sheet=int(active_config.get("sanity", {}).get("max_crops_per_sheet", 24)),
            )
        )
        metrics_config = dict(active_config)
        if skip_metrics:
            metrics_config["metrics"] = {**dict(metrics_config.get("metrics", {})), "enabled": False}
        metrics_summary = compute_distribution_metrics(
            dataset_dir=output_dir,
            source_samples=source_samples,
            config=metrics_config,
            device=active_device,
            seed=seed,
        )
        result = {
            "name": name,
            "backend": backend_name,
            "output_dir": str(output_dir),
            "annotations_path": str(output_dir / "annotations.json"),
            "unfiltered_annotations_path": str(output_dir / "annotations_unfiltered.json"),
            "n_source_images": len(source_samples),
            "n_generated_images": int(n_generated),
            "audit": audit_summary,
            "retry": retry_summary,
            "layout_overlay_paths": layout_overlay_paths,
            "filtered_layout_overlay_paths": filtered_layout_overlay_paths,
            "sanity_check_paths": sanity_paths,
            "metrics": metrics_summary,
        }
        _write_json(output_dir / "metadata" / "production_summary.json", result)
        results.append(result)

    summary = {
        "yolo_dataset_yaml": str(_repo_path(dataset_yaml)),
        "output_root": str(root),
        "device": active_device,
        "n_source_images": len(source_samples),
        "generators": results,
    }
    _write_json(root / "summary.json", summary)
    return summary


def generate_regiondiff_candidate_dataset(
    *,
    model_kind: str,
    artifact_dir: str | Path,
    yolo_dataset_yaml: str | Path,
    output_dir: str | Path,
    max_samples: int = 2,
    batch_size: int = 1,
    image_size: int = 512,
    steps: int = 2,
    seed: int = 7,
    device: str = "cpu",
    t_scale: float = 1000.0,
    train_target: str = "v",
    guidance_scale: float = 1.0,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Backward-compatible tiny export for older smoke callers.

    This keeps the old API alive for test fixtures and ad-hoc smoke runs. The
    production entrypoint is :func:`generate_production_synthetic_datasets`.
    """

    del model_kind, artifact_dir, batch_size, image_size, steps, seed, device, t_scale, train_target, guidance_scale, precision
    source_samples, dataset_payload = load_full_train_samples(yolo_dataset_yaml)
    source_samples = source_samples[: max(0, int(max_samples))]
    arrays = [np.zeros((512, 512), dtype=np.float32) for _ in source_samples]
    output = export_generated_candidate_dataset(
        output_dir=output_dir,
        source_samples=source_samples,
        generated_arrays=arrays,
        dataset_payload=dataset_payload,
        generator_kind="legacy_smoke_placeholder",
    )
    return {
        "output_dir": str(output),
        "n_generated_samples": len(source_samples),
        "annotations_path": str(output / "annotations.json"),
        "summary_path": str(output / "metadata" / "summary.json"),
    }
