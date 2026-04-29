"""Experiment B helpers for full-train YOLO synthetic augmentation."""

from __future__ import annotations

import json
import math
import random
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

from src.algorithms.inference.rare_layout_dataset_tools import (
    audit_generated_layout_dataset,
    export_audit_results,
    load_filter_from_run_or_checkpoint,
    load_sampler_from_pipeline,
    sample_layout_batch,
)
from src.algorithms.stable_diffusion.layout_data import build_layout_prompt
from src.core.configs.yolo_experiment_config import YOLOExperimentConfig
from src.core.data.layout_batching import collate_layout_batch
from src.core.normalization import RAW_UINT16_PERCENTILE, sd_output_to_npy
from src.core.paths import repo_root


@dataclass(frozen=True)
class YOLOBox:
    """One normalized YOLO-format box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @classmethod
    def from_line(cls, line: str) -> "YOLOBox":
        parts = str(line).strip().split()
        if len(parts) != 5:
            raise ValueError(f"Expected YOLO label line with 5 columns, got: {line!r}")
        return cls(
            class_id=int(float(parts[0])),
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )

    def to_line(self) -> str:
        return (
            f"{int(self.class_id)} {self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )

    def xyxy_abs(self, *, image_w: int, image_h: int) -> tuple[float, float, float, float]:
        w = max(0.0, float(self.width)) * float(image_w)
        h = max(0.0, float(self.height)) * float(image_h)
        cx = float(self.x_center) * float(image_w)
        cy = float(self.y_center) * float(image_h)
        x1 = min(max(cx - 0.5 * w, 0.0), float(image_w))
        y1 = min(max(cy - 0.5 * h, 0.0), float(image_h))
        x2 = min(max(cx + 0.5 * w, 0.0), float(image_w))
        y2 = min(max(cy + 0.5 * h, 0.0), float(image_h))
        return x1, y1, x2, y2

    def coco_bbox_abs(self, *, image_w: int, image_h: int) -> list[float]:
        x1, y1, x2, y2 = self.xyxy_abs(image_w=image_w, image_h=image_h)
        return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


@dataclass(frozen=True)
class YOLOTrainSample:
    """One source full-train sample and its layout labels."""

    index: int
    image_path: Path
    label_path: Path
    boxes: list[YOLOBox]

    @property
    def stem(self) -> str:
        return self.image_path.stem


def _repo_path(path_like: str | Path | None) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like)
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_yolo_yaml(dataset_yaml: str | Path) -> dict[str, Any]:
    path = _repo_path(dataset_yaml)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {dataset_yaml}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "train" not in payload or "names" not in payload:
        raise ValueError(f"YOLO dataset YAML is missing required keys train/names: {path}")
    payload["_yaml_path"] = str(path)
    return payload


def _resolve_dataset_entry(dataset_payload: dict[str, Any], key: str) -> Path:
    value = dataset_payload.get(key)
    if value is None:
        raise ValueError(f"YOLO dataset YAML is missing {key!r}.")
    path = Path(str(value))
    if not path.is_absolute():
        base = Path(str(dataset_payload.get("path", Path(dataset_payload["_yaml_path"]).parent)))
        path = base / path
    return path.resolve()


def _labels_dir_for_images_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("images")
        parts[index] = "labels"
        return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def _normalise_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(k): str(v) for k, v in raw_names.items()}
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    raise TypeError("YOLO dataset YAML names must be a mapping or list.")


def load_full_train_samples(dataset_yaml: str | Path) -> tuple[list[YOLOTrainSample], dict[str, Any]]:
    """Load source full-train images and labels from a YOLO dataset YAML."""

    payload = _load_yolo_yaml(dataset_yaml)
    train_images_dir = _resolve_dataset_entry(payload, "train")
    train_labels_dir = _labels_dir_for_images_dir(train_images_dir)
    if not train_images_dir.is_dir():
        raise FileNotFoundError(f"YOLO full-train images directory not found: {train_images_dir}")
    if not train_labels_dir.is_dir():
        raise FileNotFoundError(f"YOLO full-train labels directory not found: {train_labels_dir}")

    image_paths = sorted(
        path
        for path in train_images_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise ValueError(f"No training images found under {train_images_dir}")

    samples: list[YOLOTrainSample] = []
    for index, image_path in enumerate(image_paths):
        label_path = train_labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing YOLO label file for {image_path}: {label_path}")
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        samples.append(
            YOLOTrainSample(
                index=index,
                image_path=image_path,
                label_path=label_path,
                boxes=[YOLOBox.from_line(line) for line in lines],
            )
        )
    return samples, payload


def validate_experiment_b_config(cfg: YOLOExperimentConfig) -> None:
    """Fail loudly on invalid Experiment B combinations."""

    mode = str(cfg.experiment_b.mode)
    if mode not in {"plain", "fm_aug", "sd_aug", "precomputed_aug"}:
        raise ValueError("experiment_b.mode must be one of: plain, fm_aug, sd_aug, precomputed_aug.")
    threshold = float(cfg.experiment_b.invalid_instance_ratio_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("experiment_b.invalid_instance_ratio_threshold must be in [0, 1].")

    dataset_yaml = _repo_path(cfg.data.dataset_yaml)
    full_train_yaml = _repo_path(cfg.data.full_train_dataset_yaml)
    if dataset_yaml != full_train_yaml:
        raise ValueError(
            "Experiment B is full-train only: data.dataset_yaml must match "
            "data.full_train_dataset_yaml."
        )

    has_fm = bool(cfg.experiment_b.fm.checkpoint_path)
    has_sd_stage1 = bool(cfg.experiment_b.sd.stage1_dir)
    has_sd_lora = bool(cfg.experiment_b.sd.lora_dir)
    has_precomputed = bool(cfg.experiment_b.precomputed_dataset_dir)
    if mode == "plain" and (has_fm or has_sd_stage1 or has_sd_lora or has_precomputed):
        raise ValueError("experiment_b.mode=plain must not configure generation sources.")
    if mode == "fm_aug" and not has_fm:
        raise ValueError("experiment_b.mode=fm_aug requires experiment_b.fm.checkpoint_path.")
    if mode == "sd_aug" and int(has_sd_stage1) + int(has_sd_lora) != 1:
        raise ValueError("experiment_b.mode=sd_aug requires exactly one of experiment_b.sd.stage1_dir or sd.lora_dir.")
    if mode == "precomputed_aug" and not has_precomputed:
        raise ValueError("experiment_b.mode=precomputed_aug requires experiment_b.precomputed_dataset_dir.")
    if mode != "precomputed_aug" and has_precomputed:
        raise ValueError("precomputed_dataset_dir is only valid when experiment_b.mode=precomputed_aug.")
    if mode != "sd_aug" and (has_sd_stage1 or has_sd_lora):
        raise ValueError("SD source paths are only valid when experiment_b.mode=sd_aug.")
    if mode != "fm_aug" and has_fm:
        raise ValueError("FM checkpoint paths are only valid when experiment_b.mode=fm_aug.")
    if int(cfg.experiment_b.filter.batch_size) <= 0:
        raise ValueError("experiment_b.filter.batch_size must be positive.")
    if int(cfg.experiment_b.fm.batch_size) <= 0:
        raise ValueError("experiment_b.fm.batch_size must be positive.")
    if int(cfg.experiment_b.fm.steps) <= 0:
        raise ValueError("experiment_b.fm.steps must be positive.")
    if int(cfg.experiment_b.sd.sd_steps) <= 0:
        raise ValueError("experiment_b.sd.sd_steps must be positive.")
    if int(cfg.experiment_b.sd.max_tries) <= 0:
        raise ValueError("experiment_b.sd.max_tries must be positive.")
    if str(cfg.experiment_b.sd.precision) not in {"fp16", "bf16", "fp32"}:
        raise ValueError("experiment_b.sd.precision must be one of: fp16, bf16, fp32.")
    if str(cfg.experiment_b.sd.prompt_mode) not in {"constant", "class_list"}:
        raise ValueError("experiment_b.sd.prompt_mode must be one of: constant, class_list.")


def _array_to_png_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3:
        arr = arr.astype(np.float32).mean(axis=-1)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.dtype == np.uint16 or float(np.nanmax(arr)) > 255.0:
        low = float(np.nanpercentile(arr, 1.0))
        high = float(np.nanpercentile(arr, 99.0))
    else:
        low = float(np.nanmin(arr))
        high = float(np.nanmax(arr))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _save_png_from_array(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_array_to_png_uint8(array), mode="L").convert("RGB").save(path, format="PNG")


def _image_size_from_array(array: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return int(arr.shape[1]), int(arr.shape[0])
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        return int(arr.shape[2]), int(arr.shape[1])
    if arr.ndim == 3:
        return int(arr.shape[1]), int(arr.shape[0])
    raise ValueError(f"Unsupported generated image shape: {tuple(arr.shape)}")


def _size_bin_thresholds_from_samples(samples: Sequence[YOLOTrainSample]) -> list[float]:
    areas = [float(box.width * box.height) for sample in samples for box in sample.boxes]
    if not areas:
        return [0.002, 0.01]
    quantiles = np.quantile(np.asarray(areas, dtype=np.float32), [1.0 / 3.0, 2.0 / 3.0])
    return [float(value) for value in quantiles.tolist()]


def _write_generated_coco(
    *,
    output_dir: Path,
    samples: Sequence[YOLOTrainSample],
    generated_arrays: Sequence[np.ndarray],
    dataset_payload: dict[str, Any],
    cfg: YOLOExperimentConfig,
    generator_kind: str,
) -> None:
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    names = _normalise_names(dataset_payload["names"])
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, (sample, generated_array) in enumerate(zip(samples, generated_arrays), start=1):
        file_name = f"sample_{image_id:06d}.npy"
        np.save(images_dir / file_name, generated_array)
        image_w, image_h = _image_size_from_array(generated_array)
        coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": image_w,
                "height": image_h,
            }
        )
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
                "source_index": sample.index,
                "source_image_path": str(sample.image_path),
                "source_label_path": str(sample.label_path),
                "n_objects": len(sample.boxes),
                "labels": [box.to_line() for box in sample.boxes],
            }
        )

    annotations = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": class_id, "name": name} for class_id, name in sorted(names.items())],
    }
    _write_json(output_dir / "annotations.json", annotations)
    with (metadata_dir / "provenance.jsonl").open("w", encoding="utf-8") as handle:
        for row in provenance_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_json(
        metadata_dir / "summary.json",
        {
            "dataset_id": "yolo_full_train",
            "split": "train",
            "selection_mode": "full_train_1_to_1",
            "rarity_aggregation": "none",
            "n_requested_samples": len(samples),
            "n_generated_samples": len(generated_arrays),
            "n_annotations": len(coco_annotations),
            "size_bin_thresholds": _size_bin_thresholds_from_samples(samples),
            "generator_kind": generator_kind,
            "source_dataset_yaml": str(_repo_path(cfg.data.dataset_yaml)),
            "samples": provenance_rows,
        },
    )


def _make_layout_batch_samples(
    samples: Sequence[YOLOTrainSample],
    *,
    image_size: int,
    names: dict[int, str],
) -> list[dict[str, Any]]:
    batch_samples: list[dict[str, Any]] = []
    for sample in samples:
        boxes = [
            box.xyxy_abs(image_w=int(image_size), image_h=int(image_size))
            for box in sample.boxes
        ]
        labels = [box.class_id for box in sample.boxes]
        batch_samples.append(
            {
                "pixel_values": torch.zeros(3, int(image_size), int(image_size), dtype=torch.float32),
                "boxes_xyxy": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.long),
                "n_objects": len(labels),
                "file_name": sample.image_path.name,
                "image_id": sample.index,
                "label_names": [names.get(int(label), str(int(label))) for label in labels],
            }
        )
    return batch_samples


def generate_fm_candidates(
    *,
    cfg: YOLOExperimentConfig,
    samples: Sequence[YOLOTrainSample],
    dataset_payload: dict[str, Any],
    output_dir: Path,
    device: str,
) -> Path:
    """Generate one layout-conditioned FM candidate for every source image."""

    checkpoint_path = _repo_path(cfg.experiment_b.fm.checkpoint_path)
    preset_path = _repo_path(cfg.experiment_b.fm.preset_path)
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing FM checkpoint: {cfg.experiment_b.fm.checkpoint_path}")
    if preset_path is None or not preset_path.is_file():
        raise FileNotFoundError(f"Missing FM preset: {cfg.experiment_b.fm.preset_path}")
    pipeline_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "UNET" else checkpoint_path.parent
    _preset, _layout_meta, _unet, _vae, sampler = load_sampler_from_pipeline(
        pipeline_dir,
        preset_path,
        checkpoint_path.name,
        device,
    )

    rng_seed = int(cfg.training.seed)
    names = _normalise_names(dataset_payload["names"])
    layout_samples = _make_layout_batch_samples(samples, image_size=cfg.data.image_size, names=names)
    generated_arrays: list[np.ndarray] = []
    batch_size = max(1, int(cfg.experiment_b.fm.batch_size))
    for start_idx in range(0, len(layout_samples), batch_size):
        chunk = layout_samples[start_idx:start_idx + batch_size]
        batch = collate_layout_batch(chunk)
        generated = sample_layout_batch(
            sampler,
            batch,
            steps=int(cfg.experiment_b.fm.steps),
            seed=rng_seed + int(start_idx),
        )
        generated_arrays.extend(
            image.detach().cpu().to(torch.float32).numpy().astype(np.float32, copy=False)
            for image in generated
        )
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if len(generated_arrays) != len(samples):
        raise RuntimeError(f"FM generated {len(generated_arrays)} images for {len(samples)} source images.")
    _write_generated_coco(
        output_dir=output_dir,
        samples=samples,
        generated_arrays=generated_arrays,
        dataset_payload=dataset_payload,
        cfg=cfg,
        generator_kind="fm_aug",
    )
    return output_dir


def _load_sd_pipeline(cfg: YOLOExperimentConfig, *, device: str):
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    weight_dtype = dtype_map[str(cfg.experiment_b.sd.precision)]
    normalization_mode = RAW_UINT16_PERCENTILE
    artifact_prompt_text = "thermal image"
    source_dir = cfg.experiment_b.sd.stage1_dir or cfg.experiment_b.sd.lora_dir
    source_path = _repo_path(source_dir)
    if source_path is None or not source_path.exists():
        raise FileNotFoundError(f"Missing SD source directory: {source_dir}")

    if (source_path / "stage1_manifest.json").is_file():
        from src.algorithms.stable_diffusion.models import load_stage1_pipeline

        pipe, manifest = load_stage1_pipeline(
            stage1_dir=str(source_path),
            base_model=cfg.experiment_b.sd.base_model,
            torch_dtype=weight_dtype,
        )
        normalization_mode = str(manifest.get("normalization_mode", RAW_UINT16_PERCENTILE))
        artifact_prompt_text = str(manifest.get("prompt_text") or artifact_prompt_text)
    else:
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.experiment_b.sd.base_model,
            torch_dtype=weight_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.load_lora_weights(str(source_path))

    pipe.to(device)
    return pipe, weight_dtype, normalization_mode, artifact_prompt_text


def generate_sd_candidates(
    *,
    cfg: YOLOExperimentConfig,
    samples: Sequence[YOLOTrainSample],
    dataset_payload: dict[str, Any],
    output_dir: Path,
    device: str,
) -> Path:
    """Generate one SD stage-1 candidate for every source image."""

    pipe, weight_dtype, normalization_mode, artifact_prompt_text = _load_sd_pipeline(cfg, device=device)
    names = _normalise_names(dataset_payload["names"])
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=weight_dtype)
        if str(device).startswith("cuda") and weight_dtype != torch.float32
        else nullcontext()
    )
    generated_arrays: list[np.ndarray] = []
    for sample in samples:
        label_names = [names.get(box.class_id, str(box.class_id)) for box in sample.boxes]
        prompt = build_layout_prompt(
            label_names=label_names,
            prompt_mode=str(cfg.experiment_b.sd.prompt_mode),
            constant_prompt=artifact_prompt_text,
            thermal_scene_suffix="in thermal scene.",
            caption=None,
            use_captions_if_available=False,
        )
        base_seed = int(cfg.training.seed) + int(sample.index)
        image = None
        last_flagged = False
        for attempt in range(max(1, int(cfg.experiment_b.sd.max_tries))):
            seed = base_seed + attempt
            generator = torch.Generator(device=device).manual_seed(seed)
            with autocast_ctx:
                result = pipe(
                    prompt,
                    negative_prompt=cfg.experiment_b.sd.negative_prompt,
                    num_inference_steps=int(cfg.experiment_b.sd.sd_steps),
                    guidance_scale=float(cfg.experiment_b.sd.guidance),
                    generator=generator,
                )
            image = result.images[0]
            nsfw = getattr(result, "nsfw_content_detected", None)
            if nsfw is None:
                last_flagged = False
                break
            last_flagged = any(bool(item) for item in nsfw) if isinstance(nsfw, list) else bool(nsfw)
            if not last_flagged:
                break
        if image is None:
            raise RuntimeError(f"SD generation produced no image for {sample.image_path}")
        generated_arrays.append(sd_output_to_npy(image, normalization_mode=normalization_mode))

    if len(generated_arrays) != len(samples):
        raise RuntimeError(f"SD generated {len(generated_arrays)} images for {len(samples)} source images.")
    _write_generated_coco(
        output_dir=output_dir,
        samples=samples,
        generated_arrays=generated_arrays,
        dataset_payload=dataset_payload,
        cfg=cfg,
        generator_kind="sd_aug",
    )
    return output_dir


def _resolve_filter_source(cfg: YOLOExperimentConfig) -> tuple[Optional[Path], Optional[Path]]:
    if cfg.experiment_b.filter.checkpoint_path:
        checkpoint_path = _repo_path(cfg.experiment_b.filter.checkpoint_path)
        return None, checkpoint_path
    checkpoint_dir = _repo_path(cfg.experiment_b.filter.checkpoint_dir)
    if checkpoint_dir is None:
        raise ValueError("experiment_b.filter.checkpoint_dir or checkpoint_path must be set.")
    if checkpoint_dir.name == "checkpoints":
        run_dir = checkpoint_dir.parent
        if (run_dir / "metrics" / "summary.json").is_file():
            return run_dir, None
        return None, checkpoint_dir / "best.pt"
    if (checkpoint_dir / "checkpoints" / "best.pt").is_file():
        return checkpoint_dir, None
    return None, checkpoint_dir / "best.pt"


def audit_generated_candidates(
    *,
    cfg: YOLOExperimentConfig,
    generated_dataset_dir: Path,
    device: str,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    """Audit generated candidates and return kept/discarded image rows."""

    run_dir, checkpoint_path = _resolve_filter_source(cfg)
    model, summary, threshold, input_size, context_ratio, resolved_run_dir = load_filter_from_run_or_checkpoint(
        device=device,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        threshold_override=None,
    )
    if str(summary.get("classifier_mode", "")) != "multiclass":
        raise ValueError(
            "Experiment B requires the multiclass foreground/background filter. "
            "An instance is valid only when the classifier predicts the expected class "
            "and passes that class threshold."
        )
    instance_rows, image_rows, stats = audit_generated_layout_dataset(
        dataset_dir=generated_dataset_dir,
        filter_model=model,
        threshold=threshold,
        filter_input_size=input_size,
        context_ratio=context_ratio,
        device=device,
        crop_batch_size=int(cfg.experiment_b.filter.batch_size),
        show_progress=True,
        classifier_summary=summary,
    )
    audit_dir = generated_dataset_dir / "filter_audit"
    export_audit_results(
        output_dir=audit_dir,
        summary={
            "generated_dataset_dir": str(generated_dataset_dir),
            "filter_run_dir": None if resolved_run_dir is None else str(resolved_run_dir),
            "checkpoint_summary": summary,
            "threshold": float(threshold) if isinstance(threshold, (int, float)) else dict(threshold),
            "input_size": int(input_size),
            "context_ratio": float(context_ratio),
            "max_discarded_valid_threshold": 1.0,
            "score_alpha": 1.0,
            "score_beta": 1.0,
            "stats": stats,
            "min_valid_object_fraction_sweep": "",
        },
        instance_rows=instance_rows,
        image_rows=image_rows,
    )
    annotation_filter_summary = _write_filtered_annotations_from_audit(
        generated_dataset_dir=generated_dataset_dir,
        instance_rows=instance_rows,
    )
    classified_rows = classify_generated_image_rows(
        image_rows,
        invalid_instance_ratio_threshold=float(cfg.experiment_b.invalid_instance_ratio_threshold),
    )
    discard_summary = build_instance_discard_summary(
        instance_rows=instance_rows,
        classified_image_rows=classified_rows,
        invalid_instance_ratio_threshold=float(cfg.experiment_b.invalid_instance_ratio_threshold),
    )
    _write_json(audit_dir / "experiment_b_discard_summary.json", discard_summary)
    print(
        "[Experiment B] filter annotation summary: "
        f"{annotation_filter_summary['n_invalid_annotations_removed']}/"
        f"{annotation_filter_summary['n_annotations_unfiltered']} annotations removed; "
        "generated images are retained."
    )
    return classified_rows, audit_dir, discard_summary


def _write_filtered_annotations_from_audit(
    *,
    generated_dataset_dir: Path,
    instance_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Remove invalid instance annotations while retaining every generated image."""

    annotations_path = generated_dataset_dir / "annotations.json"
    unfiltered_path = generated_dataset_dir / "annotations_unfiltered.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Generated dataset is missing annotations.json: {generated_dataset_dir}")
    if not unfiltered_path.is_file():
        shutil.copy2(annotations_path, unfiltered_path)
    payload = _load_json(unfiltered_path)
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
    summary = {
        "n_images": len(filtered_payload["images"]),
        "n_annotations_unfiltered": len(payload.get("annotations", [])),
        "n_annotations": len(filtered_annotations),
        "n_invalid_annotations_removed": len(payload.get("annotations", [])) - len(filtered_annotations),
    }
    _write_json(generated_dataset_dir / "metadata" / "filtered_annotation_summary.json", summary)
    metadata_path = generated_dataset_dir / "metadata" / "summary.json"
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        metadata.update(summary)
        metadata["annotations_path"] = "annotations.json"
        metadata["unfiltered_annotations_path"] = "annotations_unfiltered.json"
        _write_json(metadata_path, metadata)
    return summary


def classify_generated_image_rows(
    image_rows: Sequence[dict[str, Any]],
    *,
    invalid_instance_ratio_threshold: float,
) -> list[dict[str, Any]]:
    """Apply Experiment B's invalid-instance-ratio discard rule."""

    threshold = float(invalid_instance_ratio_threshold)
    classified: list[dict[str, Any]] = []
    for row in image_rows:
        n_instances = int(row.get("n_instances", 0) or 0)
        n_invalid = int(row.get("n_negative_instances", 0) or 0)
        invalid_ratio = float(n_invalid / max(1, n_instances))
        enriched = dict(row)
        enriched["invalid_instance_ratio"] = invalid_ratio
        enriched["discarded_by_invalid_instance_ratio"] = bool(invalid_ratio > threshold)
        classified.append(enriched)
    return classified


def load_precomputed_generated_image_rows(generated_dataset_dir: Path) -> list[dict[str, Any]]:
    """Load precomputed generated image rows in Experiment-B candidate format."""

    annotations_path = generated_dataset_dir / "annotations.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Precomputed generated dataset is missing annotations.json: {generated_dataset_dir}")
    payload = _load_json(annotations_path)
    images = payload.get("images", [])
    if not isinstance(images, list) or not images:
        raise ValueError(f"Precomputed generated dataset has no images: {annotations_path}")

    rows: list[dict[str, Any]] = []
    anns_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for ann in payload.get("annotations", []):
        anns_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    for image in images:
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        image_path = generated_dataset_dir / "images" / file_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Generated image listed in annotations.json is missing: {image_path}")
        n_instances = len(anns_by_image_id.get(image_id, []))
        rows.append(
            {
                "generated_image_id": image_id,
                "generated_file_name": file_name,
                "n_instances": n_instances,
                "n_negative_instances": 0,
                "invalid_instance_ratio": 0.0,
                "discarded_by_invalid_instance_ratio": False,
            }
        )
    rows.sort(key=lambda row: int(row["generated_image_id"]))
    return rows


def _empty_discard_bucket(group_value: str) -> dict[str, Any]:
    return {
        "group": str(group_value),
        "total_instance_count": 0,
        "discarded_instance_count": 0,
        "instance_discard_ratio": 0.0,
        "valid_instance_count": 0,
        "invalid_instance_count": 0,
        "discarded_valid_instance_count": 0,
        "discarded_invalid_instance_count": 0,
        "valid_instance_discard_ratio": 0.0,
        "invalid_instance_discard_ratio": 0.0,
    }


def _finalize_discard_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    total = int(bucket["total_instance_count"])
    valid = int(bucket["valid_instance_count"])
    invalid = int(bucket["invalid_instance_count"])
    bucket["instance_discard_ratio"] = float(bucket["discarded_instance_count"] / max(1, total))
    bucket["valid_instance_discard_ratio"] = float(bucket["discarded_valid_instance_count"] / max(1, valid)) if valid else 0.0
    bucket["invalid_instance_discard_ratio"] = float(bucket["discarded_invalid_instance_count"] / max(1, invalid)) if invalid else 0.0
    return bucket


def _sorted_discard_buckets(buckets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_finalize_discard_bucket(bucket) for bucket in buckets.values()]
    rows.sort(
        key=lambda row: (
            -float(row["instance_discard_ratio"]),
            -int(row["discarded_instance_count"]),
            str(row["group"]),
        )
    )
    return rows


def build_instance_discard_summary(
    *,
    instance_rows: Sequence[dict[str, Any]],
    classified_image_rows: Sequence[dict[str, Any]],
    invalid_instance_ratio_threshold: float,
) -> dict[str, Any]:
    """Summarize instances discarded by the image-level invalid-ratio threshold."""

    discarded_image_ids = {
        int(row["generated_image_id"])
        for row in classified_image_rows
        if bool(row.get("discarded_by_invalid_instance_ratio", False))
    }
    global_bucket = _empty_discard_bucket("global")
    by_category: dict[str, dict[str, Any]] = {}
    by_size: dict[str, dict[str, Any]] = {}
    by_category_size: dict[str, dict[str, Any]] = {}

    def update(bucket: dict[str, Any], *, is_valid: bool, is_discarded: bool) -> None:
        bucket["total_instance_count"] += 1
        bucket["valid_instance_count" if is_valid else "invalid_instance_count"] += 1
        if is_discarded:
            bucket["discarded_instance_count"] += 1
            bucket["discarded_valid_instance_count" if is_valid else "discarded_invalid_instance_count"] += 1

    for row in instance_rows:
        category = str(row.get("category_name", row.get("expected_category_name", row.get("category_id", "unknown"))))
        size_bin = str(row.get("size_bin", "unknown"))
        category_size = f"{category} | {size_bin}"
        is_valid = bool(row.get("is_positive", False))
        is_discarded = int(row["generated_image_id"]) in discarded_image_ids

        update(global_bucket, is_valid=is_valid, is_discarded=is_discarded)
        update(by_category.setdefault(category, _empty_discard_bucket(category)), is_valid=is_valid, is_discarded=is_discarded)
        update(by_size.setdefault(size_bin, _empty_discard_bucket(size_bin)), is_valid=is_valid, is_discarded=is_discarded)
        update(
            by_category_size.setdefault(category_size, _empty_discard_bucket(category_size)),
            is_valid=is_valid,
            is_discarded=is_discarded,
        )

    global_row = _finalize_discard_bucket(global_bucket)
    return {
        "invalid_instance_ratio_threshold": float(invalid_instance_ratio_threshold),
        "discarded_image_count": int(len(discarded_image_ids)),
        "total_image_count": int(len(classified_image_rows)),
        "image_discard_ratio": float(len(discarded_image_ids) / max(1, len(classified_image_rows))),
        "global": global_row,
        "by_category": _sorted_discard_buckets(by_category),
        "by_size": _sorted_discard_buckets(by_size),
        "by_category_size": _sorted_discard_buckets(by_category_size),
    }


def _source_sample_by_generated_id(samples: Sequence[YOLOTrainSample]) -> dict[int, YOLOTrainSample]:
    return {idx: sample for idx, sample in enumerate(samples, start=1)}


def _source_sample_by_generated_rows(
    samples: Sequence[YOLOTrainSample],
    generated_dataset_dir: Path,
) -> dict[int, YOLOTrainSample]:
    """Resolve source samples for generated rows, using provenance when present."""

    mapping = _source_sample_by_generated_id(samples)
    provenance_path = generated_dataset_dir / "metadata" / "provenance.jsonl"
    if not provenance_path.is_file():
        return mapping

    by_source_index = {int(sample.index): sample for sample in samples}
    with provenance_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            generated_id = int(row.get("generated_image_id", 0))
            source_index = row.get("source_index")
            if generated_id <= 0 or source_index is None:
                continue
            sample = by_source_index.get(int(source_index))
            if sample is not None:
                mapping[generated_id] = sample
    return mapping


def _load_filtered_generated_coco(generated_dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    annotations_path = generated_dataset_dir / "annotations.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(
            "Experiment B precomputed augmentation requires filtered annotations.json "
            f"under {generated_dataset_dir}"
        )
    payload = _load_json(annotations_path)
    images = list(payload.get("images", []))
    anns_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for ann in payload.get("annotations", []):
        anns_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)
    images.sort(key=lambda row: int(row["id"]))
    return images, anns_by_image_id


def _annotation_to_yolo_line(ann: Mapping[str, Any], *, image_w: float, image_h: float) -> str:
    x, y, w, h = [float(value) for value in ann["bbox"]]
    x_center = (x + 0.5 * w) / max(1.0, image_w)
    y_center = (y + 0.5 * h) / max(1.0, image_h)
    width = w / max(1.0, image_w)
    height = h / max(1.0, image_h)
    return (
        f"{int(ann['category_id'])} "
        f"{min(max(x_center, 0.0), 1.0):.8f} "
        f"{min(max(y_center, 0.0), 1.0):.8f} "
        f"{min(max(width, 0.0), 1.0):.8f} "
        f"{min(max(height, 0.0), 1.0):.8f}"
    )


def export_augmented_yolo_dataset(
    *,
    cfg: YOLOExperimentConfig,
    source_samples: Sequence[YOLOTrainSample],
    dataset_payload: dict[str, Any],
    generated_dataset_dir: Path,
    classified_rows: Sequence[dict[str, Any]],
    discard_summary: Optional[dict[str, Any]] = None,
) -> Path:
    """Export a YOLO dataset with all real samples plus all generated images."""

    output_root = _repo_path(cfg.experiment_b.augmented_yolo_root)
    if output_root is None:
        raise ValueError("experiment_b.augmented_yolo_root must be set.")
    output_root = output_root / cfg.output.experiment_name / str(cfg.experiment_b.mode)
    if output_root.exists():
        shutil.rmtree(output_root)
    train_images_dir = output_root / "images" / "train"
    train_labels_dir = output_root / "labels" / "train"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    for sample in source_samples:
        image_dst = train_images_dir / f"real_{sample.image_path.name}"
        label_dst = train_labels_dir / f"real_{sample.stem}.txt"
        shutil.copy2(sample.image_path, image_dst)
        shutil.copy2(sample.label_path, label_dst)

    source_by_generated_id = _source_sample_by_generated_rows(source_samples, generated_dataset_dir)
    generated_images, anns_by_image_id = _load_filtered_generated_coco(generated_dataset_dir)
    for image_info in generated_images:
        generated_id = int(image_info["id"])
        sample = source_by_generated_id.get(generated_id)
        generated_path = generated_dataset_dir / "images" / str(image_info["file_name"])
        generated_array = np.load(generated_path)
        synthetic_stem = (
            f"synthetic_{sample.index:06d}_{sample.stem}"
            if sample is not None
            else f"synthetic_generated_{generated_id:06d}"
        )
        _save_png_from_array(generated_array, train_images_dir / f"{synthetic_stem}.png")
        image_w = float(image_info.get("width", _image_size_from_array(generated_array)[0]))
        image_h = float(image_info.get("height", _image_size_from_array(generated_array)[1]))
        label_lines = [
            _annotation_to_yolo_line(ann, image_w=image_w, image_h=image_h)
            for ann in anns_by_image_id.get(generated_id, [])
        ]
        (train_labels_dir / f"{synthetic_stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )

    names = _normalise_names(dataset_payload["names"])
    yaml_path = output_root / "full_train_synthetic_aug.yaml"
    yaml_payload = {
        "path": str(output_root.resolve()),
        "train": str(train_images_dir.resolve()),
        "val": str(_resolve_dataset_entry(dataset_payload, "val")),
        "test": str(_resolve_dataset_entry(dataset_payload, "test")),
        "names": {idx: names[idx] for idx in sorted(names)},
        "nc": int(dataset_payload.get("nc", len(names))),
    }
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")

    flagged_rows = [row for row in classified_rows if bool(row.get("discarded_by_invalid_instance_ratio", False))]
    _write_json(
        output_root / "experiment_b_augmented_dataset_summary.json",
        {
            "mode": cfg.experiment_b.mode,
            "source_dataset_yaml": str(_repo_path(cfg.data.dataset_yaml)),
            "augmented_dataset_yaml": str(yaml_path.resolve()),
            "generated_dataset_dir": str(generated_dataset_dir.resolve()),
            "invalid_instance_ratio_threshold": float(cfg.experiment_b.invalid_instance_ratio_threshold),
            "n_real_images": len(source_samples),
            "n_generated_images": len(generated_images),
            "n_kept_synthetic_images": len(generated_images),
            "n_discarded_synthetic_images": 0,
            "n_flagged_by_invalid_instance_ratio": len(flagged_rows),
            "kept_generated_image_ids": [int(row["id"]) for row in generated_images],
            "discarded_generated_image_ids": [],
            "flagged_generated_image_ids": [int(row["generated_image_id"]) for row in flagged_rows],
            "discard_summary": discard_summary,
        },
    )
    return yaml_path


def prepare_experiment_b_dataset(cfg: YOLOExperimentConfig, *, device: str) -> dict[str, Any]:
    """Generate, filter, and export the Experiment B augmented YOLO dataset."""

    validate_experiment_b_config(cfg)
    if str(cfg.experiment_b.mode) == "plain":
        return {"mode": "plain", "dataset_yaml": str(_repo_path(cfg.data.dataset_yaml))}

    random.seed(int(cfg.training.seed))
    np.random.seed(int(cfg.training.seed))
    torch.manual_seed(int(cfg.training.seed))
    source_samples, dataset_payload = load_full_train_samples(cfg.data.dataset_yaml)

    mode = str(cfg.experiment_b.mode)
    if mode == "precomputed_aug":
        generated_dataset_dir = _repo_path(cfg.experiment_b.precomputed_dataset_dir)
        if generated_dataset_dir is None or not generated_dataset_dir.is_dir():
            raise FileNotFoundError(f"Precomputed generated dataset not found: {cfg.experiment_b.precomputed_dataset_dir}")
    else:
        generated_root = _repo_path(cfg.experiment_b.generated_dataset_dir)
        if generated_root is None:
            raise ValueError("experiment_b.generated_dataset_dir must be set.")
        generated_dataset_dir = generated_root / cfg.output.experiment_name / mode
        if generated_dataset_dir.exists():
            shutil.rmtree(generated_dataset_dir)
        generated_dataset_dir.mkdir(parents=True, exist_ok=True)

    if mode == "fm_aug":
        generate_fm_candidates(
            cfg=cfg,
            samples=source_samples,
            dataset_payload=dataset_payload,
            output_dir=generated_dataset_dir,
            device=device,
        )
    elif mode == "sd_aug":
        generate_sd_candidates(
            cfg=cfg,
            samples=source_samples,
            dataset_payload=dataset_payload,
            output_dir=generated_dataset_dir,
            device=device,
        )
    elif mode == "precomputed_aug":
        pass
    else:
        raise ValueError(f"Unsupported Experiment B mode: {cfg.experiment_b.mode}")

    if bool(cfg.experiment_b.filter.enabled):
        classified_rows, audit_dir, discard_summary = audit_generated_candidates(
            cfg=cfg,
            generated_dataset_dir=generated_dataset_dir,
            device=device,
        )
    else:
        classified_rows = load_precomputed_generated_image_rows(generated_dataset_dir)
        audit_dir = generated_dataset_dir / "filter_audit_disabled"
        audit_dir.mkdir(parents=True, exist_ok=True)
        discard_summary = {
            "filter_enabled": False,
            "discarded_image_count": 0,
            "total_image_count": len(classified_rows),
            "image_discard_ratio": 0.0,
        }
        _write_json(audit_dir / "experiment_b_discard_summary.json", discard_summary)
    augmented_yaml = export_augmented_yolo_dataset(
        cfg=cfg,
        source_samples=source_samples,
        dataset_payload=dataset_payload,
        generated_dataset_dir=generated_dataset_dir,
        classified_rows=classified_rows,
        discard_summary=discard_summary,
    )
    return {
        "mode": cfg.experiment_b.mode,
        "source_dataset_yaml": str(_repo_path(cfg.data.dataset_yaml)),
        "generated_dataset_dir": str(generated_dataset_dir.resolve()),
        "filter_audit_dir": str(audit_dir.resolve()),
        "discard_summary_path": str((audit_dir / "experiment_b_discard_summary.json").resolve()),
        "discard_summary": discard_summary,
        "augmented_dataset_yaml": str(augmented_yaml.resolve()),
        "n_source_images": len(source_samples),
        "n_generated_images": len(classified_rows),
        "n_kept_synthetic_images": len(classified_rows),
        "n_flagged_by_invalid_instance_ratio": sum(
            int(bool(row.get("discarded_by_invalid_instance_ratio", False)))
            for row in classified_rows
        ),
    }
