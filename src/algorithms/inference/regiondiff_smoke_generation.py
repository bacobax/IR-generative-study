"""Offline mini generation for smoked RegionDiff artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler
from src.algorithms.inference.unconditional_sd_sampler import UnconditionalStableDiffusionSampler
from src.algorithms.stable_diffusion.layout_data import build_layout_prompt
from src.algorithms.stable_diffusion.layout_models import load_stage2_layout_pipeline
from src.algorithms.training.yolo_experiment_b import YOLOTrainSample, load_full_train_samples
from src.core.data.layout_batching import collate_layout_batch
from src.core.normalization import RAW_UINT16_PERCENTILE, sd_output_to_npy


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
        collate_layout_batch(layout_samples[start:start + max(1, int(batch_size))])
        for start in range(0, len(layout_samples), max(1, int(batch_size)))
    ]


def export_generated_candidate_dataset(
    *,
    output_dir: str | Path,
    source_samples: Sequence[YOLOTrainSample],
    generated_arrays: Sequence[np.ndarray],
    dataset_payload: dict[str, Any],
    generator_kind: str,
) -> Path:
    """Write generated arrays in the candidate format consumed by Experiment B."""

    output = Path(output_dir)
    images_dir = output / "images"
    metadata_dir = output / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    names = _normalise_names(dataset_payload["names"])
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, (sample, generated_array) in enumerate(zip(source_samples, generated_arrays), start=1):
        file_name = f"sample_{image_id:06d}.npy"
        np.save(images_dir / file_name, np.asarray(generated_array))
        image_w, image_h = _image_size_from_array(np.asarray(generated_array))
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

    (output / "annotations.json").write_text(
        json.dumps(
            {
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [{"id": class_id, "name": name} for class_id, name in sorted(names.items())],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with (metadata_dir / "provenance.jsonl").open("w", encoding="utf-8") as handle:
        for row in provenance_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (metadata_dir / "summary.json").write_text(
        json.dumps(
            {
                "generator_kind": generator_kind,
                "n_generated_samples": len(generated_arrays),
                "n_annotations": len(coco_annotations),
                "source_dataset_yaml": str(dataset_payload.get("_yaml_path", "")),
                "samples": provenance_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output


def _pipeline_config(pipeline_dir: str | Path, *, device: str, t_scale: float, train_target: str):
    return SimpleNamespace(
        pipeline_dir=str(pipeline_dir),
        vae_weights=None,
        t_scale=float(t_scale),
        train_target=str(train_target),
        sample_shape=None,
        device=device,
        resolved_device=lambda: device,
    )


def generate_fm_regiondiff_arrays(
    *,
    pipeline_dir: str | Path,
    batches: Sequence[dict[str, Any]],
    device: str,
    steps: int,
    seed: int,
    t_scale: float,
    train_target: str,
) -> list[np.ndarray]:
    sampler = FlowMatchingSampler.from_config(
        _pipeline_config(pipeline_dir, device=device, t_scale=t_scale, train_target=train_target)
    )
    arrays: list[np.ndarray] = []
    for batch_idx, batch in enumerate(tqdm(batches, desc="FM RegionDiff smoke generation")):
        z = sampler.sample_euler_layout(batch, steps=steps, seed=seed + batch_idx)
        decoded = sampler.decode(z)
        arrays.extend(image.detach().cpu().to(torch.float32).numpy() for image in decoded)
    return arrays


def generate_dm_regiondiff_arrays(
    *,
    pipeline_dir: str | Path,
    batches: Sequence[dict[str, Any]],
    device: str,
    steps: int,
    seed: int,
) -> list[np.ndarray]:
    cfg = SimpleNamespace(
        output=SimpleNamespace(model_dir=str(pipeline_dir)),
        sampling=SimpleNamespace(sample_shape=None),
        resolved_device=lambda: device,
    )
    sampler = UnconditionalStableDiffusionSampler.from_config(cfg)
    arrays: list[np.ndarray] = []
    for batch_idx, batch in enumerate(tqdm(batches, desc="DM RegionDiff smoke generation")):
        z = sampler.sample_layout(batch, steps=steps, seed=seed + batch_idx)
        decoded = sampler.decode(z)
        arrays.extend(image.detach().cpu().to(torch.float32).numpy() for image in decoded)
    return arrays


def generate_sd15_regiondiff_arrays(
    *,
    stage2_dir: str | Path,
    samples: Sequence[YOLOTrainSample],
    names: dict[int, str],
    device: str,
    steps: int,
    seed: int,
    guidance_scale: float,
    precision: str,
) -> list[np.ndarray]:
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    weight_dtype = dtype_map[str(precision)]
    pipe, manifest = load_stage2_layout_pipeline(stage2_dir=str(stage2_dir), torch_dtype=weight_dtype)
    pipe.to(device)
    normalization_mode = str(manifest.get("normalization_mode", RAW_UINT16_PERCENTILE))
    arrays: list[np.ndarray] = []
    for idx, sample in enumerate(tqdm(samples, desc="SD1.5 RegionDiff smoke generation")):
        batch = collate_layout_batch([_sample_to_layout_dict(sample, image_size=512, names=names)])
        label_names = [names.get(box.class_id, str(box.class_id)) for box in sample.boxes]
        prompt = build_layout_prompt(
            label_names=label_names,
            prompt_mode="class_list",
            constant_prompt="thermal image",
            thermal_scene_suffix="in thermal scene.",
            caption=None,
            use_captions_if_available=False,
        )
        generator = torch.Generator(device=device).manual_seed(int(seed) + idx)
        result = pipe(
            prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            cross_attention_kwargs={
                "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(device=device),
                "labels": batch["labels"].to(device=device),
                "object_mask": batch["object_mask"].to(device=device),
            },
        )
        image = result.images[0]
        if isinstance(image, Image.Image):
            arrays.append(sd_output_to_npy(image, normalization_mode=normalization_mode))
        else:
            arrays.append(np.asarray(image))
    return arrays


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
    """Generate a tiny RegionDiff candidate dataset from a smoke artifact."""

    source_samples, dataset_payload = load_full_train_samples(yolo_dataset_yaml)
    source_samples = source_samples[: max(0, int(max_samples))]
    if not source_samples:
        raise ValueError("No source samples selected for RegionDiff smoke generation.")
    names = _normalise_names(dataset_payload["names"])
    batches = build_layout_batches(
        source_samples,
        image_size=int(image_size),
        names=names,
        batch_size=max(1, int(batch_size)),
    )

    kind = str(model_kind)
    if kind == "fm":
        arrays = generate_fm_regiondiff_arrays(
            pipeline_dir=artifact_dir,
            batches=batches,
            device=device,
            steps=int(steps),
            seed=int(seed),
            t_scale=float(t_scale),
            train_target=str(train_target),
        )
    elif kind == "dm":
        arrays = generate_dm_regiondiff_arrays(
            pipeline_dir=artifact_dir,
            batches=batches,
            device=device,
            steps=int(steps),
            seed=int(seed),
        )
    elif kind in {"sd15_finetune", "sd15_lora"}:
        arrays = generate_sd15_regiondiff_arrays(
            stage2_dir=artifact_dir,
            samples=source_samples,
            names=names,
            device=device,
            steps=int(steps),
            seed=int(seed),
            guidance_scale=float(guidance_scale),
            precision=str(precision),
        )
    else:
        raise ValueError("model_kind must be one of: fm, dm, sd15_finetune, sd15_lora.")

    output = export_generated_candidate_dataset(
        output_dir=output_dir,
        source_samples=source_samples,
        generated_arrays=arrays[: len(source_samples)],
        dataset_payload=dataset_payload,
        generator_kind=f"regiondiff_{kind}",
    )
    return {
        "model_kind": kind,
        "artifact_dir": str(Path(artifact_dir)),
        "output_dir": str(output),
        "n_generated_samples": len(source_samples),
        "annotations_path": str(output / "annotations.json"),
        "summary_path": str(output / "metadata" / "summary.json"),
    }
