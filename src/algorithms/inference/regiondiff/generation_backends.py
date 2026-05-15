"""RegionDiff synthetic generation backend generation functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from src.algorithms.inference.rare_layout_dataset_tools import sample_layout_batch
from src.algorithms.training.yolo_experiment_b import YOLOTrainSample
from src.core.data.layout_batching import collate_layout_batch

from .backend_loaders import (
    _build_regiondiff_sd_layout_prompts,
    _load_regiondiff_sampler,
    _load_regiondiff_sd_layout_pipeline,
    _load_regiondiff_sd_sampler,
    _load_stay_sampler,
    _pipeline_images_to_arrays,
    _remap_layout_batch_labels,
)
from .dataset_io import (
    _generated_sample_exists,
    _normalise_names,
    _sample_to_layout_dict,
    _save_generated_array,
    build_layout_batches,
    initialize_generated_candidate_dataset,
)


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

__all__ = [name for name in globals() if not name.startswith("__")]
