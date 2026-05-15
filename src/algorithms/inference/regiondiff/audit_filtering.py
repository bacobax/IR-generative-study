"""RegionDiff synthetic generation audit, filtering, retry, and metric helpers."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.algorithms.inference.rare_layout_dataset_tools import (
    audit_generated_layout_dataset,
    export_audit_results,
    load_filter_from_run_or_checkpoint,
)
from src.algorithms.training.yolo_experiment_b import YOLOTrainSample
from src.evaluation.feature_extractors import build_feature_extractor, extract_features
from src.evaluation.generative_metrics import compute_fid, compute_kid
from src.evaluation.mmd import compute_rbf_mmd

from .dataset_io import (
    DEFAULT_FILTER_RUN_DIR,
    _array_to_png_uint8,
    _category_color,
    _write_csv,
    _write_json,
    _write_jsonl,
    export_generated_candidate_dataset,
)


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

__all__ = [name for name in globals() if not name.startswith("__")]
