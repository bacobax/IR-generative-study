"""RegionDiff synthetic generation production orchestration."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.algorithms.training.yolo_experiment_b import load_full_train_samples
from src.core.paths import repo_root

from . import audit_filtering, generation_backends
from .dataset_io import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_YOLO_DATASET_YAML,
    _load_yaml,
    _repo_path,
    _write_json,
    export_generated_candidate_dataset,
)

GENERATOR_BACKENDS = generation_backends.GENERATOR_BACKENDS
STREAMING_GENERATOR_BACKENDS = generation_backends.STREAMING_GENERATOR_BACKENDS


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


def _resolve_generation_device(config: Mapping[str, Any], device: str | None = None) -> str:
    if device not in (None, ""):
        return str(device)
    config_device = config.get("device")
    if config_device not in (None, ""):
        value = str(config_device).strip()
        if value.lower() not in {"none", "null"}:
            return value
    return "cuda" if torch.cuda.is_available() else "cpu"

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
    active_device = _resolve_generation_device(active_config, device=device)
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
            metrics_summary = audit_filtering.compute_distribution_metrics(
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
                audit_summary, retry_summary = audit_filtering._retry_streamed_generation_with_filter(
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
                arrays, audit_summary, retry_summary = audit_filtering._retry_generation_with_filter(
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
        layout_overlay_paths = audit_filtering.render_layout_overlay_previews(
            dataset_dir=output_dir,
            max_images=max_layout_overlays,
            annotations_filename="annotations_unfiltered.json",
            output_dir_name="layout_overlays",
        )
        filtered_layout_overlay_paths = audit_filtering.render_layout_overlay_previews(
            dataset_dir=output_dir,
            max_images=max_layout_overlays,
            annotations_filename="annotations.json",
            output_dir_name="filtered_layout_overlays",
        )
        sanity_paths = audit_filtering.render_sanity_check_images(
            dataset_dir=output_dir,
            max_images=int(active_config.get("sanity", {}).get("max_images", 24)),
        )
        sanity_paths.extend(
            audit_filtering.render_filter_crop_contact_sheets(
                dataset_dir=output_dir,
                max_crops_per_sheet=int(active_config.get("sanity", {}).get("max_crops_per_sheet", 24)),
            )
        )
        metrics_config = dict(active_config)
        if skip_metrics:
            metrics_config["metrics"] = {**dict(metrics_config.get("metrics", {})), "enabled": False}
        metrics_summary = audit_filtering.compute_distribution_metrics(
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

__all__ = [name for name in globals() if not name.startswith("__")]
