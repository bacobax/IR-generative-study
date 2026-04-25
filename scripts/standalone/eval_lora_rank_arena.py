#!/usr/bin/env python3
"""Evaluate Stable Diffusion LoRA ranks with FID, KID, and MMD."""

from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from src.core.configs.config_loader import load_yaml
from src.core.data.dataset_targets import resolve_dataset_target
from src.core.normalization import UINT8_LINEAR
from src.core.paths import repo_root
from src.evaluation.feature_extractors import build_feature_extractor, extract_features
from src.evaluation.generative_metrics import (
    compute_fid,
    compute_kid,
    metrics_row_to_jsonable,
    sort_metric_rows,
)
from src.evaluation.mmd import compute_rbf_mmd


SUPPORTED_FEATURE_EXTRACTORS = {"dinov2", "inception"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SD LoRA ranks before downstream experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to the LoRA rank arena YAML config.")
    parser.add_argument("--force-generate", action="store_true", help="Regenerate samples even when cached images exist.")
    parser.add_argument("--force-features", action="store_true", help="Re-extract image features even when caches exist.")
    parser.add_argument("--force-metrics", action="store_true", help="Recompute metrics even when metrics.json exists.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override generation.n_samples.")
    parser.add_argument("--feature-extractor", type=str, default=None, choices=sorted(SUPPORTED_FEATURE_EXTRACTORS))
    parser.add_argument("--reference-split", type=str, default=None, help="Override data.reference_split.")
    return parser.parse_args(argv)


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_resolved_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_yaml(args.config)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a mapping in config file {args.config!r}.")

    overrides: Dict[str, Any] = {}
    if args.n_samples is not None:
        overrides.setdefault("generation", {})["n_samples"] = int(args.n_samples)
    if args.feature_extractor is not None:
        overrides.setdefault("metrics", {})["feature_extractor"] = str(args.feature_extractor)
    if args.reference_split is not None:
        overrides.setdefault("data", {})["reference_split"] = str(args.reference_split)
    if overrides:
        cfg = _deep_merge(dict(cfg), overrides)
    return cfg


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path


def get_device(config: Mapping[str, Any]) -> str:
    requested = config.get("experiment", {}).get("device")
    if requested:
        return str(requested)
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_weight_dtype(config: Mapping[str, Any], device: str) -> torch.dtype:
    precision = str(config.get("experiment", {}).get("mixed_precision", "auto")).lower()
    if precision == "auto":
        return torch.float16 if str(device).startswith("cuda") else torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision in {"fp32", "no", "none"}:
        return torch.float32
    raise ValueError(f"Unsupported mixed_precision={precision!r}.")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_reference_images(
    *,
    dataset_id: str,
    reference_split: str,
    max_real_images: int | None = None,
) -> tuple[List[Path], str, Path]:
    """Resolve a repo-native split and return deterministic real image paths."""
    target = resolve_dataset_target(dataset_id)
    split_dir = target.split_dir(reference_split)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    paths = sorted(split_dir.glob("*.npy"))
    images_dir = split_dir / "images"
    if not paths and images_dir.is_dir():
        paths = sorted(images_dir.glob("*.npy"))
    if max_real_images is not None:
        paths = paths[: int(max_real_images)]
    if not paths:
        raise ValueError(f"No real .npy images found in {split_dir}.")
    return paths, target.normalization_mode, split_dir


def validate_lora_ranks(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ranks = list(config.get("lora_arena", {}).get("ranks", []))
    if not ranks:
        raise ValueError("lora_arena.ranks must contain at least one rank entry.")

    labels = set()
    validated: List[Dict[str, Any]] = []
    for entry in ranks:
        rank = int(entry["rank"])
        label = str(entry.get("label") or f"lora_r{rank}")
        if label in labels:
            raise ValueError(f"Duplicate LoRA rank label: {label!r}.")
        labels.add(label)

        checkpoint_path = resolve_repo_path(str(entry["checkpoint_path"]))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing LoRA checkpoint for {label}: {checkpoint_path}")
        if not _is_diffusers_lora_checkpoint(checkpoint_path):
            raise FileNotFoundError(
                "LoRA checkpoint must be a Diffusers LoRA directory containing "
                f"pytorch_lora_weights.safetensors/.bin or a .safetensors file: {checkpoint_path}"
            )
        validated.append(
            {
                "rank": rank,
                "label": label,
                "checkpoint_path": checkpoint_path,
                "checkpoint_path_config": str(entry["checkpoint_path"]),
            }
        )
    return validated


def _is_diffusers_lora_checkpoint(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in {".safetensors", ".bin"}
    return (
        (path / "pytorch_lora_weights.safetensors").is_file()
        or (path / "pytorch_lora_weights.bin").is_file()
    )


def expected_generated_paths(output_dir: Path, n_samples: int) -> List[Path]:
    return [output_dir / f"sample_{idx:06d}.png" for idx in range(int(n_samples))]


def validate_generated_image_set(output_dir: Path, n_samples: int) -> List[Path]:
    expected = expected_generated_paths(output_dir, n_samples)
    expected_names = {path.name for path in expected}
    actual_names = {path.name for path in output_dir.glob("sample_*.png")} if output_dir.is_dir() else set()
    extra = sorted(actual_names - expected_names)
    if extra:
        raise RuntimeError(f"Found unexpected generated images in {output_dir}: {extra[:5]}")
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Generated image count is incomplete in {output_dir}: missing {len(missing)} files.")
    return expected


def _build_pipeline(config: Mapping[str, Any], rank_entry: Mapping[str, Any], *, device: str, dtype: torch.dtype):
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    model_cfg = config.get("model", {})
    base_model = str(model_cfg.get("base_model_name_or_path", "runwayml/stable-diffusion-v1-5"))
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    scheduler_name = str(model_cfg.get("scheduler", "ddim")).lower()
    if scheduler_name != "ddim":
        raise ValueError(f"Unsupported scheduler={scheduler_name!r}; only 'ddim' is implemented.")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    prediction_type = model_cfg.get("prediction_type")
    if prediction_type is not None and hasattr(pipe.scheduler, "register_to_config"):
        pipe.scheduler.register_to_config(prediction_type=str(prediction_type))

    pipe.load_lora_weights(str(rank_entry["checkpoint_path"]))
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_samples_for_rank(
    *,
    config: Mapping[str, Any],
    rank_entry: Mapping[str, Any],
    output_dir: Path,
    device: str,
    dtype: torch.dtype,
    force: bool,
) -> List[Path]:
    """Generate exactly N prompt-only SD samples for one LoRA rank."""
    generation_cfg = config.get("generation", {})
    n_samples = int(generation_cfg.get("n_samples", 1000))
    if not bool(generation_cfg.get("save_images", True)):
        raise ValueError("generation.save_images must be true because metrics are computed from saved images.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output_dir.glob("sample_*.png"):
            path.unlink()
        metadata_path = output_dir / "metadata.jsonl"
        if metadata_path.exists():
            metadata_path.unlink()

    expected = expected_generated_paths(output_dir, n_samples)
    resume_existing = bool(generation_cfg.get("resume_existing_images", True))
    missing = [path for path in expected if not path.is_file()]
    if resume_existing and not missing:
        return validate_generated_image_set(output_dir, n_samples)
    if not resume_existing:
        missing = expected

    # Rank selection requires every rank to use the same sample count,
    # resolution, preprocessing, real split, and feature extractor.
    pipe = _build_pipeline(config, rank_entry, device=device, dtype=dtype)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if str(device).startswith("cuda") and dtype != torch.float32
        else nullcontext()
    )

    prompt = str(generation_cfg.get("prompt", "thermal image"))
    negative_prompt = str(generation_cfg.get("negative_prompt", ""))
    batch_size = int(generation_cfg.get("batch_size", 8))
    height = int(generation_cfg.get("height", generation_cfg.get("image_size", 512)))
    width = int(generation_cfg.get("width", generation_cfg.get("image_size", 512)))
    steps = int(generation_cfg.get("num_inference_steps", 40))
    guidance = float(generation_cfg.get("guidance_scale", 1.0))
    seed = int(config.get("experiment", {}).get("seed", 42))

    missing_indices = [int(path.stem.split("_")[-1]) for path in missing]
    for start in tqdm(range(0, len(missing_indices), batch_size), desc=f"Generating {rank_entry['label']}"):
        batch_indices = missing_indices[start : start + batch_size]
        generators = [
            torch.Generator(device=device).manual_seed(seed + idx)
            for idx in batch_indices
        ]
        with autocast_ctx:
            result = pipe(
                [prompt] * len(batch_indices),
                negative_prompt=[negative_prompt] * len(batch_indices),
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                generator=generators,
            )

        for idx, image in zip(batch_indices, result.images):
            path = output_dir / f"sample_{idx:06d}.png"
            image.convert("RGB").save(path)

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for idx in range(n_samples):
            record = {
                "file_name": f"sample_{idx:06d}.png",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed + idx,
                "rank": int(rank_entry["rank"]),
                "label": str(rank_entry["label"]),
                "checkpoint_path": str(rank_entry["checkpoint_path"]),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    del pipe
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return validate_generated_image_set(output_dir, n_samples)


def rank_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    ranking_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    keys = [
        str(ranking_cfg.get("primary", "kid")),
        str(ranking_cfg.get("secondary", "fid")),
        str(ranking_cfg.get("tertiary", "mmd")),
    ]
    return sort_metric_rows(rows, ranking_keys=keys)


def compute_metric_rows(
    *,
    config: Mapping[str, Any],
    ranks: Sequence[Mapping[str, Any]],
    real_features: np.ndarray,
    generated_features_by_label: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    metrics_cfg = config.get("metrics", {})
    seed = int(config.get("experiment", {}).get("seed", 42))
    rows: List[Dict[str, Any]] = []
    min_feature_count = min(
        [real_features.shape[0]]
        + [features.shape[0] for features in generated_features_by_label.values()]
    )
    requested_kid_subset = int(metrics_cfg.get("kid", {}).get("subset_size", 1000))
    if requested_kid_subset > min_feature_count:
        warnings.warn(
            "KID subset_size exceeds available features; reducing to "
            f"{min_feature_count}.",
            RuntimeWarning,
        )

    for rank_entry in ranks:
        label = str(rank_entry["label"])
        gen_features = generated_features_by_label[label]
        row: Dict[str, Any] = {
            "label": label,
            "rank": int(rank_entry["rank"]),
            "checkpoint_path": str(rank_entry["checkpoint_path"]),
            "num_real": int(real_features.shape[0]),
            "num_generated": int(gen_features.shape[0]),
        }
        if bool(metrics_cfg.get("compute_fid", True)):
            row["fid"] = compute_fid(real_features, gen_features)
        if bool(metrics_cfg.get("compute_kid", True)):
            kid_cfg = metrics_cfg.get("kid", {})
            row["kid"] = compute_kid(
                real_features,
                gen_features,
                subsets=int(kid_cfg.get("subsets", 100)),
                subset_size=min(int(kid_cfg.get("subset_size", 1000)), min_feature_count),
                seed=seed,
            )
        if bool(metrics_cfg.get("compute_mmd", True)):
            mmd_cfg = metrics_cfg.get("mmd", {})
            row["mmd"] = compute_rbf_mmd(
                real_features,
                gen_features,
                bandwidths=mmd_cfg.get("bandwidths", [0.1, 1.0, 10.0]),
            )
        rows.append(metrics_row_to_jsonable(row))
    return rows


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "rank", "checkpoint_path", "num_real", "num_generated", "kid", "fid", "mmd"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def save_outputs(
    *,
    output_root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    ranked_rows: Sequence[Mapping[str, Any]],
) -> None:
    selected = dict(ranked_rows[0]) if ranked_rows else {}
    metrics_payload = {
        "experiment": config.get("experiment", {}),
        "feature_extractor": config.get("metrics", {}).get("feature_extractor"),
        "metrics": list(rows),
    }
    ranking_payload = {
        "ranking": list(ranked_rows),
        "selected_top1": selected,
        "ranking_keys": {
            "primary": config.get("ranking", {}).get("primary", "kid"),
            "secondary": config.get("ranking", {}).get("secondary", "fid"),
            "tertiary": config.get("ranking", {}).get("tertiary", "mmd"),
        },
    }
    (output_root / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "ranking.json").write_text(
        json.dumps(ranking_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary_csv(output_root / "summary.csv", ranked_rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_resolved_config(args)
    output_root = resolve_repo_path(config.get("output", {}).get("root_dir", "./artifacts/evaluations/lora_rank_arena"))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    feature_name = str(config.get("metrics", {}).get("feature_extractor", "dinov2")).lower()
    if feature_name not in SUPPORTED_FEATURE_EXTRACTORS:
        raise ValueError(f"Unsupported feature extractor: {feature_name!r}.")
    # Do not mix feature extractors across one arena run; ranking is only
    # meaningful when every rank is embedded by the same model.

    seed = int(config.get("experiment", {}).get("seed", 42))
    set_reproducibility(seed)
    device = get_device(config)
    dtype = get_weight_dtype(config, device)
    ranks = validate_lora_ranks(config)

    data_cfg = config.get("data", {})
    reference_split = str(data_cfg.get("reference_split", "val"))
    # This pipeline is for LoRA rank selection only, before RegionDiff.
    # Prefer val/eval over test for selection; every metric below uses this
    # same real reference split.
    real_paths, real_normalization, split_dir = discover_reference_images(
        dataset_id=str(data_cfg.get("dataset_id", "flir_private_proxy_alignment_v18")),
        reference_split=reference_split,
        max_real_images=data_cfg.get("max_real_images"),
    )

    generated_root = output_root / "generated"
    generated_paths_by_label: Dict[str, List[Path]] = {}
    for rank_entry in ranks:
        label = str(rank_entry["label"])
        generated_paths_by_label[label] = generate_samples_for_rank(
            config=config,
            rank_entry=rank_entry,
            output_dir=generated_root / label,
            device=device,
            dtype=dtype,
            force=bool(args.force_generate),
        )

    features_root = output_root / "features"
    extractor = build_feature_extractor(feature_name, config.get("metrics", {}), device)
    real_features = extract_features(
        real_paths,
        extractor,
        batch_size=int(config.get("generation", {}).get("batch_size", 8)),
        cache_path=features_root / f"real_{reference_split}_{feature_name}.npz",
        force=bool(args.force_features),
        normalization_mode=real_normalization,
        metadata={
            "split": reference_split,
            "split_dir": str(split_dir),
            "image_size": int(data_cfg.get("image_size", 512)),
        },
    )

    generated_features_by_label: Dict[str, np.ndarray] = {}
    for rank_entry in ranks:
        label = str(rank_entry["label"])
        generated_features_by_label[label] = extract_features(
            generated_paths_by_label[label],
            extractor,
            batch_size=int(config.get("generation", {}).get("batch_size", 8)),
            cache_path=features_root / f"generated_{label}_{feature_name}.npz",
            force=bool(args.force_features),
            normalization_mode=UINT8_LINEAR,
            metadata={
                "label": label,
                "rank": int(rank_entry["rank"]),
                "image_size": int(data_cfg.get("image_size", 512)),
            },
        )

    metrics_path = output_root / "metrics.json"
    if metrics_path.is_file() and not args.force_metrics:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows = list(payload.get("metrics", []))
    else:
        rows = compute_metric_rows(
            config=config,
            ranks=ranks,
            real_features=real_features,
            generated_features_by_label=generated_features_by_label,
        )

    ranked_rows = rank_metric_rows(rows, ranking_cfg=config.get("ranking", {}))
    save_outputs(output_root=output_root, config=config, rows=rows, ranked_rows=ranked_rows)

    selected = ranked_rows[0]
    print("\nLoRA Rank Arena completed.\n")
    print(f"Feature extractor: {feature_name}")
    print(f"Reference split: {reference_split}")
    print(f"Samples per rank: {int(config.get('generation', {}).get('n_samples', 1000))}")
    print("\nRanking:")
    for idx, row in enumerate(ranked_rows, start=1):
        print(
            f"{idx}. {row['label']} | "
            f"KID={float(row.get('kid', float('nan'))):.6g} | "
            f"FID={float(row.get('fid', float('nan'))):.6g} | "
            f"MMD={float(row.get('mmd', float('nan'))):.6g}"
        )
    print("\nSelected top-1:")
    print(f"rank={selected['rank']}")
    print(f"checkpoint={selected['checkpoint_path']}")


if __name__ == "__main__":
    main()
