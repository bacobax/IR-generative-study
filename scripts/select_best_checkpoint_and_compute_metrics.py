#!/usr/bin/env python3
"""Standalone post-training checkpoint selection and generative metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.standalone.generate_checkpoint_quality_comparison import (  # noqa: E402
    _build_fm_sampler,
    _build_sd_sampler,
    _dataset_for_conditional,
    _normalization_mode_from_preset,
    detect_run_kind,
    resolve_run_dirs,
    tensor_to_output_array,
)
from src.algorithms.inference.layout_flow_matching_sampler import LayoutFlowMatchingSampler  # noqa: E402
from src.algorithms.inference.rare_layout_dataset_tools import sample_layout_batch  # noqa: E402
from src.algorithms.stable_diffusion.models import (  # noqa: E402
    LORA_WEIGHT_FILENAMES,
    load_lora_weights_compat,
    load_stage1_manifest,
    load_stage1_pipeline,
)
from src.core.artifacts import read_artifact_manifest  # noqa: E402
from src.core.configs.config_loader import load_yaml  # noqa: E402
from src.core.data.dataset_targets import resolve_dataset_target  # noqa: E402
from src.core.normalization import UINT8_LINEAR, raw_array_to_png_uint8, sd_output_to_npy  # noqa: E402
from src.evaluation.feature_extractors import build_feature_extractor, extract_features  # noqa: E402
from src.evaluation.generative_metrics import compute_fid, compute_kid  # noqa: E402
from src.evaluation.intra_lpips import compute_intra_lpips  # noqa: E402
from src.evaluation.mmd import compute_rbf_mmd  # noqa: E402


NATIVE_EPOCH_RE = re.compile(r"^(?P<stem>unet_(?:fm|sd_uncond))_epoch_(?P<epoch>\d+)(?:_ckpt)?\.pt$")
DIFFUSERS_STEP_RE = re.compile(r"^checkpoint-(?P<step>\d+)$")


@dataclass(frozen=True)
class CheckpointCandidate:
    checkpoint_identifier: str
    checkpoint_path: str
    checkpoint_kind: str
    epoch: int | None = None
    step: int | None = None
    source: str = ""


@dataclass(frozen=True)
class ExcludedCheckpoint:
    path: str
    reason: str


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: list[CheckpointCandidate]
    excluded: list[ExcludedCheckpoint]


@dataclass(frozen=True)
class RunResolution:
    run_identifier: str
    run_dir: Path
    model_type: str
    sampler_name: str | None
    sampling_config_path: Path | None
    preset: dict[str, Any]
    generation_backend_used: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Checkpoint-selection YAML config.")
    return parser.parse_args(argv)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (CheckpointCandidate, ExcludedCheckpoint, DiscoveryResult)):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def sanitize_identifier(value: str, *, field_name: str) -> str:
    identifier = str(value).strip().replace(" ", "_")
    if not identifier:
        raise ValueError(f"{field_name} must be non-empty.")
    if "/" in identifier or "\\" in identifier:
        raise ValueError(f"{field_name} must not contain path separators: {value!r}")
    return identifier


def resolve_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def expected_generated_paths(images_dir: Path, n_images: int) -> list[Path]:
    return [images_dir / f"sample_{idx:06d}.npy" for idx in range(int(n_images))]


def validate_or_prepare_generation_dir(
    images_dir: Path,
    *,
    n_images: int,
    overwrite: bool,
) -> tuple[list[int], bool]:
    if overwrite and images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    expected = expected_generated_paths(images_dir, n_images)
    expected_names = {path.name for path in expected}
    actual_names = {path.name for path in images_dir.glob("sample_*.npy")}
    extra = sorted(actual_names - expected_names)
    if extra:
        raise RuntimeError(f"Unexpected generated files in {images_dir}: {extra[:5]}")
    missing = [idx for idx, path in enumerate(expected) if not path.is_file()]
    return missing, not missing


def make_stage_seeds(config: Mapping[str, Any]) -> dict[str, list[int]]:
    base = int(config.get("generation_seed", 1234))
    stage1_n = int(config.get("stage1_num_images", 200))
    stage2_n = int(config.get("stage2_extra_images", 800))
    stage3_n = int(config.get("stage3_extra_images", 1000))
    seeds = {
        "stage1": [base + int(config.get("stage1_seed_offset", 0)) + idx for idx in range(stage1_n)],
        "stage2": [base + int(config.get("stage2_seed_offset", 100000)) + idx for idx in range(stage2_n)],
        "stage3": [base + int(config.get("stage3_seed_offset", 200000)) + idx for idx in range(stage3_n)],
    }
    flattened = [seed for values in seeds.values() for seed in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Stage generation seeds overlap; adjust stage seed offsets.")
    return seeds


def _has_lora_weights(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in {".safetensors", ".bin"}
    return any((path / filename).is_file() for filename in LORA_WEIGHT_FILENAMES)


def _has_sd_stage1_final_export(path: Path) -> bool:
    return _has_lora_weights(path) or (path / "unet").is_dir()


def _checkpoint_id_for_native_epoch(path: Path) -> tuple[str, int] | None:
    match = NATIVE_EPOCH_RE.match(path.name)
    if match is None or path.name.endswith("_ckpt.pt"):
        return None
    digits = match.group("epoch")
    return f"epoch_{digits}", int(digits)


def _latest_native_epoch(unet_dir: Path) -> tuple[Path, str, int] | None:
    rows: list[tuple[int, str, Path]] = []
    for path in unet_dir.glob("unet_*_epoch_*.pt"):
        parsed = _checkpoint_id_for_native_epoch(path)
        if parsed is None:
            continue
        identifier, epoch = parsed
        rows.append((epoch, identifier, path))
    if not rows:
        return None
    epoch, identifier, path = max(rows, key=lambda item: item[0])
    return path, identifier, epoch


def discover_candidate_checkpoints(
    run_dir: str | Path,
    *,
    model_type: str | None = None,
    checkpoint_min_epoch: int = 50,
    checkpoint_min_step: int | None = None,
) -> DiscoveryResult:
    run_path = Path(run_dir)
    candidates: list[CheckpointCandidate] = []
    excluded: list[ExcludedCheckpoint] = []
    seen_paths: set[Path] = set()

    def include(candidate: CheckpointCandidate) -> None:
        resolved = Path(candidate.checkpoint_path).resolve()
        if resolved in seen_paths:
            excluded.append(ExcludedCheckpoint(str(candidate.checkpoint_path), "duplicate checkpoint path"))
            return
        seen_paths.add(resolved)
        candidates.append(candidate)

    def exclude(path: Path, reason: str) -> None:
        excluded.append(ExcludedCheckpoint(str(path), reason))

    normalized_model_type = (model_type or infer_model_type(run_path, None)).lower()
    stage1_manifest = run_path / "stage1_manifest.json"
    if normalized_model_type in {"sd_lora", "sd_stage1", "stable_diffusion_lora"} or stage1_manifest.is_file():
        if stage1_manifest.is_file() and _has_sd_stage1_final_export(run_path):
            include(
                CheckpointCandidate(
                    checkpoint_identifier="final",
                    checkpoint_path=str(run_path),
                    checkpoint_kind="final",
                    source="sd_stage1_final_export",
                )
            )
        else:
            exclude(run_path, "missing final stage1_manifest.json or exported stage-1 weights")

        for path in sorted(run_path.iterdir() if run_path.is_dir() else []):
            match = DIFFUSERS_STEP_RE.match(path.name)
            if match is None:
                continue
            step = int(match.group("step"))
            if checkpoint_min_step is not None and step < int(checkpoint_min_step):
                exclude(path, f"step {step} < checkpoint_min_step {checkpoint_min_step}")
                continue
            if not _has_lora_weights(path):
                exclude(path, "missing Diffusers LoRA checkpoint weights")
                continue
            include(
                CheckpointCandidate(
                    checkpoint_identifier=f"step_{step:06d}",
                    checkpoint_path=str(path),
                    checkpoint_kind="step",
                    step=step,
                    source="diffusers_checkpoint_dir",
                )
            )
        if not candidates:
            raise FileNotFoundError(f"No valid SD stage-1 checkpoints found under {run_path}")
        return DiscoveryResult(candidates=candidates, excluded=excluded)

    unet_dir = run_path / "UNET" if run_path.name != "UNET" else run_path
    if not unet_dir.is_dir():
        raise FileNotFoundError(f"No UNET directory found for native checkpoint discovery: {unet_dir}")

    best_paths = [
        unet_dir / "unet_fm_best.pt",
        unet_dir / "unet_sd_uncond_best.pt",
        unet_dir / "best.pt",
    ]
    best_path = next((path for path in best_paths if path.is_file()), None)
    if best_path is None:
        exclude(unet_dir / "best.pt", "best checkpoint missing")
    else:
        include(
            CheckpointCandidate(
                checkpoint_identifier="best",
                checkpoint_path=str(best_path),
                checkpoint_kind="best",
                source="native_best",
            )
        )

    latest = _latest_native_epoch(unet_dir)
    if latest is None:
        exclude(unet_dir, "final/latest native epoch checkpoint missing")
    else:
        path, _epoch_identifier, epoch = latest
        include(
            CheckpointCandidate(
                checkpoint_identifier="final",
                checkpoint_path=str(path),
                checkpoint_kind="final",
                epoch=epoch,
                source="native_latest_epoch_as_final",
            )
        )

    for path in sorted(unet_dir.glob("unet_*_epoch_*.pt")):
        parsed = _checkpoint_id_for_native_epoch(path)
        if parsed is None:
            continue
        identifier, epoch = parsed
        if epoch < int(checkpoint_min_epoch):
            exclude(path, f"epoch {epoch} < checkpoint_min_epoch {checkpoint_min_epoch}")
            continue
        include(
            CheckpointCandidate(
                checkpoint_identifier=identifier,
                checkpoint_path=str(path),
                checkpoint_kind="epoch",
                epoch=epoch,
                source="native_epoch",
            )
        )

    if not candidates:
        raise FileNotFoundError(f"No valid native checkpoints found under {unet_dir}")
    return DiscoveryResult(candidates=candidates, excluded=excluded)


def find_sampling_config_for_run(run_dir: Path, model_type: str) -> Path | None:
    search_roots = []
    if model_type in {"flow_matching", "latent_flow_matching", "fm"}:
        search_roots.append(REPO_ROOT / "configs" / "fm" / "train")
    elif model_type in {"sd_uncond", "diffusion", "latent_diffusion"}:
        search_roots.append(REPO_ROOT / "configs" / "sd_uncond" / "train")
    elif model_type in {"sd_lora", "sd_stage1", "stable_diffusion_lora"}:
        search_roots.append(REPO_ROOT / "configs" / "sd" / "train")
    else:
        search_roots.extend([
            REPO_ROOT / "configs" / "fm" / "train",
            REPO_ROOT / "configs" / "sd_uncond" / "train",
            REPO_ROOT / "configs" / "sd" / "train",
        ])

    for root in search_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.yaml")):
            try:
                data = load_yaml(path)
            except Exception:
                continue
            output_dir = (
                data.get("output", {}).get("model_dir")
                if isinstance(data.get("output"), Mapping)
                else None
            ) or data.get("output_dir")
            if not output_dir:
                continue
            resolved = resolve_path(output_dir)
            if resolved is not None and resolved.resolve() == run_dir.resolve():
                return path
    return None


def infer_model_type(run_dir: Path, config_model_type: str | None) -> str:
    if config_model_type:
        return str(config_model_type)
    manifest = read_artifact_manifest(run_dir)
    if manifest is not None:
        if manifest.model_family == "flow_matching":
            return "latent_flow_matching"
        if manifest.model_family == "stable_diffusion":
            return "sd_uncond"
    stage1_manifest = run_dir / "stage1_manifest.json"
    if stage1_manifest.is_file():
        data = json.loads(stage1_manifest.read_text(encoding="utf-8"))
        if data.get("baseline_mode") == "sd_ir_lora":
            return "sd_lora"
        return "sd_stage1"
    if (run_dir / "SCHEDULER").is_dir():
        return "sd_uncond"
    if (run_dir / "UNET").is_dir():
        return "latent_flow_matching"
    raise ValueError(
        f"Could not infer model_type for {run_dir}. Set model_type in the run entry."
    )


def resolve_run(run_entry: Mapping[str, Any], config: Mapping[str, Any]) -> RunResolution:
    run_identifier = sanitize_identifier(str(run_entry["run_identifier"]), field_name="run_identifier")
    run_dir = resolve_path(str(run_entry["run_dir"]))
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist for {run_identifier}: {run_entry.get('run_dir')}")
    model_type = infer_model_type(run_dir, run_entry.get("model_type"))
    sampler_name = run_entry.get("sampler_name")
    sampling_config_path = resolve_path(run_entry.get("sampling_config_path") or config.get("sampling_config_path"))
    if sampling_config_path is None:
        sampling_config_path = find_sampling_config_for_run(run_dir, model_type)
    preset: dict[str, Any] = {}
    if sampling_config_path is not None:
        if not sampling_config_path.is_file():
            raise FileNotFoundError(f"sampling_config_path not found: {sampling_config_path}")
        preset = load_yaml(sampling_config_path)
    elif model_type not in {"sd_lora", "sd_stage1", "stable_diffusion_lora"}:
        manifest = read_artifact_manifest(run_dir)
        if manifest is not None:
            preset = {
                "training": {
                    "t_scale": manifest.task.get("t_scale", 1000.0),
                    "train_target": manifest.task.get("train_target", "v"),
                }
            }

    if not preset and model_type not in {"sd_lora", "sd_stage1", "stable_diffusion_lora"}:
        raise ValueError(
            f"Could not resolve sampling config for {run_identifier}. "
            "Set sampling_config_path or provide a run with artifact_manifest.json."
        )

    backend = {
        "latent_flow_matching": "native_flow_matching_sampler",
        "flow_matching": "native_flow_matching_sampler",
        "fm": "native_flow_matching_sampler",
        "sd_uncond": "native_unconditional_sd_sampler",
        "diffusion": "native_unconditional_sd_sampler",
        "latent_diffusion": "native_unconditional_sd_sampler",
        "sd_lora": "diffusers_stable_diffusion_lora",
        "sd_stage1": "diffusers_stable_diffusion_stage1",
        "stable_diffusion_lora": "diffusers_stable_diffusion_lora",
    }.get(str(model_type), "")
    if not backend:
        raise ValueError(f"Unsupported model_type={model_type!r} for run {run_identifier}.")
    return RunResolution(
        run_identifier=run_identifier,
        run_dir=run_dir,
        model_type=str(model_type),
        sampler_name=str(sampler_name) if sampler_name is not None else None,
        sampling_config_path=sampling_config_path,
        preset=preset,
        generation_backend_used=backend,
    )


def discover_reference_images(
    config: Mapping[str, Any],
    run: RunResolution,
    *,
    split_override: str | None = None,
) -> tuple[list[Path], str, Path]:
    real_reference_path = resolve_path(config.get("real_reference_path"))
    limit = config.get("real_reference_num_samples")
    if real_reference_path is not None:
        if not real_reference_path.exists():
            raise FileNotFoundError(f"real_reference_path not found: {real_reference_path}")
        if real_reference_path.is_dir():
            paths = sorted(real_reference_path.glob("*.npy"))
            if not paths and (real_reference_path / "images").is_dir():
                paths = sorted((real_reference_path / "images").glob("*.npy"))
        else:
            paths = [real_reference_path]
        if limit is not None:
            paths = paths[: int(limit)]
        if not paths:
            raise ValueError(f"No real reference .npy images found in {real_reference_path}")
        return paths, UINT8_LINEAR, real_reference_path

    dataset_id = config.get("dataset_id")
    if not dataset_id and run.preset:
        dataset_id = run.preset.get("data", {}).get("dataset_id") if isinstance(run.preset.get("data"), Mapping) else None
        dataset_id = dataset_id or run.preset.get("dataset_id")
    if not dataset_id and (run.run_dir / "stage1_manifest.json").is_file():
        dataset_id = load_stage1_manifest(str(run.run_dir)).get("dataset_id")
    if not dataset_id:
        raise ValueError(
            f"Could not infer real reference dataset for {run.run_identifier}. "
            "Set dataset_id or real_reference_path in the evaluation config."
        )
    split = str(split_override or config.get("real_reference_split", "val"))
    target = resolve_dataset_target(str(dataset_id))
    split_dir = target.split_dir(split)
    paths = sorted(split_dir.glob("*.npy"))
    if not paths and (split_dir / "images").is_dir():
        paths = sorted((split_dir / "images").glob("*.npy"))
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        raise ValueError(f"No real reference .npy images found in {split_dir}")
    return paths, target.normalization_mode, split_dir


def get_device(config: Mapping[str, Any]) -> str:
    requested = config.get("device")
    if requested:
        return str(requested)
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_weight_dtype(config: Mapping[str, Any], device: str) -> torch.dtype:
    precision = str(config.get("mixed_precision", "auto")).lower()
    if precision == "auto":
        return torch.float16 if str(device).startswith("cuda") else torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision in {"fp32", "no", "none"}:
        return torch.float32
    raise ValueError(f"Unsupported mixed_precision={precision!r}")


def build_sd_stage1_pipeline(run: RunResolution, checkpoint: CheckpointCandidate, *, config: Mapping[str, Any], device: str):
    dtype = get_weight_dtype(config, device)
    manifest = load_stage1_manifest(str(run.run_dir))
    base_model = config.get("base_model_name_or_path") or manifest.get("pretrained_model_name_or_path")
    if checkpoint.checkpoint_kind == "final":
        pipe, _manifest = load_stage1_pipeline(
            stage1_dir=str(run.run_dir),
            base_model=str(base_model) if base_model else None,
            torch_dtype=dtype,
        )
    else:
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            str(base_model),
            revision=manifest.get("revision"),
            variant=manifest.get("variant"),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        load_lora_weights_compat(pipe, checkpoint.checkpoint_path)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, manifest


def _save_preview(path: Path, arr: np.ndarray, *, normalization_mode: str) -> None:
    preview_path = path.with_suffix(".png")
    preview = raw_array_to_png_uint8(arr, normalization_mode=normalization_mode)
    Image.fromarray(preview, mode="L").save(preview_path)


def generate_sd_stage1_samples(
    *,
    run: RunResolution,
    checkpoint: CheckpointCandidate,
    images_dir: Path,
    seeds: Sequence[int],
    config: Mapping[str, Any],
    device: str,
) -> None:
    pipe, manifest = build_sd_stage1_pipeline(run, checkpoint, config=config, device=device)
    prompt = str(config.get("generation_prompt") or manifest.get("prompt_text") or "thermal image")
    negative_prompt = str(config.get("negative_prompt", ""))
    steps = int(config.get("num_inference_steps", config.get("sd_steps", 40)))
    guidance_scale = float(config.get("guidance_scale", 1.0))
    height = int(config.get("height", config.get("image_size", 512)))
    width = int(config.get("width", config.get("image_size", 512)))
    normalization_mode = str(manifest.get("normalization_mode", UINT8_LINEAR))

    for idx, seed in tqdm(list(enumerate(seeds)), desc=f"Generating {checkpoint.checkpoint_identifier}", unit="img"):
        output_path = images_dir / f"sample_{idx:06d}.npy"
        if output_path.is_file():
            continue
        generator = torch.Generator(device=device).manual_seed(int(seed))
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        arr = sd_output_to_npy(result.images[0], normalization_mode=normalization_mode)
        np.save(output_path, arr)
        _save_preview(output_path, arr, normalization_mode=normalization_mode)

    del pipe
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _sample_native_one(sampler, *, model_family: str, seed: int, steps: int, sample: Mapping[str, Any] | None = None):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available() and str(getattr(sampler, "device", "")).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    if sample is None:
        if model_family == "sd":
            latents = sampler.sample(steps=steps, batch_size=1)
        else:
            latents = sampler.sample_euler(steps=steps, batch_size=1)
        return sampler.decode(latents).detach().cpu()[0]

    from src.core.data.layout_batching import collate_layout_batch

    batch = collate_layout_batch([dict(sample)])
    if model_family == "sd":
        latents = sampler.sample_layout(batch, steps=steps, seed=int(seed))
        return sampler.decode(latents).detach().cpu()[0]
    if isinstance(sampler, LayoutFlowMatchingSampler):
        return sample_layout_batch(sampler, batch, steps=steps, seed=int(seed)).detach().cpu()[0]
    latents = sampler.sample_euler_layout(batch, steps=steps)
    return sampler.decode(latents).detach().cpu()[0]


def generate_native_samples(
    *,
    run: RunResolution,
    checkpoint: CheckpointCandidate,
    images_dir: Path,
    seeds: Sequence[int],
    config: Mapping[str, Any],
    device: str,
) -> None:
    run_dirs = resolve_run_dirs(run.run_dir)
    run_kind = detect_run_kind(run_dirs.pipeline_dir, run.preset, model_family="auto")
    steps = int(config.get("num_inference_steps", config.get("steps", 50)))
    split = str(config.get("layout_reference_split", config.get("real_reference_split", "val")))
    dataset_id = str(config.get("dataset_id") or run.preset.get("data", {}).get("dataset_id", ""))
    layout_samples: list[Mapping[str, Any]] = []
    if run_kind.layout_conditioned:
        dataset = _dataset_for_conditional(
            run.preset,
            split=split,
            dataset_root=None,
            dataset_id=dataset_id or None,
        )
        if len(dataset) < len(seeds):
            raise ValueError(
                f"Layout-conditioned generation requested {len(seeds)} samples, "
                f"but split {split!r} has only {len(dataset)} records."
            )
        layout_samples = [dataset[idx] for idx in range(len(seeds))]

    if run_kind.model_family == "sd":
        sampler = _build_sd_sampler(
            pipeline_dir=run_dirs.pipeline_dir,
            preset=run.preset,
            checkpoint_path=Path(checkpoint.checkpoint_path),
            device=device,
        )
    else:
        categories = {}
        if layout_samples:
            try:
                dataset = _dataset_for_conditional(run.preset, split=split, dataset_root=None, dataset_id=dataset_id or None)
                categories = dict(dataset.category_id_to_name)
            except Exception:
                categories = {}
        sampler = _build_fm_sampler(
            pipeline_dir=run_dirs.pipeline_dir,
            preset=run.preset,
            checkpoint_path=Path(checkpoint.checkpoint_path),
            device=device,
            layout_variant=run_kind.layout_variant,
            category_id_to_name=categories,
        )

    normalization_mode = _normalization_mode_from_preset(run.preset)
    for idx, seed in tqdm(list(enumerate(seeds)), desc=f"Generating {checkpoint.checkpoint_identifier}", unit="img"):
        output_path = images_dir / f"sample_{idx:06d}.npy"
        if output_path.is_file():
            continue
        sample = layout_samples[idx] if layout_samples else None
        image = _sample_native_one(
            sampler,
            model_family=run_kind.model_family,
            seed=int(seed),
            steps=steps,
            sample=sample,
        )
        arr = tensor_to_output_array(image, normalization_mode=normalization_mode)
        np.save(output_path, arr)

    del sampler
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def ensure_generated_stage(
    *,
    run: RunResolution,
    checkpoint: CheckpointCandidate,
    stage_dir: Path,
    seeds: Sequence[int],
    config: Mapping[str, Any],
    device: str,
) -> tuple[Path, bool]:
    images_dir = stage_dir / "generated_npy_images"
    missing, complete = validate_or_prepare_generation_dir(
        images_dir,
        n_images=len(seeds),
        overwrite=bool(config.get("overwrite_existing_generations", False)),
    )
    if complete:
        metadata = {
            "cached": True,
            "num_generated_images": len(seeds),
            "generation_seed_list": list(map(int, seeds)),
            "timestamp": utc_timestamp(),
        }
        save_json(stage_dir / "generation_metadata.json", metadata)
        return images_dir, True

    active_seeds = list(seeds)
    if run.generation_backend_used.startswith("diffusers_stable_diffusion"):
        generate_sd_stage1_samples(
            run=run,
            checkpoint=checkpoint,
            images_dir=images_dir,
            seeds=active_seeds,
            config=config,
            device=device,
        )
    else:
        generate_native_samples(
            run=run,
            checkpoint=checkpoint,
            images_dir=images_dir,
            seeds=active_seeds,
            config=config,
            device=device,
        )
    missing_after, _complete_after = validate_or_prepare_generation_dir(
        images_dir,
        n_images=len(seeds),
        overwrite=False,
    )
    if missing_after:
        raise RuntimeError(f"Generation incomplete in {images_dir}: missing {len(missing_after)} files")
    save_json(
        stage_dir / "generation_metadata.json",
        {
            "cached": False,
            "num_generated_images": len(seeds),
            "generation_seed_list": list(map(int, seeds)),
            "checkpoint_identifier": checkpoint.checkpoint_identifier,
            "checkpoint_path": checkpoint.checkpoint_path,
            "generation_backend_used": run.generation_backend_used,
            "timestamp": utc_timestamp(),
        },
    )
    return images_dir, False


def _features_for_paths(
    *,
    paths: Sequence[Path],
    extractor,
    cache_path: Path,
    config: Mapping[str, Any],
    normalization_mode: str,
) -> np.ndarray:
    return extract_features(
        paths,
        extractor,
        batch_size=max(1, int(config.get("metric_batch_size") or config.get("generation_batch_size") or 8)),
        cache_path=cache_path,
        force=bool(config.get("overwrite_existing_metrics", False)),
        normalization_mode=normalization_mode,
        metadata={"num_images": len(paths)},
    )


def compute_metrics_from_paths(
    *,
    real_features: np.ndarray,
    generated_features: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    include_fid: bool,
    include_kid: bool,
    include_mmd: bool,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    min_count = min(real_features.shape[0], generated_features.shape[0])
    if include_fid and bool(config.get("compute_fid", True)):
        metrics["FID"] = compute_fid(real_features, generated_features)
    if include_kid and bool(config.get("compute_kid", True)):
        kid_cfg = config.get("kid", {}) if isinstance(config.get("kid"), Mapping) else {}
        metrics["KID"] = compute_kid(
            real_features,
            generated_features,
            subsets=int(kid_cfg.get("subsets", 100)),
            subset_size=min(int(kid_cfg.get("subset_size", 1000)), min_count),
            seed=int(seed),
        )
    if include_mmd and bool(config.get("compute_mmd", True)):
        mmd_cfg = config.get("mmd", {}) if isinstance(config.get("mmd"), Mapping) else {}
        metrics["MMD"] = compute_rbf_mmd(
            real_features,
            generated_features,
            bandwidths=mmd_cfg.get("bandwidths", [0.1, 1.0, 10.0]),
        )
    return metrics


def add_weighted_normalized_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    kid_weight: float,
    fid_weight: float,
) -> list[dict[str, Any]]:
    mutable = [dict(row) for row in rows]
    for source_key, norm_key in (("KID", "normalized_KID"), ("FID", "normalized_FID")):
        values = [float(row[source_key]) for row in mutable]
        min_value = min(values)
        max_value = max(values)
        denom = max_value - min_value
        for row in mutable:
            row[norm_key] = 0.0 if denom == 0.0 else (float(row[source_key]) - min_value) / denom
    for row in mutable:
        row["kid_weight"] = float(kid_weight)
        row["fid_weight"] = float(fid_weight)
        row["selection_score"] = (
            float(kid_weight) * float(row["normalized_KID"])
            + float(fid_weight) * float(row["normalized_FID"])
        )
    ranked = sorted(mutable, key=lambda row: (float(row["selection_score"]), float(row["KID"]), float(row["FID"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def rank_by_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    ranked = sorted((dict(row) for row in rows), key=lambda row: float(row[metric]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _stage_paths(run_output_dir: Path, checkpoint: CheckpointCandidate, stage: str) -> Path:
    return run_output_dir / checkpoint.checkpoint_identifier / stage


def _image_paths(images_dir: Path, count: int) -> list[Path]:
    paths = expected_generated_paths(images_dir, count)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} generated images in {images_dir}")
    return paths


def run_stage1(
    *,
    run: RunResolution,
    discovery: DiscoveryResult,
    run_output_dir: Path,
    real_features: np.ndarray,
    extractor,
    config: Mapping[str, Any],
    seeds: Sequence[int],
    real_reference_path: Path,
    device: str,
) -> list[dict[str, Any]]:
    metrics_path = run_output_dir / "stage1_metrics.json"
    if metrics_path.is_file() and not bool(config.get("overwrite_existing_metrics", False)):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return list(payload.get("ranking", payload.get("metrics", [])))

    rows = []
    for checkpoint in discovery.candidates:
        images_dir, _cached = ensure_generated_stage(
            run=run,
            checkpoint=checkpoint,
            stage_dir=_stage_paths(run_output_dir, checkpoint, "stage1"),
            seeds=seeds,
            config=config,
            device=device,
        )
        gen_paths = _image_paths(images_dir, len(seeds))
        gen_features = _features_for_paths(
            paths=gen_paths,
            extractor=extractor,
            cache_path=run_output_dir / "features" / f"{checkpoint.checkpoint_identifier}_stage1.npz",
            config=config,
            normalization_mode=UINT8_LINEAR,
        )
        metrics = compute_metrics_from_paths(
            real_features=real_features,
            generated_features=gen_features,
            config=config,
            seed=int(config.get("generation_seed", 1234)),
            include_fid=True,
            include_kid=True,
            include_mmd=False,
        )
        rows.append(
            {
                "run_identifier": run.run_identifier,
                "run_dir": str(run.run_dir),
                "checkpoint_identifier": checkpoint.checkpoint_identifier,
                "checkpoint_path": checkpoint.checkpoint_path,
                "model_type": run.model_type,
                "sampler_name": run.sampler_name,
                "generation_backend_used": run.generation_backend_used,
                "generated_image_folder": str(images_dir),
                "num_generated_images": len(seeds),
                "real_reference_path": str(real_reference_path),
                **metrics,
                "generation_seed_start": int(seeds[0]) if seeds else None,
                "generation_seed_list": list(map(int, seeds)),
                "metric_feature_extractor": str(config.get("metric_feature_extractor", "inception")),
                "timestamp": utc_timestamp(),
            }
        )
    ranked = add_weighted_normalized_scores(
        rows,
        kid_weight=float(config.get("kid_weight", 0.8)),
        fid_weight=float(config.get("fid_weight", 0.2)),
    )
    payload = {
        "metrics": ranked,
        "ranking": ranked,
        "selected_top_k_checkpoints": [row["checkpoint_identifier"] for row in ranked[: int(config.get("top_k_checkpoints", 3))]],
        "timestamp": utc_timestamp(),
    }
    save_json(metrics_path, payload)
    return ranked


def run_stage2(
    *,
    run: RunResolution,
    top_candidates: Sequence[CheckpointCandidate],
    run_output_dir: Path,
    real_features: np.ndarray,
    extractor,
    config: Mapping[str, Any],
    stage1_seeds: Sequence[int],
    stage2_seeds: Sequence[int],
    device: str,
) -> list[dict[str, Any]]:
    metrics_path = run_output_dir / "stage2_metrics.json"
    if metrics_path.is_file() and not bool(config.get("overwrite_existing_metrics", False)):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return list(payload.get("ranking", payload.get("metrics", [])))

    rows = []
    for checkpoint in top_candidates:
        stage2_images_dir, _cached = ensure_generated_stage(
            run=run,
            checkpoint=checkpoint,
            stage_dir=_stage_paths(run_output_dir, checkpoint, "stage2"),
            seeds=stage2_seeds,
            config=config,
            device=device,
        )
        stage1_images_dir = _stage_paths(run_output_dir, checkpoint, "stage1") / "generated_npy_images"
        combined_paths = [
            *_image_paths(stage1_images_dir, len(stage1_seeds)),
            *_image_paths(stage2_images_dir, len(stage2_seeds)),
        ]
        gen_features = _features_for_paths(
            paths=combined_paths,
            extractor=extractor,
            cache_path=run_output_dir / "features" / f"{checkpoint.checkpoint_identifier}_stage1_stage2.npz",
            config=config,
            normalization_mode=UINT8_LINEAR,
        )
        metrics = compute_metrics_from_paths(
            real_features=real_features,
            generated_features=gen_features,
            config=config,
            seed=int(config.get("generation_seed", 1234)) + 17,
            include_fid=False,
            include_kid=True,
            include_mmd=False,
        )
        rows.append(
            {
                "run_identifier": run.run_identifier,
                "run_dir": str(run.run_dir),
                "checkpoint_identifier": checkpoint.checkpoint_identifier,
                "checkpoint_path": checkpoint.checkpoint_path,
                "model_type": run.model_type,
                "sampler_name": run.sampler_name,
                "generation_backend_used": run.generation_backend_used,
                "stage1_generated_image_folder": str(stage1_images_dir),
                "stage2_generated_image_folder": str(stage2_images_dir),
                "total_generated_images": len(combined_paths),
                **metrics,
                "generation_seed_start": int(stage2_seeds[0]) if stage2_seeds else None,
                "generation_seed_list": list(map(int, stage2_seeds)),
                "metric_feature_extractor": str(config.get("metric_feature_extractor", "inception")),
                "timestamp": utc_timestamp(),
            }
        )
    ranked = rank_by_metric(rows, "KID")
    save_json(
        metrics_path,
        {
            "metrics": ranked,
            "ranking": ranked,
            "selected_best_checkpoint": ranked[0]["checkpoint_identifier"] if ranked else None,
            "timestamp": utc_timestamp(),
        },
    )
    return ranked


def run_final_metrics(
    *,
    run: RunResolution,
    selected: CheckpointCandidate,
    run_output_dir: Path,
    real_paths: Sequence[Path],
    lpips_real_paths: Sequence[Path],
    real_features: np.ndarray,
    real_normalization: str,
    lpips_real_normalization: str,
    extractor,
    config: Mapping[str, Any],
    seeds_by_stage: Mapping[str, Sequence[int]],
    real_reference_path: Path,
    lpips_reference_path: Path,
    device: str,
) -> dict[str, Any]:
    metrics_path = run_output_dir / "final_metrics.json"
    if metrics_path.is_file() and not bool(config.get("overwrite_existing_metrics", False)):
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    stage3_images_dir, _cached = ensure_generated_stage(
        run=run,
        checkpoint=selected,
        stage_dir=_stage_paths(run_output_dir, selected, "stage3"),
        seeds=seeds_by_stage["stage3"],
        config=config,
        device=device,
    )
    stage1_images_dir = _stage_paths(run_output_dir, selected, "stage1") / "generated_npy_images"
    stage2_images_dir = _stage_paths(run_output_dir, selected, "stage2") / "generated_npy_images"
    generated_paths = [
        *_image_paths(stage1_images_dir, len(seeds_by_stage["stage1"])),
        *_image_paths(stage2_images_dir, len(seeds_by_stage["stage2"])),
        *_image_paths(stage3_images_dir, len(seeds_by_stage["stage3"])),
    ]
    gen_features = _features_for_paths(
        paths=generated_paths,
        extractor=extractor,
        cache_path=run_output_dir / "features" / f"{selected.checkpoint_identifier}_final_2000.npz",
        config=config,
        normalization_mode=UINT8_LINEAR,
    )
    metrics = compute_metrics_from_paths(
        real_features=real_features,
        generated_features=gen_features,
        config=config,
        seed=int(config.get("generation_seed", 1234)) + 29,
        include_fid=True,
        include_kid=True,
        include_mmd=True,
    )
    lpips_result = None
    if bool(config.get("compute_intra_lpips", True)):
        lpips_result = compute_intra_lpips(
            real_paths=lpips_real_paths,
            generated_paths=generated_paths,
            backbone=str(config.get("lpips_backbone", "alex")),
            device=device,
            batch_size=max(1, int(config.get("lpips_batch_size") or config.get("metric_batch_size") or 8)),
            real_normalization_mode=lpips_real_normalization,
            generated_normalization_mode=UINT8_LINEAR,
        )
        metrics["Intra-LPIPS"] = lpips_result.value

    payload = {
        "run_identifier": run.run_identifier,
        "run_dir": str(run.run_dir),
        "selected_checkpoint_identifier": selected.checkpoint_identifier,
        "selected_checkpoint_path": selected.checkpoint_path,
        "model_type": run.model_type,
        "sampler_name": run.sampler_name,
        "generation_backend_used": run.generation_backend_used,
        "selection_reason": "lowest KID on combined 1000-image stage1+stage2 set",
        "stage1_generated_image_folder": str(stage1_images_dir),
        "stage2_generated_image_folder": str(stage2_images_dir),
        "stage3_generated_image_folder": str(stage3_images_dir),
        "total_generated_images": len(generated_paths),
        "real_reference_path": str(real_reference_path),
        **metrics,
        "metric_directions": {
            "KID": "lower_is_better",
            "FID": "lower_is_better",
            "MMD": "lower_is_better",
            "Intra-LPIPS": "higher_is_better",
        },
        "metric_feature_extractor": str(config.get("metric_feature_extractor", "inception")),
        "LPIPS_backbone_config": {
            "lpips_metric_type": str(config.get("lpips_metric_type", "intra_lpips")),
            "lpips_backbone": str(config.get("lpips_backbone", "alex")),
            "lpips_reference_path": str(lpips_reference_path),
            "diagnostics": lpips_result.to_dict() if lpips_result is not None else None,
        },
        "generation_seed_information": {key: list(map(int, values)) for key, values in seeds_by_stage.items()},
        "timestamp": utc_timestamp(),
        "full_relevant_config_snapshot": dict(config),
    }
    save_json(metrics_path, payload)
    return payload


def write_text_summary(
    path: Path,
    *,
    run: RunResolution,
    discovery: DiscoveryResult,
    stage1_ranking: Sequence[Mapping[str, Any]],
    top3: Sequence[CheckpointCandidate],
    stage2_ranking: Sequence[Mapping[str, Any]],
    final_metrics: Mapping[str, Any],
) -> None:
    lines = [
        f"Run: {run.run_identifier}",
        f"Run directory: {run.run_dir}",
        f"Generation backend: {run.generation_backend_used}",
        f"Sampler: {run.sampler_name}",
        "",
        "Candidate checkpoints considered:",
    ]
    for candidate in discovery.candidates:
        lines.append(f"- {candidate.checkpoint_identifier}: {candidate.checkpoint_path}")
    lines.append("")
    lines.append("Excluded checkpoints:")
    if discovery.excluded:
        for excluded in discovery.excluded:
            lines.append(f"- {excluded.path}: {excluded.reason}")
    else:
        lines.append("- none")
    lines.extend(["", "Stage-1 ranking:"])
    for row in stage1_ranking:
        lines.append(
            f"{row['rank']}. {row['checkpoint_identifier']} "
            f"KID={float(row.get('KID', math.nan)):.6g} "
            f"FID={float(row.get('FID', math.nan)):.6g} "
            f"score={float(row.get('selection_score', math.nan)):.6g}"
        )
    lines.extend(["", "Top-3 selected checkpoints:"])
    for candidate in top3:
        lines.append(f"- {candidate.checkpoint_identifier}")
    lines.extend(["", "Stage-2 KID ranking:"])
    for row in stage2_ranking:
        lines.append(f"{row['rank']}. {row['checkpoint_identifier']} KID={float(row.get('KID', math.nan)):.6g}")
    lines.extend(
        [
            "",
            f"Final selected checkpoint: {final_metrics.get('selected_checkpoint_identifier')}",
            "Final 2000-image metrics:",
        ]
    )
    for key in ("KID", "FID", "MMD", "Intra-LPIPS"):
        if key in final_metrics:
            lines.append(f"- {key}: {float(final_metrics[key]):.6g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(run_entry: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    run = resolve_run(run_entry, config)
    run_output_dir = resolve_path(config.get("output_root") or "/scratch/bacobax02")
    if run_output_dir is None:
        raise ValueError("output_root cannot be empty.")
    run_output_dir = run_output_dir / run.run_identifier
    run_output_dir.mkdir(parents=True, exist_ok=True)

    discovery = discover_candidate_checkpoints(
        run.run_dir,
        model_type=run.model_type,
        checkpoint_min_epoch=int(config.get("checkpoint_min_epoch", 50)),
        checkpoint_min_step=config.get("checkpoint_min_step"),
    )
    seeds_by_stage = make_stage_seeds(config)
    device = get_device(config)

    real_paths, real_normalization, real_reference_path = discover_reference_images(config, run)
    lpips_real_paths, lpips_real_normalization, lpips_reference_path = discover_reference_images(
        config,
        run,
        split_override=str(config.get("lpips_reference_split", "train")),
    )
    feature_name = str(config.get("metric_feature_extractor", "inception")).lower()
    extractor = build_feature_extractor(feature_name, config, device)
    real_features = _features_for_paths(
        paths=real_paths,
        extractor=extractor,
        cache_path=run_output_dir / "features" / f"real_{feature_name}.npz",
        config=config,
        normalization_mode=real_normalization,
    )

    stage1_ranking = run_stage1(
        run=run,
        discovery=discovery,
        run_output_dir=run_output_dir,
        real_features=real_features,
        extractor=extractor,
        config=config,
        seeds=seeds_by_stage["stage1"],
        real_reference_path=real_reference_path,
        device=device,
    )
    by_id = {candidate.checkpoint_identifier: candidate for candidate in discovery.candidates}
    top_k = min(int(config.get("top_k_checkpoints", 3)), len(stage1_ranking))
    top_candidates = [by_id[row["checkpoint_identifier"]] for row in stage1_ranking[:top_k]]
    stage2_ranking = run_stage2(
        run=run,
        top_candidates=top_candidates,
        run_output_dir=run_output_dir,
        real_features=real_features,
        extractor=extractor,
        config=config,
        stage1_seeds=seeds_by_stage["stage1"],
        stage2_seeds=seeds_by_stage["stage2"],
        device=device,
    )
    selected = by_id[stage2_ranking[0]["checkpoint_identifier"]]
    final_metrics = run_final_metrics(
        run=run,
        selected=selected,
        run_output_dir=run_output_dir,
        real_paths=real_paths,
        lpips_real_paths=lpips_real_paths,
        real_features=real_features,
        real_normalization=real_normalization,
        lpips_real_normalization=lpips_real_normalization,
        extractor=extractor,
        config=config,
        seeds_by_stage=seeds_by_stage,
        real_reference_path=real_reference_path,
        lpips_reference_path=lpips_reference_path,
        device=device,
    )

    summary = {
        "all_candidate_checkpoints": discovery.candidates,
        "excluded_checkpoints": discovery.excluded,
        "stage_1_full_ranking": stage1_ranking,
        "selected_top_3_checkpoints": [candidate.checkpoint_identifier for candidate in top_candidates],
        "stage_2_full_ranking": stage2_ranking,
        "final_selected_checkpoint": selected.checkpoint_identifier,
        "final_metrics": final_metrics,
    }
    save_json(run_output_dir / "checkpoint_selection_summary.json", summary)
    write_text_summary(
        run_output_dir / "checkpoint_selection_summary.txt",
        run=run,
        discovery=discovery,
        stage1_ranking=stage1_ranking,
        top3=top_candidates,
        stage2_ranking=stage2_ranking,
        final_metrics=final_metrics,
    )
    return {"run_identifier": run.run_identifier, "output_dir": str(run_output_dir), **summary}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = resolve_path(args.config)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in config: {config_path}")
    if str(config.get("save_generated_format", "npy")).lower() != "npy":
        raise ValueError("Only save_generated_format='npy' is supported by this pipeline.")
    runs = config.get("runs") or []
    if not runs:
        raise ValueError("Evaluation config must contain at least one run under 'runs'.")

    summaries = []
    for run_entry in runs:
        summaries.append(run_one(run_entry, config))
    print(json.dumps(_jsonable({"runs": summaries}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
