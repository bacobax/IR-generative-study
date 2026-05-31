#!/usr/bin/env python3
"""Generate paired best/latest samples for checkpoint visual comparison."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

from src.algorithms.inference.flow_matching_sampler import (
    FlowMatchingSampler,
    _maybe_wrap_regiondiff_unet,
)
from src.algorithms.inference.layout_flow_matching_sampler import LayoutFlowMatchingSampler
from src.algorithms.inference.rare_layout_dataset_tools import (
    build_layout_dataset,
    load_json,
    sample_layout_batch,
)
from src.algorithms.inference.unconditional_sd_sampler import UnconditionalStableDiffusionSampler
from src.core.data.layout_batching import collate_layout_batch
from src.core.data.dataset_targets import resolve_dataset_target
from src.core.diffusers_compat import import_diffusers_attr
from src.core.normalization import (
    RAW_UINT16_PERCENTILE,
    SENTINEL2_REFLECTANCE,
    UINT8_LINEAR,
)
from src.models.fm_unet import build_fm_unet_from_config, load_unet_config
from src.models.stay_layout_conditioned_unet import build_stay_layout_conditioned_unet
from src.models.vae import (
    build_vae_from_config,
    freeze_vae,
    is_diffusers_vae_config,
    load_diffusers_vae_config,
    load_vae_config,
    load_vae_weights,
)


@dataclass(frozen=True)
class ResolvedRunDirs:
    pipeline_dir: Path
    unet_dir: Path


@dataclass(frozen=True)
class CheckpointChoice:
    role: str
    path: Path
    epoch: Optional[int]
    source: str


@dataclass(frozen=True)
class RunKind:
    model_family: str
    layout_conditioned: bool
    layout_variant: str


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve_run_dirs(weights_dir: str | Path) -> ResolvedRunDirs:
    path = Path(weights_dir)
    if path.name == "UNET":
        return ResolvedRunDirs(pipeline_dir=path.parent, unet_dir=path)
    unet_dir = path / "UNET"
    if unet_dir.exists() or not path.suffix:
        return ResolvedRunDirs(pipeline_dir=path, unet_dir=unet_dir)
    raise ValueError(f"--weights_dir must be a run directory or UNET directory, got {weights_dir!r}")


def _loadable_checkpoint(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "unet_state" in state:
            state = state["unet_state"]
        return isinstance(state, dict) and bool(state)
    except Exception as exc:
        print(f"[warn] Skipping unloadable checkpoint {path}: {exc}")
        return False


def _first_loadable(role: str, candidates: Sequence[Tuple[Path, Optional[int], str]]) -> CheckpointChoice:
    for path, epoch, source in candidates:
        if _loadable_checkpoint(path):
            return CheckpointChoice(role=role, path=path, epoch=epoch, source=source)
    candidate_text = ", ".join(str(path) for path, _epoch, _source in candidates)
    raise FileNotFoundError(f"No loadable {role} checkpoint found. Tried: {candidate_text}")


def _epoch_from_name(path: Path) -> Optional[int]:
    match = re.search(r"_epoch_(\d+)(?:_ckpt)?\.pt$", path.name)
    return int(match.group(1)) if match else None


def _epoch_candidates(unet_dir: Path, *, ckpt: bool) -> List[Tuple[Path, Optional[int], str]]:
    rows: List[Tuple[int, Path]] = []
    pattern = "*_epoch_*_ckpt.pt" if ckpt else "*_epoch_*.pt"
    for path in unet_dir.glob(pattern):
        is_ckpt = path.name.endswith("_ckpt.pt")
        if bool(is_ckpt) != bool(ckpt):
            continue
        epoch = _epoch_from_name(path)
        if epoch is not None:
            rows.append((epoch, path))
    rows.sort(key=lambda item: item[0], reverse=True)
    source = "latest_epoch_ckpt" if ckpt else "latest_epoch_weights"
    return [(path, epoch, source) for epoch, path in rows]


def _checkpoint_stems(unet_dir: Path) -> List[str]:
    stems = []
    for stem in ("unet_fm", "unet_sd_uncond"):
        if any(unet_dir.glob(f"{stem}_*.pt")):
            stems.append(stem)
    return stems or ["unet_fm", "unet_sd_uncond"]


def _best_epoch_metadata_candidates(unet_dir: Path) -> List[Tuple[Path, Optional[int], str]]:
    """Resolve best epoch weights from full checkpoint metadata when available."""
    candidates: List[Tuple[Path, Optional[int], str]] = []
    metadata_paths = [path for path, _epoch, _source in _epoch_candidates(unet_dir, ckpt=True)]
    metadata_paths.extend(
        path
        for path in sorted(
            unet_dir.glob("*last*.pt"),
            key=lambda item: (item.stat().st_mtime if item.exists() else 0.0, item.name),
            reverse=True,
        )
    )
    seen = set()
    for metadata_path in metadata_paths:
        if metadata_path in seen or not metadata_path.is_file():
            continue
        seen.add(metadata_path)
        try:
            ckpt = torch.load(metadata_path, map_location="cpu")
        except Exception as exc:
            print(f"[warn] Could not inspect checkpoint metadata {metadata_path}: {exc}")
            continue
        if not isinstance(ckpt, dict):
            continue
        best_epoch = ckpt.get("best_epoch")
        if best_epoch is None:
            continue
        try:
            epoch_num = int(best_epoch) + 1
        except (TypeError, ValueError):
            continue
        if epoch_num <= 0:
            continue
        for stem in _checkpoint_stems(unet_dir):
            candidates.append((unet_dir / f"{stem}_epoch_{epoch_num}.pt", epoch_num, "best_epoch_metadata"))
    return candidates


def resolve_checkpoint_pair(unet_dir: str | Path) -> Dict[str, CheckpointChoice]:
    unet_dir = Path(unet_dir)
    best_names = [
        "unet_fm_best.pt",
        "unet_sd_uncond_best.pt",
        "best.pt",
    ]
    best_candidates = [(unet_dir / name, None, "best_name") for name in best_names]
    best_candidates.extend(
        (path, _epoch_from_name(path), "best_glob")
        for path in sorted(unet_dir.glob("*best.pt"))
        if path.name not in set(best_names)
    )
    best_candidates.extend(_best_epoch_metadata_candidates(unet_dir))
    best_candidates.extend(
        (path, epoch, "best_fallback_latest_epoch_weights")
        for path, epoch, _source in _epoch_candidates(unet_dir, ckpt=False)
    )
    best_candidates.extend(
        (path, epoch, "best_fallback_latest_epoch_ckpt")
        for path, epoch, _source in _epoch_candidates(unet_dir, ckpt=True)
    )

    last_candidates = sorted(
        unet_dir.glob("*last*.pt"),
        key=lambda path: (path.stat().st_mtime if path.exists() else 0.0, path.name),
        reverse=True,
    )
    latest_candidates: List[Tuple[Path, Optional[int], str]] = [
        (path, _epoch_from_name(path), "last") for path in last_candidates
    ]
    latest_candidates.extend(_epoch_candidates(unet_dir, ckpt=False))
    latest_candidates.extend(_epoch_candidates(unet_dir, ckpt=True))

    return {
        "best": _first_loadable("best", best_candidates),
        "latest": _first_loadable("latest", latest_candidates),
    }


def detect_run_kind(
    pipeline_dir: str | Path,
    preset: Dict[str, Any],
    *,
    model_family: str = "auto",
) -> RunKind:
    pipeline_dir = Path(pipeline_dir)
    if model_family == "auto":
        model_family = "sd" if (pipeline_dir / "SCHEDULER").exists() or "stable_diffusion" in pipeline_dir.parts else "fm"

    layout_cfg = preset.get("layout_conditioning", {}) or {}
    has_stay_meta = (pipeline_dir / "layout_conditioning.json").is_file()
    has_regiondiff_meta = (pipeline_dir / "regiondiff_config.json").is_file()
    layout_conditioned = bool(has_stay_meta or has_regiondiff_meta or layout_cfg.get("enabled", False))
    variant = str(layout_cfg.get("variant") or "")
    if has_stay_meta:
        try:
            variant = str(load_json(pipeline_dir / "layout_conditioning.json").get("variant") or variant)
        except Exception:
            pass
    elif has_regiondiff_meta and not variant:
        variant = "regiondiff_v1"
    return RunKind(model_family=str(model_family), layout_conditioned=layout_conditioned, layout_variant=variant)


def _state_dict_from_checkpoint(path: Path, *, map_location: str | torch.device) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "unet_state" in state:
        state = state["unet_state"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {path} does not contain a state dict")
    return state


def _resolve_unet_config(pipeline_dir: Path, preset: Dict[str, Any]) -> Dict[str, Any]:
    saved = pipeline_dir / "UNET" / "config.json"
    if saved.is_file():
        return load_unet_config(str(saved))
    config_path = preset.get("model", {}).get("unet_config")
    if not config_path:
        raise FileNotFoundError(f"No UNET/config.json in {pipeline_dir / 'UNET'} and no model.unet_config in preset")
    return load_unet_config(str(config_path))


def _infer_vae_downsample_factor(vae_config: Dict[str, Any]) -> int:
    for key in ("num_channels", "block_out_channels", "down_block_types"):
        values = vae_config.get(key)
        if isinstance(values, (list, tuple)) and values:
            return 2 ** max(0, len(values) - 1)
    raise ValueError("Cannot infer VAE downsample factor from VAE config")


def _apply_training_sample_size(
    unet_cfg: Dict[str, Any],
    preset: Dict[str, Any],
    vae_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Mirror training-time latent sample-size resolution for sparse run folders."""
    resolved = dict(unet_cfg)
    if vae_cfg is None:
        return resolved
    image_size = preset.get("data", {}).get("image_size")
    if image_size is None:
        return resolved
    factor = _infer_vae_downsample_factor(vae_cfg)
    image_size = int(image_size)
    if image_size % factor != 0:
        raise ValueError(f"image_size={image_size} is not divisible by VAE downsample factor={factor}")
    latent_size = image_size // factor
    if int(resolved.get("sample_size", latent_size)) != latent_size:
        print(
            "[info] Adjusting UNET sample_size from "
            f"{resolved.get('sample_size')} to {latent_size} "
            f"using image_size={image_size} and VAE downsample factor={factor}."
        )
    resolved["sample_size"] = latent_size
    return resolved


def _resolve_vae_config(pipeline_dir: Path, preset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    saved = pipeline_dir / "VAE" / "config.json"
    if saved.is_file():
        return load_vae_config(str(saved))
    model_cfg = preset.get("model", {}) or {}
    pretrained = model_cfg.get("vae_pretrained_model_name_or_path")
    if pretrained:
        return load_diffusers_vae_config(
            str(pretrained),
            subfolder=model_cfg.get("vae_pretrained_subfolder", "vae"),
            revision=model_cfg.get("vae_revision"),
            variant=model_cfg.get("vae_variant"),
        )
    vae_config = model_cfg.get("vae_config")
    if vae_config:
        return load_vae_config(str(vae_config))
    return None


def _build_vae(
    pipeline_dir: Path,
    preset: Dict[str, Any],
    device: str | torch.device,
    *,
    vae_cfg: Optional[Dict[str, Any]] = None,
):
    if vae_cfg is None:
        vae_cfg = _resolve_vae_config(pipeline_dir, preset)
    if vae_cfg is None:
        return None
    vae = build_vae_from_config(vae_cfg, device=device)
    vae_weights = preset.get("model", {}).get("vae_weights")
    if vae_weights:
        load_vae_weights(vae, str(vae_weights), map_location=device)
    else:
        saved_best = pipeline_dir / "VAE" / "vae_best.pt"
        if saved_best.is_file() and not is_diffusers_vae_config(vae_cfg):
            load_vae_weights(vae, str(saved_best), map_location=device)
    return freeze_vae(vae)


def _layout_meta_from_preset(
    preset: Dict[str, Any],
    unet_cfg: Dict[str, Any],
    category_id_to_name: Dict[int, str],
    checkpoint_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    cfg = preset.get("layout_conditioning", {}) or {}
    state_num_classes = 0
    if checkpoint_state is not None:
        class_embedding = checkpoint_state.get("object_encoder.class_embedding.weight")
        if torch.is_tensor(class_embedding) and class_embedding.ndim >= 2:
            state_num_classes = int(class_embedding.shape[0])
    num_classes = max(
        int(cfg.get("num_classes") or 0),
        int((max(category_id_to_name) + 1) if category_id_to_name else 1),
        state_num_classes,
    )
    return {
        "variant": str(cfg.get("variant", "stay_v2")),
        "num_classes": num_classes,
        "class_embed_dim": int(cfg.get("class_embed_dim", 48)),
        "bbox_embed_dim": int(cfg.get("bbox_embed_dim", 48)),
        "object_embed_dim": int(cfg.get("object_embed_dim", 64)),
        "image_in_channels": int(unet_cfg.get("in_channels", 4)),
        "category_id_to_name": {str(k): v for k, v in category_id_to_name.items()},
        "use_style_latent": bool(cfg.get("use_style_latent", True)),
        "style_latent_dim": int(cfg.get("style_latent_dim", 16)),
        "style_seed": int(cfg.get("style_seed", 1234)),
        "mask_resolution": int(cfg.get("mask_resolution", 16)),
        "mask_hidden_channels": int(cfg.get("mask_hidden_channels", 32)),
        "mask_threshold": float(cfg.get("mask_threshold", 0.5)),
        "edge_dilation": int(cfg.get("edge_dilation", 1)),
        "injection_mode": str(cfg.get("injection_mode", "ea_norm")),
        "use_masked_context": bool(cfg.get("use_masked_context", True)),
        "mask_overlap_loss_weight": float(cfg.get("mask_overlap_loss_weight", 0.05)),
        "mask_sharpness_loss_weight": float(cfg.get("mask_sharpness_loss_weight", 0.01)),
        "mask_activation_loss_weight": float(cfg.get("mask_activation_loss_weight", 0.01)),
    }


def _build_fm_sampler(
    *,
    pipeline_dir: Path,
    preset: Dict[str, Any],
    checkpoint_path: Path,
    device: str,
    layout_variant: str,
    category_id_to_name: Optional[Dict[int, str]] = None,
):
    vae_cfg = _resolve_vae_config(pipeline_dir, preset)
    unet_cfg = _apply_training_sample_size(_resolve_unet_config(pipeline_dir, preset), preset, vae_cfg)
    vae = _build_vae(pipeline_dir, preset, device, vae_cfg=vae_cfg)
    checkpoint_state = _state_dict_from_checkpoint(checkpoint_path, map_location=device)

    if layout_variant == "stay_v2":
        if (pipeline_dir / "layout_conditioning.json").is_file():
            layout_meta = load_json(pipeline_dir / "layout_conditioning.json")
        else:
            layout_meta = _layout_meta_from_preset(
                preset,
                unet_cfg,
                category_id_to_name or {},
                checkpoint_state=checkpoint_state,
            )
        meta_category_names = {
            int(key): value for key, value in layout_meta.get("category_id_to_name", {}).items()
        }
        unet = build_stay_layout_conditioned_unet(
            unet_cfg,
            image_in_channels=int(layout_meta["image_in_channels"]),
            num_classes=int(layout_meta["num_classes"]),
            class_embed_dim=int(layout_meta["class_embed_dim"]),
            bbox_embed_dim=int(layout_meta["bbox_embed_dim"]),
            object_embed_dim=int(layout_meta["object_embed_dim"]),
            use_style_latent=bool(layout_meta["use_style_latent"]),
            style_latent_dim=int(layout_meta["style_latent_dim"]),
            style_seed=int(layout_meta["style_seed"]),
            mask_resolution=int(layout_meta["mask_resolution"]),
            mask_hidden_channels=int(layout_meta["mask_hidden_channels"]),
            mask_threshold=float(layout_meta["mask_threshold"]),
            edge_dilation=int(layout_meta["edge_dilation"]),
            injection_mode=str(layout_meta["injection_mode"]),
            use_masked_context=bool(layout_meta["use_masked_context"]),
            mask_overlap_loss_weight=float(layout_meta["mask_overlap_loss_weight"]),
            mask_sharpness_loss_weight=float(layout_meta["mask_sharpness_loss_weight"]),
            mask_activation_loss_weight=float(layout_meta["mask_activation_loss_weight"]),
            category_id_to_name=meta_category_names,
            device=device,
        )
    else:
        unet = build_fm_unet_from_config(unet_cfg, device=device)
        unet = _maybe_wrap_regiondiff_unet(
            unet,
            pipeline_dir=str(pipeline_dir),
            backbone_kind="fm_unet2d",
        )
        unet = torch.nn.Module.to(unet, device)

    unet.load_state_dict(checkpoint_state, strict=True)
    unet.eval()
    sampler_cls = LayoutFlowMatchingSampler if layout_variant == "stay_v2" else FlowMatchingSampler
    if vae is not None:
        return sampler_cls.from_stable(
            unet,
            vae,
            device=device,
            t_scale=float(preset.get("training", {}).get("t_scale", 1000.0)),
            train_target=str(preset.get("training", {}).get("train_target", "v")),
        )
    return sampler_cls(
        unet,
        device=device,
        t_scale=float(preset.get("training", {}).get("t_scale", 1000.0)),
        train_target=str(preset.get("training", {}).get("train_target", "v")),
    )


def _build_sd_sampler(
    *,
    pipeline_dir: Path,
    preset: Dict[str, Any],
    checkpoint_path: Path,
    device: str,
):
    vae_cfg = _resolve_vae_config(pipeline_dir, preset)
    unet_cfg = _apply_training_sample_size(_resolve_unet_config(pipeline_dir, preset), preset, vae_cfg)
    unet = build_fm_unet_from_config(unet_cfg, device=device)
    unet = _maybe_wrap_regiondiff_unet(
        unet,
        pipeline_dir=str(pipeline_dir),
        backbone_kind="sd_uncond_unet2d",
    )
    unet = torch.nn.Module.to(unet, device)
    unet.load_state_dict(_state_dict_from_checkpoint(checkpoint_path, map_location=device), strict=True)
    unet.eval()

    vae = _build_vae(pipeline_dir, preset, device, vae_cfg=vae_cfg)
    if vae is None:
        raise FileNotFoundError("SD sampling requires a VAE config from the run folder or preset")

    DDPMScheduler = import_diffusers_attr("diffusers", "DDPMScheduler")
    scheduler_dir = pipeline_dir / "SCHEDULER"
    if scheduler_dir.is_dir():
        noise_scheduler = DDPMScheduler.from_pretrained(str(scheduler_dir))
    else:
        diffusion_cfg = preset.get("diffusion", {}) or {}
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(diffusion_cfg.get("num_train_timesteps", 1000)),
            beta_schedule=str(diffusion_cfg.get("beta_schedule", "scaled_linear")),
            beta_start=float(diffusion_cfg.get("beta_start", 0.00085)),
            beta_end=float(diffusion_cfg.get("beta_end", 0.012)),
            prediction_type=str(diffusion_cfg.get("prediction_type", "epsilon")),
        )
    return UnconditionalStableDiffusionSampler.from_stable(
        unet,
        vae,
        noise_scheduler,
        device=device,
    )


def tensor_to_output_array(image: torch.Tensor, *, normalization_mode: str) -> np.ndarray:
    arr = image.detach().cpu().to(torch.float32).numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    scaled = (np.clip(arr, -1.0, 1.0) + 1.0) / 2.0
    if normalization_mode == RAW_UINT16_PERCENTILE:
        from src.core.constants import P0001_PERCENTILE_RAW_IMAGES, RAW_RANGE

        raw = scaled * RAW_RANGE + P0001_PERCENTILE_RAW_IMAGES
        return np.clip(np.rint(raw), 0, 65535).astype(np.uint16)
    if normalization_mode == UINT8_LINEAR:
        return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)
    if normalization_mode == SENTINEL2_REFLECTANCE:
        return np.clip(np.rint(scaled * 10000.0), 0, 10000).astype(np.uint16)
    raise ValueError(
        f"Unknown normalization_mode={normalization_mode!r}. "
        f"Expected one of: {RAW_UINT16_PERCENTILE!r}, "
        f"{UINT8_LINEAR!r}, {SENTINEL2_REFLECTANCE!r}"
    )


def tensor_array_to_preview_uint8(arr: np.ndarray) -> np.ndarray:
    """Create a grayscale preview from a generated tensor array."""
    preview = arr
    if preview.ndim == 3:
        preview = preview.mean(axis=0)
    preview = preview.astype(np.float32, copy=False)
    if float(np.nanmax(preview)) <= 1.5 and float(np.nanmin(preview)) >= -1.5:
        preview = (np.clip(preview, -1.0, 1.0) + 1.0) * 127.5
        return np.clip(preview, 0, 255).astype(np.uint8)
    lo = float(np.nanpercentile(preview, 1.0))
    hi = float(np.nanpercentile(preview, 99.0))
    if hi <= lo:
        return np.zeros_like(preview, dtype=np.uint8)
    return np.clip((preview - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def save_generated_image(
    *,
    images_dir: Path,
    previews_dir: Path,
    image_id: int,
    image: torch.Tensor,
    normalization_mode: str,
) -> str:
    file_name = f"sample_{image_id:06d}.npy"
    arr = tensor_to_output_array(image, normalization_mode=normalization_mode)
    np.save(images_dir / file_name, arr)
    preview = tensor_array_to_preview_uint8(arr)
    Image.fromarray(preview, mode="L").save(previews_dir / f"sample_{image_id:06d}.png")
    return file_name


def _coco_bbox_from_xyxy(box_xyxy: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def initialize_output_dir(output_dir: Path, *, overwrite: bool) -> Tuple[Path, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    previews_dir = output_dir / "previews"
    images_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    return images_dir, previews_dir


def export_conditional_comparison_split(
    *,
    output_dir: str | Path,
    records: Sequence[Dict[str, Any]],
    generated_images: Sequence[torch.Tensor],
    categories: Sequence[Dict[str, Any]],
    checkpoint: CheckpointChoice,
    model_family: str,
    layout_variant: str,
    split: str,
    dataset_id: str,
    steps: int,
    seed: int,
    normalization_mode: str = UINT8_LINEAR,
    overwrite: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    images_dir, previews_dir = initialize_output_dir(output_dir, overwrite=overwrite)
    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    provenance_rows: List[Dict[str, Any]] = []
    annotation_id = 1

    for image_id, (sample, image) in enumerate(zip(records, generated_images), start=1):
        file_name = save_generated_image(
            images_dir=images_dir,
            previews_dir=previews_dir,
            image_id=image_id,
            image=image,
            normalization_mode=normalization_mode,
        )
        _, image_h, image_w = image.shape
        coco_images.append(
            {"id": image_id, "file_name": file_name, "width": int(image_w), "height": int(image_h)}
        )
        labels = sample["labels"].tolist()
        boxes_xyxy = sample["boxes_xyxy"].tolist()
        for object_idx, (label, box_xyxy) in enumerate(zip(labels, boxes_xyxy)):
            bbox = _coco_bbox_from_xyxy(box_xyxy)
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "source_image_id": sample.get("image_id"),
                    "source_file_name": sample.get("file_name"),
                    "object_index": object_idx,
                }
            )
            annotation_id += 1
        provenance_rows.append(
            {
                "generated_image_id": image_id,
                "generated_file_name": file_name,
                "source_image_id": sample.get("image_id"),
                "source_file_name": sample.get("file_name"),
                "n_objects": int(sample.get("n_objects", len(labels))),
                "checkpoint_role": checkpoint.role,
                "checkpoint_path": str(checkpoint.path),
                "checkpoint_source": checkpoint.source,
                "checkpoint_epoch": checkpoint.epoch,
            }
        )

    save_json(output_dir / "annotations.json", {"images": coco_images, "annotations": coco_annotations, "categories": list(categories)})
    write_jsonl(output_dir / "metadata" / "provenance.jsonl", provenance_rows)
    summary = {
        "checkpoint_role": checkpoint.role,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_source": checkpoint.source,
        "checkpoint_epoch": checkpoint.epoch,
        "model_family": model_family,
        "layout_conditioned": True,
        "layout_variant": layout_variant,
        "split": split,
        "dataset_id": dataset_id,
        "steps": int(steps),
        "seed": int(seed),
        "n_generated_samples": len(coco_images),
        "n_annotations": len(coco_annotations),
    }
    save_json(output_dir / "metadata" / "summary.json", summary)
    return summary


def export_unconditional_comparison_split(
    *,
    output_dir: str | Path,
    generated_images: Sequence[torch.Tensor],
    checkpoint: CheckpointChoice,
    model_family: str,
    steps: int,
    seed: int,
    normalization_mode: str = UINT8_LINEAR,
    overwrite: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    images_dir, previews_dir = initialize_output_dir(output_dir, overwrite=overwrite)
    provenance_rows: List[Dict[str, Any]] = []
    for image_id, image in enumerate(generated_images, start=1):
        file_name = save_generated_image(
            images_dir=images_dir,
            previews_dir=previews_dir,
            image_id=image_id,
            image=image,
            normalization_mode=normalization_mode,
        )
        provenance_rows.append(
            {
                "generated_image_id": image_id,
                "generated_file_name": file_name,
                "checkpoint_role": checkpoint.role,
                "checkpoint_path": str(checkpoint.path),
                "checkpoint_source": checkpoint.source,
                "checkpoint_epoch": checkpoint.epoch,
            }
        )

    write_jsonl(output_dir / "metadata" / "provenance.jsonl", provenance_rows)
    summary = {
        "checkpoint_role": checkpoint.role,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_source": checkpoint.source,
        "checkpoint_epoch": checkpoint.epoch,
        "model_family": model_family,
        "layout_conditioned": False,
        "steps": int(steps),
        "seed": int(seed),
        "n_generated_samples": len(generated_images),
    }
    save_json(output_dir / "metadata" / "summary.json", summary)
    return summary


def _sample_unconditional(
    sampler,
    *,
    model_family: str,
    n_samples: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    generated = 0
    with tqdm(total=n_samples, desc="Generating random samples", unit="img") as pbar:
        while generated < n_samples:
            bs = min(batch_size, n_samples - generated)
            torch.manual_seed(int(seed) + generated)
            if torch.cuda.is_available() and str(getattr(sampler, "device", "")).startswith("cuda"):
                torch.cuda.manual_seed_all(int(seed) + generated)
            if model_family == "sd":
                latents = sampler.sample(steps=steps, batch_size=bs)
            else:
                latents = sampler.sample_euler(steps=steps, batch_size=bs)
            images = sampler.decode(latents).detach().cpu()
            out.extend([image for image in images])
            generated += bs
            pbar.update(bs)
    return out


def _sample_conditional(
    sampler,
    *,
    model_family: str,
    samples: Sequence[Dict[str, Any]],
    batch_size: int,
    steps: int,
    seed: int,
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for start_idx in tqdm(range(0, len(samples), batch_size), desc="Generating layout samples"):
        chunk = list(samples[start_idx:start_idx + batch_size])
        batch = collate_layout_batch(chunk)
        if model_family == "sd":
            latents = sampler.sample_layout(batch, steps=steps, seed=int(seed) + start_idx)
            images = sampler.decode(latents).detach().cpu()
        elif isinstance(sampler, LayoutFlowMatchingSampler):
            images = sample_layout_batch(sampler, batch, steps=steps, seed=int(seed) + start_idx)
        else:
            latents = sampler.sample_euler_layout(batch, steps=steps, seed=int(seed) + start_idx)
            images = sampler.decode(latents).detach().cpu()
        out.extend([image for image in images])
    return out


def _dataset_for_conditional(
    preset: Dict[str, Any],
    *,
    split: str,
    dataset_root: Optional[str],
    dataset_id: Optional[str],
):
    return build_layout_dataset(
        preset,
        split=split,
        dataset_root=(Path(dataset_root) if dataset_root else None),
        dataset_id=dataset_id,
    )


def _normalization_mode_from_preset(preset: Dict[str, Any], fallback: str = UINT8_LINEAR) -> str:
    dataset_id = preset.get("data", {}).get("dataset_id")
    if not dataset_id:
        return fallback
    try:
        return resolve_dataset_target(str(dataset_id)).normalization_mode
    except Exception:
        return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate best/latest checkpoint visual comparisons.")
    parser.add_argument("--weights_dir", required=True, help="Run directory or UNET weights directory.")
    parser.add_argument("--preset_path", required=True, help="Training preset used for this run.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_family", choices=["auto", "fm", "sd"], default="auto")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--dataset_root", default="")
    parser.add_argument("--dataset_id", default="")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args.weights_dir)
    preset = load_yaml(args.preset_path)
    run_kind = detect_run_kind(run_dirs.pipeline_dir, preset, model_family=args.model_family)
    checkpoints = resolve_checkpoint_pair(run_dirs.unet_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalization_mode = _normalization_mode_from_preset(preset)
    selected_samples: List[Dict[str, Any]] = []
    categories: List[Dict[str, Any]] = []
    dataset_id = str(args.dataset_id or preset.get("data", {}).get("dataset_id", ""))
    if run_kind.layout_conditioned:
        dataset = _dataset_for_conditional(
            preset,
            split=args.split,
            dataset_root=args.dataset_root,
            dataset_id=(args.dataset_id or None),
        )
        normalization_mode = dataset.normalization_mode
        n_select = min(int(args.max_samples), len(dataset))
        selected_samples = [dataset[idx] for idx in range(n_select)]
        categories = [
            {"id": int(category_id), "name": str(name)}
            for category_id, name in sorted(dataset.category_id_to_name.items())
        ]
        if not dataset_id:
            dataset_id = str(preset.get("data", {}).get("dataset_id", ""))

    summaries: Dict[str, Any] = {
        "pipeline_dir": str(run_dirs.pipeline_dir),
        "unet_dir": str(run_dirs.unet_dir),
        "preset_path": str(Path(args.preset_path)),
        "model_family": run_kind.model_family,
        "layout_conditioned": run_kind.layout_conditioned,
        "layout_variant": run_kind.layout_variant,
        "max_samples": int(args.max_samples),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "checkpoints": {},
    }

    for role in ("best", "latest"):
        checkpoint = checkpoints[role]
        print(f"[{role}] checkpoint: {checkpoint.path}")
        if run_kind.model_family == "sd":
            sampler = _build_sd_sampler(
                pipeline_dir=run_dirs.pipeline_dir,
                preset=preset,
                checkpoint_path=checkpoint.path,
                device=device,
            )
        else:
            sampler = _build_fm_sampler(
                pipeline_dir=run_dirs.pipeline_dir,
                preset=preset,
                checkpoint_path=checkpoint.path,
                device=device,
                layout_variant=run_kind.layout_variant,
                category_id_to_name={category["id"]: category["name"] for category in categories},
            )

        if run_kind.layout_conditioned:
            generated = _sample_conditional(
                sampler,
                model_family=run_kind.model_family,
                samples=selected_samples,
                batch_size=max(1, int(args.batch_size)),
                steps=int(args.steps),
                seed=int(args.seed),
            )
            summary = export_conditional_comparison_split(
                output_dir=output_dir / role,
                records=selected_samples,
                generated_images=generated,
                categories=categories,
                checkpoint=checkpoint,
                model_family=run_kind.model_family,
                layout_variant=run_kind.layout_variant,
                split=args.split,
                dataset_id=dataset_id,
                steps=int(args.steps),
                seed=int(args.seed),
                normalization_mode=normalization_mode,
                overwrite=args.overwrite,
            )
        else:
            generated = _sample_unconditional(
                sampler,
                model_family=run_kind.model_family,
                n_samples=int(args.max_samples),
                batch_size=max(1, int(args.batch_size)),
                steps=int(args.steps),
                seed=int(args.seed),
            )
            summary = export_unconditional_comparison_split(
                output_dir=output_dir / role,
                generated_images=generated,
                checkpoint=checkpoint,
                model_family=run_kind.model_family,
                steps=int(args.steps),
                seed=int(args.seed),
                normalization_mode=normalization_mode,
                overwrite=args.overwrite,
            )
        summaries["checkpoints"][role] = summary
        del sampler
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    save_json(output_dir / "summary.json", summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
