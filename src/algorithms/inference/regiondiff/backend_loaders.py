"""RegionDiff synthetic generation backend loader helpers."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler
from src.algorithms.inference.layout_flow_matching_sampler import LayoutFlowMatchingSampler
from src.algorithms.inference.unconditional_sd_sampler import UnconditionalStableDiffusionSampler
from src.core.diffusers_compat import import_diffusers_attr
from src.models.fm_unet import build_fm_unet_from_config, load_unet_config
from src.models.regiondiffusion_factory import build_regiondiff_wrapper
from src.models.stay_layout_conditioned_unet import build_stay_layout_conditioned_unet
from src.models.vae import build_vae_from_config, freeze_vae, load_diffusers_vae_config, load_vae_config, load_vae_weights

from .dataset_io import (
    STAGE2_CHECKPOINT_UNET_WEIGHTS_NAME,
    STAGE2_LAYOUT_MANIFEST_NAME,
    STAGE2_REGIONDIFF_CONFIG_NAME,
    STAGE2_UNET_WEIGHTS_NAME,
    _load_yaml,
    _normalise_names,
    _repo_path,
)


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


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"_epoch_(\d+)\.pt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _resolve_unet_checkpoint_path(
    checkpoint_path: str | Path,
    *,
    preferred_names: Sequence[str],
) -> Path:
    path = _repo_path(checkpoint_path)
    if path is None:
        raise FileNotFoundError(f"Missing checkpoint path: {checkpoint_path}")
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    search_dirs = [path / "UNET", path]
    for search_dir in search_dirs:
        for name in preferred_names:
            candidate = search_dir / name
            if candidate.is_file():
                return candidate

    candidates: list[Path] = []
    for search_dir in search_dirs:
        if search_dir.is_dir():
            candidates.extend(sorted(search_dir.glob("*.pt")))
            candidates.extend(sorted(search_dir.glob("*.safetensors")))
    if not candidates:
        raise FileNotFoundError(
            f"No UNET checkpoint file found under {path}. Expected one of "
            f"{', '.join(preferred_names)} or an epoch checkpoint in UNET/."
        )

    candidates.sort(key=lambda item: (_checkpoint_epoch(item), item.name))
    return candidates[-1]


def _find_unet_checkpoint_in_dir(path: Path) -> Path | None:
    for search_dir in (path / "UNET", path):
        if not search_dir.is_dir():
            continue
        candidates = list(search_dir.glob("*.pt")) + list(search_dir.glob("*.safetensors"))
        if candidates:
            candidates.sort(key=lambda item: (_checkpoint_epoch(item), item.name))
            return candidates[-1]
    return None


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
        unet_checkpoint = _find_unet_checkpoint_in_dir(path)
        if unet_checkpoint is not None:
            path = unet_checkpoint
        else:
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
    checkpoint_path = _resolve_unet_checkpoint_path(
        generator_cfg["checkpoint_path"],
        preferred_names=("unet_fm_best.pt",),
    )
    preset_path = _repo_path(generator_cfg["preset_path"])
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
    checkpoint_path = _resolve_unet_checkpoint_path(
        generator_cfg["checkpoint_path"],
        preferred_names=("unet_fm_best.pt",),
    )
    preset_path = _repo_path(generator_cfg["preset_path"])
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
    checkpoint_path = _resolve_unet_checkpoint_path(
        generator_cfg["checkpoint_path"],
        preferred_names=("unet_sd_uncond_best.pt",),
    )
    preset_path = _repo_path(generator_cfg["preset_path"])
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

__all__ = [name for name in globals() if not name.startswith("__")]
