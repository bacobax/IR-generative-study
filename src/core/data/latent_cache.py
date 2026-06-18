"""Disk latent caching for VAE-encoder trainings.

Every VAE-encoder training (flow matching family, SD15, SDXL) re-encodes the same
images through the frozen VAE on every epoch. The encode is ``no_grad`` and, for a
fixed input, deterministic, so it is pure wasted GPU time and memory.

This module provides a persistent disk cache keyed by the 4-tuple
``<VAE, dataset, augmentation, normalization_method>``. On the first run for a
given tuple, every base image plus its enumerated augmentation variants is encoded
once and the *scaled posterior* ``(mu, sigma)`` is written to disk. Later runs with
the same tuple load the latents and skip the VAE entirely (it need not even be
loaded onto the GPU).

Design notes
------------
* We store the posterior ``mu`` and ``sigma`` (already multiplied by the VAE
  ``scaling_factor`` by the encode adapter), NOT a pre-sampled latent. Training
  samples ``z = mu + sigma * noise`` fresh each step, preserving the stochastic
  posterior regularisation the trainers rely on.
* Augmentation is *materialised*: instead of a probabilistic per-epoch transform,
  each base sample is expanded into a fixed, finite set of deterministic variants
  (e.g. ``[identity, hflip]``) that are all encoded and cached. The augmentation
  combo is part of the cache key. Continuous/non-enumerable transforms (random
  crop, free rotation) are rejected when caching is enabled.
* Normalization method is a first-class component of the key: latents encoded
  under ``per_image_minmax`` are never reused for a ``raw_uint16_percentile`` run
  and vice-versa.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.core import paths

CACHE_FORMAT_VERSION = 1
# Files larger than this are fingerprinted by (size, mtime) instead of full hash.
_LARGE_FILE_BYTES = 256 * 1024 * 1024


# ── Cache-key construction ────────────────────────────────────────────────────


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Optional[str]) -> Optional[dict]:
    """Stable fingerprint of a file used in the cache key.

    Small files are hashed by content (robust to path moves); large files fall
    back to ``(size, mtime)`` to avoid hashing multi-hundred-MB checkpoints.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return {"name": p.name, "exists": False}
    st = p.stat()
    fp: dict[str, Any] = {"name": p.name, "size": int(st.st_size)}
    if st.st_size <= _LARGE_FILE_BYTES:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        fp["sha256"] = h.hexdigest()
    else:
        fp["mtime"] = int(st.st_mtime)
    return fp


def build_vae_identity(model_cfg: Any) -> dict:
    """Identity dict for the VAE selected by a training model config."""
    return {
        "pretrained_model_name_or_path": getattr(
            model_cfg, "vae_pretrained_model_name_or_path", None
        ),
        "pretrained_subfolder": getattr(model_cfg, "vae_pretrained_subfolder", None),
        "revision": getattr(model_cfg, "vae_revision", None),
        "variant": getattr(model_cfg, "vae_variant", None),
        "vae_config": _file_fingerprint(getattr(model_cfg, "vae_config", None)),
        "vae_weights": _file_fingerprint(getattr(model_cfg, "vae_weights", None)),
    }


def build_dataset_identity(
    *,
    dataset_id: Optional[str],
    train_dir: Optional[str],
    val_dir: Optional[str],
    split: str,
    image_size: int,
    subset_manifest: Optional[str],
) -> dict:
    """Identity dict for the dataset + split feeding the cache.

    When a named ``dataset_id`` is present it pins identity; otherwise the
    absolute split directories are used so ad-hoc path-based datasets still get a
    stable, distinct key.
    """
    return {
        "dataset_id": dataset_id,
        "train_dir": None if dataset_id else (os.path.abspath(train_dir) if train_dir else None),
        "val_dir": None if dataset_id else (os.path.abspath(val_dir) if val_dir else None),
        "split": split,
        "image_size": int(image_size),
        "subset_manifest": subset_manifest,
    }


def build_cache_key(
    *,
    vae_identity: dict,
    dataset_identity: dict,
    normalization_mode: str,
    aug_combo: dict,
) -> str:
    """Deterministic short key for the ``<VAE, dataset, aug, normalization>`` tuple."""
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "vae": vae_identity,
        "dataset": dataset_identity,
        "normalization": normalization_mode,
        "augmentation": aug_combo,
    }
    return _sha256_text(_stable_json(payload))[:16]


# ── Augmentation enumeration (materialisation) ────────────────────────────────


@dataclass(frozen=True)
class AugVariant:
    """A single deterministic augmentation variant of a base sample."""

    name: str
    hflip: bool = False


def _max_prob(augment_config: Any, *names: str) -> float:
    if augment_config is None:
        return 0.0
    return max(
        (float(getattr(augment_config, n, 0.0) or 0.0) for n in names),
        default=0.0,
    )


def enumerate_aug_variants(augment_config: Any) -> list[AugVariant]:
    """Turn a probabilistic augment config into a finite list of variants.

    Only horizontal flip is materialisable today: it maps to ``[identity, hflip]``.
    Crop/rotation are continuous/non-enumerable and are rejected so the user
    explicitly disables them (or disables caching) rather than getting a silently
    wrong, frozen augmentation.
    """
    crop_on = _max_prob(augment_config, "p_crop_warmup", "p_crop_max", "p_crop_final") > 0.0
    rot_on = _max_prob(augment_config, "p_rot_warmup", "p_rot_max", "p_rot_final") > 0.0
    if crop_on or rot_on:
        raise ValueError(
            "Latent caching does not support crop/rotation augmentation "
            "(non-enumerable continuous transforms). Disable crop/rotation in the "
            "augment config or disable latent_cache."
        )
    hflip_on = _max_prob(augment_config, "p_hflip_warmup", "p_hflip_max", "p_hflip_final") > 0.0
    variants = [AugVariant(name="id")]
    if hflip_on:
        variants.append(AugVariant(name="hflip", hflip=True))
    return variants


def aug_combo_descriptor(variants: Sequence[AugVariant]) -> dict:
    return {"variants": [v.name for v in variants]}


def apply_variant(sample: Any, variant: AugVariant, *, image_size: int) -> dict:
    """Apply a deterministic variant to a sample, returning a dict.

    Bare-tensor samples (non-layout datasets) are wrapped as
    ``{"pixel_values": tensor}``. Horizontal flip mirrors both the image and any
    ``boxes_xyxy`` so layout annotations stay aligned (same formula as
    :class:`AnnotationLayoutDataset`).
    """
    if isinstance(sample, torch.Tensor):
        out: dict[str, Any] = {"pixel_values": sample}
    else:
        out = dict(sample)

    if variant.hflip:
        out["pixel_values"] = torch.flip(out["pixel_values"], dims=[-1])
        boxes = out.get("boxes_xyxy")
        if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
            flipped = boxes.clone()
            flipped[:, 0] = float(image_size) - boxes[:, 2]
            flipped[:, 2] = float(image_size) - boxes[:, 0]
            out["boxes_xyxy"] = flipped
        out["horizontal_flipped"] = True
    return out


# ── Disk I/O helpers ──────────────────────────────────────────────────────────


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_json(obj: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(_stable_json(obj))
    os.replace(tmp, path)


def _cache_complete(cache_dir: Path, expected_entries: int) -> bool:
    meta_path = cache_dir / "meta.json"
    manifest_path = cache_dir / "manifest.json"
    if not (meta_path.exists() and manifest_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(
        meta.get("complete")
        and meta.get("format_version") == CACHE_FORMAT_VERSION
        and meta.get("n_entries") == expected_entries
    )


# ── Precompute pass ───────────────────────────────────────────────────────────


@torch.no_grad()
def precompute_latents(
    *,
    base_dataset: Dataset,
    vae,
    device: Any,
    variants: Sequence[AugVariant],
    cache_dir: Path,
    image_size: int,
    store_dtype: str = "fp16",
    rebuild: bool = False,
    key_payload: Optional[dict] = None,
    desc: str = "latents",
) -> Path:
    """Encode every (base sample × variant) once and write latents to ``cache_dir``.

    Idempotent and resumable: existing per-entry files are skipped unless
    ``rebuild`` is set. Writes ``manifest.json`` (ordered entry list) and
    ``meta.json`` (completion marker + human-readable key) on success.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_base = len(base_dataset)
    n_var = len(variants)
    expected_entries = n_base * n_var

    if _cache_complete(cache_dir, expected_entries) and not rebuild:
        return cache_dir

    dtype = torch.float16 if store_dtype == "fp16" else torch.float32
    vae.eval()
    entries: list[str] = []
    for base_idx in tqdm(range(n_base), desc=f"Caching {desc}"):
        sample = base_dataset[base_idx]
        for var_idx, variant in enumerate(variants):
            global_idx = base_idx * n_var + var_idx
            entry_name = f"entry_{global_idx:07d}.pt"
            entry_path = cache_dir / entry_name
            entries.append(entry_name)
            if entry_path.exists() and not rebuild:
                continue

            v_sample = apply_variant(sample, variant, image_size=image_size)
            pixel_values = v_sample["pixel_values"]
            x = pixel_values.unsqueeze(0).to(device)
            mu, sigma = vae.encode(x)

            payload = dict(v_sample)
            # Keep pixel_values (cheap vs the GPU encode we skip) so visualization,
            # ground-truth display and the existing collate work unchanged.
            payload["pixel_values"] = pixel_values.to(dtype)
            payload["latent_mu"] = mu.squeeze(0).to("cpu", dtype)
            payload["latent_sigma"] = sigma.squeeze(0).to("cpu", dtype)
            payload["image_size"] = int(image_size)
            payload["variant"] = variant.name
            _atomic_save(payload, entry_path)

    _atomic_write_json(
        {
            "entries": entries,
            "n_base": n_base,
            "n_variants": n_var,
            "variants": [v.name for v in variants],
        },
        cache_dir / "manifest.json",
    )
    _atomic_write_json(
        {
            "format_version": CACHE_FORMAT_VERSION,
            "n_entries": expected_entries,
            "store_dtype": store_dtype,
            "complete": True,
            "key": key_payload,
        },
        cache_dir / "meta.json",
    )
    return cache_dir


# ── Cache-backed dataset ──────────────────────────────────────────────────────


class LatentCacheDataset(Dataset):
    """Serves cached posterior ``(mu, sigma)`` + carried annotation fields.

    Replaces ``pixel_values`` with ``latent_mu`` / ``latent_sigma`` (restored to
    float32). Carries through all other fields written during precompute (boxes,
    labels, file names, ...). ``set_epoch`` is a no-op because augmentation is
    already materialised into the expanded entry set.
    """

    def __init__(self, cache_dir: Any):
        self.cache_dir = Path(cache_dir)
        manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.entries: list[str] = list(manifest["entries"])
        self.n_variants: int = int(manifest.get("n_variants", 1))

    def __len__(self) -> int:
        return len(self.entries)

    def set_epoch(self, epoch: int) -> None:  # noqa: D401 - augmentation materialised
        return None

    def __getitem__(self, idx: int) -> dict:
        payload = torch.load(self.cache_dir / self.entries[idx], map_location="cpu")
        payload["latent_mu"] = payload["latent_mu"].float()
        payload["latent_sigma"] = payload["latent_sigma"].float()
        if "pixel_values" in payload:
            payload["pixel_values"] = payload["pixel_values"].float()
        return payload


# ── Orchestration ─────────────────────────────────────────────────────────────


def _build_frozen_vae(model_cfg: Any, device: Any, strict_load: bool):
    from src.models.vae import (
        build_vae_from_config,
        freeze_vae,
        load_vae_weights,
        resolve_vae_config_from_model_config,
    )

    vae_cfg = resolve_vae_config_from_model_config(model_cfg)
    vae = build_vae_from_config(vae_cfg, device=device)
    weights = getattr(model_cfg, "vae_weights", None)
    if weights:
        load_vae_weights(vae, weights, strict=strict_load, map_location=device)
    return freeze_vae(vae)


def resolve_cache_dir(
    *,
    model_cfg: Any,
    dataset_id: Optional[str],
    train_dir: Optional[str],
    val_dir: Optional[str],
    split: str,
    image_size: int,
    subset_manifest: Optional[str],
    normalization_mode: str,
    variants: Sequence[AugVariant],
    cache_root: Optional[str],
) -> tuple[Path, dict]:
    """Compute the cache directory + human-readable key payload for a run."""
    vae_identity = build_vae_identity(model_cfg)
    dataset_identity = build_dataset_identity(
        dataset_id=dataset_id,
        train_dir=train_dir,
        val_dir=val_dir,
        split=split,
        image_size=image_size,
        subset_manifest=subset_manifest,
    )
    aug_combo = aug_combo_descriptor(variants)
    key = build_cache_key(
        vae_identity=vae_identity,
        dataset_identity=dataset_identity,
        normalization_mode=normalization_mode,
        aug_combo=aug_combo,
    )
    root = Path(cache_root) if cache_root else (paths.cache_root() / "latents")
    cache_dir = root / key / split
    key_payload = {
        "vae": vae_identity,
        "dataset": dataset_identity,
        "normalization": normalization_mode,
        "augmentation": aug_combo,
    }
    return cache_dir, key_payload


def build_latent_cache_dataset(
    *,
    base_dataset: Dataset,
    model_cfg: Any,
    dataset_id: Optional[str],
    train_dir: Optional[str],
    val_dir: Optional[str],
    split: str,
    image_size: int,
    subset_manifest: Optional[str],
    normalization_mode: str,
    augment_config: Any,
    latent_cache_cfg: Any,
    device: Any,
    strict_load: bool = True,
) -> LatentCacheDataset:
    """Build (or reuse) the disk latent cache and return a cache-backed dataset.

    The VAE is only constructed/loaded when the cache is incomplete — a warm cache
    never touches the VAE, which is the whole point.
    """
    variants = enumerate_aug_variants(augment_config)
    cache_dir, key_payload = resolve_cache_dir(
        model_cfg=model_cfg,
        dataset_id=dataset_id,
        train_dir=train_dir,
        val_dir=val_dir,
        split=split,
        image_size=image_size,
        subset_manifest=subset_manifest,
        normalization_mode=normalization_mode,
        variants=variants,
        cache_root=getattr(latent_cache_cfg, "cache_root", None),
    )
    expected_entries = len(base_dataset) * len(variants)
    rebuild = bool(getattr(latent_cache_cfg, "rebuild", False))

    if rebuild or not _cache_complete(cache_dir, expected_entries):
        print(
            f"[latent_cache] building cache for split={split} "
            f"({len(base_dataset)} base × {len(variants)} variants) -> {cache_dir}"
        )
        vae = _build_frozen_vae(model_cfg, device, strict_load)
        try:
            precompute_latents(
                base_dataset=base_dataset,
                vae=vae,
                device=device,
                variants=variants,
                cache_dir=cache_dir,
                image_size=image_size,
                store_dtype=getattr(latent_cache_cfg, "store_dtype", "fp16"),
                rebuild=rebuild,
                key_payload=key_payload,
                desc=f"{split} latents",
            )
        finally:
            del vae
            from src.core.training_utils import release_cuda_cache

            release_cuda_cache()
    else:
        print(f"[latent_cache] reusing warm cache for split={split} -> {cache_dir}")

    return LatentCacheDataset(cache_dir)
