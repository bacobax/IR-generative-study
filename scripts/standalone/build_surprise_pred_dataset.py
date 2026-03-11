#!/usr/bin/env python3
"""
build_surprise_pred_dataset.py

Builds a dataset folder with the following structure:

  DS_ROOT/
    [vae_model_name]/
      latents/{stem}.npy          ← VAE-encoded latent (float32 numpy)
      config.json                 ← build provenance + CLI args
      annotations.json            ← per-stem record with surprise + GMM scores
      gmm.pkl                     ← fitted GaussianMixture (sklearn, pickle)
    clusters/
      centers_{dino_name}_k{n_clusters}.pt
      maps_{dino_name}_k{n_clusters}/{stem}_clusters.npy

Pipeline:
  Step 1 – Fit global MiniBatchKMeans cluster centres on DINOv2 patch tokens;
            assign per-image cluster maps.
  Step 2 – Encode each image through the VAE using exactly the same path as
            StableFlowMatchingPipeline.encode_fm_input().
  Step 3 – Load a trained MaskedClusterModel checkpoint; compute
            typicality / surprise scores and semantic embeddings.
  Step 4 – Fit a GaussianMixture on the semantic embeddings; compute
            log-likelihood scores.
  Step 5 – Join everything by stem; write annotations.json + config.json.

Reuse policy:
  • DINO logic  →  copied verbatim from fm_src/notebooks/dinov2_study.ipynb
  • VAE encode  →  StableFlowMatchingPipeline (fm_src/pipelines/flow_matching_pipeline.py)
  • Typicality / embeddings  →  train_cluster_reconstruction.py
"""

import argparse
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.configs.config_loader import apply_yaml_defaults
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── add repo root to sys.path ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── project imports ───────────────────────────────────────────────────────────
from fm_src.pipelines.flow_matching_pipeline import StableFlowMatchingPipeline  # noqa: E402
from src.core.constants import (  # noqa: E402
    P0001_PERCENTILE_RAW_IMAGES,
    P9999_PERCENTILE_RAW_IMAGES,
    RAW_RANGE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.core.normalization import raw_to_norm as _to_sd_tensor  # noqa: E402
from src.core.normalization import resize_and_normalize_256 as transform_256_vae  # noqa: E402
from src.core.data.datasets import NPYStemDataset  # noqa: E402
from train_cluster_reconstruction import (  # noqa: E402
    ClusterReconDataset,
    MaskedClusterModel,
    compute_typicality_scores,
    extract_semantic_embeddings,
)

# ── sklearn optional ──────────────────────────────────────────────────────────
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.mixture import GaussianMixture
    _SKLEARN_OK = True
except ImportError as _e:
    _SKLEARN_OK = False
    _SKLEARN_ERR = str(_e)


# =============================================================================
# Normalisation constants (from src.core.constants)
# =============================================================================

_A = P0001_PERCENTILE_RAW_IMAGES
_B = P9999_PERCENTILE_RAW_IMAGES
_S = RAW_RANGE


# =============================================================================
# Dataset helpers
# =============================================================================

def _collate_stem(batch):
    """Collate (tensor, stem) pairs into (Tensor[B,...], List[str])."""
    tensors, stems = zip(*batch)
    return torch.stack(tensors, dim=0), list(stems)


def _collate_cluster(batch):
    """Collate ClusterReconDataset dicts."""
    lbl = torch.stack([b["cluster_lbl"] for b in batch])
    return {"cluster_lbl": lbl, "stem": [b["stem"] for b in batch]}


# =============================================================================
# Image transforms  (copied from fm_src/notebooks/dinov2_study.ipynb)
# =============================================================================

def transform_256_dino(x: torch.Tensor) -> torch.Tensor:
    """
    Resize to 256×256 + per-image min/max → [-1, 1].
    Copied from fm_src/notebooks/dinov2_study.ipynb (transform_256).
    """
    x = F.interpolate(
        x.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
    ).squeeze(0)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    x = 2.0 * x - 1.0
    return x


# =============================================================================
# DINO helpers   (copied verbatim from fm_src/notebooks/dinov2_study.ipynb)
# =============================================================================

# Module-level cache so stats tensors are not re-created on every call.
_IMNET_MEAN: Optional[torch.Tensor] = None
_IMNET_STD: Optional[torch.Tensor] = None


def _get_imnet_stats(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    global _IMNET_MEAN, _IMNET_STD
    if _IMNET_MEAN is None or _IMNET_MEAN.device.type != device.split(":")[0]:
        _IMNET_MEAN = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        _IMNET_STD  = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return _IMNET_MEAN, _IMNET_STD


@torch.no_grad()
def to_dino_input(x_b1hw: torch.Tensor, device: str) -> torch.Tensor:
    """
    Copied from fm_src/notebooks/dinov2_study.ipynb (to_dino_input).
    x_b1hw: (B,1,H,W) in [-1, 1] → ImageNet-normalised (B,3,224,224).
    """
    imnet_mean, imnet_std = _get_imnet_stats(device)
    x = (x_b1hw + 1.0) * 0.5                                          # [0,1]
    x = x.repeat(1, 3, 1, 1)                                          # 3-channel
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - imnet_mean) / imnet_std
    return x


@torch.no_grad()
def extract_patch_tokens(
    x_b1hw: torch.Tensor,
    dino: nn.Module,
    device: str,
) -> torch.Tensor:
    """
    Copied from fm_src/notebooks/dinov2_study.ipynb (extract_patch_tokens).
    Returns patch tokens of shape (B, N, D).
    For ViT-S/14 at 224px: N = 16×16 = 256, D = 384.
    """
    x = to_dino_input(x_b1hw.to(device), device)
    out = dino.forward_features(x)
    return out["x_norm_patchtokens"]          # (B, N, D)


@torch.no_grad()
def assign_clusters(
    pt_bnd: torch.Tensor,
    centers_kd: torch.Tensor,
) -> torch.Tensor:
    """
    Copied from fm_src/notebooks/dinov2_study.ipynb (assign_clusters).
    pt_bnd: (B, N, D);  centers_kd: (k, D) → ids: (B, N) cluster indices.
    """
    B, N, D = pt_bnd.shape
    X = pt_bnd.reshape(B * N, D)
    d = torch.cdist(X, centers_kd)           # (B*N, k)
    return torch.argmin(d, dim=1).reshape(B, N)


# =============================================================================
# Step 1 – build global clusters + per-image maps
# =============================================================================

def build_clusters(
    args: argparse.Namespace,
    img_dir: Path,
    clusters_root: Path,
    device: str,
    stems_ordered: List[str],
) -> Path:
    """
    1a) Fit MiniBatchKMeans on DINOv2 patch tokens.
    1b) Assign each image to nearest centre and store a (grid×grid) int16
        cluster map.

    Returns:
        maps_dir: path to DS_ROOT/clusters/maps_{dino_name}_k{n_clusters}/
    """
    if not _SKLEARN_OK:
        raise RuntimeError(f"scikit-learn is required for clustering: {_SKLEARN_ERR}")

    centers_path = clusters_root / f"centers_{args.dino_name}_k{args.n_clusters}.pt"
    maps_dir = clusters_root / f"maps_{args.dino_name}_k{args.n_clusters}"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # ── load DINOv2 ──────────────────────────────────────────────────────────
    print(f"\n[step 1] Loading DINOv2 model: {args.dino_name} ...")
    dino: nn.Module = torch.hub.load("facebookresearch/dinov2", args.dino_name)
    dino.eval().to(device)

    # ── 1a: fit MiniBatchKMeans ───────────────────────────────────────────────
    if centers_path.exists() and not args.overwrite_clusters:
        print(f"[step 1a] Cluster centres already exist at {centers_path}. Skipping KMeans fit.")
    else:
        print(f"[step 1a] Fitting MiniBatchKMeans (k={args.n_clusters}) ...")

        dino_ds = NPYStemDataset(
            root_dir=str(img_dir),
            transform=transform_256_dino,
            stem_list=stems_ordered,
        )
        dino_loader = DataLoader(
            dino_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_stem,
        )

        mbk = MiniBatchKMeans(
            n_clusters=args.n_clusters,
            batch_size=8192,
            random_state=args.seed,
            verbose=0,
            max_iter=100,
            n_init="auto",
        )

        total_seen = 0
        for x_b1hw, _ in tqdm(dino_loader, desc="  KMeans partial_fit"):
            x_b1hw = x_b1hw.to(device)
            pt = extract_patch_tokens(x_b1hw, dino, device)   # (B, N, D)
            B, N, D = pt.shape
            X = pt.float().cpu().reshape(B * N, D).numpy()
            mbk.partial_fit(X)
            total_seen += B * N

        print(f"  Total patch vectors seen: {total_seen:,}")
        centers = torch.from_numpy(mbk.cluster_centers_).float()  # (k, D)
        torch.save(
            {"dino_name": args.dino_name, "k": args.n_clusters, "centers": centers},
            centers_path,
        )
        print(f"  Saved centres → {centers_path}")

    # ── 1b: assign cluster maps ───────────────────────────────────────────────
    stems_needing = [
        s for s in stems_ordered
        if not (maps_dir / f"{s}_clusters.npy").exists() or args.overwrite_clusters
    ]

    if not stems_needing:
        print("[step 1b] All cluster maps already exist. Skipping.")
    else:
        print(f"[step 1b] Assigning cluster maps for {len(stems_needing)} stems ...")

        ckpt = torch.load(centers_path, map_location="cpu")
        centers_dev = ckpt["centers"].to(device)

        map_ds = NPYStemDataset(
            root_dir=str(img_dir),
            transform=transform_256_dino,
            stem_list=stems_needing,
        )
        map_loader = DataLoader(
            map_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate_stem,
        )

        for x_b1hw, batch_stems in tqdm(map_loader, desc="  Assigning cluster maps"):
            x_b1hw = x_b1hw.to(device)
            pt = extract_patch_tokens(x_b1hw, dino, device)        # (B, N, D)
            ids = assign_clusters(pt, centers_dev)                  # (B, N)

            N = ids.shape[1]
            grid = int(math.sqrt(N))
            assert grid * grid == N, (
                f"Patch count N={N} is not a perfect square; "
                f"cannot reshape to (grid, grid). Check DINOv2 input resolution."
            )
            ids_grid = (
                ids.reshape(-1, grid, grid).detach().cpu().to(torch.int16).numpy()
            )

            for i, stem in enumerate(batch_stems):
                np.save(str(maps_dir / f"{stem}_clusters.npy"), ids_grid[i])

        print(f"  Saved cluster maps → {maps_dir}")

    # Free GPU memory used by DINO
    del dino
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return maps_dir


# =============================================================================
# Step 2 – extract VAE latents
# =============================================================================

def extract_latents(
    args: argparse.Namespace,
    img_dir: Path,
    latents_dir: Path,
    stems: List[str],
    device: str,
) -> None:
    """
    Encode each image using StableFlowMatchingPipeline.encode_fm_input(),
    which calls vae.encode() + vae.sampling() — identical to the FM training
    pipeline.  Images are preprocessed with transform_256_vae (percentile-based
    normalisation, same as train_sfm.py).
    """
    stems_needing = [
        s for s in stems
        if not (latents_dir / f"{s}.npy").exists() or args.overwrite_latents
    ]

    if not stems_needing:
        print("\n[step 2] All latents already exist. Skipping.")
        return

    print(f"\n[step 2] Extracting VAE latents for {len(stems_needing)} stems ...")

    # ── build pipeline ────────────────────────────────────────────────────────
    pipeline = StableFlowMatchingPipeline(device=device)
    pipeline.build_from_configs(vae_json=args.vae_config, save_configs=False)
    pipeline.load_pretrained(vae_path=args.vae_weights, set_eval=True)
    pipeline.freeze_vae()

    # ── dataset ───────────────────────────────────────────────────────────────
    ds = NPYStemDataset(
        root_dir=str(img_dir),
        transform=transform_256_vae,
        stem_list=stems_needing,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_stem,
    )

    latents_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for x_batch, batch_stems in tqdm(loader, desc="  Encoding latents"):
            x_batch = x_batch.to(device)
            # encode_fm_input = vae.encode(x) + vae.sampling(mu, sigma)
            z = pipeline.encode_fm_input(x_batch)    # (B, C, H', W')
            for i, stem in enumerate(batch_stems):
                np.save(
                    str(latents_dir / f"{stem}.npy"),
                    z[i].cpu().float().numpy(),
                )

    print(f"  Saved latents → {latents_dir}")

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Step 3 – surprise scores, semantic embeddings, GMM
# =============================================================================

def compute_scores(
    args: argparse.Namespace,
    maps_dir: Path,
    img_dir: Path,
    vae_out_dir: Path,
    stems: List[str],
    device: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Loads the MaskedClusterModel checkpoint and:
      3a) extracts per-sample semantic embeddings
      3b) optionally computes typicality (surprise) scores
      3c) fits a GaussianMixture on embeddings, stores log-likelihood scores

    Returns:
        surp_by_stem:  stem → mean(-log p)  [NaN if --eval_typicality not set]
        gmm_by_stem:   stem → GMM log-likelihood
    """
    if not _SKLEARN_OK:
        raise RuntimeError(f"scikit-learn is required for GMM: {_SKLEARN_ERR}")

    print(f"\n[step 3] Loading MaskedClusterModel: {args.masked_model_ckpt}")
    ckpt = torch.load(args.masked_model_ckpt, map_location=device)

    k_regions  = int(ckpt["k_regions"])
    mask_id    = int(ckpt.get("mask_id",    k_regions))
    emb_dim    = int(ckpt["emb_dim"])
    hidden_dim = int(ckpt["hidden_dim"])

    model = MaskedClusterModel(
        k_regions=k_regions, emb_dim=emb_dim, hidden=hidden_dim
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  k_regions={k_regions}  emb_dim={emb_dim}  hidden_dim={hidden_dim}  mask_id={mask_id}")

    # ClusterReconDataset uses cluster_maps_dir and stems; img_dir is stored
    # but not accessed in __getitem__.
    cluster_ds = ClusterReconDataset(
        img_dir=img_dir,
        cluster_maps_dir=maps_dir,
        stems=stems,
        k_regions=k_regions,
    )
    loader = DataLoader(
        cluster_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_cluster,
    )

    # ── 3a: semantic embeddings ───────────────────────────────────────────────
    print("[step 3a] Extracting semantic embeddings ...")
    emb_arr, emb_stems = extract_semantic_embeddings(loader, model, device)
    emb_by_stem: Dict[str, np.ndarray] = {
        s: emb_arr[i] for i, s in enumerate(emb_stems)
    }

    # ── 3b: typicality / surprise ─────────────────────────────────────────────
    if args.eval_typicality:
        print("[step 3b] Computing typicality (surprise) scores ...")
        surp_arr, surp_stems = compute_typicality_scores(
            loader=loader,
            model=model,
            device=device,
            mask_id=mask_id,
            typicality_chunk=args.typicality_chunk,
            max_batches=args.typicality_max_batches,
        )
        surp_by_stem: Dict[str, float] = {
            s: float(surp_arr[i]) for i, s in enumerate(surp_stems)
        }
    else:
        print("[step 3b] Skipping typicality (pass --eval_typicality to enable). "
              "surprise_raw/minmax will be null in annotations.json.")
        surp_by_stem = {s: float("nan") for s in stems}

    # ── 3c: GaussianMixture ───────────────────────────────────────────────────
    print("[step 3c] Fitting GaussianMixture ...")
    valid_stems = [s for s in stems if s in emb_by_stem]
    if not valid_stems:
        raise RuntimeError("No valid semantic embeddings found. Cannot fit GMM.")

    emb_matrix = np.stack([emb_by_stem[s] for s in valid_stems], axis=0)  # (N, D)

    gmm = GaussianMixture(
        n_components=args.gmm_n_components,
        covariance_type=args.gmm_covariance_type,
        random_state=args.seed,
        max_iter=200,
    )
    gmm.fit(emb_matrix)
    gmm_scores_raw = gmm.score_samples(emb_matrix)   # (N,) log-likelihood
    gmm_by_stem: Dict[str, float] = {
        s: float(gmm_scores_raw[i]) for i, s in enumerate(valid_stems)
    }

    # Save the fitted GMM so it can be loaded at inference time
    gmm_path = vae_out_dir / "gmm.pkl"
    with open(str(gmm_path), "wb") as f:
        pickle.dump(gmm, f, protocol=4)
    print(f"  Saved GMM → {gmm_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return surp_by_stem, gmm_by_stem


# =============================================================================
# Step 4 – assemble + write annotations.json
# =============================================================================

def _minmax_normalize(v: float, lo: float, hi: float) -> Optional[float]:
    if not (lo == lo) or not (v == v):   # NaN check via self-comparison
        return None
    span = hi - lo
    if span < 1e-12:
        return 0.0
    return float((v - lo) / span)


def save_annotations(
    args: argparse.Namespace,
    vae_out_dir: Path,
    latents_dir: Path,
    maps_dir: Path,
    stems: List[str],
    surp_by_stem: Dict[str, float],
    gmm_by_stem: Dict[str, float],
) -> None:
    """
    Computes dataset-level min/max for surprise and GMM scores and writes
    annotations.json with per-stem records.
    """
    # Collect finite values for dataset-level stats
    surp_vals = np.array(
        [surp_by_stem.get(s, float("nan")) for s in stems], dtype=np.float64
    )
    gmm_vals = np.array(
        [gmm_by_stem.get(s, float("nan")) for s in stems], dtype=np.float64
    )

    finite_surp = surp_vals[np.isfinite(surp_vals)]
    finite_gmm  = gmm_vals[np.isfinite(gmm_vals)]

    surp_min = float(finite_surp.min()) if len(finite_surp) > 0 else 0.0
    surp_max = float(finite_surp.max()) if len(finite_surp) > 0 else 1.0
    gmm_min  = float(finite_gmm.min())  if len(finite_gmm)  > 0 else 0.0
    gmm_max  = float(finite_gmm.max())  if len(finite_gmm)  > 0 else 1.0

    records = []
    for stem in stems:
        latent_path = latents_dir / f"{stem}.npy"
        if not latent_path.exists():
            continue   # latent was not produced for this stem — skip

        surp_raw = surp_by_stem.get(stem, float("nan"))
        gmm_raw  = gmm_by_stem.get(stem, float("nan"))

        records.append({
            "stem":             stem,
            "latent_relpath":   f"latents/{stem}.npy",
            # relative from DS_ROOT/vae_model_name/ to clusters/
            "cluster_map_relpath": (
                f"../clusters/maps_{args.dino_name}_k{args.n_clusters}"
                f"/{stem}_clusters.npy"
            ),
            "surprise_raw":     surp_raw     if np.isfinite(surp_raw) else None,
            "surprise_minmax":  _minmax_normalize(surp_raw, surp_min, surp_max),
            "gmm_score_raw":    gmm_raw      if np.isfinite(gmm_raw)  else None,
            "gmm_score_minmax": _minmax_normalize(gmm_raw,  gmm_min,  gmm_max),
        })

    payload = {
        "dataset_stats": {
            "n_samples":        len(records),
            "surprise_min":     surp_min,
            "surprise_max":     surp_max,
            "gmm_score_min":    gmm_min,
            "gmm_score_max":    gmm_max,
            "eval_typicality":  args.eval_typicality,
        },
        "images": records,
    }

    ann_path = vae_out_dir / "annotations.json"
    ann_path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved annotations → {ann_path}  ({len(records)} records)")


# =============================================================================
# Config JSON
# =============================================================================

def save_config(args: argparse.Namespace, vae_out_dir: Path) -> None:
    cfg = {
        "vae_model_name":  args.vae_model_name,
        "vae_config":      str(args.vae_config),
        "vae_weights":     str(args.vae_weights),
        "dino_name":       args.dino_name,
        "n_clusters":      args.n_clusters,
        "masked_model_ckpt": str(args.masked_model_ckpt),
        "data_root":       str(args.data_root),
        "split":           args.split,
        # ── preprocessing details ──────────────────────────────────────────
        "image_size_vae":          256,
        "image_size_dino_input":   256,
        "dino_input_after_transform": 224,
        "vae_normalization":       "percentile_based_clamp_to_sd",
        "vae_normalization_constants": {"A": _A, "B": _B, "S": _S},
        "dino_normalization":      "per_image_min_max_then_imagenet",
        "pipeline_class":          "StableFlowMatchingPipeline",
        "encode_method":           "encode_fm_input → vae.encode + vae.sampling",
        # ── CLI args (full record for reproducibility) ─────────────────────
        "cli_args": vars(args),
    }
    cfg_path = vae_out_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"  Saved config → {cfg_path}")


# =============================================================================
# Argument parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build surprise-prediction dataset (latents + cluster maps + scores).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file. CLI flags override config values.")

    # ── mandatory ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--created_dataset", type=str, required=True,
        help="Path to DS_ROOT — the top-level output directory.",
    )
    p.add_argument(
        "--vae_model_name", type=str, required=True,
        help="Unique name for this VAE; used as the sub-folder name inside DS_ROOT.",
    )
    p.add_argument(
        "--vae_config", type=str, required=True,
        help="Path to the VAE architecture config JSON.",
    )
    p.add_argument(
        "--vae_weights", type=str, required=True,
        help="Path to the pretrained VAE weights (.pt).",
    )
    p.add_argument(
        "--masked_model_ckpt", type=str, required=True,
        help="Path to the MaskedClusterModel checkpoint (.pt).",
    )
    p.add_argument(
        "--n_clusters", type=int, required=True,
        help="Number of semantic cluster regions K.",
    )

    # ── data ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--data_root", type=str, default="./v18",
        help="Root of the image dataset (same convention as train_cluster_reconstruction).",
    )
    p.add_argument(
        "--split", type=str, default="train",
        help="Sub-folder of data_root to process.",
    )
    p.add_argument(
        "--dino_name", type=str, default="dinov2_vits14",
        help="DINOv2 model variant (e.g. dinov2_vits14, dinov2_vitb14).",
    )
    p.add_argument(
        "--max_items", type=int, default=0,
        help="If > 0, only process the first N images (useful for quick debug runs).",
    )

    # ── compute ───────────────────────────────────────────────────────────────
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--device",       type=str, default="auto")
    p.add_argument("--seed",         type=int, default=0)

    # ── typicality / surprise ─────────────────────────────────────────────────
    p.add_argument(
        "--eval_typicality", action="store_true", default=False,
        help=(
            "Compute per-sample typicality (surprise) scores via masked prediction. "
            "Expensive — O(H*W) forward passes per image."
        ),
    )
    p.add_argument(
        "--typicality_chunk", type=int, default=32,
        help=(
            "Number of masked positions processed in one forward pass inside "
            "compute_typicality_scores.  Larger = faster but more VRAM."
        ),
    )
    p.add_argument(
        "--typicality_max_batches", type=int, default=0,
        help="If > 0, limit typicality computation to the first N loader batches.",
    )

    # ── GMM ───────────────────────────────────────────────────────────────────
    p.add_argument(
        "--gmm_n_components", type=int, default=8,
        help="Number of mixture components for the GaussianMixture model.",
    )
    p.add_argument(
        "--gmm_covariance_type", type=str, default="full",
        choices=["full", "tied", "diag", "spherical"],
        help="Covariance type for the GaussianMixture model.",
    )

    # ── overwrite flags ───────────────────────────────────────────────────────
    p.add_argument(
        "--overwrite_clusters", action="store_true", default=False,
        help="Re-fit KMeans and re-assign cluster maps even if they already exist.",
    )
    p.add_argument(
        "--overwrite_latents", action="store_true", default=False,
        help="Re-encode latents even if they already exist on disk.",
    )

    preliminary, _ = p.parse_known_args()
    apply_yaml_defaults(p, preliminary.config)
    return p.parse_args()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args = parse_args()

    # ── seed + device ─────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    print(f"Device: {device}")

    # ── directory layout ──────────────────────────────────────────────────────
    ds_root = Path(args.created_dataset).resolve()
    data_root = Path(args.data_root).resolve()
    img_dir = data_root / args.split

    def _is_subpath(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    # Safety guard: built dataset must live in a different tree than source data.
    if _is_subpath(ds_root, data_root) or _is_subpath(data_root, ds_root):
        raise ValueError(
            "`--created_dataset` must be different from and non-overlapping with "
            "`--data_root`. Choose a separate output folder (e.g. ./generated/... ).\n"
            f"Got created_dataset={ds_root} and data_root={data_root}."
        )

    clusters_root = ds_root / "clusters"
    vae_out_dir  = ds_root / args.vae_model_name
    latents_dir  = vae_out_dir / "latents"

    for d in (ds_root, clusters_root, vae_out_dir, latents_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    # ── discover stems ────────────────────────────────────────────────────────
    all_files = sorted(f for f in os.listdir(str(img_dir)) if f.endswith(".npy"))
    if args.max_items > 0:
        all_files = all_files[: args.max_items]
    if not all_files:
        raise RuntimeError(f"No .npy files found in {img_dir}")

    stems_ordered: List[str] = [Path(f).stem for f in all_files]
    print(f"Total stems discovered: {len(stems_ordered):,}")

    # ── step 1: clusters ──────────────────────────────────────────────────────
    maps_dir = build_clusters(args, img_dir, clusters_root, device, stems_ordered)

    # Only keep stems that actually have a cluster map
    stems = [s for s in stems_ordered if (maps_dir / f"{s}_clusters.npy").exists()]
    n_missing_map = len(stems_ordered) - len(stems)
    if n_missing_map:
        print(f"Warning: {n_missing_map} stem(s) have no cluster map — skipping.")

    # ── step 2: latents ───────────────────────────────────────────────────────
    extract_latents(args, img_dir, latents_dir, stems, device)

    # Only keep stems that also have a latent
    stems = [s for s in stems if (latents_dir / f"{s}.npy").exists()]
    n_missing_lat = len(stems_ordered) - len(stems)
    if n_missing_lat:
        print(f"Warning: {n_missing_lat} stem(s) have no latent — skipping.")

    print(f"\nStems with both cluster map + latent: {len(stems):,}")
    if not stems:
        raise RuntimeError("No complete samples (map + latent) found. "
                           "Check data_root / split path.")

    # ── step 3: scores ────────────────────────────────────────────────────────
    surp_by_stem, gmm_by_stem = compute_scores(
        args, maps_dir, img_dir, vae_out_dir, stems, device
    )

    # ── step 4/5: save ────────────────────────────────────────────────────────
    print()
    save_annotations(args, vae_out_dir, latents_dir, maps_dir, stems, surp_by_stem, gmm_by_stem)
    save_config(args, vae_out_dir)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n✓ Dataset built successfully.")
    print(f"  {ds_root}/")
    print(f"    {args.vae_model_name}/latents/       ← {len(stems)} latent .npy files")
    print(f"    {args.vae_model_name}/config.json")
    print(f"    {args.vae_model_name}/annotations.json")
    print(f"    {args.vae_model_name}/gmm.pkl")
    print(f"    clusters/centers_{args.dino_name}_k{args.n_clusters}.pt")
    print(f"    clusters/maps_{args.dino_name}_k{args.n_clusters}/  "
          f"← {len(stems)} cluster maps")


if __name__ == "__main__":
    main()
