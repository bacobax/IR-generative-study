#!/usr/bin/env python3
"""
analyze_fm_subsampling_coverage.py

Coverage/diversity analysis for Flow-Matching generated samples.

Scores used for subsampling:
  scoreA = pred_surprise          (from SurprisePredictor)
  scoreB = 1 - pred_gmm           (from SurprisePredictor)

Five subset strategies per score type:
  U      – uniform random K (no replacement)
  Splus  – weighted by s_i' = clamp(score_norm, eps, 1)       [prefers HIGH score]
  Sminus – weighted by (1 - s_i') = clamp(1-score_norm, eps, 1) [prefers LOW score]
  Tplus  – uniform from top ceil(top_pct*N) pool by score
  Tminus – uniform from bottom ceil(top_pct*N) pool by score

Per-sample cache layout (NO giant arrays):
  OUT_DIR/
    cache/
      gen/
        z/{idx:07d}.npy          single latent float32 (C,H,W)
        x/{idx:07d}.npy          decoded image float32 (1,256,256) in [-1,1]
        scores/{idx:07d}.json    {"surprise":..., "gmm":..., "one_minus_gmm":...}
        meta.json
      tokens/                    only if --cache_tokens
        gen/{idx:07d}.npy        pooled DINO feat float32 (D,)
        real/{stem}.npy          pooled DINO feat float32 (D,)
    subsets/
      scoreA_surprise/           indices_*.npy + metrics.json
      scoreB_one_minus_gmm/      indices_*.npy + metrics.json
    figures/
      *.png

Real images are NEVER written/cached as images; only optional DINO token
caching for real images is allowed (keyed by stem).

Usage:
  python analyze_fm_subsampling_coverage.py \\
      --out_dir ./analysis_out \\
      --fm_pipeline_dir ./serious_runs/stable_training_t_scaled \\
      --vae_weights ./fm_src/vae_best.pt \\
      --surprise_ckpt ./runs/surprise_predictor_longrun/vae_x4_best_minmax_h256_s0/best_model.pt \\
      --real_data_root ./v18 --split train \\
      --N 2000 --K 400 --reuse_cache --cache_tokens
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo-canonical normalization (matches train_sfm.py / train_fm.py / train_vae.py)
# p0.001 and p99.999 percentiles of the raw uint16 sensor image distribution.
# ---------------------------------------------------------------------------
_RAW_A: float = 11667.0   # p0.001 percentile
_RAW_B: float = 13944.0   # p99.999 percentile
_RAW_S: float = _RAW_B - _RAW_A


def to_sd_tensor_and_x(x: torch.Tensor) -> torch.Tensor:
    """
    Repo-canonical raw uint16 → [-1, 1] normalisation.
    Identical to train_sfm.py::to_sd_tensor_and_x.
    """
    return torch.clamp((x.to(torch.float32) - _RAW_A) / _RAW_S, 0.0, 1.0) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# DINO preprocessing — mirrors SurprisePredictor internals exactly
# ---------------------------------------------------------------------------
_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def per_image_minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-image min/max normalisation → [-1, 1].
    (B, C, H, W) → (B, C, H, W).
    Matches SurprisePredictor._per_image_minmax.
    """
    B = x.shape[0]
    flat = x.view(B, -1)
    lo = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    hi = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    x = (x - lo) / (hi - lo + eps)   # → [0, 1]
    return 2.0 * x - 1.0              # → [-1, 1]


def to_dino_input(x_b1hw: torch.Tensor, device: str) -> torch.Tensor:
    """
    (B, 1, H, W) in [-1, 1] → (B, 3, 224, 224) with ImageNet normalisation.
    Matches SurprisePredictor._to_dino_input.
    """
    x = (x_b1hw + 1.0) * 0.5                                                   # [0, 1]
    x = x.expand(-1, 3, -1, -1)                                                # 3-channel
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = _IMNET_MEAN.to(device)
    std  = _IMNET_STD.to(device)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Fréchet distance (FID-like on pooled DINO features)
# ---------------------------------------------------------------------------
def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """
    Matrix square root via scipy.linalg.sqrtm with eigenvalue-based fallback.
    """
    try:
        from scipy.linalg import sqrtm
        result = sqrtm(M)
        if np.iscomplexobj(result):
            result = result.real
        return result
    except Exception:
        # Eigenvalue-based fallback: M = V * diag(λ) * V^T  ⟹  M^½ = V * diag(√λ) * V^T
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0.0)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def frechet_distance(feats_real: np.ndarray, feats_gen: np.ndarray) -> float:
    """
    FID-like Fréchet distance on pooled DINO features.
    dist = ||m1 - m2||² + Tr(C1 + C2 - 2·√(C1·C2))
    """
    m1 = feats_real.mean(axis=0)
    m2 = feats_gen.mean(axis=0)
    C1 = np.cov(feats_real, rowvar=False)
    C2 = np.cov(feats_gen,  rowvar=False)

    diff        = m1 - m2
    sq_prod     = _matrix_sqrt(C1 @ C2)
    trace_term  = float(np.trace(C1) + np.trace(C2) - 2.0 * np.trace(sq_prod))
    dist        = float(diff @ diff) + trace_term
    return dist


# ---------------------------------------------------------------------------
# Pairwise distance (memory-efficient)
# ---------------------------------------------------------------------------
def _cdist_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise squared Euclidean distances using the identity:
        ||a - b||² = ||a||² + ||b||² - 2·a·b

    Only materialises (N_A, N_B), never (N_A, N_B, D).  This is much more
    memory-efficient than the naive broadcast approach.
    """
    A = np.asarray(A, dtype=np.float64)  # float64 to reduce cancellation
    B = np.asarray(B, dtype=np.float64)
    sq_A = (A * A).sum(axis=1, keepdims=True)        # (N_A, 1)
    sq_B = (B * B).sum(axis=1, keepdims=True)        # (N_B, 1)
    D    = sq_A + sq_B.T - 2.0 * (A @ B.T)           # (N_A, N_B)
    np.maximum(D, 0.0, out=D)                         # clamp numerical noise
    return D.astype(np.float32)


# ---------------------------------------------------------------------------
# kNN coverage helpers
# ---------------------------------------------------------------------------
def precompute_real_knn(
    real_feats: np.ndarray,
    k: int = 5,
    tau_percentile: float = 95.0,
) -> Dict:
    """
    Pre-compute real-to-real kNN structure (done ONCE, reused for every subset).

    Returns
    -------
    dict with keys:
        rr_knn_radius : (M,) k-th nearest real-neighbor distance for each real sample
        rr_1nn        : (M,) 1-NN distance for each real sample
        tau_global    : global tau (tau_percentile of 1-NN distances)
    """
    rr = _cdist_sq(real_feats, real_feats)
    np.fill_diagonal(rr, np.inf)
    rr_dists = np.sqrt(rr)                                    # (M, M)

    # Sort each row to get k-th NN distances
    rr_sorted = np.sort(rr_dists, axis=1)                     # columns ascending
    rr_1nn    = rr_sorted[:, 0]                                # 1st NN
    k_col     = min(k - 1, rr_sorted.shape[1] - 1)
    rr_knn    = rr_sorted[:, k_col]                            # k-th NN radius

    tau       = float(np.percentile(rr_1nn, tau_percentile))

    return {
        "rr_knn_radius": rr_knn,
        "rr_1nn":        rr_1nn,
        "tau_global":    tau,
        "k":             k,
    }


def knn_coverage_metrics(
    real_feats: np.ndarray,
    gen_feats:  np.ndarray,
    real_knn:   Dict,
) -> Dict:
    """
    Coverage metrics between real and generated pooled DINO features.

    Uses THREE complementary definitions so the metric is informative even
    when real and generated distributions do not tightly overlap:

    1. **coverage_knn** (Improved Precision/Recall, Kynkäänniemi et al. 2019):
       For each real sample r_i, check if its nearest generated neighbor
       falls within r_i's k-th real-NN radius.  coverage = mean(indicator).

    2. **coverage_2x / coverage_3x**:  same, but with 2× / 3× relaxed radii.
       Useful when distributions are far apart — allows comparing which
       subsampling strategy *relatively* brings gen samples closer to real.

    3. **coverage_at_tau** (original global-tau definition, kept for reference).

    Also reports the **nn_ratio** = median(gen_nn / knn_radius), i.e. how many
    times farther generated neighbours are compared to real-manifold density.
    """
    rr_knn_radius = real_knn["rr_knn_radius"]
    tau_global    = real_knn["tau_global"]

    # Real-to-gen NN distances
    rg     = _cdist_sq(real_feats, gen_feats)
    rg_nn  = np.sqrt(rg.min(axis=1))                          # (M,)

    # Coverage with per-sample kNN radius (standard)
    ratio = rg_nn / (rr_knn_radius + 1e-12)
    coverage_knn = float((ratio <= 1.0).mean())
    coverage_2x  = float((ratio <= 2.0).mean())
    coverage_3x  = float((ratio <= 3.0).mean())

    # Original global-tau coverage (kept for reference)
    coverage_tau  = float((rg_nn <= tau_global).mean())

    return {
        "mean_nn":          float(rg_nn.mean()),
        "median_nn":        float(np.median(rg_nn)),
        "tau":              tau_global,
        "coverage_at_tau":  coverage_tau,
        "coverage_knn":     coverage_knn,
        "coverage_2x":      coverage_2x,
        "coverage_3x":      coverage_3x,
        "nn_ratio_median":  float(np.median(ratio)),
        "nn_ratio_mean":    float(np.mean(ratio)),
    }


# ---------------------------------------------------------------------------
# Diversity proxy
# ---------------------------------------------------------------------------
def diversity_metrics(feats: np.ndarray, max_pairs: int = 10_000, seed: int = 0) -> Dict:
    """
    Intra-subset diversity:
      - mean_pairwise_dist : mean L2 between pairs (sampled if n*(n-1)/2 > max_pairs)
      - trace_cov          : Tr(empirical covariance)
    """
    n   = len(feats)
    rng = np.random.default_rng(seed)

    if n * (n - 1) // 2 > max_pairs:
        # Approximate with random pairs
        idx = rng.integers(0, n, size=(max_pairs, 2))
        idx = idx[idx[:, 0] != idx[:, 1]]
        diffs = feats[idx[:, 0]] - feats[idx[:, 1]]
        mean_pw = float(np.sqrt((diffs ** 2).sum(axis=1)).mean())
    else:
        dmat = _cdist_sq(feats, feats)
        vals = dmat[np.triu_indices(n, k=1)]
        mean_pw = float(np.sqrt(vals).mean()) if len(vals) > 0 else 0.0

    trace_cov = 0.0
    if feats.shape[0] > 1:
        trace_cov = float(np.trace(np.cov(feats, rowvar=False)))

    return {
        "mean_pairwise_dist": mean_pw,
        "trace_cov":          trace_cov,
    }


# ---------------------------------------------------------------------------
# Subsampling strategies
# ---------------------------------------------------------------------------
def build_subsets(
    scores:   np.ndarray,
    K:        int,
    top_pct:  float,
    seed:     int,
    eps:      float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Build 5 index subsets of size K from N generated samples.

    Parameters
    ----------
    scores  : (N,) raw scores for this score type
    K       : number of indices to select
    top_pct : fraction of N used as the Tplus/Tminus candidate pool.
              E.g. top_pct=0.10 means pool = ceil(0.10 * N) highest/lowest samples.
    seed    : RNG seed
    eps     : floor value to prevent zero sampling probability

    Strategies
    ----------
    U      : uniform random K (without replacement, no bias)

    Splus  : weighted sampling ∝ s_i' where
               s_i' = clamp(score_norm_i, eps, 1)
               score_norm_i = (s_i - s_min) / (s_max - s_min + eps)
             Result: samples with HIGH score are more likely to be selected.

    Sminus : weighted sampling ∝ (1 - s_i') where
               (1 - s_i') = clamp(1 - score_norm_i, eps, 1)
             This is the explicit complement of Splus.
             Result: samples with LOW score are more likely to be selected.

    Tplus  : deterministic top pool = ceil(top_pct * N) highest-score indices;
             then draw K uniformly (without replacement) from that pool.
             If pool_size <= K, take all pool indices.

    Tminus : same as Tplus but for the ceil(top_pct * N) lowest-score indices.
    """
    N   = len(scores)
    rng = np.random.default_rng(seed)

    # Min-max normalise scores to [0, 1] (robust to degenerate range)
    s_min = float(scores.min())
    s_max = float(scores.max())
    score_norm = (scores - s_min) / (s_max - s_min + eps)   # (N,)

    # --- U: uniform random ---
    U = rng.choice(N, size=K, replace=False)

    # --- Splus: p_i ∝ s_i' (high-score bias) ---
    p_plus = np.clip(score_norm, eps, 1.0)
    p_plus = p_plus / p_plus.sum()
    Splus  = rng.choice(N, size=K, replace=False, p=p_plus)

    # --- Sminus: p_i ∝ (1 - s_i') (low-score bias; explicit complement) ---
    p_minus = np.clip(1.0 - score_norm, eps, 1.0)
    p_minus = p_minus / p_minus.sum()
    Sminus  = rng.choice(N, size=K, replace=False, p=p_minus)

    # --- Tplus: pool = top ceil(top_pct * N) by score ---
    pool_size = max(K, math.ceil(top_pct * N))
    top_pool  = np.argsort(scores)[::-1][:pool_size]   # descending
    Tplus     = top_pool.copy() if len(top_pool) <= K else rng.choice(top_pool, size=K, replace=False)

    # --- Tminus: pool = bottom ceil(top_pct * N) by score ---
    bot_pool  = np.argsort(scores)[:pool_size]          # ascending
    Tminus    = bot_pool.copy() if len(bot_pool) <= K else rng.choice(bot_pool, size=K, replace=False)

    return {
        "U":      U.astype(np.int64),
        "Splus":  Splus.astype(np.int64),
        "Sminus": Sminus.astype(np.int64),
        "Tplus":  Tplus.astype(np.int64),
        "Tminus": Tminus.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# FM pipeline loader
# ---------------------------------------------------------------------------
def setup_fm_pipeline(args, device: str):
    """
    Build and return a ready-to-sample StableFlowMatchingPipeline.

    Two loading modes (matching repo usage):
      1. --fm_pipeline_dir  – auto-load from pipeline folder (UNET/ + VAE/)
      2. --fm_config_json + --fm_weights  – explicit UNet config/weights;
         --vae_config + --vae_weights for the VAE.
    """
    # Ensure fm_src package is importable
    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from fm_src.pipelines.flow_matching_pipeline import StableFlowMatchingPipeline

    t_scale = getattr(args, "fm_t_scale", 1000.0)

    if getattr(args, "fm_pipeline_dir", None):
        # --- folder-based auto-load (identical to generate_datasets.py) ---
        pipe = StableFlowMatchingPipeline(
            device=device,
            t_scale=t_scale,
            model_dir=args.fm_pipeline_dir,
        )
        pipe.load_from_pipeline_folder_auto(
            args.fm_pipeline_dir,
            strict=True,
            map_location=device,
            set_eval=True,
        )
        # Optional: override VAE weights (e.g. --vae_weights ./fm_src/vae_best.pt)
        if getattr(args, "vae_weights", None):
            print(f"[FM] Overriding VAE weights: {args.vae_weights}")
            pipe.load_vae_weights(args.vae_weights, strict=True, map_location=device)
            pipe.vae.eval()

    else:
        # --- explicit config + weights mode ---
        pipe = StableFlowMatchingPipeline(device=device, t_scale=t_scale)
        pipe.build_from_configs(
            vae_json=args.vae_config,
            unet_json=args.fm_config_json,
            save_configs=False,
        )
        if args.vae_weights:
            pipe.load_vae_weights(args.vae_weights, strict=True, map_location=device)
        pipe.load_unet_weights(args.fm_weights, strict=True, map_location=device)
        pipe.vae.eval()
        pipe.unet.eval()

    pipe.freeze_vae()
    return pipe


# ---------------------------------------------------------------------------
# DINO feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_dino_features(
    dino_model,
    images_b1hw: torch.Tensor,
    device: str,
) -> np.ndarray:
    """
    Extract pooled DINOv2 patch-token features for a batch.

    images_b1hw : (B, 1, H, W) float32 in [-1, 1]
    returns      : (B, D) float32 numpy array
    """
    # Step 1: per-image min/max stretch (same as SurprisePredictor._per_image_minmax)
    stretched = per_image_minmax(images_b1hw)            # (B, 1, H, W) in [-1, 1]

    # Step 2: map to DINO input (mirrors _to_dino_input)
    dino_in  = to_dino_input(stretched, device)           # (B, 3, 224, 224)

    # Step 3: extract patch tokens and mean-pool
    out      = dino_model.forward_features(dino_in)
    tokens   = out["x_norm_patchtokens"]                  # (B, N_patches, D)
    pooled   = tokens.mean(dim=1)                         # (B, D)
    return pooled.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FM subsampling coverage/diversity analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
    p.add_argument("--out_dir",        required=True, help="Output root directory.")
    p.add_argument("--surprise_ckpt",  required=True, help="SurprisePredictor .pt checkpoint.")
    p.add_argument("--real_data_root", required=True,
                   help="Root of real data; images are at REAL_DATA_ROOT/split/*.npy.")

    # --- FM pipeline (mutually exclusive loading modes) ---
    p.add_argument("--fm_pipeline_dir", default=None,
                   help="Folder with UNET/ and VAE/ subdirs (auto-load, easiest).")
    p.add_argument("--fm_config_json",  default=None,
                   help="UNet config JSON (or combined VAE+UNet JSON).")
    p.add_argument("--fm_weights",      default=None, help="UNet .pt weights path.")
    p.add_argument("--vae_config",      default=None, help="VAE config JSON path.")
    p.add_argument("--vae_weights",     default=None, help="VAE .pt weights path.")
    p.add_argument("--fm_t_scale",      type=float, default=1000.0,
                   help="Time-scaling factor applied to t before UNet (matches FM training).")
    p.add_argument("--fm_steps",        type=int,   default=50,
                   help="Number of Euler steps for FM sampling.")

    # --- data ---
    p.add_argument("--split",     default="train",        help="Real data split subfolder.")
    p.add_argument("--dino_name", default="dinov2_vits14", help="DINOv2 hub model name.")
    p.add_argument("--max_n_real",  type=int, default=0,
                   help="Randomly sample this many real images (0 = all).")
    p.add_argument("--max_real", dest="max_n_real", type=int,
                   help="Deprecated alias for --max_n_real.")

    # --- subsampling ---
    p.add_argument("--N",       type=int,   default=5000,
                   help="Number of samples to generate.")
    p.add_argument("--K",       type=int,   default=1000,
                   help="Subset size (must be < N).")
    p.add_argument("--top_pct", type=float, default=0.1,
                   help="Fraction of N used as Tplus/Tminus candidate pool.")
    p.add_argument("--seed",    type=int,   default=0)

    # --- runtime ---
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--device",       default="auto",
                   help="'auto', 'cuda', 'cuda:1', 'cpu', etc.")
    p.add_argument("--tau_percentile", type=float, default=95.0,
                   help="Percentile of real-NN distances used as coverage threshold τ.")
    p.add_argument("--coverage_k", type=int, default=5,
                   help="k for per-sample kNN coverage (k-th real NN radius).")

    # --- cache flags ---
    p.add_argument("--reuse_cache",    action="store_true",
                   help="Skip regeneration for samples already cached on disk.")
    p.add_argument("--overwrite_cache", action="store_true",
                   help="Delete and recreate all generated caches before running.")
    p.add_argument("--cache_tokens",   action="store_true",
                   help="Cache per-sample pooled DINO features (gen + real tokens).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # ---- device ----
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[device] {device}")

    # ---- validate pipeline args ----
    if args.fm_pipeline_dir is None:
        if args.fm_config_json is None or args.fm_weights is None:
            raise ValueError(
                "Provide either --fm_pipeline_dir (auto-load folder) or "
                "both --fm_config_json AND --fm_weights for explicit loading."
            )

    assert args.K < args.N, f"K={args.K} must be strictly less than N={args.N}"

    # ---- directory layout ----
    out_dir      = Path(args.out_dir)
    cache_dir    = out_dir / "cache"
    gen_z_dir    = cache_dir / "gen" / "z"
    gen_x_dir    = cache_dir / "gen" / "x"
    gen_sc_dir   = cache_dir / "gen" / "scores"
    tok_gen_dir  = cache_dir / "tokens" / "gen"
    tok_real_dir = cache_dir / "tokens" / "real"
    subsets_dir  = out_dir / "subsets"
    figures_dir  = out_dir / "figures"

    # Handle --overwrite_cache: wipe gen cache (z, x, scores)
    if args.overwrite_cache:
        gen_cache = cache_dir / "gen"
        if gen_cache.exists():
            shutil.rmtree(gen_cache)
            print("[cache] Existing gen cache deleted (--overwrite_cache).")

    for d in [gen_z_dir, gen_x_dir, gen_sc_dir, subsets_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.cache_tokens:
        tok_gen_dir.mkdir(parents=True, exist_ok=True)
        tok_real_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # =========================================================================
    # STEP 1+2  Generate N latents Z and decode to images X (per-sample cache)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1+2  FM generation")
    print("=" * 60)

    # Which indices still need to be generated?
    need_gen = [
        idx for idx in range(args.N)
        if not (
            args.reuse_cache
            and (gen_z_dir / f"{idx:07d}.npy").exists()
            and (gen_x_dir / f"{idx:07d}.npy").exists()
        )
    ]

    if need_gen:
        print(f"[gen] Loading FM pipeline …")
        pipe = setup_fm_pipeline(args, device)
        pipe.unet.eval()
        pipe.vae.eval()

        print(f"[gen] Generating {len(need_gen)} / {args.N} samples "
              f"(batch={args.batch_size}, steps={args.fm_steps}) …")

        with torch.no_grad():
            i = 0
            with tqdm(total=len(need_gen), desc="FM generate", unit="img") as pbar:
                while i < len(need_gen):
                    batch_idxs = need_gen[i : i + args.batch_size]
                    bs = len(batch_idxs)

                    # sample_euler returns raw latents; decode via VAE
                    z_gen = pipe.sample_euler(steps=args.fm_steps, batch_size=bs)   # (bs, C, H, W)
                    x_gen = pipe.decode_fm_output(z_gen)                             # (bs, ?, H, W) [-1,1]

                    # Enforce single channel
                    if x_gen.shape[1] != 1:
                        x_gen = x_gen[:, :1]

                    # Resize to 256×256 if the VAE outputs a different spatial size
                    if x_gen.shape[-2] != 256 or x_gen.shape[-1] != 256:
                        x_gen = F.interpolate(
                            x_gen, size=(256, 256), mode="bilinear", align_corners=False
                        )

                    z_np = z_gen.float().cpu().numpy()
                    x_np = x_gen.float().cpu().numpy()

                    for j, idx in enumerate(batch_idxs):
                        # Save latent (C, H, W) and decoded image (1, 256, 256) per-sample
                        np.save(gen_z_dir / f"{idx:07d}.npy", z_np[j].astype(np.float32))
                        np.save(gen_x_dir / f"{idx:07d}.npy", x_np[j].astype(np.float32))

                    i += bs
                    pbar.update(bs)

        del pipe
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        print(f"[gen] All {args.N} samples already cached (--reuse_cache). Skipping.")

    # Write run meta
    meta_path = cache_dir / "gen" / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "N": args.N,
                "seed": args.seed,
                "fm_steps": args.fm_steps,
                "fm_t_scale": args.fm_t_scale,
                "fm_pipeline_dir": args.fm_pipeline_dir,
                "fm_config_json": args.fm_config_json,
                "fm_weights": args.fm_weights,
                "vae_config": args.vae_config,
                "vae_weights": args.vae_weights,
                "surprise_ckpt": args.surprise_ckpt,
                "dino_name": args.dino_name,
                "K": args.K,
                "top_pct": args.top_pct,
                "real_data_root": args.real_data_root,
                "split": args.split,
            },
            f,
            indent=2,
        )

    # =========================================================================
    # STEP 3  SurprisePredictor inference → per-sample JSON scores
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3  Surprise predictor scoring")
    print("=" * 60)

    need_score = [
        idx for idx in range(args.N)
        if not (args.reuse_cache and (gen_sc_dir / f"{idx:07d}.json").exists())
    ]

    if need_score:
        # Import SurprisePredictor from repo — no code duplication
        repo_root = Path(__file__).parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from train_surprise_predictor import SurprisePredictor

        sp_ckpt = torch.load(args.surprise_ckpt, map_location="cpu")
        sp_args = sp_ckpt.get("args", {})

        # Prefer values saved in checkpoint; fall back to CLI args
        vae_cfg  = sp_args.get("vae_config",  args.vae_config)
        vae_wts  = sp_args.get("vae_weights", args.vae_weights)
        dino_nm  = sp_args.get("dino_name",   args.dino_name)
        h_dim    = int(sp_args.get("hidden_dim", 256))

        if vae_cfg is None or vae_wts is None:
            raise ValueError(
                "SurprisePredictor needs vae_config and vae_weights. "
                "Pass --vae_config / --vae_weights or ensure they are saved in the checkpoint."
            )

        print(f"[predictor] Building SurprisePredictor (hidden={h_dim}, dino={dino_nm}) …")
        predictor = SurprisePredictor(
            vae_config_path=vae_cfg,
            vae_weights_path=vae_wts,
            dino_name=dino_nm,
            hidden_dim=h_dim,
            device=device,
        ).to(device)
        predictor.load_state_dict(sp_ckpt["model_state"], strict=False)
        predictor.eval()

        print(f"[predictor] Scoring {len(need_score)} samples …")
        with torch.no_grad():
            i = 0
            while i < len(need_score):
                batch_idxs = need_score[i : i + args.batch_size]

                zs = [np.load(gen_z_dir / f"{idx:07d}.npy") for idx in batch_idxs]
                z_t = torch.tensor(
                    np.stack(zs, axis=0), dtype=torch.float32, device=device
                )

                pred_s, pred_g = predictor(z_t)
                pred_s = pred_s.float().cpu().numpy()
                pred_g = pred_g.float().cpu().numpy()

                for j, idx in enumerate(batch_idxs):
                    with open(gen_sc_dir / f"{idx:07d}.json", "w") as fp:
                        json.dump(
                            {
                                "surprise":      float(pred_s[j]),
                                "gmm":           float(pred_g[j]),
                                "one_minus_gmm": float(1.0 - pred_g[j]),
                            },
                            fp,
                        )

                i += len(batch_idxs)
                if i % 500 == 0 or i >= len(need_score):
                    print(f"  [{i}/{len(need_score)}] scored")

        del predictor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        print(f"[predictor] All {args.N} scores already cached (--reuse_cache). Skipping.")

    # =========================================================================
    # STEP 4+5  DINO feature extraction (generated + real)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4+5  DINO feature extraction")
    print("=" * 60)

    # Load DINOv2
    print(f"[dino] Loading {args.dino_name} …")
    dino_model = torch.hub.load("facebookresearch/dinov2", args.dino_name)
    dino_model = dino_model.to(device).eval()
    for param in dino_model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # 5a  Generated features
    # ------------------------------------------------------------------
    print("[dino] Extracting features for generated samples …")
    gen_feat_map: Dict[int, np.ndarray] = {}

    # Separate cached from uncached
    need_dino_gen: List[int] = []
    for idx in range(args.N):
        tok_path = tok_gen_dir / f"{idx:07d}.npy"
        if args.cache_tokens and tok_path.exists():
            gen_feat_map[idx] = np.load(tok_path).astype(np.float32)
        else:
            need_dino_gen.append(idx)

    for i in range(0, len(need_dino_gen), args.batch_size):
        batch_idxs = need_dino_gen[i : i + args.batch_size]
        imgs = [np.load(gen_x_dir / f"{idx:07d}.npy") for idx in batch_idxs]
        # shape: (B, 1, 256, 256)
        imgs_t = torch.tensor(
            np.stack(imgs, axis=0), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            feats = extract_dino_features(dino_model, imgs_t, device)   # (B, D)

        for j, idx in enumerate(batch_idxs):
            f = feats[j].astype(np.float32)
            gen_feat_map[idx] = f
            if args.cache_tokens:
                np.save(tok_gen_dir / f"{idx:07d}.npy", f)

        if (i + len(batch_idxs)) % 1000 == 0 or i + len(batch_idxs) >= len(need_dino_gen):
            print(f"  gen dino [{i + len(batch_idxs)}/{len(need_dino_gen)}]")

    # Assemble in index order → (N, D)
    gen_feats = np.stack([gen_feat_map[idx] for idx in range(args.N)], axis=0)
    print(f"[dino] gen_feats shape: {gen_feats.shape}")

    # ------------------------------------------------------------------
    # 5b  Real features  (images loaded on-demand; NEVER written to disk)
    # ------------------------------------------------------------------
    print("[dino] Extracting features for real samples …")
    real_dir = Path(args.real_data_root) / args.split
    real_paths = sorted(real_dir.glob("*.npy"))
    if args.max_n_real > 0:
        if args.max_n_real >= len(real_paths):
            print(
                f"[dino] --max_n_real={args.max_n_real} >= available={len(real_paths)}; "
                "using all real samples."
            )
        else:
            rng_real = np.random.default_rng(args.seed)
            sel_idx = rng_real.choice(len(real_paths), size=args.max_n_real, replace=False)
            # Keep deterministic processing/caching order while preserving random subset selection.
            real_paths = [real_paths[i] for i in sorted(sel_idx.tolist())]
            print(f"[dino] Randomly selected {len(real_paths)} real samples (--max_n_real).")
    if not real_paths:
        raise RuntimeError(f"No .npy files found in {real_dir}")
    print(f"[dino] Found {len(real_paths)} real samples in {real_dir}")

    real_feats_list:  List[np.ndarray] = []
    real_stems_list:  List[str]        = []

    # Accumulate batches
    _rbatch_imgs:  List[torch.Tensor] = []
    _rbatch_stems: List[str]          = []

    def _flush_real_batch() -> None:
        """Process and save accumulated real-image batch; do NOT save images."""
        if not _rbatch_imgs:
            return
        imgs_t = torch.stack(_rbatch_imgs, dim=0).to(device)   # (B, 1, H, W)
        with torch.no_grad():
            feats = extract_dino_features(dino_model, imgs_t, device)
        for k, stem in enumerate(_rbatch_stems):
            f = feats[k].astype(np.float32)
            real_feats_list.append(f)
            real_stems_list.append(stem)
            if args.cache_tokens:
                # Optional: cache pooled DINO tokens for real (saves future DINO passes)
                # NOTE: we save TOKENS only — the normalised image is NEVER written
                np.save(tok_real_dir / f"{stem}.npy", f)
        _rbatch_imgs.clear()
        _rbatch_stems.clear()

    for npy_path in tqdm(real_paths, desc="real DINO"):
        stem     = npy_path.stem
        tok_path = tok_real_dir / f"{stem}.npy"

        if args.cache_tokens and tok_path.exists():
            # Load cached token directly (the image itself was never stored)
            real_feats_list.append(np.load(tok_path).astype(np.float32))
            real_stems_list.append(stem)
            continue

        # Load uint16 .npy, apply canonical repo normalisation to [-1, 1]
        # The normalised tensor is used only for DINO; it is NEVER saved to disk.
        arr = np.load(npy_path)                           # uint16, (H,W) or (1,H,W)
        if arr.ndim == 2:
            arr = arr[np.newaxis]                         # → (1, H, W)
        x_t = torch.from_numpy(arr.astype(np.float32))   # (1, H, W) still raw
        x_t = to_sd_tensor_and_x(x_t)                    # → [-1, 1]  (canonical repo fn)
        x_t = x_t.unsqueeze(0)                           # → (1, 1, H, W)
        if x_t.shape[-2] != 256 or x_t.shape[-1] != 256:
            x_t = F.interpolate(
                x_t, size=(256, 256), mode="bilinear", align_corners=False
            )
        # Accumulate (keep on CPU until batch is full)
        _rbatch_imgs.append(x_t.squeeze(0).cpu())        # (1, H, W)
        _rbatch_stems.append(stem)

        if len(_rbatch_imgs) >= args.batch_size:
            _flush_real_batch()

    _flush_real_batch()   # flush remainder

    real_feats = np.stack(real_feats_list, axis=0)   # (M, D)
    print(f"[dino] real_feats shape: {real_feats.shape}")

    del dino_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # =========================================================================
    # STEP 6  Load scores and build subsets
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6  Subsampling")
    print("=" * 60)

    scores_surprise   = np.empty(args.N, dtype=np.float32)
    scores_one_m_gmm  = np.empty(args.N, dtype=np.float32)

    for idx in range(args.N):
        with open(gen_sc_dir / f"{idx:07d}.json") as f:
            d = json.load(f)
        scores_surprise[idx]  = d["surprise"]
        scores_one_m_gmm[idx] = d["one_minus_gmm"]

    print(f"  surprise   : min={scores_surprise.min():.4f}  max={scores_surprise.max():.4f}  "
          f"mean={scores_surprise.mean():.4f}")
    print(f"  1-gmm      : min={scores_one_m_gmm.min():.4f}  max={scores_one_m_gmm.max():.4f}  "
          f"mean={scores_one_m_gmm.mean():.4f}")

    score_types = {
        "scoreA_surprise":      scores_surprise,
        "scoreB_one_minus_gmm": scores_one_m_gmm,
    }

    all_subsets: Dict[str, Dict[str, np.ndarray]] = {}
    for score_name, scores in score_types.items():
        subsets = build_subsets(
            scores, K=args.K, top_pct=args.top_pct, seed=args.seed
        )
        all_subsets[score_name] = subsets

        sdir = subsets_dir / score_name
        sdir.mkdir(parents=True, exist_ok=True)
        for sub_name, indices in subsets.items():
            np.save(sdir / f"indices_{sub_name}.npy", indices)

        print(f"  [{score_name}] subsets saved: {list(subsets.keys())}")

    # =========================================================================
    # STEP 7  Metrics
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 7  Metrics")
    print("=" * 60)

    # Pre-compute real-real kNN structure ONCE (reused for every subset)
    coverage_k = getattr(args, "coverage_k", 5)
    real_knn = precompute_real_knn(
        real_feats, k=coverage_k, tau_percentile=args.tau_percentile
    )
    print(f"  real-real kNN (k={coverage_k}):")
    print(f"    1-NN  mean={real_knn['rr_1nn'].mean():.4f}  "
          f"median={np.median(real_knn['rr_1nn']):.4f}  "
          f"p95={np.percentile(real_knn['rr_1nn'], 95):.4f}")
    print(f"    k-NN  mean={real_knn['rr_knn_radius'].mean():.4f}  "
          f"median={np.median(real_knn['rr_knn_radius']):.4f}")
    print(f"    τ_global (p{args.tau_percentile:.0f} of 1-NN) = {real_knn['tau_global']:.4f}")

    def _metrics(gen_f: np.ndarray) -> Dict:
        fd   = frechet_distance(real_feats, gen_f)
        knn  = knn_coverage_metrics(real_feats, gen_f, real_knn)
        div  = diversity_metrics(gen_f, seed=args.seed)
        return {
            "frechet_dist":    fd,
            "n_gen":           int(len(gen_f)),
            "n_real":          int(len(real_feats)),
            "coverage_k":      coverage_k,
            **knn,
            **div,
        }

    all_metrics: Dict[str, Dict[str, Dict]] = {}

    for score_name, subsets in all_subsets.items():
        run_metrics: Dict[str, Dict] = {}

        # FULL set baseline
        run_metrics["FULL"] = _metrics(gen_feats)

        for sub_name, indices in subsets.items():
            m = _metrics(gen_feats[indices])
            run_metrics[sub_name] = m
            print(
                f"  [{score_name}/{sub_name:6s}]  "
                f"FD={m['frechet_dist']:8.4f}  "
                f"cov_knn={m['coverage_knn']:.4f}  "
                f"cov_2x={m['coverage_2x']:.4f}  "
                f"cov_3x={m['coverage_3x']:.4f}  "
                f"ratio={m['nn_ratio_median']:.3f}  "
                f"mean_pw={m['mean_pairwise_dist']:.4f}"
            )

        sdir = subsets_dir / score_name
        with open(sdir / "metrics.json", "w") as f:
            json.dump(run_metrics, f, indent=2)

        all_metrics[score_name] = run_metrics

    # =========================================================================
    # STEP 8  Plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 8  Plotting")
    print("=" * 60)

    for score_name, scores in score_types.items():
        subsets      = all_subsets[score_name]
        run_metrics  = all_metrics[score_name]

        pool_size     = max(args.K, math.ceil(args.top_pct * args.N))
        sorted_scores = np.sort(scores)
        # Threshold for Tplus: score at the boundary between pool and rest (descending order)
        tplus_thresh  = float(sorted_scores[-min(pool_size, len(sorted_scores))])
        # Threshold for Tminus (ascending order)
        tminus_thresh = float(sorted_scores[min(pool_size - 1, len(sorted_scores) - 1)])

        # ---- 8a: Score histogram ----
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(scores, bins=80, color="steelblue", alpha=0.75, edgecolor="none")
        ax.axvline(tplus_thresh,  color="red",    linestyle="--", linewidth=1.5,
                   label=f"T+ lower bound ({tplus_thresh:.3f})")
        ax.axvline(tminus_thresh, color="orange", linestyle="--", linewidth=1.5,
                   label=f"T- upper bound ({tminus_thresh:.3f})")
        ax.set_title(f"Score distribution — {score_name}")
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"hist_{score_name}.png", dpi=150)
        plt.close(fig)

        # ---- 8b: Bar plots of metrics across subsets ----
        subset_order  = ["FULL", "U", "Splus", "Sminus", "Tplus", "Tminus"]
        present_order = [s for s in subset_order if s in run_metrics]
        metric_keys   = ["frechet_dist", "coverage_knn", "coverage_2x", "coverage_3x",
                         "nn_ratio_median", "mean_pairwise_dist", "trace_cov"]
        metric_labels = ["Fréchet dist ↓",
                         f"Coverage kNN(k={coverage_k}) ↑",
                         "Coverage 2× radius ↑",
                         "Coverage 3× radius ↑",
                         "NN ratio (median) ↓",
                         "Mean PW dist ↑",
                         "Trace(Cov) ↑"]
        palette       = ["#aaaaaa", "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

        fig, axes = plt.subplots(1, len(metric_keys), figsize=(4 * len(metric_keys), 4.5))
        for ax, mk, ml in zip(axes, metric_keys, metric_labels):
            vals   = [run_metrics[s].get(mk, 0.0) for s in present_order]
            colors = palette[: len(present_order)]
            x_pos  = range(len(present_order))
            bars   = ax.bar(x_pos, vals, color=colors)
            ax.set_title(ml, fontsize=9)
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(present_order, rotation=35, ha="right", fontsize=8)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )
        fig.suptitle(f"Subset metrics — {score_name}", y=1.02, fontsize=10)
        fig.tight_layout()
        fig.savefig(figures_dir / f"barplot_{score_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- 8c: PCA scatter charts ----
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("[plot] sklearn not available; skipping PCA scatter.")
            continue

        # Fit a single PCA basis on real + ALL generated to keep plots comparable.
        pca_2d = PCA(n_components=2, random_state=args.seed)
        all_for_fit = np.concatenate([real_feats, gen_feats], axis=0)
        all_pc = pca_2d.fit_transform(all_for_fit)

        pc_real = all_pc[: len(real_feats)]
        pc_gen_all = all_pc[len(real_feats):]
        var_str = " | ".join(
            f"PC{i+1}: {v:.1%}" for i, v in enumerate(pca_2d.explained_variance_ratio_)
        )

        # 8c-1) PCA chart: all generated vs real
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            pc_real[:, 0], pc_real[:, 1],
            s=7, c="gray", alpha=0.30, label="real", zorder=1
        )
        sc = ax.scatter(
            pc_gen_all[:, 0], pc_gen_all[:, 1],
            s=12, c=scores, cmap="viridis", alpha=0.70,
            label="gen (ALL)", zorder=2
        )
        plt.colorbar(sc, ax=ax, label="score")
        ax.set_title(f"PCA — real vs all generated   [{score_name}]", fontsize=10)
        ax.set_xlabel(f"PC1 ({var_str})")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"pca_{score_name}_ALL_vs_real.png", dpi=150)
        plt.close(fig)

        # 8c-2) PCA chart: all subsets vs real (multi-panel)
        subset_panels = ["U", "Splus", "Sminus", "Tplus", "Tminus"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for panel_idx, sub_name in enumerate(subset_panels):
            ax = axes[panel_idx]
            if sub_name not in subsets:
                ax.axis("off")
                continue

            sub_idx = subsets[sub_name]
            pc_sub = pc_gen_all[sub_idx]
            sub_scores = scores[sub_idx]

            ax.scatter(
                pc_real[:, 0], pc_real[:, 1],
                s=6, c="gray", alpha=0.25, label="real", zorder=1
            )
            sc_sub = ax.scatter(
                pc_sub[:, 0], pc_sub[:, 1],
                s=16, c=sub_scores, cmap="viridis", alpha=0.75,
                label=f"gen ({sub_name})", zorder=2
            )
            ax.set_title(f"{sub_name} vs real", fontsize=10)
            ax.grid(alpha=0.15)
            ax.legend(loc="upper right", fontsize=7, markerscale=1.5)
            plt.colorbar(sc_sub, ax=ax, fraction=0.046, pad=0.02)

        # Last empty panel used for text/legend info.
        info_ax = axes[-1]
        info_ax.axis("off")
        info_ax.text(
            0.05, 0.80,
            "PCA basis fitted on:\nreal + all generated\n\n"
            "Panels: each subset vs real\n"
            f"score type: {score_name}",
            fontsize=10, va="top"
        )

        fig.suptitle(
            f"PCA — all subsets vs real   [{score_name}]\n{var_str}",
            fontsize=12,
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(figures_dir / f"pca_{score_name}_all_subsets_vs_real.png", dpi=150)
        plt.close(fig)

    # =========================================================================
    # End-of-run summary table
    # =========================================================================
    line = "=" * 80
    print(f"\n{line}")
    print("END-OF-RUN SUMMARY")
    print(line)
    col_w = 18
    hdr = (
        f"{'Subset':>{col_w}}  "
        f"{'FD':>9}  "
        f"{'cov_knn':>9}  "
        f"{'cov_2x':>9}  "
        f"{'cov_3x':>9}  "
        f"{'nn_ratio':>9}  "
        f"{'mean_pw':>9}  "
        f"{'trc_cov':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for score_name, run_metrics in all_metrics.items():
        print(f"\n  ● {score_name}")
        for sname in ["FULL", "U", "Splus", "Sminus", "Tplus", "Tminus"]:
            if sname not in run_metrics:
                continue
            m = run_metrics[sname]
            print(
                f"    {sname:>{col_w - 4}}  "
                f"{m['frechet_dist']:>9.3f}  "
                f"{m['coverage_knn']:>9.4f}  "
                f"{m['coverage_2x']:>9.4f}  "
                f"{m['coverage_3x']:>9.4f}  "
                f"{m['nn_ratio_median']:>9.3f}  "
                f"{m['mean_pairwise_dist']:>9.3f}  "
                f"{m['trace_cov']:>9.3f}"
            )

    print(f"\n[Done]  Results written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
