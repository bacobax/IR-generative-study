#!/usr/bin/env python3
"""
Distribution Shift Analysis: Real vs Generated Images
Analyzes multiple generators and compares their output distributions to real data
using Inception, VAE, and DINOv2 feature extractors.

Metrics computed per extractor:
  - MMD, Wasserstein Distance, Mean Euclidean Distance, KL Divergence,
    Trace Difference, FID Score, Precision, Coverage

Results are saved in per-extractor sub-folders inside --output_dir.
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.models as models
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

from src.core.constants import (
    P0001_PERCENTILE_RAW_IMAGES,
    P9999_PERCENTILE_RAW_IMAGES,
    RAW_RANGE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.core.normalization import raw_to_norm_numpy as _normalize_uint16_to_m1p1
from fm_src.pipelines.flow_matching_pipeline import StableFlowMatchingPipeline

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distribution shift analysis for real vs generated IR samples (.npy)."
    )
    parser.add_argument("--real_dir", type=str, default="./data/raw/v18/images",
                        help="Folder containing real .npy samples.")
    parser.add_argument("--generated_dir", type=str, default="./artifacts/generated/main",
                        help="Folder containing generated subfolders with .npy files.")
    parser.add_argument("--output_dir", type=str, default="./artifacts/analysis/main",
                        help="Where to save plots and metrics.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to load per dataset (real + each generator).")
    parser.add_argument("--metrics_max_samples", type=int, default=None,
                        help="Optional cap on feature samples used for metrics.")
    parser.add_argument("--metrics_pca_dim", type=int, default=256,
                        help="PCA dim for metrics (reduces cost of covariance). Set 0 to disable.")
    parser.add_argument("--skip_kl", action="store_true",
                        help="Skip KL divergence metric (most expensive).")
    parser.add_argument("--tsne_perplexity", type=int, default=30,
                        help="Base t-SNE perplexity (auto-clamped to sample count).")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device override, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae_config", type=str, default="./artifacts/checkpoints/vae/vae_runs/vae_fm_x4/VAE/config.json",
                        help="Path to VAE config JSON for VAE-based feature extraction.")
    parser.add_argument("--vae_weights", type=str, default="./artifacts/checkpoints/vae/vae_runs/vae_fm_x4/VAE/vae_best.pt",
                        help="Path to VAE weights for VAE-based feature extraction.")
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14",
                        help="DINOv2 model name for torch.hub (e.g. dinov2_vits14, dinov2_vitb14).")
    parser.add_argument("--precision_coverage_k", type=int, default=5,
                        help="k for k-NN used in Precision and Coverage metrics.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A = P0001_PERCENTILE_RAW_IMAGES
B = P9999_PERCENTILE_RAW_IMAGES
S = RAW_RANGE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_environment(output_dir: str, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    sns.set_style("darkgrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 10
    os.makedirs(output_dir, exist_ok=True)


def load_images(image_path, max_images=None):
    """Load 1-channel .npy images from a directory.

    Expects uint16 .npy files.  All images are normalised to [-1, 1]
    via the percentile linear mapping.
    """
    images = []
    image_paths = []

    image_files = glob.glob(os.path.join(image_path, "**/*.npy"), recursive=True)

    if not image_files:
        print(f"Warning: No .npy files found in {image_path}")
        return np.array([]), []

    if max_images is not None:
        image_files = image_files[:max_images]

    print(f"Loading {len(image_files)} images from {image_path}")

    for img_file in tqdm(image_files, desc=f"Loading {Path(image_path).name}"):
        try:
            arr = np.load(img_file)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")
            images.append(arr)
            image_paths.append(img_file)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue

    if len(images) == 0:
        return np.array([]), []

    result = np.array(images)
    print(f"  -> Loaded dtype={result.dtype}, range=[{result.min():.1f}, {result.max():.1f}]")
    result = _normalize_uint16_to_m1p1(result)
    print(f"  -> Normalised to [-1, 1]  range=[{result.min():.4f}, {result.max():.4f}]")

    return result, image_paths


def find_generated_folders(base_path):
    """Find all folders containing generated images."""
    folders = {}
    if os.path.isdir(base_path):
        for item in sorted(os.listdir(base_path)):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                image_files = glob.glob(os.path.join(item_path, "**/*.npy"), recursive=True)
                if len(image_files) > 0:
                    folders[item] = item_path
    return folders


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def extract_features_inception(images, model, device, batch_size=16):
    """Extract features from 1-channel images using InceptionV3."""
    features = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Extracting features (Inception)"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch_images = images[start_idx:end_idx]

            batch = torch.from_numpy(batch_images).unsqueeze(1).to(device)
            batch = (batch + 1.0) / 2.0
            batch = batch.clamp(0.0, 1.0)
            batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            batch = (batch - mean) / std

            feat = model(batch)
            features.append(feat.cpu().numpy())

    return np.vstack(features)


def extract_features_vae(images, vae, device, batch_size=16):
    """Extract flattened VAE mu features from 1-channel images."""
    features = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Extracting features (VAE)"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch_images = images[start_idx:end_idx]

            batch = torch.from_numpy(batch_images).unsqueeze(1).float().to(device)
            if batch.shape[2] != 256 or batch.shape[3] != 256:
                batch = F.interpolate(batch, size=(256, 256), mode="bilinear", align_corners=False)

            z_mu, _ = vae.encode(batch)
            feat = z_mu.flatten(start_dim=1)
            features.append(feat.cpu().numpy())

    return np.vstack(features)


def extract_features_dinov2(images, model, device, batch_size=16):
    """Extract CLS-token features from 1-channel images using DINOv2.

    DINOv2 expects 3-channel 224x224 images normalised with ImageNet stats.
    """
    features = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Extracting features (DINOv2)"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch_images = images[start_idx:end_idx]

            batch = torch.from_numpy(batch_images).unsqueeze(1).float().to(device)
            batch = (batch + 1.0) / 2.0
            batch = batch.clamp(0.0, 1.0)
            batch = batch.repeat(1, 3, 1, 1)
            # DINOv2 ViT-S/14 expects multiples of 14; 224 is the standard size
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = (batch - mean) / std

            out = model.forward_features(batch)
            cls = out["x_norm_clstoken"]              # (B, D)
            feats = cls
            features.append(feats.cpu().numpy())

    return np.vstack(features)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_distribution_metrics(real_features, gen_features, progress=None, skip_kl=False,
                                  mmd_sigma=None):
    """Compute distribution shift metrics between real and generated feature sets."""
    metrics = {}
    if len(real_features) == 0 or len(gen_features) == 0:
        return metrics

    def _pairwise_sq_dists(a, b):
        """Compute pairwise squared Euclidean distances between rows of a and b."""
        a_sq = np.sum(a ** 2, axis=1, keepdims=True)  # (N, 1)
        b_sq = np.sum(b ** 2, axis=1, keepdims=True)  # (M, 1)
        return np.maximum(a_sq + b_sq.T - 2.0 * a @ b.T, 0.0)  # (N, M)

    def mmd_rbf(x, y, sigma=None):
        """Unbiased RBF-kernel MMD² between samples x and y.

        Uses the median heuristic for bandwidth selection when sigma is None.
        """
        n = len(x)
        m = len(y)
        if n < 2 or m < 2:
            return 0.0

        d_xx = _pairwise_sq_dists(x, x)  # (n, n)
        d_yy = _pairwise_sq_dists(y, y)  # (m, m)
        d_xy = _pairwise_sq_dists(x, y)  # (n, m)

        if sigma is None:
            # Median heuristic: sigma² = median of all pairwise squared distances
            all_dists = np.concatenate([
                d_xx[np.triu_indices(n, k=1)],
                d_yy[np.triu_indices(m, k=1)],
                d_xy.ravel(),
            ])
            med_sq = float(np.median(all_dists))
            sigma_sq = med_sq if med_sq > 0 else 1.0
        else:
            sigma_sq = sigma ** 2

        k_xx = np.exp(-d_xx / (2.0 * sigma_sq))
        k_yy = np.exp(-d_yy / (2.0 * sigma_sq))
        k_xy = np.exp(-d_xy / (2.0 * sigma_sq))

        # Unbiased estimator: exclude diagonal for k_xx and k_yy
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)

        mmd2 = (k_xx.sum() / (n * (n - 1))
                + k_yy.sum() / (m * (m - 1))
                - 2.0 * k_xy.mean())

        return max(0.0, float(mmd2))

    # Joint normalisation using real-feature statistics only, so mean/scale
    # differences between real and generated distributions are preserved.
    real_mean = real_features.mean(axis=0)
    real_std = real_features.std(axis=0) + 1e-8
    real_feat_norm = (real_features - real_mean) / real_std
    gen_feat_norm = (gen_features - real_mean) / real_std

    if progress is not None:
        progress.set_postfix_str("MMD")
    metrics["MMD"] = mmd_rbf(real_feat_norm, gen_feat_norm, sigma=mmd_sigma)
    if progress is not None:
        progress.update(1)

    real_agg = real_features.mean(axis=0)
    gen_agg = gen_features.mean(axis=0)
    if progress is not None:
        progress.set_postfix_str("Wasserstein")
    metrics["Wasserstein_Distance"] = wasserstein_distance(real_agg, gen_agg)
    if progress is not None:
        progress.update(1)

    if progress is not None:
        progress.set_postfix_str("MeanDist")
    metrics["Mean_Euclidean_Distance"] = float(np.linalg.norm(real_agg - gen_agg))
    if progress is not None:
        progress.update(1)

    real_cov = np.cov(real_features.T) + 1e-8 * np.eye(real_features.shape[1])
    gen_cov = np.cov(gen_features.T) + 1e-8 * np.eye(gen_features.shape[1])

    if skip_kl:
        metrics["KL_Divergence"] = np.nan
        if progress is not None:
            progress.update(1)
    else:
        if progress is not None:
            progress.set_postfix_str("KL")
        try:
            kl_div = 0.5 * (
                np.trace(np.linalg.inv(gen_cov) @ real_cov)
                + np.sum((gen_agg - real_agg) ** 2 / np.diag(gen_cov))
                - real_features.shape[1]
                + np.linalg.slogdet(gen_cov)[1]
                - np.linalg.slogdet(real_cov)[1]
            )
            metrics["KL_Divergence"] = kl_div
        except Exception:
            metrics["KL_Divergence"] = np.inf
        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.set_postfix_str("Trace")
    metrics["Trace_Difference"] = float(np.trace(real_cov) - np.trace(gen_cov))
    if progress is not None:
        progress.update(1)

    return metrics


def calculate_fid_score(real_features, gen_features):
    """Calculate Frechet Inception Distance (FID)."""
    if len(real_features) == 0 or len(gen_features) == 0:
        return np.inf

    real_mean = np.mean(real_features, axis=0)
    gen_mean = np.mean(gen_features, axis=0)

    real_cov = np.cov(real_features.T) + 1e-8 * np.eye(real_features.shape[1])
    gen_cov = np.cov(gen_features.T) + 1e-8 * np.eye(gen_features.shape[1])

    mean_diff = np.sum((real_mean - gen_mean) ** 2)

    try:
        eigvals, eigvecs = np.linalg.eigh(real_cov)
        eigvals[eigvals < 0] = 0
        sqrt_real_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        product = sqrt_real_cov @ gen_cov @ sqrt_real_cov
        eigvals_product = np.linalg.eigvalsh(product)
        eigvals_product[eigvals_product < 0] = 0
        trace_term = np.trace(real_cov) + np.trace(gen_cov) - 2 * np.sum(np.sqrt(eigvals_product))

        fid = mean_diff + trace_term
    except Exception:
        fid = np.inf

    return max(0, fid)


def compute_precision_coverage(real_features, gen_features, k=5):
    """Compute Improved Precision (Kynkaanniemi et al., 2019) and
    Coverage (Naeem et al., 2020) between real and generated feature sets.

    Precision = fraction of generated samples that fall inside the
                k-NN manifold of the real samples.
    Coverage  = fraction of real samples whose k-NN ball is "hit"
                by at least one generated sample.

    Parameters
    ----------
    real_features : np.ndarray, shape (N_r, D)
    gen_features  : np.ndarray, shape (N_g, D)
    k : int
        Number of nearest neighbours to define local radii.

    Returns
    -------
    precision : float in [0, 1]
    coverage  : float in [0, 1]
    """
    if len(real_features) == 0 or len(gen_features) == 0:
        return 0.0, 0.0

    # Pairwise L2 distances
    # real-real distances for neighbourhood radii
    dist_rr = cdist(real_features, real_features, metric="euclidean")  # (N_r, N_r)
    # For each real sample, k-th NN distance (skip self which is 0)
    # Sort each row; index 0 is self (distance 0), so k-th NN is at index k
    dist_rr_sorted = np.sort(dist_rr, axis=1)
    # Clamp k to available neighbours
    k_eff = min(k, dist_rr.shape[1] - 1)
    if k_eff < 1:
        return 0.0, 0.0
    radii_real = dist_rr_sorted[:, k_eff]  # (N_r,)

    # gen-to-real distances
    dist_gr = cdist(gen_features, real_features, metric="euclidean")  # (N_g, N_r)

    # --- Precision ---
    # For each generated sample, check if it falls inside ANY real sample's ball
    # g_j is inside r_i's ball iff dist(g_j, r_i) <= radii_real[i]
    inside = dist_gr <= radii_real[np.newaxis, :]  # (N_g, N_r)
    precision = float(np.any(inside, axis=1).mean())

    # --- Coverage ---
    # For each real sample, check if at least one generated sample falls in its ball
    covered = np.any(inside, axis=0)  # (N_r,)
    coverage = float(covered.mean())

    return precision, coverage


def compute_kl_divergence_discrete(real_data, gen_data, bins=50):
    """Compute KL divergence from histograms."""
    real_flat = real_data.flatten()
    gen_flat = gen_data.flatten()

    hist_real, _ = np.histogram(real_flat, bins=bins, range=(-1, 1))
    hist_gen, _ = np.histogram(gen_flat, bins=bins, range=(-1, 1))

    hist_real = hist_real / np.sum(hist_real)
    hist_gen = hist_gen / np.sum(hist_gen)

    epsilon = 1e-10
    hist_real = hist_real + epsilon
    hist_gen = hist_gen + epsilon

    kl_div = np.sum(hist_real * (np.log(hist_real) - np.log(hist_gen)))

    return kl_div, hist_real, hist_gen


# ---------------------------------------------------------------------------
# Unified per-extractor pipeline
# ---------------------------------------------------------------------------

def _maybe_subsample_and_pca(real_feat, gen_feat, args):
    """Optionally subsample and PCA-reduce features for metrics."""
    real_use = real_feat
    gen_use = gen_feat

    if args.metrics_max_samples is not None:
        n_real = min(args.metrics_max_samples, len(real_feat))
        n_gen = min(args.metrics_max_samples, len(gen_feat))
        real_idx = np.random.choice(len(real_feat), n_real, replace=False)
        gen_idx = np.random.choice(len(gen_feat), n_gen, replace=False)
        real_use = real_feat[real_idx]
        gen_use = gen_feat[gen_idx]

    if args.metrics_pca_dim and args.metrics_pca_dim > 0:
        combined = np.vstack([real_use, gen_use])
        max_components = min(combined.shape[0], combined.shape[1])
        n_components = min(args.metrics_pca_dim, max_components)
        if n_components >= 1 and n_components < combined.shape[1]:
            pca = PCA(n_components=n_components)
            combined = pca.fit_transform(combined)
            real_use = combined[: len(real_use)]
            gen_use = combined[len(real_use):]

    return real_use, gen_use


def run_extractor_pipeline(
    extractor_name,
    real_features,
    generated_features,
    generated_datasets,
    real_images,
    args,
    output_dir,
):
    """Compute all metrics, save CSV, and produce plots for one extractor.

    Parameters
    ----------
    extractor_name : e.g. "Inception", "VAE", "DINOv2"
    real_features : (N_real, D) feature array
    generated_features : {gen_name: (N_gen, D)} dict
    generated_datasets : {gen_name: {'images': ..., 'paths': ...}}
    real_images : raw normalised images (for pixel-level plots, shared)
    args : parsed CLI args
    output_dir : per-extractor output folder
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Metrics ----
    print(f"\n{'='*70}")
    print(f"COMPUTING METRICS ({extractor_name})")
    print("=" * 70)

    all_metrics = {}
    fid_scores = {}
    precision_scores = {}
    coverage_scores = {}

    for gen_name, gen_feat in tqdm(generated_features.items(), desc=f"Metrics ({extractor_name})"):
        print(f"\n{gen_name} ({extractor_name}):")

        real_use, gen_use = _maybe_subsample_and_pca(real_features, gen_feat, args)

        n_metric_steps = 8  # 5 distribution + FID + Precision + Coverage
        metric_steps = tqdm(total=n_metric_steps, desc=f"  {gen_name} metrics", leave=False)

        metrics = compute_distribution_metrics(
            real_use, gen_use, progress=metric_steps, skip_kl=args.skip_kl
        )

        metric_steps.set_postfix_str("FID")
        fid = calculate_fid_score(real_use, gen_use)
        metric_steps.update(1)

        metric_steps.set_postfix_str("Precision & Coverage")
        prec, cov = compute_precision_coverage(real_use, gen_use, k=args.precision_coverage_k)
        metric_steps.update(1)
        metric_steps.update(1)

        metric_steps.close()

        all_metrics[gen_name] = metrics
        fid_scores[gen_name] = fid
        precision_scores[gen_name] = prec
        coverage_scores[gen_name] = cov

        print(f"  MMD:                        {metrics.get('MMD', np.nan):.6f}")
        print(f"  Wasserstein Distance:       {metrics.get('Wasserstein_Distance', np.nan):.6f}")
        print(f"  Mean Euclidean Distance:    {metrics.get('Mean_Euclidean_Distance', np.nan):.6f}")
        print(f"  KL Divergence:              {metrics.get('KL_Divergence', np.nan):.6f}")
        print(f"  Trace Difference:           {metrics.get('Trace_Difference', np.nan):.6f}")
        print(f"  FID Score:                  {fid:.4f}")
        print(f"  Precision (k={args.precision_coverage_k}):           {prec:.4f}")
        print(f"  Coverage  (k={args.precision_coverage_k}):           {cov:.4f}")

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print(f"METRICS SUMMARY ({extractor_name})")
    print("=" * 70)

    rows = []
    for gen_name in all_metrics:
        row = {"Generator": gen_name}
        row.update(all_metrics[gen_name])
        row["FID_Score"] = fid_scores[gen_name]
        row["Precision"] = precision_scores[gen_name]
        row["Coverage"] = coverage_scores[gen_name]
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ---- Ranking ----
    print(f"\n{'-'*70}")
    print(f"GENERATOR RANKING - {extractor_name} (by FID Score - Lower is Better)")
    print("-" * 70)
    ranked = sorted(fid_scores.items(), key=lambda x: x[1])
    for rank, (gen_name, fid) in enumerate(ranked, 1):
        print(f"  {rank}. {gen_name:.<30} FID = {fid:.4f}")
    best_gen, best_fid = ranked[0]
    print(f"\n  Best ({extractor_name}): {best_gen} (FID: {best_fid:.4f})")

    # ---- Visualisations ----
    gen_names = list(all_metrics.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(gen_names)))

    # -- t-SNE --
    if len(generated_features) > 0:
        print(f"Creating t-SNE plot ({extractor_name})...")
        num_gen = len(generated_features)
        fig, axes = plt.subplots(1, num_gen, figsize=(8 * num_gen, 6))
        if num_gen == 1:
            axes = [axes]

        for idx, (gen_name, gen_feat) in enumerate(generated_features.items()):
            combined = np.vstack([real_features, gen_feat])
            n_samples = combined.shape[0]
            if n_samples < 3:
                continue
            max_perp = max(2, (n_samples - 1) // 3)
            perplexity = min(args.tsne_perplexity, max_perp)
            tsne = TSNE(n_components=2, random_state=args.seed, perplexity=perplexity, max_iter=1000)
            tsne_feat = tsne.fit_transform(combined)

            labels = np.array(["Real"] * len(real_features) + ["Generated"] * len(gen_feat))
            ax = axes[idx]
            ax.scatter(tsne_feat[labels == "Real", 0], tsne_feat[labels == "Real", 1],
                       c="blue", label="Real", alpha=0.6, s=50, edgecolors="navy")
            ax.scatter(tsne_feat[labels == "Generated", 0], tsne_feat[labels == "Generated", 1],
                       c="red", label="Generated", alpha=0.6, s=50, edgecolors="darkred")
            ax.set_title(f"t-SNE: {gen_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"t-SNE Feature Distribution ({extractor_name})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tsne_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: tsne_distribution.png")

    # -- PCA --
    if len(generated_features) > 0:
        print(f"Creating PCA plot ({extractor_name})...")
        num_gen = len(generated_features)
        fig, axes = plt.subplots(1, num_gen, figsize=(8 * num_gen, 6))
        if num_gen == 1:
            axes = [axes]

        for idx, (gen_name, gen_feat) in enumerate(generated_features.items()):
            combined = np.vstack([real_features, gen_feat])
            pca = PCA(n_components=2)
            pca_feat = pca.fit_transform(combined)

            labels = np.array(["Real"] * len(real_features) + ["Generated"] * len(gen_feat))
            ax = axes[idx]
            ax.scatter(pca_feat[labels == "Real", 0], pca_feat[labels == "Real", 1],
                       c="blue", label="Real", alpha=0.6, s=50, edgecolors="navy")
            ax.scatter(pca_feat[labels == "Generated", 0], pca_feat[labels == "Generated", 1],
                       c="red", label="Generated", alpha=0.6, s=50, edgecolors="darkred")
            var = pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({var[0]:.2%})")
            ax.set_ylabel(f"PC2 ({var[1]:.2%})")
            ax.set_title(f"PCA: {gen_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"PCA Feature Distribution ({extractor_name})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pca_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: pca_distribution.png")

    # -- Metrics comparison bar charts (2x3: FID, MMD, WD, MeanDist, Precision, Coverage) --
    print(f"Creating metrics comparison plots ({extractor_name})...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].barh(gen_names, [fid_scores[n] for n in gen_names], color=colors, edgecolor="black")
    axes[0, 0].set_xlabel("FID Score (Lower is Better)")
    axes[0, 0].set_title("FID Score")
    axes[0, 0].invert_yaxis()

    axes[0, 1].barh(gen_names, [all_metrics[n].get("MMD", np.nan) for n in gen_names], color=colors, edgecolor="black")
    axes[0, 1].set_xlabel("MMD (Lower is Better)")
    axes[0, 1].set_title("Maximum Mean Discrepancy")
    axes[0, 1].invert_yaxis()

    axes[0, 2].barh(gen_names, [all_metrics[n].get("Wasserstein_Distance", np.nan) for n in gen_names], color=colors, edgecolor="black")
    axes[0, 2].set_xlabel("Wasserstein Distance (Lower is Better)")
    axes[0, 2].set_title("Wasserstein Distance")
    axes[0, 2].invert_yaxis()

    axes[1, 0].barh(gen_names, [all_metrics[n].get("Mean_Euclidean_Distance", np.nan) for n in gen_names], color=colors, edgecolor="black")
    axes[1, 0].set_xlabel("Mean Euclidean Distance (Lower is Better)")
    axes[1, 0].set_title("Mean Distance")
    axes[1, 0].invert_yaxis()

    axes[1, 1].barh(gen_names, [precision_scores[n] for n in gen_names], color=colors, edgecolor="black")
    axes[1, 1].set_xlabel("Precision (Higher is Better)")
    axes[1, 1].set_title(f"Precision (k={args.precision_coverage_k})")
    axes[1, 1].set_xlim(0, 1.05)
    axes[1, 1].invert_yaxis()

    axes[1, 2].barh(gen_names, [coverage_scores[n] for n in gen_names], color=colors, edgecolor="black")
    axes[1, 2].set_xlabel("Coverage (Higher is Better)")
    axes[1, 2].set_title(f"Coverage (k={args.precision_coverage_k})")
    axes[1, 2].set_xlim(0, 1.05)
    axes[1, 2].invert_yaxis()

    plt.suptitle(f"Distribution Metrics Comparison ({extractor_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: metrics_comparison.png")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_environment(args.output_dir, args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}\n")

    print("=" * 70)
    print("DISTRIBUTION SHIFT ANALYSIS")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------ #
    # 1. Load images
    # ------------------------------------------------------------------ #
    print("1. LOADING IMAGES\n" + "-" * 70)
    real_images, real_paths = load_images(args.real_dir, args.max_samples)
    print(f"Real images shape: {real_images.shape}\n")

    generated_folders = find_generated_folders(args.generated_dir)
    print(f"Found {len(generated_folders)} generator(s):\n")

    generated_datasets = {}
    for gen_name, gen_path in generated_folders.items():
        print(f"  - {gen_name}")
        images, paths = load_images(gen_path, args.max_samples)
        if len(images) > 0:
            generated_datasets[gen_name] = {"images": images, "paths": paths}
        print(f"    Loaded {len(images)} images")

    if len(real_images) == 0:
        print("\nNo real images found. Check --real_dir.")
        return 1
    if len(generated_datasets) == 0:
        print("\nNo generated images found. Check --generated_dir.")
        return 1

    print(f"\nLoaded {len(real_images)} real images and {len(generated_datasets)} generator(s)\n")

    # ------------------------------------------------------------------ #
    # 2. Extract features for all three extractors
    # ------------------------------------------------------------------ #
    print("2. EXTRACTING FEATURES\n" + "-" * 70)

    # --- Inception ---
    print("Loading InceptionV3 model...")
    inception = models.inception_v3(weights="IMAGENET1K_V1")
    inception.fc = torch.nn.Identity()
    inception.eval()
    inception = inception.to(device)

    print("Extracting Inception features from real images...")
    real_feat_inception = extract_features_inception(real_images, inception, device)
    print(f"  Real Inception features shape: {real_feat_inception.shape}")

    gen_feat_inception = {}
    for gen_name, gen_data in tqdm(generated_datasets.items(), desc="Generators (Inception)", leave=False):
        print(f"Extracting Inception features from {gen_name}...")
        gf = extract_features_inception(gen_data["images"], inception, device)
        gen_feat_inception[gen_name] = gf
        print(f"  {gen_name} Inception features shape: {gf.shape}")

    del inception
    torch.cuda.empty_cache()
    print()

    # --- VAE ---
    print("Loading VAE model for feature extraction...")
    vae_pipeline = StableFlowMatchingPipeline(
        device=str(device),
        t_scale=1.0,
        model_dir=".",
        from_norm_to_display=lambda x: (x + 1) / 2,
    ).build_from_configs(
        vae_json=args.vae_config,
    )
    vae_pipeline.load_vae_weights(args.vae_weights)
    vae_pipeline.vae.eval()
    vae_model = vae_pipeline.vae

    print("Extracting VAE features from real images...")
    real_feat_vae = extract_features_vae(real_images, vae_model, device)
    print(f"  Real VAE features shape: {real_feat_vae.shape}")

    gen_feat_vae = {}
    for gen_name, gen_data in tqdm(generated_datasets.items(), desc="Generators (VAE)", leave=False):
        print(f"Extracting VAE features from {gen_name}...")
        gf = extract_features_vae(gen_data["images"], vae_model, device)
        gen_feat_vae[gen_name] = gf
        print(f"  {gen_name} VAE features shape: {gf.shape}")

    del vae_model, vae_pipeline
    torch.cuda.empty_cache()
    print()

    # --- DINOv2 ---
    print(f"Loading DINOv2 model ({args.dino_model})...")
    dino = torch.hub.load("facebookresearch/dinov2", args.dino_model)
    dino.eval()
    dino = dino.to(device)

    print("Extracting DINOv2 features from real images...")
    real_feat_dino = extract_features_dinov2(real_images, dino, device)
    print(f"  Real DINOv2 features shape: {real_feat_dino.shape}")

    gen_feat_dino = {}
    for gen_name, gen_data in tqdm(generated_datasets.items(), desc="Generators (DINOv2)", leave=False):
        print(f"Extracting DINOv2 features from {gen_name}...")
        gf = extract_features_dinov2(gen_data["images"], dino, device)
        gen_feat_dino[gen_name] = gf
        print(f"  {gen_name} DINOv2 features shape: {gf.shape}")

    del dino
    torch.cuda.empty_cache()
    print()

    # ------------------------------------------------------------------ #
    # 3. Run per-extractor metric + visualisation pipeline
    # ------------------------------------------------------------------ #
    extractors = [
        ("Inception", real_feat_inception, gen_feat_inception),
        ("VAE", real_feat_vae, gen_feat_vae),
        ("DINOv2", real_feat_dino, gen_feat_dino),
    ]

    all_dfs = {}
    for ext_name, real_feat, gen_feat in extractors:
        ext_output_dir = os.path.join(args.output_dir, ext_name)
        df = run_extractor_pipeline(
            extractor_name=ext_name,
            real_features=real_feat,
            generated_features=gen_feat,
            generated_datasets=generated_datasets,
            real_images=real_images,
            args=args,
            output_dir=ext_output_dir,
        )
        all_dfs[ext_name] = df

    # ------------------------------------------------------------------ #
    # 4. Combined CSV with all extractors side-by-side
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("COMBINED METRICS SUMMARY (all extractors)")
    print("=" * 70)

    gen_names = list(generated_datasets.keys())
    combined_rows = []
    for gen_name in gen_names:
        row = {"Generator": gen_name}
        for ext_name, df in all_dfs.items():
            df_row = df[df["Generator"] == gen_name]
            if len(df_row) == 0:
                continue
            for col in df.columns:
                if col == "Generator":
                    continue
                row[f"{ext_name}_{col}"] = df_row[col].values[0]
        combined_rows.append(row)

    combined_df = pd.DataFrame(combined_rows)
    combined_csv = os.path.join(args.output_dir, "metrics_summary_combined.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(combined_df.to_string(index=False))
    print(f"\nSaved: {combined_csv}")

    # ------------------------------------------------------------------ #
    # 5. Extractor-agnostic plots (pixel stats, sample grids, KL)
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("CREATING SHARED VISUALISATIONS")
    print("=" * 70)

    # Sample comparison
    print("Creating sample comparisons...")
    num_samples = min(4, len(real_images))
    real_indices = np.random.choice(len(real_images), num_samples, replace=False)

    for gen_name, gen_data in generated_datasets.items():
        gen_images = gen_data["images"]
        num_samples_gen = min(num_samples, len(gen_images))
        gen_indices = np.random.choice(len(gen_images), num_samples_gen, replace=False)

        fig, axes = plt.subplots(num_samples_gen, 2, figsize=(8, 4 * num_samples_gen))
        if num_samples_gen == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples_gen):
            axes[i, 0].imshow(real_images[real_indices[i]], cmap="gray", vmin=-1, vmax=1)
            axes[i, 0].set_title("Real", fontsize=11, fontweight="bold")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gen_images[gen_indices[i]], cmap="gray", vmin=-1, vmax=1)
            axes[i, 1].set_title(gen_name, fontsize=11, fontweight="bold")
            axes[i, 1].axis("off")

        plt.suptitle(f"Real vs {gen_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"samples_{gen_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: samples_{gen_name}.png")

    # Pixel statistics
    print("Creating pixel statistics plots...")
    real_gray = real_images
    num_generators = len(generated_datasets)
    fig, axes = plt.subplots(3, num_generators, figsize=(6 * num_generators, 12))
    if num_generators == 1:
        axes = axes.reshape(-1, 1)

    for gen_idx, (gen_name, gen_data) in enumerate(generated_datasets.items()):
        gen_images = gen_data["images"]
        gen_gray = gen_images

        axes[0, gen_idx].hist(real_gray.flatten(), bins=50, alpha=0.7, label="Real", color="blue", density=True)
        axes[0, gen_idx].hist(gen_gray.flatten(), bins=50, alpha=0.7, label="Generated", color="red", density=True)
        axes[0, gen_idx].set_title(f"{gen_name}: Intensity")
        axes[0, gen_idx].legend()
        axes[0, gen_idx].grid(True, alpha=0.3)

        real_means = np.mean(real_images, axis=(1, 2))
        gen_means = np.mean(gen_images, axis=(1, 2))
        axes[1, gen_idx].hist(real_means, bins=30, alpha=0.7, label="Real", color="blue", density=True)
        axes[1, gen_idx].hist(gen_means, bins=30, alpha=0.7, label="Generated", color="red", density=True)
        axes[1, gen_idx].set_title(f"{gen_name}: Mean Values")
        axes[1, gen_idx].legend()
        axes[1, gen_idx].grid(True, alpha=0.3)

        real_stds = np.std(real_images, axis=(1, 2))
        gen_stds = np.std(gen_images, axis=(1, 2))
        axes[2, gen_idx].hist(real_stds, bins=30, alpha=0.7, label="Real", color="blue", density=True)
        axes[2, gen_idx].hist(gen_stds, bins=30, alpha=0.7, label="Generated", color="red", density=True)
        axes[2, gen_idx].set_title(f"{gen_name}: Std Deviation")
        axes[2, gen_idx].legend()
        axes[2, gen_idx].grid(True, alpha=0.3)

    plt.suptitle("Pixel Statistics Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pixel_statistics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pixel_statistics.png")

    # KL divergence
    print("Creating KL divergence plots...")
    fig, axes = plt.subplots(2, num_generators, figsize=(8 * num_generators, 10))
    if num_generators == 1:
        axes = axes.reshape(2, -1)

    for gen_idx, (gen_name, gen_data) in enumerate(generated_datasets.items()):
        gen_images = gen_data["images"]
        gen_gray = gen_images

        kl_div, hist_real, hist_gen = compute_kl_divergence_discrete(real_gray, gen_gray)
        bins_range = np.arange(len(hist_real))

        axes[0, gen_idx].bar(bins_range - 0.2, hist_real, width=0.4, label="Real", alpha=0.7, color="blue")
        axes[0, gen_idx].bar(bins_range + 0.2, hist_gen, width=0.4, label="Generated", alpha=0.7, color="red")
        axes[0, gen_idx].set_title(f"{gen_name}: Histogram")
        axes[0, gen_idx].legend()
        axes[0, gen_idx].grid(True, alpha=0.3)

        axes[1, gen_idx].barh([gen_name], [kl_div], color="steelblue", edgecolor="black", height=0.5)
        axes[1, gen_idx].set_title(f"KL Divergence: {kl_div:.4f}")

    plt.suptitle("KL Divergence Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "kl_divergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: kl_divergence.png")

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  Per-extractor folders:")
    for ext_name in ["Inception", "VAE", "DINOv2"]:
        ext_dir = os.path.join(args.output_dir, ext_name)
        print(f"    {ext_dir}/")
        print(f"      - metrics_summary.csv")
        print(f"      - tsne_distribution.png")
        print(f"      - pca_distribution.png")
        print(f"      - metrics_comparison.png")
    print(f"  Combined:")
    print(f"    - metrics_summary_combined.csv")
    print(f"    - samples_*.png")
    print(f"    - pixel_statistics.png")
    print(f"    - kl_divergence.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
