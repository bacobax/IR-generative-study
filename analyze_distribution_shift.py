#!/usr/bin/env python3
"""
Distribution Shift Analysis: Real vs Generated Images
Analyzes multiple generators and compares their output distributions to real data.
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.models as models
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distribution shift analysis for real vs generated IR samples (.npy)."
    )
    parser.add_argument("--real_dir", type=str, default="./v18/images",
                        help="Folder containing real .npy samples.")
    parser.add_argument("--generated_dir", type=str, default="./generated",
                        help="Folder containing generated subfolders with .npy files.")
    parser.add_argument("--output_dir", type=str, default="./analysis_results",
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
    return parser.parse_args()


def setup_environment(output_dir: str, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    os.makedirs(output_dir, exist_ok=True)


# Raw IR normalization constants (from sd_src/helpers.py)
_A_RAW = 11667.0
_B_RAW = 13944.0


def _normalize_to_m1p1(arr: np.ndarray) -> np.ndarray:
    """Normalize raw IR values to [-1, 1]."""
    return (2.0 * np.clip((arr - _A_RAW) / (_B_RAW - _A_RAW), 0.0, 1.0) - 1.0).astype(np.float32)


def _needs_normalization(images: np.ndarray) -> bool:
    """Detect whether images are raw IR (large values) vs already [-1, 1]."""
    sample = images[:min(5, len(images))]
    return float(np.abs(sample).max()) > 2.0


def load_images(image_path, max_images=None, normalize=True):
    """Load 1-channel .npy images from a directory.

    If images appear to be raw IR values (range >> [-1,1]),
    they are automatically normalized to [-1,1].
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
            arr = np.load(img_file).astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")
            images.append(arr)
            image_paths.append(img_file)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue

    result = np.array(images)
    if normalize and len(result) > 0 and _needs_normalization(result):
        print(f"  → Auto-normalizing raw IR values to [-1, 1]")
        result = _normalize_to_m1p1(result)

    return result, image_paths


def find_generated_folders(base_path):
    """Find all folders containing generated images."""
    folders = {}
    
    if os.path.isdir(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                image_files = glob.glob(os.path.join(item_path, "**/*.npy"), recursive=True)
                if len(image_files) > 0:
                    folders[item] = item_path
    
    return folders


def extract_features(images, model, device, batch_size=16):
    """Extract features from 1-channel images using InceptionV3."""
    features = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Extracting features"):
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


def compute_distribution_metrics(real_features, gen_features, progress=None, skip_kl=False):
    """Compute distribution shift metrics."""
    metrics = {}
    
    if len(real_features) == 0 or len(gen_features) == 0:
        return metrics
    
    def mmd(x, y):
        xx = np.dot(x, x.T)
        yy = np.dot(y, y.T)
        xy = np.dot(x, y.T)
        return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)
    
    real_feat_norm = (real_features - real_features.mean(axis=0)) / (real_features.std(axis=0) + 1e-8)
    gen_feat_norm = (gen_features - gen_features.mean(axis=0)) / (gen_features.std(axis=0) + 1e-8)
    
    if progress is not None:
        progress.set_postfix_str("MMD")
    metrics['MMD'] = mmd(real_feat_norm, gen_feat_norm)
    if progress is not None:
        progress.update(1)
    
    real_agg = real_features.mean(axis=0)
    gen_agg = gen_features.mean(axis=0)
    if progress is not None:
        progress.set_postfix_str("Wasserstein")
    metrics['Wasserstein_Distance'] = wasserstein_distance(real_agg, gen_agg)
    if progress is not None:
        progress.update(1)

    if progress is not None:
        progress.set_postfix_str("MeanDist")
    metrics['Mean_Euclidean_Distance'] = np.linalg.norm(real_agg - gen_agg)
    if progress is not None:
        progress.update(1)
    
    real_cov = np.cov(real_features.T) + 1e-8 * np.eye(real_features.shape[1])
    gen_cov = np.cov(gen_features.T) + 1e-8 * np.eye(gen_features.shape[1])
    
    if skip_kl:
        metrics['KL_Divergence'] = np.nan
        if progress is not None:
            progress.update(1)
    else:
        if progress is not None:
            progress.set_postfix_str("KL")
        try:
            kl_div = 0.5 * (np.trace(np.linalg.inv(gen_cov) @ real_cov) + 
                            np.sum((gen_agg - real_agg) ** 2 / np.diag(gen_cov)) -
                            real_features.shape[1] + 
                            np.linalg.slogdet(gen_cov)[1] - np.linalg.slogdet(real_cov)[1])
            metrics['KL_Divergence'] = kl_div
        except Exception:
            metrics['KL_Divergence'] = np.inf
        if progress is not None:
            progress.update(1)
    
    if progress is not None:
        progress.set_postfix_str("Trace")
    metrics['Trace_Difference'] = np.trace(real_cov) - np.trace(gen_cov)
    if progress is not None:
        progress.update(1)
    
    return metrics


def calculate_fid_score(real_features, gen_features):
    """Calculate Fréchet Inception Distance (FID)."""
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


def main():
    # ============================================================================
    # MAIN ANALYSIS
    # ============================================================================
    args = parse_args()
    setup_environment(args.output_dir, args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}\n")

    print("="*70)
    print("DISTRIBUTION SHIFT ANALYSIS")
    print("="*70 + "\n")

    # Load images
    print("1. LOADING IMAGES\n" + "-"*70)
    real_images, real_paths = load_images(args.real_dir, args.max_samples)
    print(f"Real images shape: {real_images.shape}\n")

    generated_folders = find_generated_folders(args.generated_dir)
    print(f"Found {len(generated_folders)} generator(s):\n")

    generated_datasets = {}
    for gen_name, gen_path in generated_folders.items():
        print(f"  - {gen_name}")
        images, paths = load_images(gen_path, args.max_samples)
        if len(images) > 0:
            generated_datasets[gen_name] = {'images': images, 'paths': paths}
        print(f"    Loaded {len(images)} images")

    if len(real_images) == 0:
        print("\nNo real images found. Check --real_dir.")
        return 1

    if len(generated_datasets) == 0:
        print("\nNo generated images found. Check --generated_dir.")
        return 1

    print(f"\n✓ Successfully loaded {len(real_images)} real images and {len(generated_datasets)} generator(s)\n")

    # Extract features
    print("2. EXTRACTING FEATURES\n" + "-"*70)
    print("Loading InceptionV3 model...")
    inception = models.inception_v3(weights='IMAGENET1K_V1')
    inception.fc = torch.nn.Identity()
    inception.eval()
    inception = inception.to(device)

    print("Extracting features from real images...")
    real_features = extract_features(real_images, inception, device)
    print(f"Real features shape: {real_features.shape}")

    generated_features = {}
    for gen_name, gen_data in tqdm(generated_datasets.items(), desc="Generators", leave=False):
        print(f"Extracting features from {gen_name}...")
        gen_feat = extract_features(gen_data['images'], inception, device)
        generated_features[gen_name] = gen_feat
        print(f"  {gen_name} features shape: {gen_feat.shape}")

    print()

    # Compute metrics
    print("3. COMPUTING METRICS\n" + "-"*70)
    all_metrics = {}
    fid_scores = {}

    for gen_name, gen_feat in tqdm(generated_features.items(), desc="Metrics"):
        print(f"\n{gen_name}:")
        real_feat_use = real_features
        gen_feat_use = gen_feat
        if args.metrics_max_samples is not None:
            n_real = min(args.metrics_max_samples, len(real_features))
            n_gen = min(args.metrics_max_samples, len(gen_feat))
            real_idx = np.random.choice(len(real_features), n_real, replace=False)
            gen_idx = np.random.choice(len(gen_feat), n_gen, replace=False)
            real_feat_use = real_features[real_idx]
            gen_feat_use = gen_feat[gen_idx]

        if args.metrics_pca_dim and args.metrics_pca_dim > 0:
            combined = np.vstack([real_feat_use, gen_feat_use])
            max_components = min(combined.shape[0], combined.shape[1])
            n_components = min(args.metrics_pca_dim, max_components)
            if n_components >= 1 and n_components < combined.shape[1]:
                pca = PCA(n_components=n_components)
                combined = pca.fit_transform(combined)
                real_feat_use = combined[: len(real_feat_use)]
                gen_feat_use = combined[len(real_feat_use) :]

        metric_steps = tqdm(total=6, desc=f"  {gen_name} metrics", leave=False)
        metrics = compute_distribution_metrics(
            real_feat_use, gen_feat_use, progress=metric_steps, skip_kl=args.skip_kl
        )
        metric_steps.set_postfix_str("FID")
        fid = calculate_fid_score(real_feat_use, gen_feat_use)
        metric_steps.update(1)
        metric_steps.close()

        all_metrics[gen_name] = metrics
        fid_scores[gen_name] = fid

        print(f"  MMD:                        {metrics.get('MMD', np.nan):.6f}")
        print(f"  Wasserstein Distance:       {metrics.get('Wasserstein_Distance', np.nan):.6f}")
        print(f"  Mean Euclidean Distance:    {metrics.get('Mean_Euclidean_Distance', np.nan):.6f}")
        print(f"  KL Divergence:              {metrics.get('KL_Divergence', np.nan):.6f}")
        print(f"  FID Score:                  {fid:.4f}")

    # Create summary table
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    metrics_rows = []
    for gen_name in all_metrics.keys():
        row = {'Generator': gen_name}
        row.update(all_metrics[gen_name])
        row['FID_Score'] = fid_scores.get(gen_name, np.nan)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    print(metrics_df.to_string(index=False))

    # Ranking
    print("\n" + "-"*70)
    print("GENERATOR RANKING (by FID Score - Lower is Better)")
    print("-"*70)
    ranked = sorted(fid_scores.items(), key=lambda x: x[1])
    for rank, (gen_name, fid) in enumerate(ranked, 1):
        print(f"  {rank}. {gen_name:.<30} FID = {fid:.4f}")

    best_gen, best_fid = ranked[0]
    print(f"\n  ✓ Best: {best_gen} (FID: {best_fid:.4f})")

    # Save metrics
    metrics_df.to_csv(os.path.join(args.output_dir, 'metrics_summary.csv'), index=False)
    print(f"\n✓ Metrics saved to {args.output_dir}/metrics_summary.csv\n")

    # Visualizations
    print("4. CREATING VISUALIZATIONS\n" + "-"*70)

    # t-SNE
    if len(generated_features) > 0:
        print("Creating t-SNE plots...")
        num_generators = len(generated_features)
        fig, axes = plt.subplots(1, num_generators, figsize=(8*num_generators, 6))

        if num_generators == 1:
            axes = [axes]

        for idx, (gen_name, gen_feat) in enumerate(generated_features.items()):
            combined = np.vstack([real_features, gen_feat])
            n_samples = combined.shape[0]
            if n_samples < 3:
                print(f"  [t-SNE] Skipping {gen_name} (n_samples={n_samples})")
                continue
            max_perplexity = max(2, (n_samples - 1) // 3)
            perplexity = min(args.tsne_perplexity, max_perplexity)
            tsne = TSNE(n_components=2, random_state=args.seed, perplexity=perplexity, max_iter=1000)
            tsne_feat = tsne.fit_transform(combined)

            labels = np.array(['Real'] * len(real_features) + ['Generated'] * len(gen_feat))

            ax = axes[idx]
            ax.scatter(tsne_feat[labels == 'Real', 0], tsne_feat[labels == 'Real', 1],
                       c='blue', label='Real', alpha=0.6, s=50, edgecolors='navy')
            ax.scatter(tsne_feat[labels == 'Generated', 0], tsne_feat[labels == 'Generated', 1],
                       c='red', label='Generated', alpha=0.6, s=50, edgecolors='darkred')
            ax.set_title(f't-SNE: {gen_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('t-SNE Feature Distribution Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'tsne_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: tsne_distribution.png")

    # PCA
    print("Creating PCA plots...")
    num_generators = len(generated_features)
    fig, axes = plt.subplots(1, num_generators, figsize=(8*num_generators, 6))

    if num_generators == 1:
        axes = [axes]

    for idx, (gen_name, gen_feat) in enumerate(generated_features.items()):
        combined = np.vstack([real_features, gen_feat])
        pca = PCA(n_components=2)
        pca_feat = pca.fit_transform(combined)

        labels = np.array(['Real'] * len(real_features) + ['Generated'] * len(gen_feat))

        ax = axes[idx]
        ax.scatter(pca_feat[labels == 'Real', 0], pca_feat[labels == 'Real', 1],
                   c='blue', label='Real', alpha=0.6, s=50, edgecolors='navy')
        ax.scatter(pca_feat[labels == 'Generated', 0], pca_feat[labels == 'Generated', 1],
                   c='red', label='Generated', alpha=0.6, s=50, edgecolors='darkred')

        var = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var[0]:.2%})')
        ax.set_ylabel(f'PC2 ({var[1]:.2%})')
        ax.set_title(f'PCA: {gen_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('PCA Feature Distribution Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'pca_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: pca_distribution.png")

    # Metrics comparison
    print("Creating metrics comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fid_names = list(fid_scores.keys())
    fid_vals = list(fid_scores.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(fid_names)))

    axes[0, 0].barh(fid_names, fid_vals, color=colors, edgecolor='black')
    axes[0, 0].set_xlabel('FID Score (Lower is Better)')
    axes[0, 0].set_title('FID Score Comparison')
    axes[0, 0].invert_yaxis()

    mmd_vals = [all_metrics[n].get('MMD', np.nan) for n in fid_names]
    axes[0, 1].barh(fid_names, mmd_vals, color=colors, edgecolor='black')
    axes[0, 1].set_xlabel('MMD (Lower is Better)')
    axes[0, 1].set_title('Maximum Mean Discrepancy')
    axes[0, 1].invert_yaxis()

    wd_vals = [all_metrics[n].get('Wasserstein_Distance', np.nan) for n in fid_names]
    axes[1, 0].barh(fid_names, wd_vals, color=colors, edgecolor='black')
    axes[1, 0].set_xlabel('Wasserstein Distance (Lower is Better)')
    axes[1, 0].set_title('Wasserstein Distance')
    axes[1, 0].invert_yaxis()

    me_vals = [all_metrics[n].get('Mean_Euclidean_Distance', np.nan) for n in fid_names]
    axes[1, 1].barh(fid_names, me_vals, color=colors, edgecolor='black')
    axes[1, 1].set_xlabel('Mean Euclidean Distance (Lower is Better)')
    axes[1, 1].set_title('Mean Distance')
    axes[1, 1].invert_yaxis()

    plt.suptitle('Distribution Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: metrics_comparison.png")

    # Sample comparison
    print("Creating sample comparisons...")
    num_samples = min(4, len(real_images))
    real_indices = np.random.choice(len(real_images), num_samples, replace=False)

    for gen_name, gen_data in generated_datasets.items():
        gen_images = gen_data['images']
        num_samples_gen = min(num_samples, len(gen_images))
        gen_indices = np.random.choice(len(gen_images), num_samples_gen, replace=False)

        fig, axes = plt.subplots(num_samples_gen, 2, figsize=(8, 4*num_samples_gen))

        if num_samples_gen == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples_gen):
            axes[i, 0].imshow(real_images[real_indices[i]], cmap='gray', vmin=-1, vmax=1)
            axes[i, 0].set_title('Real', fontsize=11, fontweight='bold')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gen_images[gen_indices[i]], cmap='gray', vmin=-1, vmax=1)
            axes[i, 1].set_title(gen_name, fontsize=11, fontweight='bold')
            axes[i, 1].axis('off')

        plt.suptitle(f'Real vs {gen_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'samples_{gen_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: samples_{gen_name}.png")

    # Pixel statistics
    print("Creating pixel statistics plots...")
    real_gray = real_images

    num_generators = len(generated_datasets)
    fig, axes = plt.subplots(3, num_generators, figsize=(6*num_generators, 12))

    if num_generators == 1:
        axes = axes.reshape(-1, 1)

    for gen_idx, (gen_name, gen_data) in enumerate(generated_datasets.items()):
        gen_images = gen_data['images']
        gen_gray = gen_images

        axes[0, gen_idx].hist(real_gray.flatten(), bins=50, alpha=0.7, label='Real', color='blue', density=True)
        axes[0, gen_idx].hist(gen_gray.flatten(), bins=50, alpha=0.7, label='Generated', color='red', density=True)
        axes[0, gen_idx].set_title(f'{gen_name}: Intensity')
        axes[0, gen_idx].legend()
        axes[0, gen_idx].grid(True, alpha=0.3)

        real_means = np.mean(real_images, axis=(1, 2))
        gen_means = np.mean(gen_images, axis=(1, 2))

        axes[1, gen_idx].hist(real_means, bins=30, alpha=0.7, label='Real', color='blue', density=True)
        axes[1, gen_idx].hist(gen_means, bins=30, alpha=0.7, label='Generated', color='red', density=True)
        axes[1, gen_idx].set_title(f'{gen_name}: Mean Values')
        axes[1, gen_idx].legend()
        axes[1, gen_idx].grid(True, alpha=0.3)

        real_stds = np.std(real_images, axis=(1, 2))
        gen_stds = np.std(gen_images, axis=(1, 2))

        axes[2, gen_idx].hist(real_stds, bins=30, alpha=0.7, label='Real', color='blue', density=True)
        axes[2, gen_idx].hist(gen_stds, bins=30, alpha=0.7, label='Generated', color='red', density=True)
        axes[2, gen_idx].set_title(f'{gen_name}: Std Deviation')
        axes[2, gen_idx].legend()
        axes[2, gen_idx].grid(True, alpha=0.3)

    plt.suptitle('Pixel Statistics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'pixel_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: pixel_statistics.png")

    # KL divergence
    print("Creating KL divergence plots...")
    fig, axes = plt.subplots(2, num_generators, figsize=(8*num_generators, 10))

    if num_generators == 1:
        axes = axes.reshape(2, -1)

    for gen_idx, (gen_name, gen_data) in enumerate(generated_datasets.items()):
        gen_images = gen_data['images']
        gen_gray = gen_images

        kl_div, hist_real, hist_gen = compute_kl_divergence_discrete(real_gray, gen_gray)

        bins_range = np.arange(len(hist_real))
        axes[0, gen_idx].bar(bins_range - 0.2, hist_real, width=0.4, label='Real', alpha=0.7, color='blue')
        axes[0, gen_idx].bar(bins_range + 0.2, hist_gen, width=0.4, label='Generated', alpha=0.7, color='red')
        axes[0, gen_idx].set_title(f'{gen_name}: Histogram')
        axes[0, gen_idx].legend()
        axes[0, gen_idx].grid(True, alpha=0.3)

        axes[1, gen_idx].barh([gen_name], [kl_div], color='steelblue', edgecolor='black', height=0.5)
        axes[1, gen_idx].set_title(f'KL Divergence: {kl_div:.4f}')

    plt.suptitle('KL Divergence Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'kl_divergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: kl_divergence.png")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - metrics_summary.csv")
    print(f"  - tsne_distribution.png")
    print(f"  - pca_distribution.png")
    print(f"  - metrics_comparison.png")
    print(f"  - samples_*.png (for each generator)")
    print(f"  - pixel_statistics.png")
    print(f"  - kl_divergence.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
