#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

python "$ROOT_DIR/scripts/analyze_distribution_shift.py" \
    --real_dir "$ROOT_DIR/data/raw/v18/test" \
    --generated_dir "$ROOT_DIR/artifacts/generated/main" \
    --output_dir "$ROOT_DIR/artifacts/analysis/main" \
    --max_samples 500 \
    --metrics_max_samples 500 \
    --metrics_pca_dim 128 \
    --tsne_perplexity 20 \
    --precision_coverage_k 5 \
    --dino_model "dinov2_vits14" \
    --vae_config "$ROOT_DIR/artifacts/checkpoints/vae/vae_runs/vae_fm_x4/VAE/config.json" \
    --vae_weights "$ROOT_DIR/artifacts/checkpoints/vae/vae_runs/vae_fm_x4/VAE/vae_best.pt" \
    --device cuda:0