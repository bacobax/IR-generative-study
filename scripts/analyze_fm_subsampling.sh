#!/usr/bin/env bash
set -euo pipefail

# Find project root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

# Define paths and parameters
OUT_DIR="$ROOT_DIR/artifacts/analysis/main/subsampling_coverage"
FM_PIPELINE_DIR="$ROOT_DIR/artifacts/checkpoints/flow_matching/serious_runs/stable_training_no_norm"
VAE_WEIGHTS="$ROOT_DIR/artifacts/checkpoints/vae/vae_runs/vae_fm_x4/VAE/vae_best.pt"
SURPRISE_CKPT="$ROOT_DIR/artifacts/runs/main/surprise_predictor_longrun/vae_x4_best_minmax_h256_s0/best_model.pt"
REAL_DATA_ROOT="$ROOT_DIR/data/raw/v18"

# Launch analysis
python "$ROOT_DIR/scripts/analyze_fm_subsampling_coverage.py" \
    --out_dir "$OUT_DIR" \
    --fm_pipeline_dir "$FM_PIPELINE_DIR" \
    --vae_weights "$VAE_WEIGHTS" \
    --surprise_ckpt "$SURPRISE_CKPT" \
    --real_data_root "$REAL_DATA_ROOT" \
    --split test \
    --N 1000 \
    --K 500 \
    --top_pct 0.1 \
    --batch_size 32 \
    --device cuda:2 \
    --reuse_cache \
    --cache_tokens \
    --max_n_real 1000
