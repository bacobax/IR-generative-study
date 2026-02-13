#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

python "$ROOT_DIR/analyze_distribution_shift.py" \
    --real_dir "$ROOT_DIR/v18/images" \
    --generated_dir "$ROOT_DIR/generated" \
    --output_dir "$ROOT_DIR/analysis_results" \
    --max_samples 500 \
    --metrics_max_samples 500 \
    --device cuda:0 \
    --metrics_pca_dim 128 \
    --tsne_perplexity 20 \