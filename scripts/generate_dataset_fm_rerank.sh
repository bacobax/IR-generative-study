#!/usr/bin/env bash
# FM generation: rejection reranking via surprise predictor energy.
# Generates fm_n_candidates independent Euler trajectories per slot,
# then keeps the one with the lowest surprise energy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

python "$ROOT_DIR/generate_datasets.py" \
    --mode fm \
    --fm_pipeline_dir "$ROOT_DIR/serious_runs/stable_training_no_norm" \
    --fm_vae_weights  "$ROOT_DIR/serious_runs/stable_training_no_norm/VAE/vae_best.pt" \
    --max_samples 200 \
    --output_dir "$ROOT_DIR/generated/fm_rerank_surprise" \
    --fm_steps 100 \
    --fm_batch_size 4 \
    --fm_guidance_method rerank \
    --fm_surprise_ckpt "$ROOT_DIR/runs/surprise_predictor/best_model.pt" \
    --fm_energy_mode surprise \
    --fm_sign minimize \
    --fm_n_candidates 8 \
    --device cuda:2
