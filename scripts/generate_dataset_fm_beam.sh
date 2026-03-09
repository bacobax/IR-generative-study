#!/usr/bin/env bash
# FM generation: beam sampling pruned by surprise predictor energy.
# Maintains beam_size parallel trajectories; at each step each beam is
# expanded into branch_factor perturbed copies and the best beam_size
# are kept according to the predictor energy.
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
    --output_dir "$ROOT_DIR/generated/fm_beam_surprise" \
    --fm_steps 100 \
    --fm_batch_size 2 \
    --fm_guidance_method beam \
    --fm_surprise_ckpt "$ROOT_DIR/runs/surprise_predictor_longer_run/vae_x4_best_minmax_h256_s0/best_model.pt" \
    --fm_energy_mode surprise \
    --fm_sign minimize \
    --fm_beam_size 4 \
    --fm_branch_factor 2 \
    --fm_sigma_perturb 0.05 \
    --device cuda:2
