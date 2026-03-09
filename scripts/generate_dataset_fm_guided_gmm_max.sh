#!/usr/bin/env bash
# Guided FM generation: Euler + GMM energy, MINIMISE
# Guidance pushes toward samples better fitting the GMM (higher GMM probability).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

python "$ROOT_DIR/generate_datasets.py" \
    --mode fm \
    --fm_pipeline_dir "$ROOT_DIR/serious_runs/stable_training_no_norm_longrun" \
    --fm_vae_weights  "$ROOT_DIR/serious_runs/stable_training_no_norm_longrun/VAE/vae_best.pt" \
    --max_samples 200 \
    --output_dir "$ROOT_DIR/generated/fm_guided_gmm_max" \
    --fm_steps 100 \
    --fm_batch_size 4 \
    --fm_guidance_method euler_guided \
    --fm_surprise_ckpt "$ROOT_DIR/runs/surprise_predictor_longer_run/vae_x4_best_minmax_h256_s0/best_model.pt" \
    --fm_energy_mode gmm \
    --fm_sign maximize \
    --fm_guidance_scale 1.2 \
    --fm_lambda_start 0.2 \
    --fm_lambda_end 2 \
    --fm_lambda_schedule cosine \
    --fm_grad_clip_norm 1.0 \
    --fm_use_ddim_hat \
    --device cuda:2
