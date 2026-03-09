#!/usr/bin/env bash
# FM generation: plain Euler followed by latent gradient refinement.
# Post-sampling refinement runs num_refine_steps steps of gradient
# descent on the surprise energy in latent space.
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
    --output_dir "$ROOT_DIR/generated/fm_refine_surprise" \
    --fm_steps 100 \
    --fm_batch_size 4 \
    --fm_guidance_method refine \
    --fm_surprise_ckpt "$ROOT_DIR/runs/surprise_predictor/best_model.pt" \
    --fm_energy_mode surprise \
    --fm_sign maximize \
    --fm_num_refine_steps 20 \
    --fm_refine_step_size 0.005 \
    --fm_grad_clip_norm 1.0 \
    --device cuda:2
