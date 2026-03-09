#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

# Unstable (pixel-space) Flow Matching Training Script
# Usage: ./run_unstable_training.sh
# To resume: RESUME_CKPT=./path/to/checkpoint.pt ./run_unstable_training.sh

# Data paths
TRAIN_DIR="$ROOT_DIR/v18/train/"
VAL_DIR="$ROOT_DIR/v18/val/"

# Model configs
UNET_CONFIG="$ROOT_DIR/fm_src/non_stable_unet_config.json"

# Output directory
MODEL_DIR="$ROOT_DIR/serious_runs/pixel_fm_x0/"

# Training params
EPOCHS=300
BATCH_SIZE=8
NUM_WORKERS=4
SAVE_EVERY_N_EPOCHS=30
SAMPLE_BATCH_SIZE=4
T_SCALE=1000
TRAIN_TARGET="x0"

# Augmentation schedule (warmup -> ramp -> decay)
WARMUP_FRAC=0.1
RAMP_FRAC=0.4

P_CROP_WARMUP=0.05
P_CROP_MAX=0.20
P_CROP_FINAL=0.05

P_ROT_WARMUP=0.05
P_ROT_MAX=0.30
P_ROT_FINAL=0.05

# Resume checkpoint (leave empty for fresh training)
# RESUME_CKPT="$ROOT_DIR/serious_runs/pixel_fm_x0/UNET/unet_fm_epoch_150_ckpt.pt"
RESUME_CKPT=
# Build the command
CMD=(
    python "$ROOT_DIR/train_fm.py"
    --train_dir "$TRAIN_DIR"
    --val_dir "$VAL_DIR"
    --unet_config "$UNET_CONFIG"
    --model_dir "$MODEL_DIR"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
    --sample_batch_size "$SAMPLE_BATCH_SIZE"
    --t_scale "$T_SCALE"
    --train-target "$TRAIN_TARGET"
    --warmup_frac "$WARMUP_FRAC"
    --ramp_frac "$RAMP_FRAC"
    --p_crop_warmup "$P_CROP_WARMUP"
    --p_crop_max "$P_CROP_MAX"
    --p_crop_final "$P_CROP_FINAL"
    --p_rot_warmup "$P_ROT_WARMUP"
    --p_rot_max "$P_ROT_MAX"
    --p_rot_final "$P_ROT_FINAL"
)

# Add resume flag if checkpoint provided
if [ -n "$RESUME_CKPT" ]; then
    CMD+=(--resume "$RESUME_CKPT")
    echo "Resuming from checkpoint: $RESUME_CKPT"
fi

echo "Starting training..."
printf '%q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
