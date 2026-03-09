#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

# =====================================================================
# ControlNet Flow Matching Training  (Stage 2)
#
# Requires a trained stage-1 pipeline folder that contains:
#   UNET/config.json + unet_fm_best.pt   (or unet_fm_epoch_*.pt)
#   VAE/config.json  + vae_best.pt        (or vae_epoch_*.pt)
# =====================================================================

# ---------- Data paths ----------
TRAIN_DIR="$ROOT_DIR/v18/train/"
VAL_DIR="$ROOT_DIR/v18/val/"
TRAIN_ANNOTATIONS="$ROOT_DIR/v18/train/annotations.json"
VAL_ANNOTATIONS="$ROOT_DIR/v18/val/annotations.json"

# ---------- Stage-1 pipeline (frozen UNet + VAE) ----------
STAGE1_PIPELINE_DIR="$ROOT_DIR/serious_runs/stable_training_t_scaled/"

# ---------- Output ----------
MODEL_DIR="$ROOT_DIR/controlnet_runs/bbox_controlnet/"

# ---------- Training ----------
EPOCHS=100
BATCH_SIZE=8
NUM_WORKERS=4
LR=1e-4
CONDITIONING_SCALE=1.0
CONDITIONING_DROPOUT=0.1
SAVE_EVERY_N_EPOCHS=10
SAMPLE_STEPS=50
T_SCALE=1000

# ---------- Resume (uncomment to resume) ----------
# RESUME_CKPT="$ROOT_DIR/controlnet_runs/bbox_controlnet/CONTROLNET/controlnet_epoch_10_ckpt.pt"

# ---------- Build command ----------
CMD=(
    conda run --no-capture-output -n diffusers-dev
    python "$ROOT_DIR/train_controlnet.py"
    --train_dir "$TRAIN_DIR"
    --val_dir "$VAL_DIR"
    --train_annotations "$TRAIN_ANNOTATIONS"
    --val_annotations "$VAL_ANNOTATIONS"
    --stage1_pipeline_dir "$STAGE1_PIPELINE_DIR"
    --model_dir "$MODEL_DIR"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --lr "$LR"
    --conditioning_scale "$CONDITIONING_SCALE"
    --conditioning_dropout "$CONDITIONING_DROPOUT"
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
    --sample_steps "$SAMPLE_STEPS"
    --t_scale "$T_SCALE"
)

# Add resume flag if checkpoint provided
if [ -n "$RESUME_CKPT" ]; then
    CMD+=(--resume "$RESUME_CKPT")
    echo "Resuming from checkpoint: $RESUME_CKPT"
fi

echo "Starting ControlNet training (stage 2)..."
printf '%q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
