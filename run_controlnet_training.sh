#!/bin/bash

# =====================================================================
# ControlNet Flow Matching Training  (Stage 2)
#
# Requires a trained stage-1 pipeline folder that contains:
#   UNET/config.json + unet_fm_best.pt   (or unet_fm_epoch_*.pt)
#   VAE/config.json  + vae_best.pt        (or vae_epoch_*.pt)
# =====================================================================

# ---------- Data paths ----------
TRAIN_DIR="./v18/train/"
VAL_DIR="./v18/val/"
TRAIN_ANNOTATIONS="./v18/train/annotations.json"
VAL_ANNOTATIONS="./v18/val/annotations.json"

# ---------- Stage-1 pipeline (frozen UNet + VAE) ----------
STAGE1_PIPELINE_DIR="./serious_runs/stable_training_t_scaled/"

# ---------- Output ----------
MODEL_DIR="./controlnet_runs/bbox_controlnet/"

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
# RESUME_CKPT="./controlnet_runs/bbox_controlnet/CONTROLNET/controlnet_epoch_10_ckpt.pt"

# ---------- Build command ----------
CMD="conda run --no-capture-output -n diffusers-dev python train_controlnet.py \
    --train_dir $TRAIN_DIR \
    --val_dir $VAL_DIR \
    --train_annotations $TRAIN_ANNOTATIONS \
    --val_annotations $VAL_ANNOTATIONS \
    --stage1_pipeline_dir $STAGE1_PIPELINE_DIR \
    --model_dir $MODEL_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --conditioning_scale $CONDITIONING_SCALE \
    --conditioning_dropout $CONDITIONING_DROPOUT \
    --save_every_n_epochs $SAVE_EVERY_N_EPOCHS \
    --sample_steps $SAMPLE_STEPS \
    --t_scale $T_SCALE"

# Add resume flag if checkpoint provided
if [ -n "$RESUME_CKPT" ]; then
    CMD="$CMD --resume $RESUME_CKPT"
    echo "Resuming from checkpoint: $RESUME_CKPT"
fi

echo "Starting ControlNet training (stage 2)..."
echo "$CMD"
exec $CMD
