#!/bin/bash

# Stable Flow Matching Training Script
# Usage: ./run_stable_training.sh
# To resume: RESUME_CKPT=./path/to/checkpoint.pt ./run_stable_training.sh

# Data paths
TRAIN_DIR="./v18/train/"
VAL_DIR="./v18/val/"

# Model configs
UNET_CONFIG="fm_src/stable_unet_config.json"
VAE_CONFIG="fm_src/vae_config.json"
VAE_WEIGHTS="fm_src/vae_best.pt"

# Output directory
MODEL_DIR="./fake_runs/stable_training_t_scaled/"

# Training params
EPOCHS=300
BATCH_SIZE=8
NUM_WORKERS=4
SAVE_EVERY_N_EPOCHS=30
SAMPLE_BATCH_SIZE=4
T_SCALE=1000

# Augmentation schedule (warmup -> ramp -> decay)
# WARMUP_FRAC=0.1
# RAMP_FRAC=0.3
# P_CROP_WARMUP=0.05
# P_CROP_MAX=0.20
# P_CROP_FINAL=0.05
# P_ROT_WARMUP=0.05
# P_ROT_MAX=0.30
# P_ROT_FINAL=0.05

WARMUP_FRAC=0.1
RAMP_FRAC=0.3
P_CROP_WARMUP=0.01
P_CROP_MAX=0.01
P_CROP_FINAL=0.01
P_ROT_WARMUP=0.05
P_ROT_MAX=0.05
P_ROT_FINAL=0.05

# Resume checkpoint (leave empty for fresh training)
# RESUME_CKPT="serious_runs/stable_training_t_scaled/UNET/unet_fm_epoch_150_ckpt.pt"

# Build the command
CMD="python train_sfm.py \
    --train_dir $TRAIN_DIR \
    --val_dir $VAL_DIR \
    --unet_config $UNET_CONFIG \
    --vae_config $VAE_CONFIG \
    --vae_weights $VAE_WEIGHTS \
    --model_dir $MODEL_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --save_every_n_epochs $SAVE_EVERY_N_EPOCHS \
    --sample_batch_size $SAMPLE_BATCH_SIZE \
    --t_scale $T_SCALE \
    --warmup_frac $WARMUP_FRAC \
    --ramp_frac $RAMP_FRAC \
    --p_crop_warmup $P_CROP_WARMUP \
    --p_crop_max $P_CROP_MAX \
    --p_crop_final $P_CROP_FINAL \
    --p_rot_warmup $P_ROT_WARMUP \
    --p_rot_max $P_ROT_MAX \
    --p_rot_final $P_ROT_FINAL"

# Add resume flag if checkpoint provided
if [ -n "$RESUME_CKPT" ]; then
    CMD="$CMD --resume $RESUME_CKPT"
    echo "Resuming from checkpoint: $RESUME_CKPT"
fi

echo "Starting training..."
echo "$CMD"
exec $CMD
