#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# Residual Count Adapter Training Launcher
# ==========================================================
# Adjust paths and hyperparameters as needed.
# ==========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  ROOT_DIR="$ROOT_DIR_GIT"
else
  ROOT_DIR="$SCRIPT_DIR"
fi

# ==========================================================
# Paths
# ==========================================================

DATA_DIR="$ROOT_DIR/data/raw/v18/train"
OUTPUT_DIR="$ROOT_DIR/artifacts/checkpoints/count_adapter/runs/run_07"

# ==========================================================
# Model
# ==========================================================

MODEL_NAME="facebook/dinov2-base"
FEATURE_MODE="cls"

RANK=16
HIDDEN=16

# ==========================================================
# Training
# ==========================================================

EPOCHS=300
LR=1e-3
WEIGHT_DECAY=1e-3
GRAD_CLIP=1.0

# ==========================================================
# Loss weights
# ==========================================================

LAMBDA_PROTO=1.0
LAMBDA_COS=0.5
LAMBDA_SMOOTH=0.2

# ==========================================================
# Dataset filtering
# ==========================================================

MIN_COUNT_SAMPLES=10
BATCH_COUNTS=0

# ==========================================================
# Base embedding behavior
# ==========================================================

LEARN_BASE=false
BASE_WARMUP_EPOCHS=100

# ==========================================================
# Misc
# ==========================================================

SAVE_EVERY=100
DEVICE=auto
SEED=42

# ==========================================================
# Caching & Early stopping
# ==========================================================

CACHE=true
CACHE_DIR="$ROOT_DIR/data/cache/dino_cache"
PATIENCE=50

# ==========================================================
# Launch training
# ==========================================================

python "$ROOT_DIR/scripts/train_count_adapter.py" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --feature-mode "$FEATURE_MODE" \
  --rank "$RANK" \
  --hidden "$HIDDEN" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --lambda-proto "$LAMBDA_PROTO" \
  --lambda-cos "$LAMBDA_COS" \
  --lambda-smooth "$LAMBDA_SMOOTH" \
  --min-count-samples "$MIN_COUNT_SAMPLES" \
  --batch-counts "$BATCH_COUNTS" \
  --save-every "$SAVE_EVERY" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --base-warmup-epochs "$BASE_WARMUP_EPOCHS" \
  --patience "$PATIENCE" \
  --cache \
  --cache-dir "$CACHE_DIR" \
  --loco \
  