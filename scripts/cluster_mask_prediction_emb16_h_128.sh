#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

# -------- Fixed configuration --------
DATA_ROOT="$ROOT_DIR/v18"
SPLIT="train"
DINO_NAME="dinov2_vits14"
K_REGIONS="5"

SEED="0"
VAL_FRAC="0.20"
TEST_FRAC="0.10"

USE_PROJ="1"      # 1=true, 0=false
EMB_DIM="16"
HIDDEN_DIM="128"

LR="2e-4"
WEIGHT_DECAY="1e-4"
DEVICE="cuda:0"       # auto | cpu | cuda
BATCH_SIZE="128"
EPOCHS="150"
NUM_WORKERS="4"
MASK_PROB="0.50"

EARLY_METRIC="val_masked_acc"  # val_loss | val_masked_acc | val_full_acc
EARLY_MODE="max"               # min | max
PATIENCE="7"
MIN_DELTA="1e-4"
SAVE_BEST="1"     # 1=true, 0=false

RUNS_ROOT="$ROOT_DIR/runs/cluster_reconstruction"
RUN_NAME="emb${EMB_DIM}_hidden${HIDDEN_DIM}_proj_mask50_k${K_REGIONS}"
OUT_DIR=""        # if set, overrides RUNS_ROOT/RUN_NAME

# Typicality (optional, expensive)
EVAL_TYPICALITY="0"      # 1=true, 0=false
TYPICALITY_CHUNK="32"
TYPICALITY_MAX_BATCHES="0"

# -------- Build command --------
CMD=(
  python "$ROOT_DIR/train_cluster_reconstruction.py"
  --data_root "$DATA_ROOT"
  --split "$SPLIT"
  --dino_name "$DINO_NAME"
  --k_regions "$K_REGIONS"
  --seed "$SEED"
  --val_frac "$VAL_FRAC"
  --test_frac "$TEST_FRAC"
  --emb_dim "$EMB_DIM"
  --hidden_dim "$HIDDEN_DIM"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --num_workers "$NUM_WORKERS"
  --mask_prob "$MASK_PROB"
  --early_metric "$EARLY_METRIC"
  --early_mode "$EARLY_MODE"
  --patience "$PATIENCE"
  --min_delta "$MIN_DELTA"
  --runs_root "$RUNS_ROOT"
  --run_name "$RUN_NAME"
  --typicality_chunk "$TYPICALITY_CHUNK"
  --typicality_max_batches "$TYPICALITY_MAX_BATCHES"
)

if [[ -n "$OUT_DIR" ]]; then
  CMD+=(--out_dir "$OUT_DIR")
fi

if [[ "$USE_PROJ" == "1" ]]; then
  CMD+=(--use_proj)
else
  CMD+=(--no_use_proj)
fi

if [[ "$SAVE_BEST" == "1" ]]; then
  CMD+=(--save_best)
else
  CMD+=(--no_save_best)
fi

if [[ "$EVAL_TYPICALITY" == "1" ]]; then
  CMD+=(--eval_typicality)
fi

echo "Launching cluster reconstruction training with TensorBoard logging..."
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
