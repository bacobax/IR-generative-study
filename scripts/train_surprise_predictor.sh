#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

# -------- Dataset (must match build_surprise_pred_dataset.sh output) --------
DS_ROOT="$ROOT_DIR/data/derived/surprise_pred_dataset"
VAE_MODEL_NAME="vae_x4_best"
VAE_CONFIG="$ROOT_DIR/configs/models/fm/vae_config.json"
VAE_WEIGHTS="$ROOT_DIR/artifacts/checkpoints/vae/vae_runs/vae_fm_x4/VAE/vae_best.pt"

# -------- Model --------
DINO_NAME="dinov2_vits14"
HIDDEN_DIM="256"

# -------- Targets --------
TARGET_MODE="minmax"       # minmax | raw

# -------- Splits --------
SEED="0"
VAL_FRAC="0.15"
TEST_FRAC="0.10"

# -------- Training --------
EPOCHS="500"
BATCH_SIZE="16"
NUM_WORKERS="4"
LR="1e-4"
WEIGHT_DECAY="1e-4"
DEVICE="cuda:2"            # auto | cpu | cuda[:index]
AMP="0"                    # 1=true, 0=false
LOSS_W_SURPRISE="1.0"
LOSS_W_GMM="1.0"

# -------- Early stopping --------
EARLY_METRIC="val_loss"    # val_loss | val_spearman_surprise | val_spearman_gmm
EARLY_MODE=""               # auto from metric if empty
PATIENCE="400"
MIN_DELTA="1e-4"

# -------- Checkpointing --------
OUT_DIR="$ROOT_DIR/artifacts/runs/main/surprise_predictor_longer_run"
RUN_NAME=""                 # empty = auto-generated
SAVE_EVERY="0"              # 0 = only best + final
RESUME=""                   # path to checkpoint to resume from

# -------- Build command --------
CMD=(
  python "$ROOT_DIR/scripts/train_surprise_predictor.py"
  --ds_root "$DS_ROOT"
  --vae_model_name "$VAE_MODEL_NAME"
  --vae_config "$VAE_CONFIG"
  --vae_weights "$VAE_WEIGHTS"
  --dino_name "$DINO_NAME"
  --hidden_dim "$HIDDEN_DIM"
  --target_mode "$TARGET_MODE"
  --seed "$SEED"
  --val_frac "$VAL_FRAC"
  --test_frac "$TEST_FRAC"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --device "$DEVICE"
  --loss_weights "$LOSS_W_SURPRISE" "$LOSS_W_GMM"
  --early_metric "$EARLY_METRIC"
  --patience "$PATIENCE"
  --min_delta "$MIN_DELTA"
  --out_dir "$OUT_DIR"
  --save_every "$SAVE_EVERY"
)

if [[ -n "$EARLY_MODE" ]]; then
  CMD+=(--early_mode "$EARLY_MODE")
fi

if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run_name "$RUN_NAME")
fi

if [[ -n "$RESUME" ]]; then
  CMD+=(--resume "$RESUME")
fi

if [[ "$AMP" == "1" ]]; then
  CMD+=(--amp)
fi

echo "Launching surprise predictor training..."
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
