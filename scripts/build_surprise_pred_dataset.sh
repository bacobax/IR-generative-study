#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

# -------- Required arguments --------
CREATED_DATASET="$ROOT_DIR/surprise_pred_dataset"
VAE_MODEL_NAME="vae_x4_best"
VAE_CONFIG="$ROOT_DIR/fm_src/vae_config.json"
VAE_WEIGHTS="$ROOT_DIR/vae_runs/vae_fm_x4/VAE/vae_best.pt"
MASKED_MODEL_CKPT="$ROOT_DIR/runs/cluster_reconstruction/emb32_hidden256_proj_mask50_k5/best_model.pt"
DINO_NAME="dinov2_vits14"
N_CLUSTERS="5"

# -------- Data / split --------
DATA_ROOT="$ROOT_DIR/v18"
SPLIT="train"
MAX_ITEMS="0"            # 0 = all items

# -------- Compute --------
BATCH_SIZE="32"
NUM_WORKERS="4"
DEVICE="cuda:0"          # auto | cpu | cuda[:index]
SEED="0"

# -------- Typicality / surprise --------
EVAL_TYPICALITY="1"      # 1=true, 0=false
TYPICALITY_CHUNK="32"
TYPICALITY_MAX_BATCHES="0"  # 0 = no limit

# -------- GMM --------
GMM_N_COMPONENTS="10"
GMM_COVARIANCE_TYPE="full"   # full | tied | diag | spherical

# -------- Overwrite behavior --------
OVERWRITE_CLUSTERS="1"   # 1=true, 0=false
OVERWRITE_LATENTS="1"    # 1=true, 0=false

# -------- Build command --------
CMD=(
  python "$ROOT_DIR/build_surprise_pred_dataset.py"
  --created_dataset "$CREATED_DATASET"
  --vae_model_name "$VAE_MODEL_NAME"
  --vae_config "$VAE_CONFIG"
  --vae_weights "$VAE_WEIGHTS"
  --masked_model_ckpt "$MASKED_MODEL_CKPT"
  --dino_name "$DINO_NAME"
  --n_clusters "$N_CLUSTERS"
  --data_root "$DATA_ROOT"
  --split "$SPLIT"
  --max_items "$MAX_ITEMS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
  --typicality_chunk "$TYPICALITY_CHUNK"
  --typicality_max_batches "$TYPICALITY_MAX_BATCHES"
  --gmm_n_components "$GMM_N_COMPONENTS"
  --gmm_covariance_type "$GMM_COVARIANCE_TYPE"
)

if [[ "$EVAL_TYPICALITY" == "1" ]]; then
  CMD+=(--eval_typicality)
fi

if [[ "$OVERWRITE_CLUSTERS" == "1" ]]; then
  CMD+=(--overwrite_clusters)
fi

if [[ "$OVERWRITE_LATENTS" == "1" ]]; then
  CMD+=(--overwrite_latents)
fi

echo "Launching surprise-pred dataset build..."
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
