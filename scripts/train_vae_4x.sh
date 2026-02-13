#!/usr/bin/env bash
set -euo pipefail

# Example: adjust paths and hyperparameters as needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  ROOT_DIR="$ROOT_DIR_GIT"
else
  ROOT_DIR="$SCRIPT_DIR"
fi

python "$ROOT_DIR/train_vae.py" \
  --train-dir "$ROOT_DIR/v18/train" \
  --val-dir "$ROOT_DIR/v18/val" \
  --image-size 256 \
  --batch-size 8 \
  --num-workers 4 \
  --pin-memory \
  --epochs 200 \
  --t-scale 1000 \
  --model-dir "$ROOT_DIR/vae_runs/vae_fm_x4" \
  --vae-json "$ROOT_DIR/fm_src/vae_config.json" \
  --log-dir "$ROOT_DIR/runs/autoencoder_kl" \
  --patience 4 \
  --min-delta 0 \
  --gpu-prefer memory \
  --min-free-mb 4096
