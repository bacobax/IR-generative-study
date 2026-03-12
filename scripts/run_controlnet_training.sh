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
# Uses YAML config for all parameters (3-layer merge: defaults → YAML → CLI).
#
# Requires a trained stage-1 pipeline folder that contains:
#   UNET/config.json + unet_fm_best.pt   (or unet_fm_epoch_*.pt)
#   VAE/config.json  + vae_best.pt        (or vae_epoch_*.pt)
# =====================================================================

# Config file — edit this or use a preset from configs/controlnet/train/presets/
CONFIG="$ROOT_DIR/configs/controlnet/train/default.yaml"

# ---------- Resume (uncomment to resume) ----------
# RESUME_CKPT="$ROOT_DIR/controlnet_runs/bbox_controlnet/CONTROLNET/controlnet_epoch_10_ckpt.pt"

# ---------- Build command ----------
CMD=(
    conda run --no-capture-output -n diffusers-dev
    python "$ROOT_DIR/train_controlnet.py"
    --config "$CONFIG"
)

# Add resume flag if checkpoint provided
if [ -n "${RESUME_CKPT:-}" ]; then
    CMD+=(--resume "$RESUME_CKPT")
    echo "Resuming from checkpoint: $RESUME_CKPT"
fi

# Any additional CLI overrides can be appended here, e.g.:
# CMD+=(--epochs 50 --batch_size 16)

echo "Starting ControlNet training (stage 2)..."
printf '%q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
