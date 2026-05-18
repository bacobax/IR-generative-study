#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../..")"
cd "$ROOT_DIR"

CONFIG_PATH="configs/fm/train/presets/stay_layout_pixel_flir_v2.yaml"
MODEL_DIR="./artifacts/checkpoints/flow_matching/serious_runs/stay_layout_pixel_flir_v2"
LOG_DIR="./artifacts/runs/main/flow_matching/stay_layout_pixel_flir_v2"
DEBUG_DIR="./artifacts/debug/stay_layout_pixel_flir_v2"

mkdir -p "$MODEL_DIR" "$LOG_DIR" "$DEBUG_DIR"

echo "Config:      $CONFIG_PATH"
echo "Model dir:   $MODEL_DIR"
echo "Log dir:     $LOG_DIR"
echo "Debug dir:   $DEBUG_DIR"

echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir $LOG_DIR"


python -m src.cli.train_flow_matching --config "$CONFIG_PATH" --resume /projets/Fbassignana/diffusers_try/flow_matching_trial/artifacts/checkpoints/flow_matching/serious_runs/stay_layout_pixel_flir_v2/UNET/unet_fm_epoch_90_ckpt.pt "$@"

