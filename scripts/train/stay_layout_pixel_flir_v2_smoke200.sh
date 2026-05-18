#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../..")"
cd "$ROOT_DIR"

CONFIG_PATH="configs/fm/train/presets/stay_layout_pixel_flir_v2_smoke200.yaml"
MODEL_DIR="./artifacts/checkpoints/flow_matching/serious_runs/stay_layout_pixel_flir_v2_smoke200"
LOG_DIR="./artifacts/runs/test/flow_matching/stay_layout_pixel_flir_v2_smoke200"
DEBUG_DIR="./artifacts/debug/stay_layout_pixel_flir_v2_smoke200"

mkdir -p "$MODEL_DIR" "$LOG_DIR" "$DEBUG_DIR"

echo "Config:      $CONFIG_PATH"
echo "Model dir:   $MODEL_DIR"
echo "Log dir:     $LOG_DIR"
echo "Debug dir:   $DEBUG_DIR"

echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir $LOG_DIR"


python -m src.cli.train_flow_matching --config "$CONFIG_PATH" "$@"

