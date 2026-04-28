#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../..")"
cd "$ROOT_DIR"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
accelerate launch --mixed_precision=fp16 \
  -m src.cli.train_sd_layout --config configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8.yaml "$@"
