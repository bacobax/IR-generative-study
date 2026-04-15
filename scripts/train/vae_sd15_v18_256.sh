#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../..")"
cd "$ROOT_DIR"
conda run -n diffusers-dev python -m src.cli.train_vae --config configs/vae/train/presets/v18_sd15_vae_x8_256.yaml "$@"
