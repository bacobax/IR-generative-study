#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../..")"
cd "$ROOT_DIR"
python -m src.cli.train_vae --config configs/vae/train/presets/flir_private_proxy_alignment_v18_vae_x4_512.yaml "$@"
