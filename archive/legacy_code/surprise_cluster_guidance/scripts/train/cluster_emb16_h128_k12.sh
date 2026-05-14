#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../..")"
cd "$ROOT_DIR"
python scripts/standalone/train_cluster_reconstruction.py --config configs/auxiliary/cluster_reconstruction/presets/emb16_h128_k12.yaml "$@"
