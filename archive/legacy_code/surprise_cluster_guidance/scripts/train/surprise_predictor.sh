#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../..")"
cd "$ROOT_DIR"
python scripts/standalone/train_surprise_predictor.py --config configs/auxiliary/surprise_predictor/presets/default.yaml "$@"
