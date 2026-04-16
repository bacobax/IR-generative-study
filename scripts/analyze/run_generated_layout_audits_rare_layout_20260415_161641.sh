#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEVICE="${1:-cuda:0}"

cd "${ROOT_DIR}"

python scripts/standalone/filter_generated_layout_dataset.py \
  --config configs/auxiliary/generated_layout_audit/presets/binary_rare_layout_20260415_161641.yaml \
  --device "${DEVICE}"

python scripts/standalone/filter_generated_layout_dataset.py \
  --config configs/auxiliary/generated_layout_audit/presets/multiclass_rare_layout_20260415_161641.yaml \
  --device "${DEVICE}"
