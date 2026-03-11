#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../..")"
cd "$ROOT_DIR"

# Launch guided FM variants in parallel
pids=()
for preset in guided_combo_maxmin guided_combo_maxmin_tune_{1a,1b,2a,2b} guided_combo_minmax; do
  bash scripts/generate/fm_${preset}.sh &
  pids+=($!)
done

# Wait for all generation jobs
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All guided FM generation complete. Running analysis..."
bash scripts/analyze/distribution_shift.sh
