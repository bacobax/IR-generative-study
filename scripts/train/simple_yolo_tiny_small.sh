#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
enter_repo_root "${SCRIPT_DIR}"

PYTHON=/export/livia/home/vision/Fbassignana/miniconda3/envs/diffusers-dev/bin/python
CONFIG=configs/yolo/exp_v18_simple_yolo_tiny/small.yaml
LOG_FILE=artifacts/runs/yolo/exp_v18_simple_yolo_tiny/small_train.log

mkdir -p "$(dirname "${LOG_FILE}")"

# Redirect all output through tee inside the script — avoids pipe fragility
# from external shell command chaining in tmux.
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[launcher] $(date '+%Y-%m-%d %H:%M:%S') starting simple_yolo_tiny small"
echo "[launcher] config: ${CONFIG}"
echo "[launcher] log:    ${LOG_FILE}"
echo "[launcher] device: ${CUDA_VISIBLE_DEVICES}"
echo "---"

exec "${PYTHON}" -u -m src.cli.train_yolo \
    --action train \
    --config "${CONFIG}" \
    --device "${CUDA_VISIBLE_DEVICES}" \
    "$@"
