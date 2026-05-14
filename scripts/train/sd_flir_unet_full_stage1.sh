#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
enter_repo_root "${SCRIPT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
run_accelerate_module_config fp16 src.cli.train_sd configs/sd/train/presets/flir_unet_full_stage1.yaml "$@"
