#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
enter_repo_root "${SCRIPT_DIR}"
run_python_module_config src.cli.train_vae configs/vae/train/presets/flir_private_proxy_alignment_v18_vae_x4_512.yaml "$@"
