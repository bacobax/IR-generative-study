#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
enter_repo_root "${SCRIPT_DIR}"
run_python_module_config src.cli.train configs/fm/train/presets/stay_layout_latent_flir_sd15_512.yaml "$@"
