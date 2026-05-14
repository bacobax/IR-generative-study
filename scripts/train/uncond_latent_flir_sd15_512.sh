#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
enter_repo_root "${SCRIPT_DIR}"
run_python_module_config src.cli.train_sd_uncond configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512.yaml "$@"
