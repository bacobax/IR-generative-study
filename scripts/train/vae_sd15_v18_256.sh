#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
enter_repo_root "${SCRIPT_DIR}"
CONFIG_REL="configs/vae/train/presets/v18_sd15_vae_x8_256.yaml"
require_config "${CONFIG_REL}"
conda run -n diffusers-dev python -m src.cli.train_vae --config "${CONFIG_REL}" "$@"
