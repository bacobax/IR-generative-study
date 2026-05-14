#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
enter_repo_root "${SCRIPT_DIR}"
run_python_script_config scripts/standalone/train_count_adapter.py configs/auxiliary/count_adapter/presets/run_07.yaml "$@"
