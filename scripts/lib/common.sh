#!/usr/bin/env bash
# Shared helpers for simple config-driven launchers.

set -euo pipefail

repo_root_from_script_dir() {
  local script_dir="$1"
  git -C "${script_dir}" rev-parse --show-toplevel 2>/dev/null || {
    cd "${script_dir}/../.."
    pwd
  }
}

enter_repo_root() {
  local script_dir="$1"
  ROOT_DIR="$(repo_root_from_script_dir "${script_dir}")"
  export ROOT_DIR
  cd "${ROOT_DIR}"
}

require_config() {
  local config_rel="$1"
  if [[ ! -f "${config_rel}" ]]; then
    echo "Missing config: ${config_rel}" >&2
    echo "Repo root: ${ROOT_DIR:-$(pwd)}" >&2
    return 1
  fi
}

run_python_module_config() {
  local module="$1"
  local config_rel="$2"
  shift 2
  require_config "${config_rel}"
  python -m "${module}" --config "${config_rel}" "$@"
}

run_python_script_config() {
  local script_rel="$1"
  local config_rel="$2"
  shift 2
  require_config "${config_rel}"
  python "${script_rel}" --config "${config_rel}" "$@"
}

run_accelerate_module_config() {
  local mixed_precision="$1"
  local module="$2"
  local config_rel="$3"
  shift 3
  require_config "${config_rel}"
  accelerate launch --mixed_precision="${mixed_precision}" \
    -m "${module}" --config "${config_rel}" "$@"
}
