#!/usr/bin/env bash
# Shared runtime helpers for Slurm jobs.

set -euo pipefail

slurm_activate_conda() {
  if [[ -n "${CONDA_ACTIVATE_PATH:-}" ]]; then
    # Tamia jobs use an environment-specific activate script.
    source "${CONDA_ACTIVATE_PATH}"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    return
  fi

  echo "ERROR: conda not found in PATH" >&2
  return 1
}

slurm_init_runtime() {
  PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
  ENV_NAME="${ENV_NAME:-diffusers-dev}"
  JOB_ID="${SLURM_JOB_ID:-manual}"
  export PROJECT_ROOT ENV_NAME JOB_ID

  cd "${PROJECT_ROOT}"
  mkdir -p "${PROJECT_ROOT}/logs"
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

  slurm_activate_conda

  echo "Host: $(hostname)"
  echo "Start time: $(date)"
  echo "Project root: ${PROJECT_ROOT}"
  echo "Conda env: ${ENV_NAME}"
  echo "CUDA_VISIBLE_DEVICES from Slurm: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
  echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
  echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
  echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-<unset>}"
  echo "SLURM_MEM_PER_CPU: ${SLURM_MEM_PER_CPU:-<unset>}"
  echo "CPU count: $(nproc 2>/dev/null || echo '<unknown>')"
}

slurm_config_path() {
  local config_rel="$1"
  if [[ "${config_rel}" = /* ]]; then
    printf '%s\n' "${config_rel}"
  else
    printf '%s/%s\n' "${PROJECT_ROOT}" "${config_rel}"
  fi
}

slurm_require_file() {
  local path="$1"
  local label="${2:-file}"
  if [[ -f "${path}" ]]; then
    return
  fi
  if [[ "${path}" != /* && -f "${PROJECT_ROOT}/${path}" ]]; then
    return
  fi
  echo "ERROR: ${label} not found: ${path}" >&2
  return 1
}

slurm_require_path() {
  local path="$1"
  local label="${2:-path}"
  if [[ -e "${path}" ]]; then
    return
  fi
  if [[ "${path}" != /* && -e "${PROJECT_ROOT}/${path}" ]]; then
    return
  fi
  echo "ERROR: ${label} not found: ${path}" >&2
  return 1
}

slurm_print_python_diagnostics() {
  which python
  python --version
}

slurm_print_gpu_diagnostics() {
  nvidia-smi || true
}

slurm_grep_config_keys() {
  local title="$1"
  local pattern="$2"
  local config="${3:-${CONFIG:-}}"
  shift 3 || true
  echo "${title}"
  grep "$@" "${pattern}" "${config}" || true
}

slurm_run_timed() {
  local log_prefix="$1"
  local label="$2"
  shift 2
  if [[ "${1:-}" != "--" ]]; then
    echo "ERROR: slurm_run_timed expects: LOG_PREFIX LABEL -- command..." >&2
    return 2
  fi
  shift

  local log_out="${log_prefix}.out"
  local log_err="${log_prefix}.err"
  echo "${label} stdout log: ${log_out}"
  echo "${label} stderr log: ${log_err}"

  set +e
  /usr/bin/time -v "$@" > "${log_out}" 2> "${log_err}"
  local status=$?
  set -e

  echo "${label} exit code: ${status}"
  echo "Stdout log: ${log_out}"
  echo "Stderr log: ${log_err}"
  echo "End time: $(date)"
  return "${status}"
}
