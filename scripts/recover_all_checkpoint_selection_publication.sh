#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-dry-run}"
DELETE_INVALID_ANALYSIS="${DELETE_INVALID_ANALYSIS:-0}"
ALLOW_HEAVY_METRICS="${ALLOW_HEAVY_METRICS:-0}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/checkpoint_selection_recovery}"

BEN_ROOT="${BEN_ROOT:-/scratch/bacobax2/bigearthnet_s2_b08_5x5_stride3_checkpoint_selection_publication_single_runs}"
FLIR_ROOT="${FLIR_ROOT:-/scratch/bacobax2/flir_checkpoint_selection_publication_single_runs}"

BEN_CONFIG_DIR="${BEN_CONFIG_DIR:-${PROJECT_ROOT}/configs/eval/publication_single_runs/bigearthnet_s2_b08_5x5_stride3}"
FLIR_CONFIG_DIR="${FLIR_CONFIG_DIR:-${PROJECT_ROOT}/configs/eval/publication_single_runs/flir}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

case "${MODE}" in
  dry-run|execute)
    ;;
  *)
    echo "ERROR: MODE must be dry-run or execute, got '${MODE}'." >&2
    exit 2
    ;;
esac

run_root() {
  local root="$1"
  local config_dir="$2"
  local label="$3"

  if [[ ! -d "${root}" ]]; then
    echo "Skipping ${label}: root does not exist: ${root}" >&2
    return 0
  fi
  if [[ ! -d "${config_dir}" ]]; then
    echo "ERROR: config directory does not exist for ${label}: ${config_dir}" >&2
    exit 1
  fi

  echo "Scanning ${label}"
  echo "  root: ${root}"
  echo "  configs: ${config_dir}"

  while IFS=$'\t' read -r run_id config_path; do
    if [[ -z "${run_id}" || -z "${config_path}" ]]; then
      continue
    fi

    local run_dir="${root}/${run_id}"
    if [[ ! -d "${run_dir}" ]]; then
      echo "  config has no current run directory, skipping: ${run_id}"
      continue
    fi

    local log_file="${LOG_DIR}/${label}_${run_id}_${MODE}.log"
    local args=(
      scripts/recover_checkpoint_selection_publication.py
      --root "${root}"
      --config "${config_path}"
      --only-run "${run_id}"
      --log-file "${log_file}"
    )

    if [[ "${MODE}" == "execute" ]]; then
      args+=(--execute)
    else
      args+=(--dry-run)
    fi
    if [[ "${DELETE_INVALID_ANALYSIS}" == "1" || "${DELETE_INVALID_ANALYSIS}" == "true" ]]; then
      args+=(--delete-invalid-analysis)
    fi
    if [[ "${ALLOW_HEAVY_METRICS}" == "1" || "${ALLOW_HEAVY_METRICS}" == "true" ]]; then
      args+=(--allow-heavy-metrics)
    fi

    echo "  recovering ${run_id}"
    echo "    config: ${config_path}"
    echo "    log: ${log_file}"
    "${PYTHON_BIN}" "${args[@]}"
  done < <(
    "${PYTHON_BIN}" - "${config_dir}" <<'PY'
import sys
from pathlib import Path
import yaml

config_dir = Path(sys.argv[1])
rows = []
for path in sorted(config_dir.glob("*.yaml")):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if data.get("pipeline_mode") != "clean_fid_selection_publication":
        continue
    runs = data.get("runs") or []
    if len(runs) != 1:
        continue
    run_id = str(runs[0].get("run_identifier") or "").strip()
    if not run_id:
        continue
    rows.append((run_id, str(path)))

for run_id, path in rows:
    print(f"{run_id}\t{path}")
PY
  )
}

run_root "${BEN_ROOT}" "${BEN_CONFIG_DIR}" "bigearthnet"
run_root "${FLIR_ROOT}" "${FLIR_CONFIG_DIR}" "flir"
