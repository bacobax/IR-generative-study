#!/usr/bin/env bash
set -euo pipefail

# Submit the three FLIR SDXL Stage-1 LoRA r8 jobs independently:
#   - full train split
#   - 2k stratified subset
#   - 5k stratified subset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
  else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
  fi
else
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd -P)"
fi

if [[ "${PROJECT_ROOT}" == "${HOME}"/* ]]; then
  echo "WARNING: FIR HOME has a small quota; $HOME/project/<project-id> or $HOME/scratch is recommended for experiment repos." >&2
  echo "         Current resolved PROJECT_ROOT: ${PROJECT_ROOT}" >&2
fi

DRY_RUN="${DRY_RUN:-0}"

FULL_WORKER="${PROJECT_ROOT}/slurm/fir/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_full_fir.slurm"
TWO_K_WORKER="${PROJECT_ROOT}/slurm/fir/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_train_2000_fir.slurm"
FIVE_K_WORKER="${PROJECT_ROOT}/slurm/fir/flir/sd_adaptation/train_flir_sdxl_lora_stage1_r8_train_5000_fir.slurm"

cd "${PROJECT_ROOT}"
echo "Project root: ${PROJECT_ROOT}"

for path in "${FULL_WORKER}" "${TWO_K_WORKER}" "${FIVE_K_WORKER}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing Slurm worker: ${path}" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN}" = "1" || "${DRY_RUN}" = "true" ]]; then
  echo "DRY RUN: commands that would be submitted"
  # echo sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${FULL_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${TWO_K_WORKER}"
  # echo sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${FIVE_K_WORKER}"
  exit 0
fi

echo "Submitting FLIR SDXL LoRA r8 jobs"
# sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${FULL_WORKER}"
sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${TWO_K_WORKER}"
# sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${FIVE_K_WORKER}"
