#!/usr/bin/env bash
set -euo pipefail

# Submit the three BigEarthNet Sentinel-2 B08 SDXL Stage-1 LoRA r8 jobs independently:
#   - full train split
#   - 2040-sample tile-balanced subset
#   - 5100-sample tile-balanced subset

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

if [[ "${PROJECT_ROOT}" == /home/* ]]; then
  echo "ERROR: Killarney does not allow sbatch submissions from /home." >&2
  echo "       Run this from the repo path under /project or /scratch, or set PROJECT_ROOT to that path." >&2
  echo "       Current resolved PROJECT_ROOT: ${PROJECT_ROOT}" >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"

FULL_WORKER="${PROJECT_ROOT}/slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_full_kl.slurm"
TWO_K_WORKER="${PROJECT_ROOT}/slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_2040_kl.slurm"
FIVE_K_WORKER="${PROJECT_ROOT}/slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_sdxl_lora_stage1_r8_train_5100_kl.slurm"

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
  echo sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${FULL_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${TWO_K_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${FIVE_K_WORKER}"
  exit 0
fi

echo "Submitting BigEarthNet SDXL LoRA r8 jobs"
sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${FULL_WORKER}"
sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${TWO_K_WORKER}"
sbatch --chdir="${PROJECT_ROOT}" --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}" "${FIVE_K_WORKER}"

