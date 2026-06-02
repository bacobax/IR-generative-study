#!/usr/bin/env bash
set -euo pipefail

# Submit the 3 x 3 FLIR training sweep on Fir:
#   - unconditional latent FM OT
#   - unconditional latent diffusion
#   - Stage-1 SD IR LoRA r8 adaptation
# across full train, 5k subset, and 2k subset configs.
#
# Walltimes are scaled from the matching baseline launchers using the local
# v18 train split size (10,742 samples), rounded up to practical hour blocks:
#   FM/SD full: 24h -> 5k: 12h, 2k: 5h
#   LoRA full: 12h -> 5k: 6h, 2k: 3h

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

FM_WORKER="${PROJECT_ROOT}/slurm/fir/flir/flow_matching/train_stable_fm_hflip_ot_fir.slurm"
SD_WORKER="${PROJECT_ROOT}/slurm/fir/flir/diffusion/train_stable_sd_hflip_fir.slurm"
SD_STAGE1_WORKER="${PROJECT_ROOT}/slurm/fir/flir/sd_adaptation/train_flir_unet_full_domainstudio_512_fir.slurm"

FM_FULL_CONFIG="configs/datasets/flir/flow_matching/uncond_latent_flir_sd15_512_b64_hflip_ot.yaml"
FM_5K_CONFIG="configs/datasets/flir/flow_matching/uncond_latent_flir_sd15_512_b64_hflip_ot_train_5000.yaml"
FM_2K_CONFIG="configs/datasets/flir/flow_matching/uncond_latent_flir_sd15_512_b64_hflip_ot_train_2000.yaml"
SD_FULL_CONFIG="configs/datasets/flir/diffusion/uncond_latent_flir_sd15_512_b64_hflip.yaml"
SD_5K_CONFIG="configs/datasets/flir/diffusion/uncond_latent_flir_sd15_512_b64_hflip_train_5000.yaml"
SD_2K_CONFIG="configs/datasets/flir/diffusion/uncond_latent_flir_sd15_512_b64_hflip_train_2000.yaml"
LORA_FULL_CONFIG="configs/datasets/flir/sd_adaptation/flir_lora_stage1_r8.yaml"
LORA_5K_CONFIG="configs/datasets/flir/sd_adaptation/flir_lora_stage1_r8_train_5000.yaml"
LORA_2K_CONFIG="configs/datasets/flir/sd_adaptation/flir_lora_stage1_r8_train_2000.yaml"

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"

for path in "${FM_WORKER}" "${SD_WORKER}" "${SD_STAGE1_WORKER}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing Slurm worker: ${path}" >&2
    exit 1
  fi
done

for config in \
  "${FM_FULL_CONFIG}" "${FM_5K_CONFIG}" "${FM_2K_CONFIG}" \
  "${SD_FULL_CONFIG}" "${SD_5K_CONFIG}" "${SD_2K_CONFIG}" \
  "${LORA_FULL_CONFIG}" "${LORA_5K_CONFIG}" "${LORA_2K_CONFIG}"; do
  if [[ ! -f "${config}" ]]; then
    echo "ERROR: missing config: ${PROJECT_ROOT}/${config}" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN}" = "1" || "${DRY_RUN}" = "true" ]]; then
  echo "DRY RUN: commands that would be submitted"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=fm-ot-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_FULL_CONFIG}" "${FM_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=fm-ot-5k --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_5K_CONFIG}" "${FM_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=fm-ot-2k --time=05:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_2K_CONFIG}" "${FM_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-uncond-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_FULL_CONFIG}" "${SD_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-uncond-5k --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_5K_CONFIG}" "${SD_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-uncond-2k --time=05:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_2K_CONFIG}" "${SD_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-lora-r8-full --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_FULL_CONFIG}" "${SD_STAGE1_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-lora-r8-5k --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_5K_CONFIG}" "${SD_STAGE1_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-lora-r8-2k --time=05:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_2K_CONFIG}" "${SD_STAGE1_WORKER}"
  exit 0
fi

echo "Submitting FM OT jobs"
sbatch --chdir="${PROJECT_ROOT}" --job-name=fm-ot-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_FULL_CONFIG}" "${FM_WORKER}"
sbatch --chdir="${PROJECT_ROOT}" --job-name=fm-ot-5k --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_5K_CONFIG}" "${FM_WORKER}"
sbatch --chdir="${PROJECT_ROOT}" --job-name=fm-ot-2k --time=05:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_2K_CONFIG}" "${FM_WORKER}"

echo "Submitting SD uncond jobs"
sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-uncond-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_FULL_CONFIG}" "${SD_WORKER}"
sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-uncond-5k --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_5K_CONFIG}" "${SD_WORKER}"
sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-uncond-2k --time=05:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_2K_CONFIG}" "${SD_WORKER}"

echo "Submitting SD LoRA r8 jobs"
sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-lora-r8-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_FULL_CONFIG}" "${SD_STAGE1_WORKER}"
sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-lora-r8-5k --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_5K_CONFIG}" "${SD_STAGE1_WORKER}"
sbatch --chdir="${PROJECT_ROOT}" --job-name=sd-lora-r8-2k --time=05:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_2K_CONFIG}" "${SD_STAGE1_WORKER}"
