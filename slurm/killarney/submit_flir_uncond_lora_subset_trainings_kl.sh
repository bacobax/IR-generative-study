#!/usr/bin/env bash
set -euo pipefail

# Submit the 3 x 3 FLIR training sweep on Killarney:
#   - unconditional latent FM OT
#   - unconditional latent diffusion
#   - Stage-1 SD IR LoRA r8 adaptation
# across full train, 5k subset, and 2k subset configs.
#
# Walltimes are scaled from existing Killarney baselines using the local
# v18 train split size (10,742 samples), rounded up to practical hour blocks:
#   FM/SD full: 24h -> 5k: 12h, 2k: 5h
#   LoRA full: 12h -> 5k: 6h, 2k: 3h

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}}"
DRY_RUN="${DRY_RUN:-0}"

FM_WORKER="${SCRIPT_DIR}/train_stable_fm_hflip_ot_kl.slurm"
SD_WORKER="${SCRIPT_DIR}/train_stable_sd_hflip_kl.slurm"
SD_STAGE1_WORKER="${SCRIPT_DIR}/train_flir_unet_full_domainstudio_512_kl.slurm"

submit_job() {
  local label="$1"
  local worker="$2"
  local config_rel="$3"
  local walltime="$4"
  local job_name="$5"

  local config_path="${PROJECT_ROOT}/${config_rel}"
  if [[ ! -f "${config_path}" ]]; then
    echo "ERROR: missing config for ${label}: ${config_path}" >&2
    return 1
  fi
  if [[ ! -f "${worker}" ]]; then
    echo "ERROR: missing worker for ${label}: ${worker}" >&2
    return 1
  fi

  local cmd=(
    sbatch
    --job-name="${job_name}"
    --time="${walltime}"
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CONFIG_REL=${config_rel}"
    "${worker}"
  )

  echo "${label}: ${config_rel} (${walltime})"
  if [[ "${DRY_RUN}" = "1" || "${DRY_RUN}" = "true" ]]; then
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

submit_job \
  "FM OT full" \
  "${FM_WORKER}" \
  "configs/fm/train/presets/uncond_latent_flir_sd15_512_b64_hflip_ot.yaml" \
  "24:00:00" \
  "fm-ot-full"

submit_job \
  "FM OT 5k" \
  "${FM_WORKER}" \
  "configs/fm/train/presets/uncond_latent_flir_sd15_512_b64_hflip_ot_train_5000.yaml" \
  "12:00:00" \
  "fm-ot-5k"

submit_job \
  "FM OT 2k" \
  "${FM_WORKER}" \
  "configs/fm/train/presets/uncond_latent_flir_sd15_512_b64_hflip_ot_train_2000.yaml" \
  "05:00:00" \
  "fm-ot-2k"

submit_job \
  "SD uncond full" \
  "${SD_WORKER}" \
  "configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512_b64_hflip.yaml" \
  "24:00:00" \
  "sd-uncond-full"

submit_job \
  "SD uncond 5k" \
  "${SD_WORKER}" \
  "configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512_b64_hflip_train_5000.yaml" \
  "12:00:00" \
  "sd-uncond-5k"

submit_job \
  "SD uncond 2k" \
  "${SD_WORKER}" \
  "configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512_b64_hflip_train_2000.yaml" \
  "05:00:00" \
  "sd-uncond-2k"

submit_job \
  "SD LoRA r8 full" \
  "${SD_STAGE1_WORKER}" \
  "configs/sd/train/presets/flir_lora_stage1_r8.yaml" \
  "12:00:00" \
  "sd-lora-r8-full"

submit_job \
  "SD LoRA r8 5k" \
  "${SD_STAGE1_WORKER}" \
  "configs/sd/train/presets/flir_lora_stage1_r8_train_5000.yaml" \
  "06:00:00" \
  "sd-lora-r8-5k"

submit_job \
  "SD LoRA r8 2k" \
  "${SD_STAGE1_WORKER}" \
  "configs/sd/train/presets/flir_lora_stage1_r8_train_2000.yaml" \
  "03:00:00" \
  "sd-lora-r8-2k"
