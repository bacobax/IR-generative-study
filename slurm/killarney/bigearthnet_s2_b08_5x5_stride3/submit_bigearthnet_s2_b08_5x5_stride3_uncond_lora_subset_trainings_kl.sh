#!/usr/bin/env bash
set -euo pipefail

# Submit the 3 x 3 BigEarthNet Sentinel-2 B08 5x5 stride-3 training sweep on Killarney:
#   - unconditional latent FM OT
#   - unconditional latent diffusion
#   - Stage-1 SD LoRA r8 adaptation
# across full train, 5100-sample subset, and 2040-sample subset configs.
#
# Walltimes mirror the comparable FLIR sweep.

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

FM_WORKER="${PROJECT_ROOT}/slurm/killarney/bigearthnet_s2_b08_5x5_stride3/flow_matching/train_bigearthnet_s2_b08_5x5_stride3_fm_kl.slurm"
SD_WORKER="${PROJECT_ROOT}/slurm/killarney/bigearthnet_s2_b08_5x5_stride3/diffusion/train_bigearthnet_s2_b08_5x5_stride3_sd_uncond_kl.slurm"
SD_STAGE1_WORKER="${PROJECT_ROOT}/slurm/killarney/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/train_bigearthnet_s2_b08_5x5_stride3_lora_kl.slurm"

FM_FULL_CONFIG="configs/datasets/bigearthnet_s2_b08_5x5_stride3/flow_matching/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip_ot.yaml"
FM_5100_CONFIG="configs/datasets/bigearthnet_s2_b08_5x5_stride3/flow_matching/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip_ot_train_5100.yaml"
FM_2040_CONFIG="configs/datasets/bigearthnet_s2_b08_5x5_stride3/flow_matching/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip_ot_train_2040.yaml"
SD_FULL_CONFIG="configs/datasets/bigearthnet_s2_b08_5x5_stride3/diffusion/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip.yaml"
SD_5100_CONFIG="configs/datasets/bigearthnet_s2_b08_5x5_stride3/diffusion/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip_train_5100.yaml"
SD_2040_CONFIG="configs/datasets/bigearthnet_s2_b08_5x5_stride3/diffusion/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip_train_2040.yaml"
LORA_FULL_CONFIG="configs/datasets/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/bigearthnet_s2_b08_5x5_stride3_lora_stage1_r8.yaml"
LORA_5100_CONFIG="configs/datasets/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/bigearthnet_s2_b08_5x5_stride3_lora_stage1_r8_train_5100.yaml"
LORA_2040_CONFIG="configs/datasets/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/bigearthnet_s2_b08_5x5_stride3_lora_stage1_r8_train_2040.yaml"

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"

for path in "${FM_WORKER}" "${SD_WORKER}" "${SD_STAGE1_WORKER}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing Slurm worker: ${path}" >&2
    exit 1
  fi
done

for config in \
  "${FM_FULL_CONFIG}" "${FM_5100_CONFIG}" "${FM_2040_CONFIG}" \
  "${SD_FULL_CONFIG}" "${SD_5100_CONFIG}" "${SD_2040_CONFIG}" \
  "${LORA_FULL_CONFIG}" "${LORA_5100_CONFIG}" "${LORA_2040_CONFIG}"; do
  if [[ ! -f "${config}" ]]; then
    echo "ERROR: missing config: ${PROJECT_ROOT}/${config}" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN}" = "1" || "${DRY_RUN}" = "true" ]]; then
  echo "DRY RUN: commands that would be submitted"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-fm-ot-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_FULL_CONFIG}",SIZE=full "${FM_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-fm-ot-5100 --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_5100_CONFIG}",SIZE=5100 "${FM_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-fm-ot-2040 --time=05:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_2040_CONFIG}",SIZE=2040 "${FM_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-uncond-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_FULL_CONFIG}",SIZE=full "${SD_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-uncond-5100 --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_5100_CONFIG}",SIZE=5100 "${SD_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-uncond-2040 --time=05:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_2040_CONFIG}",SIZE=2040 "${SD_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-lora-r8-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_FULL_CONFIG}",SIZE=full "${SD_STAGE1_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-lora-r8-5100 --time=12:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_5100_CONFIG}",SIZE=5100 "${SD_STAGE1_WORKER}"
  echo sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-lora-r8-2040 --time=06:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_2040_CONFIG}",SIZE=2040 "${SD_STAGE1_WORKER}"
  exit 0
fi

echo "Submitting BigEarthNet FM OT jobs"
# sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-fm-ot-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_FULL_CONFIG}",SIZE=full "${FM_WORKER}"
# sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-fm-ot-5100 --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_5100_CONFIG}",SIZE=5100 "${FM_WORKER}"
sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-fm-ot-2040 --time=05:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${FM_2040_CONFIG}",SIZE=2040 "${FM_WORKER}"

# echo "Submitting BigEarthNet SD uncond jobs"
# sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-uncond-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_FULL_CONFIG}",SIZE=full "${SD_WORKER}"
# sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-uncond-5100 --time=15:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_5100_CONFIG}",SIZE=5100 "${SD_WORKER}"
# sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-uncond-2040 --time=05:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${SD_2040_CONFIG}",SIZE=2040 "${SD_WORKER}"

# echo "Submitting BigEarthNet SD LoRA r8 jobs"
# sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-lora-r8-full --time=24:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_FULL_CONFIG}",SIZE=full "${SD_STAGE1_WORKER}"
# sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-lora-r8-5100 --time=12:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_5100_CONFIG}",SIZE=5100 "${SD_STAGE1_WORKER}"
# sbatch --chdir="${PROJECT_ROOT}" --job-name=ben-sd-lora-r8-2040 --time=06:00:00 --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_REL="${LORA_2040_CONFIG}",SIZE=2040 "${SD_STAGE1_WORKER}"
