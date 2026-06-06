#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/configs" ]]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
  else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
  fi
else
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd -P)"
fi

if [[ "${PROJECT_ROOT}" == "${HOME}/projects/"* ]]; then
  PROJECT_CANDIDATE="/project/${PROJECT_ROOT#"${HOME}/projects/"}"
  if [[ -d "${PROJECT_CANDIDATE}/configs" ]]; then
    PROJECT_ROOT="$(cd "${PROJECT_CANDIDATE}" && pwd -P)"
  fi
fi

if [[ "${PROJECT_ROOT}" == /home/* ]]; then
  echo "ERROR: Killarney does not allow sbatch submission from /home." >&2
  echo "       Run this from the repo path under /project or /scratch, or set PROJECT_ROOT to that path." >&2
  echo "       Current resolved PROJECT_ROOT: ${PROJECT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

echo "Submitting publication checkpoint-selection jobs from ${PROJECT_ROOT}"
echo "Comment out any sbatch line in this file to skip that run."

# BigEarthNet flow matching
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_fm_ot_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_fm_ot_train_2040_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_fm_ot_train_5100_publication_kl.slurm

# BigEarthNet SD 1.5 LoRA
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd15_lora_stage1_r8_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd15_lora_stage1_r8_train_2040_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd15_lora_stage1_r8_train_5100_publication_kl.slurm

# BigEarthNet SD 1.5 unconditional latent diffusion
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd_uncond_hflip_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd_uncond_hflip_train_2040_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd_uncond_hflip_train_5100_publication_kl.slurm

# BigEarthNet SDXL LoRA
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sdxl_lora_stage1_r8_train_2040_publication_kl.slurm

# # FLIR flow matching
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/flir/checkpoint_selection_publication/select_flir_fm_ot_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/flir/checkpoint_selection_publication/select_flir_fm_ot_train_2000_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/flir/checkpoint_selection_publication/select_flir_fm_ot_train_5000_publication_kl.slurm

# FLIR SD 1.5 LoRA
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/flir/checkpoint_selection_publication/select_flir_sd15_lora_stage1_r8_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/flir/checkpoint_selection_publication/select_flir_sd15_lora_stage1_r8_train_2000_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/flir/checkpoint_selection_publication/select_flir_sd15_lora_stage1_r8_train_5000_publication_kl.slurm

# FLIR SD 1.5 unconditional latent diffusion
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/flir/checkpoint_selection_publication/select_flir_sd_uncond_hflip_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/flir/checkpoint_selection_publication/select_flir_sd_uncond_hflip_train_2000_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/flir/checkpoint_selection_publication/select_flir_sd_uncond_hflip_train_5000_publication_kl.slurm

# FLIR SDXL LoRA
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/flir/checkpoint_selection_publication/select_flir_sdxl_lora_stage1_r8_train_2000_publication_kl.slurm
