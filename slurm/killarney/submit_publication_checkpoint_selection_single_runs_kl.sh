#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"

echo "Submitting publication checkpoint-selection jobs from ${PROJECT_ROOT}"
echo "Comment out any sbatch line in this file to skip that run."

# BigEarthNet flow matching
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_fm_ot_train_2040_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_fm_ot_train_5100_publication_kl.slurm

# BigEarthNet SD 1.5 LoRA
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd15_lora_stage1_r8_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd15_lora_stage1_r8_train_2040_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd15_lora_stage1_r8_train_5100_publication_kl.slurm

# BigEarthNet SD 1.5 unconditional latent diffusion
# sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd_uncond_hflip_publication_kl.slurm
# sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd_uncond_hflip_train_2040_publication_kl.slurm
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sd_uncond_hflip_train_5100_publication_kl.slurm

# BigEarthNet SDXL LoRA
sbatch --chdir="${PROJECT_ROOT}" slurm/killarney/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection_publication/select_ben_sdxl_lora_stage1_r8_train_2040_publication_kl.slurm

# FLIR flow matching
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
