#!/usr/bin/env bash
set -euo pipefail

sbatch slurm/killarney/ben/diff_scratch/diff_scratch_ben_uncond_2k.slurm
sbatch slurm/killarney/ben/diff_scratch/diff_scratch_ben_uncond_5k.slurm
sbatch slurm/killarney/ben/diff_scratch/diff_scratch_ben_uncond_full.slurm
sbatch slurm/killarney/ben/fm_scratch/fm_scratch_ben_uncond_2k.slurm
sbatch slurm/killarney/ben/fm_scratch/fm_scratch_ben_uncond_5k.slurm
sbatch slurm/killarney/ben/fm_scratch/fm_scratch_ben_uncond_full.slurm
sbatch slurm/killarney/ben/lora_sd15/lora_sd15_ben_2k.slurm
sbatch slurm/killarney/ben/lora_sd15/lora_sd15_ben_5k.slurm
sbatch slurm/killarney/ben/lora_sd15/lora_sd15_ben_full.slurm
sbatch slurm/killarney/ben/lora_sdxl/lora_sdxl_ben_2k.slurm
sbatch slurm/killarney/ben/lora_sdxl/lora_sdxl_ben_5k.slurm
sbatch slurm/killarney/ben/lora_sdxl/lora_sdxl_ben_full.slurm
