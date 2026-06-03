#!/usr/bin/env bash
set -euo pipefail

sbatch slurm/tamia/flir/diff_scratch/diff_scratch_flir_uncond_full_no_hflip.slurm
sbatch slurm/tamia/flir/fm_scratch/fm_scratch_flir_uncond_full_no_hflip.slurm
