#!/bin/bash
#SBATCH --job-name=stable-sd
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/b/bacobax2/IR-generative-study}"
CONDA_ACTIVATE_PATH="${CONDA_ACTIVATE_PATH:-/scratch/b/bacobax2/conda-envs/diffusers-dev/bin/activate}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512_b64.yaml"
CONFIG="$(slurm_config_path "${CONFIG_REL}")"

slurm_init_runtime

echo "Config: ${CONFIG}"
slurm_print_python_diagnostics
slurm_require_file "${CONFIG}" "config"

slurm_grep_config_keys "Configured batch size:" "batch_size:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured mixed precision:" "mixed_precision:" "${CONFIG}" -n | head -n 2 || true
slurm_print_gpu_diagnostics

slurm_run_timed "${PROJECT_ROOT}/logs/sduncond-${JOB_ID}" "SD Uncond" --   python -m src.cli.train_sd_uncond     --config "${CONFIG}"     --mixed_precision auto || exit $?
