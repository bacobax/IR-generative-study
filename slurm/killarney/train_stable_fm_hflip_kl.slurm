#!/bin/bash
#SBATCH --job-name=stable-fm-hf
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="diffusers-dev"
CONFIG_REL="configs/fm/train/presets/uncond_latent_flir_sd15_512_b64_hflip.yaml"
CONFIG="$(slurm_config_path "${CONFIG_REL}")"

slurm_init_runtime

echo "Config: ${CONFIG}"
slurm_print_python_diagnostics
slurm_require_file "${CONFIG}" "config"

slurm_grep_config_keys "Configured batch size:" "batch_size:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured horizontal flip probabilities:" "p_hflip_" "${CONFIG}" -n
slurm_grep_config_keys "Configured mixed precision:" "mixed_precision:" "${CONFIG}" -n | head -n 2 || true
slurm_print_gpu_diagnostics

slurm_run_timed "${PROJECT_ROOT}/logs/fm-hflip-${JOB_ID}" "FM hflip" --   python -m src.cli.train     --config "${CONFIG}"     --mixed_precision auto || exit $?
