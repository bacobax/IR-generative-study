#!/bin/bash
#SBATCH --job-name=sd-dstudio-unet
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="${CONFIG_REL:-configs/sd/train/presets/flir_unet_full_domainstudio_512.yaml}"
CONFIG="$(slurm_config_path "${CONFIG_REL}")"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

slurm_init_runtime

echo "Config: ${CONFIG}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
slurm_print_python_diagnostics
slurm_require_file "${CONFIG}" "config"

slurm_grep_config_keys "Selected training settings:" "^(baseline_mode|unet_train_mode|resolution|train_batch_size|gradient_accumulation_steps|max_train_steps|learning_rate|mixed_precision|enable_xformers_memory_efficient_attention):" "${CONFIG}" -nE
slurm_grep_config_keys "Selected DomainStudio settings:" "^(domainstudio_enabled|domainstudio_source_prompt|domainstudio_target_prompt|domainstudio_lambda_|domainstudio_loss_every_n_steps|domainstudio_min_pairwise_batch|domainstudio_prior_data_dir|domainstudio_prior_cache_dir|domainstudio_generate_prior_if_missing):" "${CONFIG}" -nE
slurm_print_gpu_diagnostics

slurm_run_timed "${PROJECT_ROOT}/logs/sd-domainstudio-unet-${JOB_ID}" "DomainStudio full U-Net Stage-1" --   python -m src.cli.train_sd     --config "${CONFIG}" || exit $?
