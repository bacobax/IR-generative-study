#!/bin/bash
#SBATCH --job-name=sd-dstudio-unet
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="${CONFIG_REL:-configs/sd/train/presets/flir_unet_full_domainstudio_512.yaml}"
if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG="${CONFIG_REL}"
else
  CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
fi

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

cd "${PROJECT_ROOT}"
mkdir -p logs
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -n "${CONDA_ACTIVATE_PATH:-}" ]]; then
  source "${CONDA_ACTIVATE_PATH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
else
  echo "ERROR: conda not found in PATH" >&2
  exit 1
fi

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env: ${ENV_NAME}"
echo "Config: ${CONFIG}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
which python
python --version

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi

echo "Selected training settings:"
grep -nE "^(baseline_mode|unet_train_mode|resolution|train_batch_size|gradient_accumulation_steps|max_train_steps|learning_rate|mixed_precision|enable_xformers_memory_efficient_attention):" "${CONFIG}" || true
echo "Selected DomainStudio settings:"
grep -nE "^(domainstudio_enabled|domainstudio_source_prompt|domainstudio_target_prompt|domainstudio_lambda_|domainstudio_loss_every_n_steps|domainstudio_min_pairwise_batch|domainstudio_prior_data_dir|domainstudio_prior_cache_dir|domainstudio_generate_prior_if_missing):" "${CONFIG}" || true
nvidia-smi || true

python -m src.cli.adapt_stable_diffusion \
  --config "${CONFIG}"
