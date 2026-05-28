#!/bin/bash
#SBATCH --job-name=ben-sd-5x5s3
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-2
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%A_%a.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%A_%a.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}"

SIZES=(full 5100 2040)
CONFIG_RELS=(
  configs/datasets/bigearthnet_s2_b08_5x5_stride3/diffusion/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip.yaml
  configs/datasets/bigearthnet_s2_b08_5x5_stride3/diffusion/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip_train_5100.yaml
  configs/datasets/bigearthnet_s2_b08_5x5_stride3/diffusion/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip_train_2040.yaml
)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "${CONFIG_REL:-}" ]]; then
  SIZE="${SIZE:-custom}"
else
  if (( TASK_ID < 0 || TASK_ID >= ${#SIZES[@]} )); then
    echo "ERROR: unsupported SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0..$((${#SIZES[@]} - 1))" >&2
    exit 2
  fi
  SIZE="${SIZES[${TASK_ID}]}"
  CONFIG_REL="${CONFIG_RELS[${TASK_ID}]}"
fi
CONFIG="$(slurm_config_path "${CONFIG_REL}")"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

slurm_init_runtime

echo "BigEarthNet subset size: ${SIZE}"
echo "Config: ${CONFIG}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
slurm_print_python_diagnostics
slurm_require_file "${CONFIG}" "config"
slurm_grep_config_keys "Configured dataset:" "dataset_id:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured subset manifest:" "subset_manifest:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured batch size:" "batch_size:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured diffusion:" "^diffusion:" "${CONFIG}" -n -A8 || true
slurm_print_gpu_diagnostics

slurm_run_timed "${PROJECT_ROOT}/logs/bigearthnet-sd-uncond-${SIZE}-${JOB_ID}" "BigEarthNet SD uncond ${SIZE}" -- \
  python -m src.cli.train_latent_diffusion \
    --config "${CONFIG}" \
    --mixed_precision "${MIXED_PRECISION}" || exit $?
