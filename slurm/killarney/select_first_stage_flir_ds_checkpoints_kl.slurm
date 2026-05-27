#!/bin/bash
#SBATCH --job-name=sel-flir-stage1
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
CONFIG_REL="${CONFIG_REL:-configs/eval/checkpoint_selection_first_stage_flir_ds_trainings.yaml}"
CONFIG="$(slurm_config_path "${CONFIG_REL}")"
PREFLIGHT="${PREFLIGHT:-0}"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

slurm_init_runtime

echo "Checkpoint-selection config: ${CONFIG}"
echo "PREFLIGHT: ${PREFLIGHT}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
slurm_print_python_diagnostics
slurm_print_gpu_diagnostics
slurm_require_file "${CONFIG}" "checkpoint-selection config"

echo "Configured runs:"
grep -n "run_identifier:" "${CONFIG}" || true
echo "Configured generation sizes:"
grep -nE "^(stage1_num_images|stage2_extra_images|stage3_extra_images|generation_batch_size|num_inference_steps|output_root|analysis_output_root):" "${CONFIG}" || true

EXTRA_ARGS=()
if [ "${PREFLIGHT}" = "1" ] || [ "${PREFLIGHT}" = "true" ]; then
  EXTRA_ARGS+=(--preflight)
fi

slurm_run_timed "${PROJECT_ROOT}/logs/select-flir-stage1-${JOB_ID}" "FLIR first-stage checkpoint selection" -- \
  python scripts/select_best_checkpoint_and_compute_metrics.py \
    --config "${CONFIG}" \
    "${EXTRA_ARGS[@]}" || exit $?
