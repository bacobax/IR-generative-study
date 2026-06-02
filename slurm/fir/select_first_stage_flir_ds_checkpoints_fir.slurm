#!/bin/bash
#SBATCH --job-name=sel-flir-stage1
#SBATCH --account=def-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="${CONFIG_REL:-configs/eval/checkpoint_selection_first_stage_flir_ds_trainings.yaml}"
if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG="${CONFIG_REL}"
else
  CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
fi
PREFLIGHT="${PREFLIGHT:-0}"
GENERATION_SMOKE_TEST="${GENERATION_SMOKE_TEST:-0}"
CLEANUP_CHECKPOINTS="${CLEANUP_CHECKPOINTS:-0}"

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
echo "Checkpoint-selection config: ${CONFIG}"
echo "PREFLIGHT: ${PREFLIGHT}"
echo "GENERATION_SMOKE_TEST: ${GENERATION_SMOKE_TEST}"
echo "CLEANUP_CHECKPOINTS: ${CLEANUP_CHECKPOINTS}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
which python
python --version
nvidia-smi || true

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: checkpoint-selection config not found: ${CONFIG}" >&2
  exit 1
fi

echo "Configured runs:"
grep -n "run_identifier:" "${CONFIG}" || true
echo "Configured generation sizes:"
grep -nE "^(stage1_num_images|stage2_extra_images|stage3_extra_images|generation_batch_size|num_inference_steps|output_root|analysis_output_root):" "${CONFIG}" || true

EXTRA_ARGS=()
if [ "${PREFLIGHT}" = "1" ] || [ "${PREFLIGHT}" = "true" ]; then
  EXTRA_ARGS+=(--preflight)
fi
if [ "${GENERATION_SMOKE_TEST}" = "1" ] || [ "${GENERATION_SMOKE_TEST}" = "true" ]; then
  EXTRA_ARGS+=(--generation-smoke-test)
fi
if [ "${CLEANUP_CHECKPOINTS}" = "1" ] || [ "${CLEANUP_CHECKPOINTS}" = "true" ]; then
  EXTRA_ARGS+=(--cleanup-checkpoints)
fi

python scripts/select_best_checkpoint_and_compute_metrics.py \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"

echo "End time: $(date)"
