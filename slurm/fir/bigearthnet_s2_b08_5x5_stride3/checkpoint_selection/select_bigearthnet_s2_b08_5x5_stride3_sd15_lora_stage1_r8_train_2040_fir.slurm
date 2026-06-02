#!/bin/bash
#SBATCH --job-name=sel-ben-lora-2040
#SBATCH --account=def-mpederso
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="configs/eval/bigearthnet_s2_b08_5x5_stride3_stage1_single_runs/checkpoint_selection_bigearthnet_s2_b08_5x5_stride3_sd15_lora_stage1_r8_train_2040.yaml"
CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
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
which python
python --version
nvidia-smi || true

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: checkpoint-selection config not found: ${CONFIG}" >&2
  exit 1
fi

grep -n "run_identifier:" "${CONFIG}" || true
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

python scripts/select_best_checkpoint_and_compute_metrics.py --config "${CONFIG}" "${EXTRA_ARGS[@]}"
echo "End time: $(date)"
