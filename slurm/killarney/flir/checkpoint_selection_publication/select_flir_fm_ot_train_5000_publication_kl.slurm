#!/bin/bash
#SBATCH --job-name=pub-flir-fm-5000
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="${CONFIG_REL:-configs/eval/publication_single_runs/flir/checkpoint_selection_flir_fm_ot_train_5000.yaml}"
if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG="${CONFIG_REL}"
else
  CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
fi
PREFLIGHT="${PREFLIGHT:-0}"
GENERATION_SMOKE_TEST="${GENERATION_SMOKE_TEST:-0}"

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
echo "Publication single-run config: ${CONFIG}"
echo "PREFLIGHT: ${PREFLIGHT}"
echo "GENERATION_SMOKE_TEST: ${GENERATION_SMOKE_TEST}"
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

echo "Configured run:"
grep -n "run_identifier:" "${CONFIG}" || true
echo "Configured publication sizes and metrics:"
grep -nE "selection_num_images:|final_total_images:|cleanup_generated_images:|cleanup_dry_run:|compute_clean_fid:|compute_fd_dinov2:|compute_kid:|compute_mmd:|num_inference_steps:|generation_batch_size:|output_root:" "${CONFIG}" || true

EXTRA_ARGS=()
if [ "${PREFLIGHT}" = "1" ] || [ "${PREFLIGHT}" = "true" ]; then
  EXTRA_ARGS+=(--preflight)
fi
if [ "${GENERATION_SMOKE_TEST}" = "1" ] || [ "${GENERATION_SMOKE_TEST}" = "true" ]; then
  EXTRA_ARGS+=(--generation-smoke-test)
fi

python scripts/select_best_checkpoint_and_compute_metrics.py \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"

echo "End time: $(date)"
