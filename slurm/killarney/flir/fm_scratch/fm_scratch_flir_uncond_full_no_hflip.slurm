#!/bin/bash
#SBATCH --job-name=fm_scratch_flir_uncond_full_no_hflip
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}"
CONFIG_REL="configs/fm/train/presets/fm_scratch_flir_uncond_full_no_hflip.yaml"
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

JOB_ID="${SLURM_JOB_ID:-manual}"
export PROJECT_ROOT ENV_NAME JOB_ID

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
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
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
which python
python --version
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi

grep -nE "^(extends|dataset_id|subset_manifest|resolution|image_size|batch_size|train_batch_size|gradient_accumulation_steps|num_train_epochs|epochs|checkpointing_epochs|checkpoints_total_limit|save_optimizer_state|resume_from_checkpoint|output_dir|model_dir|log_dir|debug_dir):" "${CONFIG}" || true
nvidia-smi || true

LOG_PREFIX="${PROJECT_ROOT}/logs/fm_scratch_flir_uncond_full_no_hflip-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "fm_scratch_flir_uncond_full_no_hflip stdout log: ${LOG_OUT}"
echo "fm_scratch_flir_uncond_full_no_hflip stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v python -m src.cli.train_flow_matching --config "${CONFIG}" --mixed_precision "${MIXED_PRECISION}" > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e

echo "fm_scratch_flir_uncond_full_no_hflip exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
