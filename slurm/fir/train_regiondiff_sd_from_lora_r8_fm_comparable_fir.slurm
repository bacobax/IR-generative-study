#!/bin/bash
#SBATCH --job-name=rdiff-lora-r8
#SBATCH --account=def-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="diffusers-dev"
CONFIG_REL="configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8_fm_comparable.yaml"
if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG="${CONFIG_REL}"
else
  CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
fi
STAGE1_DIR="${PROJECT_ROOT}/artifacts/checkpoints/stable_diffusion/lora_runs/flir_sd15_lora_stage1_r8"
STAGE1_MANIFEST="${STAGE1_DIR}/stage1_manifest.json"
STAGE1_LORA_WEIGHTS="${STAGE1_DIR}/pytorch_lora_weights.safetensors"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

if [ "${DEBUG_SEGFAULT:-0}" = "1" ]; then
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_SHOW_CPP_STACKTRACES=1
    NUM_WORKERS=0
fi

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
echo "CUDA_VISIBLE_DEVICES from Slurm: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-<unset>}"
echo "SLURM_MEM_PER_CPU: ${SLURM_MEM_PER_CPU:-<unset>}"
echo "CPU count: $(nproc 2>/dev/null || echo '<unknown>')"

echo "Config: ${CONFIG}"
echo "Stage-1 LoRA dir: ${STAGE1_DIR}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "RESUME_FROM_CHECKPOINT: ${RESUME_FROM_CHECKPOINT:-<disabled>}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "DEBUG_SEGFAULT: ${DEBUG_SEGFAULT:-0}"
which python
python --version
if [[ ! -f "${CONFIG}" ]]; then
  if [[ "${CONFIG}" != /* && -f "${PROJECT_ROOT}/${CONFIG}" ]]; then
    :
  else
    echo "ERROR: config not found: ${CONFIG}" >&2
    exit 1
  fi
fi
if [[ ! -f "${STAGE1_MANIFEST}" ]]; then
  if [[ "${STAGE1_MANIFEST}" != /* && -f "${PROJECT_ROOT}/${STAGE1_MANIFEST}" ]]; then
    :
  else
    echo "ERROR: stage-1 manifest not found: ${STAGE1_MANIFEST}" >&2
    exit 1
  fi
fi
if [[ ! -f "${STAGE1_LORA_WEIGHTS}" ]]; then
  if [[ "${STAGE1_LORA_WEIGHTS}" != /* && -f "${PROJECT_ROOT}/${STAGE1_LORA_WEIGHTS}" ]]; then
    :
  else
    echo "ERROR: stage-1 LoRA weights not found: ${STAGE1_LORA_WEIGHTS}" >&2
    exit 1
  fi
fi

echo "Configured batch size:"

grep -n "batch_size:" "${CONFIG}" | head -n 2 || true
echo "Configured num workers:"
grep -n "num_workers:" "${CONFIG}" | head -n 2 || true
echo "Configured Stage-1 source:"
grep -n "stage1_dir:\|stage1_checkpoint:" "${CONFIG}" || true
echo "Configured train mode and epochs:"
grep -n "train_mode:\|num_train_epochs:" "${CONFIG}" || true
echo "Configured mixed precision:"
grep -n "mixed_precision:" "${CONFIG}" | head -n 2 || true
nvidia-smi || true

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

LOG_PREFIX="${PROJECT_ROOT}/logs/regiondiff-sd-lora-r8-fm-comparable-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "RegionDiff SD-layout from LoRA r8 stdout log: ${LOG_OUT}"
echo "RegionDiff SD-layout from LoRA r8 stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v python -m src.cli.adapt_stable_diffusion     --config "${CONFIG}"     --mixed_precision "${MIXED_PRECISION}"     --num_workers "${NUM_WORKERS}"     "${RESUME_ARGS[@]}" > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e

echo "RegionDiff SD-layout from LoRA r8 exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
