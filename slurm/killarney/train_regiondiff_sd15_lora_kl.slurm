#!/bin/bash
#SBATCH --job-name=rdiff-sd-lora
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="diffusers-dev"
CONFIG_REL="configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8.yaml"
if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG="${CONFIG_REL}"
else
  CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
fi
LORA_RUN_DIR="${PROJECT_ROOT}/artifacts/checkpoints/stable_diffusion/lora_runs/flir_sd15_lora_stage1_r8"
LORA_MANIFEST="${LORA_RUN_DIR}/stage1_manifest.json"
LORA_WEIGHTS="${LORA_RUN_DIR}/pytorch_lora_weights.safetensors"
FUSED_UNET_CHECKPOINT="${LORA_RUN_DIR}/fused_unet/export"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-0}"

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
echo "LoRA run dir: ${LORA_RUN_DIR}"
echo "LoRA weights: ${LORA_WEIGHTS}"
echo "Fused UNet checkpoint target: ${FUSED_UNET_CHECKPOINT}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
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
if [[ ! -f "${LORA_MANIFEST}" ]]; then
  if [[ "${LORA_MANIFEST}" != /* && -f "${PROJECT_ROOT}/${LORA_MANIFEST}" ]]; then
    :
  else
    echo "ERROR: LoRA stage-1 manifest not found: ${LORA_MANIFEST}" >&2
    exit 1
  fi
fi
if [[ ! -f "${LORA_WEIGHTS}" ]]; then
  if [[ "${LORA_WEIGHTS}" != /* && -f "${PROJECT_ROOT}/${LORA_WEIGHTS}" ]]; then
    :
  else
    echo "ERROR: LoRA weights not found: ${LORA_WEIGHTS}" >&2
    exit 1
  fi
fi

echo "Configured stage1 source:"

grep -n -A4 "^stage1:" "${CONFIG}" || true
echo "Configured batch size:"
grep -n "batch_size:" "${CONFIG}" | head -n 2 || true
echo "Configured num workers:"
grep -n "num_workers:" "${CONFIG}" | head -n 2 || true
echo "Configured train mode:"
grep -n "train_mode:" "${CONFIG}" | head -n 2 || true
echo "Configured mixed precision:"
grep -n "mixed_precision:" "${CONFIG}" | head -n 2 || true
nvidia-smi || true

LOG_PREFIX="${PROJECT_ROOT}/logs/regiondiff-sd-lora-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "RegionDiff SD LoRA stdout log: ${LOG_OUT}"
echo "RegionDiff SD LoRA stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v python -m src.cli.adapt_stable_diffusion     --config "${CONFIG}"     --mixed_precision "${MIXED_PRECISION}"     --num_workers "${NUM_WORKERS}" > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e

echo "RegionDiff SD LoRA exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
