#!/bin/bash
#SBATCH --job-name=rdiff-lora-r8
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
ENV_NAME="diffusers-dev"
CONFIG_REL="configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8_fm_comparable.yaml"
CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
STAGE1_DIR="${PROJECT_ROOT}/artifacts/checkpoints/stable_diffusion/lora_runs/flir_sd15_lora_stage1_r8"
STAGE1_MANIFEST="${STAGE1_DIR}/stage1_manifest.json"
STAGE1_LORA_WEIGHTS="${STAGE1_DIR}/pytorch_lora_weights.safetensors"
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

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
else
    echo "ERROR: conda not found in PATH"
    exit 1
fi

JOB_ID="${SLURM_JOB_ID:-manual}"

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env: ${ENV_NAME}"
echo "CUDA_VISIBLE_DEVICES from Slurm: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
echo "Config: ${CONFIG}"
echo "Stage-1 LoRA dir: ${STAGE1_DIR}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "DEBUG_SEGFAULT: ${DEBUG_SEGFAULT:-0}"
which python
python --version

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: config not found: ${CONFIG}"
    exit 1
fi

if [ ! -f "${STAGE1_MANIFEST}" ]; then
    echo "ERROR: stage-1 manifest not found: ${STAGE1_MANIFEST}"
    exit 1
fi

if [ ! -f "${STAGE1_LORA_WEIGHTS}" ]; then
    echo "ERROR: stage-1 LoRA weights not found: ${STAGE1_LORA_WEIGHTS}"
    exit 1
fi

echo "Configured batch size:"
grep -n "batch_size:" "${CONFIG}" | head -n 1 || true
echo "Configured num workers:"
grep -n "num_workers:" "${CONFIG}" | head -n 1 || true
echo "Configured Stage-1 source:"
grep -n "stage1_dir:\\|stage1_checkpoint:" "${CONFIG}" || true
echo "Configured train mode and epochs:"
grep -n "train_mode:\\|num_train_epochs:" "${CONFIG}" || true
echo "Configured mixed precision:"
grep -n "mixed_precision:" "${CONFIG}" | head -n 1 || true

nvidia-smi || true

set +e
/usr/bin/time -v python -m src.cli.train_sd_layout \
  --config "${CONFIG}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --num_workers "${NUM_WORKERS}" \
  > "${PROJECT_ROOT}/logs/regiondiff-sd-lora-r8-fm-comparable-${JOB_ID}.out" \
  2> "${PROJECT_ROOT}/logs/regiondiff-sd-lora-r8-fm-comparable-${JOB_ID}.err"
STATUS=$?
set -e

echo "RegionDiff SD-layout from LoRA r8 exit code: ${STATUS}"
echo "End time: $(date)"
exit "${STATUS}"
