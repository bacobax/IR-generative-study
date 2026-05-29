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

set -euo pipefail
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
echo "CUDA_VISIBLE_DEVICES from Slurm: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-<unset>}"
echo "SLURM_MEM_PER_CPU: ${SLURM_MEM_PER_CPU:-<unset>}"
echo "CPU count: $(nproc 2>/dev/null || echo '<unknown>')"

echo "BigEarthNet subset size: ${SIZE}"
echo "Config: ${CONFIG}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
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
echo "Configured dataset:"
grep -n "dataset_id:" "${CONFIG}" | head -n 2 || true
echo "Configured subset manifest:"
grep -n "subset_manifest:" "${CONFIG}" | head -n 2 || true
echo "Configured batch size:"
grep -n "batch_size:" "${CONFIG}" | head -n 2 || true
echo "Configured diffusion:"
grep -n -A8 "^diffusion:" "${CONFIG}" || true
nvidia-smi || true

LOG_PREFIX="${PROJECT_ROOT}/logs/bigearthnet-sd-uncond-${SIZE}-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "BigEarthNet SD uncond ${SIZE} stdout log: ${LOG_OUT}"
echo "BigEarthNet SD uncond ${SIZE} stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v \
  python -m src.cli.train_latent_diffusion \
    --config "${CONFIG}" \
    --mixed_precision "${MIXED_PRECISION}" > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e

echo "BigEarthNet SD uncond ${SIZE} exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
