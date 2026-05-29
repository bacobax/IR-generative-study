#!/bin/bash
#SBATCH --job-name=qcmp-sd-hf
#SBATCH --account=aip-mpederso
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=15G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
WEIGHTS_DIR="${WEIGHTS_DIR:-artifacts/checkpoints/stable_diffusion/uncond_runs/uncond_latent_flir_sd15_512_hflip/UNET}"
PRESET_PATH="${PRESET_PATH:-configs/datasets/flir/diffusion/uncond_latent_flir_sd15_512_b64_hflip.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/generated/checkpoint_compare/uncond_sd_latent_flir_sd15_512_hflip}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-0}"
MODEL_FAMILY="${MODEL_FAMILY:-sd}"
SPLIT="${SPLIT:-train}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-0}"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

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

echo "Run: unconditional latent SD hflip"
echo "Weights dir: ${WEIGHTS_DIR}"
echo "Preset path: ${PRESET_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Steps: ${STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Seed: ${SEED}"
echo "Device: ${DEVICE}"
echo "Overwrite: ${OVERWRITE}"
which python
python --version
nvidia-smi || true
if [[ ! -e "${WEIGHTS_DIR}" ]]; then
  if [[ "${WEIGHTS_DIR}" != /* && -e "${PROJECT_ROOT}/${WEIGHTS_DIR}" ]]; then
    :
  else
    echo "ERROR: weights dir not found: ${WEIGHTS_DIR}" >&2
    exit 1
  fi
fi
if [[ ! -f "${PRESET_PATH}" ]]; then
  if [[ "${PRESET_PATH}" != /* && -f "${PROJECT_ROOT}/${PRESET_PATH}" ]]; then
    :
  else
    echo "ERROR: preset path not found: ${PRESET_PATH}" >&2
    exit 1
  fi
fi

EXTRA_ARGS=()
if [ "${OVERWRITE}" = "1" ] || [ "${OVERWRITE}" = "true" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

LOG_PREFIX="${PROJECT_ROOT}/logs/qcmp-uncond-sd-hflip-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "Checkpoint quality comparison stdout log: ${LOG_OUT}"
echo "Checkpoint quality comparison stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v python scripts/standalone/generate_checkpoint_quality_comparison.py     --weights_dir "${WEIGHTS_DIR}"     --preset_path "${PRESET_PATH}"     --output_dir "${OUTPUT_DIR}"     --model_family "${MODEL_FAMILY}"     --split "${SPLIT}"     --max_samples "${MAX_SAMPLES}"     --steps "${STEPS}"     --batch_size "${BATCH_SIZE}"     --seed "${SEED}"     --device "${DEVICE}"     "${EXTRA_ARGS[@]}" > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e

echo "Checkpoint quality comparison exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
