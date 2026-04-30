#!/bin/bash
#SBATCH --job-name=qcmp-fm-hf-ot
#SBATCH --account=aip-mpederso
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=15G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"

WEIGHTS_DIR="${WEIGHTS_DIR:-artifacts/checkpoints/flow_matching/serious_runs/uncond_latent_flir_sd15_512_hflip_ot/UNET}"
PRESET_PATH="${PRESET_PATH:-configs/fm/train/presets/uncond_latent_flir_sd15_512_b64_hflip_ot.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/generated/checkpoint_compare/uncond_latent_flir_sd15_512_hflip_ot}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-0}"
MODEL_FAMILY="${MODEL_FAMILY:-fm}"
SPLIT="${SPLIT:-train}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-0}"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

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
LOG_OUT="${PROJECT_ROOT}/logs/qcmp-uncond-fm-hflip-ot-${JOB_ID}.out"
LOG_ERR="${PROJECT_ROOT}/logs/qcmp-uncond-fm-hflip-ot-${JOB_ID}.err"

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env: ${ENV_NAME}"
echo "Run: unconditional FM hflip OT"
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

if [ ! -e "${PROJECT_ROOT}/${WEIGHTS_DIR}" ] && [ ! -e "${WEIGHTS_DIR}" ]; then
    echo "ERROR: weights dir not found: ${WEIGHTS_DIR}"
    exit 1
fi
if [ ! -f "${PROJECT_ROOT}/${PRESET_PATH}" ] && [ ! -f "${PRESET_PATH}" ]; then
    echo "ERROR: preset path not found: ${PRESET_PATH}"
    exit 1
fi

EXTRA_ARGS=()
if [ "${OVERWRITE}" = "1" ] || [ "${OVERWRITE}" = "true" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

set +e
/usr/bin/time -v python scripts/standalone/generate_checkpoint_quality_comparison.py \
  --weights_dir "${WEIGHTS_DIR}" \
  --preset_path "${PRESET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_family "${MODEL_FAMILY}" \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --steps "${STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_OUT}" \
  2> "${LOG_ERR}"
STATUS=$?
set -e

echo "Exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
exit "${STATUS}"
