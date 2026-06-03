#!/bin/bash
#SBATCH --job-name=lora_sdxl_ben_5k
#SBATCH --account=def-mpederso
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
CONFIG_REL="configs/datasets/ben/sdxl_adaptation/lora_sdxl_ben_5k.yaml"
OUTPUT_REL="artifacts/checkpoints/stable_diffusion_xl/lora_runs/lora_sdxl_ben_5k"
OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_REL}"
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

LOG_PREFIX="${PROJECT_ROOT}/logs/lora_sdxl_ben_5k-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "lora_sdxl_ben_5k stdout log: ${LOG_OUT}"
echo "lora_sdxl_ben_5k stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v accelerate launch --mixed_precision="${MIXED_PRECISION}" -m src.cli.train_sdxl --config "${CONFIG}" > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e


echo "Cleaning non-inference checkpoint state under ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -path "*/checkpoint-*" -type f \( \
  -name "optimizer*" -o \
  -name "scheduler*" -o \
  -name "random_states_*.pkl" -o \
  -name "scaler.pt" \
\) -print -delete || true

mkdir -p "${OUTPUT_DIR}/last" "${OUTPUT_DIR}/best"
for artifact_name in pytorch_lora_weights.safetensors pytorch_lora_weights.bin stage1_manifest.json artifact_manifest.json; do
  if [[ -f "${OUTPUT_DIR}/${artifact_name}" ]]; then
    cp "${OUTPUT_DIR}/${artifact_name}" "${OUTPUT_DIR}/last/${artifact_name}"
    cp "${OUTPUT_DIR}/${artifact_name}" "${OUTPUT_DIR}/best/${artifact_name}"
  fi
done

find "${OUTPUT_DIR}" -maxdepth 2 -type f \( \
  -name "pytorch_lora_weights.safetensors" -o \
  -name "pytorch_lora_weights.bin" -o \
  -name "stage1_manifest.json" -o \
  -name "artifact_manifest.json" \
\) -print | sort

echo "lora_sdxl_ben_5k exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
