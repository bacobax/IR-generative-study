#!/bin/bash
#SBATCH --job-name=flir-sdxl-r8-5k
#SBATCH --account=aip-mpederso
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="configs/datasets/flir/sdxl_adaptation/flir_sdxl_lora_stage1_r8_train_5000.yaml"
OUTPUT_REL="artifacts/checkpoints/stable_diffusion_xl/lora_runs/flir_sdxl_lora_stage1_r8_train_5000"
RUN_LABEL="train_5000"
CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_REL}"

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
echo "Run label: ${RUN_LABEL}"
echo "Training config: ${CONFIG}"
echo "Output dir: ${OUTPUT_DIR}"
which python
python --version
nvidia-smi || true

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi

grep -nE "^(extends|dataset_id|subset_manifest|resolution|train_batch_size|gradient_accumulation_steps|num_train_epochs|checkpointing_epochs|checkpoints_total_limit|save_optimizer_state|output_dir):" "${CONFIG}" || true

/usr/bin/time -v accelerate launch --mixed_precision=fp16 \
  -m src.cli.train_sdxl --config "${CONFIG}"

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

echo "End time: $(date)"

