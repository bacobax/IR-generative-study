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

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
ENV_NAME="diffusers-dev"
CONFIG_REL="configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8.yaml"
CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
LORA_RUN_DIR="${PROJECT_ROOT}/artifacts/checkpoints/stable_diffusion/lora_runs/flir_sd15_lora_stage1_r8"
LORA_MANIFEST="${LORA_RUN_DIR}/stage1_manifest.json"
LORA_WEIGHTS="${LORA_RUN_DIR}/pytorch_lora_weights.safetensors"
FUSED_UNET_CHECKPOINT="${LORA_RUN_DIR}/fused_unet/export"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

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
echo "Submit dir / project root: ${PROJECT_ROOT}"
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

which python
python --version

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: config not found: ${CONFIG}"
    exit 1
fi

if [ ! -f "${LORA_MANIFEST}" ]; then
    echo "ERROR: LoRA stage-1 manifest not found: ${LORA_MANIFEST}"
    exit 1
fi

if [ ! -f "${LORA_WEIGHTS}" ]; then
    echo "ERROR: LoRA weights not found: ${LORA_WEIGHTS}"
    exit 1
fi

echo "Configured stage1 source:"
grep -n -A4 "^stage1:" "${CONFIG}" || true
echo "Configured batch size:"
grep -n "batch_size:" "${CONFIG}" | head -n 1 || true
echo "Configured train mode:"
grep -n "train_mode:" "${CONFIG}" | head -n 1 || true
echo "Configured mixed precision:"
grep -n "mixed_precision:" "${CONFIG}" | head -n 1 || true

nvidia-smi || true

set +e
/usr/bin/time -v python -m src.cli.train_sd_layout \
  --config "${CONFIG}" \
  --mixed_precision fp16 \
  > "${PROJECT_ROOT}/logs/regiondiff-sd-lora-${JOB_ID}.out" \
  2> "${PROJECT_ROOT}/logs/regiondiff-sd-lora-${JOB_ID}.err"
STATUS=$?
set -e

echo "RegionDiff SD LoRA exit code: ${STATUS}"
echo "End time: $(date)"
exit "${STATUS}"
