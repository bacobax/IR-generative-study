#!/bin/bash
#SBATCH --job-name=rdiff-kd-sel-l005
#SBATCH --account=aip-mpederso
#SBATCH --time=00:06:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="${CONFIG_REL:-configs/fm/train/presets/regiondiff_attention_kd_latent_flir_sd15_512_selected_person_car_truck_l005.yaml}"
CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
PRETRAINED_UNET="${PROJECT_ROOT}/artifacts/checkpoints/flow_matching/serious_runs/uncond_latent_flir_sd15_512_hflip_ot/UNET/unet_fm_epoch_300.pt"
DEFAULT_TEACHER_CHECKPOINT="${PROJECT_ROOT}/artifacts/checkpoints/stable_diffusion/layout_runs/flir_sd15_regiondiff_stage2_from_lora_r8_fm_comparable"

MIXED_PRECISION="${MIXED_PRECISION:-auto}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-}"
RESUME="${RESUME:-artifacts/checkpoints/flow_matching/serious_runs/regiondiff_attention_kd_latent_flir_sd15_512_selected_person_car_truck_l005/UNET/unet_fm_epoch_125_ckpt.pt}"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

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
LOG_PREFIX="${PROJECT_ROOT}/logs/regiondiff-attn-kd-selected-l005-${JOB_ID}"

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env: ${ENV_NAME}"
echo "CUDA_VISIBLE_DEVICES from Slurm: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-<unset>}"
echo "Config: ${CONFIG}"
echo "Pretrained UNet: ${PRETRAINED_UNET}"
echo "Default teacher checkpoint: ${DEFAULT_TEACHER_CHECKPOINT}"
echo "Teacher checkpoint override: ${TEACHER_CHECKPOINT:-<config default>}"
echo "Resume checkpoint: ${RESUME:-<disabled>}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"

which python
python --version

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: config not found: ${CONFIG}"
    exit 1
fi

if [ ! -f "${PRETRAINED_UNET}" ]; then
    echo "ERROR: pretrained UNet checkpoint not found: ${PRETRAINED_UNET}"
    exit 1
fi

if [ -n "${TEACHER_CHECKPOINT}" ]; then
    if [ ! -e "${TEACHER_CHECKPOINT}" ]; then
        echo "ERROR: TEACHER_CHECKPOINT does not exist: ${TEACHER_CHECKPOINT}"
        exit 1
    fi
elif [ ! -e "${DEFAULT_TEACHER_CHECKPOINT}" ]; then
    echo "WARNING: default teacher checkpoint not found: ${DEFAULT_TEACHER_CHECKPOINT}"
    echo "         If the YAML path differs on this cluster, submit with:"
    echo "         sbatch --export=ALL,TEACHER_CHECKPOINT=/path/to/teacher ${BASH_SOURCE[0]}"
fi

echo "Configured batch size:"
grep -n "batch_size:" "${CONFIG}" | head -n 1 || true
echo "Configured num workers:"
grep -n "num_workers:" "${CONFIG}" | head -n 1 || true
echo "Configured distillation block:"
grep -n -A14 "^distillation:" "${CONFIG}" || true
echo "Configured selected categories:"
grep -n "selected_categories:" "${CONFIG}" || true
echo "Configured output:"
grep -n -A4 "^output:" "${CONFIG}" || true

nvidia-smi || true

TRAIN_ARGS=(
  --config "${CONFIG}"
  --mixed_precision "${MIXED_PRECISION}"
  --num_workers "${NUM_WORKERS}"
)

if [ -n "${TEACHER_CHECKPOINT}" ]; then
  TRAIN_ARGS+=(--distillation_teacher_checkpoint "${TEACHER_CHECKPOINT}")
fi

if [ -n "${RESUME}" ]; then
  TRAIN_ARGS+=(--resume "${RESUME}")
fi

set +e
/usr/bin/time -v python -m src.cli.train "${TRAIN_ARGS[@]}" \
  > "${LOG_PREFIX}.out" \
  2> "${LOG_PREFIX}.err"
STATUS=$?
set -e

echo "RegionDiff attention KD selected-category FM exit code: ${STATUS}"
echo "Stdout log: ${LOG_PREFIX}.out"
echo "Stderr log: ${LOG_PREFIX}.err"
echo "End time: $(date)"
exit "${STATUS}"
