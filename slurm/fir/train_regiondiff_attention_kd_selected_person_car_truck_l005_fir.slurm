#!/bin/bash
#SBATCH --job-name=rdiff-kd-sel-l005
#SBATCH --account=def-mpederso
#SBATCH --time=00:06:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="${CONFIG_REL:-configs/fm/train/presets/regiondiff_attention_kd_latent_flir_sd15_512_selected_person_car_truck_l005.yaml}"
if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG="${CONFIG_REL}"
else
  CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
fi
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
echo "Pretrained UNet: ${PRETRAINED_UNET}"
echo "Default teacher checkpoint: ${DEFAULT_TEACHER_CHECKPOINT}"
echo "Teacher checkpoint override: ${TEACHER_CHECKPOINT:-<config default>}"
echo "Resume checkpoint: ${RESUME:-<disabled>}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
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
if [[ ! -f "${PRETRAINED_UNET}" ]]; then
  if [[ "${PRETRAINED_UNET}" != /* && -f "${PROJECT_ROOT}/${PRETRAINED_UNET}" ]]; then
    :
  else
    echo "ERROR: pretrained UNet checkpoint not found: ${PRETRAINED_UNET}" >&2
    exit 1
  fi
fi

if [ -n "${TEACHER_CHECKPOINT}" ]; then
if [[ ! -e "${TEACHER_CHECKPOINT}" ]]; then
  if [[ "${TEACHER_CHECKPOINT}" != /* && -e "${PROJECT_ROOT}/${TEACHER_CHECKPOINT}" ]]; then
    :
  else
    echo "ERROR: TEACHER_CHECKPOINT not found: ${TEACHER_CHECKPOINT}" >&2
    exit 1
  fi
fi
elif [ ! -e "${DEFAULT_TEACHER_CHECKPOINT}" ]; then
    echo "WARNING: default teacher checkpoint not found: ${DEFAULT_TEACHER_CHECKPOINT}"
    echo "         If the YAML path differs on this cluster, submit with:"
    echo "         sbatch --export=ALL,TEACHER_CHECKPOINT=/path/to/teacher ${BASH_SOURCE[0]}"
fi

echo "Configured batch size:"

grep -n "batch_size:" "${CONFIG}" | head -n 2 || true
echo "Configured num workers:"
grep -n "num_workers:" "${CONFIG}" | head -n 2 || true
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

LOG_PREFIX="${PROJECT_ROOT}/logs/regiondiff-attn-kd-selected-l005-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "RegionDiff attention KD selected-category FM stdout log: ${LOG_OUT}"
echo "RegionDiff attention KD selected-category FM stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v python -m src.cli.train_flow_matching "${TRAIN_ARGS[@]}" > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e

echo "RegionDiff attention KD selected-category FM exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
