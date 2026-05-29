#!/bin/bash
#SBATCH --job-name=rdiff-fm-ot-hf
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
CONFIG_REL="configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_ot_b64_hflip.yaml"
if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG="${CONFIG_REL}"
else
  CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
fi
PRETRAINED_UNET="${PROJECT_ROOT}/artifacts/checkpoints/flow_matching/serious_runs/uncond_latent_flir_sd15_512_hflip_ot/UNET/unet_fm_epoch_300.pt"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

echo "Configured batch size:"

grep -n "batch_size:" "${CONFIG}" | head -n 2 || true
echo "Configured horizontal flip probabilities:"
grep -n "p_hflip_" "${CONFIG}" || true
echo "Configured path mode:"
grep -n -A4 "^path:" "${CONFIG}" || true
echo "Configured mixed precision:"
grep -n "mixed_precision:" "${CONFIG}" | head -n 2 || true
echo "Configured sampling/debug:"
grep -n -A6 "^sampling:" "${CONFIG}" || true
nvidia-smi || true

LOG_PREFIX="${PROJECT_ROOT}/logs/regiondiff-fm-hflip-ot-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "RegionDiff FM hflip OT stdout log: ${LOG_OUT}"
echo "RegionDiff FM hflip OT stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v python -m src.cli.train_flow_matching     --config "${CONFIG}"     --mixed_precision auto > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e

echo "RegionDiff FM hflip OT exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
