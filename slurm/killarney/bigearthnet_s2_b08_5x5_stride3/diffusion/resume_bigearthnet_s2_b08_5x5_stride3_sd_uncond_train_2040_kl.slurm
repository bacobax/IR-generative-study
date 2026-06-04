#!/bin/bash
#SBATCH --job-name=ben-sd-res-2040
#SBATCH --account=aip-mpederso
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}"
SIZE="${SIZE:-2040}"
CONFIG_REL="${CONFIG_REL:-configs/datasets/bigearthnet_s2_b08_5x5_stride3/diffusion/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_b64_hflip_train_2040.yaml}"
RESUME="${RESUME:-${PROJECT_ROOT}/artifacts/checkpoints/stable_diffusion/uncond_runs/uncond_latent_bigearthnet_s2_b08_5x5_stride3_sd15_512_hflip_train_2040/UNET/unet_sd_uncond_epoch_150_ckpt.pt}"
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
echo "RESUME: ${RESUME}"
which python
python --version
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi
echo "Configured dataset:"
grep -n "dataset_id:" "${CONFIG}" | head -n 2 || true
echo "Configured subset manifest:"
grep -n "subset_manifest:" "${CONFIG}" | head -n 2 || true
echo "Configured output:"
grep -n -A5 "^output:" "${CONFIG}" || true
echo "Configured diffusion:"
grep -n -A8 "^diffusion:" "${CONFIG}" || true
nvidia-smi || true

LOG_PREFIX="${PROJECT_ROOT}/logs/bigearthnet-sd-uncond-resume-${SIZE}-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "BigEarthNet SD uncond resume ${SIZE} stdout log: ${LOG_OUT}"
echo "BigEarthNet SD uncond resume ${SIZE} stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v \
  python -m src.cli.train_latent_diffusion \
    --config "${CONFIG}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --resume "${RESUME}" > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e

echo "BigEarthNet SD uncond resume ${SIZE} exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
