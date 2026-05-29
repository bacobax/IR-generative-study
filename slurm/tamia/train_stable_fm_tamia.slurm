#!/bin/bash
#SBATCH --job-name=stable-fm
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/b/bacobax2/IR-generative-study}"
CONDA_ACTIVATE_PATH="${CONDA_ACTIVATE_PATH:-/scratch/b/bacobax2/conda-envs/diffusers-dev/bin/activate}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="configs/fm/train/presets/uncond_latent_flir_sd15_512_b64.yaml"
if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG="${CONFIG_REL}"
else
  CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
fi

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

echo "Configured batch size:"

grep -n "batch_size:" "${CONFIG}" | head -n 2 || true
echo "Configured mixed precision:"
grep -n "mixed_precision:" "${CONFIG}" | head -n 2 || true
nvidia-smi || true

LOG_PREFIX="${PROJECT_ROOT}/logs/fm-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "FM stdout log: ${LOG_OUT}"
echo "FM stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v python -m src.cli.train_flow_matching     --config "${CONFIG}"     --mixed_precision auto > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e

echo "FM exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
