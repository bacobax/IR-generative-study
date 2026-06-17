#!/bin/bash
#SBATCH --job-name=stay-fm-v18
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="diffusers-dev"
CONFIG_REL="configs/fm/train/presets/stay_layout_latent_v18_x4_512.yaml"
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
echo "Configured horizontal flip probabilities:"
grep -n "p_hflip_" "${CONFIG}" || true
echo "Configured mixed precision:"
grep -n "mixed_precision:" "${CONFIG}" | head -n 2 || true
echo "Configured patience / weight_decay:"
grep -n "patience:\|weight_decay:" "${CONFIG}" | head -n 4 || true
nvidia-smi || true

LOG_PREFIX="${PROJECT_ROOT}/logs/stay-layout-fm-v18-${JOB_ID}"
LOG_OUT="${LOG_PREFIX}.out"
LOG_ERR="${LOG_PREFIX}.err"
echo "STAY layout FM v18 stdout log: ${LOG_OUT}"
echo "STAY layout FM v18 stderr log: ${LOG_ERR}"

set +e
/usr/bin/time -v python -m src.cli.train_flow_matching \
    --config "${CONFIG}" \
    --mixed_precision auto > "${LOG_OUT}" 2> "${LOG_ERR}"
STATUS=$?
set -e

echo "STAY layout FM v18 exit code: ${STATUS}"
echo "Stdout log: ${LOG_OUT}"
echo "Stderr log: ${LOG_ERR}"
echo "End time: $(date)"
if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi
