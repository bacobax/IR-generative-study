#!/bin/bash
#SBATCH --job-name=stable-fm-ot-hf
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="${CONFIG_REL:-configs/fm/train/presets/uncond_latent_flir_sd15_512_b64_hflip_ot.yaml}"
if [[ "${CONFIG_REL}" = /* ]]; then
  CONFIG="${CONFIG_REL}"
else
  CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
fi

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
echo "Config: ${CONFIG}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
which python
python --version

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi

echo "Configured batch size:"
grep -n "batch_size:" "${CONFIG}" | head -n 2 || true
echo "Configured horizontal flip probabilities:"
grep -n "p_hflip_" "${CONFIG}" || true
echo "Configured path mode:"
grep -n -A3 "^path:" "${CONFIG}" || true
echo "Configured mixed precision:"
grep -n "mixed_precision:" "${CONFIG}" | head -n 2 || true
nvidia-smi || true

python -m src.cli.train_flow_matching \
  --config "${CONFIG}" \
  --mixed_precision auto
