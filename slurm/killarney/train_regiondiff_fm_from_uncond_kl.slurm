#!/bin/bash
#SBATCH --job-name=rdiff-fm
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
CONFIG_REL="configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_b64.yaml"
CONFIG="${PROJECT_ROOT}/${CONFIG_REL}"
PRETRAINED_UNET="${PROJECT_ROOT}/artifacts/checkpoints/flow_matching/serious_runs/uncond_latent_flir_sd15_512/UNET/unet_fm_best.pt"

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
echo "Pretrained UNet: ${PRETRAINED_UNET}"

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

echo "Configured batch size:"
grep -n "batch_size:" "${CONFIG}" | head -n 1 || true
echo "Configured epochs:"
grep -n "epochs:" "${CONFIG}" | head -n 1 || true
echo "Configured mixed precision:"
grep -n "mixed_precision:" "${CONFIG}" | head -n 1 || true

nvidia-smi || true

set +e
/usr/bin/time -v python -m src.cli.train \
  --config "${CONFIG}" \
  --mixed_precision auto \
  > "${PROJECT_ROOT}/logs/regiondiff-fm-${JOB_ID}.out" \
  2> "${PROJECT_ROOT}/logs/regiondiff-fm-${JOB_ID}.err"
STATUS=$?
set -e

echo "RegionDiff FM exit code: ${STATUS}"
echo "End time: $(date)"
exit "${STATUS}"
