#!/bin/bash
#SBATCH --job-name=stable-fm
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /scratch/b/bacobax2/IR-generative-study
mkdir -p logs

source /scratch/b/bacobax2/conda-envs/diffusers-dev/bin/activate

CONFIG="configs/fm/train/presets/uncond_latent_flir_sd15_512_b64.yaml"
JOB_ID="${SLURM_JOB_ID:-manual}"

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES from Slurm: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-<unset>}"
echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-<unset>}"
echo "SLURM_MEM_PER_CPU: ${SLURM_MEM_PER_CPU:-<unset>}"
echo "CPU count: $(nproc 2>/dev/null || echo '<unknown>')"
echo "Config: ${CONFIG}"
echo "Configured batch size:"
grep -n "batch_size:" "${CONFIG}" | head -n 1 || true
echo "Configured mixed precision:"
grep -n "mixed_precision:" "${CONFIG}" | head -n 1 || true

nvidia-smi || true

set +e
/usr/bin/time -v python -m src.cli.train \
  --config "${CONFIG}" \
  --mixed_precision auto \
  > "logs/fm-${JOB_ID}.out" 2> "logs/fm-${JOB_ID}.err"
STATUS=$?
set -e

echo "FM exit code: ${STATUS}"
echo "End time: $(date)"
exit "${STATUS}"
