#!/bin/bash
#SBATCH --job-name=stable-dual
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus=h100:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /scratch/b/bacobax2/IR-generative-study
mkdir -p logs

# Attiva l'env se serve
source /scratch/b/bacobax2/conda-envs/diffusers-dev/bin/activate

echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "CUDA_VISIBLE_DEVICES before pinning: ${CUDA_VISIBLE_DEVICES:-<unset>}"

nvidia-smi || true

CUDA_VISIBLE_DEVICES=0 python -m src.cli.train \
  --config configs/fm/train/presets/uncond_latent_flir_sd15_512.yaml \
  > logs/fm-${SLURM_JOB_ID}.out 2> logs/fm-${SLURM_JOB_ID}.err &
PID1=$!

wait $PID1
STATUS1=$?

wait $PID2
STATUS2=$?

echo "FM exit code: $STATUS1"
echo "SD Uncond exit code: $STATUS2"
echo "End time: $(date)"

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
  exit 1
fi