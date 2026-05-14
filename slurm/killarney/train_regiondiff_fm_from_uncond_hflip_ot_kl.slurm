#!/bin/bash
#SBATCH --job-name=rdiff-fm-ot-hf
#SBATCH --account=aip-mpederso
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="diffusers-dev"
CONFIG_REL="configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_ot_b64_hflip.yaml"
CONFIG="$(slurm_config_path "${CONFIG_REL}")"
PRETRAINED_UNET="${PROJECT_ROOT}/artifacts/checkpoints/flow_matching/serious_runs/uncond_latent_flir_sd15_512_hflip_ot/UNET/unet_fm_epoch_300.pt"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

slurm_init_runtime

echo "Config: ${CONFIG}"
echo "Pretrained UNet: ${PRETRAINED_UNET}"
slurm_print_python_diagnostics
slurm_require_file "${CONFIG}" "config"
slurm_require_file "${PRETRAINED_UNET}" "pretrained UNet checkpoint"

slurm_grep_config_keys "Configured batch size:" "batch_size:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured horizontal flip probabilities:" "p_hflip_" "${CONFIG}" -n
slurm_grep_config_keys "Configured path mode:" "^path:" "${CONFIG}" -n -A4
slurm_grep_config_keys "Configured mixed precision:" "mixed_precision:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured sampling/debug:" "^sampling:" "${CONFIG}" -n -A6
slurm_print_gpu_diagnostics

slurm_run_timed "${PROJECT_ROOT}/logs/regiondiff-fm-hflip-ot-${JOB_ID}" "RegionDiff FM hflip OT" --   python -m src.cli.train     --config "${CONFIG}"     --mixed_precision auto || exit $?
