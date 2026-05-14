#!/bin/bash
#SBATCH --job-name=rdiff-kd-sel-l005
#SBATCH --account=aip-mpederso
#SBATCH --time=00:06:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="${ENV_NAME:-diffusers-dev}"
CONFIG_REL="${CONFIG_REL:-configs/fm/train/presets/regiondiff_attention_kd_latent_flir_sd15_512_selected_person_car_truck_l005.yaml}"
CONFIG="$(slurm_config_path "${CONFIG_REL}")"
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

slurm_init_runtime

echo "Config: ${CONFIG}"
echo "Pretrained UNet: ${PRETRAINED_UNET}"
echo "Default teacher checkpoint: ${DEFAULT_TEACHER_CHECKPOINT}"
echo "Teacher checkpoint override: ${TEACHER_CHECKPOINT:-<config default>}"
echo "Resume checkpoint: ${RESUME:-<disabled>}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
slurm_print_python_diagnostics
slurm_require_file "${CONFIG}" "config"
slurm_require_file "${PRETRAINED_UNET}" "pretrained UNet checkpoint"

if [ -n "${TEACHER_CHECKPOINT}" ]; then
    slurm_require_path "${TEACHER_CHECKPOINT}" "TEACHER_CHECKPOINT"
elif [ ! -e "${DEFAULT_TEACHER_CHECKPOINT}" ]; then
    echo "WARNING: default teacher checkpoint not found: ${DEFAULT_TEACHER_CHECKPOINT}"
    echo "         If the YAML path differs on this cluster, submit with:"
    echo "         sbatch --export=ALL,TEACHER_CHECKPOINT=/path/to/teacher ${BASH_SOURCE[0]}"
fi

slurm_grep_config_keys "Configured batch size:" "batch_size:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured num workers:" "num_workers:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured distillation block:" "^distillation:" "${CONFIG}" -n -A14
slurm_grep_config_keys "Configured selected categories:" "selected_categories:" "${CONFIG}" -n
slurm_grep_config_keys "Configured output:" "^output:" "${CONFIG}" -n -A4
slurm_print_gpu_diagnostics

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

slurm_run_timed "${PROJECT_ROOT}/logs/regiondiff-attn-kd-selected-l005-${JOB_ID}" "RegionDiff attention KD selected-category FM" --   python -m src.cli.train "${TRAIN_ARGS[@]}" || exit $?
