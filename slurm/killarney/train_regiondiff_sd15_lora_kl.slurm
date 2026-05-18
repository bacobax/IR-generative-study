#!/bin/bash
#SBATCH --job-name=rdiff-sd-lora
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
CONFIG_REL="configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8.yaml"
CONFIG="$(slurm_config_path "${CONFIG_REL}")"
LORA_RUN_DIR="${PROJECT_ROOT}/artifacts/checkpoints/stable_diffusion/lora_runs/flir_sd15_lora_stage1_r8"
LORA_MANIFEST="${LORA_RUN_DIR}/stage1_manifest.json"
LORA_WEIGHTS="${LORA_RUN_DIR}/pytorch_lora_weights.safetensors"
FUSED_UNET_CHECKPOINT="${LORA_RUN_DIR}/fused_unet/export"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-0}"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

if [ "${DEBUG_SEGFAULT:-0}" = "1" ]; then
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_SHOW_CPP_STACKTRACES=1
    NUM_WORKERS=0
fi

slurm_init_runtime

echo "Config: ${CONFIG}"
echo "LoRA run dir: ${LORA_RUN_DIR}"
echo "LoRA weights: ${LORA_WEIGHTS}"
echo "Fused UNet checkpoint target: ${FUSED_UNET_CHECKPOINT}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "DEBUG_SEGFAULT: ${DEBUG_SEGFAULT:-0}"
slurm_print_python_diagnostics
slurm_require_file "${CONFIG}" "config"
slurm_require_file "${LORA_MANIFEST}" "LoRA stage-1 manifest"
slurm_require_file "${LORA_WEIGHTS}" "LoRA weights"

slurm_grep_config_keys "Configured stage1 source:" "^stage1:" "${CONFIG}" -n -A4
slurm_grep_config_keys "Configured batch size:" "batch_size:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured num workers:" "num_workers:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured train mode:" "train_mode:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured mixed precision:" "mixed_precision:" "${CONFIG}" -n | head -n 2 || true
slurm_print_gpu_diagnostics

slurm_run_timed "${PROJECT_ROOT}/logs/regiondiff-sd-lora-${JOB_ID}" "RegionDiff SD LoRA" --   python -m src.cli.adapt_stable_diffusion     --config "${CONFIG}"     --mixed_precision "${MIXED_PRECISION}"     --num_workers "${NUM_WORKERS}" || exit $?
