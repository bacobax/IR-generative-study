#!/bin/bash
#SBATCH --job-name=rdiff-lora-r8
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
CONFIG_REL="configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8_fm_comparable.yaml"
CONFIG="$(slurm_config_path "${CONFIG_REL}")"
STAGE1_DIR="${PROJECT_ROOT}/artifacts/checkpoints/stable_diffusion/lora_runs/flir_sd15_lora_stage1_r8"
STAGE1_MANIFEST="${STAGE1_DIR}/stage1_manifest.json"
STAGE1_LORA_WEIGHTS="${STAGE1_DIR}/pytorch_lora_weights.safetensors"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"

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
echo "Stage-1 LoRA dir: ${STAGE1_DIR}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "RESUME_FROM_CHECKPOINT: ${RESUME_FROM_CHECKPOINT:-<disabled>}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "DEBUG_SEGFAULT: ${DEBUG_SEGFAULT:-0}"
slurm_print_python_diagnostics
slurm_require_file "${CONFIG}" "config"
slurm_require_file "${STAGE1_MANIFEST}" "stage-1 manifest"
slurm_require_file "${STAGE1_LORA_WEIGHTS}" "stage-1 LoRA weights"

slurm_grep_config_keys "Configured batch size:" "batch_size:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured num workers:" "num_workers:" "${CONFIG}" -n | head -n 2 || true
slurm_grep_config_keys "Configured Stage-1 source:" "stage1_dir:\|stage1_checkpoint:" "${CONFIG}" -n
slurm_grep_config_keys "Configured train mode and epochs:" "train_mode:\|num_train_epochs:" "${CONFIG}" -n
slurm_grep_config_keys "Configured mixed precision:" "mixed_precision:" "${CONFIG}" -n | head -n 2 || true
slurm_print_gpu_diagnostics

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

slurm_run_timed "${PROJECT_ROOT}/logs/regiondiff-sd-lora-r8-fm-comparable-${JOB_ID}" "RegionDiff SD-layout from LoRA r8" --   python -m src.cli.train_sd_layout     --config "${CONFIG}"     --mixed_precision "${MIXED_PRECISION}"     --num_workers "${NUM_WORKERS}"     "${RESUME_ARGS[@]}" || exit $?
