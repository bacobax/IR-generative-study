#!/bin/bash
#SBATCH --job-name=sd-lora-rdiff
#SBATCH --account=aip-mpederso
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.out
#SBATCH --error=/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/logs/%x-%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
ENV_NAME="diffusers-dev"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
STAGE1_NUM_WORKERS="${STAGE1_NUM_WORKERS:-4}"
STAGE2_NUM_WORKERS="${STAGE2_NUM_WORKERS:-0}"

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
    STAGE1_NUM_WORKERS=0
    STAGE2_NUM_WORKERS=0
fi

slurm_init_runtime

echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "STAGE1_NUM_WORKERS: ${STAGE1_NUM_WORKERS}"
echo "STAGE2_NUM_WORKERS: ${STAGE2_NUM_WORKERS}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "DEBUG_SEGFAULT: ${DEBUG_SEGFAULT:-0}"
slurm_print_python_diagnostics
slurm_print_gpu_diagnostics
slurm_require_file "configs/sd/train/presets/flir_lora_stage1_r8.yaml" "stage-1 config"
slurm_require_file "configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8.yaml" "stage-2 config"

slurm_run_timed "${PROJECT_ROOT}/logs/sd-lora-rdiff-${JOB_ID}" "SD LoRA r8 then RegionDiff" --   python scripts/train/run_sd_stage1_then_regiondiff.py     --project-root "${PROJECT_ROOT}"     --stage1-config configs/sd/train/presets/flir_lora_stage1_r8.yaml     --stage2-config configs/sd_layout/train/presets/flir_regiondiff_sd15_lora_stage2_r8.yaml     --mixed-precision "${MIXED_PRECISION}"     --stage1-num-workers "${STAGE1_NUM_WORKERS}"     --stage2-num-workers "${STAGE2_NUM_WORKERS}" || exit $?
