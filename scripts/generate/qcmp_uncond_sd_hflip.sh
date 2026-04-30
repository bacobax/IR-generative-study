#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || { cd "${SCRIPT_DIR}/../.." && pwd; })"
cd "${ROOT_DIR}"

WEIGHTS_DIR="${WEIGHTS_DIR:-artifacts/checkpoints/stable_diffusion/uncond_runs/uncond_latent_flir_sd15_512_hflip/UNET}"
PRESET_PATH="${PRESET_PATH:-configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512_b64_hflip.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/generated/checkpoint_compare/uncond_sd_latent_flir_sd15_512_hflip}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-0}"
MODEL_FAMILY="${MODEL_FAMILY:-sd}"
SPLIT="${SPLIT:-train}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-0}"

export FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY="${FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [ "${OVERWRITE}" = "1" ] || [ "${OVERWRITE}" = "true" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

echo "Run: unconditional latent SD hflip"
echo "Project root: ${ROOT_DIR}"
echo "Weights dir: ${WEIGHTS_DIR}"
echo "Preset path: ${PRESET_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max samples: ${MAX_SAMPLES}; steps: ${STEPS}; batch size: ${BATCH_SIZE}; seed: ${SEED}"
echo "Device: ${DEVICE}; overwrite: ${OVERWRITE}"

/usr/bin/time -v python scripts/standalone/generate_checkpoint_quality_comparison.py \
  --weights_dir "${WEIGHTS_DIR}" \
  --preset_path "${PRESET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_family "${MODEL_FAMILY}" \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --steps "${STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
