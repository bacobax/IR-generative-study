#!/usr/bin/env bash
set -euo pipefail

# ====== EDIT THESE ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  ROOT_DIR="$ROOT_DIR_GIT"
else
  ROOT_DIR="$SCRIPT_DIR"
fi
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export TRAIN_DIR="$ROOT_DIR/v18"   # must contain images + metadata.jsonl
export OUTPUT_DIR="$ROOT_DIR/stable_diffusion_15_out/out_ir_lora_sd15r8_generic_prompt"
export VALID_PROMPT="overhead infrared surveillance image with any people or objects"
# ========================

# Optional: choose GPU(s)
export CUDA_VISIBLE_DEVICES=1

# Recommended for stability / speed
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

#--enable_xformers_memory_efficient_attention 

accelerate launch --mixed_precision=fp16 \
  "$ROOT_DIR/train_sd.py" \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --train_data_dir "$TRAIN_DIR" \
  --image_column "image" \
  --caption_column "text" \
  --resolution 512 \
  --center_crop \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0 \
  --max_grad_norm 1.0 \
  --num_train_epochs 100 \
  --checkpointing_steps 1000 \
  --validation_prompt "$VALID_PROMPT" \
  --validation_epochs 1 \
  --num_validation_images 4 \
  --rank 8 \
  --output_dir "$OUTPUT_DIR" \
  --report_to "tensorboard" \
  --logging_dir "$ROOT_DIR/logs" \
  --seed 42 \
  --use_ir_preprocessing \
  --generic_prompt

