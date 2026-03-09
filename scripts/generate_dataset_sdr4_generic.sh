#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    ROOT_DIR="$ROOT_DIR_GIT"
else
    ROOT_DIR="$SCRIPT_DIR"
fi

python "$ROOT_DIR/generate_datasets.py" \
    --mode sd15 \
    --lora_dir "$ROOT_DIR/stable_diffusion_15_out/out_ir_lora_sd15r4_generic_prompt/checkpoint-10000" \
    --max_samples 200 \
    --output_dir "$ROOT_DIR/generated/sd15_r4_generic" \
    --max_tries 25 \
    --sd_steps 100 \
    --generic_prompt \
    --device cuda:1