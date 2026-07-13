#!/usr/bin/env bash
# Launch all 4 tiny-YOLO runs at 150 epochs across exactly 2 GPUs.
# Two sequential "lanes" (one per GPU) → at most 2 jobs run concurrently, and the
# next job in a lane starts as soon as the previous one finishes (max GPU use under
# the 2-GPU cap). fmaug trains on the already-built merged real+synth dataset (no
# regeneration); the 3 baselines train on the real-only v18 full_train split.
set -u
cd "$(dirname "$0")/.."

EP=150
CFG=configs/yolo/exp_v18_simple_yolo_tiny
MERGED=artifacts/generated/yolo/exp_b/augmented_yolo/small_v4_fmaug/fm_aug/full_train_synthetic_aug.yaml
LOG=artifacts/logs
mkdir -p "$LOG"

run() { # name gpu extra_args...
  # Pin the physical GPU with CUDA_VISIBLE_DEVICES so the process sees exactly one
  # device as cuda:0 (passing --device cuda:1 was NOT honored by the trainer and
  # both lanes collided on GPU0). This guarantees lane isolation under the 2-GPU cap.
  local name="$1" gpu="$2"; shift 2
  echo "[$(date +%H:%M:%S)] START $name on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" conda run -n diffusers-dev python -m src.cli.train_yolo \
    --action train --device cuda:0 --epochs "$EP" "$@" \
    > "$LOG/cmp150_${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $?)"
}

# Lane A on physical GPU 0 : fmaug (merged dataset) -> fast
( run fmaug 0 --config "$CFG/small_v4_fmaug.yaml" --dataset_yaml "$MERGED"
  run fast  0 --config "$CFG/small_v4_fast.yaml"
) > "$LOG/cmp150_laneA.log" 2>&1 &
LANE_A=$!

# Lane B on physical GPU 1 : rareaug -> genaug
( run rareaug 1 --config "$CFG/small_v4_rareaug.yaml"
  run genaug  1 --config "$CFG/small_v4_genaug.yaml"
) > "$LOG/cmp150_laneB.log" 2>&1 &
LANE_B=$!

wait $LANE_A $LANE_B
echo "[$(date +%H:%M:%S)] ALL 4 RUNS COMPLETE"
