#!/usr/bin/env bash
set -euo pipefail



SCRIPTS=(
  "/projets/Fbassignana/diffusers_try/flow_matching_trial/scripts/generate_dataset_fm_guided_combo_maxmin.sh"
  "/projets/Fbassignana/diffusers_try/flow_matching_trial/scripts/generate_dataset_fm_guided_combo_maxmin_tune_1a.sh"
  "/projets/Fbassignana/diffusers_try/flow_matching_trial/scripts/generate_dataset_fm_guided_combo_maxmin_tune_1b.sh"
  "/projets/Fbassignana/diffusers_try/flow_matching_trial/scripts/generate_dataset_fm_guided_combo_maxmin_tune_2a.sh"
  "/projets/Fbassignana/diffusers_try/flow_matching_trial/scripts/generate_dataset_fm_guided_combo_maxmin_tune_2b.sh"
  "/projets/Fbassignana/diffusers_try/flow_matching_trial/scripts/generate_dataset_fm_guided_combo_minmax.sh"
)

pids=()
declare -A pid2name

kill_all() {
  # terminate all children still running
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  # give a moment, then hard-kill if needed
  sleep 1 || true
  for pid in "${pids[@]:-}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}

trap 'kill_all' INT TERM

# Start all
for s in "${SCRIPTS[@]}"; do
  bash "$s" &
  pid=$!
  pids+=("$pid")
  pid2name["$pid"]="$s"
  echo "Started $s (pid=$pid)"
done

# Track running pids
running=("${pids[@]}")

# Loop: wait for any job to finish; if it failed -> kill the rest; if ok -> keep waiting
while ((${#running[@]} > 0)); do
  # wait for *any* child to finish
  if wait -n "${running[@]}"; then
    # one finished successfully; remove finished PIDs from running list
    still=()
    for pid in "${running[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        still+=("$pid")
      fi
    done
    running=("${still[@]}")
  else
    # a job failed; capture its exit code and stop everything
    code=$?
    echo "A job failed (exit=$code). Stopping remaining jobs..."
    kill_all
    exit "$code"
  fi
done

echo "All jobs finished successfully."
# Start next script after all concurrent jobs completed
NEXT_SCRIPT="/projets/Fbassignana/diffusers_try/flow_matching_trial/scripts/analysis.sh"
echo "Starting follow-up script: $NEXT_SCRIPT"
bash "$NEXT_SCRIPT"

exit 0