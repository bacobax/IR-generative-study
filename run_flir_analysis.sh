#!/usr/bin/env bash
# Launches the FLIR subgroup analysis backend (FastAPI/uvicorn) and frontend
# (Vite) together. Ctrl+C stops both, including any processes they spawn
# (e.g. uvicorn's reload worker, vite's esbuild process).

set -uo pipefail
set -m  # each background job gets its own process group, so we can kill it as a unit

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_DIR="$ROOT_DIR/frontend/flir-subgroup-analysis"

cd "$ROOT_DIR"

# Free the backend port if a previous run left a server dangling.
EXISTING_PIDS="$(lsof -tiTCP:"$BACKEND_PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$EXISTING_PIDS" ]]; then
  echo "Port $BACKEND_PORT is already in use by PID(s) $EXISTING_PIDS - stopping them."
  kill $EXISTING_PIDS 2>/dev/null || true
  sleep 1
fi

PIDS=()

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Shutting down..."
  for pid in "${PIDS[@]:-}"; do
    [[ -z "$pid" ]] && continue
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    [[ -z "$pid" ]] && continue
    kill -KILL -- "-$pid" 2>/dev/null || true
  done
  exit 0
}
trap cleanup INT TERM EXIT

python "$ROOT_DIR/serve_flir_analysis.py" --host "$HOST" --port "$BACKEND_PORT" &
PIDS+=("$!")

(cd "$FRONTEND_DIR" && npm run dev) &
PIDS+=("$!")

echo "Backend  -> http://$HOST:$BACKEND_PORT (pid ${PIDS[0]})"
echo "Frontend -> http://127.0.0.1:5173 (pid ${PIDS[1]})"
echo "Press Ctrl+C to stop both."

# bash 3.2 (macOS default) has no `wait -n`, so poll until either exits.
while kill -0 "${PIDS[0]}" 2>/dev/null && kill -0 "${PIDS[1]}" 2>/dev/null; do
  sleep 1
done
