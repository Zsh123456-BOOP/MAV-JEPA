#!/usr/bin/env bash
set -euo pipefail

# Wait until one physical GPU is idle enough, then run the requested command with
# only that GPU visible. Defaults are intentionally conservative and can be
# overridden by environment variables.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AUTORUN_DIR="${AUTORUN_DIR:-outputs/autostart}"
mkdir -p "$AUTORUN_DIR"

LOG_FILE="${AUTORUN_LOG:-$AUTORUN_DIR/wait_for_gpu.log}"
LOCK_FILE="${AUTORUN_LOCK:-$AUTORUN_DIR/wait_for_gpu.lock}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-1024}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-5}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-300}"

exec > >(tee -a "$LOG_FILE") 2>&1
exec 9>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "$(date -Is) another autostart watcher is already running: $LOCK_FILE"
    exit 2
  fi
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "$#" -eq 0 && -z "${RUN_COMMAND:-}" ]]; then
  set -- bash scripts/81_run_full_sweeps.sh
elif [[ "$#" -eq 0 ]]; then
  set -- bash -lc "$RUN_COMMAND"
fi

echo "$(date -Is) waiting for a GPU: ids=[$GPU_IDS], max_used_mib=$GPU_MAX_USED_MIB, max_util=$GPU_MAX_UTIL"
echo "$(date -Is) command: $*"

while true; do
  selected_gpu="$(
    python - "$GPU_IDS" "$GPU_MAX_USED_MIB" "$GPU_MAX_UTIL" <<'PY'
import subprocess
import sys

allowed = {item.strip() for item in sys.argv[1].replace(",", " ").split() if item.strip()}
max_used = float(sys.argv[2])
max_util = float(sys.argv[3])

try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
except Exception:
    raise SystemExit(1)

best = None
for line in out.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        continue
    idx, used, util = parts[0], float(parts[1]), float(parts[2])
    if allowed and idx not in allowed:
        continue
    if used <= max_used and util <= max_util:
        candidate = (used, util, int(idx), idx)
        if best is None or candidate < best:
            best = candidate
if best is not None:
    print(best[3])
PY
  )" || selected_gpu=""

  if [[ -n "$selected_gpu" ]]; then
    export CUDA_VISIBLE_DEVICES="$selected_gpu"
    export GPU_INDEX="$selected_gpu"
    export MAV_JEPA_AUTORUN_GPU="$selected_gpu"
    cat > "$AUTORUN_DIR/last_start.json" <<JSON
{
  "start_time": "$(date -Is)",
  "gpu_index": "$selected_gpu",
  "command": "$*"
}
JSON
    echo "$(date -Is) selected GPU $selected_gpu; starting command"
    set +e
    "$@"
    status="$?"
    set -e
    cat > "$AUTORUN_DIR/last_exit.json" <<JSON
{
  "end_time": "$(date -Is)",
  "gpu_index": "$selected_gpu",
  "exit_code": $status
}
JSON
    echo "$(date -Is) command finished with exit code $status"
    exit "$status"
  fi

  echo "$(date -Is) no GPU below thresholds; sleeping ${POLL_INTERVAL_SEC}s"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
  sleep "$POLL_INTERVAL_SEC"
done
