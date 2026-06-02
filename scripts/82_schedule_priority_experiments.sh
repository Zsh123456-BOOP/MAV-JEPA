#!/usr/bin/env bash
set -euo pipefail

# Priority scheduler for the current MAV-JEPA experiment phase.
#
# Policy:
# - Quick single-seed experiments are launched immediately on idle GPUs.
# - Multi-seed experiments are launched only when GPU 0 is idle, or when at
#   least two GPUs are idle. This keeps the current GPU-0 chain from being
#   disturbed while still using spare capacity.
# - Each run trains first, then generates/evaluates predictions for that run.
# - Final aggregation is run once after all background jobs finish.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${MAV_JEPA_CONDA_ENV:-mav-jepa}"
CONDA_BIN="${CONDA_BIN:-${HOME}/anaconda3/bin/conda}"
if [[ ! -x "$CONDA_BIN" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
fi
if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Cannot find conda. Set CONDA_BIN." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$("$CONDA_BIN" run -n "$CONDA_ENV" python -c 'import sys; print(sys.executable)')}"
ENV_BIN="$(dirname "$PYTHON_BIN")"
export PATH="$ENV_BIN:$PATH"

GPU_IDS="${GPU_IDS:-0 1 2 3}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-1024}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-5}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-120}"

TASK="${TASK:-gsm8k}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LR="${LR:-2e-5}"
RANK="${RANK:-16}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
VIEW_MAX_LENGTH="${VIEW_MAX_LENGTH:-256}"
EDGE_BUDGET="${EDGE_BUDGET:-1}"
MAX_PROCESS_RSS_GB="${MAX_PROCESS_RSS_GB:-48}"
MAX_SYSTEM_MEMORY_PCT="${MAX_SYSTEM_MEMORY_PCT:-85}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-10}"

GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
GENERATION_MAX_PROMPT_LENGTH="${GENERATION_MAX_PROMPT_LENGTH:-1024}"
GENERATION_GSM8K_MAX_NEW_TOKENS="${GENERATION_GSM8K_MAX_NEW_TOKENS:-192}"
GENERATION_SPIDER_MAX_NEW_TOKENS="${GENERATION_SPIDER_MAX_NEW_TOKENS:-128}"
GENERATION_OVERWRITE="${GENERATION_OVERWRITE:-0}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MAV_JEPA_USE_TORCH_PROFILER="${MAV_JEPA_USE_TORCH_PROFILER:-0}"

latest_real_pilot_dir() {
  find outputs -maxdepth 1 -type d -name 'real_pilot_gsm8k_*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

DEFAULT_OUTPUTS_DIR="$(latest_real_pilot_dir)"
if [[ -z "$DEFAULT_OUTPUTS_DIR" ]]; then
  DEFAULT_OUTPUTS_DIR="outputs/priority_gsm8k_$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)"
fi
OUTPUTS_DIR="${OUTPUTS_DIR:-$DEFAULT_OUTPUTS_DIR}"
SCHED_DIR="${SCHED_DIR:-$OUTPUTS_DIR/scheduler}"
mkdir -p "$SCHED_DIR"

LOG_FILE="${LOG_FILE:-$SCHED_DIR/scheduler.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "$(date -Is) scheduler starting"
echo "$(date -Is) outputs_dir=$OUTPUTS_DIR"
echo "$(date -Is) gpu_ids=[$GPU_IDS] idle_thresholds: memory<=${GPU_MAX_USED_MIB}MiB util<=${GPU_MAX_UTIL}%"

gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
}

idle_gpus() {
  "$PYTHON_BIN" - "$GPU_IDS" "$GPU_MAX_USED_MIB" "$GPU_MAX_UTIL" <<'PY'
import subprocess
import sys

allowed = {item.strip() for item in sys.argv[1].replace(",", " ").split() if item.strip()}
max_used = float(sys.argv[2])
max_util = float(sys.argv[3])

out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
)
idle = []
for line in out.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        continue
    idx, used, util = parts[0], float(parts[1]), float(parts[2])
    if idx in allowed and used <= max_used and util <= max_util:
        idle.append(idx)
print(" ".join(idle))
PY
}

active_pids=()
active_gpus=()

prune_active_jobs() {
  local next_pids=()
  local next_gpus=()
  local i pid
  for i in "${!active_pids[@]}"; do
    pid="${active_pids[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      next_pids+=("$pid")
      next_gpus+=("${active_gpus[$i]}")
    fi
  done
  active_pids=("${next_pids[@]}")
  active_gpus=("${next_gpus[@]}")
}

gpu_is_reserved() {
  local gpu="$1"
  local reserved
  prune_active_jobs
  for reserved in "${active_gpus[@]}"; do
    [[ "$reserved" == "$gpu" ]] && return 0
  done
  return 1
}

available_gpus() {
  local gpu
  for gpu in $(idle_gpus); do
    if ! gpu_is_reserved "$gpu"; then
      printf '%s\n' "$gpu"
    fi
  done
}

first_idle_gpu() {
  available_gpus | head -1
}

idle_gpu_count() {
  available_gpus | wc -l | tr -d ' '
}

gpu0_is_idle() {
  available_gpus | grep -qx "0"
}

lr_tag() {
  sed -e 's/-/m/g' -e 's/\./p/g' <<<"$1"
}

kind_for_method() {
  case "$1" in
    sft_lora|original_llm_jepa_lora|original_llm_jepa_random_dropout) echo "baselines" ;;
    mv_sft_lora|mv_jepa_fixed_lambda|mav_jepa_full|mv_jepa_adaptive_lambda|mv_jepa_adaptive_edge_dropout) echo "mav" ;;
    mav_qr_stopgrad_p25_l005|mav_qr_stopgrad_p50_l005|mav_qr_normmse_p25_l005) echo "mav" ;;
    mav_qr_stopgrad_p125_l005|mav_qra_safe_all_p25_l005|mav_qa_only_p25_l005|mav_ra_only_p25_l005) echo "mav" ;;
    mav_qr_p125_l003_cap003|mav_qr_p125_l003_cap003_nostrip) echo "mav" ;;
    mav_rspan_qrpre_rsuf_p125_l003|mav_qr_rspan_prior_p125_l003|mav_qr_rspan_answerweak_p125_l003) echo "mav" ;;
    *) echo "Unknown method: $1" >&2; exit 2 ;;
  esac
}

run_name_for() {
  local task="$1" method="$2" seed="$3" lr="$4" rank="$5"
  printf '%s_%s_seed%s_lr%s_r%s_full' "$task" "$method" "$seed" "$(lr_tag "$lr")" "$rank"
}

run_complete_with_predictions() {
  local run_dir="$1"
  [[ -f "$run_dir/run_status.json" && -f "$run_dir/results.json" && -f "$run_dir/predictions.jsonl" ]] || return 1
  "$PYTHON_BIN" - "$run_dir" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
status = json.loads((run / "run_status.json").read_text()).get("status")
results = json.loads((run / "results.json").read_text())
ok = status == "success" and results.get("status") == "success" and results.get("eval_status", "success") == "success"
raise SystemExit(0 if ok else 1)
PY
}

run_experiment() {
  local gpu="$1" method="$2" seed="$3" label="$4"
  local kind run_name run_dir port gen_args=()
  kind="$(kind_for_method "$method")"
  run_name="$(run_name_for "$TASK" "$method" "$seed" "$LR" "$RANK")"
  run_dir="$OUTPUTS_DIR/runs/$run_name"
  port="$((29600 + gpu))"

  if run_complete_with_predictions "$run_dir"; then
    echo "$(date -Is) [$label] skip completed $run_name"
    return 0
  fi

  echo "$(date -Is) [$label] gpu=$gpu start train $run_name"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export GPU_INDEX="$gpu"
    "$PYTHON_BIN" scripts/run_task06_matrix.py \
      --kind "$kind" \
      --tasks "$TASK" \
      --methods "$method" \
      --seeds "$seed" \
      --learning_rates "$LR" \
      --lora_ranks "$RANK" \
      --epochs "$EPOCHS" \
      --max_length "$MAX_LENGTH" \
      --view_max_length "$VIEW_MAX_LENGTH" \
      --batch_size "$BATCH_SIZE" \
      --grad_accum "$GRAD_ACCUM" \
      --edge_budget "$EDGE_BUDGET" \
      --gpu_index "$gpu" \
      --master_port "$port" \
      --outputs_dir "$OUTPUTS_DIR" \
      --max_process_rss_gb "$MAX_PROCESS_RSS_GB" \
      --max_system_memory_pct "$MAX_SYSTEM_MEMORY_PCT" \
      --monitor_interval_sec "$MONITOR_INTERVAL_SEC"

    gen_args=(
      --run_dir "$run_dir"
      --outputs_dir "$OUTPUTS_DIR"
      --tasks "$TASK"
      --batch_size "$GENERATION_BATCH_SIZE"
      --max_prompt_length "$GENERATION_MAX_PROMPT_LENGTH"
      --gsm8k_max_new_tokens "$GENERATION_GSM8K_MAX_NEW_TOKENS"
      --spider_max_new_tokens "$GENERATION_SPIDER_MAX_NEW_TOKENS"
      --evaluate
    )
    if [[ "$GENERATION_OVERWRITE" == "1" ]]; then
      gen_args+=(--overwrite)
    fi

    echo "$(date -Is) [$label] gpu=$gpu start generate/evaluate $run_name"
    "$PYTHON_BIN" scripts/55_generate_predictions.py "${gen_args[@]}"
    echo "$(date -Is) [$label] gpu=$gpu done $run_name"
  )
}

launch_job() {
  local gpu="$1" method="$2" seed="$3" label="$4"
  local log="$SCHED_DIR/${label}_${method}_seed${seed}_gpu${gpu}.log"
  local pid
  (
    set -euo pipefail
    run_experiment "$gpu" "$method" "$seed" "$label"
  ) > >(tee -a "$log") 2>&1 &
  pid="$!"
  pids+=("$pid")
  active_pids+=("$pid")
  active_gpus+=("$gpu")
  echo "$(date -Is) launched pid=$pid label=$label method=$method seed=$seed gpu=$gpu log=$log"
}

wait_for_any_gpu() {
  local gpu=""
  while [[ -z "$gpu" ]]; do
    gpu="$(first_idle_gpu)"
    if [[ -z "$gpu" ]]; then
      echo "$(date -Is) no idle GPU for quick queue; sleeping ${POLL_INTERVAL_SEC}s" >&2
      gpu_snapshot >&2 || true
      sleep "$POLL_INTERVAL_SEC"
    fi
  done
  echo "$gpu"
}

wait_for_multiseed_slot() {
  local gpu="" idle_count
  while [[ -z "$gpu" ]]; do
    idle_count="$(idle_gpu_count)"
    if gpu0_is_idle || [[ "$idle_count" -ge 2 ]]; then
      gpu="$(first_idle_gpu)"
      break
    fi
    echo "$(date -Is) multi-seed waits: gpu0 busy and idle_gpu_count=$idle_count; sleeping ${POLL_INTERVAL_SEC}s" >&2
    gpu_snapshot >&2 || true
    sleep "$POLL_INTERVAL_SEC"
  done
  echo "$gpu"
}

pids=()

quick_methods=(
  original_llm_jepa_lora
  original_llm_jepa_random_dropout
  mv_jepa_fixed_lambda
  mv_jepa_adaptive_lambda
  mv_jepa_adaptive_edge_dropout
)

multi_seed_items=(
  "1 sft_lora"
  "1 original_llm_jepa_lora"
  "1 original_llm_jepa_random_dropout"
  "1 mav_jepa_full"
  "2 sft_lora"
  "2 original_llm_jepa_lora"
  "2 original_llm_jepa_random_dropout"
  "2 mav_jepa_full"
)

for method in "${quick_methods[@]}"; do
  gpu="$(wait_for_any_gpu)"
  launch_job "$gpu" "$method" "0" "quick"
  sleep 5
done

for item in "${multi_seed_items[@]}"; do
  read -r seed method <<<"$item"
  gpu="$(wait_for_multiseed_slot)"
  launch_job "$gpu" "$method" "$seed" "multiseed"
  sleep 5
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "$(date -Is) all scheduled jobs finished; refreshing aggregate"
"$PYTHON_BIN" scripts/60_evaluate_all.py --outputs_dir "$OUTPUTS_DIR" --tasks "$TASK" || status=1
"$PYTHON_BIN" scripts/70_aggregate_results.py \
  --outputs_dir "$OUTPUTS_DIR" \
  --output_csv "$OUTPUTS_DIR/aggregate/results.csv" \
  --make_plots || status=1

echo "$(date -Is) scheduler finished status=$status"
exit "$status"
