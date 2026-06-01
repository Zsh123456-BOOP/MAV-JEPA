#!/usr/bin/env bash
set -euo pipefail

# Run the next-stage GSM8K/Spider training sweep, then refresh evaluation,
# aggregation, ablations, and analysis artifacts. Defaults are intentionally
# smoke-scale after the 2026-06 memory incident; opt into full sweeps by setting
# SWEEP_SMOKE=0 and explicit seeds/lrs/ranks.
#
# Tune the sweep size with env:
#   SWEEP_TASKS="gsm8k spider"
#   SWEEP_SEEDS="0 1 2"
#   SWEEP_LEARNING_RATES="1e-5 2e-5 4e-5"
#   SWEEP_LORA_RANKS="32 64 128"
#   SWEEP_SMOKE=0
#   SWEEP_LIMIT=...

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${MAV_JEPA_CONDA_ENV:-mav-jepa}"
CONDA_BIN="${CONDA_BIN:-${HOME}/anaconda3/bin/conda}"
GPU_INDEX="${GPU_INDEX:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_INDEX}"

SWEEP_TASKS="${SWEEP_TASKS:-gsm8k spider}"
SWEEP_SEEDS="${SWEEP_SEEDS:-0}"
SWEEP_LEARNING_RATES="${SWEEP_LEARNING_RATES:-2e-5}"
SWEEP_LORA_RANKS="${SWEEP_LORA_RANKS:-16}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-1}"
SWEEP_MAX_LENGTH="${SWEEP_MAX_LENGTH:-512}"
SWEEP_VIEW_MAX_LENGTH="${SWEEP_VIEW_MAX_LENGTH:-256}"
SWEEP_BATCH_SIZE="${SWEEP_BATCH_SIZE:-1}"
SWEEP_GRAD_ACCUM="${SWEEP_GRAD_ACCUM:-4}"
SWEEP_EDGE_BUDGET="${SWEEP_EDGE_BUDGET:-1}"
SWEEP_SMOKE="${SWEEP_SMOKE:-1}"
SWEEP_OVERWRITE="${SWEEP_OVERWRITE:-0}"
SWEEP_OUTPUTS_DIR="${SWEEP_OUTPUTS_DIR:-outputs}"
SWEEP_MAX_PROCESS_RSS_GB="${SWEEP_MAX_PROCESS_RSS_GB:-64}"
SWEEP_MAX_SYSTEM_MEMORY_PCT="${SWEEP_MAX_SYSTEM_MEMORY_PCT:-90}"
RUN_BASELINES="${RUN_BASELINES:-0}"
RUN_MAV="${RUN_MAV:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MAV_JEPA_USE_TORCH_PROFILER="${MAV_JEPA_USE_TORCH_PROFILER:-0}"

if [[ ! -x "$CONDA_BIN" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
fi

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Cannot find conda. Set CONDA_BIN or install conda." >&2
  exit 1
fi

PYTHON_BIN="$("$CONDA_BIN" run -n "$CONDA_ENV" python -c 'import sys; print(sys.executable)')"
ENV_BIN="$(dirname "$PYTHON_BIN")"
export PATH="$ENV_BIN:$PATH"

common_args=(
  --tasks "$SWEEP_TASKS"
  --seeds "$SWEEP_SEEDS"
  --learning_rates "$SWEEP_LEARNING_RATES"
  --lora_ranks "$SWEEP_LORA_RANKS"
  --epochs "$SWEEP_EPOCHS"
  --max_length "$SWEEP_MAX_LENGTH"
  --view_max_length "$SWEEP_VIEW_MAX_LENGTH"
  --batch_size "$SWEEP_BATCH_SIZE"
  --grad_accum "$SWEEP_GRAD_ACCUM"
  --edge_budget "$SWEEP_EDGE_BUDGET"
  --gpu_index "$GPU_INDEX"
  --outputs_dir "$SWEEP_OUTPUTS_DIR"
  --max_process_rss_gb "$SWEEP_MAX_PROCESS_RSS_GB"
  --max_system_memory_pct "$SWEEP_MAX_SYSTEM_MEMORY_PCT"
)

if [[ -n "${SWEEP_LIMIT:-}" ]]; then
  common_args+=(--limit "$SWEEP_LIMIT")
fi

if [[ "$SWEEP_SMOKE" == "1" ]]; then
  common_args+=(--smoke)
fi

if [[ "$SWEEP_OVERWRITE" == "1" ]]; then
  common_args+=(--overwrite)
fi

echo "$(date -Is) starting MAV-JEPA sweep on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "$(date -Is) tasks=[$SWEEP_TASKS] seeds=[$SWEEP_SEEDS] lrs=[$SWEEP_LEARNING_RATES] ranks=[$SWEEP_LORA_RANKS]"

if [[ "$RUN_BASELINES" == "1" ]]; then
  "$PYTHON_BIN" scripts/run_task06_matrix.py --kind baselines "${common_args[@]}"
fi

if [[ "$RUN_MAV" == "1" ]]; then
  "$PYTHON_BIN" scripts/run_task06_matrix.py --kind mav "${common_args[@]}"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  "$PYTHON_BIN" scripts/60_evaluate_all.py --outputs_dir "$SWEEP_OUTPUTS_DIR" --tasks gsm8k spider
fi

if [[ "$RUN_PLOTS" == "1" ]]; then
  "$PYTHON_BIN" scripts/70_aggregate_results.py --outputs_dir "$SWEEP_OUTPUTS_DIR" --output_csv "$SWEEP_OUTPUTS_DIR/aggregate/results.csv" --make_plots
else
  "$PYTHON_BIN" scripts/70_aggregate_results.py --outputs_dir "$SWEEP_OUTPUTS_DIR" --output_csv "$SWEEP_OUTPUTS_DIR/aggregate/results.csv"
fi

echo "$(date -Is) MAV-JEPA sweep finished"
