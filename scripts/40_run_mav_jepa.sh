#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${MAV_JEPA_CONDA_ENV:-mav-jepa}"
CONDA_BIN="${CONDA_BIN:-${HOME}/anaconda3/bin/conda}"
GPU_INDEX="${GPU_INDEX:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_INDEX}"

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

"$PYTHON_BIN" scripts/run_task06_matrix.py --kind mav "$@"
