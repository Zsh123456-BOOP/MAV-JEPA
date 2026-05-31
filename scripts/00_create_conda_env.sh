#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${MAV_JEPA_CONDA_ENV:-mav-jepa}"
PYTHON_VERSION="${MAV_JEPA_PYTHON:-3.11}"
TORCH_VERSION="${MAV_JEPA_TORCH:-2.7.1}"
TORCHVISION_VERSION="${MAV_JEPA_TORCHVISION:-0.22.1}"
TORCH_CUDA_INDEX="${MAV_JEPA_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BIN="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  CONDA_BIN="$HOME/anaconda3/bin/conda"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
  CONDA_BIN="$HOME/miniforge3/bin/conda"
else
  echo "Could not find conda. Install Anaconda/Miniconda/Miniforge first." >&2
  exit 1
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists; reusing it."
else
  "$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "$TORCH_CUDA_INDEX"

"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -r requirements_mav.txt

# Align with the upstream setup.sh-tested package family while keeping CUDA
# PyTorch installed from the explicit wheel index above.
"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install \
  "transformers==4.55.2" "peft==0.17.0" "numpy==2.3.2"

echo "Conda env '$ENV_NAME' is ready."
echo "Validate with: CUDA_VISIBLE_DEVICES=0 $CONDA_BIN run -n $ENV_NAME bash scripts/00_env_check.sh"
