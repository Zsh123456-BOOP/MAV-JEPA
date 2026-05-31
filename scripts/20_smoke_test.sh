#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-gsm8k}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
FALLBACK_MODEL="${FALLBACK_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
CONDA_ENV="${MAV_JEPA_CONDA_ENV:-mav-jepa}"
CONDA_BIN="${CONDA_BIN:-${HOME}/anaconda3/bin/conda}"
GPU_INDEX="${GPU_INDEX:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_INDEX}"
export CUDA_VISIBLE_DEVICES

if [[ ! -x "$CONDA_BIN" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
fi

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Cannot find conda. Set CONDA_BIN or install conda." >&2
  exit 1
fi

PYTHON_BIN="$("$CONDA_BIN" run -n "$CONDA_ENV" python -c 'import sys; print(sys.executable)')"
ENV_BIN="$(dirname "$PYTHON_BIN")"
PYTHON=("$PYTHON_BIN")
TORCHRUN="$ENV_BIN/torchrun"

mkdir -p outputs/help data/debug outputs/smoke

"${PYTHON[@]}" finetune.py --help > outputs/help/finetune_help.txt

"${PYTHON[@]}" - <<'PY'
import json
from pathlib import Path

sources = {
    "data/debug/gsm8k_64_train.jsonl": "datasets/gsm8k_train.jsonl",
    "data/debug/gsm8k_64_test.jsonl": "datasets/gsm8k_test.jsonl",
}
for dst, src in sources.items():
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        raise FileNotFoundError(src)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with src_path.open("r", encoding="utf-8") as fin, dst_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if kept >= 64:
                break
            record = json.loads(line)
            if "messages" not in record:
                raise ValueError(f"{src} record is missing messages")
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    if kept != 64:
        raise ValueError(f"{src} only provided {kept} records")
PY

MODEL_META="outputs/smoke/model_resolution.json"
MODEL_PATH="$("${PYTHON[@]}" scripts/05_resolve_model.py \
  --model "$MODEL" \
  --source "$MODEL_SOURCE" \
  --fallback "$FALLBACK_MODEL" \
  --allow_fallback \
  --output_json "$MODEL_META")"

git_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

run_smoke() {
  local run_name="$1"
  local method="$2"
  shift 2
  local out_dir="outputs/smoke/${run_name}"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  cp "$MODEL_META" "$out_dir/model_resolution.json"

  "${PYTHON[@]}" - "$out_dir" "$run_name" "$method" "$MODEL" "$MODEL_PATH" "$MODEL_SOURCE" "$git_commit" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

out_dir, run_name, method, requested_model, model_path, model_source, git_commit = sys.argv[1:]
model_meta = json.loads((Path(out_dir) / "model_resolution.json").read_text(encoding="utf-8"))
config = {
    "run_id": run_name,
    "task": "gsm8k",
    "method": method,
    "git_commit": git_commit,
    "requested_model": requested_model,
    "model_name_or_path": model_path,
    "model_source": model_meta.get("model_source", model_source),
    "model_fallback": model_meta.get("model_fallback"),
    "fallback_reason": model_meta.get("fallback_reason"),
    "seed": 0,
    "num_epochs": 1,
    "num_train_examples": 64,
    "num_eval_examples": 64,
    "lora": True,
    "lora_rank": 16,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "start_time": datetime.now(timezone.utc).isoformat(),
    "efficiency_fields": [
        "wall_clock_sec",
        "gpu_hours",
        "peak_vram_gb",
        "avg_steps_per_sec",
        "avg_tokens_per_sec",
        "estimated_total_flops",
        "jepa_edges_per_step",
        "lambda_history",
        "edge_sampling_frequency",
        "same_flop_accuracy",
    ],
}
(Path(out_dir) / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

  local mem_trace="$out_dir/gpu_mem_mib.txt"
  (
    while true; do
      nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits >> "$mem_trace" 2>/dev/null || true
      sleep 5
    done
  ) &
  local monitor_pid=$!

  local start_ts end_ts status
  start_ts="$(date +%s)"
  set +e
  (
    set -o pipefail
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$TORCHRUN" --nproc_per_node=1 finetune.py "$@"
  ) 2>&1 | tee "$out_dir/train.log"
  pipe_status=("${PIPESTATUS[@]}")
  status="${pipe_status[0]}"
  set -e
  end_ts="$(date +%s)"
  kill "$monitor_pid" >/dev/null 2>&1 || true
  wait "$monitor_pid" 2>/dev/null || true

  "${PYTHON[@]}" - "$out_dir" "$status" "$start_ts" "$end_ts" "$mem_trace" <<'PY'
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(sys.argv[1])
status = int(sys.argv[2])
start_ts = int(sys.argv[3])
end_ts = int(sys.argv[4])
mem_trace = Path(sys.argv[5])
wall = max(0, end_ts - start_ts)
mem_values = []
if mem_trace.exists():
    for line in mem_trace.read_text(encoding="utf-8").splitlines():
        try:
            mem_values.append(float(line.strip()))
        except ValueError:
            pass
peak_vram_gb = round(max(mem_values) / 1024, 3) if mem_values else None
train_log = (out_dir / "train.log").read_text(encoding="utf-8", errors="replace")
lower_log = train_log.lower()
loss_finite = not bool(re.search(r"\b(nan|inf|-inf)\b", lower_log))
checkpoint_count = len(list(out_dir.glob("checkpoint-*")))
has_jepa_log = "jepa_loss" in train_log or "jepa loss" in lower_log

config_path = out_dir / "run_config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
config.update(
    {
        "end_time": datetime.now(timezone.utc).isoformat(),
        "wall_clock_sec": wall,
        "gpu_hours": round(wall / 3600, 6),
        "peak_vram_gb": peak_vram_gb,
    }
)
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

metrics = {
    "event": "run_completed",
    "status": "success" if status == 0 else "failed",
    "wall_clock_sec": wall,
    "gpu_hours": round(wall / 3600, 6),
    "peak_vram_gb": peak_vram_gb,
    "avg_steps_per_sec": None,
    "avg_tokens_per_sec": None,
    "estimated_total_flops": None,
    "jepa_edges_per_step": None,
    "lambda_history": None,
    "edge_sampling_frequency": None,
    "same_flop_accuracy": None,
}
(out_dir / "metrics.jsonl").write_text(json.dumps(metrics) + "\n", encoding="utf-8")

results = {
    "status": metrics["status"],
    "exit_code": status,
    "loss_finite": loss_finite,
    "checkpoint_count": checkpoint_count,
    "checkpoint_created": checkpoint_count > 0,
    "has_jepa_loss_log": has_jepa_log,
    "wall_clock_sec": wall,
    "gpu_hours": metrics["gpu_hours"],
    "peak_vram_gb": peak_vram_gb,
}
(out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

if status != 0:
    raise SystemExit(status)
if not loss_finite:
    raise SystemExit("Loss log contains NaN or Inf")
if checkpoint_count < 1:
    raise SystemExit("No checkpoint-* directory was created")
PY
}

common_args=(
  --train_file data/debug/gsm8k_64_train.jsonl
  --eval_file data/debug/gsm8k_64_test.jsonl
  --model_name "$MODEL_PATH"
  --max_length 512
  --batch_size 1
  --grad_accum 4
  --num_epochs 1
  --learning_rate 2e-5
  --lora
  --lora_rank 16
  --finetune_seed 0
  --eval_steps 1
  --keep_output_dir
)

run_smoke original_sft sft_lora \
  "${common_args[@]}" \
  --output_dir outputs/smoke/original_sft \
  --regular

run_smoke original_jepa original_llm_jepa_lora \
  "${common_args[@]}" \
  --output_dir outputs/smoke/original_jepa \
  --predictors 1 \
  --last_token -3 \
  --lbd 1.0 \
  --jepa_ratio 0.25 \
  --track_flop \
  --debug 5

"${PYTHON[@]}" - <<'PY'
import json
from pathlib import Path

summary = {}
for name in ["original_sft", "original_jepa"]:
    summary[name] = json.loads((Path("outputs/smoke") / name / "results.json").read_text(encoding="utf-8"))
print(json.dumps(summary, indent=2))
PY
