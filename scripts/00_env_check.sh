#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/env

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3)
else
  echo "No Python interpreter found. Install Python or set PATH before running this check." >&2
  exit 1
fi

"${PYTHON_CMD[@]}" - <<'PY'
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

out_dir = Path("outputs/env")
out_dir.mkdir(parents=True, exist_ok=True)
report_path = out_dir / "env_report.json"

packages = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "modelscope",
]

report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python_version": sys.version.replace("\n", " "),
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "packages": {},
    "missing_packages": [],
    "cuda_available": False,
    "cuda_device_count": 0,
    "gpus": [],
    "bf16_supported": False,
    "precision": "unknown",
    "one_gpu_smoke_available": False,
    "four_gpu_runs_available": False,
    "status": "failed",
    "notes": [],
}

for package in packages:
    try:
        importlib.import_module(package)
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "unknown"
        report["packages"][package] = version
    except Exception as exc:
        report["packages"][package] = None
        report["missing_packages"].append({"package": package, "error": repr(exc)})

if not any(item["package"] == "torch" for item in report["missing_packages"]):
    import torch

    report["torch_cuda_version"] = torch.version.cuda
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count()) if report["cuda_available"] else 0
    report["one_gpu_smoke_available"] = report["cuda_device_count"] >= 1
    report["four_gpu_runs_available"] = report["cuda_device_count"] >= 4
    if report["cuda_available"]:
        for idx in range(report["cuda_device_count"]):
            props = torch.cuda.get_device_properties(idx)
            report["gpus"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        try:
            report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            report["bf16_supported"] = False
        report["precision"] = "bf16" if report["bf16_supported"] else "fp16"
    else:
        report["notes"].append("No CUDA GPU is visible; MAV-JEPA smoke tests require at least one GPU.")

try:
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    report["nvidia_smi_query"] = smi.stdout.strip()
    if smi.stderr.strip():
        report["nvidia_smi_stderr"] = smi.stderr.strip()
except Exception as exc:
    report["nvidia_smi_error"] = repr(exc)

if report["missing_packages"]:
    report["notes"].append("Install missing Python packages with: pip install -r requirements_mav.txt")
elif not report["cuda_available"]:
    report["notes"].append("GPU check failed.")
else:
    report["status"] = "ok"
    if report["cuda_device_count"] == 1:
        report["notes"].append("Only one GPU is visible; smoke tests are allowed, 4-GPU runs are unavailable.")
    elif report["cuda_device_count"] < 4:
        report["notes"].append("Fewer than four GPUs are visible; main 4-GPU runs are unavailable.")

report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"Python: {report['python_version']}")
print(f"CUDA_VISIBLE_DEVICES: {report['cuda_visible_devices'] or '<unset>'}")
for package in packages:
    print(f"{package}: {report['packages'].get(package)}")
print(f"CUDA available: {report['cuda_available']}")
print(f"CUDA device count: {report['cuda_device_count']}")
for gpu in report["gpus"]:
    print(f"GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory_gb']} GB)")
print(f"BF16 supported: {report['bf16_supported']}")
print(f"Selected precision fallback: {report['precision']}")
print(f"Wrote {report_path}")

if report["status"] != "ok":
    raise SystemExit(1)
PY
