#!/usr/bin/env python
"""Run the Task 06 smoke/full training matrix with run artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    script: str
    extra_args: tuple[str, ...]


BASELINE_METHODS = {
    "sft_lora": MethodSpec("sft_lora", "finetune.py", ("--regular",)),
    "original_llm_jepa_lora": MethodSpec(
        "original_llm_jepa_lora",
        "finetune.py",
        ("--predictors", "1", "--last_token", "-3", "--lbd", "1.0"),
    ),
    "original_llm_jepa_random_dropout": MethodSpec(
        "original_llm_jepa_random_dropout",
        "finetune.py",
        ("--predictors", "1", "--last_token", "-3", "--lbd", "1.0", "--jepa_ratio", "0.25"),
    ),
}

MAV_METHODS = {
    "mv_sft_lora": MethodSpec("mv_sft_lora", "finetune_mv.py", ("--regular",)),
    "mv_jepa_fixed_lambda": MethodSpec(
        "mv_jepa_fixed_lambda",
        "finetune_mv.py",
        ("--mv_jepa", "--predictors", "1", "--lambda_base", "1.0", "--edge_dropout", "none"),
    ),
    "mav_jepa_full": MethodSpec(
        "mav_jepa_full",
        "finetune_mv.py",
        (
            "--mv_jepa",
            "--predictors",
            "1",
            "--lambda_base",
            "1.0",
            "--lambda_mode",
            "current_adaptive",
            "--edge_dropout",
            "adaptive",
        ),
    ),
    "mv_jepa_adaptive_lambda": MethodSpec(
        "mv_jepa_adaptive_lambda",
        "finetune_mv.py",
        (
            "--mv_jepa",
            "--predictors",
            "1",
            "--lambda_base",
            "1.0",
            "--lambda_mode",
            "current_adaptive",
            "--edge_dropout",
            "none",
        ),
    ),
    "mv_jepa_adaptive_edge_dropout": MethodSpec(
        "mv_jepa_adaptive_edge_dropout",
        "finetune_mv.py",
        (
            "--mv_jepa",
            "--predictors",
            "1",
            "--lambda_base",
            "1.0",
            "--edge_dropout",
            "adaptive",
        ),
    ),
    "mav_qr_stopgrad_p25_l005": MethodSpec(
        "mav_qr_stopgrad_p25_l005",
        "finetune_mv.py",
        (
            "--mv_jepa",
            "--predictors",
            "0",
            "--lambda_base",
            "0.05",
            "--detach_target",
            "true",
            "--target_no_grad",
            "true",
            "--allowed_edges",
            "Q_to_R",
            "--edge_dropout",
            "adaptive",
            "--edge_budget",
            "1",
            "--jepa_step_prob",
            "0.25",
            "--jepa_start_step",
            "500",
            "--jepa_warmup_steps",
            "1000",
            "--jepa_ce_ratio_cap",
            "0.05",
            "--jepa_reduce",
            "mean",
            "--strip_answer_from_reasoning",
        ),
    ),
    "mav_qr_stopgrad_p50_l005": MethodSpec(
        "mav_qr_stopgrad_p50_l005",
        "finetune_mv.py",
        (
            "--mv_jepa",
            "--predictors",
            "0",
            "--lambda_base",
            "0.05",
            "--detach_target",
            "true",
            "--target_no_grad",
            "true",
            "--allowed_edges",
            "Q_to_R",
            "--edge_dropout",
            "adaptive",
            "--edge_budget",
            "1",
            "--jepa_step_prob",
            "0.50",
            "--jepa_start_step",
            "500",
            "--jepa_warmup_steps",
            "1000",
            "--jepa_ce_ratio_cap",
            "0.05",
            "--jepa_reduce",
            "mean",
            "--strip_answer_from_reasoning",
        ),
    ),
    "mav_qr_normmse_p25_l005": MethodSpec(
        "mav_qr_normmse_p25_l005",
        "finetune_mv.py",
        (
            "--mv_jepa",
            "--predictors",
            "0",
            "--lambda_base",
            "0.05",
            "--detach_target",
            "true",
            "--target_no_grad",
            "true",
            "--allowed_edges",
            "Q_to_R",
            "--edge_dropout",
            "adaptive",
            "--edge_budget",
            "1",
            "--jepa_step_prob",
            "0.25",
            "--jepa_start_step",
            "500",
            "--jepa_warmup_steps",
            "1000",
            "--jepa_ce_ratio_cap",
            "0.05",
            "--jepa_reduce",
            "mean",
            "--mv_loss_type",
            "normalized_mse",
            "--strip_answer_from_reasoning",
        ),
    ),
    "mav_qr_stopgrad_p125_l005": MethodSpec(
        "mav_qr_stopgrad_p125_l005",
        "finetune_mv.py",
        (
            "--mv_jepa",
            "--predictors",
            "0",
            "--lambda_base",
            "0.05",
            "--detach_target",
            "true",
            "--target_no_grad",
            "true",
            "--allowed_edges",
            "Q_to_R",
            "--edge_dropout",
            "adaptive",
            "--edge_budget",
            "1",
            "--jepa_step_prob",
            "0.125",
            "--jepa_start_step",
            "500",
            "--jepa_warmup_steps",
            "1000",
            "--jepa_ce_ratio_cap",
            "0.05",
            "--jepa_reduce",
            "mean",
            "--strip_answer_from_reasoning",
        ),
    ),
    "mav_qra_safe_all_p25_l005": MethodSpec(
        "mav_qra_safe_all_p25_l005",
        "finetune_mv.py",
        (
            "--mv_jepa",
            "--predictors",
            "0",
            "--lambda_base",
            "0.05",
            "--detach_target",
            "true",
            "--target_no_grad",
            "true",
            "--edge_dropout",
            "adaptive",
            "--edge_budget",
            "1",
            "--jepa_step_prob",
            "0.25",
            "--jepa_start_step",
            "500",
            "--jepa_warmup_steps",
            "1000",
            "--jepa_ce_ratio_cap",
            "0.05",
            "--jepa_reduce",
            "mean",
            "--strip_answer_from_reasoning",
        ),
    ),
    "mav_qa_only_p25_l005": MethodSpec(
        "mav_qa_only_p25_l005",
        "finetune_mv.py",
        (
            "--mv_jepa",
            "--predictors",
            "0",
            "--lambda_base",
            "0.05",
            "--detach_target",
            "true",
            "--target_no_grad",
            "true",
            "--allowed_edges",
            "Q_to_A",
            "--min_target_tokens",
            "0",
            "--edge_dropout",
            "adaptive",
            "--edge_budget",
            "1",
            "--jepa_step_prob",
            "0.25",
            "--jepa_start_step",
            "500",
            "--jepa_warmup_steps",
            "1000",
            "--jepa_ce_ratio_cap",
            "0.05",
            "--jepa_reduce",
            "mean",
            "--strip_answer_from_reasoning",
        ),
    ),
    "mav_ra_only_p25_l005": MethodSpec(
        "mav_ra_only_p25_l005",
        "finetune_mv.py",
        (
            "--mv_jepa",
            "--predictors",
            "0",
            "--lambda_base",
            "0.05",
            "--detach_target",
            "true",
            "--target_no_grad",
            "true",
            "--allowed_edges",
            "R_to_A",
            "--min_target_tokens",
            "0",
            "--edge_dropout",
            "adaptive",
            "--edge_budget",
            "1",
            "--jepa_step_prob",
            "0.25",
            "--jepa_start_step",
            "500",
            "--jepa_warmup_steps",
            "1000",
            "--jepa_ce_ratio_cap",
            "0.05",
            "--jepa_reduce",
            "mean",
            "--strip_answer_from_reasoning",
        ),
    ),
}


class GpuMemoryMonitor:
    def __init__(self, gpu_index: str | None):
        self.gpu_index = gpu_index
        self.values_mib: list[float] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def peak_gb(self) -> float | None:
        if not self.values_mib:
            return None
        return round(max(self.values_mib) / 1024, 4)

    def _run(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if self.gpu_index not in {None, ""}:
            query.insert(1, f"--id={self.gpu_index}")
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output(query, text=True, stderr=subprocess.DEVNULL)
                for line in output.splitlines():
                    self.values_mib.append(float(line.strip()))
            except Exception:
                pass
            self.stop_event.wait(2)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise SystemExit("torchrun is not on PATH; run this script inside the MAV-JEPA conda env")

    methods = select_methods(args.kind, args.methods)
    model_path, model_meta = resolve_model(args, repo_root)
    runs = []

    for task in split_values(args.tasks):
        train_file, eval_file = task_paths(task, args.smoke)
        view_config = view_config_for(task)
        for seed in split_values(args.seeds):
            for lr in split_values(args.learning_rates):
                for rank in split_values(args.lora_ranks):
                    for method in methods:
                        runs.append(
                            {
                                "task": task,
                                "train_file": train_file,
                                "eval_file": eval_file,
                                "view_config": view_config,
                                "seed": int(seed),
                                "learning_rate": lr,
                                "lora_rank": int(rank),
                                "method": method,
                            }
                        )

    if args.limit is not None:
        runs = runs[: args.limit]

    for run in runs:
        run_one(args, repo_root, torchrun, model_path, model_meta, run)

    aggregate_csv = Path(args.outputs_dir) / "aggregate" / "results.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/70_aggregate_results.py",
            "--outputs_dir",
            args.outputs_dir,
            "--output_csv",
            str(aggregate_csv),
        ],
        check=True,
    )
    print(f"Wrote {aggregate_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 06 GSM8K/Spider baseline or MAV-JEPA matrix.")
    parser.add_argument("--kind", choices=["baselines", "mav"], required=True)
    parser.add_argument("--tasks", default="gsm8k spider")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model_source", default="modelscope", choices=["modelscope", "local", "auto"])
    parser.add_argument("--fallback_model", default=FALLBACK_MODEL)
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--smoke", action="store_true", help="Use 64-sample train/eval files.")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--learning_rates", default="2e-5")
    parser.add_argument("--lora_ranks", default="16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--view_max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--edge_budget", type=int, default=1)
    parser.add_argument("--gpu_index", default=os.environ.get("GPU_INDEX", "0"))
    parser.add_argument(
        "--master_port",
        type=int,
        help="Explicit torchrun rendezvous port. Overrides --master_port_base.",
    )
    parser.add_argument(
        "--master_port_base",
        type=int,
        default=29600,
        help="Base torchrun port; physical GPU index is added for single-card runs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow_resume_partial",
        action="store_true",
        help="Reuse an existing non-success output directory instead of archiving it before rerun.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--track_flop_original", action="store_true")
    parser.add_argument(
        "--max_process_rss_gb",
        type=float,
        default=64.0,
        help="Kill a run if the torchrun process tree exceeds this resident-memory limit. Use <=0 to disable.",
    )
    parser.add_argument(
        "--max_system_memory_pct",
        type=float,
        default=90.0,
        help="Kill a run if host memory usage exceeds this percentage. Use <=0 to disable.",
    )
    parser.add_argument("--monitor_interval_sec", type=float, default=5.0)
    return parser.parse_args()


def select_methods(kind: str, requested: str) -> list[MethodSpec]:
    available = BASELINE_METHODS if kind == "baselines" else MAV_METHODS
    if requested == "all":
        return list(available.values())
    names = split_values(requested)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"Unknown methods for {kind}: {unknown}")
    return [available[name] for name in names]


def split_values(value: str) -> list[str]:
    return [part for part in re.split(r"[\s,]+", value.strip()) if part]


def resolve_model(args: argparse.Namespace, repo_root: Path) -> tuple[str, dict[str, Any]]:
    meta_path = Path(args.outputs_dir) / "matrix" / "model_resolution.json"
    cmd = [
        sys.executable,
        "scripts/05_resolve_model.py",
        "--model",
        args.model,
        "--source",
        args.model_source,
        "--fallback",
        args.fallback_model,
        "--allow_fallback",
        "--output_json",
        str(meta_path),
    ]
    model_path = subprocess.check_output(cmd, cwd=repo_root, text=True).strip()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model_path, meta


def task_paths(task: str, smoke: bool) -> tuple[str, str]:
    suffix = "_64" if smoke else ""
    train_file = Path("data/mv") / task / f"train{suffix}.jsonl"
    eval_file = Path("data/mv") / task / f"eval{suffix}.jsonl"
    if not train_file.exists() or not eval_file.exists():
        raise FileNotFoundError(f"Missing prepared data for {task}: {train_file}, {eval_file}")
    return str(train_file), str(eval_file)


def view_config_for(task: str) -> str:
    mapping = {
        "gsm8k": "configs/views/gsm8k_qra.yaml",
        "spider": "configs/views/spider_qssql.yaml",
    }
    if task not in mapping:
        raise ValueError(f"Unsupported Task 06 task: {task}")
    return mapping[task]


def run_one(
    args: argparse.Namespace,
    repo_root: Path,
    torchrun: str,
    model_path: str,
    model_meta: dict[str, Any],
    run: dict[str, Any],
) -> None:
    method: MethodSpec = run["method"]
    lr_tag = str(run["learning_rate"]).replace("-", "m").replace(".", "p")
    preset = "smoke" if args.smoke else "full"
    run_name = f"{run['task']}_{method.name}_seed{run['seed']}_lr{lr_tag}_r{run['lora_rank']}_{preset}"
    out_dir = Path(args.outputs_dir) / "runs" / run_name
    command = build_command(args, torchrun, model_path, out_dir, run, method)
    if args.dry_run:
        print(" ".join(command))
        return

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    if (out_dir / "results.json").exists() and not args.overwrite:
        existing = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        if existing.get("status") == "success":
            print(f"Skipping completed run {run_name}")
            return
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite and not args.allow_resume_partial:
        archived = archive_partial_run(out_dir)
        print(f"Archived incomplete/non-success run {run_name} -> {archived}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = build_run_config(args, run, method, run_name, out_dir, model_path, model_meta)
    write_json(out_dir / "run_config.json", config)
    write_json(out_dir / "command.json", {"command": command})

    print(f"Starting {run_name}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", args.gpu_index)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")
    env.setdefault("MAV_JEPA_USE_TORCH_PROFILER", "0")
    start = time.time()
    monitor = GpuMemoryMonitor(args.gpu_index)
    monitor.start()
    resource_reason = None
    with (out_dir / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
        while process.poll() is None:
            time.sleep(max(1.0, args.monitor_interval_sec))
            resource_reason = check_resource_limits(process.pid, args)
            if resource_reason:
                log.write(f"\nRESOURCE_GUARD: {resource_reason}\n")
                log.flush()
                terminate_process_tree(process)
                break
        if process.poll() is None:
            process.wait()
    monitor.stop()
    wall = time.time() - start
    finalize_run(out_dir, config, process.returncode, wall, monitor.peak_gb(), resource_reason=resource_reason)
    if resource_reason:
        raise SystemExit(f"{run_name} stopped by resource guard: {resource_reason}; see {out_dir / 'train.log'}")
    if process.returncode != 0:
        raise SystemExit(f"{run_name} failed with exit code {process.returncode}; see {out_dir / 'train.log'}")
    print(f"Completed {run_name}: {wall:.1f}s")


def build_command(
    args: argparse.Namespace,
    torchrun: str,
    model_path: str,
    out_dir: Path,
    run: dict[str, Any],
    method: MethodSpec,
) -> list[str]:
    base = [
        torchrun,
        "--nproc_per_node=1",
        "--master_port",
        str(resolve_master_port(args)),
        method.script,
        "--train_file",
        run["train_file"],
        "--eval_file",
        run["eval_file"],
        "--output_dir",
        str(out_dir),
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--grad_accum",
        str(args.grad_accum),
        "--num_epochs",
        str(args.epochs),
        "--learning_rate",
        str(run["learning_rate"]),
        "--lora",
        "--lora_rank",
        str(run["lora_rank"]),
        "--finetune_seed",
        str(run["seed"]),
    ]
    if method.script == "finetune.py":
        base.extend(["--model_name", model_path, "--eval_steps", "10", "--keep_output_dir"])
        if args.track_flop_original:
            base.append("--track_flop")
    else:
        base.extend(
            [
                "--model_name",
                args.model,
                "--model_source",
                args.model_source,
                "--view_config",
                run["view_config"],
                "--view_max_length",
                str(args.view_max_length),
                "--edge_budget",
                str(args.edge_budget),
                "--track_flop",
            ]
        )
    base.extend(method.extra_args)
    return base


def build_run_config(
    args: argparse.Namespace,
    run: dict[str, Any],
    method: MethodSpec,
    run_name: str,
    out_dir: Path,
    model_path: str,
    model_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_name,
        "task": run["task"],
        "method": method.name,
        "git_commit": git_commit(),
        "model": args.model,
        "requested_model": model_meta.get("requested_model", args.model),
        "model_name_or_path": model_path,
        "model_source": model_meta.get("model_source", args.model_source),
        "model_fallback": model_meta.get("model_fallback"),
        "fallback_reason": model_meta.get("fallback_reason"),
        "train_file": run["train_file"],
        "eval_file": run["eval_file"],
        "output_dir": str(out_dir),
        "seed": run["seed"],
        "learning_rate": float(run["learning_rate"]),
        "num_epochs": args.epochs,
        "lora": True,
        "lora_rank": run["lora_rank"],
        "max_length": args.max_length,
        "view_max_length": args.view_max_length if method.script != "finetune.py" else None,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "edge_budget": args.edge_budget if method.script != "finetune.py" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", args.gpu_index),
        "torchrun_master_port": resolve_master_port(args),
        "start_time": datetime.now(timezone.utc).isoformat(),
    }


def finalize_run(
    out_dir: Path,
    config: dict[str, Any],
    exit_code: int,
    wall: float,
    peak_vram_gb: float | None,
    resource_reason: str | None = None,
) -> None:
    existing_results = read_json(out_dir / "results.json")
    log_text = (out_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    parsed = parse_training_log(log_text)
    train_wall = existing_results.get("train_wall_clock_sec", existing_results.get("wall_clock_sec"))
    train_gpu_hours = existing_results.get("train_gpu_hours", existing_results.get("gpu_hours"))
    status = (
        "success"
        if exit_code == 0 and resource_reason is None and existing_results.get("status", "success") == "success"
        else "failed"
    )
    results = {
        **existing_results,
        "status": status,
        "exit_code": exit_code,
        "resource_guard_reason": resource_reason,
        "wall_clock_sec": wall,
        "gpu_hours": wall / 3600,
        "train_wall_clock_sec": train_wall,
        "train_gpu_hours": train_gpu_hours,
        "peak_vram_gb": existing_results.get("peak_vram_gb", peak_vram_gb),
        "train_loss": existing_results.get("train_loss", parsed.get("train_loss")),
        "jepa_loss": existing_results.get("jepa_loss", parsed.get("jepa_loss")),
        "estimated_total_flops": existing_results.get("estimated_total_flops"),
        "trainable_params": existing_results.get("trainable_params", parsed.get("trainable_params")),
        "checkpoint_count": len([path for path in out_dir.iterdir() if path.name.startswith("checkpoint")]),
    }
    write_json(out_dir / "results.json", results)

    merged_config = {**config, **read_json(out_dir / "run_config.json")}
    merged_config.update(
        {
            "run_id": config["run_id"],
            "task": config["task"],
            "method": config["method"],
            "learning_rate": config["learning_rate"],
            "end_time": datetime.now(timezone.utc).isoformat(),
            "wall_clock_sec": results["wall_clock_sec"],
            "gpu_hours": results["gpu_hours"],
            "train_wall_clock_sec": results.get("train_wall_clock_sec"),
            "train_gpu_hours": results.get("train_gpu_hours"),
            "peak_vram_gb": results["peak_vram_gb"],
            "trainable_params": results.get("trainable_params"),
            "status": status,
            "exit_code": exit_code,
            "resource_guard_reason": resource_reason,
        }
    )
    write_json(out_dir / "run_config.json", merged_config)
    write_json(out_dir / "run_status.json", {"status": status, "exit_code": exit_code, "resource_guard_reason": resource_reason})

    metrics_path = out_dir / "metrics.jsonl"
    if not metrics_path.exists():
        metrics = {
            "event": "run_completed",
            "status": status,
            "wall_clock_sec": results["wall_clock_sec"],
            "gpu_hours": results["gpu_hours"],
            "peak_vram_gb": results["peak_vram_gb"],
            "avg_steps_per_sec": existing_results.get("avg_steps_per_sec"),
            "avg_tokens_per_sec": existing_results.get("avg_tokens_per_sec"),
            "estimated_total_flops": results.get("estimated_total_flops"),
            "jepa_edges_per_step": existing_results.get("jepa_edges_per_step"),
            "lambda_history": existing_results.get("lambda_history"),
            "edge_sampling_frequency": existing_results.get("edge_sampling_frequency"),
            "same_flop_accuracy": existing_results.get("same_flop_accuracy"),
        }
        metrics_path.write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")


def parse_training_log(log_text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    train_loss_matches = re.findall(r"[\"']train_loss[\"']\s*:\s*([0-9.eE+-]+)", log_text)
    if train_loss_matches:
        parsed["train_loss"] = float(train_loss_matches[-1])
    jepa_matches = re.findall(r"jepa[_ ]loss[\"']?\s*[:=]\s*([0-9.eE+-]+)", log_text, flags=re.IGNORECASE)
    if jepa_matches:
        parsed["jepa_loss"] = float(jepa_matches[-1])
    trainable_matches = re.findall(r"trainable params:\s*([0-9,]+)", log_text, flags=re.IGNORECASE)
    if trainable_matches:
        parsed["trainable_params"] = int(trainable_matches[-1].replace(",", ""))
    return parsed


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def archive_partial_run(out_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = out_dir.with_name(f"{out_dir.name}.interrupted-{stamp}")
    suffix = 1
    while archived.exists():
        archived = out_dir.with_name(f"{out_dir.name}.interrupted-{stamp}-{suffix}")
        suffix += 1
    shutil.move(str(out_dir), str(archived))
    return archived


def check_resource_limits(root_pid: int, args: argparse.Namespace) -> str | None:
    rss_gb = process_tree_rss_gb(root_pid)
    if args.max_process_rss_gb > 0 and rss_gb is not None and rss_gb > args.max_process_rss_gb:
        return f"process_tree_rss_gb={rss_gb:.2f} exceeded limit {args.max_process_rss_gb:.2f}"
    used_pct = system_memory_used_pct()
    if args.max_system_memory_pct > 0 and used_pct is not None and used_pct > args.max_system_memory_pct:
        return f"system_memory_used_pct={used_pct:.2f} exceeded limit {args.max_system_memory_pct:.2f}"
    return None


def process_tree_rss_gb(root_pid: int) -> float | None:
    if os.name == "nt" or not Path("/proc").exists():
        return None
    total_kb = 0
    for pid in process_tree_pids(root_pid):
        status_path = Path("/proc") / str(pid) / "status"
        try:
            for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    total_kb += int(line.split()[1])
                    break
        except Exception:
            continue
    return total_kb / (1024 * 1024)


def process_tree_pids(root_pid: int) -> set[int]:
    pids = {root_pid}
    changed = True
    while changed:
        changed = False
        for proc_dir in Path("/proc").iterdir():
            if not proc_dir.name.isdigit():
                continue
            pid = int(proc_dir.name)
            if pid in pids:
                continue
            try:
                status = (proc_dir / "status").read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            ppid = None
            for line in status.splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                    break
            if ppid in pids:
                pids.add(pid)
                changed = True
    return pids


def system_memory_used_pct() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    values = {}
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, rest = line.split(":", 1)
            values[key] = int(rest.split()[0])
    except Exception:
        return None
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None
    return 100.0 * (1.0 - available / total)


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except Exception:
            process.kill()
        process.wait(timeout=30)


def resolve_master_port(args: argparse.Namespace) -> int:
    if args.master_port is not None:
        return int(args.master_port)
    first_gpu = str(args.gpu_index).split(",", 1)[0].strip()
    try:
        offset = int(first_gpu)
    except ValueError:
        offset = 0
    return int(args.master_port_base) + offset


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
