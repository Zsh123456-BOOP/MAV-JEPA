#!/usr/bin/env python
"""Multi-view MAV-JEPA fine-tuning entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

from mavjepa.logging_utils import setup_run_logger, write_json


FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(args.output_dir)
    requested_model = args.model_name
    model_path, model_meta = resolve_model(requested_model, args.model_source)
    args.model_name = model_path
    run_config = build_run_config(args, requested_model, model_meta)
    write_json(Path(args.output_dir) / "run_config.json", run_config)
    logger.info("Resolved model %s -> %s", requested_model, model_path)

    import torch

    from finetune import setup_model_and_tokenizer
    from mavjepa.trainer_mv import MultiViewTrainer

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer = setup_model_and_tokenizer(
        model_path,
        use_lora=args.lora,
        lora_rank=args.lora_rank,
        pretrain=False,
        debug=0,
        seed=args.finetune_seed,
    )
    trainer = MultiViewTrainer(model=model, tokenizer=tokenizer, args=args, logger=logger)
    try:
        results = trainer.train(args.train_file, args.eval_file)
    except Exception as exc:
        results = {"status": "failed", "error": repr(exc)}
        write_json(Path(args.output_dir) / "results.json", results)
        logger.exception("Training failed")
        raise
    results.update({"model_name_or_path": model_path, "model_source": model_meta["model_source"]})
    write_json(Path(args.output_dir) / "results.json", results)
    run_config.update(
        {
            "end_time": datetime.now(timezone.utc).isoformat(),
            "wall_clock_sec": results.get("wall_clock_sec"),
            "gpu_hours": results.get("gpu_hours"),
            "peak_vram_gb": results.get("peak_vram_gb"),
            "avg_steps_per_sec": results.get("avg_steps_per_sec"),
            "avg_tokens_per_sec": results.get("avg_tokens_per_sec"),
            "estimated_total_flops": results.get("estimated_total_flops"),
        }
    )
    write_json(Path(args.output_dir) / "run_config.json", run_config)
    logger.info("Training completed: %s", json.dumps(results, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune with multi-view MAV-JEPA.")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_source", default="modelscope", choices=["modelscope", "local", "auto"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--view_max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--finetune_seed", type=int, default=42)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--mv_jepa", action="store_true")
    parser.add_argument("--regular", action="store_true")
    parser.add_argument("--view_config")
    parser.add_argument("--predictors", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda_base", type=float, default=0.05)
    parser.add_argument("--lambda_min", type=float, default=0.05)
    parser.add_argument("--lambda_max", type=float, default=4.0)
    parser.add_argument("--lambda_mode", choices=["fixed", "current_adaptive", "inverse_loss"], default="fixed")
    parser.add_argument("--lambda_ema_beta", type=float, default=0.95)
    parser.add_argument("--lambda_warmup_steps", type=int, default=50)
    parser.add_argument("--edge_dropout", default="none", choices=["none", "random", "adaptive", "prior"])
    parser.add_argument("--edge_budget", type=int, default=1)
    parser.add_argument("--edge_p_min", type=float, default=0.05)
    parser.add_argument("--target_compute_ratio", type=float, default=1.25)
    parser.add_argument("--mv_loss_type", default="cosine", choices=["cosine", "safe_cosine", "mse", "normalized_mse", "l2"])
    parser.add_argument("--detach_target", type=str_to_bool, default=True)
    parser.add_argument("--jepa_start_step", type=int, default=500)
    parser.add_argument("--jepa_warmup_steps", type=int, default=1000)
    parser.add_argument("--jepa_step_prob", type=float, default=1.0)
    parser.add_argument("--jepa_step_schedule", choices=["constant", "linear_warmup"], default="constant")
    parser.add_argument("--jepa_ce_ratio_cap", type=float, default=0.05)
    parser.add_argument("--jepa_reduce", choices=["mean", "sum"], default="mean")
    parser.add_argument("--weak_edge_step_prob", type=float, default=0.0)
    parser.add_argument("--weak_edge_start_step", type=int, default=3000)
    parser.add_argument("--allowed_edges")
    parser.add_argument("--disable_answer_target_edges", action="store_true")
    parser.add_argument("--min_target_tokens", type=int, default=8)
    parser.add_argument("--target_no_grad", type=str_to_bool, default=True)
    parser.add_argument("--pooling", choices=["last", "mean", "mean_last_k"], default="last")
    parser.add_argument("--target_pooling", choices=["last", "mean", "mean_last_k"], default="mean_last_k")
    parser.add_argument("--pool_last_k", type=int, default=64)
    parser.add_argument("--strip_answer_from_reasoning", action="store_true")
    parser.add_argument("--track_flop", action="store_true")
    parser.add_argument("--same_flop", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_steps", type=int, default=1)
    return parser.parse_args()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "1", "yes", "y"}


def resolve_model(model_name: str, source: str) -> tuple[str, dict[str, str | None]]:
    meta = {
        "requested_model": model_name,
        "model_source": source,
        "model_fallback": None,
        "fallback_reason": None,
    }
    if Path(model_name).exists() or source == "local":
        meta["model_source"] = "local"
        return model_name, meta
    effective = model_name
    if model_name.lower().startswith(("meta-llama/", "google/gemma")):
        effective = FALLBACK_MODEL
        meta["model_fallback"] = FALLBACK_MODEL
        meta["fallback_reason"] = "requested model is usually gated; using Qwen fallback"
    if source in {"modelscope", "auto"}:
        from modelscope import snapshot_download

        local_path = snapshot_download(effective)
        meta["model_source"] = "modelscope"
        return local_path, meta
    return model_name, meta


def build_run_config(args: argparse.Namespace, requested_model: str, model_meta: dict[str, str | None]) -> dict[str, object]:
    return {
        "run_id": Path(args.output_dir).name,
        "git_commit": git_commit(),
        "model": requested_model,
        "model_name_or_path": args.model_name,
        **model_meta,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "task": infer_task(args.train_file, args.view_config),
        "method": "mav_jepa" if args.mv_jepa else "sft",
        "seed": args.finetune_seed,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "view_max_length": args.view_max_length,
        "lora": args.lora,
        "lora_rank": args.lora_rank,
        "view_config": args.view_config,
        "gamma": args.gamma,
        "lambda_base": args.lambda_base,
        "lambda_mode": args.lambda_mode,
        "edge_dropout": args.edge_dropout,
        "edge_budget": args.edge_budget,
        "target_compute_ratio": args.target_compute_ratio,
        "mv_loss_type": args.mv_loss_type,
        "detach_target": args.detach_target,
        "jepa_start_step": args.jepa_start_step,
        "jepa_warmup_steps": args.jepa_warmup_steps,
        "jepa_step_prob": args.jepa_step_prob,
        "jepa_step_schedule": args.jepa_step_schedule,
        "jepa_ce_ratio_cap": args.jepa_ce_ratio_cap,
        "jepa_reduce": args.jepa_reduce,
        "weak_edge_step_prob": args.weak_edge_step_prob,
        "weak_edge_start_step": args.weak_edge_start_step,
        "allowed_edges": args.allowed_edges,
        "disable_answer_target_edges": args.disable_answer_target_edges,
        "min_target_tokens": args.min_target_tokens,
        "target_no_grad": args.target_no_grad,
        "pooling": args.pooling,
        "target_pooling": args.target_pooling,
        "pool_last_k": args.pool_last_k,
        "strip_answer_from_reasoning": args.strip_answer_from_reasoning,
        "track_flop": args.track_flop,
        "same_flop": args.same_flop,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "start_time": datetime.now(timezone.utc).isoformat(),
    }


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def infer_task(train_file: str | None, view_config: str | None) -> str | None:
    text = " ".join(part for part in [train_file, view_config] if part).lower()
    for task in ["gsm8k", "spider", "hotpot"]:
        if task in text:
            return "hotpotqa" if task == "hotpot" else task
    return None


if __name__ == "__main__":
    main()
