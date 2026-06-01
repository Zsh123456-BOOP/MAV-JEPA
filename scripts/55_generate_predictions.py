#!/usr/bin/env python
"""Generate GSM8K/Spider predictions for trained MAV-JEPA runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PREDICTION_FILE = "predictions.jsonl"


def main() -> None:
    args = parse_args()
    run_dirs = select_run_dirs(args)
    if not run_dirs:
        print("No matching run directories found.")
        return

    summaries = []
    for run_dir in run_dirs:
        summaries.append(generate_for_run(run_dir, args))

    if args.aggregate:
        subprocess.run(
            [
                sys.executable,
                "scripts/70_aggregate_results.py",
                "--outputs_dir",
                args.outputs_dir,
                "--output_csv",
                str(Path(args.outputs_dir) / "aggregate" / "results.csv"),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    print(json.dumps({"generated_runs": summaries}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions for MAV-JEPA GSM8K/Spider runs.")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--run_dir", action="append", help="Specific run directory. May be repeated.")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "spider"])
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--limit", type=int, default=0, help="Limit examples per run; 0 means all eval rows.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--spider_max_new_tokens", type=int, default=128)
    parser.add_argument("--gsm8k_max_new_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--spider_db_dir")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    return parser.parse_args()


def select_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        candidates = [Path(path) for path in args.run_dir]
    else:
        candidates = list(iter_run_dirs(Path(args.outputs_dir)))

    tasks = set(args.tasks)
    methods = set(args.methods)
    selected = []
    for run_dir in candidates:
        config = read_json(run_dir / "run_config.json")
        if not config:
            continue
        task = config.get("task") or infer_task(config.get("eval_file") or config.get("train_file"))
        method = config.get("method")
        if task not in tasks:
            continue
        if "all" not in methods and method not in methods:
            continue
        if (run_dir / PREDICTION_FILE).exists() and not args.overwrite:
            print(f"Skipping {run_dir}: {PREDICTION_FILE} exists")
            continue
        selected.append(run_dir)
    return selected


def iter_run_dirs(outputs_dir: Path):
    for parent_name in ["runs", "smoke"]:
        parent = outputs_dir / parent_name
        if not parent.exists():
            continue
        for path in sorted(parent.iterdir()):
            if path.is_dir() and (path / "run_config.json").exists():
                yield path


def generate_for_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(run_dir / "run_config.json")
    task = config.get("task") or infer_task(config.get("eval_file") or config.get("train_file"))
    eval_file = config.get("eval_file")
    if not eval_file:
        raise ValueError(f"{run_dir} has no eval_file in run_config.json")
    eval_path = resolve_path(eval_file)
    rows = read_jsonl(eval_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"{eval_path} has no rows")

    artifact_kind, artifact_dir = locate_model_artifact(run_dir)
    tokenizer, model = load_model_and_tokenizer(config, artifact_kind, artifact_dir, args)

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    output_path = run_dir / PREDICTION_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_new_tokens = 0
    do_sample = args.temperature > 0
    max_new_tokens = task_max_new_tokens(task, args)

    with output_path.open("w", encoding="utf-8") as handle:
        for start_idx in range(0, len(rows), max(1, args.batch_size)):
            batch = rows[start_idx : start_idx + max(1, args.batch_size)]
            prompts = [prompt_from_record(row, tokenizer) for row in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            )
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            prompt_width = inputs["input_ids"].shape[1]
            for row, prompt, output_ids in zip(batch, prompts, outputs):
                generated_ids = output_ids[prompt_width:]
                total_new_tokens += int(generated_ids.numel())
                generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                record = prediction_record(row, prompt, generated, task)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    wall = time.time() - start
    peak_vram_gb = peak_gpu_memory_gb(torch)
    summary = {
        "generation_status": "success",
        "prediction_file": str(output_path),
        "generation_num_examples": len(rows),
        "generation_wall_clock_sec": wall,
        "generation_gpu_hours": wall / 3600 if torch.cuda.is_available() else 0.0,
        "generation_peak_vram_gb": peak_vram_gb,
        "generation_total_new_tokens": total_new_tokens,
        "generation_avg_examples_per_sec": len(rows) / max(wall, 1e-6),
        "generation_avg_new_tokens_per_sec": total_new_tokens / max(wall, 1e-6),
        "generation_max_new_tokens": max_new_tokens,
        "generation_batch_size": args.batch_size,
        "generation_artifact_kind": artifact_kind,
        "generation_artifact_dir": str(artifact_dir),
    }
    write_json(run_dir / "generation_summary.json", summary)
    update_json(run_dir / "results.json", summary)
    update_json(run_dir / "run_config.json", summary)

    if args.evaluate:
        eval_summary = evaluate_run(run_dir, task, args.spider_db_dir)
        summary.update({"eval_status": eval_summary.get("status"), "accuracy": eval_summary.get("accuracy")})
    print(f"Generated {len(rows)} predictions for {run_dir}")
    return {"run_dir": str(run_dir), **summary}


def load_model_and_tokenizer(
    config: dict[str, Any],
    artifact_kind: str,
    artifact_dir: Path,
    args: argparse.Namespace,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_dir = artifact_dir if has_tokenizer(artifact_dir) else resolve_base_model(config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = dtype_from_arg(args.torch_dtype, torch)
    if artifact_kind == "adapter":
        from peft import PeftModel

        base_path = resolve_base_model(config)
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True,
        )
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, artifact_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            artifact_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True,
        )
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model


def locate_model_artifact(run_dir: Path) -> tuple[str, Path]:
    candidates = [run_dir / "checkpoint-final", run_dir]
    checkpoints = sorted(
        [path for path in run_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates.extend(checkpoints)
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return "adapter", candidate
    for candidate in candidates:
        if (candidate / "config.json").exists() and has_model_weights(candidate):
            return "full", candidate
    raise FileNotFoundError(f"No adapter or full-model checkpoint found under {run_dir}")


def resolve_base_model(config: dict[str, Any]) -> str:
    model_path = config.get("model_name_or_path")
    if model_path and Path(str(model_path)).exists():
        return str(model_path)
    model_id = config.get("model") or config.get("requested_model") or model_path
    if not model_id:
        raise ValueError("run_config.json does not identify a base model")
    source = config.get("model_source", "modelscope")
    if source in {"modelscope", "auto"}:
        from modelscope import snapshot_download

        return str(snapshot_download(str(model_id)))
    return str(model_id)


def prompt_from_record(record: dict[str, Any], tokenizer: Any) -> str:
    messages = record.get("messages") or []
    prompt_messages = [message for message in messages if message.get("role") != "assistant"][:2]
    if not prompt_messages and "views" in record:
        prompt_messages = [
            {"role": "system", "content": default_system_prompt(record.get("task"))},
            {"role": "user", "content": record["views"].get("Q") or record["views"].get("QS") or ""},
        ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages) + "\nassistant:"


def prediction_record(row: dict[str, Any], prompt: str, generated: str, task: str | None) -> dict[str, Any]:
    messages = row.get("messages") or []
    gold = None
    for message in reversed(messages):
        if message.get("role") == "assistant":
            gold = message.get("content")
            break
    if gold is None and row.get("views"):
        gold = row["views"].get("A") or row["views"].get("SQL")
    return {
        "id": row.get("id"),
        "task": task or row.get("task"),
        "prediction": generated,
        "generated": generated,
        "gold": gold,
        "db_id": (row.get("meta") or {}).get("db_id") or row.get("db_id"),
        "prompt": prompt,
    }


def evaluate_run(run_dir: Path, task: str | None, spider_db_dir: str | None) -> dict[str, Any]:
    module_path = REPO_ROOT / "scripts" / "60_evaluate_all.py"
    spec = importlib.util.spec_from_file_location("mav_jepa_eval_all", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config = read_json(run_dir / "run_config.json")
    return module.evaluate_run(run_dir, config, task or config.get("task"), spider_db_dir)


def task_max_new_tokens(task: str | None, args: argparse.Namespace) -> int:
    if task == "gsm8k":
        return args.gsm8k_max_new_tokens
    if task == "spider":
        return args.spider_max_new_tokens
    return args.max_new_tokens


def dtype_from_arg(name: str, torch: Any) -> Any:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def has_model_weights(path: Path) -> bool:
    return any((path / name).exists() for name in ["model.safetensors", "pytorch_model.bin"])


def has_tokenizer(path: Path) -> bool:
    return any((path / name).exists() for name in ["tokenizer.json", "tokenizer_config.json"])


def peak_gpu_memory_gb(torch: Any) -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**3), 4)


def default_system_prompt(task: str | None) -> str:
    if task == "spider":
        return "Convert natural language to SQL."
    return "Answer the math question, show steps."


def infer_task(path_text: str | None) -> str | None:
    text = str(path_text or "").lower()
    for task in ["gsm8k", "spider"]:
        if task in text:
            return task
    return None


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_json(path: Path, updates: dict[str, Any]) -> None:
    payload = read_json(path)
    payload.update(updates)
    write_json(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
