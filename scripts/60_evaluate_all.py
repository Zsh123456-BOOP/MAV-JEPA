#!/usr/bin/env python
"""Evaluate generated predictions for MAV-JEPA runs when available."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mavjepa.metrics import evaluate_gsm8k_rows, evaluate_spider_rows, load_jsonl


PREDICTION_FILENAMES = [
    "predictions.jsonl",
    "generations.jsonl",
    "eval_predictions.jsonl",
    "outputs.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MAV-JEPA outputs for GSM8K and Spider.")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "spider"])
    parser.add_argument("--spider_db_dir")
    args = parser.parse_args()

    tasks = set(args.tasks)
    evaluated = []
    for run_dir in iter_run_dirs(Path(args.outputs_dir)):
        config = read_json(run_dir / "run_config.json")
        task = config.get("task") or infer_task(config.get("train_file"))
        if task not in tasks:
            continue
        summary = evaluate_run(run_dir, config, task, args.spider_db_dir)
        evaluated.append(summary)
    report_path = Path(args.outputs_dir) / "aggregate" / "evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"runs": evaluated}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Evaluated {len(evaluated)} runs; wrote {report_path}")


def iter_run_dirs(outputs_dir: Path):
    for parent_name in ["runs", "smoke"]:
        parent = outputs_dir / parent_name
        if not parent.exists():
            continue
        for path in sorted(parent.iterdir()):
            if path.is_dir() and (path / "run_config.json").exists():
                yield path


def evaluate_run(run_dir: Path, config: dict[str, Any], task: str, spider_db_dir: str | None) -> dict[str, Any]:
    prediction_path = find_prediction_file(run_dir)
    results_path = run_dir / "results.json"
    results = read_json(results_path)
    if prediction_path is None:
        metrics = null_metrics(task)
        summary = {
            "run_name": config.get("run_id") or run_dir.name,
            "task": task,
            "status": "no_predictions",
            "prediction_file": None,
            **metrics,
        }
        update_results(results_path, results, metrics, "no_predictions")
        write_json(run_dir / "eval_summary.json", summary)
        return summary

    try:
        rows = align_prediction_rows(prediction_path, config.get("eval_file"))
        if task == "gsm8k":
            metrics = evaluate_gsm8k_rows(rows)
            metrics["exact_match"] = metrics["final_answer_exact_match"]
            metrics["accuracy"] = metrics["numeric_exact_match"]
        elif task == "spider":
            metrics = evaluate_spider_rows(rows, spider_db_dir=spider_db_dir)
            metrics["exact_match"] = metrics["sql_string_exact_match"]
            metrics["exec_acc"] = metrics["execution_accuracy"]
            metrics["accuracy"] = metrics["execution_accuracy"] or metrics["sql_string_exact_match"]
        else:
            metrics = null_metrics(task)
        status = "success"
    except Exception as exc:
        metrics = null_metrics(task)
        metrics["evaluation_error"] = repr(exc)
        status = "failed"

    summary = {
        "run_name": config.get("run_id") or run_dir.name,
        "task": task,
        "status": status,
        "prediction_file": str(prediction_path),
        **metrics,
    }
    update_results(results_path, results, metrics, status)
    write_json(run_dir / "eval_summary.json", summary)
    return summary


def align_prediction_rows(prediction_path: Path, eval_file: str | None) -> list[dict[str, Any]]:
    predictions = load_jsonl(prediction_path)
    gold_rows = load_jsonl(eval_file) if eval_file and Path(eval_file).exists() else []
    gold_by_id = {row.get("id"): row for row in gold_rows if row.get("id")}
    rows = []
    for idx, pred in enumerate(predictions):
        gold_record = gold_by_id.get(pred.get("id")) or (gold_rows[idx] if idx < len(gold_rows) else {})
        rows.append(
            {
                "id": pred.get("id") or gold_record.get("id"),
                "prediction": first_present(pred, ["prediction", "generated", "output", "response", "text", "sql"]),
                "gold": first_present(pred, ["gold", "target", "answer", "gold_sql"])
                or gold_from_record(gold_record),
                "db_id": first_present(pred, ["db_id", "database_id"]) or gold_record.get("meta", {}).get("db_id"),
            }
        )
    return rows


def gold_from_record(record: dict[str, Any]) -> str | None:
    if not record:
        return None
    if "views" in record and "A" in record["views"]:
        return record["views"]["A"]
    if "views" in record and "SQL" in record["views"]:
        return record["views"]["SQL"]
    messages = record.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content")
    return None


def null_metrics(task: str) -> dict[str, Any]:
    base = {"accuracy": None, "exact_match": None, "exec_acc": None, "num_eval_examples": 0}
    if task == "gsm8k":
        base.update({"final_answer_exact_match": None, "numeric_exact_match": None})
    elif task == "spider":
        base.update({"sql_string_exact_match": None, "execution_accuracy": None})
    return base


def update_results(results_path: Path, results: dict[str, Any], metrics: dict[str, Any], eval_status: str) -> None:
    merged = {**results, **metrics, "eval_status": eval_status}
    write_json(results_path, merged)


def find_prediction_file(run_dir: Path) -> Path | None:
    for name in PREDICTION_FILENAMES:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def infer_task(train_file: str | None) -> str | None:
    text = (train_file or "").lower()
    for task in ["gsm8k", "spider", "hotpotqa", "hotpot"]:
        if task in text:
            return "hotpotqa" if task == "hotpot" else task
    return None


if __name__ == "__main__":
    main()
