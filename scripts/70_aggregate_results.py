#!/usr/bin/env python
"""Aggregate MAV-JEPA run artifacts into Task 06/07 CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RESULT_COLUMNS = [
    "run_name",
    "task",
    "model",
    "method",
    "seed",
    "lora_rank",
    "lr",
    "accuracy",
    "exact_match",
    "exec_acc",
    "train_loss",
    "jepa_loss",
    "flops",
    "wall_clock_sec",
    "trainable_params",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MAV-JEPA experiment results.")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--output_csv", default="outputs/aggregate/results.csv")
    parser.add_argument("--make_plots", action="store_true", help="Reserved for Task 09.")
    args = parser.parse_args()

    rows = [row_for_run(path) for path in iter_run_dirs(Path(args.outputs_dir))]
    rows = [row for row in rows if row is not None]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")


def iter_run_dirs(outputs_dir: Path):
    for parent_name in ["runs", "smoke"]:
        parent = outputs_dir / parent_name
        if not parent.exists():
            continue
        for path in sorted(parent.iterdir()):
            if path.is_dir() and (path / "run_config.json").exists():
                yield path


def row_for_run(path: Path) -> dict[str, str] | None:
    config = read_json(path / "run_config.json")
    if not config:
        return None
    results = read_json(path / "results.json")
    row = {
        "run_name": config.get("run_id") or path.name,
        "task": config.get("task") or infer_task(config.get("train_file")),
        "model": config.get("model") or config.get("requested_model") or config.get("model_name_or_path"),
        "method": config.get("method"),
        "seed": config.get("seed"),
        "lora_rank": config.get("lora_rank"),
        "lr": config.get("learning_rate") or config.get("lr"),
        "accuracy": results.get("accuracy"),
        "exact_match": first_present(results, ["exact_match", "final_answer_exact_match", "sql_string_exact_match"]),
        "exec_acc": first_present(results, ["exec_acc", "execution_accuracy"]),
        "train_loss": results.get("train_loss"),
        "jepa_loss": results.get("jepa_loss"),
        "flops": results.get("estimated_total_flops") or config.get("estimated_total_flops"),
        "wall_clock_sec": results.get("wall_clock_sec") or config.get("wall_clock_sec"),
        "trainable_params": results.get("trainable_params") or config.get("trainable_params"),
    }
    return {key: csv_value(row.get(key)) for key in RESULT_COLUMNS}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def infer_task(train_file: str | None) -> str | None:
    if not train_file:
        return None
    lowered = train_file.lower()
    for task in ["gsm8k", "spider", "hotpotqa", "hotpot"]:
        if task in lowered:
            return "hotpotqa" if task == "hotpot" else task
    return None


def csv_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


if __name__ == "__main__":
    main()
