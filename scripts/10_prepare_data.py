#!/usr/bin/env python
"""Prepare normalized MAV-JEPA multi-view data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mavjepa.view_builders import GSM8KViewBuilder, SpiderViewBuilder, iter_jsonl
from mavjepa.view_schema import save_data_report


TASK_FILES = {
    "gsm8k": ("datasets/gsm8k_train.jsonl", "datasets/gsm8k_test.jsonl"),
    "spider": ("datasets/spider_train.jsonl", "datasets/spider_test.jsonl"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MAV-JEPA multi-view data.")
    parser.add_argument("--task", required=True, choices=["gsm8k", "spider"])
    parser.add_argument("--source", default="auto", choices=["auto", "original", "hf"])
    parser.add_argument("--input_train")
    parser.add_argument("--input_eval")
    parser.add_argument("--spider_db_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    train_path, eval_path, selected_source = resolve_inputs(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "gsm8k":
        train_builder = GSM8KViewBuilder(source=selected_source, split="train")
        eval_builder = GSM8KViewBuilder(source=selected_source, split="eval")
    else:
        train_builder = SpiderViewBuilder(source=selected_source, split="train", spider_db_dir=args.spider_db_dir)
        eval_builder = SpiderViewBuilder(source=selected_source, split="eval", spider_db_dir=args.spider_db_dir)

    train_records = train_builder.build_records(read_rows(train_path, args.task, "train", args.limit, selected_source))
    eval_records = eval_builder.build_records(read_rows(eval_path, args.task, "eval", args.limit, selected_source))

    write_jsonl(out_dir / "train.jsonl", train_records)
    write_jsonl(out_dir / "eval.jsonl", eval_records)
    write_jsonl(out_dir / "train_64.jsonl", train_records[:64])
    write_jsonl(out_dir / "eval_64.jsonl", eval_records[:64])

    report: dict[str, Any] = {
        "task": args.task,
        "source": selected_source,
        "input_train": str(train_path) if train_path else None,
        "input_eval": str(eval_path) if eval_path else None,
        "output_dir": str(out_dir),
        "train": train_builder.stats.as_dict(),
        "eval": eval_builder.stats.as_dict(),
        "created_files": [
            "train.jsonl",
            "eval.jsonl",
            "train_64.jsonl",
            "eval_64.jsonl",
            "data_report.json",
        ],
    }
    save_data_report(report, out_dir / "data_report.json")
    print(json.dumps(report, indent=2))


def resolve_inputs(args: argparse.Namespace) -> tuple[Path | None, Path | None, str]:
    if args.source in {"auto", "original"}:
        default_train, default_eval = TASK_FILES[args.task]
        train_path = Path(args.input_train or default_train)
        eval_path = Path(args.input_eval or default_eval)
        if train_path.exists() and eval_path.exists():
            return train_path, eval_path, "original"
        if args.source == "original":
            raise FileNotFoundError(f"Could not find original files: {train_path}, {eval_path}")
    if args.source in {"auto", "hf"}:
        return None, None, "hf"
    raise FileNotFoundError(f"No usable inputs found for {args.task}")


def read_rows(path: Path | None, task: str, split: str, limit: int, source: str) -> list[dict[str, Any]]:
    if source == "original":
        rows = list(iter_jsonl(path))
    else:
        rows = list(read_hf_rows(task, split))
    if limit > 0:
        return rows[:limit]
    return rows


def read_hf_rows(task: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        hf_split = "train" if split == "train" else "test"
        return [dict(row) for row in ds[hf_split]]
    if task == "spider":
        ds = load_dataset("xlangai/spider", "spider")
        hf_split = "train" if split == "train" else "validation"
        return [dict(row) for row in ds[hf_split]]
    raise ValueError(task)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
