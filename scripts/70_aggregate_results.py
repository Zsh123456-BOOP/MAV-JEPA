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
    write_ablation_outputs(Path(args.outputs_dir), rows)
    if args.make_plots:
        make_plots(Path(args.outputs_dir), rows)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def make_plots(outputs_dir: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = outputs_dir / "figures"
    analysis_dir = outputs_dir / "analysis"
    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = list(iter_run_dirs(outputs_dir))
    tasks = ["gsm8k", "spider"]
    for task in tasks:
        task_dirs = [path for path in run_dirs if run_task(path) == task]
        plot_loss_curves(plt, task, task_dirs, figures_dir / f"loss_curves_{task}.png")
        plot_edge_sampling(plt, task, task_dirs, figures_dir / f"edge_sampling_{task}.png")
        plot_lambda_dynamics(plt, task, task_dirs, figures_dir / f"lambda_dynamics_{task}.png")
    plot_compute_vs_score(plt, rows, figures_dir / "compute_vs_score.png")
    write_view_edge_table(analysis_dir / "view_edge_table.md", run_dirs)
    error_cases_path = analysis_dir / "error_cases.jsonl"
    if not error_cases_path.exists():
        error_cases_path.write_text("", encoding="utf-8")


def run_task(run_dir: Path) -> str | None:
    config = read_json(run_dir / "run_config.json")
    return config.get("task") or infer_task(config.get("train_file"))


def plot_loss_curves(plt: Any, task: str, run_dirs: list[Path], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for run_dir in run_dirs:
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        steps = [row.get("step") for row in metrics if row.get("step") is not None]
        if not steps:
            continue
        for key, style in [("ce_loss", "-"), ("jepa_loss", "--"), ("total_loss", ":")]:
            values = [row.get(key) for row in metrics if row.get("step") is not None]
            if any(value is not None for value in values):
                ax.plot(steps, values, style, linewidth=1.2, label=f"{run_dir.name}:{key}")
                plotted = True
    if not plotted:
        ax.text(0.5, 0.5, f"No step loss metrics for {task}", ha="center", va="center")
    ax.set_title(f"{task} loss curves")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=6, loc="best") if plotted else None
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_edge_sampling(plt: Any, task: str, run_dirs: list[Path], output_path: Path) -> None:
    counts: dict[str, int] = {}
    for run_dir in run_dirs:
        results = read_json(run_dir / "results.json")
        for edge, count in (results.get("edge_sampling_frequency") or {}).items():
            counts[edge] = counts.get(edge, 0) + int(count)
        for row in read_jsonl(run_dir / "metrics.jsonl"):
            for edge in row.get("active_edges") or []:
                counts[edge] = counts.get(edge, 0) + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    if counts:
        names = sorted(counts)
        ax.bar(names, [counts[name] for name in names], color="#2f6f6d")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(0.5, 0.5, f"No edge sampling metrics for {task}", ha="center", va="center")
    ax.set_title(f"{task} edge sampling")
    ax.set_ylabel("selected count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_lambda_dynamics(plt: Any, task: str, run_dirs: list[Path], output_path: Path) -> None:
    series: dict[str, list[tuple[int, float]]] = {}
    for run_dir in run_dirs:
        for row in read_jsonl(run_dir / "metrics.jsonl"):
            step = row.get("step")
            if step is None:
                continue
            for edge, value in (row.get("lambda_by_edge") or {}).items():
                if value is not None:
                    series.setdefault(edge, []).append((int(step), float(value)))
    fig, ax = plt.subplots(figsize=(7, 4))
    if series:
        for edge, points in sorted(series.items()):
            ax.plot([p[0] for p in points], [p[1] for p in points], label=edge)
        ax.legend(fontsize=7, loc="best")
    else:
        ax.text(0.5, 0.5, f"No lambda dynamics for {task}", ha="center", va="center")
    ax.set_title(f"{task} lambda dynamics")
    ax.set_xlabel("step")
    ax.set_ylabel("lambda")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_compute_vs_score(plt: Any, rows: list[dict[str, str]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for row in rows:
        flops = parse_float(row.get("flops"))
        score = parse_float(row.get("exact_match"))
        if score is None:
            score = parse_float(row.get("train_loss"))
        if flops is None or score is None:
            continue
        ax.scatter(flops, score, s=28)
        ax.annotate(f"{row.get('task')}:{row.get('method')}", (flops, score), fontsize=6)
        plotted = True
    if plotted:
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No paired compute/score metrics yet", ha="center", va="center")
    ax.set_title("Compute vs score/loss")
    ax.set_xlabel("estimated FLOPs")
    ax.set_ylabel("exact match if available, else train loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_view_edge_table(output_path: Path, run_dirs: list[Path]) -> None:
    lines = [
        "# View Edge Sampling Table",
        "",
        "| Task | Run | Edge | Count |",
        "|---|---|---|---:|",
    ]
    emitted = False
    for run_dir in run_dirs:
        task = run_task(run_dir) or "unknown"
        counts: dict[str, int] = {}
        results = read_json(run_dir / "results.json")
        for edge, count in (results.get("edge_sampling_frequency") or {}).items():
            counts[edge] = counts.get(edge, 0) + int(count)
        for row in read_jsonl(run_dir / "metrics.jsonl"):
            for edge in row.get("active_edges") or []:
                counts[edge] = counts.get(edge, 0) + 1
        for edge, count in sorted(counts.items()):
            lines.append(f"| {task} | {run_dir.name} | {edge} | {count} |")
            emitted = True
    if not emitted:
        lines.append("| null | null | null | 0 |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_float(value: str | None) -> float | None:
    if value in {None, "", "null"}:
        return None
    try:
        return float(value)
    except ValueError:
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


ABLATION_MAP = {
    "sft_lora": "A0",
    "original_llm_jepa_lora": "A1",
    "mv_jepa_fixed_lambda": "A2",
    "mv_jepa_adaptive_lambda": "A3",
    "mv_jepa_adaptive_edge_dropout": "A4",
    "mav_jepa_full": "A5",
}

ABLATION_LABELS = {
    "A0": "SFT + LoRA",
    "A1": "Original LLM-JEPA + LoRA",
    "A2": "MV-JEPA fixed lambda, no adaptive dropout",
    "A3": "MV-JEPA + adaptive lambda only",
    "A4": "MV-JEPA + adaptive edge dropout only",
    "A5": "MAV-JEPA full",
}


def write_ablation_outputs(outputs_dir: Path, rows: list[dict[str, str]]) -> None:
    aggregate_dir = outputs_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        ablation = ABLATION_MAP.get(row.get("method"))
        if not ablation or row.get("task") not in {"gsm8k", "spider"}:
            continue
        item = {
            "ablation": ablation,
            "description": ABLATION_LABELS[ablation],
            "task": row["task"],
            "method": row["method"],
            "run_name": row["run_name"],
            "exact_match": row["exact_match"],
            "exec_acc": row["exec_acc"],
            "train_loss": row["train_loss"],
            "jepa_loss": row["jepa_loss"],
            "flops": row["flops"],
            "wall_clock_sec": row["wall_clock_sec"],
            "lr": row.get("lr", "null"),
        }
        key = (row["task"], ablation)
        if key not in selected or ablation_prefer(item, selected[key]):
            selected[key] = item
    ablation_rows = list(selected.values())
    ablation_rows.sort(key=lambda item: (item["task"], item["ablation"], item["run_name"]))
    csv_path = aggregate_dir / "ablation.csv"
    columns = [
        "ablation",
        "description",
        "task",
        "method",
        "run_name",
        "exact_match",
        "exec_acc",
        "train_loss",
        "jepa_loss",
        "flops",
        "wall_clock_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in ablation_rows])
    (aggregate_dir / "ablation.md").write_text(build_ablation_report(ablation_rows, rows), encoding="utf-8")


def ablation_prefer(candidate: dict[str, str], current: dict[str, str]) -> bool:
    candidate_score = int(candidate.get("lr") != "null") + int(candidate.get("train_loss") != "null")
    current_score = int(current.get("lr") != "null") + int(current.get("train_loss") != "null")
    return candidate_score > current_score


def build_ablation_report(rows: list[dict[str, str]], all_rows: list[dict[str, str]]) -> str:
    lines = [
        "# MAV-JEPA Ablation Smoke Report",
        "",
        "This report is generated from available GSM8K and Spider runs. Quality metrics remain `null` until prediction files are generated; current comparisons use smoke-run training loss and compute fields only.",
        "",
        "## Runs",
        "",
        "| Task | Ablation | Method | Train loss | JEPA loss | FLOPs | Wall clock sec | Exact match | Exec acc |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['ablation']} | {row['method']} | {row['train_loss']} | {row['jepa_loss']} | {row['flops']} | {row['wall_clock_sec']} | {row['exact_match']} | {row['exec_acc']} |"
        )
    lines.extend(["", "## Required Questions", ""])
    lines.append(question_answer(rows, "Does multi-view help over original two-view JEPA?", "A1", "A2"))
    lines.append(question_answer(rows, "Does adaptive lambda reduce tuning sensitivity?", "A2", "A3"))
    lines.append(random_dropout_answer(rows, all_rows))
    lines.append(edge_answer(rows))
    lines.append("")
    return "\n".join(lines)


def question_answer(rows: list[dict[str, str]], question: str, left: str, right: str) -> str:
    comparisons = []
    for task in sorted({row["task"] for row in rows}):
        left_row = first_ablation(rows, task, left)
        right_row = first_ablation(rows, task, right)
        if not left_row or not right_row:
            comparisons.append(f"{task}: missing {left} or {right}")
            continue
        comparisons.append(
            f"{task}: {left} train_loss={left_row['train_loss']}, {right} train_loss={right_row['train_loss']}, quality metrics={right_row['exact_match']}"
        )
    return f"1. {question} {'; '.join(comparisons) if comparisons else 'No matching runs yet.'}"


def edge_answer(rows: list[dict[str, str]]) -> str:
    present = sorted({row["method"] for row in rows if row["ablation"] in {"A2", "A3", "A4", "A5"}})
    if not present:
        return "1. Which view edges are useful or harmful? No multi-view ablation runs are available yet."
    return (
        "1. Which view edges are useful or harmful? Current smoke runs exercise the configured GSM8K and Spider edges, "
        "but edge-removal ablations are optional and have not been run; use per-run `metrics.jsonl` edge frequencies for diagnostics. "
        f"Available multi-view methods: {', '.join(present)}."
    )


def random_dropout_answer(rows: list[dict[str, str]], all_rows: list[dict[str, str]]) -> str:
    comparisons = []
    for task in sorted({row["task"] for row in rows}):
        adaptive = first_ablation(rows, task, "A4")
        random_rows = [
            row
            for row in all_rows
            if row.get("task") == task and row.get("method") == "original_llm_jepa_random_dropout"
        ]
        random_rows.sort(key=lambda row: (row.get("lr") == "null", row.get("train_loss") == "null"))
        random_row = random_rows[0] if random_rows else None
        if not adaptive or not random_row:
            comparisons.append(f"{task}: missing A4 or random-dropout baseline")
            continue
        comparisons.append(
            f"{task}: random_dropout train_loss={random_row['train_loss']}, wall={random_row['wall_clock_sec']}; A4 train_loss={adaptive['train_loss']}, wall={adaptive['wall_clock_sec']}, quality metrics={adaptive['exact_match']}"
        )
    return (
        "1. Does adaptive edge dropout beat random dropout at similar compute? "
        f"{'; '.join(comparisons) if comparisons else 'No matching runs yet.'}"
    )


def first_ablation(rows: list[dict[str, str]], task: str, ablation: str) -> dict[str, str] | None:
    for row in rows:
        if row["task"] == task and row["ablation"] == ablation:
            return row
    return None


if __name__ == "__main__":
    main()
