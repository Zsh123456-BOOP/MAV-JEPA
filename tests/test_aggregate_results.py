import csv
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "70_aggregate_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_results", SCRIPT_PATH)
aggregate_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = aggregate_results
SPEC.loader.exec_module(aggregate_results)


def write_run(
    root: Path,
    name: str,
    method: str,
    seed: int,
    accuracy: float,
    wall: float,
    flops: int,
    generation_num_examples: int | None = None,
) -> None:
    run = root / "runs" / name
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "task": "gsm8k",
                "method": method,
                "seed": seed,
                "train_wall_clock_sec": wall,
            }
        ),
        encoding="utf-8",
    )
    (run / "results.json").write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "exact_match": accuracy - 0.01,
                "estimated_total_flops": flops,
                "generation_num_examples": generation_num_examples,
            }
        ),
        encoding="utf-8",
    )


def test_summary_by_method_outputs_mean_and_std(tmp_path: Path):
    write_run(tmp_path, "run0", "mv_sft_lora", 0, 0.5, 10.0, 100)
    write_run(tmp_path, "run1", "mv_sft_lora", 1, 0.4, 20.0, 300)
    rows = [aggregate_results.row_for_run(path) for path in aggregate_results.iter_run_dirs(tmp_path)]

    aggregate_results.write_summary_by_method(tmp_path, rows)

    summary = list(csv.DictReader((tmp_path / "aggregate" / "summary_by_method.csv").open()))
    assert summary[0]["method"] == "mv_sft_lora"
    assert float(summary[0]["accuracy_mean"]) == 0.45
    assert float(summary[0]["train_wall_clock_mean"]) == 15.0
    assert float(summary[0]["flops_mean"]) == 200.0
    assert float(summary[0]["accuracy_std"]) > 0


def test_summary_by_method_separates_eval_size(tmp_path: Path):
    write_run(tmp_path, "run256", "mav_r3", 0, 0.5, 10.0, 100, generation_num_examples=256)
    write_run(tmp_path, "runfull", "mav_r3", 0, 0.4, 20.0, 300, generation_num_examples=1319)
    rows = [aggregate_results.row_for_run(path) for path in aggregate_results.iter_run_dirs(tmp_path)]

    aggregate_results.write_summary_by_method(tmp_path, rows)

    summary = list(csv.DictReader((tmp_path / "aggregate" / "summary_by_method.csv").open()))
    assert {row["eval_examples"] for row in summary} == {"256", "1319"}
    assert len(summary) == 2
