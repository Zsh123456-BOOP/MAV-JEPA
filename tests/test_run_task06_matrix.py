from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_task06_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_task06_matrix", SCRIPT_PATH)
run_task06_matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = run_task06_matrix
SPEC.loader.exec_module(run_task06_matrix)


def test_master_port_defaults_to_base_plus_gpu_index():
    args = Namespace(master_port=None, master_port_base=29600, gpu_index="2")

    assert run_task06_matrix.resolve_master_port(args) == 29602


def test_master_port_override_wins():
    args = Namespace(master_port=29777, master_port_base=29600, gpu_index="2")

    assert run_task06_matrix.resolve_master_port(args) == 29777


def test_build_command_passes_master_port_to_torchrun(tmp_path):
    args = Namespace(
        master_port=None,
        master_port_base=29600,
        gpu_index="3",
        max_length=512,
        batch_size=1,
        grad_accum=4,
        epochs=1,
        lora=True,
        track_flop_original=False,
        model="Qwen/Qwen2.5-1.5B-Instruct",
        model_source="modelscope",
        view_max_length=256,
        edge_budget=1,
    )
    run = {
        "train_file": "data/mv/gsm8k/train.jsonl",
        "eval_file": "data/mv/gsm8k/eval.jsonl",
        "learning_rate": "2e-5",
        "lora_rank": 64,
        "seed": 0,
        "view_config": "configs/views/gsm8k_qra.yaml",
    }
    method = run_task06_matrix.BASELINE_METHODS["sft_lora"]

    command = run_task06_matrix.build_command(args, "torchrun", "model-path", tmp_path, run, method)

    assert command[:4] == ["torchrun", "--nproc_per_node=1", "--master_port", "29603"]


def test_revised_qr_method_uses_safe_jepa_controls(tmp_path):
    args = Namespace(
        master_port=None,
        master_port_base=29600,
        gpu_index="0",
        max_length=512,
        batch_size=1,
        grad_accum=4,
        epochs=2,
        lora=True,
        track_flop_original=False,
        model="Qwen/Qwen2.5-1.5B-Instruct",
        model_source="modelscope",
        view_max_length=256,
        edge_budget=1,
    )
    run = {
        "train_file": "data/mv/gsm8k/train.jsonl",
        "eval_file": "data/mv/gsm8k/eval.jsonl",
        "learning_rate": "2e-5",
        "lora_rank": 16,
        "seed": 0,
        "view_config": "configs/views/gsm8k_qra.yaml",
    }
    method = run_task06_matrix.MAV_METHODS["mav_qr_stopgrad_p25_l005"]

    command = run_task06_matrix.build_command(args, "torchrun", "model-path", tmp_path, run, method)

    assert "finetune_mv.py" in command
    assert command[command.index("--allowed_edges") + 1] == "Q_to_R"
    assert command[command.index("--jepa_step_prob") + 1] == "0.25"
    assert command[command.index("--jepa_reduce") + 1] == "mean"
    assert command[command.index("--jepa_ce_ratio_cap") + 1] == "0.05"


def test_qra_diagnostic_methods_encode_expected_edge_scope(tmp_path):
    args = Namespace(
        master_port=None,
        master_port_base=29600,
        gpu_index="0",
        max_length=512,
        batch_size=1,
        grad_accum=4,
        epochs=2,
        lora=True,
        track_flop_original=False,
        model="Qwen/Qwen2.5-1.5B-Instruct",
        model_source="modelscope",
        view_max_length=256,
        edge_budget=1,
    )
    run = {
        "train_file": "data/mv/gsm8k/train.jsonl",
        "eval_file": "data/mv/gsm8k/eval.jsonl",
        "learning_rate": "2e-5",
        "lora_rank": 16,
        "seed": 0,
        "view_config": "configs/views/gsm8k_qra.yaml",
    }

    low_freq = run_task06_matrix.build_command(
        args,
        "torchrun",
        "model-path",
        tmp_path,
        run,
        run_task06_matrix.MAV_METHODS["mav_qr_stopgrad_p125_l005"],
    )
    all_edge = run_task06_matrix.build_command(
        args,
        "torchrun",
        "model-path",
        tmp_path,
        run,
        run_task06_matrix.MAV_METHODS["mav_qra_safe_all_p25_l005"],
    )
    qa_only = run_task06_matrix.build_command(
        args,
        "torchrun",
        "model-path",
        tmp_path,
        run,
        run_task06_matrix.MAV_METHODS["mav_qa_only_p25_l005"],
    )

    assert low_freq[low_freq.index("--jepa_step_prob") + 1] == "0.125"
    assert "--allowed_edges" not in all_edge
    assert qa_only[qa_only.index("--allowed_edges") + 1] == "Q_to_A"
    assert qa_only[qa_only.index("--min_target_tokens") + 1] == "0"


def test_finalize_run_uses_outer_wall_clock_for_cost(tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "train.log").write_text("", encoding="utf-8")
    (out_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({"status": "success", "wall_clock_sec": 12.5, "gpu_hours": 12.5 / 3600}),
        encoding="utf-8",
    )
    config = {"run_id": "r", "task": "gsm8k", "method": "m", "learning_rate": 2e-5}

    run_task06_matrix.finalize_run(out_dir, config, exit_code=0, wall=45.0, peak_vram_gb=3.2)

    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert results["wall_clock_sec"] == 45.0
    assert results["gpu_hours"] == 45.0 / 3600
    assert results["train_wall_clock_sec"] == 12.5
    assert results["train_gpu_hours"] == 12.5 / 3600
    run_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["wall_clock_sec"] == 45.0
    assert run_config["train_wall_clock_sec"] == 12.5
    assert run_config["resource_guard_reason"] is None
