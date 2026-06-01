from argparse import Namespace
import importlib.util
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
