import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "55_generate_predictions.py"
SPEC = importlib.util.spec_from_file_location("generate_predictions", SCRIPT_PATH)
generate_predictions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generate_predictions
SPEC.loader.exec_module(generate_predictions)


class DummyTokenizer:
    chat_template = None


def test_locate_model_artifact_prefers_checkpoint_final_adapter(tmp_path):
    run_dir = tmp_path / "run"
    adapter_dir = run_dir / "checkpoint-final"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    kind, path = generate_predictions.locate_model_artifact(run_dir)

    assert kind == "adapter"
    assert path == adapter_dir


def test_locate_model_artifact_prefers_final_full_model_over_step_adapter(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "model.safetensors").write_text("", encoding="utf-8")
    step_dir = run_dir / "checkpoint-16"
    step_dir.mkdir()
    (step_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    kind, path = generate_predictions.locate_model_artifact(run_dir)

    assert kind == "full"
    assert path == run_dir


def test_prompt_from_record_uses_system_and_user_only():
    record = {
        "messages": [
            {"role": "system", "content": "Solve."},
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "gold response"},
        ]
    }

    prompt = generate_predictions.prompt_from_record(record, DummyTokenizer())

    assert "system: Solve." in prompt
    assert "user: Q?" in prompt
    assert "gold response" not in prompt
