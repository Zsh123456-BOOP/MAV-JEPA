import json
from pathlib import Path

from mavjepa.view_builders import GSM8KViewBuilder, SpiderViewBuilder, split_gsm8k_answer
from mavjepa.view_schema import validate_mv_record


def test_gsm8k_splits_marker_answer():
    reasoning, answer, quality = split_gsm8k_answer("one\n#### 42")
    assert reasoning == "one"
    assert answer == "42"
    assert quality == 1.0


def test_gsm8k_fallback_last_number():
    reasoning, answer, quality = split_gsm8k_answer("No marker, final is $1,234.")
    assert reasoning.startswith("No marker")
    assert answer == "1,234"
    assert quality == 0.7


def test_gsm8k_builder_record_valid():
    raw = {
        "messages": [
            {"role": "system", "content": "Answer the math question, show steps."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2=4\n#### 4"},
        ]
    }
    builder = GSM8KViewBuilder(split="train")
    record = builder.build_record(raw, 0)
    ok, errors = validate_mv_record(record)
    assert ok, errors
    assert record["views"]["Q"] == "What is 2+2?"
    assert record["views"]["A"] == "4"
    assert {edge["name"] for edge in record["edges"]} == {"Q_to_R", "R_to_A", "Q_to_A"}


def test_spider_missing_db_skips_result_edge(tmp_path: Path):
    raw = {
        "messages": [
            {"role": "system", "content": "Convert natural language to SQL."},
            {"role": "user", "content": "For db_id:[store_1]\n\nWhat are the names?"},
            {"role": "assistant", "content": "SELECT name FROM playlists;"},
        ]
    }
    builder = SpiderViewBuilder(split="train", spider_db_dir=tmp_path / "missing")
    record = builder.build_record(raw, 1)
    ok, errors = validate_mv_record(record)
    assert ok, errors
    assert "RESULT" not in record["views"]
    assert "SQL_to_RESULT" not in {edge["name"] for edge in record["edges"]}
    assert builder.stats.missing_db == 1


def test_prepare_shape_json_serializable():
    raw = {
        "question": "What is 3+5?",
        "answer": "3+5=8\n#### 8",
    }
    builder = GSM8KViewBuilder(source="hf", split="train")
    record = builder.build_record(raw, 3)
    json.dumps(record)
    ok, errors = validate_mv_record(record)
    assert ok, errors
