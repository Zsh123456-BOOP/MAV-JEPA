import json
from pathlib import Path

from mavjepa.view_builders import (
    GSM8KViewBuilder,
    SpiderViewBuilder,
    last_k_reasoning_sentences,
    mask_final_answer,
    split_gsm8k_answer,
    split_rationale_span,
)
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
    assert {"Q_to_R", "R_to_A", "Q_to_A", "QR_to_A_STMT"}.issubset(
        {edge["name"] for edge in record["edges"]}
    )
    assert {"R_FULL", "R_STRIP", "R_MASKANS", "R_LAST", "R_LAST_MASK", "QR_MASKANS"}.issubset(record["views"])


def test_gsm8k_builder_adds_rationale_span_views_for_long_reasoning():
    reasoning = " ".join(f"step{i}" for i in range(80))
    raw = {
        "messages": [
            {"role": "system", "content": "Answer the math question, show steps."},
            {"role": "user", "content": "What is the total?"},
            {"role": "assistant", "content": f"{reasoning}\n#### 42"},
        ]
    }
    builder = GSM8KViewBuilder(split="train")
    record = builder.build_record(raw, 0)

    assert {"R_PRE", "R_SUF", "QR_PRE", "QR", "A_STMT"}.issubset(record["views"])
    assert "QRPRE_to_RSUF" in {edge["name"] for edge in record["edges"]}


def test_rationale_span_split_requires_usable_suffix():
    prefix, suffix = split_rationale_span("short reasoning", min_suffix_tokens=16)

    assert prefix == ""
    assert suffix == ""


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


def test_mask_final_answer_preserves_reasoning_structure():
    text = "Then 20 + 22 = 42. Therefore, the answer is 42."

    masked, stats = mask_final_answer(text, "42")

    assert "Therefore" in masked
    assert "<ANS>" in masked
    assert "42" not in masked.split("Therefore")[-1]
    assert stats["mask_replacements"] >= 1


def test_r_last_mask_min_tokens():
    reasoning = "We first compute the subtotal carefully. Then add the remaining values to get 42. Therefore, the answer is 42."
    raw = {
        "messages": [
            {"role": "system", "content": "Answer the math question, show steps."},
            {"role": "user", "content": "What is the total?"},
            {"role": "assistant", "content": f"{reasoning}\n#### 42"},
        ]
    }
    builder = GSM8KViewBuilder(split="train")
    record = builder.build_record(raw, 0)

    assert "Q_to_R_LAST_MASK" in {edge["name"] for edge in record["edges"]}
    assert len(record["views"]["R_LAST_MASK"].split()) >= 8


def test_last_k_reasoning_sentences_returns_tail():
    assert last_k_reasoning_sentences("A first. B second. C third.", k=2) == "B second. C third."
