"""Evaluation metrics for MAV-JEPA tasks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .spider_exec import execute_sql_with_timeout, find_spider_db


NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = str(text).strip().replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(".")
    match = NUMBER_RE.search(cleaned)
    if not match:
        return None
    value = match.group(0).replace(",", "").replace("$", "").rstrip(".")
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, "g")


def extract_final_answer(text: str | None) -> str:
    if text is None:
        return ""
    answer = str(text).strip()
    if "####" in answer:
        answer = answer.rsplit("####", 1)[-1].strip()
    else:
        stripped = re.sub(r"^answer\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
        if stripped != answer:
            answer = stripped
        else:
            matches = NUMBER_RE.findall(answer)
            if matches:
                answer = matches[-1]
    return answer.rstrip(".").strip()


def gsm8k_scores(prediction: str | None, gold: str | None) -> dict[str, bool]:
    pred_answer = extract_final_answer(prediction)
    gold_answer = extract_final_answer(gold)
    pred_number = normalize_number(pred_answer)
    gold_number = normalize_number(gold_answer)
    return {
        "final_answer_exact_match": pred_answer == gold_answer and bool(gold_answer),
        "numeric_exact_match": pred_number is not None and pred_number == gold_number,
    }


def strip_sql_markdown(text: str | None) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    match = CODE_FENCE_RE.search(value)
    if match:
        value = match.group(1).strip()
    return value


def normalize_sql(sql: str | None) -> str:
    value = strip_sql_markdown(sql)
    value = re.sub(r"\s+", " ", value).strip().rstrip(";")
    return value.lower()


def spider_scores(
    prediction: str | None,
    gold_sql: str | None,
    db_id: str | None = None,
    spider_db_dir: str | Path | None = None,
) -> dict[str, bool | None]:
    pred_sql = strip_sql_markdown(prediction)
    gold = strip_sql_markdown(gold_sql)
    sql_string_exact = normalize_sql(pred_sql) == normalize_sql(gold) and bool(normalize_sql(gold))
    exec_acc: bool | None = None
    db_path = find_spider_db(spider_db_dir, db_id or "")
    if db_path is not None:
        try:
            pred_status, pred_rows = execute_sql_with_timeout(db_path, pred_sql)
            gold_status, gold_rows = execute_sql_with_timeout(db_path, gold)
            exec_acc = pred_status == "ok" and gold_status == "ok" and pred_rows == gold_rows
        except Exception:
            exec_acc = False
    return {"sql_string_exact_match": sql_string_exact, "execution_accuracy": exec_acc}


def mean_bool(values: list[bool | None]) -> float | None:
    scored = [value for value in values if value is not None]
    if not scored:
        return None
    return sum(1 for value in scored if value) / len(scored)


def evaluate_gsm8k_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    final_scores = []
    numeric_scores = []
    for row in rows:
        scores = gsm8k_scores(row.get("prediction"), row.get("gold"))
        final_scores.append(scores["final_answer_exact_match"])
        numeric_scores.append(scores["numeric_exact_match"])
    return {
        "num_eval_examples": len(rows),
        "final_answer_exact_match": mean_bool(final_scores),
        "numeric_exact_match": mean_bool(numeric_scores),
    }


def evaluate_spider_rows(rows: list[dict[str, Any]], spider_db_dir: str | Path | None = None) -> dict[str, float | int | None]:
    exact_scores = []
    exec_scores = []
    for row in rows:
        scores = spider_scores(row.get("prediction"), row.get("gold"), row.get("db_id"), spider_db_dir)
        exact_scores.append(scores["sql_string_exact_match"])
        exec_scores.append(scores["execution_accuracy"])
    return {
        "num_eval_examples": len(rows),
        "sql_string_exact_match": mean_bool(exact_scores),
        "execution_accuracy": mean_bool(exec_scores),
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
