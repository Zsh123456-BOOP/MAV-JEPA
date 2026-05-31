"""Task-specific builders for MAV-JEPA multi-view JSONL records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .spider_exec import execute_sql_with_timeout, find_spider_db, serialize_result, sqlite_schema_string
from .view_schema import remove_empty_views_and_edges, validate_mv_record


FINAL_MARKER = "####"
LAST_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
SPIDER_DB_RE = re.compile(r"db_id:\[([^\]]+)\]")


@dataclass
class BuildStats:
    kept: int = 0
    skipped: int = 0
    truncated: int = 0
    malformed: int = 0
    missing_answer: int = 0
    missing_db: int = 0
    sql_exec_ok: int = 0
    sql_exec_error: int = 0
    sql_exec_timeout: int = 0
    validation_errors: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kept": self.kept,
            "skipped": self.skipped,
            "truncated": self.truncated,
            "malformed": self.malformed,
            "missing_answer": self.missing_answer,
            "missing_db": self.missing_db,
            "sql_exec_ok": self.sql_exec_ok,
            "sql_exec_error": self.sql_exec_error,
            "sql_exec_timeout": self.sql_exec_timeout,
            "validation_errors": self.validation_errors,
        }


class BaseViewBuilder:
    task = "base"

    def __init__(self, source: str = "original", split: str = "train", tokenizer: Any | None = None):
        self.source = source
        self.split = split
        self.tokenizer = tokenizer
        self.stats = BuildStats()

    def build_record(self, raw: dict[str, Any], index: int) -> dict[str, Any] | None:
        raise NotImplementedError

    def build_records(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        records = []
        for index, raw in enumerate(rows):
            record = self.build_record(raw, index)
            if record is not None:
                records.append(record)
        return records

    def truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        self.stats.truncated += 1
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            decoded = self.tokenizer.decode(ids[:max_chars], skip_special_tokens=True)
            return decoded
        return text[: max_chars - 3] + "..."

    def _validate_or_skip(self, record: dict[str, Any]) -> dict[str, Any] | None:
        record = remove_empty_views_and_edges(record)
        ok, errors = validate_mv_record(record)
        if ok:
            self.stats.kept += 1
            return record
        self.stats.skipped += 1
        self.stats.malformed += 1
        for error in errors:
            self.stats.validation_errors[error] = self.stats.validation_errors.get(error, 0) + 1
        return None


class GSM8KViewBuilder(BaseViewBuilder):
    task = "gsm8k"

    def build_record(self, raw: dict[str, Any], index: int) -> dict[str, Any] | None:
        messages = normalize_messages(raw, task=self.task)
        if not messages:
            self.stats.skipped += 1
            self.stats.malformed += 1
            return None
        question = messages[1]["content"].strip()
        assistant = messages[2]["content"].strip()
        if not assistant:
            self.stats.skipped += 1
            self.stats.missing_answer += 1
            return None
        reasoning, answer, answer_quality = split_gsm8k_answer(assistant)
        edges = [
            {"src": "Q", "tgt": "R", "name": "Q_to_R", "quality": 1.0},
            {"src": "R", "tgt": "A", "name": "R_to_A", "quality": answer_quality},
            {"src": "Q", "tgt": "A", "name": "Q_to_A", "quality": 0.7 if answer_quality >= 1.0 else 0.3},
        ]
        record = {
            "id": f"gsm8k-{self.split}-{index:06d}",
            "task": self.task,
            "messages": messages,
            "views": {
                "Q": self.truncate(question, 2048),
                "R": self.truncate(reasoning, 4096),
                "A": self.truncate(answer, 512),
            },
            "edges": edges,
            "meta": {"source": self.source, "split": self.split, "original_index": index},
        }
        return self._validate_or_skip(record)


class SpiderViewBuilder(BaseViewBuilder):
    task = "spider"

    def __init__(
        self,
        source: str = "original",
        split: str = "train",
        spider_db_dir: str | Path | None = None,
        tokenizer: Any | None = None,
    ):
        super().__init__(source=source, split=split, tokenizer=tokenizer)
        self.spider_db_dir = Path(spider_db_dir) if spider_db_dir else None

    def build_record(self, raw: dict[str, Any], index: int) -> dict[str, Any] | None:
        messages = normalize_messages(raw, task=self.task)
        if not messages:
            self.stats.skipped += 1
            self.stats.malformed += 1
            return None
        user = messages[1]["content"].strip()
        sql = messages[2]["content"].strip()
        if not sql:
            self.stats.skipped += 1
            self.stats.missing_answer += 1
            return None
        db_id, question = parse_spider_user(user, raw)
        schema = f"Database: {db_id}\nSchema unavailable: SQLite database not found"
        result = ""
        db_path = find_spider_db(self.spider_db_dir, db_id)
        if db_path is None:
            self.stats.missing_db += 1
        else:
            try:
                schema = sqlite_schema_string(db_path, db_id)
            except Exception as exc:
                schema = f"Database: {db_id}\nSchema introspection failed: {exc!r}"
                self.stats.sql_exec_error += 1
            status, payload = execute_sql_with_timeout(db_path, sql)
            if status == "ok":
                result = serialize_result(payload)
                self.stats.sql_exec_ok += 1
            elif status == "timeout":
                self.stats.sql_exec_timeout += 1
            else:
                self.stats.sql_exec_error += 1

        views = {
            "Q": self.truncate(question, 2048),
            "S": self.truncate(schema, 4096),
            "QS": self.truncate(f"Question: {question}\n\n{schema}", 4096),
            "SQL": self.truncate(sql, 1024),
        }
        if result:
            views["RESULT"] = self.truncate(result, 512)

        edges = [
            {"src": "QS", "tgt": "SQL", "name": "QS_to_SQL", "quality": 1.0},
            {"src": "Q", "tgt": "SQL", "name": "Q_to_SQL", "quality": 0.7},
            {"src": "Q", "tgt": "S", "name": "Q_to_S", "quality": 0.4},
        ]
        if result:
            edges.append({"src": "SQL", "tgt": "RESULT", "name": "SQL_to_RESULT", "quality": 0.8})

        record = {
            "id": f"spider-{self.split}-{index:06d}",
            "task": self.task,
            "messages": messages,
            "views": views,
            "edges": edges,
            "meta": {
                "source": self.source,
                "split": self.split,
                "original_index": index,
                "db_id": db_id,
                "db_path": str(db_path) if db_path else None,
            },
        }
        return self._validate_or_skip(record)


class HotpotQAViewBuilder(BaseViewBuilder):
    task = "hotpotqa"

    def build_record(self, raw: dict[str, Any], index: int) -> dict[str, Any] | None:
        raise NotImplementedError("HotpotQA is intentionally deferred until GSM8K and Spider are stable.")


def split_gsm8k_answer(text: str) -> tuple[str, str, float]:
    if FINAL_MARKER in text:
        reasoning, answer = text.rsplit(FINAL_MARKER, 1)
        return reasoning.strip(), answer.strip(), 1.0
    matches = LAST_NUMBER_RE.findall(text)
    if matches:
        return text.strip(), matches[-1].replace("$", ""), 0.7
    return text.strip(), text.strip(), 0.3


def parse_spider_user(user: str, raw: dict[str, Any]) -> tuple[str, str]:
    db_id = raw.get("db_id") or raw.get("database_id")
    match = SPIDER_DB_RE.search(user)
    if match:
        db_id = match.group(1)
    db_id = str(db_id or "unknown")
    question = SPIDER_DB_RE.sub("", user)
    question = question.replace("For \n\n", "").strip()
    return db_id, question


def normalize_messages(raw: dict[str, Any], task: str) -> list[dict[str, str]] | None:
    messages = raw.get("messages")
    if isinstance(messages, list) and len(messages) >= 3:
        normalized = []
        for message in messages[:3]:
            if not isinstance(message, dict):
                return None
            normalized.append({"role": str(message.get("role", "")), "content": str(message.get("content", ""))})
        return normalized

    if task == "gsm8k":
        question = raw.get("question") or raw.get("problem")
        answer = raw.get("answer") or raw.get("solution")
        if question and answer:
            return [
                {"role": "system", "content": "Answer the math question, show steps."},
                {"role": "user", "content": str(question)},
                {"role": "assistant", "content": str(answer)},
            ]
    if task == "spider":
        question = raw.get("question")
        sql = raw.get("query") or raw.get("sql")
        db_id = raw.get("db_id") or raw.get("database_id")
        if question and sql:
            user = f"For db_id:[{db_id or 'unknown'}]\n\n{question}"
            return [
                {"role": "system", "content": "Convert natural language to SQL."},
                {"role": "user", "content": user},
                {"role": "assistant", "content": str(sql)},
            ]
    return None


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
