"""SQLite helpers for Spider view construction and evaluation."""

from __future__ import annotations

import concurrent.futures
import sqlite3
from pathlib import Path
from typing import Any


def is_read_only_sql(sql: str) -> bool:
    stripped = sql.strip().lower()
    return stripped.startswith("select") or stripped.startswith("with")


def find_spider_db(spider_db_dir: str | Path | None, db_id: str) -> Path | None:
    if not spider_db_dir or not db_id:
        return None
    root = Path(spider_db_dir)
    candidates = [
        root / "database" / db_id / f"{db_id}.sqlite",
        root / db_id / f"{db_id}.sqlite",
        root / f"{db_id}.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def sqlite_schema_string(db_path: Path, db_id: str) -> str:
    lines = [f"Database: {db_id}"]
    with sqlite3.connect(str(db_path), timeout=5.0) as conn:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (table_name,) in table_rows:
            cols = conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
            col_names = [str(row[1]) for row in cols]
            lines.append(f"Table {table_name}: {', '.join(col_names)}")
        fk_parts = []
        for (table_name,) in table_rows:
            for fk in conn.execute(f"PRAGMA foreign_key_list({quote_identifier(table_name)})").fetchall():
                fk_parts.append(f"{table_name}.{fk[3]} -> {fk[2]}.{fk[4]}")
        if fk_parts:
            lines.append("Foreign keys: " + "; ".join(fk_parts))
    return "\n".join(lines)


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def execute_sql_with_timeout(db_path: Path, sql: str, timeout_sec: float = 5.0) -> tuple[str, Any]:
    if not is_read_only_sql(sql):
        return "blocked", "Only SELECT/WITH statements are allowed"

    def _run() -> list[tuple[Any, ...]]:
        with sqlite3.connect(str(db_path), timeout=timeout_sec) as conn:
            cursor = conn.execute(sql)
            return cursor.fetchmany(4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            return "ok", future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            return "timeout", None
        except Exception as exc:  # sqlite errors are counted, not fatal.
            return "error", repr(exc)


def serialize_result(rows: list[tuple[Any, ...]], max_rows: int = 3, max_chars: int = 512) -> str:
    sample = rows[:max_rows]
    if sample:
        columns = len(sample[0])
    else:
        columns = 0
    text = f"Rows: {len(rows)}; Columns: {columns}; Sample: {sample}"
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text
