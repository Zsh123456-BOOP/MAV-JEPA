"""Validation helpers for MAV-JEPA multi-view records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {"id", "task", "messages", "views", "edges", "meta"}
REQUIRED_MESSAGE_KEYS = {"role", "content"}
REQUIRED_EDGE_KEYS = {"src", "tgt", "name"}


def validate_mv_record(record: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = REQUIRED_TOP_LEVEL - set(record)
    if missing:
        errors.append(f"missing top-level keys: {sorted(missing)}")

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
    else:
        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"messages[{idx}] must be an object")
                continue
            msg_missing = REQUIRED_MESSAGE_KEYS - set(message)
            if msg_missing:
                errors.append(f"messages[{idx}] missing keys: {sorted(msg_missing)}")
            if not str(message.get("content", "")).strip():
                errors.append(f"messages[{idx}] content is empty")

    views = record.get("views")
    if not isinstance(views, dict) or not views:
        errors.append("views must be a non-empty object")
        views = {}
    else:
        for name, value in views.items():
            if not isinstance(name, str) or not name:
                errors.append("view names must be non-empty strings")
            if not isinstance(value, str) or not value.strip():
                errors.append(f"view {name!r} is empty")

    edges = record.get("edges")
    if not isinstance(edges, list):
        errors.append("edges must be a list")
    else:
        for idx, edge in enumerate(edges):
            if not isinstance(edge, dict):
                errors.append(f"edges[{idx}] must be an object")
                continue
            edge_missing = REQUIRED_EDGE_KEYS - set(edge)
            if edge_missing:
                errors.append(f"edges[{idx}] missing keys: {sorted(edge_missing)}")
            src = edge.get("src")
            tgt = edge.get("tgt")
            if src not in views:
                errors.append(f"edges[{idx}] src {src!r} missing from views")
            if tgt not in views:
                errors.append(f"edges[{idx}] tgt {tgt!r} missing from views")
            quality = edge.get("quality", 1.0)
            try:
                quality_f = float(quality)
                if quality_f < 0:
                    errors.append(f"edges[{idx}] quality must be non-negative")
            except (TypeError, ValueError):
                errors.append(f"edges[{idx}] quality must be numeric")

    meta = record.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta must be an object")

    return not errors, errors


def save_data_report(stats: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remove_empty_views_and_edges(record: dict[str, Any]) -> dict[str, Any]:
    views = {k: v for k, v in record.get("views", {}).items() if isinstance(v, str) and v.strip()}
    edges = []
    for edge in record.get("edges", []):
        if edge.get("src") in views and edge.get("tgt") in views:
            edges.append(edge)
    record["views"] = views
    record["edges"] = edges
    return record
