#!/usr/bin/env python
"""Resolve model IDs to local paths, preferring ModelScope for server runs."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path


DEFAULT_FALLBACK = "Qwen/Qwen2.5-1.5B-Instruct"


def is_gated_family(model_id: str) -> bool:
    lower = model_id.lower()
    return lower.startswith("meta-llama/") or lower.startswith("google/gemma")


def resolve_with_modelscope(model_id: str, cache_dir: str | None) -> str:
    from modelscope import snapshot_download

    kwargs = {"model_id": model_id}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    with contextlib.redirect_stdout(sys.stderr):
        return snapshot_download(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a model for MAV-JEPA runs.")
    parser.add_argument("--model", required=True, help="Requested model ID or local path.")
    parser.add_argument("--source", default="modelscope", choices=["modelscope", "local", "auto"])
    parser.add_argument("--fallback", default=DEFAULT_FALLBACK)
    parser.add_argument("--allow_fallback", action="store_true")
    parser.add_argument("--cache_dir", default=os.environ.get("MODELSCOPE_CACHE"))
    parser.add_argument("--output_json", help="Optional path for resolution metadata.")
    args = parser.parse_args()

    requested = args.model
    meta = {
        "requested_model": requested,
        "effective_model": requested,
        "model_name_or_path": requested,
        "model_source": args.source,
        "model_fallback": None,
        "fallback_reason": None,
    }

    if Path(requested).exists():
        meta["model_name_or_path"] = str(Path(requested).resolve())
        meta["model_source"] = "local"
    else:
        model_to_download = requested
        if args.allow_fallback and is_gated_family(requested):
            model_to_download = args.fallback
            meta["effective_model"] = model_to_download
            meta["model_fallback"] = model_to_download
            meta["fallback_reason"] = "requested model is usually gated; using open Qwen fallback"

        if args.source in {"modelscope", "auto"}:
            try:
                local_path = resolve_with_modelscope(model_to_download, args.cache_dir)
                meta["model_name_or_path"] = str(Path(local_path).resolve())
                meta["model_source"] = "modelscope"
            except Exception as exc:
                if not args.allow_fallback or model_to_download == args.fallback:
                    meta["fallback_reason"] = f"modelscope resolution failed: {exc!r}"
                    if args.output_json:
                        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
                        Path(args.output_json).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
                    raise
                local_path = resolve_with_modelscope(args.fallback, args.cache_dir)
                meta["effective_model"] = args.fallback
                meta["model_name_or_path"] = str(Path(local_path).resolve())
                meta["model_source"] = "modelscope"
                meta["model_fallback"] = args.fallback
                meta["fallback_reason"] = f"modelscope resolution failed for requested model: {exc!r}"
        elif args.source == "local":
            meta["model_source"] = "local"

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(meta["model_name_or_path"])


if __name__ == "__main__":
    main()
