#!/usr/bin/env python3
"""Merge retry outputs by request ID, preferring the latest successful response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_jsonl, write_json, write_jsonl


def response_failure_reason(row: dict) -> str | None:
    if row.get("error"):
        return str(row["error"])
    if not str(row.get("response") or "").strip():
        return "empty_response"
    return None


def response_quality_flags(row: dict) -> list[str]:
    finish_reason = str(row.get("finish_reason") or "").lower()
    usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    completion_tokens = usage.get("completion_tokens")
    requested_max_tokens = row.get("requested_max_tokens")
    inferred_truncation = (
        not finish_reason
        and type(completion_tokens) is int
        and type(requested_max_tokens) is int
        and completion_tokens >= requested_max_tokens
    )
    if finish_reason == "length" or inferred_truncation:
        return ["truncated_generation"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    request_rows = load_jsonl(args.requests)
    request_ids = [row["id"] for row in request_rows]
    expected = set(request_ids)
    successful: dict[str, dict] = {}
    invalid_rows = []
    quality_flags_by_id: dict[str, list[dict]] = {}
    for path in args.responses:
        for row in load_jsonl(path):
            if row.get("model") != args.model:
                raise ValueError(f"unexpected model in {path}: {row.get('model')}")
            failure = response_failure_reason(row)
            if failure:
                invalid_rows.append({"id": row.get("id"), "path": str(path), "error": failure})
                continue
            row_id = str(row["id"])
            successful[row_id] = row
            flags = [
                {"id": row_id, "path": str(path), "flag": flag}
                for flag in response_quality_flags(row)
            ]
            if flags:
                quality_flags_by_id[row_id] = flags
            else:
                quality_flags_by_id.pop(row_id, None)
    missing = [request_id for request_id in request_ids if request_id not in successful]
    unexpected = sorted(set(successful) - expected)
    rows = [successful[request_id] for request_id in request_ids if request_id in successful]
    quality_flags = [
        flag
        for row_id in sorted(quality_flags_by_id)
        for flag in quality_flags_by_id[row_id]
    ]
    write_jsonl(args.output, rows)
    report = {
        "model": args.model,
        "input_files": [str(path) for path in args.responses],
        "invalid_rows": invalid_rows,
        "quality_flags": quality_flags,
        "successful": len(rows),
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps({
        "model": args.model,
        "successful": len(rows),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "quality_flags": len(quality_flags),
    }, ensure_ascii=False))
    return 0 if not missing and not unexpected else 1


if __name__ == "__main__":
    raise SystemExit(main())
