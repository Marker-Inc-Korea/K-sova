#!/usr/bin/env python3
"""Materialize one exact response per final teacher request from ordered inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def messages_sha256(messages: Any) -> str:
    normalized = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    return hashlib.sha256(json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--response-input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite response-set outputs")
    requests = read_jsonl(args.requests)
    request_ids = [str(row.get("id") or "") for row in requests]
    if any(not value for value in request_ids) or len(request_ids) != len(set(request_ids)):
        raise ValueError("request IDs must be nonempty and unique")
    expected_hashes = {
        str(row["id"]): messages_sha256(row.get("messages") or []) for row in requests
    }

    response_by_id: dict[str, dict[str, Any]] = {}
    input_row_counts: dict[str, int] = {}
    override_counts: Counter[str] = Counter()
    all_response_ids: set[str] = set()
    for path in args.response_input:
        rows = read_jsonl(path)
        ids = [str(row.get("id") or "") for row in rows]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{path}: response IDs must be nonempty and unique")
        input_row_counts[str(path)] = len(rows)
        for row in rows:
            response_id = str(row["id"])
            if response_id in response_by_id:
                override_counts[str(path)] += 1
            response_by_id[response_id] = row
            all_response_ids.add(response_id)

    missing = sorted(set(request_ids) - set(response_by_id))
    if missing:
        raise ValueError(f"final response set is missing requests: {missing[:5]}")
    materialized: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request["id"])
        response = response_by_id[request_id]
        if str(response.get("messages_sha256") or "") != expected_hashes[request_id]:
            raise ValueError(f"response/request message hash mismatch: {request_id}")
        materialized.append(response)
    unused = sorted(all_response_ids - set(request_ids))
    report = {
        "schema_version": "ksova-teacher-response-set-report-1.0",
        "requests": len(requests),
        "response_inputs": input_row_counts,
        "overrides_by_input": dict(sorted(override_counts.items())),
        "materialized_responses": len(materialized),
        "unused_response_ids": unused,
        "message_hash_mismatches": 0,
        "response_models": dict(sorted(Counter(
            str(row.get("model") or "") for row in materialized
        ).items())),
        "hashes": {
            "requests_sha256": sha256(args.requests),
            "response_inputs_sha256": {
                str(path): sha256(path) for path in args.response_input
            },
        },
    }
    write_jsonl(args.output, materialized)
    report["hashes"]["output_sha256"] = sha256(args.output)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
