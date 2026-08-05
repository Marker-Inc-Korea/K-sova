#!/usr/bin/env python3
"""Merge response shards in canonical request order with strict coverage checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    requests = read_jsonl(args.requests)
    request_by_id = {str(row.get("id") or ""): row for row in requests}
    if "" in request_by_id or len(request_by_id) != len(requests):
        raise ValueError("canonical requests must have unique nonempty ids")

    responses: list[dict[str, Any]] = []
    input_counts: dict[str, int] = {}
    for path in args.input:
        rows = read_jsonl(path)
        input_counts[str(path)] = len(rows)
        responses.extend(rows)

    response_counts = Counter(str(row.get("id") or "") for row in responses)
    missing = sorted(set(request_by_id) - set(response_counts))
    unexpected = sorted(set(response_counts) - set(request_by_id))
    duplicates = sorted(request_id for request_id, count in response_counts.items() if count != 1)
    metadata_mismatches: list[str] = []
    response_by_id: dict[str, dict[str, Any]] = {}
    for row in responses:
        request_id = str(row.get("id") or "")
        if request_id in request_by_id and response_counts[request_id] == 1:
            if row.get("metadata", {}) != request_by_id[request_id].get("metadata", {}):
                metadata_mismatches.append(request_id)
            response_by_id[request_id] = row

    report = {
        "schema_version": "ksova-jsonl-response-merge-1.0",
        "requests": len(requests),
        "responses": len(responses),
        "input_counts": input_counts,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "duplicate_count": len(duplicates),
        "metadata_mismatch_count": len(metadata_mismatches),
        "missing": missing[:50],
        "unexpected": unexpected[:50],
        "duplicates": duplicates[:50],
        "metadata_mismatches": metadata_mismatches[:50],
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if missing or unexpected or duplicates or metadata_mismatches:
        print(json.dumps(report, ensure_ascii=False))
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(response_by_id[str(request["id"])], ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
