#!/usr/bin/env python3
"""Validate complete, reproducible target-model coverage for K-SOVA v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def messages_hash(messages: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    requests = read_jsonl(args.requests)
    outputs = read_jsonl(args.outputs)
    request_by_id = {str(row["id"]): row for row in requests}
    counts = Counter(str(row.get("id")) for row in outputs)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request["id"])
        if counts[request_id] != 1:
            errors.append({"id": request_id, "code": "coverage", "count": counts[request_id]})
    for row in outputs:
        request_id = str(row.get("id") or "")
        request = request_by_id.get(request_id)
        if request is None:
            errors.append({"id": request_id, "code": "unknown_request"})
            continue
        if row.get("model") != args.expected_model:
            errors.append({"id": request_id, "code": "model_mismatch", "actual": row.get("model")})
        if row.get("messages_sha256") != messages_hash(request["messages"]):
            errors.append({"id": request_id, "code": "messages_hash_mismatch"})
        if row.get("metadata") != request.get("metadata"):
            errors.append({"id": request_id, "code": "metadata_mismatch"})
        if row.get("error"):
            errors.append({"id": request_id, "code": "runtime_error", "detail": row.get("error")})
        finish_reason = str(row.get("finish_reason") or "").lower()
        if finish_reason == "length":
            warnings.append({"id": request_id, "code": "truncated_at_shared_token_cap"})
        elif finish_reason != "stop":
            errors.append({"id": request_id, "code": "finish_reason", "actual": row.get("finish_reason")})
        if not str(row.get("response") or "").strip():
            errors.append({"id": request_id, "code": "empty_response"})

    report = {
        "schema_version": "ksova-v2-target-output-validation-1.0",
        "pass": not errors,
        "expected_model": args.expected_model,
        "requests": len(requests),
        "outputs": len(outputs),
        "errors": errors,
        "warnings": warnings,
        "truncated_outputs": len(warnings),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "pass": not errors,
        "outputs": len(outputs),
        "errors": len(errors),
        "warnings": len(warnings),
    }, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
