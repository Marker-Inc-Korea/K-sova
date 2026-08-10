#!/usr/bin/env python3
"""Plan one transport-only retry without changing frozen validation messages."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_response(row: dict[str, Any]) -> bool:
    return all((
        not row.get("error"),
        str(row.get("finish_reason") or "").casefold() == "stop",
        bool(str(row.get("response") or "").strip()),
    ))


def token_count(
    row: dict[str, Any], base_url: str, model: str,
    template_kwargs: dict[str, Any] | None, timeout: int,
) -> int:
    payload: dict[str, Any] = {
        "model": model,
        "messages": row["messages"],
        "add_special_tokens": True,
    }
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    request = urllib.request.Request(
        base_url.rstrip("/").removesuffix("/v1") + "/tokenize",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    count = result.get("count")
    if not isinstance(count, int) or count < 1:
        raise ValueError(f"invalid tokenize response for {row.get('id')}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--safety-margin", type=int, default=16)
    parser.add_argument("--budget-tiers", default="2048,1792,1536,1280,1024,768,512")
    parser.add_argument("--chat-template-kwargs-json")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    requests = read_jsonl(args.requests)
    request_by_id = {str(row["id"]): row for row in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("duplicate request id")
    responses = read_jsonl(args.responses)
    counts = Counter(str(row.get("id") or "") for row in responses)
    response_by_id = {str(row.get("id") or ""): row for row in responses}
    unknown = set(response_by_id) - set(request_by_id)
    if unknown:
        raise ValueError(f"unknown response ids: {sorted(unknown)[:3]}")
    invalid = [
        request_by_id[request_id] for request_id in request_by_id
        if counts[request_id] != 1
        or not valid_response(response_by_id.get(request_id, {}))
    ]
    template_kwargs = (
        json.loads(args.chat_template_kwargs_json)
        if args.chat_template_kwargs_json else None
    )
    tiers = sorted(
        {int(value) for value in args.budget_tiers.split(",") if value.strip()},
        reverse=True,
    )
    if not tiers or min(tiers) < 1:
        raise ValueError("budget tiers must contain positive integers")

    token_counts: dict[str, int] = {}
    tokenize_errors: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_by_id = {
            pool.submit(
                token_count, row, args.base_url, args.model,
                template_kwargs, args.timeout,
            ): str(row["id"])
            for row in invalid
        }
        for future in concurrent.futures.as_completed(future_by_id):
            request_id = future_by_id[future]
            try:
                token_counts[request_id] = future.result()
            except Exception as exc:  # recorded as non-retryable, never silently dropped
                tokenize_errors[request_id] = f"{type(exc).__name__}: {exc}"

    retry_rows: list[dict[str, Any]] = []
    unserviceable: list[dict[str, Any]] = []
    budgets: Counter[int] = Counter()
    for row in invalid:
        request_id = str(row["id"])
        count = token_counts.get(request_id)
        if count is None:
            unserviceable.append({
                "id": request_id,
                "reason": "tokenize_error",
                "detail": tokenize_errors.get(request_id),
            })
            continue
        available = args.context_length - count - args.safety_margin
        budget = next((tier for tier in tiers if tier <= available), None)
        if budget is None:
            unserviceable.append({
                "id": request_id,
                "reason": "insufficient_context_for_minimum_retry_budget",
                "prompt_tokens": count,
                "available_output_tokens": available,
            })
            continue
        retry_rows.append({**row, "max_tokens": budget})
        budgets[budget] += 1

    write_jsonl(args.output, retry_rows)
    report = {
        "schema_version": "ksova-context-bounded-transport-retry-plan-1.0",
        "policy": "only_requests_without_a_valid_first_response; frozen_messages_unchanged",
        "requests": len(requests),
        "valid_first_responses": len(requests) - len(invalid),
        "invalid_first_responses": len(invalid),
        "planned_retries": len(retry_rows),
        "unserviceable": len(unserviceable),
        "budget_counts": {str(key): value for key, value in sorted(budgets.items(), reverse=True)},
        "context_length": args.context_length,
        "safety_margin": args.safety_margin,
        "unserviceable_details": unserviceable,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    write_json(args.report, report)
    print(json.dumps({key: report[key] for key in (
        "requests", "valid_first_responses", "invalid_first_responses",
        "planned_retries", "unserviceable", "budget_counts",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
