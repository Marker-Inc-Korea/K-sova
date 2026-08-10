#!/usr/bin/env python3
"""Tokenize chat requests and reserve only output tokens that fit the model context."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def tokenize_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/tokenize", "", ""))


def request_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in row:
        messages = row["messages"]
    else:
        messages = [
            {"role": "system", "content": row.get("system", "You are a careful assistant.")},
            {"role": "user", "content": row["prompt"]},
        ]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]


def tokenize(
    row: dict[str, Any], url: str, model: str, timeout: int
) -> tuple[int, int]:
    body = json.dumps(
        {"model": model, "messages": request_messages(row)}, ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return int(payload["count"]), int(payload["max_model_len"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--desired-max-tokens", type=int, required=True)
    parser.add_argument("--override-request-max-tokens", action="store_true")
    parser.add_argument("--context-margin", type=int, default=128)
    parser.add_argument("--min-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.requests)
    url = tokenize_url(args.base_url)

    def inspect(row: dict[str, Any]) -> tuple[int, int]:
        return tokenize(row, url, args.model, args.timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        token_counts = list(pool.map(inspect, rows))

    output = []
    adjusted = []
    insufficient = []
    input_counts = []
    for row, (input_tokens, max_model_len) in zip(rows, token_counts):
        available = max_model_len - input_tokens - args.context_margin
        requested = (
            args.desired_max_tokens
            if args.override_request_max_tokens
            else min(
                int(row.get("max_tokens", args.desired_max_tokens)),
                args.desired_max_tokens,
            )
        )
        effective = min(requested, available)
        candidate_id = str((row.get("metadata") or {}).get("candidate_id") or row["id"])
        input_counts.append(input_tokens)
        if effective < args.min_output_tokens:
            insufficient.append({
                "id": row["id"],
                "candidate_id": candidate_id,
                "input_tokens": input_tokens,
                "available_output_tokens": available,
            })
        if effective != requested:
            adjusted.append({
                "id": row["id"],
                "candidate_id": candidate_id,
                "input_tokens": input_tokens,
                "requested_max_tokens": requested,
                "effective_max_tokens": effective,
            })
        output.append({
            **row,
            "max_tokens": effective,
            "metadata": {
                **(row.get("metadata") or {}),
                "context_preflight": {
                    "input_tokens": input_tokens,
                    "max_model_len": max_model_len,
                    "context_margin": args.context_margin,
                    "desired_max_tokens": requested,
                    "effective_max_tokens": effective,
                },
            },
        })

    input_counts.sort()
    report = {
        "schema_version": "ksova-chat-context-preflight-1.0",
        "requests": len(rows),
        "model": args.model,
        "tokenize_url": url,
        "desired_max_tokens": args.desired_max_tokens,
        "override_request_max_tokens": args.override_request_max_tokens,
        "context_margin": args.context_margin,
        "input_tokens": {
            "min": input_counts[0] if input_counts else 0,
            "median": input_counts[len(input_counts) // 2] if input_counts else 0,
            "max": input_counts[-1] if input_counts else 0,
        },
        "effective_max_tokens": dict(sorted(Counter(
            row["max_tokens"] for row in output
        ).items())),
        "adjusted": adjusted,
        "insufficient": insufficient,
        "output": str(args.output),
    }
    write_jsonl(args.output, output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 1 if insufficient else 0


if __name__ == "__main__":
    raise SystemExit(main())
