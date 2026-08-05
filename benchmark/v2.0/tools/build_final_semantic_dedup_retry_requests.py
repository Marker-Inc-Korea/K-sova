#!/usr/bin/env python3
"""Build retry requests for invalid final semantic-dedup batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RETRY_INSTRUCTION = """이전 응답은 출력 계약을 완전히 충족하지 못했다.
입력의 모든 pair_id를 정확히 한 번씩 반환하고 누락하거나 새 ID를 만들지 마라.
각 judgment에는 relation, 네 개의 same_* 불리언, preferred_item, 8자 이상의 reason_ko를 모두 넣어라.
semantic_duplicate가 아니면 preferred_item은 both다. JSON 객체 하나만 출력하라."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--parse-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    requests = read_jsonl(args.requests)
    by_id = {str(row["id"]): row for row in requests}
    parse_report = json.loads(args.parse_report.read_text(encoding="utf-8"))
    invalid_ids = [str(row["request_id"]) for row in parse_report.get("invalid") or []]
    unknown = sorted(set(invalid_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown invalid request IDs: {unknown[:10]}")

    rows: list[dict[str, Any]] = []
    for request_id in invalid_ids:
        row = json.loads(json.dumps(by_id[request_id], ensure_ascii=False))
        messages = row.get("messages") or []
        system = next((value for value in messages if value.get("role") == "system"), None)
        if system is None:
            messages.insert(0, {"role": "system", "content": RETRY_INSTRUCTION})
        else:
            system["content"] = str(system.get("content") or "") + "\n\n" + RETRY_INSTRUCTION
        row["messages"] = messages
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "ksova-v2-final-semantic-dedup-retry-plan-1.0",
        "source_requests": len(requests),
        "retry_requests": len(rows),
        "request_ids": invalid_ids,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
