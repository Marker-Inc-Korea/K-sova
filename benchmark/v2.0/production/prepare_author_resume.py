#!/usr/bin/env python3
"""Preserve valid author responses and build a source-eligible resume queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import (
    parse_object_with_status,
    read_jsonl,
    structural_errors,
    write_json,
    write_jsonl,
)


def source_policy(
    request: dict[str, Any], snapshots: dict[str, list[dict[str, Any]]]
) -> str:
    metadata = request.get("metadata") or {}
    source_ids = list(dict.fromkeys(str(value) for value in metadata.get("source_ids") or []))
    available = {
        str(source.get("source_id")): source
        for source in snapshots.get(str(metadata.get("seed_id")), [])
    }
    if not source_ids or any(source_id not in available for source_id in source_ids):
        return "excluded_missing_source"
    if len(source_ids) >= 2:
        return "multi_source"
    source = available[source_ids[0]]
    if source.get("reliability_tier") in {"A", "B"} and source.get("body_read") is True:
        return "single_authoritative_ab"
    return "excluded_single_discovery_source"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, action="append", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--eligible-requests", type=Path, required=True)
    parser.add_argument("--reusable-responses", type=Path, required=True)
    parser.add_argument("--pending-requests", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    requests = read_jsonl(args.requests)
    snapshots = {
        str(row["seed_id"]): list(row.get("sources") or [])
        for row in read_jsonl(args.snapshot)
    }
    all_responses: list[dict[str, Any]] = []
    for path in args.responses:
        all_responses.extend(read_jsonl(path))
    response_counts = Counter(str(row.get("id") or "") for row in all_responses)
    response_by_id = {str(row.get("id") or ""): row for row in all_responses}

    policies = {str(row["id"]): source_policy(row, snapshots) for row in requests}
    eligible = [
        row for row in requests
        if policies[str(row["id"])] in {"multi_source", "single_authoritative_ab"}
    ]
    reusable: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    reuse_recoveries: Counter[str] = Counter()
    pending_reasons: Counter[str] = Counter()

    for request in eligible:
        request_id = str(request["id"])
        response = response_by_id.get(request_id)
        reason = None
        if response is None:
            reason = "missing_response"
        elif response_counts[request_id] != 1:
            reason = "duplicate_response"
        elif response.get("error"):
            reason = "response_error"
        elif response.get("metadata", {}) != request.get("metadata", {}):
            reason = "metadata_mismatch"
        else:
            try:
                payload, recovery = parse_object_with_status(
                    str(response.get("response") or "")
                )
                errors = structural_errors(
                    payload,
                    request["metadata"],
                    set(request["metadata"].get("source_ids") or []),
                )
                if errors:
                    reason = f"structural:{errors[0].split(':', 1)[0]}"
                elif recovery:
                    reuse_recoveries[recovery] += 1
            except (ValueError, json.JSONDecodeError) as exc:
                reason = f"parse:{type(exc).__name__}"
        if reason is None:
            reusable.append(response)
        else:
            pending.append(request)
            pending_reasons[reason] += 1

    write_jsonl(args.eligible_requests, eligible)
    write_jsonl(args.reusable_responses, reusable)
    write_jsonl(args.pending_requests, pending)
    eligible_seed_ids = {
        str(row.get("metadata", {}).get("seed_id")) for row in eligible
    }
    report = {
        "schema_version": "ksova-v2-author-resume-plan-1.0",
        "input_requests": len(requests),
        "input_responses": len(all_responses),
        "request_source_policy": dict(sorted(Counter(policies.values()).items())),
        "eligible_requests": len(eligible),
        "eligible_seeds": len(eligible_seed_ids),
        "reusable_responses": len(reusable),
        "reusable_recoveries": dict(sorted(reuse_recoveries.items())),
        "pending_requests": len(pending),
        "pending_reasons": dict(sorted(pending_reasons.items())),
        "outputs": {
            "eligible_requests": str(args.eligible_requests),
            "reusable_responses": str(args.reusable_responses),
            "pending_requests": str(args.pending_requests),
        },
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
