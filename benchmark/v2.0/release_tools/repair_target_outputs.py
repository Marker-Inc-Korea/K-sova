#!/usr/bin/env python3
"""Prepare targeted retries and atomically merge repaired model outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
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


def unique_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row.get("id") or ""): row for row in rows}
    if "" in result or len(result) != len(rows):
        raise ValueError(f"{label} has missing or duplicate ids")
    return result


def prepare(args: argparse.Namespace) -> int:
    requests = read_jsonl(args.requests)
    outputs = read_jsonl(args.outputs)
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    request_by_id = unique_by_id(requests, "requests")
    output_by_id = unique_by_id(outputs, "outputs")
    codes = set(args.error_code)
    validation_values = validation.get(args.validation_section, [])
    retry_ids = [
        str(error.get("id") or "")
        for error in validation_values
        if str(error.get("code") or "") in codes
    ]
    if not retry_ids or len(retry_ids) != len(set(retry_ids)):
        raise ValueError("retry ids are empty or duplicated")
    missing = sorted(set(retry_ids) - set(request_by_id))
    if missing:
        raise ValueError(f"retry ids absent from requests:{missing}")
    if sorted(set(retry_ids) - set(output_by_id)):
        raise ValueError("retry ids absent from outputs")
    if (
        not args.allow_nonempty_original
        and any(str(output_by_id[value].get("response") or "").strip() for value in retry_ids)
    ):
        raise ValueError("selected retry contains a nonempty original response")
    retry_set = set(retry_ids)
    retry_requests = [row for row in requests if str(row["id"]) in retry_set]
    write_jsonl(args.retry_requests, retry_requests)
    report = {
        "schema_version": "ksova-target-output-retry-prepare-1.0",
        "pass": True,
        "canonical_requests": len(requests),
        "original_outputs": len(outputs),
        "validation_section": args.validation_section,
        "error_codes": sorted(codes),
        "allow_nonempty_original": args.allow_nonempty_original,
        "retry_requests": len(retry_requests),
        "retry_ids": [str(row["id"]) for row in retry_requests],
        "retry_requests_sha256": sha256(args.retry_requests),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def merge(args: argparse.Namespace) -> int:
    requests = read_jsonl(args.requests)
    originals = read_jsonl(args.original_outputs)
    retries = read_jsonl(args.retry_outputs)
    request_by_id = unique_by_id(requests, "requests")
    original_by_id = unique_by_id(originals, "original outputs")
    retry_by_id = unique_by_id(retries, "retry outputs")
    if set(original_by_id) != set(request_by_id):
        raise ValueError("original output coverage differs from canonical requests")
    retry_record_models = sorted({str(row.get("model") or "") for row in retries})
    rejected_retries: list[dict[str, str]] = []
    accepted_retry_by_id: dict[str, dict[str, Any]] = {}
    for value, row in retry_by_id.items():
        if value not in request_by_id:
            raise ValueError(f"unknown retry output id:{value}")
        reason = ""
        if row.get("error"):
            reason = "runtime_error"
        elif str(row.get("finish_reason") or "").lower() != "stop":
            reason = "finish_reason_not_stop"
        elif not str(row.get("response") or "").strip():
            reason = "empty_response"
        if reason:
            if not args.keep_original_on_failed_retry:
                raise ValueError(f"retry rejected ({reason}):{value}")
            rejected_retries.append({"id": value, "reason": reason})
            continue
        accepted_retry_by_id[value] = row
    retry_by_id = accepted_retry_by_id
    if args.record_model_name:
        if any(str(row.get("model") or "") != args.record_model_name for row in originals):
            raise ValueError("original outputs do not match --record-model-name")
        retry_by_id = {
            value: {**row, "model": args.record_model_name}
            for value, row in retry_by_id.items()
        }
    merged_by_id = {**original_by_id, **retry_by_id}
    merged = [merged_by_id[str(row["id"])] for row in requests]
    write_jsonl(args.output, merged)
    report = {
        "schema_version": "ksova-target-output-retry-merge-1.0",
        "pass": True,
        "outputs": len(merged),
        "replaced_outputs": len(retry_by_id),
        "replacement_ids": sorted(retry_by_id),
        "rejected_retries": rejected_retries,
        "kept_original_ids": sorted(row["id"] for row in rejected_retries),
        "retry_record_models": retry_record_models,
        "merged_record_model": args.record_model_name or None,
        "empty_responses": sum(not str(row.get("response") or "").strip() for row in merged),
        "runtime_errors": sum(bool(row.get("error")) for row in merged),
        "merged_sha256": sha256(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--requests", type=Path, required=True)
    prepare_parser.add_argument("--outputs", type=Path, required=True)
    prepare_parser.add_argument("--validation", type=Path, required=True)
    prepare_parser.add_argument("--error-code", action="append", default=["empty_response"])
    prepare_parser.add_argument(
        "--validation-section", choices=("errors", "warnings"), default="errors"
    )
    prepare_parser.add_argument("--allow-nonempty-original", action="store_true")
    prepare_parser.add_argument("--retry-requests", type=Path, required=True)
    prepare_parser.add_argument("--report", type=Path, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--requests", type=Path, required=True)
    merge_parser.add_argument("--original-outputs", type=Path, required=True)
    merge_parser.add_argument("--retry-outputs", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--report", type=Path, required=True)
    merge_parser.add_argument("--record-model-name")
    merge_parser.add_argument(
        "--keep-original-on-failed-retry",
        action="store_true",
        help=(
            "merge only nonempty finish_reason=stop retries and retain the original "
            "output for rejected retry rows"
        ),
    )
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
