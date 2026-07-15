#!/usr/bin/env python3
"""Parse and structurally validate teacher model responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT, extract_json_object, load_json, load_jsonl, validate_teacher_case, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-output", type=Path, nargs="+", required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-role-labels", action="store_true")
    args = parser.parse_args()
    blueprints = {case["id"]: case for case in load_json(ROOT / "config/case_blueprints.json")}
    raw_rows = [row for path in args.teacher_output for row in load_jsonl(path)]
    request_rows = load_jsonl(args.requests) if args.requests else []
    request_by_id = {str(row.get("id")): row for row in request_rows}
    expected_request_ids = [str(row.get("id")) for row in request_rows]
    expected_case_ids = {
        str(row.get("metadata", {}).get("case_id") or row.get("id")) for row in request_rows
    }
    raw_request_ids = [str(row.get("id")) for row in raw_rows]
    duplicate_request_ids = sorted({
        request_id for request_id in raw_request_ids if raw_request_ids.count(request_id) > 1
    })
    missing_request_ids = sorted(set(expected_request_ids) - set(raw_request_ids))
    unexpected_request_ids = sorted(set(raw_request_ids) - set(expected_request_ids)) if request_rows else []
    parsed_by_case: dict[str, dict] = {}
    attempts = []
    for row in raw_rows:
        case_id = row.get("metadata", {}).get("case_id") or row.get("id")
        request_id = str(row.get("id"))
        parsed = extract_json_object(row.get("response", ""), expected_case_id=case_id)
        errors = ["unknown_case"] if case_id not in blueprints else (
            ["invalid_json"] if parsed is None else validate_teacher_case(
                parsed, blueprints[case_id], require_role_labels=args.require_role_labels
            )
        )
        finish_reason = str(row.get("finish_reason") or "").lower()
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        completion_tokens = usage.get("completion_tokens")
        requested_max_tokens = row.get("requested_max_tokens")
        if requested_max_tokens is None and request_id in request_by_id:
            requested_max_tokens = request_by_id[request_id].get("max_tokens")
        inferred_truncation = (
            not finish_reason
            and type(completion_tokens) is int
            and type(requested_max_tokens) is int
            and completion_tokens >= requested_max_tokens
        )
        if finish_reason == "length" or inferred_truncation:
            errors.append("truncated_generation")
        attempts.append({
            "case_id": case_id, "request_id": row.get("id"), "runtime_error": row.get("error"),
            "finish_reason": row.get("finish_reason"), "completion_tokens": completion_tokens,
            "requested_max_tokens": requested_max_tokens, "truncation_inferred": inferred_truncation,
            "parse_ok": parsed is not None, "validation_errors": errors,
        })
        if parsed is not None and case_id in blueprints:
            parsed["teacher_model"] = args.model_name
            parsed["structural_validation_errors"] = errors
            parsed_by_case[case_id] = parsed
    rows = [parsed_by_case[key] for key in sorted(parsed_by_case)]
    write_jsonl(args.output, rows)
    valid = sum(not row["structural_validation_errors"] for row in rows)
    report = {
        "model": args.model_name, "raw_attempts": len(raw_rows), "parsed_cases": len(rows),
        "structurally_valid_cases": valid, "invalid_cases": len(rows) - valid,
        "expected_cases": len(expected_case_ids) if request_rows else None,
        "missing_request_ids": missing_request_ids,
        "unexpected_request_ids": unexpected_request_ids,
        "duplicate_request_ids": duplicate_request_ids,
        "missing_or_unparsed_cases": (
            len(expected_case_ids - set(parsed_by_case)) if request_rows
            else len(raw_rows) - len(rows)
        ),
        "invalid_ids": sorted(set(
            missing_request_ids
            + [str(row["request_id"]) for row in attempts if row.get("validation_errors") or row.get("runtime_error")]
        )),
        "attempts": attempts,
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps({key: report[key] for key in ("model", "raw_attempts", "parsed_cases", "structurally_valid_cases", "invalid_cases")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
