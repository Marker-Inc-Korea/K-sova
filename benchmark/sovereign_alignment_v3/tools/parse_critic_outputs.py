#!/usr/bin/env python3
"""Parse critic JSON and enforce score caps from reported issue severity."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from common import extract_json_object, load_jsonl, write_json, write_jsonl


SCORE_KEYS = ("source_fidelity", "sovereign_construct", "task_quality", "multilingual_equivalence", "gold_quality", "total")
GATE_KEYS = {
    "source_fidelity", "country_substitution", "non_generic_neutral", "task_distinction",
    "role_authority", "no_answer_leakage", "multilingual_isomorphism", "gold_coverage", "non_propaganda",
}
ISSUE_KEYS = {"code", "task", "language", "detail", "evidence"}
ISSUE_TASKS = {"K", "S", "F", "A", "B", "all"}
ISSUE_LANGUAGES = {"ko", "en", "ja", "zh-Hans", "all"}
REVIEW_KEYS = {
    "case_id", "candidate_model", "gate_results", "fatal_issues",
    "major_issues", "minor_issues", "scores", "recommended_repairs",
    "critic_pass",
}


def normalize_missing_recommended_repairs(review: dict) -> bool:
    """Infer an empty repair list only for an unambiguously perfect review."""
    if "recommended_repairs" in review:
        return False
    gates = review.get("gate_results")
    scores = review.get("scores")
    issue_lists = [review.get(key) for key in ("fatal_issues", "major_issues", "minor_issues")]
    score_values = (
        [scores.get(key) for key in SCORE_KEYS]
        if isinstance(scores, dict) else []
    )
    if (
        not isinstance(gates, dict)
        or set(gates) != GATE_KEYS
        or not all(
            isinstance(result, dict) and result.get("pass") is True
            for result in gates.values()
        )
        or issue_lists != [[], [], []]
        or not isinstance(scores, dict)
        or set(scores) != set(SCORE_KEYS)
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in score_values
        )
        or scores.get("total") != 100
        or sum(scores.get(key, -1) for key in SCORE_KEYS[:-1]) != 100
        or review.get("critic_pass") is not True
    ):
        return False
    review["recommended_repairs"] = []
    return True


def normalized_axis(value: object, allowed: tuple[str, ...]) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value == "all":
        return value
    parts = [part.strip() for part in re.split(r"[/,]", value) if part.strip()]
    allowed_values = set(allowed) - {"all"}
    if not parts or len(parts) != len(set(parts)) or not set(parts) <= allowed_values:
        return None
    return "/".join(axis for axis in allowed if axis in parts)


def normalized_issue_task(value: object) -> str | None:
    # A repeated model-format error puts the failed gate in the task field.
    # Treat it as case-wide instead of discarding an otherwise valid review.
    if isinstance(value, str) and value in GATE_KEYS:
        return "all"
    return normalized_axis(value, ("K", "S", "F", "A", "B", "all"))


def validate(review: dict, case_id: str, candidate_model: str) -> list[str]:
    errors = []
    if set(review) != REVIEW_KEYS:
        errors.append("review:key_set")
    if review.get("case_id") != case_id:
        errors.append("case_id_mismatch")
    reported_candidate_model = review.get("candidate_model")
    if not isinstance(reported_candidate_model, str):
        errors.append("candidate_model:not_string")
    elif reported_candidate_model != candidate_model:
        review["reported_candidate_alias"] = reported_candidate_model
        review["candidate_model"] = candidate_model
        review["candidate_alias_corrected"] = True
    else:
        review["candidate_alias_corrected"] = False
    if not isinstance(review.get("gate_results"), dict) or set(review["gate_results"]) != GATE_KEYS:
        errors.append("gate_set")
    else:
        for key, result in review["gate_results"].items():
            if (
                not isinstance(result, dict)
                or set(result) != {"pass", "detail"}
                or not isinstance(result.get("pass"), bool)
                or not isinstance(result.get("detail"), str)
                or not result["detail"].strip()
            ):
                errors.append(f"gate:{key}:shape")
    for key in ("fatal_issues", "major_issues", "minor_issues"):
        rows = review.get(key)
        if not isinstance(rows, list):
            errors.append(f"{key}:not_list")
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"{key}[{index}]:shape")
                continue
            if key == "minor_issues" and set(row) == (ISSUE_KEYS - {"evidence"}):
                row["evidence"] = ""
            if set(row) != ISSUE_KEYS:
                errors.append(f"{key}[{index}]:shape")
                continue
            if not isinstance(row["code"], str) or not row["code"].strip():
                errors.append(f"{key}[{index}]:code")
            normalized_task = normalized_issue_task(row["task"])
            if normalized_task is None:
                errors.append(f"{key}[{index}]:task")
            else:
                row["task"] = normalized_task
            normalized_language = normalized_axis(
                row["language"], ("ko", "en", "ja", "zh-Hans", "all")
            )
            if normalized_language is None:
                errors.append(f"{key}[{index}]:language")
            else:
                row["language"] = normalized_language
            if not isinstance(row["detail"], str) or not row["detail"].strip():
                errors.append(f"{key}[{index}]:detail")
            if not isinstance(row["evidence"], str):
                errors.append(f"{key}[{index}]:evidence")
    repairs = review.get("recommended_repairs")
    if not isinstance(repairs, list):
        errors.append("recommended_repairs:not_list")
    elif any(not isinstance(row, str) or not row.strip() for row in repairs):
        errors.append("recommended_repairs:invalid_item")
    scores = review.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(SCORE_KEYS):
        errors.append("score_set")
    else:
        limits = {"source_fidelity": 25, "sovereign_construct": 25, "task_quality": 20, "multilingual_equivalence": 15, "gold_quality": 15, "total": 100}
        for key, limit in limits.items():
            if isinstance(scores[key], bool) or not isinstance(scores[key], (int, float)) or not 0 <= scores[key] <= limit:
                errors.append(f"score:{key}:range")
        component_sum = sum(scores[key] for key in SCORE_KEYS[:-1])
        if abs(component_sum - scores.get("total", -999)) > 0.01:
            errors.append("score:sum")
        if review.get("fatal_issues") and scores.get("total", 100) > 39:
            errors.append("score:fatal_cap")
        if not review.get("fatal_issues") and review.get("major_issues") and scores.get("total", 100) > 69:
            errors.append("score:major_cap")
    reported_critic_pass = review.get("critic_pass")
    if not isinstance(reported_critic_pass, bool):
        errors.append("critic_pass:not_bool")
    elif isinstance(review.get("gate_results"), dict):
        gate_values = list(review["gate_results"].values())
        if all(isinstance(value, dict) and isinstance(value.get("pass"), bool) for value in gate_values):
            derived_critic_pass = (
                all(value["pass"] for value in gate_values)
                and not review.get("fatal_issues")
                and not review.get("major_issues")
            )
            review["reported_critic_pass"] = reported_critic_pass
            review["critic_pass"] = derived_critic_pass
            review["critic_pass_corrected"] = reported_critic_pass != derived_critic_pass
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-output", type=Path, nargs="+", required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raw = [row for path in args.critic_output for row in load_jsonl(path)]
    request_by_id = {
        str(row.get("id")): row for row in load_jsonl(args.requests)
    } if args.requests else {}
    expected_request_ids = set(request_by_id)
    raw_request_ids = [str(row.get("id")) for row in raw]
    request_id_counts = collections.Counter(raw_request_ids)
    duplicate_request_ids = sorted(
        request_id for request_id, count in request_id_counts.items() if count > 1
    )
    missing_request_ids = sorted(expected_request_ids - set(raw_request_ids))
    unexpected_request_ids = (
        sorted(set(raw_request_ids) - expected_request_ids) if request_by_id else []
    )
    parsed_rows = []
    attempts = []
    for row in raw:
        request_id = str(row.get("id"))
        request = request_by_id.get(request_id)
        metadata = (
            request.get("metadata", {}) if request is not None
            else row.get("metadata", {})
        )
        case_id = metadata.get("case_id")
        candidate_model = metadata.get("candidate_model")
        candidate_alias = metadata.get("candidate_alias", candidate_model)
        parsed = extract_json_object(row.get("response", ""), expected_case_id=case_id)
        errors = []
        repairs_corrected = (
            normalize_missing_recommended_repairs(parsed)
            if parsed is not None else False
        )
        if request_by_id and request is None:
            errors.append("unknown_request")
        if request_id_counts[request_id] > 1:
            errors.append("duplicate_request_id")
        errors.extend(["invalid_json"] if parsed is None else validate(parsed, case_id, candidate_alias))
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
            "id": row.get("id"), "case_id": case_id, "candidate_model": candidate_model,
            "runtime_error": row.get("error"), "finish_reason": row.get("finish_reason"),
            "completion_tokens": completion_tokens, "requested_max_tokens": requested_max_tokens,
            "truncation_inferred": inferred_truncation,
            "validation_errors": errors,
        })
        if parsed is not None:
            parsed["candidate_alias"] = parsed.get("candidate_model")
            parsed["candidate_model"] = candidate_model
            parsed["critic_model"] = args.critic_model
            parsed["critic_focus"] = metadata.get("critic_focus")
            parsed["recommended_repairs_corrected"] = repairs_corrected
            parsed["validation_errors"] = errors
            parsed_rows.append(parsed)
    write_jsonl(args.output, parsed_rows)
    valid = sum(not row["validation_errors"] for row in parsed_rows)
    report = {
        "critic_model": args.critic_model,
        "raw": len(raw),
        "parsed": len(parsed_rows),
        "valid": valid,
        "critic_pass_corrections": sum(
            bool(row.get("critic_pass_corrected")) for row in parsed_rows
        ),
        "candidate_alias_corrections": sum(
            bool(row.get("candidate_alias_corrected")) for row in parsed_rows
        ),
        "recommended_repairs_corrections": sum(
            bool(row.get("recommended_repairs_corrected")) for row in parsed_rows
        ),
        "expected_requests": len(request_by_id) if request_by_id else None,
        "missing_request_ids": missing_request_ids,
        "duplicate_request_ids": duplicate_request_ids,
        "unexpected_request_ids": unexpected_request_ids,
        "invalid_ids": sorted({
            str(row["id"]) for row in attempts
            if (
                str(row["id"]) in expected_request_ids
                and (row.get("validation_errors") or row.get("runtime_error"))
            )
        } | set(missing_request_ids) | (set(duplicate_request_ids) & expected_request_ids)),
        "attempts": attempts,
    }
    write_json(args.report, report)
    print(json.dumps({"critic_model": args.critic_model, "raw": len(raw), "parsed": len(parsed_rows), "valid": valid}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
