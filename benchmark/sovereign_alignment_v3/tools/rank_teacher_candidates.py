#!/usr/bin/env python3
"""Aggregate structural validity and blind critic scores for teacher selection."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from common import load_json, load_jsonl, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-report", type=Path, action="append", default=[])
    parser.add_argument("--critic-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-case", action="append", default=[])
    parser.add_argument("--expected-reviews-per-case", type=int, default=2)
    parser.add_argument("--eligible-model", action="append", default=[])
    args = parser.parse_args()
    teacher_reports = [load_json(path) for path in args.teacher_report]
    reviews = [row for path in args.critic_file for row in load_jsonl(path)]
    by_candidate: dict[str, dict] = {}
    excluded_cases = set(args.exclude_case)
    for report in teacher_reports:
        model = report["model"]
        attempts = [row for row in report.get("attempts", []) if row.get("case_id") not in excluded_cases]
        attempted = len(attempts) if attempts else report["raw_attempts"]
        structurally_valid = (
            sum(not row.get("validation_errors") for row in attempts)
            if attempts else report["structurally_valid_cases"]
        )
        by_candidate[model] = {
            "candidate_model": model,
            "attempted": attempted,
            "attempted_cases": sorted({str(row.get("case_id")) for row in attempts}),
            "structurally_valid": structurally_valid,
            "structural_pass_rate": structurally_valid / max(attempted, 1),
            "critic_scores": [], "fatal_reviews": 0, "major_reviews": 0, "critic_passes": 0,
            "gate_failures": collections.Counter(),
        }
    duplicate_reviews = []
    unknown_candidate_reviews = []
    seen_review_keys: set[tuple[str, str, str, str]] = set()
    reviews_by_candidate_case: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for review in reviews:
        model = review.get("candidate_model")
        case_id = str(review.get("case_id"))
        if case_id in excluded_cases:
            continue
        if model not in by_candidate:
            unknown_candidate_reviews.append(f"{model}::{case_id}")
            continue
        if review.get("validation_errors"):
            continue
        review_key = (
            str(model), case_id, str(review.get("critic_model")), str(review.get("critic_focus")),
        )
        if review_key in seen_review_keys:
            duplicate_reviews.append("::".join(review_key))
            continue
        seen_review_keys.add(review_key)
        reviews_by_candidate_case[(str(model), case_id)].append(review)
        data = by_candidate[model]
        data["critic_scores"].append(review["scores"]["total"])
        data["fatal_reviews"] += int(bool(review.get("fatal_issues")))
        data["major_reviews"] += int(bool(review.get("major_issues")))
        data["critic_passes"] += int(bool(review.get("critic_pass")))
        for gate, result in review.get("gate_results", {}).items():
            if not result.get("pass"):
                data["gate_failures"][gate] += 1
    eligible_models = set(args.eligible_model)
    ranking = []
    for data in by_candidate.values():
        scores = data.pop("critic_scores")
        data["critic_review_count"] = len(scores)
        data["expected_critic_reviews"] = data["attempted"] * args.expected_reviews_per_case
        data["critic_coverage"] = round(
            min(1.0, len(scores) / max(data["expected_critic_reviews"], 1)), 4
        )
        data["critic_mean"] = round(statistics.mean(scores), 3) if scores else None
        data["critic_median"] = round(statistics.median(scores), 3) if scores else None
        data["gate_failures"] = dict(sorted(data["gate_failures"].items()))
        coverage_issues = []
        for case_id in data.pop("attempted_cases"):
            case_reviews = reviews_by_candidate_case[(data["candidate_model"], case_id)]
            critic_models = {str(review.get("critic_model")) for review in case_reviews}
            critic_focuses = {str(review.get("critic_focus")) for review in case_reviews}
            if len(case_reviews) != args.expected_reviews_per_case:
                coverage_issues.append(f"{case_id}:reviews={len(case_reviews)}")
            if len(critic_models) != args.expected_reviews_per_case:
                coverage_issues.append(f"{case_id}:critic_models={len(critic_models)}")
            if args.expected_reviews_per_case == 2 and critic_focuses != {"independent", "korean"}:
                coverage_issues.append(f"{case_id}:focuses={','.join(sorted(critic_focuses))}")
        data["critic_case_coverage_complete"] = not coverage_issues
        data["critic_coverage_issues"] = coverage_issues
        mean = data["critic_mean"] or 0
        data["selection_score"] = round(
            30 * data["structural_pass_rate"] + 0.70 * mean * data["critic_coverage"]
            - 5 * data["fatal_reviews"] - 2 * data["major_reviews"], 3
        )
        data["primary_eligible"] = not eligible_models or data["candidate_model"] in eligible_models
        ranking.append(data)
    ranking.sort(key=lambda row: (row["selection_score"], row["structural_pass_rate"]), reverse=True)
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    eligible_ranking = [row for row in ranking if row["primary_eligible"]]

    def selection_gate_errors(row: dict) -> list[str]:
        errors = []
        minimum_structural = max(row["attempted"] - 1, 1)
        if row["structurally_valid"] < minimum_structural:
            errors.append(f"structural={row['structurally_valid']}<{minimum_structural}")
        if row["fatal_reviews"]:
            errors.append(f"fatal_reviews={row['fatal_reviews']}")
        if row["critic_coverage"] != 1.0:
            errors.append(f"critic_coverage={row['critic_coverage']}")
        if not row["critic_case_coverage_complete"]:
            errors.append("critic_case_coverage_incomplete")
        return errors

    for row in ranking:
        row["selection_gate_errors"] = selection_gate_errors(row)

    review_integrity_ok = not duplicate_reviews and not unknown_candidate_reviews
    selected = (
        eligible_ranking[0]
        if review_integrity_ok and eligible_ranking and not eligible_ranking[0]["selection_gate_errors"]
        else None
    )
    report = {
        "ranking": ranking,
        "selection_rule": (
            "30% structural pass + 70% mean blind-critic score multiplied by critic coverage, "
            "with fatal/major penalties"
        ),
        "excluded_cases": sorted(set(args.exclude_case)),
        "expected_reviews_per_case": args.expected_reviews_per_case,
        "eligible_models": sorted(eligible_models) if eligible_models else "all",
        "review_integrity": {
            "duplicate_reviews": sorted(set(duplicate_reviews)),
            "unknown_candidate_reviews": sorted(set(unknown_candidate_reviews)),
            "status": "pass" if review_integrity_ok else "fail",
        },
        "decision": selected["candidate_model"] if selected else None,
        "decision_global_rank": selected["rank"] if selected else None,
        "provisional_leader": eligible_ranking[0]["candidate_model"] if eligible_ranking else None,
        "provisional_leader_gate_errors": (
            eligible_ranking[0]["selection_gate_errors"] if eligible_ranking else ["no_eligible_model"]
        ),
        "status": "selected" if selected else "manual_or_repair_required",
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "decision": report["decision"], "ranking": [(row["rank"], row["candidate_model"], row["selection_score"]) for row in ranking]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
