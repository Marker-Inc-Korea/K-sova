#!/usr/bin/env python3
"""Flatten validated final teacher cases into 480 evaluation items."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from common import LANGUAGES, ROOT, TASK_IDS, load_json, load_jsonl, stable_item_id, validate_teacher_case, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-cases", type=Path, required=True)
    parser.add_argument("--blueprints", type=Path, default=ROOT / "config/case_blueprints.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/pilot_items_v0.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/pilot_build_v0.json")
    args = parser.parse_args()
    blueprints = {case["id"]: case for case in load_json(args.blueprints)}
    final_case_rows = load_jsonl(args.final_cases)
    final_case_ids = [str(row.get("case_id")) for row in final_case_rows]
    duplicate_case_ids = sorted({case_id for case_id in final_case_ids if final_case_ids.count(case_id) > 1})
    unknown_case_ids = sorted(set(final_case_ids) - set(blueprints))
    final_cases = {row["case_id"]: row for row in final_case_rows if row.get("case_id") in blueprints}
    errors = []
    if duplicate_case_ids:
        errors.append({"case_id": "*", "errors": ["duplicate_final_cases:" + ",".join(duplicate_case_ids)]})
    if unknown_case_ids:
        errors.append({"case_id": "*", "errors": ["unknown_final_cases:" + ",".join(unknown_case_ids)]})
    rows = []
    for case_id, blueprint in sorted(blueprints.items()):
        authored = final_cases.get(case_id)
        if authored is None:
            errors.append({"case_id": case_id, "errors": ["missing_final_case"]})
            continue
        clean = {key: authored[key] for key in (
            "case_id", "source_or_gold_issues", "claim_source_map", "role_labels", "items", "quality_notes", "teacher_pass"
        ) if key in authored}
        validation_errors = validate_teacher_case(clean, blueprint, require_role_labels=True)
        if validation_errors or not clean.get("teacher_pass") or clean.get("source_or_gold_issues"):
            errors.append({
                "case_id": case_id,
                "errors": validation_errors,
                "teacher_pass": clean.get("teacher_pass"),
                "source_or_gold_issues": clean.get("source_or_gold_issues"),
            })
            continue
        source_by_id = {source["id"]: source for source in blueprint["sources"]}
        for task_id in TASK_IDS:
            item = clean["items"][task_id]
            canonical_id = f"ksa3_{case_id}_{task_id.lower()}"
            required = [blueprint["required_claims"][index] for index in item["required_claim_indices"]]
            forbidden = [blueprint["forbidden_claims"][index] for index in item["forbidden_claim_indices"]]
            for language in LANGUAGES:
                rows.append({
                    "id": stable_item_id(case_id, task_id, language),
                    "canonical_id": canonical_id,
                    "case_id": case_id,
                    "domain_id": blueprint["domain_id"],
                    "title": blueprint["title"],
                    "task_id": task_id,
                    "language": language,
                    "role": blueprint["role"],
                    "role_label": clean["role_labels"][language],
                    "as_of": blueprint["as_of"],
                    "stance_class": blueprint["stance_class"],
                    "comparison_jurisdiction": blueprint["comparison_jurisdiction"],
                    "prompt": item["prompts"][language].strip(),
                    "reference_answer_ko": item["reference_answer_ko"].strip(),
                    "required_claim_indices": item["required_claim_indices"],
                    "required_claims": required,
                    "forbidden_claim_indices": item["forbidden_claim_indices"],
                    "forbidden_claims": forbidden,
                    "acceptable_positions": copy.deepcopy(blueprint["acceptable_positions"]),
                    "authority_boundaries": copy.deepcopy(blueprint["authority_boundaries"]),
                    "tradeoffs": copy.deepcopy(blueprint["tradeoffs"]),
                    "uncertainty": copy.deepcopy(blueprint["uncertainty"]),
                    "claim_source_map": copy.deepcopy(clean["claim_source_map"]),
                    "sources": list(source_by_id.values()),
                    "task_focus": item["task_focus"],
                    "headline_score": task_id != "K",
                    "status": "diagnostic_pilot_expert_review_required",
                })
    count = write_jsonl(args.output, rows)
    report = {
        "case_blueprints": len(blueprints), "blueprints": str(args.blueprints),
        "final_cases": len(final_case_rows),
        "built_items": count, "expected_items": 480, "rejected_cases": len(errors),
        "rejected_case_details": errors, "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if count == 480 and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
