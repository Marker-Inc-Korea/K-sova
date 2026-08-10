#!/usr/bin/env python3
"""Apply final semantic-duplicate judgments while never removing frozen O items."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RELATIONS = {"semantic_duplicate", "related_but_distinct", "unrelated"}
PREFERENCES = {"item_a", "item_b", "both"}
BOOLS = (
    "same_required_judgment", "same_aligned_conclusion", "same_decisive_basis",
    "same_practical_consequence",
)
BOOL_ALIASES = {
    "same_required_judgment": ("same_required_jjudgment",),
}


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def parse_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    return value


def quality(row: dict[str, Any]) -> tuple[Any, ...]:
    validation = row.get("semantic_validation") or {}
    qwen = row.get("qwen_validation") or {}
    oss = row.get("oss_validation") or {}
    answers = row.get("answers") or {}
    lengths = [
        len(str(answers[direction].get("answer_ko") or ""))
        for direction in ("korean_aligned", "context_neutral", "korean_opposed")
        if isinstance(answers.get(direction), dict)
    ]
    source_validation = row.get("source_validation") or {}
    return (
        int(validation.get("quality_score") or min(
            int(qwen.get("quality_score") or 0),
            int(oss.get("quality_score") or 0),
        )),
        bool(source_validation.get("effective_pass")),
        min(lengths, default=0),
        len(row.get("sources") or []),
        str(row.get("item_id")),
    )


def parse_batch(
    request: dict[str, Any], response: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    expected = set(request["metadata"]["pair_ids"])
    errors: list[str] = []
    normalizations: list[dict[str, str]] = []
    payload: dict[str, Any] = {}
    if response.get("error"):
        errors.append(str(response["error"]))
    elif response.get("metadata") != request.get("metadata"):
        errors.append("metadata_mismatch")
    if not errors:
        try:
            payload = parse_object(str(response.get("response") or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"parse:{exc}")
    values = payload.get("judgments") if not errors else None
    if not errors and not isinstance(values, list):
        errors.append("invalid_judgments")
    parsed_batch: list[dict[str, Any]] = []
    if not errors:
        found_ids: list[str] = []
        for value in values:
            if not isinstance(value, dict):
                errors.append("invalid_judgment_object")
                continue
            pair_id = str(value.get("pair_id") or "")
            found_ids.append(pair_id)
            for canonical, aliases in BOOL_ALIASES.items():
                if canonical in value:
                    continue
                usable = [alias for alias in aliases if isinstance(value.get(alias), bool)]
                if len(usable) == 1:
                    alias = usable[0]
                    value[canonical] = value[alias]
                    normalizations.append({
                        "request_id": str(request["id"]),
                        "pair_id": pair_id,
                        "from": alias,
                        "to": canonical,
                    })
            if pair_id not in expected:
                errors.append(f"unexpected_pair:{pair_id}")
            if value.get("relation") not in RELATIONS:
                errors.append(f"invalid_relation:{pair_id}")
            if value.get("preferred_item") not in PREFERENCES:
                errors.append(f"invalid_preference:{pair_id}")
            if any(not isinstance(value.get(key), bool) for key in BOOLS):
                errors.append(f"invalid_boolean:{pair_id}")
            if not isinstance(value.get("reason_ko"), str) or len(value["reason_ko"].strip()) < 8:
                errors.append(f"invalid_reason:{pair_id}")
            parsed_batch.append(value)
        if set(found_ids) != expected or len(found_ids) != len(expected):
            errors.append("pair_coverage_mismatch")
    return parsed_batch, sorted(set(errors)), normalizations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--new", type=Path, nargs="+", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--distinct-new", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    frozen = read_jsonl([args.frozen])
    new_rows = read_jsonl(args.new)
    frozen_ids = {row["item_id"] for row in frozen}
    all_by_id = {row["item_id"]: row for row in frozen + new_rows}
    new_by_id = {row["item_id"]: row for row in new_rows}
    requests = read_jsonl([args.requests])
    responses = read_jsonl(args.responses)
    responses_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        responses_by_id[str(response.get("id"))].append(response)

    judgments: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    schema_normalizations: list[dict[str, str]] = []
    selected_retry_attempts = 0
    for request in requests:
        attempts = responses_by_id.get(str(request["id"])) or []
        response: dict[str, Any] | None = None
        parsed_batch: list[dict[str, Any]] = []
        attempt_errors: list[dict[str, Any]] = []
        selected_normalizations: list[dict[str, str]] = []
        selected_index = -1
        for index in range(len(attempts) - 1, -1, -1):
            candidate_batch, errors, normalizations = parse_batch(request, attempts[index])
            if not errors:
                response = attempts[index]
                parsed_batch = candidate_batch
                selected_normalizations = normalizations
                selected_index = index
                break
            attempt_errors.append({"attempt": index + 1, "errors": errors})
        if response is None:
            invalid.append({
                "request_id": request["id"],
                "errors": ["missing_response"] if not attempts else attempt_errors[0]["errors"],
                "attempts": attempt_errors,
            })
            continue
        schema_normalizations.extend(selected_normalizations)
        selected_retry_attempts += int(selected_index > 0)
        for value in parsed_batch:
            left_id, right_id = value["pair_id"].split("__", 1)
            judgments.append({
                **value,
                "item_a_id": left_id,
                "item_b_id": right_id,
                "auditor_model": response.get("model"),
                "messages_sha256": response.get("messages_sha256"),
            })

    if invalid:
        # Missing duplicate decisions must never be interpreted as evidence of distinctness.
        report = {
            "schema_version": "ksova-v2-final-semantic-dedup-report-1.0",
            "pass": False,
            "requests": len(requests),
            "response_files": [str(path) for path in args.responses],
            "response_attempts": len(responses),
            "selected_retry_attempts": selected_retry_attempts,
            "parsed_judgments": len(judgments),
            "invalid": invalid,
            "schema_normalizations": schema_normalizations,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        return 1

    graph: dict[str, set[str]] = defaultdict(set)
    preference_votes: Counter[str] = Counter()
    for row in judgments:
        if row["relation"] != "semantic_duplicate":
            continue
        left, right = row["item_a_id"], row["item_b_id"]
        graph[left].add(right)
        graph[right].add(left)
        if row["preferred_item"] == "item_a":
            preference_votes[left] += 1
        elif row["preferred_item"] == "item_b":
            preference_votes[right] += 1

    clusters: list[list[str]] = []
    seen: set[str] = set()
    for item_id in sorted(graph):
        if item_id in seen:
            continue
        stack, component = [item_id], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(graph[current] - component)
        seen.update(component)
        clusters.append(sorted(component))

    removed: dict[str, dict[str, Any]] = {}
    representative_by_id: dict[str, str] = {}
    for cluster in clusters:
        frozen_members = [item_id for item_id in cluster if item_id in frozen_ids]
        if frozen_members:
            # Every human-approved item remains frozen, even if two frozen items are related.
            representative = max(frozen_members, key=lambda item_id: (preference_votes[item_id], item_id))
            for item_id in cluster:
                if item_id not in frozen_ids:
                    removed[item_id] = {
                        "reason": "semantic_duplicate_of_frozen_approved",
                        "representative": representative,
                        "cluster": cluster,
                    }
                    representative_by_id[item_id] = representative
            continue
        representative = max(
            cluster,
            key=lambda item_id: (preference_votes[item_id], *quality(all_by_id[item_id])),
        )
        for item_id in cluster:
            if item_id != representative:
                removed[item_id] = {
                    "reason": "semantic_duplicate_within_new_pool",
                    "representative": representative,
                    "cluster": cluster,
                }
                representative_by_id[item_id] = representative

    distinct = []
    for item_id, row in sorted(new_by_id.items()):
        if item_id in removed:
            continue
        duplicate_audit = {
            "status": "distinct_after_audit",
            "compared_pairs": sum(
                item_id in {judgment["item_a_id"], judgment["item_b_id"]}
                for judgment in judgments
            ),
        }
        distinct.append({**row, "final_duplicate_audit": duplicate_audit})

    for path, rows in ((args.judgments, judgments), (args.distinct_new, distinct)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    report = {
        "schema_version": "ksova-v2-final-semantic-dedup-report-1.0",
        "pass": True,
        "requests": len(requests),
        "response_files": [str(path) for path in args.responses],
        "response_attempts": len(responses),
        "selected_retry_attempts": selected_retry_attempts,
        "parsed_judgments": len(judgments),
        "relations": dict(sorted(Counter(row["relation"] for row in judgments).items())),
        "duplicate_clusters": clusters,
        "removed_new": removed,
        "frozen_preserved": len(frozen),
        "input_new": len(new_rows),
        "distinct_new": len(distinct),
        "invalid": [],
        "schema_normalizations": schema_normalizations,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": True,
        "judgments": len(judgments),
        "duplicate_clusters": len(clusters),
        "removed_new": len(removed),
        "distinct_new": len(distinct),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
