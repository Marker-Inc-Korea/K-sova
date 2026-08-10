#!/usr/bin/env python3
"""Fail-closed GLM semantic deduplication of new SFT examples against protected data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact(row: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "example_id": row["example_id"],
        "form": row.get("form"),
        "generation_profile": row.get("generation_profile", "baseline"),
        "question_ko": row.get("question_ko"),
        "answer_ko": str(row.get("answer_ko") or "")[:300],
    }


def prompt(
    material_id: str,
    references: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    contract = {
        "material_id": material_id,
        "distinct_candidate_ids": ["중복 그룹에 속하지 않는 새 candidate example_id"],
        "duplicate_groups": [{
            "member_ids": ["서로 같은 판단을 요구하는 2개 이상의 example_id"],
            "keep_id": "보존할 example_id",
            "reason_ko": "요구 판단·최소 결론·결정적 근거·실제 결과가 같은 이유",
        }],
    }
    return f"""같은 K-SOVA 소재에서 만든 학습 예시의 의미 중복만 판정하라.

[보호된 기존 예시]
{json.dumps(references, ensure_ascii=False, indent=2)}

[새 후보 예시]
{json.dumps(candidates, ensure_ascii=False, indent=2)}

판정 규칙:
1. 문장이나 형식이 달라도 요구 판단, 한국의 최소 결론, 결정적 근거, 실제 결과가 같아 답변을 거의 그대로
   바꿔 쓸 수 있으면 한 duplicate_group이다.
2. 같은 사건이어도 판단 주체, 권리·책임, 적용 조건, 비용 귀속, 불확실성 경계 또는 실제 대응 결과가 달라
   별도 논증이 필요하면 distinct다. 단순히 같은 소재라는 이유로 합치지 않는다.
3. 모든 새 후보 ID는 distinct_candidate_ids 또는 정확히 한 duplicate_group에 한 번만 포함한다.
4. 보호된 기존 예시는 distinct_candidate_ids에 넣지 않는다. 중복 그룹에 보호된 기존 예시가 하나라도 있으면
   keep_id는 반드시 그 보호 예시 중 하나다. 보호 예시는 삭제 대상으로 삼지 않는다.
5. 새 후보끼리만 중복이면 더 독립적으로 완결되고 판단 기능이 선명한 하나를 keep_id로 고른다.
6. 품질·사실성 재검수나 문장 교정은 하지 않는다. JSON 객체 하나만 출력한다.

[출력 계약]
{json.dumps(contract, ensure_ascii=False, indent=2)}"""


def command_build(args: argparse.Namespace) -> int:
    references = read_jsonl(args.reference)
    candidates = read_jsonl(args.candidates)
    reference_by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        reference_by_material[str(row["material_id"])].append(row)
    for row in candidates:
        candidate_by_material[str(row["material_id"])].append(row)
    requests = []
    for material_id, rows in sorted(candidate_by_material.items()):
        protected = reference_by_material.get(material_id, [])
        requests.append({
            "id": f"ksova_sft_semantic_dedup__{args.run_tag}__{material_id}",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict semantic redundancy auditor. "
                        "Do not rewrite or fact-check examples."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt(
                        material_id,
                        [compact(row, "protected_reference") for row in protected],
                        [compact(row, "new_candidate") for row in rows],
                    ),
                },
            ],
            "max_tokens": 4096,
            "metadata": {
                "purpose": "ksova_sft_semantic_dedup",
                "run_tag": args.run_tag,
                "material_id": material_id,
                "protected_ids": [row["example_id"] for row in protected],
                "candidate_ids": [row["example_id"] for row in rows],
                "single_round": True,
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-sft-semantic-dedup-plan-1.0",
        "protected_examples": len(references),
        "candidate_examples": len(candidates),
        "materials": len(candidate_by_material),
        "requests": len(requests),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def parse_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("response is not an object")
    return payload


def command_parse(args: argparse.Namespace) -> int:
    candidates = read_jsonl(args.candidates)
    candidate_by_id = {row["example_id"]: row for row in candidates}
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    counts = Counter(str(row.get("id") or "") for row in responses)
    response_by_id = {str(row.get("id") or ""): row for row in responses}
    kept_ids: set[str] = set()
    quarantined: dict[str, dict[str, Any]] = {}
    invalid_requests = []

    for request in requests:
        request_id = str(request["id"])
        metadata = request["metadata"]
        expected = set(metadata["candidate_ids"])
        protected = set(metadata["protected_ids"])
        response = response_by_id.get(request_id)
        invalid_reason = None
        payload: dict[str, Any] = {}
        if (
            response is None or counts[request_id] != 1 or response.get("error")
            or response.get("finish_reason") != "stop"
        ):
            invalid_reason = "missing_duplicate_runtime_or_nonstop"
        else:
            try:
                payload = parse_object(str(response.get("response") or ""))
            except (ValueError, json.JSONDecodeError) as exc:
                invalid_reason = f"parse:{exc}"

        distinct: set[str] = set()
        groups: list[dict[str, Any]] = []
        if invalid_reason is None:
            raw_distinct = payload.get("distinct_candidate_ids")
            raw_groups = payload.get("duplicate_groups")
            if payload.get("material_id") != metadata["material_id"]:
                invalid_reason = "material_id_mismatch"
            elif not isinstance(raw_distinct, list) or not all(
                isinstance(value, str) for value in raw_distinct
            ):
                invalid_reason = "invalid_distinct_ids"
            elif not isinstance(raw_groups, list):
                invalid_reason = "invalid_duplicate_groups"
            else:
                distinct = set(raw_distinct)
                if len(distinct) != len(raw_distinct) or not distinct <= expected:
                    invalid_reason = "invalid_or_duplicate_distinct_ids"
                seen_group_candidates: set[str] = set()
                for group in raw_groups:
                    if invalid_reason is not None:
                        break
                    if not isinstance(group, dict):
                        invalid_reason = "group_not_object"
                        break
                    members = group.get("member_ids")
                    keep_id = group.get("keep_id")
                    reason = group.get("reason_ko")
                    if (
                        not isinstance(members, list) or len(members) < 2
                        or not all(isinstance(value, str) for value in members)
                        or len(set(members)) != len(members)
                        or not set(members) <= expected | protected
                        or not set(members) & expected
                        or keep_id not in set(members)
                        or not isinstance(reason, str) or len(reason.strip()) < 20
                    ):
                        invalid_reason = "invalid_group_contract"
                        break
                    if set(members) & protected and keep_id not in protected:
                        invalid_reason = "protected_reference_not_kept"
                        break
                    candidate_members = set(members) & expected
                    if candidate_members & (seen_group_candidates | distinct):
                        invalid_reason = "candidate_coverage_overlap"
                        break
                    seen_group_candidates |= candidate_members
                    groups.append(group)
                if invalid_reason is None and distinct | seen_group_candidates != expected:
                    invalid_reason = "candidate_coverage_mismatch"

        if invalid_reason is not None:
            invalid_requests.append({"request_id": request_id, "reason": invalid_reason})
            for example_id in expected:
                quarantined[example_id] = {
                    "reason": "invalid_semantic_dedup_audit",
                    "request_id": request_id,
                    "detail": invalid_reason,
                }
            continue

        kept_ids |= distinct
        for group in groups:
            keep_id = group["keep_id"]
            if keep_id in expected:
                kept_ids.add(keep_id)
            for example_id in set(group["member_ids"]) & expected:
                if example_id == keep_id:
                    continue
                quarantined[example_id] = {
                    "reason": "semantic_duplicate",
                    "duplicate_of": keep_id,
                    "group": group,
                    "request_id": request_id,
                }

    covered = kept_ids | set(quarantined)
    for example_id in set(candidate_by_id) - covered:
        quarantined[example_id] = {"reason": "not_covered_by_semantic_dedup"}
    kept = [
        {
            **row,
            "semantic_dedup_gate": {
                "policy": "single_glm_group_audit_no_rewrite",
                "effective_pass": True,
            },
        }
        for example_id, row in candidate_by_id.items() if example_id in kept_ids
    ]
    quarantine_rows = [
        {
            **candidate_by_id[example_id],
            "status": "sft_quarantined_semantic_duplicate",
            "semantic_dedup_gate": {
                "policy": "single_glm_group_audit_no_rewrite",
                "effective_pass": False,
                **reason,
            },
        }
        for example_id, reason in quarantined.items()
    ]
    write_jsonl(args.kept, kept)
    write_jsonl(args.quarantine, quarantine_rows)
    report = {
        "schema_version": "ksova-sft-semantic-dedup-report-1.0",
        "candidate_examples": len(candidates),
        "kept": len(kept),
        "quarantined": len(quarantine_rows),
        "invalid_requests": invalid_requests,
        "quarantine_reasons": dict(sorted(Counter(
            row["semantic_dedup_gate"]["reason"] for row in quarantine_rows
        ).items())),
        "kept_output": str(args.kept),
        "quarantine_output": str(args.quarantine),
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "invalid_requests"}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--reference", type=Path, required=True)
    build.add_argument("--candidates", type=Path, required=True)
    build.add_argument("--run-tag", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.set_defaults(func=command_build)
    parse = commands.add_parser("parse")
    parse.add_argument("--candidates", type=Path, required=True)
    parse.add_argument("--requests", type=Path, required=True)
    parse.add_argument("--responses", type=Path, required=True)
    parse.add_argument("--kept", type=Path, required=True)
    parse.add_argument("--quarantine", type=Path, required=True)
    parse.add_argument("--report", type=Path, required=True)
    parse.set_defaults(func=command_parse)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
