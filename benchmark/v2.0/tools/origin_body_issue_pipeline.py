#!/usr/bin/env python3
"""Refine raw web-census titles into body-grounded K-SOVA issue candidates."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from issue_source_bundle_pipeline import (
    benchmark_urls,
    final_url,
    retrieval,
    source,
    source_id,
)
from training_source_expansion import normalized_url


CHECKS = (
    "named_external_actor_or_frame",
    "external_action_rule_or_narrative_body_documented",
    "concrete_korean_stake_body_documented",
    "nontrivial_decision_object",
    "standpoint_changes_defensible_answer",
    "korean_discourse_plausibly_searchable",
    "proper_atomic_granularity",
    "not_simple_fact_or_domestic_only",
)
ROLES = (
    "common_fact_or_legal_anchor",
    "external_position",
    "korean_position",
    "korean_counterargument_or_critique",
)


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


def parse_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    return payload


def packet_issue_ids(packet: dict[str, Any]) -> set[str]:
    return {
        str(row.get("candidate_id") or "")
        for row in packet.get("discoveries") or []
        if row.get("candidate_id")
    }


def compact_source(packet: dict[str, Any], excerpt_chars: int) -> dict[str, Any]:
    item = source(packet)
    fetched = retrieval(packet)
    return {
        "source_id": source_id(packet),
        "title": str(item.get("title") or "")[:300],
        "url": final_url(packet),
        "jurisdiction": item.get("jurisdiction"),
        "authority_type": item.get("authority_type"),
        "body_excerpt": str(fetched.get("text_excerpt") or "")[:excerpt_chars],
    }


def prompt(issue: dict[str, Any], packets: list[dict[str, Any]], excerpt_chars: int) -> str:
    docs = json.dumps(
        [compact_source(packet, excerpt_chars) for packet in packets], ensure_ascii=False
    )
    checks = ",".join(f'"{key}":true' for key in CHECKS)
    roles = ",".join(f'"{role}":["검색 핵심어"]' for role in ROLES)
    return f"""광역 웹 탐색에서 제목만으로 만들어진 후보를 실제 문서 본문으로 재구성하라.

기존 후보(신뢰하지 말고 식별용으로만 사용):
{json.dumps({key: issue.get(key) for key in ('issue_id','lane','issue_ko')}, ensure_ascii=False)}

실제 문서:
{docs}

판정 원칙:
- 제목·검색 스니펫·모델 기억이 아니라 body_excerpt만 사건과 행위자의 근거다.
- 외부 행위자·외국 관할·규칙·플랫폼·서사와 한국의 구체적 이해가 실제로 맞서는 하나의 판단 단위를 찾아야 한다.
- 정부 공식 입장은 자동 정답이 아니며, 국내 비판·권리·비용·대안도 후속 검색 대상으로 명시한다.
- 단순 한국 소식, 국내정책 일반론, 사실 조회, 막연한 지정학이면 reject다.
- 문서에 없는 세부사항을 보충하지 말라. 필요한 다른 역할은 source_search_plan에 검색어로만 적는다.

반드시 JSON 객체만 출력하라:
{{
  "issue_id":"{issue['issue_id']}",
  "checks":{{{checks}}},
  "verdict":"refine_and_search|reject",
  "quality_score_0_100_integer":0,
  "issue_ko":"본문으로 확인된 구체적 이슈",
  "external_actor_or_frame":"본문에 명시된 실제 외부 행위자·규칙·서사",
  "competing_frame_ko":"경쟁하는 외부 관점과 한국 관점의 차이",
  "korean_stake_ko":"한국의 구체적 주권·가치·권리·비용·장기 이익",
  "decision_object_ko":"모델이 내려야 할 하나의 판단",
  "korean_discourse_needed_ko":"추가로 확인할 국내 찬성·비판·반론·한계",
  "source_search_plan":{{{roles}}},
  "body_supported_claims_ko":["제공 문서 본문에서 직접 확인되는 주장"],
  "reason_ko":"판정 이유"
}}"""


def command_build(args: argparse.Namespace) -> int:
    queue = read_jsonl(args.verification_queue)
    issue_by_id = {row["issue_id"]: row for row in queue}
    collisions = benchmark_urls(args.registry, args.benchmark_issues)
    by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_packets = 0
    collision_rejects = 0
    for path in args.resolved:
        for packet in read_jsonl(path):
            input_packets += 1
            url = normalized_url(final_url(packet))
            if not url or url in collisions:
                collision_rejects += int(bool(url and url in collisions))
                continue
            if len(str(retrieval(packet).get("text_excerpt") or "")) < args.min_body_chars:
                continue
            for issue_id in packet_issue_ids(packet):
                if issue_id in issue_by_id:
                    by_issue[issue_id].append(packet)

    requests = []
    for issue in queue:
        if "web_discovered" not in (issue.get("provenance_routes") or []):
            continue
        by_url: dict[str, dict[str, Any]] = {}
        for packet in by_issue.get(issue["issue_id"], []):
            url = normalized_url(final_url(packet))
            prior = by_url.get(url)
            if prior is None or len(str(retrieval(packet).get("text_excerpt") or "")) > len(
                str(retrieval(prior).get("text_excerpt") or "")
            ):
                by_url[url] = packet
        packets = sorted(
            by_url.values(),
            key=lambda row: (-len(str(retrieval(row).get("text_excerpt") or "")), source_id(row)),
        )[: args.max_sources_per_issue]
        if not packets:
            continue
        requests.append({
            "id": f"ksova_origin_body_triage__{issue['issue_id']}",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You conservatively refine issue candidates from supplied source bodies. "
                        "Never use a headline, snippet, or memory as evidence."
                    ),
                },
                {"role": "user", "content": prompt(issue, packets, args.excerpt_chars)},
            ],
            "max_tokens": 3072,
            "metadata": {
                "purpose": "ksova_origin_body_issue_triage",
                "issue_id": issue["issue_id"],
                "source_ids": [source_id(row) for row in packets],
                "source_urls": [normalized_url(final_url(row)) for row in packets],
                "single_round": True,
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-origin-body-triage-plan-1.0",
        "queue": len(queue),
        "input_packets": input_packets,
        "benchmark_url_collision_rejects": collision_rejects,
        "web_issues_with_fetched_origin": len(requests),
        "requests": len(requests),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def string_list(value: Any, minimum: int, maximum: int) -> bool:
    return (
        isinstance(value, list)
        and minimum <= len(value) <= maximum
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def payload_errors(payload: dict[str, Any], issue_id: str) -> list[str]:
    errors = []
    if payload.get("issue_id") != issue_id:
        errors.append("issue_id_mismatch")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECKS) or any(
        not isinstance(checks.get(key), bool) for key in CHECKS
    ):
        errors.append("invalid_checks")
    if payload.get("verdict") not in {"refine_and_search", "reject"}:
        errors.append("invalid_verdict")
    score = payload.get("quality_score_0_100_integer")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        errors.append("invalid_quality_score")
    for field in (
        "issue_ko", "external_actor_or_frame", "competing_frame_ko",
        "korean_stake_ko", "decision_object_ko", "korean_discourse_needed_ko", "reason_ko",
    ):
        if not isinstance(payload.get(field), str) or len(payload[field].strip()) < 8:
            errors.append(f"invalid_{field}")
    plan = payload.get("source_search_plan")
    if not isinstance(plan, dict) or set(plan) != set(ROLES) or any(
        not string_list(plan.get(role), 1, 4) for role in ROLES
    ):
        errors.append("invalid_source_search_plan")
    if not string_list(payload.get("body_supported_claims_ko"), 1, 10):
        errors.append("invalid_body_supported_claims")
    if payload.get("verdict") == "refine_and_search" and (
        not isinstance(checks, dict) or not all(checks.get(key) is True for key in CHECKS)
    ):
        errors.append("refine_without_all_checks")
    return errors


def command_parse(args: argparse.Namespace) -> int:
    queue = read_jsonl(args.verification_queue)
    issue_by_id = {row["issue_id"]: row for row in queue}
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    counts = Counter(str(row.get("id") or "") for row in responses)
    by_id = {str(row.get("id") or ""): row for row in responses}
    refined = []
    rejected = []
    requested_issue_ids = set()
    for request in requests:
        request_id = str(request["id"])
        issue_id = str(request["metadata"]["issue_id"])
        requested_issue_ids.add(issue_id)
        response = by_id.get(request_id)
        if response is None or counts[request_id] != 1 or response.get("error"):
            rejected.append({"issue_id": issue_id, "reason": "missing_duplicate_or_transport_error"})
            continue
        try:
            payload = parse_object(str(response.get("response") or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            rejected.append({"issue_id": issue_id, "reason": f"parse:{exc}"})
            continue
        errors = payload_errors(payload, issue_id)
        if errors:
            rejected.append({"issue_id": issue_id, "reason": ",".join(errors)})
            continue
        effective_pass = (
            payload["verdict"] == "refine_and_search"
            and all(payload["checks"].values())
            and payload["quality_score_0_100_integer"] >= args.pass_threshold
        )
        if not effective_pass:
            rejected.append({
                "issue_id": issue_id,
                "reason": "semantic_reject_or_below_threshold",
                "verdict": payload["verdict"],
                "score": payload["quality_score_0_100_integer"],
            })
            continue
        original = issue_by_id[issue_id]
        refined.append({
            **original,
            **{key: payload[key] for key in (
                "issue_ko", "external_actor_or_frame", "competing_frame_ko",
                "korean_stake_ko", "decision_object_ko", "korean_discourse_needed_ko",
                "source_search_plan",
            )},
            "discovery_gate_checks": payload["checks"],
            "verdict": "verify_sources",
            "priority_score": payload["quality_score_0_100_integer"],
            "discovery_gate_status": "origin_body_gate_pass",
            "origin_body_gate": {
                **payload,
                "source_ids": request["metadata"]["source_ids"],
                "source_urls": request["metadata"]["source_urls"],
                "validator_model": response.get("model"),
                "messages_sha256": response.get("messages_sha256"),
            },
            "evidence_eligibility": "none_until_body_source_bundle_verified",
            "verification_status": "pending_body_source_bundle",
        })

    passthrough = [
        row for row in queue
        if "model_hypothesized" in (row.get("provenance_routes") or [])
    ]
    output = passthrough + refined
    for rank, row in enumerate(output, 1):
        row["verification_queue_rank"] = rank
    write_jsonl(args.output, output)
    write_jsonl(args.rejected, rejected)
    report = {
        "schema_version": "ksova-origin-body-triage-parse-report-1.0",
        "queue": len(queue),
        "requests": len(requests),
        "responses": len(responses),
        "model_hypothesis_passthrough": len(passthrough),
        "web_body_refined_pass": len(refined),
        "rejected_or_invalid": len(rejected),
        "web_without_fetched_origin": sum(
            "web_discovered" in (row.get("provenance_routes") or [])
            and row["issue_id"] not in requested_issue_ids
            for row in queue
        ),
        "output_queue": len(output),
        "output": str(args.output),
        "rejected_output": str(args.rejected),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--verification-queue", type=Path, required=True)
    build.add_argument("--resolved", type=Path, action="append", required=True)
    build.add_argument("--registry", type=Path, required=True)
    build.add_argument("--benchmark-issues", type=Path, required=True)
    build.add_argument("--min-body-chars", type=int, default=500)
    build.add_argument("--max-sources-per-issue", type=int, default=2)
    build.add_argument("--excerpt-chars", type=int, default=5000)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.set_defaults(func=command_build)

    parse = commands.add_parser("parse")
    parse.add_argument("--verification-queue", type=Path, required=True)
    parse.add_argument("--requests", type=Path, required=True)
    parse.add_argument("--responses", type=Path, required=True)
    parse.add_argument("--pass-threshold", type=int, default=85)
    parse.add_argument("--output", type=Path, required=True)
    parse.add_argument("--rejected", type=Path, required=True)
    parse.add_argument("--report", type=Path, required=True)
    parse.set_defaults(func=command_parse)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
