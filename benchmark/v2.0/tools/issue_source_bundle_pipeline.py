#!/usr/bin/env python3
"""Build and validate body-grounded source bundles for K-SOVA issue candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from training_source_expansion import normalized_url


CHECKS = (
    "same_atomic_decision_object",
    "external_competing_frame_body_documented",
    "concrete_korean_stake_body_documented",
    "korean_public_discourse_body_documented",
    "korean_counterargument_or_limits_body_documented",
    "shared_fact_or_legal_anchor_body_documented",
    "source_attribution_roles_clear",
    "not_simple_fact_or_domestic_only",
)

SOURCE_ROLES = {
    "external_competing_position",
    "korean_position_or_interest_argument",
    "korean_counterargument_or_limits",
    "shared_fact_or_legal_anchor",
    "irrelevant",
}


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
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
        raise ValueError("response is not an object")
    return payload


def source(packet: dict[str, Any]) -> dict[str, Any]:
    value = packet.get("source")
    return value if isinstance(value, dict) else packet


def retrieval(packet: dict[str, Any]) -> dict[str, Any]:
    value = source(packet).get("retrieval")
    return value if isinstance(value, dict) else {}


def final_url(packet: dict[str, Any]) -> str:
    item = source(packet)
    return str(retrieval(packet).get("final_url") or item.get("url") or "")


def host(packet: dict[str, Any]) -> str:
    value = (urllib.parse.urlsplit(final_url(packet)).hostname or "").casefold()
    return value[4:] if value.startswith("www.") else value


def source_id(packet: dict[str, Any]) -> str:
    item = source(packet)
    return str(item.get("source_id") or packet.get("source_lead_id") or "")


def source_class(packet: dict[str, Any]) -> str:
    item = source(packet)
    jurisdiction = str(item.get("jurisdiction") or "")
    authority = str(item.get("authority_type") or "")
    tier = str(item.get("reliability_tier") or "")
    if authority == "domestic_public_discourse":
        return "korean_discourse"
    if jurisdiction == "KR" and tier == "A":
        return "korean_official_or_legal"
    if jurisdiction == "KR":
        return "korean_secondary"
    if tier == "A":
        return "external_official_or_legal"
    return "external_secondary"


def benchmark_urls(registry_path: Path, issues_path: Path) -> set[str]:
    registry = {row["source_id"]: row for row in read_jsonl(registry_path)}
    used_ids = {
        source_value
        for issue in read_jsonl(issues_path)
        for source_value in issue.get("source_ids") or []
    }
    return {
        normalized_url(str(registry[value].get("url") or ""))
        for value in used_ids if value in registry and registry[value].get("url")
    }


def packet_issue_ids(packet: dict[str, Any]) -> set[str]:
    return {
        str(row.get("candidate_id") or "")
        for row in packet.get("discoveries") or []
        if row.get("candidate_id")
    }


def packet_relevance(
    packet: dict[str, Any], issue_id: str, source_role: str | None = None
) -> float:
    values = []
    for discovery in packet.get("discoveries") or []:
        if str(discovery.get("candidate_id") or "") != issue_id:
            continue
        if source_role is not None and str(discovery.get("source_role") or "") != source_role:
            continue
        score = discovery.get("retrieval_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            values.append(float(score))
            continue
        rank = discovery.get("rank")
        if isinstance(rank, int) and rank > 0:
            values.append(max(0.0, 20.0 - rank))
    return max(values, default=0.0)


def choose_sources(rows: list[dict[str, Any]], issue_id: str, limit: int) -> list[dict[str, Any]]:
    # Exact URL duplicates are interchangeable for this stage.  Prefer longer
    # fetched text, then round-robin across evidence ecosystems.
    by_url: dict[str, dict[str, Any]] = {}

    def preference(row: dict[str, Any]) -> tuple[int, int]:
        category = source_class(row)
        policy_priority = (
            3 if category in {"korean_official_or_legal", "external_official_or_legal"}
            else 2 if category == "korean_discourse"
            else 1
        )
        return len(str(retrieval(row).get("text_excerpt") or "")), policy_priority

    for row in rows:
        url = normalized_url(final_url(row))
        if not url:
            continue
        prior = by_url.get(url)
        if prior is None or preference(row) > preference(prior):
            by_url[url] = row
    selected = []
    selected_ids = set()
    seen_hosts = set()

    # If deterministic retrieval supplied separate role rankings, take at most
    # one strong candidate from each role before the ecosystem-diversity fill.
    # The role is still only a retrieval hint: the GLM body audit below must
    # independently confirm what the document actually supports.
    role_order = (
        "external_position", "korean_position",
        "korean_counterargument_or_critique", "common_fact_or_legal_anchor",
    )
    class_order_by_role = {
        "external_position": (
            "external_official_or_legal", "external_secondary",
            "korean_discourse", "korean_official_or_legal", "korean_secondary",
        ),
        "korean_position": (
            "korean_official_or_legal", "korean_discourse", "korean_secondary",
            "external_official_or_legal", "external_secondary",
        ),
        "korean_counterargument_or_critique": (
            "korean_discourse", "korean_secondary", "korean_official_or_legal",
            "external_secondary", "external_official_or_legal",
        ),
        "common_fact_or_legal_anchor": (
            "external_official_or_legal", "korean_official_or_legal",
            "external_secondary", "korean_discourse", "korean_secondary",
        ),
    }
    for role in role_order:
        class_order = class_order_by_role[role]
        candidates = [
            row for row in by_url.values()
            if packet_relevance(row, issue_id, role) > 0
            and source_id(row) not in selected_ids
        ]
        candidates.sort(key=lambda row: (
            class_order.index(source_class(row)),
            -packet_relevance(row, issue_id, role),
            0 if str(source(row).get("reliability_tier")) == "A" else 1,
            -len(str(retrieval(row).get("text_excerpt") or "")),
            source_id(row),
        ))
        if not candidates:
            continue
        unseen_host = next((row for row in candidates if host(row) not in seen_hosts), None)
        chosen = unseen_host or candidates[0]
        selected.append(chosen)
        selected_ids.add(source_id(chosen))
        seen_hosts.add(host(chosen))
        if len(selected) >= limit:
            return selected

    buckets: dict[str, deque[dict[str, Any]]] = {}
    order = (
        "external_official_or_legal", "korean_discourse",
        "korean_official_or_legal", "external_secondary", "korean_secondary",
    )
    for category in order:
        values = [
            row for row in by_url.values()
            if source_class(row) == category and source_id(row) not in selected_ids
        ]
        values.sort(key=lambda row: (
            -packet_relevance(row, issue_id),
            0 if str(source(row).get("reliability_tier")) == "A" else 1,
            -len(str(retrieval(row).get("text_excerpt") or "")),
            source_id(row),
        ))
        buckets[category] = deque(values)
    while len(selected) < limit and any(buckets.values()):
        made_progress = False
        for category in order:
            while buckets[category]:
                row = buckets[category].popleft()
                if host(row) in seen_hosts and any(
                    host(candidate) not in seen_hosts for values in buckets.values() for candidate in values
                ):
                    continue
                selected.append(row)
                seen_hosts.add(host(row))
                made_progress = True
                break
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    return selected


def compact_source(packet: dict[str, Any], excerpt_chars: int) -> dict[str, Any]:
    item = source(packet)
    fetched = retrieval(packet)
    return {
        "source_id": source_id(packet),
        "title": str(item.get("title") or "")[:300],
        "url": final_url(packet),
        "host": host(packet),
        "jurisdiction": item.get("jurisdiction"),
        "authority_type": item.get("authority_type"),
        "reliability_tier": item.get("reliability_tier"),
        "selection_class": source_class(packet),
        "body_excerpt": str(fetched.get("text_excerpt") or "")[:excerpt_chars],
    }


def validation_prompt(issue: dict[str, Any], packets: list[dict[str, Any]], excerpt_chars: int) -> str:
    source_payload = json.dumps(
        [compact_source(row, excerpt_chars) for row in packets], ensure_ascii=False
    )
    checks = ",".join(f'"{key}":true' for key in CHECKS)
    return f"""다음 K-SOVA 후보와 실제로 수집된 문서 본문을 검증하라.

후보:
{json.dumps({key: issue.get(key) for key in ('issue_id','lane','issue_ko','external_actor_or_frame','competing_frame_ko','korean_stake_ko','decision_object_ko','korean_discourse_needed_ko')}, ensure_ascii=False)}

문서:
{source_payload}

원칙:
- 제목이나 검색어가 아니라 body_excerpt에 실제로 적힌 내용만 근거로 삼는다.
- 정부 공식 입장은 여러 논거 중 하나이지 자동 정답이 아니다.
- 외부 경쟁 관점, 한국의 구체적 이해관계, 한국 사회의 논거와 그 한계·반론, 공통 사실/법적 앵커가 같은 판단 대상을 다루는지 본다.
- 한 문서가 여러 역할을 할 수 있지만, body_excerpt가 그 역할을 실제로 뒷받침해야 한다.
- complete는 모든 checks가 true이고 missing_roles가 비어 있을 때만 가능하다.
- 자료가 더 필요하면 needs_source, 후보 자체가 K-SOVA가 아니거나 서로 다른 판단 대상을 억지로 묶었으면 reject다.

반드시 JSON 객체만 출력하라:
{{
  "issue_id":"{issue['issue_id']}",
  "source_assessments":[{{
    "source_id":"제공된 source_id",
    "usable":true,
    "roles":["external_competing_position|korean_position_or_interest_argument|korean_counterargument_or_limits|shared_fact_or_legal_anchor|irrelevant"],
    "body_supported_claims_ko":["본문에서 확인되는 주장"],
    "attribution_actor":"이 문서의 실제 발화·판단 주체",
    "reason_ko":"판정 이유"
  }}],
  "checks":{{{checks}}},
  "verdict":"complete|needs_source|reject",
  "quality_score":0,
  "missing_roles":["부족한 역할"],
  "verified_issue_ko":"본문으로 확인된 범위만 반영한 이슈",
  "verified_external_frame_ko":"확인된 외부 경쟁 관점",
  "verified_korean_stake_ko":"확인된 한국의 이해관계",
  "verified_korean_discourse_ko":["확인된 국내 논거·반론·한계"],
  "reason_ko":"최종 판정 이유"
}}"""


def command_build_validation(args: argparse.Namespace) -> int:
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
            text = str(retrieval(packet).get("text_excerpt") or "")
            if len(text) < args.min_body_chars:
                continue
            for issue_id in packet_issue_ids(packet):
                if issue_id in issue_by_id:
                    by_issue[issue_id].append(packet)
    requests = []
    selected_sources = {}
    for issue in queue:
        issue_id = issue["issue_id"]
        selected = choose_sources(
            by_issue.get(issue_id, []), issue_id, args.max_sources_per_issue
        )
        selected_sources[issue_id] = [source_id(row) for row in selected]
        if not selected:
            continue
        requests.append({
            "id": f"ksova_issue_source_validate__{issue_id}",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You validate source bodies conservatively. Search results and model memory "
                        "are not evidence. Never fill a missing evidence role by inference."
                    ),
                },
                {"role": "user", "content": validation_prompt(issue, selected, args.excerpt_chars)},
            ],
            "max_tokens": 4096,
            "metadata": {
                "purpose": "ksova_issue_body_source_bundle_validation",
                "issue_id": issue_id,
                "source_ids": [source_id(row) for row in selected],
                "source_urls": [normalized_url(final_url(row)) for row in selected],
                "benchmark_source_collision": False,
                "single_round": True,
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-issue-source-validation-plan-1.0",
        "verification_queue": len(queue),
        "input_resolved_packets": input_packets,
        "benchmark_url_collision_rejects": collision_rejects,
        "issues_with_any_source": len(requests),
        "issues_without_source": len(queue) - len(requests),
        "requests": len(requests),
        "selected_source_count": sum(len(value) for value in selected_sources.values()),
        "selected_sources_by_issue": selected_sources,
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "selected_sources_by_issue"}, ensure_ascii=False))
    return 0


def string_list(value: Any, minimum: int = 0, maximum: int = 20) -> bool:
    return (
        isinstance(value, list)
        and minimum <= len(value) <= maximum
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def payload_errors(payload: dict[str, Any], issue_id: str, source_ids: set[str]) -> list[str]:
    errors = []
    if payload.get("issue_id") != issue_id:
        errors.append("issue_id_mismatch")
    assessments = payload.get("source_assessments")
    if not isinstance(assessments, list) or len(assessments) != len(source_ids):
        errors.append("invalid_source_assessments")
    else:
        assessed_ids = {str(row.get("source_id") or "") for row in assessments if isinstance(row, dict)}
        if assessed_ids != source_ids:
            errors.append("source_assessment_ids_mismatch")
        for row in assessments:
            if not isinstance(row, dict) or not isinstance(row.get("usable"), bool):
                errors.append("invalid_source_assessment")
                continue
            roles = row.get("roles")
            if not string_list(roles, 1, 5) or not set(roles) <= SOURCE_ROLES:
                errors.append("invalid_source_roles")
            if not string_list(row.get("body_supported_claims_ko"), 0, 10):
                errors.append("invalid_body_claims")
            for field in ("attribution_actor", "reason_ko"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    errors.append(f"invalid_{field}")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECKS) or any(
        not isinstance(checks.get(key), bool) for key in CHECKS
    ):
        errors.append("invalid_checks")
    if payload.get("verdict") not in {"complete", "needs_source", "reject"}:
        errors.append("invalid_verdict")
    score = payload.get("quality_score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not 0 <= score <= 100
    ):
        errors.append("invalid_quality_score")
    if not string_list(payload.get("missing_roles"), 0, 12):
        errors.append("invalid_missing_roles")
    for field in (
        "verified_issue_ko", "verified_external_frame_ko",
        "verified_korean_stake_ko", "reason_ko",
    ):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            errors.append(f"invalid_{field}")
    if not string_list(payload.get("verified_korean_discourse_ko"), 0, 12):
        errors.append("invalid_verified_korean_discourse")
    if payload.get("verdict") == "complete" and (
        not isinstance(checks, dict) or not all(checks.get(key) is True for key in CHECKS)
        or bool(payload.get("missing_roles"))
    ):
        errors.append("complete_without_all_gates")
    return errors


def command_parse_validation(args: argparse.Namespace) -> int:
    queue = read_jsonl(args.verification_queue)
    issue_by_id = {row["issue_id"]: row for row in queue}
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    counts = Counter(str(row.get("id") or "") for row in responses)
    by_id = {str(row.get("id") or ""): row for row in responses}
    judged = []
    invalid = []
    requested_issue_ids = set()
    for request in requests:
        request_id = str(request["id"])
        issue_id = str(request["metadata"]["issue_id"])
        requested_issue_ids.add(issue_id)
        response = by_id.get(request_id)
        if response is None or counts[request_id] != 1 or response.get("error"):
            invalid.append({"issue_id": issue_id, "reason": "missing_duplicate_or_error"})
            continue
        try:
            payload = parse_object(str(response.get("response") or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            invalid.append({"issue_id": issue_id, "reason": f"parse:{exc}"})
            continue
        errors = payload_errors(payload, issue_id, set(request["metadata"]["source_ids"]))
        if errors:
            invalid.append({"issue_id": issue_id, "reason": ",".join(errors)})
            continue
        raw_score = payload["quality_score"]
        # The frozen JSON contract named the field but did not state whether a
        # fractional or percentage scale was required.  Normalize the two
        # standard numeric encodings uniformly; this changes no verdict/check.
        fractional_score = isinstance(raw_score, float) and raw_score <= 1.0
        normalized_score = round(raw_score * 100) if fractional_score else round(raw_score)
        effective_complete = (
            payload["verdict"] == "complete"
            and all(payload["checks"].values())
            and not payload["missing_roles"]
            and normalized_score >= args.complete_threshold
        )
        effective_needs_source = (
            payload["verdict"] == "needs_source"
            and bool(payload["missing_roles"])
            and normalized_score >= args.needs_source_threshold
        )
        judged.append({
            **issue_by_id[issue_id],
            "source_bundle_validation": {
                **payload,
                "quality_score_raw": raw_score,
                "quality_score_normalized_0_100": normalized_score,
                "quality_score_normalization": (
                    "fraction_0_1_times_100" if fractional_score else "percent_0_100"
                ),
                "effective_complete": effective_complete,
                "effective_needs_source": effective_needs_source,
                "validator_model": response.get("model"),
                "messages_sha256": response.get("messages_sha256"),
            },
            "source_ids": request["metadata"]["source_ids"],
            "source_urls": request["metadata"]["source_urls"],
            "training_eligible": effective_complete,
            "status": (
                "body_source_bundle_complete" if effective_complete
                else "body_source_bundle_needs_source" if effective_needs_source
                else "body_source_bundle_rejected"
            ),
        })
    # Issues with no fetched source are explicitly retained as needs-source, not
    # silently lost from the queue.
    for issue_id, issue in issue_by_id.items():
        if issue_id not in requested_issue_ids:
            judged.append({
                **issue,
                "source_bundle_validation": None,
                "source_ids": [],
                "source_urls": [],
                "training_eligible": False,
                "status": "body_source_bundle_no_sources",
            })
    complete = [row for row in judged if row["training_eligible"]]
    needs = [row for row in judged if row["status"] in {
        "body_source_bundle_needs_source", "body_source_bundle_no_sources"
    }]
    rejected = [row for row in judged if row not in complete and row not in needs]
    write_jsonl(args.output, judged)
    write_jsonl(args.complete, complete)
    write_jsonl(args.needs_source, needs)
    write_jsonl(args.rejected, rejected)
    source_audit = {
        "schema_version": "ksova-issue-source-link-audit-1.0",
        "policy": "single_glm_body_bundle_validation_no_correction",
        "accepted_candidate_sources": {
            row["issue_id"]: {
                "source_ids": row["source_ids"],
                "source_urls": row["source_urls"],
                "validation_status": row["status"],
            }
            for row in complete
        },
    }
    write_json(args.source_audit, source_audit)
    report = {
        "schema_version": "ksova-issue-source-validation-report-1.0",
        "queue": len(queue),
        "requests": len(requests),
        "responses": len(responses),
        "judged_or_no_source": len(judged),
        "invalid": len(invalid),
        "effective_complete": len(complete),
        "effective_needs_source_or_no_source": len(needs),
        "effective_reject": len(rejected),
        "status_counts": dict(sorted(Counter(row["status"] for row in judged).items())),
        "invalid_details": invalid,
        "output": str(args.output),
        "complete_output": str(args.complete),
        "needs_source_output": str(args.needs_source),
        "rejected_output": str(args.rejected),
        "source_audit_output": str(args.source_audit),
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "invalid_details"}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-validation")
    build.add_argument("--verification-queue", type=Path, required=True)
    build.add_argument("--resolved", type=Path, action="append", required=True)
    build.add_argument("--registry", type=Path, required=True)
    build.add_argument("--benchmark-issues", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.add_argument("--max-sources-per-issue", type=int, default=4)
    build.add_argument("--min-body-chars", type=int, default=500)
    build.add_argument("--excerpt-chars", type=int, default=2200)
    build.set_defaults(func=command_build_validation)

    parse = commands.add_parser("parse-validation")
    parse.add_argument("--verification-queue", type=Path, required=True)
    parse.add_argument("--requests", type=Path, required=True)
    parse.add_argument("--responses", type=Path, required=True)
    parse.add_argument("--output", type=Path, required=True)
    parse.add_argument("--complete", type=Path, required=True)
    parse.add_argument("--needs-source", type=Path, required=True)
    parse.add_argument("--rejected", type=Path, required=True)
    parse.add_argument("--source-audit", type=Path, required=True)
    parse.add_argument("--report", type=Path, required=True)
    parse.add_argument("--complete-threshold", type=int, default=85)
    parse.add_argument("--needs-source-threshold", type=int, default=70)
    parse.set_defaults(func=command_parse_validation)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
