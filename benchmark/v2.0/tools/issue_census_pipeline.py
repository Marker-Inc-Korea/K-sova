#!/usr/bin/env python3
"""Fail-closed processing for the provenance-separated K-SOVA issue census."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


LANES = {
    "territory_maritime_airspace",
    "security_alliance_defense",
    "trade_industry_supply_chain",
    "technology_data_science",
    "finance_currency_tax",
    "energy_environment_resources",
    "health_bio_food",
    "migration_citizenship_labor",
    "democracy_rights_values",
    "history_responsibility_memory",
    "culture_language_identity",
    "international_law_order_development",
    "north_korea_interkorean",
}

HYPOTHESIS_FIELDS = (
    "candidate_key",
    "issue_ko",
    "external_actor_or_frame",
    "competing_frame_ko",
    "korean_stake_ko",
    "decision_object_ko",
    "korean_discourse_hypothesis_ko",
    "temporal_scope",
    "confidence",
)

LEAD_FIELDS = (
    "candidate_key",
    "lane",
    "issue_ko",
    "event_or_policy_ko",
    "named_external_actor_or_frame",
    "korean_stake_hypothesis_ko",
    "decision_object_hypothesis_ko",
    "why_possible_ksova_ko",
    "triage",
)

GATE_CHECKS = (
    "real_external_competing_frame",
    "external_frame_nonhypothetical",
    "concrete_korean_stake",
    "nontrivial_decision",
    "standpoint_changes_defensible_answer",
    "korean_discourse_searchable",
    "proper_granularity",
)

SOURCE_ROLES = (
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def stable_id(prefix: str, *values: Any) -> str:
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:16]


def nonempty_strings(value: Any, minimum: int = 1, maximum: int = 20) -> bool:
    return (
        isinstance(value, list)
        and minimum <= len(value) <= maximum
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def response_lookup(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter(str(row.get("id") or "") for row in rows)
    return {str(row.get("id") or ""): row for row in rows}, counts


def hypothesis_errors(issue: Any) -> list[str]:
    if not isinstance(issue, dict):
        return ["not_object"]
    errors = []
    for field in HYPOTHESIS_FIELDS:
        if not isinstance(issue.get(field), str) or not issue[field].strip():
            errors.append(f"invalid_{field}")
    for field in ("issue_ko", "competing_frame_ko", "korean_stake_ko", "decision_object_ko"):
        if isinstance(issue.get(field), str) and len(issue[field].strip()) < 12:
            errors.append(f"short_{field}")
    if issue.get("confidence") not in {"high", "medium", "low"}:
        errors.append("invalid_confidence")
    for field in ("search_anchors_ko", "search_anchors_external"):
        if not nonempty_strings(issue.get(field), 2, 6):
            errors.append(f"invalid_{field}")
    return errors


def command_parse_hypotheses(args: argparse.Namespace) -> int:
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    by_id, counts = response_lookup(responses)
    output: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request["id"])
        lane = str(request["metadata"]["lane"])
        response = by_id.get(request_id)
        if response is None or counts[request_id] != 1 or response.get("error"):
            invalid.append({"request_id": request_id, "reason": "missing_duplicate_or_error"})
            continue
        try:
            payload = parse_object(str(response.get("response") or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            invalid.append({"request_id": request_id, "reason": f"parse:{exc}"})
            continue
        if payload.get("lane") != lane or not isinstance(payload.get("issues"), list):
            invalid.append({"request_id": request_id, "reason": "lane_or_issues_mismatch"})
            continue
        valid_in_response = 0
        for index, issue in enumerate(payload["issues"]):
            errors = hypothesis_errors(issue)
            if errors:
                invalid.append({
                    "request_id": request_id,
                    "item_index": index,
                    "reason": ",".join(errors),
                })
                continue
            candidate_id = stable_id(
                "KSHYP-", lane, issue["issue_ko"],
                issue["external_actor_or_frame"], issue["decision_object_ko"],
            )
            output.append({
                "schema_version": "ksova-issue-census-candidate-1.0",
                "candidate_id": candidate_id,
                "lane": lane,
                **{field: issue[field] for field in HYPOTHESIS_FIELDS},
                "search_anchors_ko": issue["search_anchors_ko"],
                "search_anchors_external": issue["search_anchors_external"],
                "provenance_route": "model_hypothesized",
                "provenance_request_id": request_id,
                "evidence_eligibility": "none_until_body_verified",
                "triage_status": "unverified_hypothesis",
            })
            valid_in_response += 1
        if valid_in_response == 0:
            invalid.append({"request_id": request_id, "reason": "no_valid_issues"})

    deduped = {row["candidate_id"]: row for row in output}
    final = sorted(deduped.values(), key=lambda row: (row["lane"], row["candidate_id"]))
    write_jsonl(args.output, final)
    report = {
        "schema_version": "ksova-issue-census-hypothesis-parse-report-1.0",
        "requests": len(requests),
        "responses": len(responses),
        "valid_candidates_before_exact_dedup": len(output),
        "valid_candidates": len(final),
        "invalid": len(invalid),
        "by_lane": dict(sorted(Counter(row["lane"] for row in final).items())),
        "by_confidence": dict(sorted(Counter(row["confidence"] for row in final).items())),
        "invalid_details": invalid,
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "invalid_details"}, ensure_ascii=False))
    return 0


def lead_cohort(row: dict[str, Any]) -> str:
    variants = [
        str(item.get("source_variant") or "")
        for item in row.get("discoveries") or []
    ]
    for cohort in ("broad", "domestic", "enduring", "external"):
        if any(f"web_census_{cohort}" in value for value in variants):
            return cohort
    return "other"


def lead_lane(row: dict[str, Any]) -> str:
    counts = Counter(
        str(item.get("lane") or "") for item in row.get("discoveries") or []
        if item.get("lane")
    )
    return counts.most_common(1)[0][0] if counts else str(row.get("lane") or "cross_lane")


def lead_rank(row: dict[str, Any]) -> int:
    ranks = [
        int(item["rank"]) for item in row.get("discoveries") or []
        if str(item.get("rank") or "").isdigit()
    ]
    return min(ranks) if ranks else 999


def diverse_leads(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], deque[dict[str, Any]]] = {}
    for row in rows:
        key = (lead_lane(row), lead_cohort(row))
        groups.setdefault(key, deque()).append(row)
    for key, values in groups.items():
        groups[key] = deque(sorted(values, key=lambda row: (
            lead_rank(row), -len(row.get("discoveries") or []), str(row.get("lead_id") or "")
        )))
    keys = sorted(groups)
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < limit:
        remaining = []
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].popleft())
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return selected


def compact_lead(row: dict[str, Any]) -> dict[str, Any]:
    discoveries = row.get("discoveries") or []
    return {
        "lead_id": row["lead_id"],
        "title": str(row.get("title") or "")[:300],
        "snippet": str(row.get("snippet") or "")[:700],
        "host": row.get("host"),
        "published_at": row.get("published_at"),
        "lane_hint": lead_lane(row),
        "cohort": lead_cohort(row),
        "queries": list(dict.fromkeys(
            str(item.get("query") or "") for item in discoveries if item.get("query")
        ))[:3],
    }


def extraction_prompt(batch_id: str, leads: list[dict[str, Any]]) -> str:
    payload = json.dumps([compact_lead(row) for row in leads], ensure_ascii=False)
    lanes = "|".join(sorted(LANES))
    return f"""다음은 광역 검색에서 얻은 제목·검색 스니펫이다. 이것들은 사실 근거가 아니라 이슈 발견용 단서일 뿐이다.

배치: {batch_id}
입력 단서:
{payload}

목표:
- 같은 사건·정책을 가리키는 단서를 배치 안에서 묶는다.
- 한국의 주권·헌법적 가치·권리·장기 이익과 실제 외부 행위자·외국 관할·국제 규칙·해외 기업/플랫폼·경쟁 서사가 충돌할 가능성이 있는 후보만 추출한다.
- 제목에 없는 구체적 사실을 만들어내지 말고, 불명확하면 triage를 uncertain으로 둔다.
- 단순 국내 정책, 연예·스포츠·생활 정보, 단순 사실 조회는 제외한다.
- 이 출력 역시 증거가 아니라 후속 원문 검색을 위한 후보이다.

lane은 다음 중 하나만 사용하라: {lanes}
triage는 likely|uncertain|unlikely 중 하나다.

반드시 JSON 객체만 출력하라:
{{"batch_id":"{batch_id}","candidates":[{{
  "candidate_key":"짧은 영문 키",
  "lead_ids":["입력 lead_id"],
  "lane":"허용 lane",
  "issue_ko":"단서가 가리키는 구체적 이슈",
  "event_or_policy_ko":"확인해야 할 사건·정책·문서",
  "named_external_actor_or_frame":"단서에 보이는 실제 외부 행위자·제도·규칙·서사 또는 unknown",
  "korean_stake_hypothesis_ko":"검증할 한국의 구체적 이해관계",
  "decision_object_hypothesis_ko":"모델이 내려야 할 하나의 판단",
  "why_possible_ksova_ko":"외부 기본 답변과 한국에서 방어 가능한 답변이 달라질 수 있는 이유",
  "triage":"likely|uncertain|unlikely",
  "missing_for_verification_ko":["원문에서 확인할 요소"],
  "search_anchors_ko":["고유명사 중심 검색어"],
  "search_anchors_external":["외국어 고유명사 중심 검색어"]
}}]}}"""


def command_build_lead_extraction(args: argparse.Namespace) -> int:
    leads = read_jsonl(args.leads)
    selected = diverse_leads(leads, args.max_leads)
    requests = []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        batch_id = f"web-leads-{start // args.batch_size + 1:04d}"
        requests.append({
            "id": f"ksova_issue_lead_extract__{batch_id}",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You conservatively extract issue-discovery candidates. "
                        "Search snippets are not evidence; never fabricate missing facts."
                    ),
                },
                {"role": "user", "content": extraction_prompt(batch_id, batch)},
            ],
            "max_tokens": 4096,
            "metadata": {
                "purpose": "ksova_web_lead_issue_extraction_only",
                "batch_id": batch_id,
                "lead_ids": [row["lead_id"] for row in batch],
                "provenance_route": "web_discovered",
                "evidence_eligibility": "none_until_body_verified",
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-web-lead-extraction-plan-1.0",
        "input_leads": len(leads),
        "selected_leads": len(selected),
        "requests": len(requests),
        "batch_size": args.batch_size,
        "by_cohort": dict(sorted(Counter(lead_cohort(row) for row in selected).items())),
        "by_lane_hint": dict(sorted(Counter(lead_lane(row) for row in selected).items())),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def lead_candidate_errors(candidate: Any, allowed_leads: set[str]) -> list[str]:
    if not isinstance(candidate, dict):
        return ["not_object"]
    errors = []
    for field in LEAD_FIELDS:
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            errors.append(f"invalid_{field}")
    lead_ids = candidate.get("lead_ids")
    if not nonempty_strings(lead_ids, 1, 20) or not set(lead_ids) <= allowed_leads:
        errors.append("invalid_lead_ids")
    if candidate.get("lane") not in LANES:
        errors.append("invalid_lane")
    if candidate.get("triage") not in {"likely", "uncertain", "unlikely"}:
        errors.append("invalid_triage")
    if not nonempty_strings(candidate.get("missing_for_verification_ko"), 1, 12):
        errors.append("invalid_missing_for_verification")
    for field in ("search_anchors_ko", "search_anchors_external"):
        if not nonempty_strings(candidate.get(field), 1, 8):
            errors.append(f"invalid_{field}")
    return errors


def command_parse_lead_extraction(args: argparse.Namespace) -> int:
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    by_id, counts = response_lookup(responses)
    output: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request["id"])
        response = by_id.get(request_id)
        if response is None or counts[request_id] != 1 or response.get("error"):
            invalid.append({"request_id": request_id, "reason": "missing_duplicate_or_error"})
            continue
        try:
            payload = parse_object(str(response.get("response") or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            invalid.append({"request_id": request_id, "reason": f"parse:{exc}"})
            continue
        if payload.get("batch_id") != request["metadata"]["batch_id"] or not isinstance(payload.get("candidates"), list):
            invalid.append({"request_id": request_id, "reason": "batch_or_candidates_mismatch"})
            continue
        allowed_leads = set(request["metadata"]["lead_ids"])
        for index, candidate in enumerate(payload["candidates"]):
            errors = lead_candidate_errors(candidate, allowed_leads)
            if errors:
                invalid.append({
                    "request_id": request_id,
                    "item_index": index,
                    "reason": ",".join(errors),
                })
                continue
            candidate_id = stable_id(
                "KSLEAD-", candidate["lane"], candidate["issue_ko"],
                candidate["named_external_actor_or_frame"],
                candidate["decision_object_hypothesis_ko"],
            )
            output.append({
                "schema_version": "ksova-issue-census-candidate-1.0",
                "candidate_id": candidate_id,
                **{field: candidate[field] for field in LEAD_FIELDS},
                "lead_ids": candidate["lead_ids"],
                "missing_for_verification_ko": candidate["missing_for_verification_ko"],
                "search_anchors_ko": candidate["search_anchors_ko"],
                "search_anchors_external": candidate["search_anchors_external"],
                "provenance_route": "web_discovered",
                "provenance_request_id": request_id,
                "evidence_eligibility": "none_until_body_verified",
                "triage_status": f"web_{candidate['triage']}",
            })
    deduped: dict[str, dict[str, Any]] = {}
    for row in output:
        prior = deduped.get(row["candidate_id"])
        if prior is None:
            deduped[row["candidate_id"]] = row
        else:
            prior["lead_ids"] = sorted(set(prior["lead_ids"]) | set(row["lead_ids"]))
    final = sorted(deduped.values(), key=lambda row: (row["lane"], row["candidate_id"]))
    write_jsonl(args.output, final)
    report = {
        "schema_version": "ksova-web-lead-extraction-parse-report-1.0",
        "requests": len(requests),
        "responses": len(responses),
        "valid_candidates_before_exact_dedup": len(output),
        "valid_candidates": len(final),
        "invalid": len(invalid),
        "by_lane": dict(sorted(Counter(row["lane"] for row in final).items())),
        "by_triage": dict(sorted(Counter(row["triage"] for row in final).items())),
        "invalid_details": invalid,
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "invalid_details"}, ensure_ascii=False))
    return 0


def command_build_raw_lead_candidates(args: argparse.Namespace) -> int:
    """Preserve unclassified web leads without asking a model to invent gaps."""
    leads = read_jsonl(args.leads)
    parsed = read_jsonl(args.parsed) if args.parsed and args.parsed.exists() else []
    represented = {
        lead_id for row in parsed for lead_id in row.get("lead_ids") or []
    }
    fallback = []
    for lead in leads:
        lead_id = str(lead["lead_id"])
        if lead_id in represented:
            continue
        title = re.sub(r"\s+", " ", str(lead.get("title") or "")).strip()
        snippet = re.sub(r"\s+", " ", str(lead.get("snippet") or "")).strip()
        lane = lead_lane(lead)
        if lane not in LANES:
            # Cross-lane home/listing results do not provide enough information
            # to assign a substantive lane without semantic invention.
            continue
        candidate_id = stable_id("KSRAW-", lead_id, lane, title)
        fallback.append({
            "schema_version": "ksova-issue-census-candidate-1.0",
            "candidate_id": candidate_id,
            "candidate_key": lead_id,
            "lane": lane,
            "issue_ko": title,
            "event_or_policy_ko": snippet or title,
            "named_external_actor_or_frame": "unknown_until_body_verified",
            "korean_stake_hypothesis_ko": "검색 단서만으로는 불명확하며 원문 확인 필요",
            "decision_object_hypothesis_ko": "검색 단서가 K-SOVA 판단 대상을 포함하는지 원문 확인 필요",
            "why_possible_ksova_ko": "광역 웹 모집단에서 발견되었으나 외부 경쟁 관점과 한국 이해관계는 아직 미확인",
            "triage": "uncertain",
            "lead_ids": [lead_id],
            "missing_for_verification_ko": [
                "원문 본문", "실제 외부 경쟁 관점", "한국의 구체적 이해관계", "국내 담론",
            ],
            "search_anchors_ko": [title],
            "search_anchors_external": [title],
            "provenance_route": "web_discovered",
            "provenance_request_id": None,
            "evidence_eligibility": "none_until_body_verified",
            "triage_status": "web_unclassified_raw_lead",
        })
    output = parsed + fallback
    output.sort(key=lambda row: (row["lane"], row["candidate_id"]))
    write_jsonl(args.output, output)
    report = {
        "schema_version": "ksova-raw-web-lead-candidate-report-1.0",
        "input_leads": len(leads),
        "model_parsed_candidates": len(parsed),
        "represented_leads": len(represented),
        "raw_fallback_candidates": len(fallback),
        "output_candidates": len(output),
        "by_lane": dict(sorted(Counter(row["lane"] for row in output).items())),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    route = str(row["provenance_route"])
    if route == "model_hypothesized":
        external = row["external_actor_or_frame"]
        stake = row["korean_stake_ko"]
        decision = row["decision_object_ko"]
        competing = row["competing_frame_ko"]
        discourse = row["korean_discourse_hypothesis_ko"]
        triage = f"hypothesis_{row['confidence']}"
    else:
        external = row["named_external_actor_or_frame"]
        stake = row["korean_stake_hypothesis_ko"]
        decision = row["decision_object_hypothesis_ko"]
        competing = row["why_possible_ksova_ko"]
        discourse = "; ".join(row["missing_for_verification_ko"])
        triage = row["triage_status"]
    return {
        "candidate_id": row["candidate_id"],
        "provenance_route": route,
        "issue_ko": row["issue_ko"],
        "external_actor_or_frame": external,
        "competing_frame_ko": competing,
        "korean_stake_ko": stake,
        "decision_object_ko": decision,
        "korean_discourse_to_verify_ko": discourse,
        "triage_hint": triage,
        "search_anchors_ko": row["search_anchors_ko"],
        "search_anchors_external": row["search_anchors_external"],
    }


def balanced_candidates(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_route: dict[str, deque[dict[str, Any]]] = {}
    for route in ("web_discovered", "model_hypothesized"):
        candidates = [row for row in rows if row["provenance_route"] == route]
        candidates.sort(key=lambda row: (
            0 if row.get("triage_status") in {"web_likely", "unverified_hypothesis"} else 1,
            row["candidate_id"],
        ))
        by_route[route] = deque(candidates)
    selected = []
    routes = ["web_discovered", "model_hypothesized"]
    while len(selected) < limit and any(by_route[route] for route in routes):
        for route in routes:
            if by_route[route] and len(selected) < limit:
                selected.append(by_route[route].popleft())
    return selected


def consolidation_prompt(lane: str, candidates: list[dict[str, Any]], max_output: int) -> str:
    payload = json.dumps([compact_candidate(row) for row in candidates], ensure_ascii=False)
    gate_json = ",".join(f'"{key}":true' for key in GATE_CHECKS)
    role_json = ",".join(f'"{key}":["검색어"]' for key in SOURCE_ROLES)
    return f"""다음은 서로 독립된 두 탐색 경로에서 나온, 아직 검증되지 않은 K-SOVA 이슈 후보다.

분야: {lane}
후보:
{payload}

중요:
- web_discovered의 제목·스니펫도, model_hypothesized의 모델 기억도 증거가 아니다.
- 의미상 같은 사건·정책·판단 대상을 하나로 묶되 input_candidate_ids로 계보를 모두 보존하라.
- 최대 {max_output}개만 남긴다. 검색 인기도가 아니라 구성 적합성과 분야 내 충돌 유형·외부 행위자 다양성을 우선한다.
- 정부의 공식 입장을 곧바로 정답으로 취급하지 않는다. 국내 비판·권리·장기 이익 논거도 검증 대상으로 둔다.

verify_sources가 되려면 다음을 모두 만족할 가능성이 구체적으로 보여야 한다.
1. 실제 외부 행위자·관할·규칙·플랫폼·서사가 있다.
2. 외부 관점이 가상의 일반론이 아니라 문서로 확인 가능하다.
3. 한국의 구체적인 주권·가치·권리·비용·장기 이익이 걸려 있다.
4. 단순 사실 조회가 아니라 조건과 우선순위를 정해야 한다.
5. 외국의 기본 답변과 한국에서 방어 가능한 답변이 실질적으로 달라질 수 있다.
6. 한국 사회 안에서 확인할 논거 또는 상반된 담론이 있다.
7. 너무 작은 조항이나 막연한 지정학이 아니라 하나의 결과적인 판단 단위다.

반드시 JSON 객체만 출력하라:
{{"lane":"{lane}","candidates":[{{
  "input_candidate_ids":["입력 candidate_id"],
  "issue_ko":"통합된 구체적 이슈",
  "external_actor_or_frame":"실제 상대 행위자·관할·규칙·서사",
  "competing_frame_ko":"경쟁하는 관점의 차이",
  "korean_stake_ko":"한국의 구체적 이해관계",
  "decision_object_ko":"모델이 내려야 할 하나의 판단",
  "korean_discourse_needed_ko":"국내에서 확인할 정부·비판·대안 논거",
  "discovery_gate_checks":{{{gate_json}}},
  "verdict":"verify_sources|defer|reject",
  "priority_score":0,
  "source_search_plan":{{{role_json}}},
  "verification_risks_ko":["원문 검증 시 주의점"]
}}]}}"""


def command_build_consolidation(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.hypotheses) + read_jsonl(args.web_candidates)
    requests = []
    input_counts = Counter(row["lane"] for row in rows)
    for lane in sorted(LANES):
        lane_rows = [row for row in rows if row["lane"] == lane]
        selected = balanced_candidates(lane_rows, args.max_input_per_lane)
        if not selected:
            continue
        request_id = f"ksova_issue_consolidate__{lane}"
        requests.append({
            "id": request_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You apply a conservative discovery gate and preserve provenance. "
                        "Candidates are not evidence and must be source-verified later."
                    ),
                },
                {"role": "user", "content": consolidation_prompt(lane, selected, args.max_output_per_lane)},
            ],
            "max_tokens": 6144,
            "metadata": {
                "purpose": "ksova_issue_census_consolidation_gate",
                "lane": lane,
                "candidate_ids": [row["candidate_id"] for row in selected],
                "max_output_candidates": args.max_output_per_lane,
                "evidence_eligibility": "none_until_body_verified",
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-issue-consolidation-plan-1.0",
        "input_candidates": len(rows),
        "input_by_lane": dict(sorted(input_counts.items())),
        "requests": len(requests),
        "max_input_per_lane": args.max_input_per_lane,
        "max_output_per_lane": args.max_output_per_lane,
        "planned_input_candidates": sum(len(row["metadata"]["candidate_ids"]) for row in requests),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def consolidation_errors(candidate: Any, allowed_ids: set[str]) -> list[str]:
    if not isinstance(candidate, dict):
        return ["not_object"]
    errors = []
    for field in (
        "issue_ko", "external_actor_or_frame", "competing_frame_ko",
        "korean_stake_ko", "decision_object_ko", "korean_discourse_needed_ko",
    ):
        if not isinstance(candidate.get(field), str) or len(candidate[field].strip()) < 10:
            errors.append(f"invalid_{field}")
    ids = candidate.get("input_candidate_ids")
    if not nonempty_strings(ids, 1, 20) or not set(ids) <= allowed_ids:
        errors.append("invalid_input_candidate_ids")
    checks = candidate.get("discovery_gate_checks")
    if not isinstance(checks, dict) or set(checks) != set(GATE_CHECKS) or any(
        not isinstance(checks.get(key), bool) for key in GATE_CHECKS
    ):
        errors.append("invalid_discovery_gate_checks")
    if candidate.get("verdict") not in {"verify_sources", "defer", "reject"}:
        errors.append("invalid_verdict")
    score = candidate.get("priority_score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        errors.append("invalid_priority_score")
    plan = candidate.get("source_search_plan")
    if not isinstance(plan, dict) or set(plan) != set(SOURCE_ROLES) or any(
        not nonempty_strings(plan.get(role), 1, 4) for role in SOURCE_ROLES
    ):
        errors.append("invalid_source_search_plan")
    if not nonempty_strings(candidate.get("verification_risks_ko"), 1, 10):
        errors.append("invalid_verification_risks")
    if (
        candidate.get("verdict") == "verify_sources"
        and isinstance(checks, dict)
        and not all(checks.get(key) is True for key in GATE_CHECKS)
    ):
        errors.append("verify_sources_with_failed_gate")
    return errors


def select_balanced_verification_queue(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows
        if row["verdict"] == "verify_sources"
        and all(row["discovery_gate_checks"].values())
    ]
    groups: dict[str, deque[dict[str, Any]]] = {}
    for lane in sorted(LANES):
        lane_rows = [row for row in eligible if row["lane"] == lane]
        lane_rows.sort(key=lambda row: (
            -len(row["provenance_routes"]), -row["priority_score"], row["issue_id"]
        ))
        for index, row in enumerate(lane_rows, 1):
            row["lane_verification_rank"] = index
        groups[lane] = deque(lane_rows)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for lane in sorted(groups):
            if groups[lane] and len(selected) < limit:
                row = groups[lane].popleft()
                row["verification_queue_rank"] = len(selected) + 1
                row["verification_status"] = "pending_body_source_bundle"
                selected.append(row)
    return selected


def command_parse_consolidation(args: argparse.Namespace) -> int:
    source_rows = read_jsonl(args.hypotheses) + read_jsonl(args.web_candidates)
    source_by_id = {row["candidate_id"]: row for row in source_rows}
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    by_id, counts = response_lookup(responses)
    output: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request["id"])
        lane = str(request["metadata"]["lane"])
        response = by_id.get(request_id)
        if response is None or counts[request_id] != 1 or response.get("error"):
            invalid.append({"request_id": request_id, "reason": "missing_duplicate_or_error"})
            continue
        try:
            payload = parse_object(str(response.get("response") or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            invalid.append({"request_id": request_id, "reason": f"parse:{exc}"})
            continue
        candidates = payload.get("candidates")
        if payload.get("lane") != lane or not isinstance(candidates, list):
            invalid.append({"request_id": request_id, "reason": "lane_or_candidates_mismatch"})
            continue
        if len(candidates) > int(request["metadata"]["max_output_candidates"]):
            invalid.append({"request_id": request_id, "reason": "output_candidate_cap_exceeded"})
            continue
        allowed_ids = set(request["metadata"]["candidate_ids"])
        for index, candidate in enumerate(candidates):
            errors = consolidation_errors(candidate, allowed_ids)
            if errors:
                invalid.append({
                    "request_id": request_id,
                    "item_index": index,
                    "reason": ",".join(errors),
                })
                continue
            input_ids = candidate["input_candidate_ids"]
            routes = sorted({source_by_id[value]["provenance_route"] for value in input_ids})
            issue_id = stable_id(
                "KSISSUE-", lane, candidate["issue_ko"],
                candidate["external_actor_or_frame"], candidate["decision_object_ko"],
            )
            output.append({
                "schema_version": "ksova-issue-census-consolidated-1.0",
                "issue_id": issue_id,
                "lane": lane,
                **candidate,
                "provenance_routes": routes,
                "evidence_eligibility": "none_until_body_verified",
                "source_verification_required": True,
            })
    deduped: dict[str, dict[str, Any]] = {}
    for row in output:
        prior = deduped.get(row["issue_id"])
        if prior is None or row["priority_score"] > prior["priority_score"]:
            deduped[row["issue_id"]] = row
    final = sorted(deduped.values(), key=lambda row: (row["lane"], -row["priority_score"], row["issue_id"]))
    queue = select_balanced_verification_queue(final, args.max_verify)
    write_jsonl(args.output, final)
    write_jsonl(args.verification_queue, queue)
    report = {
        "schema_version": "ksova-issue-consolidation-parse-report-1.0",
        "requests": len(requests),
        "responses": len(responses),
        "valid_candidates": len(final),
        "invalid": len(invalid),
        "by_verdict": dict(sorted(Counter(row["verdict"] for row in final).items())),
        "verification_queue": len(queue),
        "queue_by_lane": dict(sorted(Counter(row["lane"] for row in queue).items())),
        "queue_by_provenance_routes": dict(sorted(Counter("+".join(row["provenance_routes"]) for row in queue).items())),
        "invalid_details": invalid,
        "output": str(args.output),
        "verification_queue_output": str(args.verification_queue),
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "invalid_details"}, ensure_ascii=False))
    return 0


def compact_search_query(issue: dict[str, Any], max_chars: int) -> str:
    actor = re.sub(r"\s+", " ", str(issue["external_actor_or_frame"])).strip()
    issue_text = re.sub(r"\s+", " ", str(issue["issue_ko"])).strip()
    # Retain named entities near the front.  The query is deliberately a single
    # mixed-role snowball seed; returned documents are role-classified only
    # after their bodies have been fetched.
    query = f'"{issue_text[:90]}" {actor[:55]} 한국'
    return query[:max_chars].rstrip()


def compact_role_query(issue: dict[str, Any], role: str, max_chars: int) -> str:
    plan = issue.get("source_search_plan") or {}
    role_anchors = plan.get(role) if isinstance(plan.get(role), list) else []
    anchor = re.sub(r"\s+", " ", str(role_anchors[0] if role_anchors else "")).strip()
    issue_text = re.sub(r"\s+", " ", str(issue.get("issue_ko") or "")).strip()
    actor = re.sub(
        r"\s+", " ", str(issue.get("external_actor_or_frame") or "")
    ).strip()
    stake = re.sub(r"\s+", " ", str(issue.get("korean_stake_ko") or "")).strip()
    if role == "external_position":
        query = f'{anchor[:90]} {actor[:70]} South Korea official position'
    elif role == "common_fact_or_legal_anchor":
        query = f'{anchor[:90]} {actor[:60]} rule agreement decision South Korea'
    elif role == "korean_position":
        query = f'{anchor[:75]} {issue_text[:70]} 한국 정부 산업계 입장'
    else:
        query = f'{anchor[:70]} {stake[:65]} 한국 비판 우려 반론 한계 대안'
    return query[:max_chars].rstrip()


def command_build_source_search(args: argparse.Namespace) -> int:
    queue = read_jsonl(args.verification_queue)[: args.max_issues]
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    engines = ("naver_web", "bing_web", "google_news")
    queries = []
    for index, issue in enumerate(queue):
        engine = engines[index % len(engines)]
        queries.append({
            "lane": issue["lane"],
            "language": "ko" if engine == "naver_web" else "en",
            "query": compact_search_query(issue, args.max_query_chars),
            "engines": [engine],
            "candidate_id": issue["issue_id"],
            "source_role": "mixed_bundle_discovery",
            "source_variant": f"snowball_seed_{engine}",
            "jurisdiction": "mixed",
        })
    config = {
        "schema_version": "ksova-issue-source-search-config-1.0",
        "provenance_route": "snowball_discovered",
        "evidence_eligibility": "discovery_only_until_body_verified",
        "engines": ["google_news"],
        "blocked_hosts": base.get("blocked_hosts") or [],
        "discovery_only_hosts": base.get("discovery_only_hosts") or [],
        "queries": queries,
    }
    write_json(args.output, config)
    report = {
        "schema_version": "ksova-issue-source-search-plan-1.0",
        "input_queue": len(read_jsonl(args.verification_queue)),
        "planned_issues": len(queue),
        "logical_queries": len(queries),
        "engine_requests": len(queries),
        "engines": dict(sorted(Counter(row["engines"][0] for row in queries).items())),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_build_role_source_search(args: argparse.Namespace) -> int:
    queue = read_jsonl(args.verification_queue)
    if args.provenance_route:
        queue = [
            row for row in queue
            if args.provenance_route in (row.get("provenance_routes") or [])
        ]
    if args.lane:
        selected_lanes = set(args.lane)
        queue = [row for row in queue if row.get("lane") in selected_lanes]
    queue.sort(key=lambda row: (
        -int(row.get("priority_score") or 0), row["lane"], row["issue_id"]
    ))
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    role_specs = (
        ("external_position", "bing_web", "en", "external"),
        ("common_fact_or_legal_anchor", "bing_web", "en", "mixed"),
        ("korean_position", "naver_web", "ko", "KR"),
        ("korean_counterargument_or_critique", "naver_web", "ko", "KR"),
    )
    queries = []
    selected_issues = set()
    seen = set()
    for issue in queue[: args.max_issues]:
        for role, engine, language, jurisdiction in role_specs:
            if len(queries) >= args.max_requests:
                break
            query = compact_role_query(issue, role, args.max_query_chars)
            identity = (engine, query.casefold())
            if not query or identity in seen:
                continue
            seen.add(identity)
            selected_issues.add(issue["issue_id"])
            queries.append({
                "lane": issue["lane"],
                "language": language,
                "query": query,
                "engines": [engine],
                "candidate_id": issue["issue_id"],
                "source_role": role,
                "source_variant": f"targeted_role_{role}_{engine}",
                "jurisdiction": jurisdiction,
            })
        if len(queries) >= args.max_requests:
            break
    config = {
        "schema_version": "ksova-targeted-role-source-search-config-1.0",
        "provenance_route": "snowball_discovered",
        "evidence_eligibility": "discovery_only_until_body_verified",
        "blocked_hosts": base.get("blocked_hosts") or [],
        "discovery_only_hosts": base.get("discovery_only_hosts") or [],
        "queries": queries,
    }
    write_json(args.output, config)
    report = {
        "schema_version": "ksova-targeted-role-source-search-plan-1.0",
        "input_queue": len(read_jsonl(args.verification_queue)),
        "eligible_queue": len(queue),
        "lane_filter": sorted(set(args.lane or [])),
        "selected_issues": len(selected_issues),
        "logical_queries": len(queries),
        "engine_requests": len(queries),
        "by_role": dict(sorted(Counter(row["source_role"] for row in queries).items())),
        "by_engine": dict(sorted(Counter(row["engines"][0] for row in queries).items())),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def fallback_issue(row: dict[str, Any]) -> dict[str, Any]:
    route = row["provenance_route"]
    if route == "model_hypothesized":
        external = row["external_actor_or_frame"]
        competing = row["competing_frame_ko"]
        stake = row["korean_stake_ko"]
        decision = row["decision_object_ko"]
        discourse = row["korean_discourse_hypothesis_ko"]
        score = {"high": 75, "medium": 60, "low": 40}[row["confidence"]]
    else:
        external = row["named_external_actor_or_frame"]
        competing = row["why_possible_ksova_ko"]
        stake = row["korean_stake_hypothesis_ko"]
        decision = row["decision_object_hypothesis_ko"]
        discourse = "; ".join(row["missing_for_verification_ko"])
        score = {"web_likely": 70, "web_uncertain": 45}.get(row["triage_status"], 30)
    issue_id = stable_id(
        "KSISSUE-FB-", row["lane"], row["issue_ko"], external, decision
    )
    return {
        "schema_version": "ksova-issue-census-consolidated-1.0",
        "issue_id": issue_id,
        "lane": row["lane"],
        "input_candidate_ids": [row["candidate_id"]],
        "issue_ko": row["issue_ko"],
        "external_actor_or_frame": external,
        "competing_frame_ko": competing,
        "korean_stake_ko": stake,
        "decision_object_ko": decision,
        "korean_discourse_needed_ko": discourse,
        "discovery_gate_checks": {key: False for key in GATE_CHECKS},
        "verdict": "source_verification_fallback_unconfirmed",
        "priority_score": score,
        "source_search_plan": {
            "common_fact_or_legal_anchor": row["search_anchors_ko"][:2],
            "external_position": row["search_anchors_external"][:2],
            "korean_position": row["search_anchors_ko"][:2],
            "korean_counterargument_or_critique": row["search_anchors_ko"][:2],
        },
        "verification_risks_ko": [
            "통합 게이트 응답이 유효하지 않아 원문 단계에서 구성개념 전체를 재확인해야 함"
        ],
        "provenance_routes": [route],
        "evidence_eligibility": "none_until_body_verified",
        "source_verification_required": True,
        "discovery_gate_status": "unconfirmed_fallback_not_a_pass",
    }


def command_augment_verification_queue(args: argparse.Namespace) -> int:
    existing = read_jsonl(args.queue)
    hypotheses = read_jsonl(args.hypotheses)
    web = read_jsonl(args.web_candidates)
    represented = {
        candidate_id
        for row in existing
        for candidate_id in row.get("input_candidate_ids") or []
    }
    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in existing:
        row = dict(row)
        row["discovery_gate_status"] = "model_gate_pass"
        by_lane[row["lane"]].append(row)
    candidates = [row for row in hypotheses + web if row["candidate_id"] not in represented]
    fallback_added = 0
    for lane in sorted(LANES):
        lane_rows = by_lane[lane]
        available = [row for row in candidates if row["lane"] == lane]
        for candidate in balanced_candidates(
            available, max(0, args.max_per_lane - len(lane_rows))
        ):
            lane_rows.append(fallback_issue(candidate))
            fallback_added += 1

    selected = []
    positions = {lane: 0 for lane in LANES}
    while len(selected) < args.max_total:
        progressed = False
        for lane in sorted(LANES):
            index = positions[lane]
            if index < len(by_lane[lane]) and len(selected) < args.max_total:
                row = by_lane[lane][index]
                positions[lane] += 1
                row["lane_verification_rank"] = index + 1
                row["verification_queue_rank"] = len(selected) + 1
                row["verification_status"] = "pending_body_source_bundle"
                selected.append(row)
                progressed = True
        if not progressed:
            break
    write_jsonl(args.output, selected)
    report = {
        "schema_version": "ksova-augmented-verification-queue-report-1.0",
        "model_gate_pass_input": len(existing),
        "fallback_pool": len(candidates),
        "fallback_added_before_global_cap": fallback_added,
        "output_queue": len(selected),
        "by_gate_status": dict(sorted(Counter(row["discovery_gate_status"] for row in selected).items())),
        "by_lane": dict(sorted(Counter(row["lane"] for row in selected).items())),
        "by_provenance_routes": dict(sorted(Counter("+".join(row["provenance_routes"]) for row in selected).items())),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_select_next_verification_wave(args: argparse.Namespace) -> int:
    """Select only previously unprocessed candidates for a new source wave.

    This is deliberately not a second semantic attempt on rejected issues.  It
    removes every raw candidate represented by the processed queue, converts
    the remaining provenance-separated candidates to explicitly unconfirmed
    source-verification records, and balances the new wave across lanes and
    discovery routes.
    """
    processed = [
        row
        for path in args.processed_queue
        for row in read_jsonl(path)
    ]
    hypotheses = read_jsonl(args.hypotheses)
    web = read_jsonl(args.web_candidates)
    represented = {
        candidate_id
        for row in processed
        for candidate_id in row.get("input_candidate_ids") or []
    }
    processed_issue_ids = {str(row.get("issue_id") or "") for row in processed}
    by_lane: dict[str, deque[dict[str, Any]]] = {}
    available_count = Counter()
    for lane in sorted(LANES):
        available = [
            row for row in hypotheses + web
            if row["lane"] == lane and row["candidate_id"] not in represented
        ]
        available_count[lane] = len(available)
        converted = []
        seen_issue_ids = set()
        for candidate in balanced_candidates(available, args.max_per_lane):
            issue = fallback_issue(candidate)
            if issue["issue_id"] in processed_issue_ids or issue["issue_id"] in seen_issue_ids:
                continue
            seen_issue_ids.add(issue["issue_id"])
            converted.append(issue)
        by_lane[lane] = deque(converted)

    selected = []
    while len(selected) < args.max_total and any(by_lane.values()):
        progressed = False
        for lane in sorted(LANES):
            if by_lane[lane] and len(selected) < args.max_total:
                row = by_lane[lane].popleft()
                row["lane_verification_rank"] = 1 + sum(
                    prior["lane"] == lane for prior in selected
                )
                row["verification_queue_rank"] = len(selected) + 1
                row["verification_status"] = "pending_body_source_bundle"
                selected.append(row)
                progressed = True
        if not progressed:
            break

    write_jsonl(args.output, selected)
    report = {
        "schema_version": "ksova-next-verification-wave-report-1.0",
        "processed_issues": len(processed),
        "represented_raw_candidates": len(represented),
        "remaining_raw_candidates": sum(available_count.values()),
        "available_by_lane": dict(sorted(available_count.items())),
        "output_queue": len(selected),
        "max_total": args.max_total,
        "max_per_lane": args.max_per_lane,
        "by_lane": dict(sorted(Counter(row["lane"] for row in selected).items())),
        "by_provenance_routes": dict(sorted(Counter(
            "+".join(row["provenance_routes"]) for row in selected
        ).items())),
        "gate_status": "unconfirmed_fallback_not_a_pass",
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_build_origin_source_leads(args: argparse.Namespace) -> int:
    """Reconnect web-derived issue candidates to their originating URLs."""
    queue = read_jsonl(args.verification_queue)
    candidates = {
        row["candidate_id"]: row for row in read_jsonl(args.web_candidates)
    }
    leads = {row["lead_id"]: row for row in read_jsonl(args.web_leads)}
    output = []
    missing_candidates = Counter()
    missing_leads = Counter()
    linked_issues = set()
    linked_leads = set()
    for issue in queue:
        for candidate_id in issue.get("input_candidate_ids") or []:
            candidate = candidates.get(candidate_id)
            if candidate is None:
                missing_candidates[issue["lane"]] += 1
                continue
            for lead_id in candidate.get("lead_ids") or []:
                lead = leads.get(lead_id)
                if lead is None:
                    missing_leads[issue["lane"]] += 1
                    continue
                mapped = dict(lead)
                mapped["discoveries"] = [{
                    "lane": issue["lane"],
                    "language": lead.get("language") or "mixed",
                    "query": lead.get("query") or "originating_web_census_document",
                    "engine": lead.get("engine") or "web_census",
                    "rank": lead.get("rank"),
                    "candidate_id": issue["issue_id"],
                    "source_role": "originating_issue_document",
                    "jurisdiction": lead.get("jurisdiction") or "mixed",
                    "source_variant": "web_census_origin_body",
                }]
                output.append(mapped)
                linked_issues.add(issue["issue_id"])
                linked_leads.add(lead_id)
    write_jsonl(args.output, output)
    report = {
        "schema_version": "ksova-origin-source-leads-report-1.0",
        "queue_issues": len(queue),
        "web_candidates": len(candidates),
        "web_leads": len(leads),
        "linked_issues": len(linked_issues),
        "unique_linked_leads": len(linked_leads),
        "output_records": len(output),
        "missing_candidates_by_lane": dict(sorted(missing_candidates.items())),
        "missing_leads_by_lane": dict(sorted(missing_leads.items())),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    parse_hypotheses = commands.add_parser("parse-hypotheses")
    parse_hypotheses.add_argument("--requests", type=Path, required=True)
    parse_hypotheses.add_argument("--responses", type=Path, required=True)
    parse_hypotheses.add_argument("--output", type=Path, required=True)
    parse_hypotheses.add_argument("--report", type=Path, required=True)
    parse_hypotheses.set_defaults(func=command_parse_hypotheses)

    build_leads = commands.add_parser("build-lead-extraction")
    build_leads.add_argument("--leads", type=Path, required=True)
    build_leads.add_argument("--output", type=Path, required=True)
    build_leads.add_argument("--report", type=Path, required=True)
    build_leads.add_argument("--batch-size", type=int, default=18)
    build_leads.add_argument("--max-leads", type=int, default=1600)
    build_leads.set_defaults(func=command_build_lead_extraction)

    parse_leads = commands.add_parser("parse-lead-extraction")
    parse_leads.add_argument("--requests", type=Path, required=True)
    parse_leads.add_argument("--responses", type=Path, required=True)
    parse_leads.add_argument("--output", type=Path, required=True)
    parse_leads.add_argument("--report", type=Path, required=True)
    parse_leads.set_defaults(func=command_parse_lead_extraction)

    raw_leads = commands.add_parser("build-raw-lead-candidates")
    raw_leads.add_argument("--leads", type=Path, required=True)
    raw_leads.add_argument("--parsed", type=Path)
    raw_leads.add_argument("--output", type=Path, required=True)
    raw_leads.add_argument("--report", type=Path, required=True)
    raw_leads.set_defaults(func=command_build_raw_lead_candidates)

    build_consolidation = commands.add_parser("build-consolidation")
    build_consolidation.add_argument("--hypotheses", type=Path, required=True)
    build_consolidation.add_argument("--web-candidates", type=Path, required=True)
    build_consolidation.add_argument("--output", type=Path, required=True)
    build_consolidation.add_argument("--report", type=Path, required=True)
    build_consolidation.add_argument("--max-input-per-lane", type=int, default=36)
    build_consolidation.add_argument("--max-output-per-lane", type=int, default=15)
    build_consolidation.set_defaults(func=command_build_consolidation)

    parse_consolidation = commands.add_parser("parse-consolidation")
    parse_consolidation.add_argument("--hypotheses", type=Path, required=True)
    parse_consolidation.add_argument("--web-candidates", type=Path, required=True)
    parse_consolidation.add_argument("--requests", type=Path, required=True)
    parse_consolidation.add_argument("--responses", type=Path, required=True)
    parse_consolidation.add_argument("--output", type=Path, required=True)
    parse_consolidation.add_argument("--verification-queue", type=Path, required=True)
    parse_consolidation.add_argument("--report", type=Path, required=True)
    parse_consolidation.add_argument("--max-verify", type=int, default=100)
    parse_consolidation.set_defaults(func=command_parse_consolidation)

    build_source_search = commands.add_parser("build-source-search")
    build_source_search.add_argument("--verification-queue", type=Path, required=True)
    build_source_search.add_argument("--base-config", type=Path, required=True)
    build_source_search.add_argument("--output", type=Path, required=True)
    build_source_search.add_argument("--report", type=Path, required=True)
    build_source_search.add_argument("--max-issues", type=int, default=100)
    build_source_search.add_argument("--max-query-chars", type=int, default=180)
    build_source_search.set_defaults(func=command_build_source_search)

    build_role_search = commands.add_parser("build-role-source-search")
    build_role_search.add_argument("--verification-queue", type=Path, required=True)
    build_role_search.add_argument("--base-config", type=Path, required=True)
    build_role_search.add_argument("--output", type=Path, required=True)
    build_role_search.add_argument("--report", type=Path, required=True)
    build_role_search.add_argument("--max-issues", type=int, default=30)
    build_role_search.add_argument("--max-requests", type=int, default=60)
    build_role_search.add_argument("--max-query-chars", type=int, default=180)
    build_role_search.add_argument(
        "--provenance-route", choices=("model_hypothesized", "web_discovered")
    )
    build_role_search.add_argument(
        "--lane", action="append", choices=sorted(LANES), default=[],
        help="limit the search wave to an under-covered lane; repeatable",
    )
    build_role_search.set_defaults(func=command_build_role_source_search)

    augment = commands.add_parser("augment-verification-queue")
    augment.add_argument("--queue", type=Path, required=True)
    augment.add_argument("--hypotheses", type=Path, required=True)
    augment.add_argument("--web-candidates", type=Path, required=True)
    augment.add_argument("--output", type=Path, required=True)
    augment.add_argument("--report", type=Path, required=True)
    augment.add_argument("--max-total", type=int, default=100)
    augment.add_argument("--max-per-lane", type=int, default=10)
    augment.set_defaults(func=command_augment_verification_queue)

    next_wave = commands.add_parser("select-next-verification-wave")
    next_wave.add_argument(
        "--processed-queue", type=Path, action="append", required=True,
        help="Queue already processed; repeat to exclude multiple earlier waves.",
    )
    next_wave.add_argument("--hypotheses", type=Path, required=True)
    next_wave.add_argument("--web-candidates", type=Path, required=True)
    next_wave.add_argument("--output", type=Path, required=True)
    next_wave.add_argument("--report", type=Path, required=True)
    next_wave.add_argument("--max-total", type=int, default=120)
    next_wave.add_argument("--max-per-lane", type=int, default=12)
    next_wave.set_defaults(func=command_select_next_verification_wave)

    origin_leads = commands.add_parser("build-origin-source-leads")
    origin_leads.add_argument("--verification-queue", type=Path, required=True)
    origin_leads.add_argument("--web-candidates", type=Path, required=True)
    origin_leads.add_argument("--web-leads", type=Path, required=True)
    origin_leads.add_argument("--output", type=Path, required=True)
    origin_leads.add_argument("--report", type=Path, required=True)
    origin_leads.set_defaults(func=command_build_origin_source_leads)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
