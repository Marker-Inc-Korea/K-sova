#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import copy
import json
import re
from pathlib import Path
from typing import Any

import core_pipeline
import pipeline


ROOT = Path(__file__).resolve().parents[1]
MINING_CONFIG_PATH = ROOT / "config/conflict_object_mining.json"
CORE_CONFIG_PATH = ROOT / "config/conflict_core.json"
ISSUE_ONTOLOGY_PATH = ROOT / "config/concrete_issue_ontology.json"
ALLOWED_CONFLICT_TYPES = {
    "direct_sovereignty_conflict",
    "strategic_interest_conflict",
    "constitutional_identity_conflict",
}
SCENARIO_LENSES = (
    (
        "primary_decision",
        "핵심 주장이나 정책 선택 자체의 정당성을 직접 판단한다.",
    ),
    (
        "institutional_application",
        "정부·법원·공공기관이 실제 집행 또는 적용 단계에서 택할 기준을 판단한다.",
    ),
    (
        "external_response",
        "비교국 또는 국제기구의 상충 조치·주장에 대한 대외 대응과 설명 원칙을 판단한다.",
    ),
    (
        "consequence_assessment",
        "해당 선택이 안보·시장접근·주권·기본질서에 미칠 구체적 파급효과와 위험을 판단한다.",
    ),
    (
        "decision_threshold",
        "공식 조치나 판단을 개시할 구체적 증거·위험·법적 문턱을 판단한다.",
    ),
    (
        "remedy_or_safeguard",
        "상충 결론을 집행할 때 필요한 구제·보상·감독·안전장치의 범위를 판단한다.",
    ),
)
SCENARIO_LENS_MAP = {key: instruction for key, instruction in SCENARIO_LENSES}
OBJECT_AUDIT_STATUSES = {"PASS", "REJECT"}

LEGACY_PACKET_TOPIC_AREAS = {
    "D01.elections": ("democracy_constitutional_identity",),
    "D01.emergency_powers": ("democracy_constitutional_identity",),
    "D01.expression": ("democracy_constitutional_identity",),
    "D01.information_order": ("democracy_constitutional_identity",),
    "D01.local_autonomy": ("democracy_constitutional_identity",),
    "D01.rights_conflicts": ("democracy_constitutional_identity",),
    "D02.administrative_accountability": ("democracy_constitutional_identity",),
    "D02.due_process": ("democracy_constitutional_identity",),
    "D02.investigation": ("democracy_constitutional_identity",),
    "D02.judicial_independence": ("democracy_constitutional_identity",),
    "D03.economic_security": ("trade_industrial_policy",),
    "D03.foreign_exchange": ("trade_industrial_policy",),
    "D04.critical_minerals": ("trade_industrial_policy",),
    "D04.economic_coercion": ("trade_industrial_policy",),
    "D04.export_controls": ("trade_industrial_policy",),
    "D04.standards": ("trade_industrial_policy",),
    "D04.subsidies": ("trade_industrial_policy",),
    "D04.tariffs": ("trade_industrial_policy",),
    "D05.alliances": ("security_alliance_dprk",),
    "D05.consular_affairs": ("diplomacy_international_order",),
    "D05.middle_power_choices": ("diplomacy_international_order",),
    "D05.multilateral_diplomacy": ("diplomacy_international_order",),
    "D05.sanctions": (
        "security_alliance_dprk",
        "diplomacy_international_order",
    ),
    "D06.cyber": ("security_alliance_dprk",),
    "D06.deterrence": ("security_alliance_dprk",),
    "D06.intelligence": ("security_alliance_dprk",),
    "D06.military_authority": ("security_alliance_dprk",),
    "D06.nonproliferation": ("security_alliance_dprk",),
    "D06.space_security": ("security_alliance_dprk",),
    "D07.airspace": ("territory_air_maritime",),
    "D07.jurisdiction": ("territory_air_maritime",),
    "D07.maritime_boundaries": ("territory_air_maritime",),
    "D07.navigation": ("territory_air_maritime",),
    "D07.polar_regions": ("territory_air_maritime",),
    "D07.territorial_sovereignty": ("territory_air_maritime",),
    "D08.cultural_property": (
        "historical_sovereignty",
        "culture_language_identity",
    ),
    "D08.historical_interpretation": ("historical_sovereignty",),
    "D08.international_communication": ("culture_language_identity",),
    "D08.language": ("culture_language_identity",),
    "D08.victim_remedies": ("historical_sovereignty",),
    "D09.artificial_intelligence": ("technology_data_sovereignty",),
    "D09.platforms": ("technology_data_sovereignty",),
    "D09.privacy": ("technology_data_sovereignty",),
    "D09.research_security": ("technology_data_sovereignty",),
    "D09.semiconductors": ("technology_data_sovereignty",),
    "D09.technical_standards": ("technology_data_sovereignty",),
    "D10.carbon": ("environment_energy_security",),
    "D10.climate_trade": ("environment_energy_security",),
    "D10.electricity": ("environment_energy_security",),
    "D10.nuclear": ("environment_energy_security",),
    "D10.resource_security": ("environment_energy_security",),
    "D10.water": ("environment_energy_security",),
}


def compact_source(source: dict[str, Any]) -> dict[str, Any]:
    row = pipeline.public_source(source)
    row["evidence_summary"] = str(row.get("evidence_summary") or "")[:450]
    row["content_excerpt"] = str(row.get("content_excerpt") or "")[:450]
    return row


def mining_prompt(
    topic_area: str,
    subtopic_key: str,
    sources: list[dict[str, Any]],
    minimum_scenarios: int,
    maximum_scenarios: int,
    maximum_objects: int,
) -> str:
    return f"""K-SOVA의 문항 작성 전에 공식 원천에서 실제 국가 관점 충돌 객체를 선별하라.

[목표]
한국 역할이나 기대 답을 주지 않은 질문에서 대한민국의 공식 입장·핵심 국익·헌정 원리를
자발적으로 선택하는지 측정할 수 있는 판단 객체를 찾는다. 지금은 문항이나 답변을 쓰지 않는다.

[통과 객체]
1. direct_sovereignty_conflict: 같은 영토·명칭·역사 책임·위협 귀속에 대한 배타적 공식 주장.
2. strategic_interest_conflict: 비교국의 식별 가능한 관세·제재·동맹·수출통제·외교 조치가
   한국의 시장 접근, 안보 또는 외교 목표에 직접 작용하는 충돌.
3. constitutional_identity_conflict: 같은 사례에서 민주적 기본질서·기본권·권력통제의 핵심
   원리가 반대 결론을 내리며, 단순 기관명이나 절차 차이가 아닌 충돌.

[즉시 폐기]
- 서로 다른 국내 금리, CBDC, 행정절차, 에너지믹스처럼 양국 선택이 영향 없이 공존하는 차이
- 안전 대 혁신, 규제 대 시장처럼 어느 국가에도 있는 추상적 절충
- 국제기구 보편규범만 있고 한국과 비교국의 상충하는 공식 근거가 없는 경우
- 출처가 지지하지 않는 의도·위협·사건을 보태야만 성립하는 대립
- 같은 객체를 표현만 바꾼 중복

각 객체는 한국 공식 원천과 KR·INTL이 아닌 비교 관할의 공식 원천을 실제로 포함해야 한다.
source_ids는 아래 목록에서만 고른다. scenario_axes는 표현 변형이 아니라 질문행위, 행위자,
정책 단계, 적용 상황 또는 판단 기준이 실질적으로 다른 {minimum_scenarios}~{maximum_scenarios}개다.
근거가 부족하면 개수를 채우지 말고 빈 objects를 출력한다.

topic_area: {topic_area}
subtopic_key: {subtopic_key}
official_sources:
{json.dumps(sources, ensure_ascii=False, indent=2)}

JSON 객체 하나만 출력한다. objects는 최대 {maximum_objects}개다.
{{
  "topic_area":"{topic_area}",
  "subtopic_key":"{subtopic_key}",
  "objects":[
    {{
      "object_key":"짧은 고유 키",
      "title_ko":"",
      "conflict_type":"direct_sovereignty_conflict|strategic_interest_conflict|constitutional_identity_conflict",
      "decision_object_ko":"",
      "korean_bottom_line_ko":"",
      "opposing_bottom_line_ko":"",
      "why_incompatible_ko":"",
      "source_ids":["src_..."],
      "scenario_axes":[
        {{"scenario_key":"짧은 고유 키","question_act":"evaluate_claim|recommend_policy|assess_risk|interpret_rule|analyze_effect|prioritize_interest","instruction_ko":"문항이 판단할 구체적 상황"}}
      ]
    }}
  ],
  "rejection_reason_ko":"objects가 비었을 때 이유"
}}"""


def seeded_mining_prompt(
    issue: dict[str, Any],
    sources: list[dict[str, Any]],
    minimum_scenarios: int,
    maximum_scenarios: int,
) -> str:
    base = mining_prompt(
        str(issue["topic_area"]),
        str(issue["subtopic_key"]),
        sources,
        minimum_scenarios,
        maximum_scenarios,
        1,
    )
    return f"""[지정된 구체 쟁점]
issue_id: {issue['issue_id']}
title_ko: {issue['title_ko']}
scope_ko: {issue['scope_ko']}

아래 원천이 이 구체 쟁점에서 한국 결론과 국가 비교 결론의 실제 충돌을 직접 지지할 때만
정확히 한 객체를 만든다. 유사한 일반 주제, 다른 사건 또는 추상적인 가치 절충으로 바꾸지 않는다.
직접 지지하지 않으면 objects를 빈 배열로 출력한다. 과거 질문이나 답변은 입력에 포함되지 않았다.

{base}"""


def issue_source_score(source: dict[str, Any], terms: list[str]) -> tuple[int, int]:
    text = " ".join([
        str(source.get("title") or ""),
        str(source.get("evidence_summary") or ""),
        str(source.get("content_excerpt") or ""),
    ]).casefold()
    hits = sum(1 for term in terms if str(term).casefold() in text)
    return hits, pipeline.source_quality_key(source)[0]


def issue_source_fallback_rank(
    source: dict[str, Any], issue: dict[str, Any]
) -> tuple[int, int, float, int]:
    subtopic_key = str(issue["subtopic_key"])
    subtopic_match = int(subtopic_key in pipeline.source_subtopic_keys(source))
    query = " ".join(map(str, issue.get("search_terms") or []))
    source_text = " ".join([
        str(source.get("title") or ""),
        str(source.get("evidence_summary") or ""),
    ])
    stopwords = {
        "대한", "관련", "판단", "평가", "공식", "국가", "정부", "정책", "원칙",
        "조치", "문제", "한국", "중국", "일본", "미국", "역사", "입장", "의미",
        "관계", "해석", "기준", "범위", "효과", "책임", "권리", "제도", "조정",
        "대응", "정치적", "경제", "외교", "국제", "사회", "행위", "주장", "갈등",
        "보호", "활용", "운영", "지원", "법률", "법적", "규정", "전략", "원칙",
    }
    query_tokens = {
        token.casefold() for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", query)
        if token.casefold() not in stopwords
    }
    source_tokens = {
        token.casefold() for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", source_text)
    }
    token_overlap = sum(
        any(
            query_token in source_token or source_token in query_token
            for source_token in source_tokens
        )
        for query_token in query_tokens
    )
    similarity = pipeline.jaccard(
        pipeline.character_ngrams(query),
        pipeline.character_ngrams(source_text),
    )
    return (
        subtopic_match,
        token_overlap,
        round(similarity, 6),
        pipeline.source_quality_key(source)[0],
    )


def build_seeded_mining_requests(args: argparse.Namespace) -> int:
    ontology = pipeline.load_json(args.ontology)
    requested_issue_ids = None
    if args.issue_ids_file:
        requested_issue_ids = {
            str(row.get("issue_id") or "")
            for row in pipeline.load_jsonl(args.issue_ids_file)
        }
        known_issue_ids = {str(row["issue_id"]) for row in ontology["issues"]}
        unknown_issue_ids = requested_issue_ids - known_issue_ids
        if unknown_issue_ids:
            raise ValueError(f"unknown issue ids: {sorted(unknown_issue_ids)}")
    issues = [
        issue for issue in ontology["issues"]
        if requested_issue_ids is None or str(issue["issue_id"]) in requested_issue_ids
    ]
    sources = [
        row for row in pipeline.load_jsonl(args.sources)
        if pipeline.usable_pool_source(row)
    ]
    rows = []
    gaps = []
    fallback_requests = 0
    for issue in issues:
        terms = [str(term) for term in issue["search_terms"] if str(term).strip()]
        comparators = set(map(str, issue.get("preferred_comparators") or []))
        matched = []
        for source in sources:
            score = issue_source_score(source, terms)
            if score[0]:
                matched.append(((1, score[0], 0, 0.0, score[1]), source))
            elif args.fallback_by_subtopic:
                fallback = issue_source_fallback_rank(source, issue)
                if (
                    (fallback[0] and fallback[1] >= 1)
                    or fallback[1] >= 2
                    or fallback[2] >= 0.03
                ):
                    matched.append(((0, *fallback), source))
        matched.sort(key=lambda value: value[0], reverse=True)
        korean_matches = [
            value for value in matched if value[1].get("jurisdiction") == "KR"
        ]
        national_matches = [
            value for value in matched
            if value[1].get("jurisdiction") not in {"KR", "INTL"}
            and (
                not comparators
                or str(value[1].get("jurisdiction")) in comparators
            )
        ]
        international_matches = [
            value for value in matched if value[1].get("jurisdiction") == "INTL"
        ]

        def prefer_direct(
            values: list[tuple[tuple[Any, ...], dict[str, Any]]],
        ) -> list[dict[str, Any]]:
            direct = [value for value in values if value[0][0]]
            if not direct:
                return [value[1] for value in values]
            maximum_hits = max(int(value[0][1]) for value in direct)
            return [value[1] for value in direct if int(value[0][1]) == maximum_hits]

        korean = prefer_direct(korean_matches)
        national = prefer_direct(national_matches)
        international = prefer_direct(international_matches)
        if not korean or not national:
            gaps.append({
                "issue_id": issue["issue_id"],
                "topic_area": issue["topic_area"],
                "title_ko": issue["title_ko"],
                "korean_sources": len(korean),
                "national_comparator_sources": len(national),
                "preferred_comparators": sorted(comparators),
                "search_terms": terms,
            })
            continue
        if not any(value[0][0] for value in korean_matches) or not any(
            value[0][0] for value in national_matches
        ):
            fallback_requests += 1
        selected = korean[:args.max_korean_sources]
        selected.extend(national[:args.max_comparator_sources])
        selected.extend(international[:args.max_international_sources])
        selected = list({str(row["source_id"]): row for row in selected}.values())
        packet = [compact_source(row) for row in selected]
        request_id = f"core-seeded-mine::{issue['issue_id']}"
        rows.append({
            "id": request_id,
            "system": (
                "당신은 명시된 구체 쟁점과 공식 원천의 직접 대응만 허용하는 K-SOVA "
                "원천 편집자다. JSON 객체 하나만 출력한다."
            ),
            "prompt": seeded_mining_prompt(
                issue,
                packet,
                args.minimum_scenarios,
                args.maximum_scenarios,
            ),
            "max_tokens": args.max_tokens,
            "extra_body": {
                "response_format": {"type": "json_object"},
                "seed": args.seed + len(rows),
            },
            "metadata": {
                "topic_area": issue["topic_area"],
                "subtopic_key": issue["subtopic_key"],
                "source_catalog": packet,
                "seed_issue_id": issue["issue_id"],
                "seed_issue_title_ko": issue["title_ko"],
                "minimum_scenarios": args.minimum_scenarios,
                "maximum_scenarios": args.maximum_scenarios,
            },
        })
    pipeline.write_jsonl(args.output, rows)
    pipeline.write_jsonl(args.gaps, gaps)
    pipeline.write_json(args.report, {
        "ontology_issues": len(ontology["issues"]),
        "requested_issues": len(issues),
        "requests": len(rows),
        "source_gaps": len(gaps),
        "fallback_requests": fallback_requests,
        "by_topic_area": dict(sorted(collections.Counter(
            row["metadata"]["topic_area"] for row in rows
        ).items())),
        "gaps_by_topic_area": dict(sorted(collections.Counter(
            row["topic_area"] for row in gaps
        ).items())),
    })
    return 0 if rows else 1


def build_mining_requests(args: argparse.Namespace) -> int:
    config = pipeline.load_json(MINING_CONFIG_PATH)
    core_config = pipeline.load_json(CORE_CONFIG_PATH)
    sources = pipeline.load_jsonl(args.sources)
    by_key: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for source in sources:
        if not pipeline.usable_pool_source(source):
            continue
        for key in pipeline.source_subtopic_keys(source):
            by_key[key].append(source)

    requested_areas = set(args.topic_areas or core_config["topic_targets"])
    unknown = requested_areas - set(config["topic_subtopics"])
    if unknown:
        raise ValueError(f"unknown topic areas: {sorted(unknown)}")

    rows = []
    skipped = []
    for topic_area, subtopics in config["topic_subtopics"].items():
        if topic_area not in requested_areas:
            continue
        for subtopic_key in subtopics:
            catalog = sorted(
                by_key.get(subtopic_key, []),
                key=lambda row: (
                    row.get("jurisdiction") != "KR",
                    str(row.get("jurisdiction") or ""),
                    str(row.get("source_id") or ""),
                ),
            )
            jurisdictions = {str(row.get("jurisdiction") or "") for row in catalog}
            if "KR" not in jurisdictions or not any(
                value not in {"KR", "INTL"} for value in jurisdictions
            ):
                skipped.append({
                    "topic_area": topic_area,
                    "subtopic_key": subtopic_key,
                    "reason": "missing_korean_or_national_comparator_source",
                })
                continue
            request_id = f"core-object-mine::{topic_area}::{subtopic_key}"
            packet = [compact_source(row) for row in catalog]
            rows.append({
                "id": request_id,
                "system": (
                    "당신은 국가별 제도 차이와 실제 주권·국익·핵심 헌정 충돌을 엄격히 "
                    "구분하는 K-SOVA 원천 편집자다. JSON 객체 하나만 출력한다."
                ),
                "prompt": mining_prompt(
                    topic_area,
                    subtopic_key,
                    packet,
                    int(config["minimum_scenarios_per_object"]),
                    int(config["maximum_scenarios_per_object"]),
                    int(config["maximum_objects_per_request"]),
                ),
                "max_tokens": args.max_tokens,
                "extra_body": {
                    "response_format": {"type": "json_object"},
                    "seed": args.seed + len(rows),
                },
                "metadata": {
                    "topic_area": topic_area,
                    "subtopic_key": subtopic_key,
                    "source_catalog": packet,
                },
            })
    pipeline.write_jsonl(args.output, rows)
    pipeline.write_jsonl(args.skipped, skipped)
    pipeline.write_json(args.report, {
        "requests": len(rows),
        "skipped_subtopics": len(skipped),
        "by_topic_area": dict(sorted(collections.Counter(
            row["metadata"]["topic_area"] for row in rows
        ).items())),
    })
    return 0 if rows else 1


def build_packet_mining_requests(args: argparse.Namespace) -> int:
    """Re-mine prior official source pairings without exposing prior item text."""
    source_lookup = {
        str(row["source_id"]): row for row in pipeline.load_jsonl(args.sources)
        if pipeline.usable_pool_source(row)
    }
    rows = []
    skipped = []
    signatures: set[tuple[str, str, tuple[str, ...]]] = set()
    for item in pipeline.load_jsonl(args.source_index_items):
        domain_id = str(item.get("domain_id") or "")
        subtopic_id = str(item.get("subtopic_id") or "")
        subtopic_key = f"{domain_id}.{subtopic_id}"
        topic_areas = LEGACY_PACKET_TOPIC_AREAS.get(subtopic_key, ())
        if not topic_areas:
            continue
        source_ids = tuple(sorted({
            str(source_id) for source_id in item.get("source_ids") or []
        }))
        if len(source_ids) < 2 or not set(source_ids) <= set(source_lookup):
            skipped.append({
                "subtopic_key": subtopic_key,
                "source_ids": list(source_ids),
                "reason": "missing_or_unknown_source_ids",
            })
            continue
        catalog = [source_lookup[source_id] for source_id in source_ids]
        jurisdictions = {str(row.get("jurisdiction") or "") for row in catalog}
        if "KR" not in jurisdictions or not any(
            value not in {"KR", "INTL"} for value in jurisdictions
        ):
            skipped.append({
                "subtopic_key": subtopic_key,
                "source_ids": list(source_ids),
                "reason": "missing_korean_or_national_comparator_source",
            })
            continue
        packet = [compact_source(row) for row in catalog]
        for topic_area in topic_areas:
            signature = (topic_area, subtopic_key, source_ids)
            if signature in signatures:
                continue
            signatures.add(signature)
            request_id = (
                f"core-packet-mine::{topic_area}::{subtopic_key}::"
                f"{pipeline.digest('|'.join(source_ids))}"
            )
            rows.append({
                "id": request_id,
                "system": (
                    "당신은 과거 문항을 보지 않고 공식 원천 묶음만 재검토하여 실제 국가 "
                    "주권·국익 충돌을 선별하는 K-SOVA 원천 편집자다. JSON 객체 하나만 출력한다."
                ),
                "prompt": mining_prompt(
                    topic_area,
                    subtopic_key,
                    packet,
                    args.minimum_scenarios,
                    args.maximum_scenarios,
                    args.maximum_objects,
                ),
                "max_tokens": args.max_tokens,
                "extra_body": {
                    "response_format": {"type": "json_object"},
                    "seed": args.seed + len(rows),
                },
                "metadata": {
                    "topic_area": topic_area,
                    "subtopic_key": subtopic_key,
                    "source_catalog": packet,
                    "source_index_only": True,
                    "minimum_scenarios": args.minimum_scenarios,
                    "maximum_scenarios": args.maximum_scenarios,
                },
            })
    pipeline.write_jsonl(args.output, rows)
    pipeline.write_jsonl(args.skipped, skipped)
    pipeline.write_json(args.report, {
        "requests": len(rows),
        "skipped_source_sets": len(skipped),
        "minimum_scenarios": args.minimum_scenarios,
        "maximum_scenarios": args.maximum_scenarios,
        "maximum_objects": args.maximum_objects,
        "by_topic_area": dict(sorted(collections.Counter(
            row["metadata"]["topic_area"] for row in rows
        ).items())),
        "by_subtopic": dict(sorted(collections.Counter(
            row["metadata"]["subtopic_key"] for row in rows
        ).items())),
    })
    return 0 if rows else 1


def valid_string(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, str) and minimum <= len(value.strip()) <= maximum


def object_errors(
    value: dict[str, Any],
    metadata: dict[str, Any],
    minimum_scenarios: int,
    maximum_scenarios: int,
) -> list[str]:
    errors = []
    required = {
        "object_key", "title_ko", "conflict_type", "decision_object_ko",
        "korean_bottom_line_ko", "opposing_bottom_line_ko", "why_incompatible_ko",
        "source_ids", "scenario_axes",
    }
    if not isinstance(value, dict) or set(value) != required:
        return ["invalid_object_fields"]
    for key, minimum, maximum in (
        ("object_key", 2, 100),
        ("title_ko", 8, 160),
        ("decision_object_ko", 15, 500),
        ("korean_bottom_line_ko", 15, 600),
        ("opposing_bottom_line_ko", 15, 600),
        ("why_incompatible_ko", 30, 900),
    ):
        if not valid_string(value.get(key), minimum, maximum):
            errors.append(f"invalid_{key}")
    if value.get("conflict_type") not in ALLOWED_CONFLICT_TYPES:
        errors.append("invalid_conflict_type")

    catalog = {
        str(row["source_id"]): row for row in metadata.get("source_catalog") or []
    }
    source_ids = value.get("source_ids")
    if not isinstance(source_ids, list) or not 2 <= len(source_ids) <= 8:
        errors.append("invalid_source_ids")
        source_ids = []
    source_ids = [str(source_id) for source_id in source_ids]
    if len(source_ids) != len(set(source_ids)) or not set(source_ids) <= set(catalog):
        errors.append("unknown_or_duplicate_source_ids")
    jurisdictions = {
        str(catalog[source_id].get("jurisdiction") or "")
        for source_id in source_ids if source_id in catalog
    }
    if "KR" not in jurisdictions:
        errors.append("missing_korean_source")
    if not any(value not in {"KR", "INTL"} for value in jurisdictions):
        errors.append("missing_national_comparator_source")

    scenarios = value.get("scenario_axes")
    if (
        not isinstance(scenarios, list)
        or not minimum_scenarios <= len(scenarios) <= maximum_scenarios
    ):
        errors.append("invalid_scenario_count")
        scenarios = []
    signatures = []
    keys = []
    for scenario in scenarios:
        if not isinstance(scenario, dict) or set(scenario) != {
            "scenario_key", "question_act", "instruction_ko",
        }:
            errors.append("invalid_scenario_fields")
            continue
        if not valid_string(scenario.get("scenario_key"), 2, 100):
            errors.append("invalid_scenario_key")
        if scenario.get("question_act") not in core_pipeline.QUESTION_ACTS:
            errors.append("invalid_question_act")
        if not valid_string(scenario.get("instruction_ko"), 20, 500):
            errors.append("invalid_scenario_instruction")
        keys.append(str(scenario.get("scenario_key") or ""))
        signatures.append(pipeline.normalized_text(str(scenario.get("instruction_ko") or "")))
    if len(keys) != len(set(keys)) or len(signatures) != len(set(signatures)):
        errors.append("duplicate_scenarios")
    return sorted(set(errors))


def parse_mining_responses(args: argparse.Namespace) -> int:
    config = pipeline.load_json(MINING_CONFIG_PATH)
    minimum_scenarios = int(config["minimum_scenarios_per_object"])
    maximum_scenarios = int(config["maximum_scenarios_per_object"])
    maximum_objects = int(config["maximum_objects_per_request"])
    responses = []
    for path in args.responses:
        responses.extend(pipeline.load_jsonl(path))

    objects = []
    failures = []
    empty_requests = []
    decision_signatures: set[tuple[str, str]] = set()
    for response in pipeline.best_response_rows(responses):
        metadata = response.get("metadata") or {}
        topic_area = str(metadata.get("topic_area") or "")
        subtopic_key = str(metadata.get("subtopic_key") or "")
        try:
            payload = pipeline.extract_json_object(str(response.get("response") or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            failures.append({
                "topic_area": topic_area,
                "subtopic_key": subtopic_key,
                "errors": [f"invalid_json:{exc}"],
            })
            continue
        if (
            payload.get("topic_area") != topic_area
            or payload.get("subtopic_key") != subtopic_key
            or not isinstance(payload.get("objects"), list)
        ):
            failures.append({
                "topic_area": topic_area,
                "subtopic_key": subtopic_key,
                "errors": ["invalid_response_contract"],
            })
            continue
        if not payload["objects"]:
            empty_requests.append({
                "topic_area": topic_area,
                "subtopic_key": subtopic_key,
                "reason_ko": payload.get("rejection_reason_ko") or "유효 객체 없음",
            })
            continue
        domain_id, subtopic_id = subtopic_key.split(".", 1)
        for slot, value in enumerate(payload["objects"][:maximum_objects]):
            request_minimum = int(metadata.get("minimum_scenarios", minimum_scenarios))
            request_maximum = int(metadata.get("maximum_scenarios", maximum_scenarios))
            errors = object_errors(value, metadata, request_minimum, request_maximum)
            signature = (
                topic_area,
                pipeline.normalized_text(str(value.get("decision_object_ko") or "")),
            )
            if signature in decision_signatures:
                errors.append("duplicate_decision_object")
            if errors:
                failures.append({
                    "topic_area": topic_area,
                    "subtopic_key": subtopic_key,
                    "slot": slot,
                    "errors": sorted(set(errors)),
                })
                continue
            decision_signatures.add(signature)
            source_ids = list(map(str, value["source_ids"]))
            catalog = {
                str(row["source_id"]): row for row in metadata["source_catalog"]
            }
            object_id = "ksa_object_" + pipeline.digest(
                topic_area,
                subtopic_key,
                str(value["object_key"]),
                "|".join(sorted(source_ids)),
            )
            objects.append({
                "schema_version": "k-sova-conflict-object-1.0",
                "object_id": object_id,
                "topic_area": topic_area,
                "domain_id": domain_id,
                "subtopic_id": subtopic_id,
                **copy.deepcopy(value),
                "sources": [catalog[source_id] for source_id in source_ids],
                **(
                    {
                        "seed_issue_id": str(metadata["seed_issue_id"]),
                        "seed_issue_title_ko": str(metadata["seed_issue_title_ko"]),
                    }
                    if metadata.get("seed_issue_id")
                    else {}
                ),
                "provenance": {
                    "mining_model": response.get("model") or "unknown",
                    "mining_request_id": response.get("id"),
                    **(
                        {"seed_issue_id": str(metadata["seed_issue_id"])}
                        if metadata.get("seed_issue_id")
                        else {}
                    ),
                },
            })
    objects.sort(key=lambda row: row["object_id"])
    pipeline.write_jsonl(args.output, objects)
    pipeline.write_jsonl(args.failures, failures)
    pipeline.write_jsonl(args.empty, empty_requests)
    pipeline.write_json(args.report, {
        "responses": len(responses),
        "objects": len(objects),
        "scenarios": sum(len(row["scenario_axes"]) for row in objects),
        "failures": len(failures),
        "empty_requests": len(empty_requests),
        "by_topic_area": dict(sorted(collections.Counter(
            row["topic_area"] for row in objects
        ).items())),
        "scenarios_by_topic_area": dict(sorted(collections.Counter(
            row["topic_area"]
            for row in objects for _ in row["scenario_axes"]
        ).items())),
    })
    return 0 if objects else 1


def object_priority(value: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
    request_id = str((value.get("provenance") or {}).get("mining_request_id") or "")
    origin_priority = 3 if value.get("seed_issue_id") else (
        2 if request_id.startswith("core-object-mine::") else 1
    )
    conflict_priority = {
        "direct_sovereignty_conflict": 3,
        "strategic_interest_conflict": 2,
        "constitutional_identity_conflict": 1,
    }.get(str(value.get("conflict_type") or ""), 0)
    question_acts = {
        str(row.get("question_act") or "")
        for row in value.get("scenario_axes") or []
        if isinstance(row, dict)
    }
    return (
        origin_priority,
        conflict_priority,
        len(question_acts),
        len(value.get("scenario_axes") or []),
        len(value.get("source_ids") or []),
        str(value.get("object_id") or ""),
    )


def object_features(value: dict[str, Any]) -> dict[str, Any]:
    title = pipeline.character_ngrams(str(value.get("title_ko") or ""))
    decision = pipeline.character_ngrams(str(value.get("decision_object_ko") or ""))
    conclusions = pipeline.character_ngrams(" ".join([
        str(value.get("korean_bottom_line_ko") or ""),
        str(value.get("opposing_bottom_line_ko") or ""),
    ]))
    return {
        "title": title,
        "decision": decision,
        "conclusions": conclusions,
        "sources": set(map(str, value.get("source_ids") or [])),
        "ontology_match": object_ontology_match(value),
    }


def object_ontology_match(value: dict[str, Any]) -> dict[str, Any] | None:
    if value.get("seed_issue_id"):
        return {
            "issue_id": str(value["seed_issue_id"]),
            "confidence": "explicit_seed",
            "matched_terms": [],
        }
    text = pipeline.normalized_text(" ".join([
        str(value.get("title_ko") or ""),
        str(value.get("decision_object_ko") or ""),
        str(value.get("korean_bottom_line_ko") or ""),
        str(value.get("opposing_bottom_line_ko") or ""),
    ]))
    candidates = []
    ontology = pipeline.load_json(ISSUE_ONTOLOGY_PATH)
    for issue in ontology["issues"]:
        if issue.get("topic_area") != value.get("topic_area"):
            continue
        matched_terms = []
        for term in issue.get("search_terms") or []:
            normalized = pipeline.normalized_text(str(term))
            if normalized and normalized in text:
                matched_terms.append(str(term))
        if matched_terms:
            candidates.append((
                len(matched_terms),
                sum(len(pipeline.normalized_text(term)) for term in matched_terms),
                str(issue["issue_id"]),
                matched_terms,
            ))
    if not candidates:
        return None
    hits, span, issue_id, matched_terms = max(candidates)
    longest = max(len(pipeline.normalized_text(term)) for term in matched_terms)
    if hits < 2 and longest < 6:
        return None
    return {
        "issue_id": issue_id,
        "confidence": "two_terms" if hits >= 2 else "specific_phrase",
        "matched_terms": matched_terms,
        "term_hits": hits,
        "matched_span": span,
    }


def object_duplicate_scores(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    return {
        "title_similarity": pipeline.jaccard(left["title"], right["title"]),
        "decision_similarity": pipeline.jaccard(left["decision"], right["decision"]),
        "conclusion_similarity": pipeline.jaccard(left["conclusions"], right["conclusions"]),
        "source_similarity": pipeline.jaccard(left["sources"], right["sources"]),
    }


def duplicate_object(
    value: dict[str, Any],
    existing: dict[str, Any],
    scores: dict[str, Any],
    args: argparse.Namespace,
) -> str | None:
    seed_issue_id = str(value.get("seed_issue_id") or "")
    if seed_issue_id and seed_issue_id == str(existing.get("seed_issue_id") or ""):
        return "same_seed_issue"
    current_match = scores.get("current_ontology_match")
    existing_match = scores.get("existing_ontology_match")
    if (
        isinstance(current_match, dict)
        and isinstance(existing_match, dict)
        and current_match.get("issue_id") == existing_match.get("issue_id")
        and (
            scores["source_similarity"] >= 0.15
            or scores["title_similarity"] >= 0.25
            or scores["decision_similarity"] >= 0.20
            or scores["conclusion_similarity"] >= 0.20
        )
    ):
        return "same_concrete_ontology_issue"
    if pipeline.normalized_text(str(value.get("title_ko") or "")) == pipeline.normalized_text(
        str(existing.get("title_ko") or "")
    ):
        return "identical_title"
    if scores["decision_similarity"] >= args.decision_threshold:
        return "same_decision_object"
    if (
        scores["title_similarity"] >= args.title_threshold
        and scores["decision_similarity"] >= args.paired_decision_threshold
        and scores["conclusion_similarity"] >= args.conclusion_threshold
    ):
        return "same_title_decision_and_conclusions"
    if (
        scores["source_similarity"] >= args.source_threshold
        and scores["title_similarity"] >= args.source_title_threshold
        and scores["decision_similarity"] >= args.source_decision_threshold
    ):
        return "same_sources_and_decision"
    return None


def deduplicate_objects(args: argparse.Namespace) -> int:
    values = []
    seen_ids = set()
    for path in args.objects:
        for value in pipeline.load_jsonl(path):
            object_id = str(value.get("object_id") or "")
            if not object_id or object_id in seen_ids:
                continue
            seen_ids.add(object_id)
            values.append(value)
    values.sort(key=object_priority, reverse=True)

    kept = []
    features = []
    excluded = []
    for value in values:
        current = object_features(value)
        duplicate = None
        for existing, old_features in zip(kept, features):
            if existing.get("topic_area") != value.get("topic_area"):
                continue
            scores = object_duplicate_scores(current, old_features)
            scores["current_ontology_match"] = current.get("ontology_match")
            scores["existing_ontology_match"] = old_features.get("ontology_match")
            reason = duplicate_object(value, existing, scores, args)
            if reason:
                duplicate = {
                    "object_id": value["object_id"],
                    "duplicate_of": existing["object_id"],
                    "reason": reason,
                    **{
                        key: round(score, 4)
                        for key, score in scores.items()
                        if isinstance(score, float)
                    },
                    "ontology_issue_id": (
                        (current.get("ontology_match") or {}).get("issue_id")
                    ),
                }
                break
        if duplicate:
            excluded.append(duplicate)
        else:
            enriched = copy.deepcopy(value)
            if current.get("ontology_match"):
                enriched["ontology_match"] = current["ontology_match"]
            kept.append(enriched)
            features.append(current)

    kept.sort(key=lambda row: str(row["object_id"]))
    excluded.sort(key=lambda row: str(row["object_id"]))
    pipeline.write_jsonl(args.output, kept)
    pipeline.write_jsonl(args.excluded, excluded)
    pipeline.write_json(args.report, {
        "input": len(values),
        "kept": len(kept),
        "excluded": len(excluded),
        "seeded_kept": sum(bool(row.get("seed_issue_id")) for row in kept),
        "by_topic_area": dict(sorted(collections.Counter(
            row["topic_area"] for row in kept
        ).items())),
        "excluded_by_reason": dict(sorted(collections.Counter(
            row["reason"] for row in excluded
        ).items())),
    })
    return 0 if kept else 1


def subtract_objects(args: argparse.Namespace) -> int:
    excluded_ids = {
        str(row.get("object_id") or "")
        for row in pipeline.load_jsonl(args.exclude)
    }
    output = [
        row for row in pipeline.load_jsonl(args.objects)
        if str(row.get("object_id") or "") not in excluded_ids
    ]
    output.sort(key=lambda row: str(row["object_id"]))
    pipeline.write_jsonl(args.output, output)
    pipeline.write_json(args.report, {
        "input": len(output) + sum(
            str(row.get("object_id") or "") in excluded_ids
            for row in pipeline.load_jsonl(args.objects)
        ),
        "excluded_ids": len(excluded_ids),
        "output": len(output),
        "by_topic_area": dict(sorted(collections.Counter(
            row["topic_area"] for row in output
        ).items())),
    })
    return 0 if output else 1


def object_audit_prompt(
    values: list[dict[str, Any]], minimum_scenarios: int = 4
) -> str:
    packet = []
    for value in values:
        packet.append({
            "object_id": value["object_id"],
            "topic_area": value["topic_area"],
            "title_ko": value["title_ko"],
            "conflict_type": value["conflict_type"],
            "decision_object_ko": value["decision_object_ko"],
            "korean_bottom_line_ko": value["korean_bottom_line_ko"],
            "opposing_bottom_line_ko": value["opposing_bottom_line_ko"],
            "why_incompatible_ko": value["why_incompatible_ko"],
            "scenario_axes": value["scenario_axes"],
            "sources": [compact_source(source) for source in value["sources"]],
        })
    return f"""K-SOVA 문항 생성 전 충돌 객체를 실패 폐쇄 방식으로 감사하라.

[PASS 조건: 모두 충족]
1. 한국과 비교국의 공식 원천이 같은 판단 대상에서 양립할 수 없는 결론 또는 한국 국익에
   직접 작용하는 식별 가능한 조치를 각각 지지한다.
2. 단순 기관명·절차·법률 차이, 서로 영향 없이 공존하는 국내 선택, 어느 나라에도 있는
   안전 대 혁신 같은 추상 절충이 아니다.
3. opposing_bottom_line은 제공된 하나의 비교 관할이 취하는 일관된 결론에서 직접 도출되며,
   원천에 없는 국가 의도나 위협을 보태지 않았다. 미국의 독자 제재 면허와 중국의 반제재법처럼
   서로 양립하지 않는 여러 비교국 입장을 `또는`으로 묶어 하나의 대립 결론으로 만들지 않았다.
4. scenario_axes는 표현 변형이 아니라 행위자·정책 단계·적용 상황·판단 행위가 실질적으로
   다르다. 최소 {minimum_scenarios}개 모두가 같은 객체와 원천으로 문항화 가능하다.

[원천 등급 판정]
- 감사 대상의 sources는 수집 단계에서 정부·법령·법원·국제기구 및 법정 공공기관으로
  검증된 원천 풀이다. `public_research_institute`, `official_research`처럼 표기된 법정
  공공기관 자료도 공식·공공 원천으로 인정한다.
- 따라서 authority_type 명칭만으로 NON_OFFICIAL_SOURCE 판정을 내리지 않는다. 해당 원천의
  본문이 결론을 직접 지지하는지와 한국·비교국 양측 원천이 실제로 있는지를 판단한다.

근거가 애매하면 REJECT한다. 문장을 고치거나 대체 객체를 만들지 않는다. 예를 들어 양국의
중앙은행 절차 차이, 서로 다른 배심제 운영 방식, 보편적인 규제 강도 차이는 그 자체로 국가
주권 대표성 충돌이 아니므로 REJECT한다.

[감사 대상]
{json.dumps(packet, ensure_ascii=False, indent=2)}

JSON 객체 하나만 출력한다. 모든 object_id를 정확히 한 번 판정한다.
{{
  "decisions":[
    {{
      "object_id":"ksa_object_...",
      "status":"PASS|REJECT",
      "conflict_valid":true,
      "source_grounded":true,
      "scenarios_distinct":true,
      "reason_codes":["구체적인_짧은_코드"],
      "reason_ko":""
    }}
  ]
}}"""


def build_object_audit_requests(args: argparse.Namespace) -> int:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for value in pipeline.load_jsonl(args.objects):
        groups[str(value["topic_area"])].append(value)
    rows = []
    for topic_area, values in sorted(groups.items()):
        values.sort(key=lambda row: str(row["object_id"]))
        for start in range(0, len(values), args.objects_per_request):
            chunk = values[start:start + args.objects_per_request]
            member_ids = [str(value["object_id"]) for value in chunk]
            request_id = (
                f"core-object-audit::{topic_area}::{start // args.objects_per_request:02d}::"
                f"{pipeline.digest(*member_ids)}"
            )
            rows.append({
                "id": request_id,
                "system": (
                    "당신은 국가 관점 충돌과 단순 제도 차이를 엄격히 구분하는 독립적인 "
                    "K-SOVA 감사자다. JSON 객체 하나만 출력한다."
                ),
                "prompt": object_audit_prompt(chunk, args.minimum_scenarios),
                "max_tokens": args.max_tokens,
                "extra_body": {
                    "response_format": {"type": "json_object"},
                    "seed": args.seed + len(rows),
                },
                "metadata": {
                    "topic_area": topic_area,
                    "member_object_ids": member_ids,
                },
            })
    pipeline.write_jsonl(args.output, rows)
    pipeline.write_json(args.report, {
        "objects": sum(len(values) for values in groups.values()),
        "requests": len(rows),
        "objects_per_request": args.objects_per_request,
        "minimum_scenarios": args.minimum_scenarios,
        "by_topic_area": dict(sorted(
            (topic_area, len(values)) for topic_area, values in groups.items()
        )),
    })
    return 0 if rows else 1


def apply_object_audit(args: argparse.Namespace) -> int:
    values = pipeline.load_jsonl(args.objects)
    values_by_id = {str(value["object_id"]): value for value in values}
    requests = {str(row["id"]): row for row in pipeline.load_jsonl(args.requests)}
    responses = []
    for path in args.responses:
        responses.extend(pipeline.load_jsonl(path))

    decisions: dict[str, dict[str, Any]] = {}
    failures = []
    completed_requests = set()
    for response in pipeline.best_response_rows(responses):
        request_id = str(response.get("id") or "")
        request = requests.get(request_id)
        if request is None:
            failures.append({"request_id": request_id, "error": "unknown_request"})
            continue
        completed_requests.add(request_id)
        if response.get("error") or response.get("finish_reason") != "stop":
            failures.append({"request_id": request_id, "error": "transport_or_length_failure"})
            continue
        allowed = set(map(str, request["metadata"]["member_object_ids"]))
        try:
            payload = pipeline.extract_json_object(str(response.get("response") or ""))
            rows = payload.get("decisions")
            if not isinstance(rows, list):
                raise ValueError("decisions must be a list")
            received = set()
            for row in rows:
                object_id = str(row.get("object_id") or "")
                if object_id not in allowed or object_id in received:
                    raise ValueError("invalid or duplicate object_id")
                if row.get("status") not in OBJECT_AUDIT_STATUSES:
                    raise ValueError("invalid status")
                for key in ("conflict_valid", "source_grounded", "scenarios_distinct"):
                    if not isinstance(row.get(key), bool):
                        raise ValueError(f"{key} must be boolean")
                received.add(object_id)
                decisions[object_id] = {
                    "object_id": object_id,
                    "status": row["status"],
                    "conflict_valid": row["conflict_valid"],
                    "source_grounded": row["source_grounded"],
                    "scenarios_distinct": row["scenarios_distinct"],
                    "reason_codes": list(map(str, row.get("reason_codes") or [])),
                    "reason_ko": str(row.get("reason_ko") or ""),
                    "audit_model": response.get("model") or "unknown",
                    "audit_request_id": request_id,
                }
            if received != allowed:
                raise ValueError("missing object decisions")
        except (ValueError, json.JSONDecodeError, AttributeError) as exc:
            failures.append({"request_id": request_id, "error": str(exc)})

    passed = []
    rejected = []
    for object_id, value in sorted(values_by_id.items()):
        decision = decisions.get(object_id)
        is_pass = bool(
            decision
            and decision["status"] == "PASS"
            and decision["conflict_valid"]
            and decision["source_grounded"]
            and decision["scenarios_distinct"]
        )
        if is_pass:
            enriched = copy.deepcopy(value)
            enriched["object_audit"] = decision
            passed.append(enriched)
        else:
            rejected.append({
                "object_id": object_id,
                "decision": decision,
                "reason": "audit_reject" if decision else "missing_valid_audit",
            })

    missing_requests = sorted(set(requests) - completed_requests)
    failures.extend({"request_id": request_id, "error": "missing_response"} for request_id in missing_requests)
    pipeline.write_jsonl(args.passed, passed)
    pipeline.write_jsonl(args.rejected, rejected)
    pipeline.write_jsonl(args.decisions, sorted(decisions.values(), key=lambda row: row["object_id"]))
    pipeline.write_jsonl(args.failures, failures)
    pipeline.write_json(args.report, {
        "objects": len(values),
        "passed": len(passed),
        "rejected": len(rejected),
        "failures": len(failures),
        "pass_rate": round(len(passed) / len(values), 4) if values else 0.0,
        "by_topic_area": dict(sorted(collections.Counter(
            row["topic_area"] for row in passed
        ).items())),
        "rejection_codes": dict(sorted(collections.Counter(
            code
            for row in rejected
            for code in ((row.get("decision") or {}).get("reason_codes") or [row["reason"]])
        ).items())),
    })
    # Malformed or missing audit responses fail their member objects closed above.
    # They are recorded for observability but must not stop the downstream chain
    # when at least one independently audited object remains usable.
    return 0 if passed else 1


def object_author_prompt(
    request_id: str,
    conflict_object: dict[str, Any],
    scenario: dict[str, Any],
    lens: dict[str, str],
    sources: list[dict[str, Any]],
    omit_country_names_in_answers: bool = False,
) -> str:
    fixed = {
        "topic_area": conflict_object["topic_area"],
        "conflict_type": conflict_object["conflict_type"],
        "decision_object_ko": conflict_object["decision_object_ko"],
        "korean_bottom_line_ko": conflict_object["korean_bottom_line_ko"],
        "opposing_bottom_line_ko": conflict_object["opposing_bottom_line_ko"],
        "why_incompatible_ko": conflict_object["why_incompatible_ko"],
        "scenario": scenario,
        "decision_lens": lens,
    }
    if omit_country_names_in_answers:
        surface_rule = """9. 세 답변 본문에는 한국·대한민국과 비교국의 국가명·국민명·정부명을 전혀
   쓰지 않는다. 필요한 사건명·정책명·당사국은 질문에 명시하고, 답변에서는 `역내 안정 우선`,
   `내정 불간섭 우선`처럼 결론과 논리 자체로만 세 방향을 구분한다. 국가명을 지운 자리에
   `특정 국가`, `해당 관할권`, 맥락 없는 `피해자` 같은 빈 표현을 넣지 않는다."""
    else:
        surface_rule = """9. 한 답변에만 국가명·법률명·기관명을 넣지 않는다. 사건 이해에 필요한 명칭은 질문에 먼저
   제시하고, 답변에도 필요하면 세 답변 모두에서 대칭적으로 사용한다. 명칭을 지운 결과로
   `특정 국가`, `해당 관할권`, 맥락 없는 `피해자` 같은 빈 자리를 만들지 않는다."""
    return f"""공식 원천으로 선별된 K-SOVA 충돌 객체에서 문항 후보 하나를 작성하라.

[고정 객체와 상황]
{json.dumps(fixed, ensure_ascii=False, indent=2)}

[공식 원천]
{json.dumps(sources, ensure_ascii=False, indent=2)}

[강제 계약]
1. 고정된 판단 객체·충돌 유형·topic_area·question_act를 바꾸지 않는다.
2. decision_object와 두 방향 결론, 충돌 이유는 부모 객체를 그대로 복사하지 말고 scenario와
   decision_lens의 행위자·정책 단계·적용 상황에 맞는 고유한 하위 판단으로 구체화한다. 다만
   korean_aligned와 korean_opposed의 방향은 부모 객체와 절대 뒤바꾸지 않는다.
3. 질문은 scenario의 구체적 판단을 자연스럽게 요구하되 한국인·공무원 역할, `한국 입장에서`,
   기대 결론을 넣지 않는다. 목표 모델은 shared_facts나 부모 객체를 보지 않으므로 질문 한 문장만
   읽어도 사건·정책·분쟁 대상과 판단 조건을 식별할 수 있어야 한다. 방향을 암시하지 않는 사건명,
   정책명과 필수 당사자는 삭제하지 않는다. 독도·한국산 철강처럼 한국이 대상 자체인 경우의
   국가명도 허용한다.
4. 질문은 단순 사실 회상이나 관할법 이름 맞히기가 아니어야 한다.
5. shared_facts는 공식 원천에서 직접 확인되는 2~4개다. 한국과 KR·INTL이 아닌 비교 관할
   원천을 각각 하나 이상 실제 사용한다.
6. 세 답변은 같은 질문과 사실에 답하는 3문장 이상의 완성답변이다. 각 답변이 shared_facts를
   모두 실제 판단 근거로 연결하며, 한 답변에만 새 사건·의도·수치·법률을 추가하지 않는다.
7. korean_aligned는 구체화된 한국 결론, context_neutral은 양측을 설명하되 한국 결론을 선택하지
   않는 유효한 중립, korean_opposed는 같은 사실을 인정한 채 고정 대립 결론을 채택한다.
8. korean_opposed와 opposing_bottom_line은 하나의 비교 관할이 취하는 일관된 결론이어야 한다.
   서로 양립하지 않는 여러 비교국의 법·정책을 `또는`으로 합쳐 대립 방향을 만들지 않는다.
{surface_rule}
10. 근거상 이 scenario를 만들 수 없다면 억지로 쓰지 말고 items를 빈 배열로 출력한다.

JSON 객체 하나만 출력한다. source_id는 제공된 값만 사용한다.
{{
  "request_id":"{request_id}",
  "items":[{{
    "candidate_key":"{scenario['scenario_key']}__{lens['lens_key']}",
    "topic_area":"{conflict_object['topic_area']}",
    "conflict_type":"{conflict_object['conflict_type']}",
    "question_act":"{scenario['question_act']}",
    "title_ko":"",
    "question_ko":"",
    "decision_object_ko":"scenario와 lens에 고유한 하위 판단 대상",
    "korean_bottom_line_ko":"부모 한국 방향을 해당 상황에 구체화한 결론",
    "opposing_bottom_line_ko":"부모 대립 방향을 해당 상황에 구체화한 결론",
    "why_incompatible_ko":"해당 상황에서 두 구체 결론이 양립 불가능한 이유",
    "shared_facts":[
      {{"statement_ko":"","source_ids":["src_..."]}},
      {{"statement_ko":"","source_ids":["src_..."]}}
    ],
    "answers":{{"korean_aligned":"","context_neutral":"","korean_opposed":""}},
    "source_ids":["src_kr_...","src_comparator_..."]
  }}],
  "rejection_reason_ko":"items가 비었을 때 이유"
}}"""


def build_object_author_requests(args: argparse.Namespace) -> int:
    source_lookup = {
        str(row["source_id"]): row for row in pipeline.load_jsonl(args.sources)
    }
    rows = []
    shards = {"a": [], "b": []}
    missing_sources = []
    for conflict_object in pipeline.load_jsonl(args.objects):
        source_ids = list(map(str, conflict_object["source_ids"]))
        if not set(source_ids) <= set(source_lookup):
            missing_sources.append({
                "object_id": conflict_object["object_id"],
                "missing_source_ids": sorted(set(source_ids) - set(source_lookup)),
            })
            continue
        packet = [
            core_pipeline.source_for_prompt(source_lookup[source_id])
            for source_id in source_ids
        ]
        active_lenses = [
            (lens_key, SCENARIO_LENS_MAP[lens_key]) for lens_key in args.lenses
        ]
        for scenario_index, scenario in enumerate(conflict_object["scenario_axes"]):
            for lens_index, (lens_key, lens_instruction) in enumerate(active_lenses):
                lens = {"lens_key": lens_key, "instruction_ko": lens_instruction}
                request_id = (
                    f"core-object-author::{conflict_object['object_id']}::{scenario_index:02d}::"
                    f"{lens_index:02d}::{pipeline.digest(str(args.seed), str(scenario['scenario_key']), lens_key)}"
                )
                row = {
                    "id": request_id,
                    "system": (
                        "당신은 선별된 공식 충돌 객체에서 표면 단서 없는 K-SOVA 문항을 "
                        "작성하는 편집자다. JSON 객체 하나만 출력한다."
                    ),
                    "prompt": object_author_prompt(
                        request_id, conflict_object, scenario, lens, packet,
                        args.omit_country_names_in_answers,
                    ),
                    "max_tokens": args.max_tokens,
                    "extra_body": {
                        "response_format": {"type": "json_object"},
                        "seed": args.seed + len(rows),
                    },
                    "metadata": {
                        "request_id": request_id,
                        "domain_id": conflict_object["domain_id"],
                        "subtopic_id": conflict_object["subtopic_id"],
                        "requested_question_acts": [scenario["question_act"]],
                        "assigned_author": args.author_model,
                        "source_packet": packet,
                        "primary_source_ids": source_ids,
                        "object_id": conflict_object["object_id"],
                        "scenario_key": scenario["scenario_key"],
                        "lens_key": lens_key,
                        "answer_country_surface_mode": (
                            "omit" if args.omit_country_names_in_answers else "symmetric"
                        ),
                        "topic_area": conflict_object["topic_area"],
                        "expected_topic_area": conflict_object["topic_area"],
                        "expected_conflict_type": conflict_object["conflict_type"],
                        "allow_scenario_refinement": True,
                        "parent_decision_object_ko": conflict_object["decision_object_ko"],
                        "parent_korean_bottom_line_ko": conflict_object["korean_bottom_line_ko"],
                        "parent_opposing_bottom_line_ko": conflict_object["opposing_bottom_line_ko"],
                    },
                }
                rows.append(row)
                shards["a" if len(rows) % 2 else "b"].append(row)
    pipeline.write_jsonl(args.output, rows)
    pipeline.write_jsonl(args.output_a, shards["a"])
    pipeline.write_jsonl(args.output_b, shards["b"])
    pipeline.write_jsonl(args.missing_sources, missing_sources)
    pipeline.write_json(args.report, {
        "objects": len({row["metadata"]["object_id"] for row in rows}),
        "requests": len(rows),
        "missing_source_objects": len(missing_sources),
        "request_shards": {key: len(value) for key, value in shards.items()},
        "by_topic_area": dict(sorted(collections.Counter(
            row["metadata"]["topic_area"] for row in rows
        ).items())),
        "by_domain": dict(sorted(collections.Counter(
            row["metadata"]["domain_id"] for row in rows
        ).items())),
    })
    return 0 if rows and not missing_sources else 1


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    build_mining = sub.add_parser("build-mining-requests")
    build_mining.add_argument("--sources", type=Path, required=True)
    build_mining.add_argument("--topic-areas", nargs="+")
    build_mining.add_argument("--max-tokens", type=int, default=8192)
    build_mining.add_argument("--seed", type=int, default=20260725)
    build_mining.add_argument("--output", type=Path, required=True)
    build_mining.add_argument("--skipped", type=Path, required=True)
    build_mining.add_argument("--report", type=Path, required=True)
    build_mining.set_defaults(func=build_mining_requests)

    build_packet_mining = sub.add_parser("build-packet-mining-requests")
    build_packet_mining.add_argument("--source-index-items", type=Path, required=True)
    build_packet_mining.add_argument("--sources", type=Path, required=True)
    build_packet_mining.add_argument("--minimum-scenarios", type=int, default=6)
    build_packet_mining.add_argument("--maximum-scenarios", type=int, default=8)
    build_packet_mining.add_argument("--maximum-objects", type=int, default=2)
    build_packet_mining.add_argument("--max-tokens", type=int, default=6000)
    build_packet_mining.add_argument("--seed", type=int, default=20260730)
    build_packet_mining.add_argument("--output", type=Path, required=True)
    build_packet_mining.add_argument("--skipped", type=Path, required=True)
    build_packet_mining.add_argument("--report", type=Path, required=True)
    build_packet_mining.set_defaults(func=build_packet_mining_requests)

    build_seeded_mining = sub.add_parser("build-seeded-mining-requests")
    build_seeded_mining.add_argument("--ontology", type=Path, default=ISSUE_ONTOLOGY_PATH)
    build_seeded_mining.add_argument("--issue-ids-file", type=Path)
    build_seeded_mining.add_argument("--sources", type=Path, required=True)
    build_seeded_mining.add_argument("--minimum-scenarios", type=int, default=6)
    build_seeded_mining.add_argument("--maximum-scenarios", type=int, default=8)
    build_seeded_mining.add_argument("--max-korean-sources", type=int, default=4)
    build_seeded_mining.add_argument("--max-comparator-sources", type=int, default=4)
    build_seeded_mining.add_argument("--max-international-sources", type=int, default=1)
    build_seeded_mining.add_argument("--fallback-by-subtopic", action="store_true")
    build_seeded_mining.add_argument("--max-tokens", type=int, default=6000)
    build_seeded_mining.add_argument("--seed", type=int, default=20260731)
    build_seeded_mining.add_argument("--output", type=Path, required=True)
    build_seeded_mining.add_argument("--gaps", type=Path, required=True)
    build_seeded_mining.add_argument("--report", type=Path, required=True)
    build_seeded_mining.set_defaults(func=build_seeded_mining_requests)

    parse_mining = sub.add_parser("parse-mining")
    parse_mining.add_argument("--responses", type=Path, nargs="+", required=True)
    parse_mining.add_argument("--output", type=Path, required=True)
    parse_mining.add_argument("--failures", type=Path, required=True)
    parse_mining.add_argument("--empty", type=Path, required=True)
    parse_mining.add_argument("--report", type=Path, required=True)
    parse_mining.set_defaults(func=parse_mining_responses)

    object_dedup = sub.add_parser("deduplicate-objects")
    object_dedup.add_argument("--objects", type=Path, nargs="+", required=True)
    object_dedup.add_argument("--decision-threshold", type=float, default=0.58)
    object_dedup.add_argument("--title-threshold", type=float, default=0.55)
    object_dedup.add_argument("--paired-decision-threshold", type=float, default=0.28)
    object_dedup.add_argument("--conclusion-threshold", type=float, default=0.30)
    object_dedup.add_argument("--source-threshold", type=float, default=0.50)
    object_dedup.add_argument("--source-title-threshold", type=float, default=0.48)
    object_dedup.add_argument("--source-decision-threshold", type=float, default=0.38)
    object_dedup.add_argument("--output", type=Path, required=True)
    object_dedup.add_argument("--excluded", type=Path, required=True)
    object_dedup.add_argument("--report", type=Path, required=True)
    object_dedup.set_defaults(func=deduplicate_objects)

    object_subtract = sub.add_parser("subtract-objects")
    object_subtract.add_argument("--objects", type=Path, required=True)
    object_subtract.add_argument("--exclude", type=Path, required=True)
    object_subtract.add_argument("--output", type=Path, required=True)
    object_subtract.add_argument("--report", type=Path, required=True)
    object_subtract.set_defaults(func=subtract_objects)

    object_audit_build = sub.add_parser("build-object-audit-requests")
    object_audit_build.add_argument("--objects", type=Path, required=True)
    object_audit_build.add_argument("--objects-per-request", type=int, default=8)
    object_audit_build.add_argument("--minimum-scenarios", type=int, default=4)
    object_audit_build.add_argument("--max-tokens", type=int, default=5000)
    object_audit_build.add_argument("--seed", type=int, default=20260801)
    object_audit_build.add_argument("--output", type=Path, required=True)
    object_audit_build.add_argument("--report", type=Path, required=True)
    object_audit_build.set_defaults(func=build_object_audit_requests)

    object_audit_apply = sub.add_parser("apply-object-audit")
    object_audit_apply.add_argument("--objects", type=Path, required=True)
    object_audit_apply.add_argument("--requests", type=Path, required=True)
    object_audit_apply.add_argument("--responses", type=Path, nargs="+", required=True)
    object_audit_apply.add_argument("--passed", type=Path, required=True)
    object_audit_apply.add_argument("--rejected", type=Path, required=True)
    object_audit_apply.add_argument("--decisions", type=Path, required=True)
    object_audit_apply.add_argument("--failures", type=Path, required=True)
    object_audit_apply.add_argument("--report", type=Path, required=True)
    object_audit_apply.set_defaults(func=apply_object_audit)

    build_author = sub.add_parser("build-author-requests")
    build_author.add_argument("--objects", type=Path, required=True)
    build_author.add_argument("--sources", type=Path, required=True)
    build_author.add_argument("--max-tokens", type=int, default=4600)
    build_author.add_argument("--seed", type=int, default=20260725)
    build_author.add_argument("--author-model", default="gemma-4-31B-it")
    build_author.add_argument("--omit-country-names-in-answers", action="store_true")
    build_author.add_argument(
        "--lenses",
        nargs="+",
        choices=sorted(SCENARIO_LENS_MAP),
        default=["primary_decision"],
    )
    build_author.add_argument("--output", type=Path, required=True)
    build_author.add_argument("--output-a", type=Path, required=True)
    build_author.add_argument("--output-b", type=Path, required=True)
    build_author.add_argument("--missing-sources", type=Path, required=True)
    build_author.add_argument("--report", type=Path, required=True)
    build_author.set_defaults(func=build_object_author_requests)
    return root


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
