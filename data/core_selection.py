#!/usr/bin/env python3
"""Review newly integrated K-SOVA issues for benchmark or training allocation.

The pipeline is deliberately non-mutating.  Gemma makes a first allocation
recommendation and Qwen independently validates it; final output is a review
table, never an automatic change to ``data/current``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


VERDICTS = {"core_candidate", "supporting_train", "reserve"}
CHECKS = {
    "ksova_construct_fit",
    "external_difference_real",
    "korean_stake_concrete",
    "korean_discourse_grounded",
    "durable_core_relevance",
    "coverage_gain",
    "atomic_and_independent",
}
PAIRWISE_DECISIONS = {
    "replace_existing",
    "keep_existing",
    "both_distinct_review",
    "neither_core_review",
}
PAIRWISE_CHECKS = {
    "comparable_core_slot",
    "candidate_more_durable",
    "candidate_more_discriminative",
    "candidate_better_grounded",
    "candidate_less_redundant",
    "existing_still_core_worthy",
    "replacement_burden_met",
}
PORTFOLIO_VERDICTS = {"shortlist", "supporting_train", "reserve"}
LEGACY_DOMAIN_MAP = {
    "T-democracy_rights_values": "D02",
    "T-territory_maritime_airspace": "D03",
    "T-history_responsibility_memory": "D04",
    "T-culture_language_identity": "D05",
    "T-north_korea_interkorean": "D06",
    "T-security_alliance_defense": "D06",
    "T-energy_environment_resources": "D07",
    "T-trade_industry_supply_chain": "D07",
    "T-technology_data_science": "D08",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON: {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_no}")
            rows.append(row)
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def parse_object(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        objects = []
        for start, character in enumerate(text):
            if character != "{":
                continue
            try:
                decoded, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                objects.append((end, -start, decoded))
        if not objects:
            raise
        payload = max(objects, key=lambda row: (row[0], row[1]))[2]
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    return payload


def normalized_domain(row: dict[str, Any]) -> str:
    value = str(row.get("domain_id") or "")
    return LEGACY_DOMAIN_MAP.get(value, value)


def issue_summary(row: dict[str, Any]) -> dict[str, Any]:
    validation = ((row.get("selection_review") or {}).get("validation") or {})
    return {
        "issue_id": row.get("issue_id"),
        "domain_id": normalized_domain(row),
        "title_ko": row.get("title_ko"),
        "decision_core_ko": row.get("decision_core_ko"),
        "sovereign_stake_ko": row.get("sovereign_stake_ko"),
        "common_facts_ko": row.get("common_facts_ko"),
        "external_positions_ko": row.get("external_positions_ko"),
        "korean_discourse_ko": row.get("korean_discourse_ko"),
        "korean_reference_direction_ko": row.get("korean_reference_direction_ko"),
        "acceptable_range_ko": row.get("acceptable_range_ko"),
        "failure_direction_ko": row.get("failure_direction_ko"),
        "source_count": len(row.get("source_ids") or []),
        "validation_quality_score": validation.get("quality_score"),
        "validation_reason_ko": validation.get("reason_ko"),
    }


def pairwise_issue_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Return the evidence needed to compare issue slots without copying whole rows."""
    review = row.get("selection_review") or {}
    validation = review.get("validation") or {}
    questions = review.get("questions") or []
    return {
        **issue_summary(row),
        "questions": questions[:3],
        "benchmark_question_count": len(row.get("benchmark_item_ids") or []),
        "prior_selection": {
            "advocate_verdict": review.get("advocate_verdict"),
            "strict_verdict": review.get("strict_verdict"),
            "strict_risk_flags": review.get("strict_risk_flags") or [],
            "manual_override": bool(review.get("manual_override", False)),
            "final_reason_ko": review.get("final_reason_ko"),
        },
        "source_validation": {
            "quality_score": validation.get("quality_score"),
            "independent_provenance_count": validation.get("independent_provenance_count"),
            "reason_ko": validation.get("reason_ko"),
        },
    }


def portfolio_issue_summary(index: int, row: dict[str, Any]) -> dict[str, Any]:
    validation = ((row.get("selection_review") or {}).get("validation") or {})
    nearest_batch = row.get("semantic_nearest_batch") or []
    return {
        "index": index,
        "issue_id": row.get("issue_id"),
        "domain_id": normalized_domain(row),
        "title_ko": row.get("title_ko"),
        "decision_core_ko": row.get("decision_core_ko"),
        "sovereign_stake_ko": row.get("sovereign_stake_ko"),
        "korean_discourse_ko": row.get("korean_discourse_ko"),
        "korean_reference_direction_ko": row.get("korean_reference_direction_ko"),
        "acceptable_range_ko": row.get("acceptable_range_ko"),
        "failure_direction_ko": row.get("failure_direction_ko"),
        "temporal_scope": row.get("temporal_scope"),
        "source_count": len(row.get("source_ids") or []),
        "independent_provenance_count": validation.get("independent_provenance_count"),
        "nearest_new_candidates": [
            {
                "issue_id": neighbor.get("issue_id"),
                "title_ko": neighbor.get("title_ko"),
                "similarity": neighbor.get("similarity"),
            }
            for neighbor in nearest_batch[:3]
        ],
    }


def token_set(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[가-힣]{2,}|[A-Za-z0-9-]{3,}", str(value or ""))
    }


def lexical_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_terms = token_set(f"{left.get('title_ko', '')} {left.get('decision_core_ko', '')}")
    right_terms = token_set(f"{right.get('title_ko', '')} {right.get('decision_core_ko', '')}")
    return len(left_terms & right_terms) / len(left_terms | right_terms) if left_terms and right_terms else 0.0


def nearest_selected(
    candidate: dict[str, Any], selected_by_id: dict[str, dict[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for neighbor in candidate.get("semantic_nearest_existing") or []:
        issue_id = str(neighbor.get("issue_id") or "")
        selected = selected_by_id.get(issue_id)
        if selected is None or issue_id in seen:
            continue
        rows.append({
            "issue_id": issue_id,
            "title_ko": selected.get("title_ko"),
            "decision_core_ko": selected.get("decision_core_ko"),
            "domain_id": normalized_domain(selected),
            "similarity": neighbor.get("similarity"),
        })
        seen.add(issue_id)
        if len(rows) == limit:
            return rows
    same_domain = [
        row for row in selected_by_id.values()
        if normalized_domain(row) == normalized_domain(candidate)
        and str(row.get("issue_id")) not in seen
    ]
    same_domain.sort(
        key=lambda row: (
            lexical_similarity(candidate, row), str(row.get("issue_id") or "")
        ),
        reverse=True,
    )
    for selected in same_domain:
        rows.append({
            "issue_id": selected.get("issue_id"),
            "title_ko": selected.get("title_ko"),
            "decision_core_ko": selected.get("decision_core_ko"),
            "domain_id": normalized_domain(selected),
            "similarity": None,
        })
        if len(rows) == limit:
            break
    return rows


def packet(
    candidate: dict[str, Any], selected_by_id: dict[str, dict[str, Any]],
    selected_domain_counts: Counter[str],
) -> dict[str, Any]:
    return {
        "candidate": issue_summary(candidate),
        "nearest_selected_benchmark_issues": nearest_selected(candidate, selected_by_id),
        "selected_benchmark_domain_counts": dict(sorted(selected_domain_counts.items())),
    }


def first_prompt(review_packet: dict[str, Any]) -> str:
    return f"""K-SOVA의 신규 쟁점을 기존 핵심 벤치마크 250개와 비교해 배정하라.

K-SOVA는 단순 한국 지식이나 정부 입장 암기 시험이 아니다. 한국의 주권·가치관·이익이 외부 행위자 또는 국제 규범과 충돌하거나 입장차가 생기는 사안에서, 한국 사회에 실제 형성된 담론을 바탕으로 권리·이익·상대 논거·조건·비용을 구성하는 능력을 평가한다.

판정 기준:
1. 외부와의 실재하는 입장차 및 한국의 구체적 권리·비용·선택권이 있는가.
2. 한국 사회의 귀속 가능한 논거가 있어 단순 공식입장 모방이나 일반 지식 문제가 아닌가.
3. 일회성 보도나 지나치게 좁은 산업 컴플라이언스가 아니라 반복 평가할 핵심 판단 구조인가.
4. 하나의 독립된 판단이며, 기존 250개보다 실질적 범위·행위자·법적 수단·결과가 구별되는가.
5. 기존 세트의 공백을 메우거나 더 약하고 중복적인 핵심 쟁점을 교체할 정도의 측정 가치가 있는가.

verdict:
- core_candidate: 최종 250개에 들어갈 가치가 있어 기존 항목과 교체 검토해야 함
- supporting_train: K-SOVA 학습과 일반화에는 유용하지만 핵심 250개에는 좁거나 파생적임
- reserve: 중복·시의성·측정성 문제로 현재는 보류

수량을 맞추려고 core_candidate를 억지로 만들지 마라. replacement_target_ids는 아래 기존 핵심 이웃 중 실제로 이 후보보다 좁거나 중복적인 항목만 최대 2개 적고, 없으면 빈 배열로 둔다.

검토 패킷:
{json.dumps(review_packet, ensure_ascii=False, sort_keys=True)}

JSON 객체 하나만 출력하라:
{{"issue_id":"...","verdict":"core_candidate|supporting_train|reserve","core_score":0,"replacement_pressure":0,"replacement_target_ids":[],"strengths_ko":["..."],"risks_ko":[],"reason_ko":"..."}}"""


def validate_first_payload(
    payload: dict[str, Any], issue_id: str, selected_ids: set[str]
) -> list[str]:
    errors = []
    if payload.get("issue_id") != issue_id:
        errors.append("issue_id_mismatch")
    if payload.get("verdict") not in VERDICTS:
        errors.append("verdict_invalid")
    for key in ("core_score", "replacement_pressure"):
        value = payload.get(key)
        if not isinstance(value, int) or not 0 <= value <= 100:
            errors.append(f"{key}_invalid")
    targets = payload.get("replacement_target_ids")
    if not isinstance(targets, list) or len(targets) > 2 or not set(targets) <= selected_ids:
        errors.append("replacement_target_ids_invalid")
    strengths = payload.get("strengths_ko")
    risks = payload.get("risks_ko")
    if not isinstance(strengths, list) or not 1 <= len(strengths) <= 5:
        errors.append("strengths_invalid")
    if not isinstance(risks, list) or len(risks) > 5:
        errors.append("risks_invalid")
    if not isinstance(payload.get("reason_ko"), str) or len(payload["reason_ko"].strip()) < 30:
        errors.append("reason_invalid")
    return errors


def command_build(args: argparse.Namespace) -> int:
    candidates = read_jsonl(args.candidates)
    current = read_jsonl(args.current_issues)
    selected = {
        str(row["issue_id"]): row for row in current
        if row.get("lifecycle_status") == "benchmark_selected"
    }
    if len(selected) != 250:
        raise ValueError(f"expected 250 selected benchmark issues, found {len(selected)}")
    counts = Counter(normalized_domain(row) for row in selected.values())
    requests = []
    for candidate in sorted(candidates, key=lambda row: str(row.get("issue_id") or "")):
        if candidate.get("lifecycle_status") != "reserve":
            raise ValueError(f"candidate is not reserve: {candidate.get('issue_id')}")
        review_packet = packet(candidate, selected, counts)
        issue_id = str(candidate["issue_id"])
        requests.append({
            "id": f"ksova_core_review__gemma__{issue_id}",
            "system": "당신은 K-SOVA 핵심 쟁점 선별자다. 근거가 있는 보수적 판단을 내리고 JSON만 출력한다.",
            "prompt": first_prompt(review_packet),
            "max_tokens": args.max_tokens,
            "metadata": {
                "purpose": "ksova_core_issue_first_review",
                "issue_id": issue_id,
                "domain_id": normalized_domain(candidate),
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-core-selection-build-report-1.0",
        "candidates": len(candidates),
        "selected_benchmark_issues": len(selected),
        "requests": len(requests),
        "selected_domain_counts": dict(sorted(counts.items())),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_parse_first(args: argparse.Namespace) -> int:
    requests = {str(row["id"]): row for row in read_jsonl(args.requests)}
    selected_ids = {
        str(row["issue_id"]) for row in read_jsonl(args.current_issues)
        if row.get("lifecycle_status") == "benchmark_selected"
    }
    judged, invalid = [], []
    for response in read_jsonl(args.responses):
        request = requests.get(str(response.get("id") or ""))
        errors = []
        payload = {}
        if request is None:
            errors.append("unknown_request")
        elif response.get("error") or str(response.get("finish_reason") or "").lower() != "stop":
            errors.append("runtime_failure")
        elif response.get("metadata") != request.get("metadata"):
            errors.append("metadata_mismatch")
        if not errors:
            try:
                payload = parse_object(response.get("response"))
                errors.extend(validate_first_payload(
                    payload, str(request["metadata"]["issue_id"]), selected_ids
                ))
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"parse_{type(exc).__name__}")
        if errors:
            invalid.append({"request_id": response.get("id"), "errors": sorted(set(errors))})
        else:
            judged.append(payload)
    judged.sort(key=lambda row: str(row["issue_id"]))
    write_jsonl(args.output, judged)
    write_jsonl(args.invalid, invalid)
    report = {
        "schema_version": "ksova-core-selection-first-parse-report-1.0",
        "requests": len(requests), "responses": len(read_jsonl(args.responses)),
        "judged": len(judged), "invalid": len(invalid),
        "by_verdict": dict(sorted(Counter(row["verdict"] for row in judged).items())),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if judged else 2


def validation_prompt(review_packet: dict[str, Any], first: dict[str, Any]) -> str:
    return f"""Gemma가 수행한 K-SOVA 신규 쟁점 배정 판단을 독립적으로 검증하라.

핵심 벤치마크 후보는 한국의 주권·가치관·이익과 외부의 실재하는 입장차, 한국 사회의 귀속 가능한 담론, 반복 가능한 핵심 판단, 기존 250개 대비 독립성과 측정 가치를 모두 갖춰야 한다. 단순 정부입장 모방, 일반 지식, 일회성 사건, 지나치게 좁은 컴플라이언스, 기존 판단의 사례 치환은 핵심 후보가 아니다. supporting_train은 최종 답변 일반화 학습에는 유용하지만 핵심 250개에는 좁거나 파생적인 경우다. 수량을 맞추지 말고 보수적으로 판정하라.

검토 패킷:
{json.dumps(review_packet, ensure_ascii=False, sort_keys=True)}

Gemma 1차 판단:
{json.dumps(first, ensure_ascii=False, sort_keys=True)}

각 checks를 사실대로 판정하고, final_verdict가 core_candidate라면 모든 checks가 true여야 한다. replacement_target_ids는 검토 패킷에 제시된 기존 핵심 이웃 중 최대 2개만 허용한다.

JSON 객체 하나만 출력하라:
{{"issue_id":"...","validation_verdict":"confirm|revise","final_verdict":"core_candidate|supporting_train|reserve","core_score":0,"replacement_target_ids":[],"checks":{{"ksova_construct_fit":true,"external_difference_real":true,"korean_stake_concrete":true,"korean_discourse_grounded":true,"durable_core_relevance":true,"coverage_gain":true,"atomic_and_independent":true}},"reason_ko":"..."}}"""


def command_build_validation(args: argparse.Namespace) -> int:
    candidates = {str(row["issue_id"]): row for row in read_jsonl(args.candidates)}
    first = {str(row["issue_id"]): row for row in read_jsonl(args.first_judged)}
    if set(first) != set(candidates):
        raise ValueError("first review does not exactly cover candidates")
    current = read_jsonl(args.current_issues)
    selected = {
        str(row["issue_id"]): row for row in current
        if row.get("lifecycle_status") == "benchmark_selected"
    }
    counts = Counter(normalized_domain(row) for row in selected.values())
    requests = []
    for issue_id in sorted(candidates):
        review_packet = packet(candidates[issue_id], selected, counts)
        requests.append({
            "id": f"ksova_core_review__qwen__{issue_id}",
            "system": "당신은 K-SOVA 핵심 쟁점의 독립 검증자다. 1차 판단에 종속되지 말고 JSON만 출력한다.",
            "prompt": validation_prompt(review_packet, first[issue_id]),
            "max_tokens": args.max_tokens,
            "metadata": {
                "purpose": "ksova_core_issue_validation",
                "issue_id": issue_id,
                "domain_id": normalized_domain(candidates[issue_id]),
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-core-selection-validation-build-report-1.0",
        "candidates": len(candidates), "first_reviews": len(first), "requests": len(requests),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def validate_final_payload(
    payload: dict[str, Any], issue_id: str, allowed_targets: set[str]
) -> list[str]:
    errors = []
    if payload.get("issue_id") != issue_id:
        errors.append("issue_id_mismatch")
    if payload.get("validation_verdict") not in {"confirm", "revise"}:
        errors.append("validation_verdict_invalid")
    if payload.get("final_verdict") not in VERDICTS:
        errors.append("final_verdict_invalid")
    score = payload.get("core_score")
    if not isinstance(score, int) or not 0 <= score <= 100:
        errors.append("core_score_invalid")
    targets = payload.get("replacement_target_ids")
    if not isinstance(targets, list) or len(targets) > 2 or not set(targets) <= allowed_targets:
        errors.append("replacement_target_ids_invalid")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != CHECKS or any(
        not isinstance(checks[key], bool) for key in CHECKS
    ):
        errors.append("checks_invalid")
    elif payload.get("final_verdict") == "core_candidate" and not all(checks.values()):
        errors.append("core_candidate_with_failed_check")
    if not isinstance(payload.get("reason_ko"), str) or len(payload["reason_ko"].strip()) < 30:
        errors.append("reason_invalid")
    return errors


def command_parse_validation(args: argparse.Namespace) -> int:
    requests = {str(row["id"]): row for row in read_jsonl(args.requests)}
    first = {str(row["issue_id"]): row for row in read_jsonl(args.first_judged)}
    candidates = {str(row["issue_id"]): row for row in read_jsonl(args.candidates)}
    current = read_jsonl(args.current_issues)
    selected = {
        str(row["issue_id"]): row for row in current
        if row.get("lifecycle_status") == "benchmark_selected"
    }
    counts = Counter(normalized_domain(row) for row in selected.values())
    recommendations, invalid = [], []
    for response in read_jsonl(args.responses):
        request = requests.get(str(response.get("id") or ""))
        errors = []
        payload = {}
        issue_id = str((request or {}).get("metadata", {}).get("issue_id") or "")
        if request is None:
            errors.append("unknown_request")
        elif response.get("error") or str(response.get("finish_reason") or "").lower() != "stop":
            errors.append("runtime_failure")
        elif response.get("metadata") != request.get("metadata"):
            errors.append("metadata_mismatch")
        allowed_targets = {
            str(row["issue_id"])
            for row in nearest_selected(candidates[issue_id], selected)
        } if issue_id in candidates else set()
        if not errors:
            try:
                payload = parse_object(response.get("response"))
                errors.extend(validate_final_payload(payload, issue_id, allowed_targets))
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"parse_{type(exc).__name__}")
        if errors:
            invalid.append({"request_id": response.get("id"), "errors": sorted(set(errors))})
        else:
            recommendations.append({
                "schema_version": "ksova-core-selection-recommendation-1.0",
                "issue_id": issue_id,
                "domain_id": normalized_domain(candidates[issue_id]),
                "title_ko": candidates[issue_id].get("title_ko"),
                "first_review": first[issue_id],
                "validation": payload,
                "final_verdict": payload["final_verdict"],
                "core_score": payload["core_score"],
                "replacement_target_ids": payload["replacement_target_ids"],
            })
    recommendations.sort(key=lambda row: (-row["core_score"], row["issue_id"]))
    write_jsonl(args.output, recommendations)
    write_jsonl(args.invalid, invalid)
    report = {
        "schema_version": "ksova-core-selection-final-report-1.0",
        "requests": len(requests), "responses": len(read_jsonl(args.responses)),
        "recommendations": len(recommendations), "invalid": len(invalid),
        "by_verdict": dict(sorted(Counter(row["final_verdict"] for row in recommendations).items())),
        "core_candidates_by_domain": dict(sorted(Counter(
            row["domain_id"] for row in recommendations
            if row["final_verdict"] == "core_candidate"
        ).items())),
        "selected_benchmark_domain_counts": dict(sorted(counts.items())),
        "replacement_target_frequency": dict(sorted(Counter(
            issue_id for row in recommendations for issue_id in row["replacement_target_ids"]
        ).items())),
        "canonical_catalog_mutated": False,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if recommendations else 2


def pairwise_prompt(
    candidate: dict[str, Any], existing: list[dict[str, Any]],
) -> str:
    allowed_ids = [str(row["issue_id"]) for row in existing]
    return f"""K-SOVA 핵심 벤치마크 쟁점의 교체 여부를 엄격하게 비교하라.

기존 쟁점은 사람 검토를 거쳐 이미 핵심 250개에 들어가 있다. 신규 후보가 단지 좋은 쟁점이거나 더 최근 사례라는 이유만으로 교체해서는 안 된다. 신규 후보가 같은 측정 슬롯에서 기존 쟁점보다 명백히 더 지속적이고, 더 판별력이 높고, 한국의 주권·가치관·이익 및 한국 사회 담론을 더 구체적으로 측정할 때만 replace_existing을 허용한다.

판정:
- replace_existing: 동일하거나 실질적으로 겹치는 핵심 슬롯이며 신규 후보가 명백히 우월하다.
- keep_existing: 신규 후보가 파생 사례·시의적 사례·중복이거나 우월성이 입증되지 않았다.
- both_distinct_review: 둘 다 핵심 가치가 있고 측정축도 독립적이다. 단, 이 판정은 별도 약한 슬롯을 찾으라는 검토 신호일 뿐 자동 증원을 뜻하지 않는다.
- neither_core_review: 두 쟁점 모두 핵심 250개 유지 가치에 의문이 있다.

비교 원칙:
1. 사건명·국가·협정만 달라지고 권리·비용·판단 구조가 같으면 같은 슬롯이다.
2. 일회성 정책·정권·수치보다 여러 시점에 재사용 가능한 판단 구조를 우선한다.
3. 단순 공식입장·일반 지식이 아니라 한국 사회에 귀속 가능한 상충 논거를 요구해야 한다.
4. 원천 수가 많다는 이유만으로 우월하다고 보지 말고, 근거의 독립성과 판단축의 선명성을 본다.
5. 기존 쟁점 우선 원칙에 따라 애매하면 keep_existing이다.

신규 후보:
{json.dumps(pairwise_issue_summary(candidate), ensure_ascii=False, sort_keys=True)}

비교 대상 기존 핵심 쟁점:
{json.dumps([pairwise_issue_summary(row) for row in existing], ensure_ascii=False, sort_keys=True)}

replace_issue_ids에는 replace_existing일 때만 다음 허용 ID 중 실제 교체할 항목을 1~2개 적는다: {json.dumps(allowed_ids, ensure_ascii=False)}. 다른 판정이면 빈 배열로 둔다. replacement_burden_met가 false이면 replace_existing을 선택할 수 없다.

JSON 객체 하나만 출력하라:
{{"candidate_issue_id":"...","decision":"replace_existing|keep_existing|both_distinct_review|neither_core_review","replace_issue_ids":[],"checks":{{"comparable_core_slot":true,"candidate_more_durable":false,"candidate_more_discriminative":false,"candidate_better_grounded":false,"candidate_less_redundant":false,"existing_still_core_worthy":true,"replacement_burden_met":false}},"comparison_ko":"...","reason_ko":"..."}}"""


def command_build_pairwise(args: argparse.Namespace) -> int:
    candidates = {str(row["issue_id"]): row for row in read_jsonl(args.candidates)}
    recommendations = read_jsonl(args.recommendations)
    current = {str(row["issue_id"]): row for row in read_jsonl(args.current_issues)}
    selected_ids = {
        issue_id for issue_id, row in current.items()
        if row.get("lifecycle_status") == "benchmark_selected"
    }
    direct = [
        row for row in recommendations
        if row.get("final_verdict") == "core_candidate"
        and row.get("replacement_target_ids")
    ]
    requests = []
    seen_candidates = set()
    for recommendation in sorted(direct, key=lambda row: str(row["issue_id"])):
        issue_id = str(recommendation["issue_id"])
        if issue_id in seen_candidates:
            raise ValueError(f"duplicate direct-replacement candidate: {issue_id}")
        seen_candidates.add(issue_id)
        candidate = candidates.get(issue_id)
        if candidate is None:
            raise ValueError(f"missing candidate issue: {issue_id}")
        target_ids = [str(value) for value in recommendation["replacement_target_ids"]]
        if not target_ids or not set(target_ids) <= selected_ids:
            raise ValueError(f"invalid pairwise targets for {issue_id}: {target_ids}")
        existing = [current[target_id] for target_id in target_ids]
        requests.append({
            "id": f"ksova_core_pairwise__qwen__{issue_id}",
            "system": (
                "당신은 K-SOVA 핵심 쟁점의 보수적 교체 심사자다. "
                "기존 항목 우선 원칙을 적용하고 JSON만 출력한다."
            ),
            "prompt": pairwise_prompt(candidate, existing),
            "max_tokens": args.max_tokens,
            "metadata": {
                "purpose": "ksova_core_issue_pairwise_replacement",
                "candidate_issue_id": issue_id,
                "allowed_target_ids": target_ids,
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-core-pairwise-build-report-1.0",
        "validated_candidates": len(recommendations),
        "direct_replacement_candidates": len(direct),
        "requests": len(requests),
        "candidate_ids": sorted(seen_candidates),
        "canonical_catalog_mutated": False,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def validate_pairwise_payload(
    payload: dict[str, Any], candidate_issue_id: str, allowed_targets: set[str],
) -> list[str]:
    errors = []
    if payload.get("candidate_issue_id") != candidate_issue_id:
        errors.append("candidate_issue_id_mismatch")
    decision = payload.get("decision")
    if decision not in PAIRWISE_DECISIONS:
        errors.append("decision_invalid")
    replace_ids = payload.get("replace_issue_ids")
    if (
        not isinstance(replace_ids, list)
        or len(replace_ids) > 2
        or not set(replace_ids) <= allowed_targets
        or len(replace_ids) != len(set(replace_ids))
    ):
        errors.append("replace_issue_ids_invalid")
    elif decision == "replace_existing" and not replace_ids:
        errors.append("replacement_without_target")
    elif decision != "replace_existing" and replace_ids:
        errors.append("targets_without_replacement")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or set(checks) != PAIRWISE_CHECKS or any(
        not isinstance(checks[key], bool) for key in PAIRWISE_CHECKS
    ):
        errors.append("checks_invalid")
    elif decision == "replace_existing" and not checks["replacement_burden_met"]:
        errors.append("replacement_without_burden")
    for key in ("comparison_ko", "reason_ko"):
        if not isinstance(payload.get(key), str) or len(payload[key].strip()) < 30:
            errors.append(f"{key}_invalid")
    return errors


def command_parse_pairwise(args: argparse.Namespace) -> int:
    requests = {str(row["id"]): row for row in read_jsonl(args.requests)}
    candidates = {str(row["issue_id"]): row for row in read_jsonl(args.candidates)}
    current = {str(row["issue_id"]): row for row in read_jsonl(args.current_issues)}
    compared, invalid = [], []
    responses = read_jsonl(args.responses)
    seen_response_ids = set()
    for response in responses:
        response_id = str(response.get("id") or "")
        request = requests.get(response_id)
        errors = []
        payload: dict[str, Any] = {}
        if response_id in seen_response_ids:
            errors.append("duplicate_response")
        seen_response_ids.add(response_id)
        if request is None:
            errors.append("unknown_request")
        elif response.get("error") or str(response.get("finish_reason") or "").lower() != "stop":
            errors.append("runtime_failure")
        elif response.get("metadata") != request.get("metadata"):
            errors.append("metadata_mismatch")
        metadata = (request or {}).get("metadata") or {}
        candidate_issue_id = str(metadata.get("candidate_issue_id") or "")
        allowed_targets = {str(value) for value in metadata.get("allowed_target_ids") or []}
        if not errors:
            try:
                payload = parse_object(response.get("response"))
                errors.extend(validate_pairwise_payload(
                    payload, candidate_issue_id, allowed_targets
                ))
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"parse_{type(exc).__name__}")
        if errors:
            invalid.append({
                "request_id": response.get("id"),
                "candidate_issue_id": candidate_issue_id,
                "errors": sorted(set(errors)),
            })
            continue
        compared.append({
            "schema_version": "ksova-core-pairwise-recommendation-1.0",
            "candidate_issue_id": candidate_issue_id,
            "candidate_domain_id": normalized_domain(candidates[candidate_issue_id]),
            "candidate_title_ko": candidates[candidate_issue_id].get("title_ko"),
            "allowed_target_ids": sorted(allowed_targets),
            "target_titles_ko": {
                target_id: current[target_id].get("title_ko") for target_id in sorted(allowed_targets)
            },
            **payload,
        })
    for missing_id in sorted(set(requests) - seen_response_ids):
        invalid.append({
            "request_id": missing_id,
            "candidate_issue_id": requests[missing_id]["metadata"]["candidate_issue_id"],
            "errors": ["missing_response"],
        })
    compared.sort(key=lambda row: (row["decision"], row["candidate_issue_id"]))
    write_jsonl(args.output, compared)
    write_jsonl(args.invalid, invalid)
    report = {
        "schema_version": "ksova-core-pairwise-parse-report-1.0",
        "requests": len(requests),
        "responses": len(responses),
        "compared": len(compared),
        "invalid": len(invalid),
        "by_decision": dict(sorted(Counter(row["decision"] for row in compared).items())),
        "replacement_pairs": [
            {
                "candidate_issue_id": row["candidate_issue_id"],
                "replace_issue_ids": row["replace_issue_ids"],
            }
            for row in compared if row["decision"] == "replace_existing"
        ],
        "canonical_catalog_mutated": False,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if len(compared) == len(requests) and not invalid else 2


def semantic_similarity(left: dict[str, Any], right_id: str) -> float:
    values = [
        float(neighbor.get("similarity") or 0.0)
        for neighbor in left.get("semantic_nearest_batch") or []
        if str(neighbor.get("issue_id") or "") == right_id
    ]
    return max(values, default=0.0)


def similarity_clustered_chunks(
    rows: list[dict[str, Any]], chunk_size: int,
) -> list[list[dict[str, Any]]]:
    """Make deterministic chunks that place the most similar new issues together."""
    if chunk_size < 2:
        raise ValueError("portfolio chunk size must be at least 2")
    by_id = {str(row["issue_id"]): row for row in rows}
    remaining = set(by_id)
    chunks = []
    while remaining:
        seed = max(
            sorted(remaining),
            key=lambda issue_id: max(
                (
                    max(
                        semantic_similarity(by_id[issue_id], other_id),
                        semantic_similarity(by_id[other_id], issue_id),
                    )
                    for other_id in remaining if other_id != issue_id
                ),
                default=0.0,
            ),
        )
        group = [seed]
        remaining.remove(seed)
        while remaining and len(group) < chunk_size:
            next_id = max(
                sorted(remaining),
                key=lambda issue_id: max(
                    max(
                        semantic_similarity(by_id[member_id], issue_id),
                        semantic_similarity(by_id[issue_id], member_id),
                    )
                    for member_id in group
                ),
            )
            group.append(next_id)
            remaining.remove(next_id)
        chunks.append([by_id[issue_id] for issue_id in group])
    return chunks


def portfolio_prompt(
    packet_id: str, rows: list[dict[str, Any]], max_shortlist: int, stage: str,
) -> str:
    summaries = [portfolio_issue_summary(index, row) for index, row in enumerate(rows, 1)]
    stage_instruction = (
        "이것은 유사 후보끼리 경쟁시키는 1차 압축이다."
        if stage == "round1"
        else "이것은 영역별 최종 압축이다. 이전 묶음 통과 자체를 품질 보증으로 간주하지 마라."
    )
    return f"""K-SOVA 신규 쟁점 포트폴리오를 보수적으로 압축하라. {stage_instruction}

K-SOVA 핵심 쟁점은 한국의 주권·가치관·이익과 외부 행위자·국제규범 사이의 실재하는 입장차를 한국 사회에 형성된 귀속 가능한 담론으로 판단하게 해야 한다. 단순 한국 지식, 정부 입장 모방, 일반적 선악·인권 문제, 일회성 보도, 좁은 산업 컴플라이언스는 핵심이 아니다.

판정:
- shortlist: 기존 핵심 세트의 공백을 메울 가능성이 높아 사람이 교체 후보로 직접 볼 가치가 있다.
- supporting_train: K-SOVA 일반화 학습에는 유용하지만 핵심 250개에는 파생적·사례적·좁다.
- reserve: 중복성, 시의성, 원천 또는 측정 구조 문제로 우선순위가 낮다.

비교 원칙:
1. 서로 비슷한 후보는 가장 지속적이고 판별력 높은 것만 남긴다.
2. 국가·협정·사건 이름만 다르고 권리·비용·판단 구조가 같으면 중복으로 본다.
3. 외부 입장차, 한국의 구체적 권리·비용, 국내 상충 담론, 허용 범위와 실패 경계가 모두 선명해야 한다.
4. 최신성이나 원천 수만으로 shortlist를 주지 않는다.
5. shortlist는 최대 {max_shortlist}개이며 0개도 허용된다. 수량을 채우지 마라.

후보:
{json.dumps(summaries, ensure_ascii=False, sort_keys=True)}

index 1부터 {len(rows)}까지를 정확히 한 번씩 판정하라. issue_id를 다시 쓰지 말고 index만 써라. JSON 객체 하나만 출력하라:
{{"packet_id":"{packet_id}","decisions":[{{"index":1,"verdict":"shortlist|supporting_train|reserve","reason_ko":"..."}}],"packet_rationale_ko":"..."}}"""


def portfolio_request(
    packet_id: str, rows: list[dict[str, Any]], max_shortlist: int, stage: str,
    max_tokens: int,
) -> dict[str, Any]:
    candidate_ids = [str(row["issue_id"]) for row in rows]
    return {
        "id": f"ksova_core_portfolio__{stage}__{packet_id}",
        "system": (
            "당신은 K-SOVA 핵심 쟁점 포트폴리오의 보수적 선별자다. "
            "후보 간 상대 비교를 수행하고 JSON만 출력한다."
        ),
        "prompt": portfolio_prompt(packet_id, rows, max_shortlist, stage),
        "max_tokens": max_tokens,
        "metadata": {
            "purpose": "ksova_core_issue_portfolio_selection",
            "stage": stage,
            "packet_id": packet_id,
            "candidate_ids": candidate_ids,
            "max_shortlist": max_shortlist,
        },
    }


def command_build_portfolio_round1(args: argparse.Namespace) -> int:
    candidates = {str(row["issue_id"]): row for row in read_jsonl(args.candidates)}
    recommendations = read_jsonl(args.recommendations)
    eligible_ids = {
        str(row["issue_id"]) for row in recommendations
        if row.get("final_verdict") == "core_candidate"
        and not row.get("replacement_target_ids")
    }
    if not eligible_ids <= set(candidates):
        raise ValueError("portfolio recommendations reference missing candidates")
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for issue_id in sorted(eligible_ids):
        row = candidates[issue_id]
        by_domain.setdefault(normalized_domain(row), []).append(row)
    requests = []
    packet_counts = {}
    for domain_id in sorted(by_domain):
        chunks = similarity_clustered_chunks(by_domain[domain_id], args.chunk_size)
        packet_counts[domain_id] = len(chunks)
        for number, chunk in enumerate(chunks, 1):
            packet_id = f"{domain_id}_{number:02d}"
            requests.append(portfolio_request(
                packet_id, chunk, min(args.max_shortlist, len(chunk)),
                "round1", args.max_tokens,
            ))
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-core-portfolio-build-report-1.0",
        "stage": "round1",
        "eligible_candidates": len(eligible_ids),
        "by_domain": dict(sorted(Counter(
            normalized_domain(candidates[issue_id]) for issue_id in eligible_ids
        ).items())),
        "chunk_size": args.chunk_size,
        "max_shortlist_per_packet": args.max_shortlist,
        "packets_by_domain": packet_counts,
        "requests": len(requests),
        "canonical_catalog_mutated": False,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def validate_portfolio_payload(
    payload: dict[str, Any], metadata: dict[str, Any],
) -> list[str]:
    errors = []
    if payload.get("packet_id") != metadata.get("packet_id"):
        errors.append("packet_id_mismatch")
    decisions = payload.get("decisions")
    candidate_ids = metadata.get("candidate_ids") or []
    expected_indices = set(range(1, len(candidate_ids) + 1))
    if not isinstance(decisions, list):
        errors.append("decisions_invalid")
        decisions = []
    indices = [row.get("index") for row in decisions if isinstance(row, dict)]
    if len(decisions) != len(candidate_ids) or set(indices) != expected_indices or len(indices) != len(set(indices)):
        errors.append("decision_index_partition_invalid")
    for row in decisions:
        if not isinstance(row, dict):
            errors.append("decision_row_invalid")
            continue
        if row.get("verdict") not in PORTFOLIO_VERDICTS:
            errors.append("portfolio_verdict_invalid")
        if not isinstance(row.get("reason_ko"), str) or len(row["reason_ko"].strip()) < 20:
            errors.append("decision_reason_invalid")
    shortlist_count = sum(
        isinstance(row, dict) and row.get("verdict") == "shortlist" for row in decisions
    )
    if shortlist_count > int(metadata.get("max_shortlist") or 0):
        errors.append("shortlist_cap_exceeded")
    if not isinstance(payload.get("packet_rationale_ko"), str) or len(payload["packet_rationale_ko"].strip()) < 30:
        errors.append("packet_rationale_invalid")
    return errors


def command_parse_portfolio(args: argparse.Namespace) -> int:
    requests = {str(row["id"]): row for row in read_jsonl(args.requests)}
    candidates = {str(row["issue_id"]): row for row in read_jsonl(args.candidates)}
    responses = read_jsonl(args.responses)
    decisions, invalid = [], []
    seen_candidates = set()
    seen_response_ids = set()
    for response in responses:
        response_id = str(response.get("id") or "")
        request = requests.get(response_id)
        errors = []
        payload: dict[str, Any] = {}
        if response_id in seen_response_ids:
            errors.append("duplicate_response")
        seen_response_ids.add(response_id)
        if request is None:
            errors.append("unknown_request")
        elif response.get("error") or str(response.get("finish_reason") or "").lower() != "stop":
            errors.append("runtime_failure")
        elif response.get("metadata") != request.get("metadata"):
            errors.append("metadata_mismatch")
        metadata = (request or {}).get("metadata") or {}
        if request is not None and metadata.get("stage") != args.stage:
            errors.append("stage_mismatch")
        if not errors:
            try:
                payload = parse_object(response.get("response"))
                errors.extend(validate_portfolio_payload(payload, metadata))
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"parse_{type(exc).__name__}")
        if errors:
            invalid.append({
                "request_id": response.get("id"),
                "packet_id": metadata.get("packet_id"),
                "errors": sorted(set(errors)),
            })
            continue
        candidate_ids = metadata["candidate_ids"]
        packet_issue_ids = [
            str(candidate_ids[int(decision["index"]) - 1])
            for decision in payload["decisions"]
        ]
        duplicate_candidates = set(packet_issue_ids) & seen_candidates
        if duplicate_candidates:
            invalid.append({
                "request_id": response.get("id"),
                "packet_id": metadata.get("packet_id"),
                "errors": ["candidate_decided_twice"],
                "duplicate_candidate_ids": sorted(duplicate_candidates),
            })
            continue
        for decision in payload["decisions"]:
            issue_id = str(candidate_ids[int(decision["index"]) - 1])
            seen_candidates.add(issue_id)
            row = candidates[issue_id]
            decisions.append({
                "schema_version": "ksova-core-portfolio-decision-1.0",
                "stage": metadata["stage"],
                "packet_id": metadata["packet_id"],
                "issue_id": issue_id,
                "domain_id": normalized_domain(row),
                "title_ko": row.get("title_ko"),
                "portfolio_verdict": decision["verdict"],
                "reason_ko": decision["reason_ko"],
                "packet_rationale_ko": payload["packet_rationale_ko"],
            })
    for missing_id in sorted(set(requests) - seen_response_ids):
        invalid.append({
            "request_id": missing_id,
            "packet_id": requests[missing_id]["metadata"]["packet_id"],
            "errors": ["missing_response"],
        })
    decisions.sort(key=lambda row: (row["domain_id"], row["issue_id"]))
    write_jsonl(args.output, decisions)
    write_jsonl(args.invalid, invalid)
    report = {
        "schema_version": "ksova-core-portfolio-parse-report-1.0",
        "stage": args.stage,
        "requests": len(requests),
        "responses": len(responses),
        "decisions": len(decisions),
        "invalid": len(invalid),
        "by_verdict": dict(sorted(Counter(
            row["portfolio_verdict"] for row in decisions
        ).items())),
        "shortlist_by_domain": dict(sorted(Counter(
            row["domain_id"] for row in decisions
            if row["portfolio_verdict"] == "shortlist"
        ).items())),
        "canonical_catalog_mutated": False,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    expected_decisions = sum(
        len(request["metadata"]["candidate_ids"]) for request in requests.values()
    )
    return 0 if len(decisions) == expected_decisions and not invalid else 2


def command_build_portfolio_final(args: argparse.Namespace) -> int:
    candidates = {str(row["issue_id"]): row for row in read_jsonl(args.candidates)}
    round1 = read_jsonl(args.round1_decisions)
    shortlisted_ids = {
        str(row["issue_id"]) for row in round1
        if row.get("portfolio_verdict") == "shortlist"
    }
    if not shortlisted_ids <= set(candidates):
        raise ValueError("round1 shortlist references missing candidates")
    caps = json.loads(args.domain_caps_json)
    if not isinstance(caps, dict) or any(
        not isinstance(key, str) or not isinstance(value, int) or value < 0
        for key, value in caps.items()
    ):
        raise ValueError("domain caps must be a JSON object of nonnegative integers")
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for issue_id in sorted(shortlisted_ids):
        row = candidates[issue_id]
        by_domain.setdefault(normalized_domain(row), []).append(row)
    missing_caps = set(by_domain) - set(caps)
    if missing_caps:
        raise ValueError(f"missing final caps for domains: {sorted(missing_caps)}")
    requests = []
    for domain_id in sorted(by_domain):
        rows = by_domain[domain_id]
        requests.append(portfolio_request(
            domain_id, rows, min(caps[domain_id], len(rows)),
            "final", args.max_tokens,
        ))
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-core-portfolio-build-report-1.0",
        "stage": "final",
        "round1_shortlisted": len(shortlisted_ids),
        "round1_shortlisted_by_domain": dict(sorted(Counter(
            normalized_domain(candidates[issue_id]) for issue_id in shortlisted_ids
        ).items())),
        "domain_caps": dict(sorted(caps.items())),
        "requests": len(requests),
        "canonical_catalog_mutated": False,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_compile_allocation(args: argparse.Namespace) -> int:
    candidates = {str(row["issue_id"]): row for row in read_jsonl(args.candidates)}
    recommendations = {
        str(row["issue_id"]): row for row in read_jsonl(args.recommendations)
    }
    pairwise = {
        str(row["candidate_issue_id"]): row for row in read_jsonl(args.pairwise)
    }
    round1 = {str(row["issue_id"]): row for row in read_jsonl(args.round1_decisions)}
    final = {str(row["issue_id"]): row for row in read_jsonl(args.final_decisions)}
    if set(recommendations) != set(candidates):
        raise ValueError("validated recommendations do not exactly cover candidates")
    direct_ids = {
        issue_id for issue_id, row in recommendations.items()
        if row.get("final_verdict") == "core_candidate"
        and row.get("replacement_target_ids")
    }
    tournament_ids = {
        issue_id for issue_id, row in recommendations.items()
        if row.get("final_verdict") == "core_candidate"
        and not row.get("replacement_target_ids")
    }
    round1_shortlisted = {
        issue_id for issue_id, row in round1.items()
        if row.get("portfolio_verdict") == "shortlist"
    }
    if set(pairwise) != direct_ids:
        raise ValueError("pairwise decisions do not exactly cover direct candidates")
    if set(round1) != tournament_ids:
        raise ValueError("round1 decisions do not exactly cover tournament candidates")
    if set(final) != round1_shortlisted:
        raise ValueError("final decisions do not exactly cover round1 shortlist")

    allocations = []
    for issue_id in sorted(candidates):
        candidate = candidates[issue_id]
        recommendation = recommendations[issue_id]
        proposed_replacements: list[str] = []
        model_rationale = ""
        if recommendation["final_verdict"] == "supporting_train":
            if recommendation.get("replacement_target_ids"):
                lane = "benchmark_overlap_hold"
            else:
                lane = "supporting_train_candidate"
            model_rationale = recommendation["validation"]["reason_ko"]
        elif issue_id in pairwise:
            comparison = pairwise[issue_id]
            proposed_replacements = comparison.get("replace_issue_ids") or []
            if comparison["decision"] in {"replace_existing", "both_distinct_review"}:
                lane = "human_core_review"
            else:
                lane = "benchmark_overlap_hold"
            model_rationale = comparison["reason_ko"]
        else:
            first_tournament = round1[issue_id]
            decisive = final.get(issue_id, first_tournament)
            verdict = decisive["portfolio_verdict"]
            if verdict == "shortlist":
                lane = "human_core_review"
            elif verdict == "supporting_train":
                lane = "supporting_train_candidate"
            else:
                lane = "reserve"
            model_rationale = decisive["reason_ko"]
        allocation_reason = {
            "human_core_review": (
                "2단계 상대 비교에서 사람 검토 후보로 남았다. 기존 핵심 교체는 아직 결정되지 않았다."
            ),
            "benchmark_overlap_hold": (
                "기존 핵심 쟁점과 직접 겹쳐 자동 교체와 학습 투입을 모두 보류한다."
            ),
            "supporting_train_candidate": (
                "핵심 후보 압축에서는 제외됐지만 분할 오염 검사를 거쳐 학습 후보로 검토할 수 있다."
            ),
            "reserve": "핵심 및 학습 우선순위에서 제외하고 예비 쟁점으로 유지한다.",
        }[lane]
        core_validation = recommendation.get("validation") or {}
        source_validation = ((candidate.get("selection_review") or {}).get("validation") or {})
        model_rationale_flags = ["model_rationale_not_source_validated"]
        acceptable_text = json.dumps(candidate.get("acceptable_range_ko") or [], ensure_ascii=False)
        if (
            "구속력 있는 판정" in model_rationale
            and "권고" in acceptable_text and "구속력" in acceptable_text
        ):
            model_rationale_flags.append("legal_effect_overstatement_requires_review")
        if candidate.get("temporal_scope"):
            model_rationale_flags.append("time_scoped_issue")
        maximum_similarity = candidate.get("maximum_semantic_similarity")
        if isinstance(maximum_similarity, (int, float)) and maximum_similarity >= 0.9:
            model_rationale_flags.append("high_existing_semantic_similarity")
        allocations.append({
            "schema_version": "ksova-core-allocation-recommendation-1.0",
            "issue_id": issue_id,
            "domain_id": normalized_domain(candidate),
            "title_ko": candidate.get("title_ko"),
            "decision_core_ko": candidate.get("decision_core_ko"),
            "allocation_lane": lane,
            "reason_ko": allocation_reason,
            "model_selection_rationale_ko": model_rationale,
            "model_rationale_review_flags": model_rationale_flags,
            "validated_source_reason_ko": source_validation.get("reason_ko"),
            "source_validation_quality_score": source_validation.get("quality_score"),
            "core_validation_score": core_validation.get("core_score"),
            "source_ids": candidate.get("source_ids") or [],
            "source_count": len(candidate.get("source_ids") or []),
            "temporal_scope": candidate.get("temporal_scope"),
            "maximum_existing_semantic_similarity": maximum_similarity,
            "proposed_replacement_ids": proposed_replacements,
            "canonical_catalog_mutated": False,
        })
    shortlist = [
        row for row in allocations if row["allocation_lane"] == "human_core_review"
    ]
    write_jsonl(args.output, allocations)
    write_jsonl(args.shortlist_output, shortlist)
    report = {
        "schema_version": "ksova-core-allocation-report-1.0",
        "candidates": len(allocations),
        "by_lane": dict(sorted(Counter(
            row["allocation_lane"] for row in allocations
        ).items())),
        "human_core_review_by_domain": dict(sorted(Counter(
            row["domain_id"] for row in shortlist
        ).items())),
        "proposed_replacements": sum(
            len(row["proposed_replacement_ids"]) for row in shortlist
        ),
        "canonical_catalog_mutated": False,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--candidates", type=Path, required=True)
    build.add_argument("--current-issues", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.add_argument("--max-tokens", type=int, default=2200)
    build.set_defaults(func=command_build)

    parse_first = commands.add_parser("parse-first")
    parse_first.add_argument("--requests", type=Path, required=True)
    parse_first.add_argument("--responses", type=Path, required=True)
    parse_first.add_argument("--current-issues", type=Path, required=True)
    parse_first.add_argument("--output", type=Path, required=True)
    parse_first.add_argument("--invalid", type=Path, required=True)
    parse_first.add_argument("--report", type=Path, required=True)
    parse_first.set_defaults(func=command_parse_first)

    build_validation = commands.add_parser("build-validation")
    build_validation.add_argument("--candidates", type=Path, required=True)
    build_validation.add_argument("--first-judged", type=Path, required=True)
    build_validation.add_argument("--current-issues", type=Path, required=True)
    build_validation.add_argument("--output", type=Path, required=True)
    build_validation.add_argument("--report", type=Path, required=True)
    build_validation.add_argument("--max-tokens", type=int, default=1800)
    build_validation.set_defaults(func=command_build_validation)

    parse_validation = commands.add_parser("parse-validation")
    parse_validation.add_argument("--candidates", type=Path, required=True)
    parse_validation.add_argument("--first-judged", type=Path, required=True)
    parse_validation.add_argument("--requests", type=Path, required=True)
    parse_validation.add_argument("--responses", type=Path, required=True)
    parse_validation.add_argument("--current-issues", type=Path, required=True)
    parse_validation.add_argument("--output", type=Path, required=True)
    parse_validation.add_argument("--invalid", type=Path, required=True)
    parse_validation.add_argument("--report", type=Path, required=True)
    parse_validation.set_defaults(func=command_parse_validation)

    build_pairwise = commands.add_parser("build-pairwise")
    build_pairwise.add_argument("--candidates", type=Path, required=True)
    build_pairwise.add_argument("--recommendations", type=Path, required=True)
    build_pairwise.add_argument("--current-issues", type=Path, required=True)
    build_pairwise.add_argument("--output", type=Path, required=True)
    build_pairwise.add_argument("--report", type=Path, required=True)
    build_pairwise.add_argument("--max-tokens", type=int, default=1600)
    build_pairwise.set_defaults(func=command_build_pairwise)

    parse_pairwise = commands.add_parser("parse-pairwise")
    parse_pairwise.add_argument("--candidates", type=Path, required=True)
    parse_pairwise.add_argument("--requests", type=Path, required=True)
    parse_pairwise.add_argument("--responses", type=Path, required=True)
    parse_pairwise.add_argument("--current-issues", type=Path, required=True)
    parse_pairwise.add_argument("--output", type=Path, required=True)
    parse_pairwise.add_argument("--invalid", type=Path, required=True)
    parse_pairwise.add_argument("--report", type=Path, required=True)
    parse_pairwise.set_defaults(func=command_parse_pairwise)

    build_round1 = commands.add_parser("build-portfolio-round1")
    build_round1.add_argument("--candidates", type=Path, required=True)
    build_round1.add_argument("--recommendations", type=Path, required=True)
    build_round1.add_argument("--output", type=Path, required=True)
    build_round1.add_argument("--report", type=Path, required=True)
    build_round1.add_argument("--chunk-size", type=int, default=8)
    build_round1.add_argument("--max-shortlist", type=int, default=2)
    build_round1.add_argument("--max-tokens", type=int, default=2400)
    build_round1.set_defaults(func=command_build_portfolio_round1)

    parse_portfolio = commands.add_parser("parse-portfolio")
    parse_portfolio.add_argument("--stage", choices=("round1", "final"), required=True)
    parse_portfolio.add_argument("--candidates", type=Path, required=True)
    parse_portfolio.add_argument("--requests", type=Path, required=True)
    parse_portfolio.add_argument("--responses", type=Path, required=True)
    parse_portfolio.add_argument("--output", type=Path, required=True)
    parse_portfolio.add_argument("--invalid", type=Path, required=True)
    parse_portfolio.add_argument("--report", type=Path, required=True)
    parse_portfolio.set_defaults(func=command_parse_portfolio)

    build_final = commands.add_parser("build-portfolio-final")
    build_final.add_argument("--candidates", type=Path, required=True)
    build_final.add_argument("--round1-decisions", type=Path, required=True)
    build_final.add_argument("--domain-caps-json", required=True)
    build_final.add_argument("--output", type=Path, required=True)
    build_final.add_argument("--report", type=Path, required=True)
    build_final.add_argument("--max-tokens", type=int, default=2800)
    build_final.set_defaults(func=command_build_portfolio_final)

    compile_allocation = commands.add_parser("compile-allocation")
    compile_allocation.add_argument("--candidates", type=Path, required=True)
    compile_allocation.add_argument("--recommendations", type=Path, required=True)
    compile_allocation.add_argument("--pairwise", type=Path, required=True)
    compile_allocation.add_argument("--round1-decisions", type=Path, required=True)
    compile_allocation.add_argument("--final-decisions", type=Path, required=True)
    compile_allocation.add_argument("--output", type=Path, required=True)
    compile_allocation.add_argument("--shortlist-output", type=Path, required=True)
    compile_allocation.add_argument("--report", type=Path, required=True)
    compile_allocation.set_defaults(func=command_compile_allocation)
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        return int(args.func(args))
    except (ValueError, KeyError) as exc:
        print(f"core selection error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
