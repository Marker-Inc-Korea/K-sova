#!/usr/bin/env python3
"""Build compact, ecosystem-balanced Korean discourse search queries.

The output is discovery-only.  It never treats a search result, headline, or
opinion as a factual anchor.  Downstream resolvers must preserve attribution
and the position-evidence-only policy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+./-]{1,})(?:\s+[A-Za-z][A-Za-z0-9+./-]{1,}){0,2}"
    r"|[0-9]{2,}[A-Za-z가-힣%·./-]*|[가-힣][가-힣·]{1,}"
)
PARTICLES = (
    "으로부터", "에게서", "에서는", "으로서", "으로써", "까지", "부터", "처럼",
    "보다", "에게", "에서", "으로", "로서", "로써", "에는", "들의", "라는",
    "이라는", "하고", "하며", "하여", "해야", "되는", "대한", "위한", "통한",
    "이나", "거나", "인지", "이고", "이며", "에도", "만을", "만이", "만의",
    "에서의", "의", "을", "를", "이", "가", "은", "는", "에", "로", "와", "과",
)
STOPWORDS = {
    "대한민국", "한국", "정부", "국가", "사회", "공식", "문서", "자료", "원문", "관련",
    "상황", "경우", "여부", "평가", "판단", "범위", "조건", "한계", "기준", "방향",
    "대응", "정책", "입장", "문제", "쟁점", "필요", "허용", "확보", "위해", "통해",
    "그리고", "그러나", "대하여", "어떻게", "것인지", "하는", "있는", "없는", "대한",
    "과제", "연구", "장기", "단기", "사전", "분야", "종류", "모든", "확인", "조치",
    "korea", "korean", "republic", "official", "policy", "report", "government",
}

# Each lane deliberately spans institutions with different social locations.
# The resolver later labels all of these as attributed position evidence only.
LANE_ECOSYSTEMS: dict[str, tuple[tuple[str, str], ...]] = {
    "democracy_rights_values": (
        ("legislative_deliberation", "nars.go.kr"),
        ("academic_debate", "kci.go.kr"),
        ("civil_society", "peoplepower21.org"),
        ("legal_civil_society", "minbyun.or.kr"),
    ),
    "energy_environment_resources": (
        ("policy_research", "keei.re.kr"),
        ("academic_debate", "kci.go.kr"),
        ("environmental_civil_society", "kfem.or.kr"),
        ("legislative_deliberation", "nars.go.kr"),
    ),
    "finance_currency_tax": (
        ("policy_research", "kiep.go.kr"),
        ("capital_market_research", "kcmi.re.kr"),
        ("academic_debate", "kci.go.kr"),
        ("industry_constituency", "korcham.net"),
    ),
    "health_bio_social_protection": (
        ("social_policy_research", "kihasa.re.kr"),
        ("health_industry_research", "khidi.or.kr"),
        ("academic_debate", "kci.go.kr"),
        ("legislative_deliberation", "nars.go.kr"),
    ),
    "migration_citizenship_labor": (
        ("labor_policy_research", "kli.re.kr"),
        ("academic_debate", "kci.go.kr"),
        ("labor_constituency", "kctu.org"),
        ("civil_society", "peoplepower21.org"),
    ),
    "technology_data_science": (
        ("digital_policy_research", "kisdi.re.kr"),
        ("academic_debate", "kci.go.kr"),
        ("civil_society", "peoplepower21.org"),
        ("industry_constituency", "korcham.net"),
    ),
    "trade_industry_supply_chain": (
        ("trade_policy_research", "kiep.go.kr"),
        ("trade_constituency", "kita.net"),
        ("industry_constituency", "korcham.net"),
        ("legislative_deliberation", "nars.go.kr"),
    ),
    "security_alliance_defense": (
        ("security_policy_research", "kida.re.kr"),
        ("strategic_policy_research", "inss.re.kr"),
        ("academic_debate", "kci.go.kr"),
        ("legislative_deliberation", "nars.go.kr"),
    ),
    "north_korea_diplomacy_human_rights": (
        ("unification_policy_research", "kinu.or.kr"),
        ("strategic_policy_research", "inss.re.kr"),
        ("academic_debate", "kci.go.kr"),
        ("legislative_deliberation", "nars.go.kr"),
    ),
    "history_responsibility_memory": (
        ("academic_debate", "kci.go.kr"),
        ("legislative_deliberation", "nars.go.kr"),
        ("civil_society", "peoplepower21.org"),
        ("public_history_research", "nahf.or.kr"),
    ),
    "culture_language_identity": (
        ("academic_debate", "kci.go.kr"),
        ("culture_policy_research", "kcti.re.kr"),
        ("legislative_deliberation", "nars.go.kr"),
        ("civil_society", "peoplepower21.org"),
    ),
    "territory_maritime_airspace": (
        ("maritime_policy_research", "kmi.re.kr"),
        ("academic_debate", "kci.go.kr"),
        ("security_policy_research", "kida.re.kr"),
        ("legislative_deliberation", "nars.go.kr"),
    ),
    "international_order_development": (
        ("foreign_policy_research", "ifans.go.kr"),
        ("economic_policy_research", "kiep.go.kr"),
        ("academic_debate", "kci.go.kr"),
        ("legislative_deliberation", "nars.go.kr"),
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_particle(token: str) -> str:
    value = token.casefold().strip("-./")
    for suffix in PARTICLES:
        if value.endswith(suffix) and len(value) - len(suffix) >= 2:
            value = value[:-len(suffix)]
            break
    return value


def field_tokens(material: dict[str, Any]) -> list[tuple[str, float]]:
    weighted_fields = (
        (str(material.get("specific_issue_ko") or ""), 5.0),
        (str(material.get("decision_object_ko") or ""), 2.0),
        (" ".join(str(value) for value in material.get("shared_facts_ko") or [])[:1200], 0.5),
    )
    output = []
    for text, weight in weighted_fields:
        for raw in TOKEN.findall(text):
            token = strip_particle(raw)
            if len(token) < 2 or token in STOPWORDS:
                continue
            output.append((token, weight))
    return output


def core_terms(materials: list[dict[str, Any]], limit: int) -> dict[str, list[str]]:
    token_rows = {str(row["material_id"]): field_tokens(row) for row in materials}
    document_frequency: Counter[str] = Counter()
    for tokens in token_rows.values():
        document_frequency.update({token for token, _ in tokens})
    total = max(1, len(materials))
    output: dict[str, list[str]] = {}
    for material in materials:
        material_id = str(material["material_id"])
        scores: Counter[str] = Counter()
        for token, weight in token_rows[material_id]:
            rarity = math.log((total + 1) / (document_frequency[token] + 1)) + 1.0
            identifier_boost = 6.0 if any(char.isdigit() for char in token) else 0.0
            acronym_boost = 4.0 if token.isascii() and (len(token) <= 8 or " " in token) else 0.0
            compound_boost = max(0, min(len(token), 10) - 4) * 0.8
            scores[token] += (
                weight * rarity + min(len(token), 12) * 0.08
                + identifier_boost + acronym_boost + compound_boost
            )
        ranked = [
            token for token, _ in sorted(scores.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        ]
        # Avoid near-duplicate inflections/compounds consuming the query budget.
        selected: list[str] = []
        for token in ranked:
            if any(token in prior or prior in token for prior in selected):
                continue
            selected.append(token)
            if len(selected) >= limit:
                break
        output[material_id] = selected
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--terms-per-query", type=int, default=6)
    parser.add_argument("--max-query-chars", type=int, default=150)
    args = parser.parse_args()

    materials = read_jsonl(args.materials)
    if not materials:
        raise ValueError("materials are empty")
    terms_by_id = core_terms(materials, args.terms_per_query)
    query_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    for material in materials:
        material_id = str(material["material_id"])
        lane = str(material.get("lane") or "")
        ecosystems = LANE_ECOSYSTEMS.get(lane)
        if not ecosystems:
            raise ValueError(f"unsupported lane for discourse search: {lane}")
        terms = terms_by_id[material_id]
        if len(terms) < 3:
            raise ValueError(f"{material_id}: fewer than three compact terms")
        core = " ".join(terms)
        material_queries = []
        for category, domain in ecosystems:
            query = f"site:{domain} {core}"[: args.max_query_chars].rstrip()
            row = {
                "lane": lane,
                "language": "ko",
                "query": query,
                "candidate_id": material_id,
                "source_role": "korean_discourse_position",
                "jurisdiction": "KR",
                "source_variant": category,
            }
            query_rows.append(row)
            material_queries.append(row)
            category_counts[category] += 1
        # One open public-sphere query lets different Korean editorial ecosystems
        # surface; it is still resolved and policy-limited as attributed position.
        public_query = f"{core} 한국 사회 토론 사설 기고"[: args.max_query_chars].rstrip()
        public_row = {
            "lane": lane,
            "language": "ko",
            "query": public_query,
            "candidate_id": material_id,
            "source_role": "korean_discourse_position",
            "jurisdiction": "KR",
            "source_variant": "media_public_sphere",
        }
        query_rows.append(public_row)
        material_queries.append(public_row)
        category_counts["media_public_sphere"] += 1
        lane_counts[lane] += 1
        seed_rows.append({
            "material_id": material_id,
            "lane": lane,
            "evidence_status": material.get("evidence_status"),
            "specific_issue_ko": material.get("specific_issue_ko"),
            "core_terms": terms,
            "queries": material_queries,
            "policy": "query_seed_only_not_training_evidence",
        })

    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    config = {
        "schema_version": "ksova-training-discourse-search-config-1.0",
        "policy": "ecosystem_balanced_discovery_only",
        "engines": ["bing_web", "duckduckgo_web", "naver_web"],
        "discovery_only_hosts": base.get("discovery_only_hosts", []),
        "blocked_hosts": base.get("blocked_hosts", ["kp"]),
        "queries": query_rows,
    }
    write_json(args.output, config)
    write_jsonl(args.seed_output, seed_rows)
    report = {
        "schema_version": "ksova-training-discourse-search-plan-report-1.0",
        "materials": len(materials),
        "queries": len(query_rows),
        "engines": config["engines"],
        "engine_requests": len(query_rows) * len(config["engines"]),
        "by_lane": dict(sorted(lane_counts.items())),
        "by_ecosystem": dict(sorted(category_counts.items())),
        "terms_per_query": args.terms_per_query,
        "max_query_chars": args.max_query_chars,
        "output": str(args.output.resolve()),
        "seed_output": str(args.seed_output.resolve()),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
