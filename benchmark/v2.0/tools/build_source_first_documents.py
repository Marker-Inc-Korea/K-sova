#!/usr/bin/env python3
"""Select unused, source-first documents for independent K-SOVA extraction.

This stage deliberately does not invent an issue hypothesis or force unrelated
documents into a bundle.  It ranks already fetched non-benchmark documents by
title-level cross-border conflict signals, then gives every selected document
its own candidate ID.  The downstream material extractor and independent
material audit remain the only semantic model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from discover_calibration_sources import base_host, trusted_jurisdiction
from training_source_expansion import normalized_url


KOREA_TERMS = (
    "한국", "대한민국", "서울", "한미", "한일", "한중", "남북", "북한",
    "korea", "korean", "seoul", "rok", "dprk", "north korea", "south korea",
)

EXTERNAL_TERMS = (
    "미국", "미 ", "중국", "일본", "러시아", "우크라이나", "유엔", "유럽연합",
    "eu", "u.s", "united states", "china", "japan", "russia", "ukraine",
    "un ", "wto", "iaea", "unesco", "nato", "canada", "vietnam", "foreign",
)

CONFLICT_TERMS = (
    "갈등", "분쟁", "충돌", "마찰", "논란", "분노", "왜곡", "주장", "압박", "위협", "반대", "항의", "제소", "보복",
    "규제", "제한", "금지", "관세", "수출통제", "제재", "감시", "침해", "구금",
    "체포", "송환", "난민", "배상", "책임", "강제", "학살", "식민", "징용",
    "위안부", "독도", "영토", "해양", "방공", "전작권", "방위비", "핵무장",
    "원자력협정", "공급망", "반도체", "지도", "데이터", "문화유산", "기원",
    "dispute", "conflict", "pressure", "threat", "oppose", "protest", "retaliat",
    "restriction", "control", "ban", "tariff", "sanction", "detain", "arrest",
    "repatriat", "refugee", "compensation", "liability", "forced", "massacre",
    "colonial", "dokdo", "territor", "maritime", "opcon", "burden", "nuclear",
    "supply chain", "semiconductor", "chip", "map data", "heritage", "origin",
    "rights", "sovereign", "visa", "currency", "trade", "export", "import",
)

LANE_TERMS = {
    "territory_maritime_airspace": (
        "독도", "영토", "해양", "이어도", "동해", "일본해", "방공", "nll", "dokdo", "territor",
        "maritime", "border", "adiz", "sea of japan", "east sea",
    ),
    "security_alliance_defense": (
        "안보", "동맹", "국방", "미사일", "전작권", "방위비", "핵무장", "군사",
        "security", "alliance", "defense", "missile", "opcon", "troop", "nuclear weapon",
    ),
    "trade_industry_supply_chain": (
        "관세", "통상", "무역", "수출", "수입", "공급망", "반도체", "방산",
        "tariff", "trade", "export", "import", "supply chain", "chip", "industry",
    ),
    "technology_data_science": (
        "데이터", "지도", "인공지능", "플랫폼", "망사용료", "기술", "ai ",
        "data", "map", "platform", "technology", "semiconductor",
    ),
    "finance_currency_tax": (
        "환율", "통화", "금융", "세금", "조세", "currency", "exchange", "finance", "tax",
    ),
    "energy_environment_resources": (
        "원전", "원자력", "오염수", "탄소", "기후", "에너지", "핵연료", "광물",
        "nuclear", "fukushima", "carbon", "climate", "energy", "mineral",
    ),
    "health_bio_food": (
        "수산물", "검역", "식품", "보건", "의료", "seafood", "food", "health", "vaccine",
    ),
    "migration_citizenship_labor": (
        "이민", "비자", "구금", "체포", "송환", "난민", "노동", "입양",
        "immigration", "visa", "detain", "arrest", "repatriat", "refugee", "labor", "adoptee",
    ),
    "democracy_rights_values": (
        "민주", "인권", "표현", "국가보안법", "사형", "권리", "계엄",
        "democracy", "human rights", "freedom", "death penalty", "martial law",
    ),
    "history_responsibility_memory": (
        "강제징용", "위안부", "식민", "학살", "배상", "과거사", "사도광산",
        "forced labor", "comfort women", "colonial", "massacre", "compensation", "history",
    ),
    "culture_language_identity": (
        "김치", "한복", "문화", "유산", "기원", "동북공정", "칠지도",
        "kimchi", "hanbok", "culture", "heritage", "origin", "identity",
    ),
    "north_korea_interkorean": (
        "북한", "남북", "탈북", "평양", "대북", "dprk", "north korea", "inter-korean",
    ),
    "international_law_order_development": (
        "국제법", "조약", "협정", "유엔", "wto", "treaty", "agreement", "united nations",
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def contains_any(value: str, needles: Iterable[str]) -> bool:
    lowered = value.casefold()
    return any(needle.casefold() in lowered for needle in needles)


def source_host(url: str) -> str:
    value = (urllib.parse.urlsplit(url).hostname or "").casefold()
    return value[4:] if value.startswith("www.") else value


def title_terms(value: str) -> set[str]:
    return set(re.findall(r"[가-힣]{2,}|[a-z0-9]{3,}", value.casefold()))


def benchmark_urls(registry: list[dict[str, Any]], issues: list[dict[str, Any]]) -> set[str]:
    by_id = {str(row.get("source_id")): row for row in registry}
    used = {str(source_id) for issue in issues for source_id in issue.get("source_ids") or []}
    return {
        normalized_url(str(by_id[source_id].get("url") or ""))
        for source_id in used if source_id in by_id and by_id[source_id].get("url")
    }


def excluded_source_ids(paths: list[Path]) -> set[str]:
    output: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            output.update(str(value) for value in row.get("source_ids") or [])
            nested = row.get("source") if isinstance(row.get("source"), dict) else row
            if nested.get("source_id"):
                output.add(str(nested["source_id"]))
    return output


def provenance_retrieval(row: dict[str, Any]) -> dict[str, Any]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return provenance.get("retrieval") if isinstance(provenance.get("retrieval"), dict) else {}


def reliability(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    tier = provenance.get("reliability_tier")
    if tier in {"A", "B"}:
        return tier
    _, inferred = trusted_jurisdiction(base_host(str(row.get("url") or "")))
    return inferred


def infer_lane(title: str, body: str, discoveries: list[dict[str, Any]]) -> str:
    # The old discovery query can be noisy.  Prefer the verified document's
    # own title and opening body, weighting the title heavily; use the old lane
    # only if the document itself has no recognizable lane signal.
    title_value = title.casefold()
    body_value = body[:2500].casefold()
    title_scored = [(
        sum(title_value.count(term.casefold()) for term in terms), lane
    ) for lane, terms in LANE_TERMS.items()]
    title_score, title_lane = max(title_scored)
    if title_score:
        return title_lane
    body_scored = [(
        sum(body_value.count(term.casefold()) for term in terms), lane
    ) for lane, terms in LANE_TERMS.items()]
    body_score, body_lane = max(body_scored)
    if body_score:
        return body_lane
    discovery_lanes = [str(value.get("lane")) for value in discoveries if value.get("lane") in LANE_TERMS]
    return Counter(discovery_lanes).most_common(1)[0][0] if discovery_lanes else "international_law_order_development"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--benchmark-issues", type=Path, required=True)
    parser.add_argument("--exclude-sources", type=Path, action="append", default=[])
    parser.add_argument("--max-documents", type=int, default=48)
    parser.add_argument("--max-per-lane", type=int, default=8)
    parser.add_argument("--max-per-host", type=int, default=4)
    parser.add_argument("--max-per-origin-query", type=int, default=2)
    parser.add_argument("--min-body-chars", type=int, default=700)
    parser.add_argument("--output-sources", type=Path, required=True)
    parser.add_argument("--output-query-config", type=Path, required=True)
    parser.add_argument("--output-documents", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    registry = read_jsonl(args.registry)
    collision_urls = benchmark_urls(registry, read_jsonl(args.benchmark_issues))
    excluded_ids = excluded_source_ids(args.exclude_sources)
    excluded = Counter()
    candidates = []
    seen_urls = set()
    for row in registry:
        source_id = str(row.get("source_id") or "")
        url = normalized_url(str(row.get("url") or ""))
        if source_id in excluded_ids:
            excluded["prior_wave_source"] += 1
            continue
        if not url or url in collision_urls or url in seen_urls:
            excluded["benchmark_url_collision" if url in collision_urls else "duplicate_or_empty_url"] += 1
            continue
        seen_urls.add(url)
        if "expansion_material" not in (row.get("usage") or []):
            excluded["not_expansion_material"] += 1
            continue
        body = str(row.get("content_excerpt") or "")
        if len(body) < args.min_body_chars:
            excluded["short_body"] += 1
            continue
        tier = reliability(row)
        if tier not in {"A", "B"}:
            excluded["untrusted_host"] += 1
            continue
        title = str(row.get("title") or "").strip()
        discoveries = (row.get("provenance") or {}).get("discoveries") or []
        # Gate on the title itself.  Summaries and long bodies are useful for
        # ranking, but allowing them to satisfy the gate admitted generic
        # encyclopedia/database pages whose titles did not identify a decision.
        title_korea = contains_any(title, KOREA_TERMS)
        title_external = contains_any(title, EXTERNAL_TERMS)
        title_conflict = contains_any(title, CONFLICT_TERMS)
        bilateral = contains_any(title, ("한미", "한일", "한중", "남북", "korea-us", "korea-japan", "u.s.-rok", "us-rok"))
        if not (title_conflict and (bilateral or (title_korea and title_external))):
            excluded["weak_title_conflict_signal"] += 1
            continue
        ranks = [value.get("rank") for value in discoveries if isinstance(value.get("rank"), int)]
        best_rank = min(ranks, default=50)
        ranked_discoveries = sorted(
            discoveries,
            key=lambda value: (
                value.get("rank") if isinstance(value.get("rank"), int) else 999,
                str(value.get("query") or ""),
            ),
        )
        origin_query = str(ranked_discoveries[0].get("query") or title) if ranked_discoveries else title
        score = (
            8 * int(title_korea) + 8 * int(title_external) + 10 * int(title_conflict) + 6 * int(bilateral)
            + 4 * int(tier == "A") + max(0, 6 - best_rank / 10)
            + min(4, len(body) / 5000)
        )
        candidates.append({
            "row": row,
            "source_id": source_id,
            "title": title,
            "url": url,
            "host": source_host(url),
            "tier": tier,
            "body": body,
            "discoveries": discoveries,
            "origin_query": origin_query,
            "lane": infer_lane(title, body, discoveries),
            "score": score,
        })

    candidates.sort(key=lambda value: (-value["score"], value["source_id"]))
    selected = []
    lane_counts = Counter()
    host_counts = Counter()
    query_counts = Counter()
    prior_titles: list[set[str]] = []
    for candidate in candidates:
        if lane_counts[candidate["lane"]] >= args.max_per_lane:
            excluded["lane_cap"] += 1
            continue
        if host_counts[candidate["host"]] >= args.max_per_host:
            excluded["host_cap"] += 1
            continue
        if query_counts[candidate["origin_query"]] >= args.max_per_origin_query:
            excluded["origin_query_cap"] += 1
            continue
        current_terms = title_terms(candidate["title"])
        if any(len(current_terms & prior) / max(1, len(current_terms | prior)) >= 0.75 for prior in prior_titles):
            excluded["near_duplicate_title"] += 1
            continue
        selected.append(candidate)
        lane_counts[candidate["lane"]] += 1
        host_counts[candidate["host"]] += 1
        query_counts[candidate["origin_query"]] += 1
        prior_titles.append(current_terms)
        if len(selected) >= args.max_documents:
            break

    packets = []
    queries = []
    document_rows = []
    for candidate in selected:
        row = candidate["row"]
        source_id = candidate["source_id"]
        candidate_id = "KSSFD-" + hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:16]
        query = "source_first_document " + candidate["title"]
        source = {
            "schema_version": row.get("schema_version"),
            "source_id": source_id,
            "title": candidate["title"],
            "url": str(row.get("url") or ""),
            "jurisdiction": row.get("jurisdiction"),
            "authority_type": row.get("authority_type"),
            "issuing_authority": row.get("publisher"),
            "reliability_tier": candidate["tier"],
            "allowed_uses": ["issue_discovery", "position_evidence", "fact_check", "generation_grounding"],
            "evidence_policy": "single_document_requires_glm_material_audit",
            "content_sha256": row.get("content_sha256"),
            "retrieval": {
                "status": "reused_verified_snapshot",
                "text_chars": len(candidate["body"]),
                "text_excerpt": candidate["body"],
                "relevance_pass": True,
                "final_url": str(row.get("url") or ""),
            },
            "evidence_summary_ko": str(row.get("evidence_summary_ko") or ""),
        }
        discovery = {
            "lane": candidate["lane"],
            "language": "mixed",
            "query": query,
            "engine": "source_first_verified_document",
            "rank": 1,
            "candidate_id": candidate_id,
            "source_role": "bilateral_or_cross_border_body",
            "jurisdiction": row.get("jurisdiction"),
            "source_variant": "independent_verified_document",
            "retrieval_score": round(candidate["score"], 6),
        }
        packets.append({
            "schema_version": "ksova-v2-trusted-discovery-source-1.0",
            "source_lead_id": source_id,
            "lead_ids": [],
            "discoveries": [discovery],
            "resolution_method": "source_first_verified_document",
            "source": source,
        })
        queries.append({
            "lane": candidate["lane"],
            "language": "mixed",
            "query": query,
            "engines": ["source_first_verified_document"],
            "candidate_id": candidate_id,
            "source_role": "bilateral_or_cross_border_body",
            "source_variant": "independent_verified_document",
            "jurisdiction": row.get("jurisdiction"),
        })
        document_rows.append({
            "candidate_id": candidate_id,
            "source_id": source_id,
            "lane": candidate["lane"],
            "score": round(candidate["score"], 6),
            "title": candidate["title"],
            "host": candidate["host"],
            "jurisdiction": row.get("jurisdiction"),
        })

    write_jsonl(args.output_sources, packets)
    write_jsonl(args.output_documents, document_rows)
    write_json(args.output_query_config, {
        "schema_version": "ksova-source-first-query-config-1.0",
        "provenance_route": "source_first_verified_document",
        "evidence_eligibility": "none_until_material_generation_and_independent_audit",
        "queries": queries,
    })
    report = {
        "schema_version": "ksova-source-first-document-report-1.0",
        "registry_sources": len(registry),
        "eligible_candidates": len(candidates),
        "selected_documents": len(selected),
        "by_lane": dict(sorted(lane_counts.items())),
        "by_host": dict(sorted(host_counts.items())),
        "origin_queries_selected": len(query_counts),
        "explicit_excluded_source_ids": len(excluded_ids),
        "benchmark_collision_urls": len(collision_urls),
        "excluded": dict(sorted(excluded.items())),
        "output_sources": str(args.output_sources),
        "output_query_config": str(args.output_query_config),
        "output_documents": str(args.output_documents),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
