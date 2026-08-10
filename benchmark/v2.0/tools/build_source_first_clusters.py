#!/usr/bin/env python3
"""Build cross-jurisdiction source-first clusters without an issue hypothesis."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from discover_calibration_sources import base_host, trusted_jurisdiction
from retrieve_registry_sources import tokens
from training_source_expansion import normalized_url


LANE_TERMS = {
    "territory_maritime_airspace": {"독도", "영토", "해양", "eez", "영공", "방공", "동해", "sea", "maritime", "territory"},
    "security_alliance_defense": {"안보", "동맹", "국방", "군사", "미사일", "핵무장", "주한미군", "nato", "defense", "security"},
    "trade_industry_supply_chain": {"통상", "무역", "공급망", "산업", "수출", "관세", "반도체", "trade", "tariff", "supply"},
    "technology_data_science": {"인공지능", "데이터", "플랫폼", "기술", "ai", "cyber", "digital", "semiconductor"},
    "finance_currency_tax": {"금융", "통화", "환율", "조세", "세금", "제재", "finance", "currency", "tax", "sanction"},
    "energy_environment_resources": {"에너지", "환경", "기후", "탄소", "원전", "자원", "energy", "climate", "carbon", "nuclear"},
    "health_bio_food": {"보건", "의료", "바이오", "식품", "검역", "질병", "health", "bio", "food", "vaccine"},
    "migration_citizenship_labor": {"이주", "이민", "국적", "난민", "노동", "재외동포", "migration", "citizenship", "refugee", "labor"},
    "democracy_rights_values": {"민주", "인권", "표현", "집회", "사형", "권리", "democracy", "rights", "freedom"},
    "history_responsibility_memory": {"역사", "강제징용", "위안부", "식민", "기억", "과거사", "history", "colonial", "reparations"},
    "culture_language_identity": {"문화", "언어", "한복", "김치", "유산", "정체성", "culture", "language", "heritage", "identity"},
    "north_korea_interkorean": {"북한", "남북", "평양", "탈북", "대북", "dprk", "pyongyang", "interkorean"},
    "international_law_order_development": {"국제법", "조약", "협정", "유엔", "개발", "원조", "treaty", "international", "united", "development"},
}

DISCOURSE_MARKERS = (
    "secondary", "reporting", "academic", "research", "institute", "media",
    "press", "civil", "labor", "industry", "opinion", "think",
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


def host(url: str) -> str:
    value = (urllib.parse.urlsplit(url).hostname or "").casefold()
    return value[4:] if value.startswith("www.") else value


def benchmark_urls(registry: list[dict[str, Any]], issues: list[dict[str, Any]]) -> set[str]:
    by_id = {row["source_id"]: row for row in registry}
    used = {
        source_id
        for issue in issues
        for source_id in issue.get("source_ids") or []
    }
    return {
        normalized_url(str(by_id[source_id].get("url") or ""))
        for source_id in used
        if source_id in by_id and by_id[source_id].get("url")
    }


def canonical_source(row: dict[str, Any]) -> dict[str, Any]:
    url = str(row.get("url") or "")
    inferred_jurisdiction, tier = trusted_jurisdiction(base_host(url))
    body = str(row.get("content_excerpt") or "")
    return {
        "schema_version": row.get("schema_version"),
        "source_id": str(row["source_id"]),
        "title": str(row.get("title") or ""),
        "url": url,
        "jurisdiction": row.get("jurisdiction") or inferred_jurisdiction,
        "authority_type": row.get("authority_type"),
        "issuing_authority": row.get("publisher"),
        "reliability_tier": tier,
        "allowed_uses": ["issue_discovery", "position_evidence", "fact_check", "generation_grounding"],
        "evidence_policy": "source_first_cluster_requires_glm_material_audit",
        "content_sha256": row.get("content_sha256"),
        "retrieval": {
            "status": "reused_verified_snapshot",
            "text_chars": len(body),
            "text_excerpt": body,
            "relevance_pass": True,
            "final_url": url,
        },
        "evidence_summary_ko": str(row.get("evidence_summary_ko") or ""),
    }


def source_text(row: dict[str, Any]) -> str:
    return " ".join((
        (str(row.get("title") or "") + " ") * 5,
        (str(row.get("evidence_summary_ko") or "") + " ") * 2,
        str((row.get("retrieval") or {}).get("text_excerpt") or "")[:10000],
    ))


def is_discourse(row: dict[str, Any]) -> bool:
    authority = str(row.get("authority_type") or "").casefold()
    return any(marker in authority for marker in DISCOURSE_MARKERS)


def infer_lane(values: list[dict[str, Any]]) -> str:
    counts = Counter(tokens(" ".join(
        str(row.get("title") or "") + " "
        + str((row.get("retrieval") or {}).get("text_excerpt") or "")[:1800]
        for row in values
    )))
    scored = []
    for lane, terms in LANE_TERMS.items():
        score = sum(counts[term] for term in terms)
        scored.append((score, lane))
    score, lane = max(scored)
    return lane if score else "international_law_order_development"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--benchmark-issues", type=Path, required=True)
    parser.add_argument(
        "--exclude-sources", type=Path, action="append", default=[],
        help="JSONL packets or materials whose source IDs must not be reused.",
    )
    parser.add_argument("--min-body-chars", type=int, default=500)
    parser.add_argument("--min-shared-terms", type=int, default=3)
    parser.add_argument("--min-pair-score", type=float, default=10.0)
    parser.add_argument("--min-coherence-score", type=float, default=8.0)
    parser.add_argument("--max-clusters", type=int, default=80)
    parser.add_argument("--max-source-reuse", type=int, default=2)
    parser.add_argument("--output-sources", type=Path, required=True)
    parser.add_argument("--output-query-config", type=Path, required=True)
    parser.add_argument("--output-clusters", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    registry = read_jsonl(args.registry)
    collisions = benchmark_urls(registry, read_jsonl(args.benchmark_issues))
    excluded_source_ids = set()
    for path in args.exclude_sources:
        for row in read_jsonl(path):
            excluded_source_ids.update(str(value) for value in row.get("source_ids") or [])
            nested = row.get("source") if isinstance(row.get("source"), dict) else row
            if nested.get("source_id"):
                excluded_source_ids.add(str(nested["source_id"]))
    excluded = Counter()
    sources = []
    seen_urls = set()
    for row in registry:
        if str(row.get("source_id") or "") in excluded_source_ids:
            excluded["prior_wave_source"] += 1
            continue
        value = canonical_source(row)
        url = normalized_url(value["url"])
        if not url or url in collisions or url in seen_urls:
            excluded[
                "benchmark_url_collision" if url in collisions
                else "duplicate_url" if url in seen_urls else "empty_url"
            ] += 1
            continue
        seen_urls.add(url)
        body = str(value["retrieval"]["text_excerpt"] or "")
        if len(body) < args.min_body_chars:
            excluded["short_body"] += 1
            continue
        if value["reliability_tier"] not in {"A", "B"}:
            excluded["untrusted_host"] += 1
            continue
        sources.append(value)

    term_counts = {row["source_id"]: Counter(tokens(source_text(row))) for row in sources}
    document_frequency = Counter()
    for counts in term_counts.values():
        document_frequency.update(counts)
    total = max(1, len(sources))
    idf = {
        term: math.log((total + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }
    by_id = {row["source_id"]: row for row in sources}
    korean = [row for row in sources if row.get("jurisdiction") == "KR"]
    external = [row for row in sources if row.get("jurisdiction") != "KR"]

    def pair_score(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, set[str]]:
        shared = set(term_counts[left["source_id"]]) & set(term_counts[right["source_id"]])
        distinctive = {
            term for term in shared
            if document_frequency[term] <= max(12, int(total * 0.12))
            and (
                any(char.isdigit() for char in term)
                or (term.isascii() and 2 <= len(term) <= 16)
                or len(term) >= 4
            )
        }
        if len(distinctive) < args.min_shared_terms:
            return 0.0, set()
        title_shared = set(tokens(str(left.get("title") or ""))) & set(
            tokens(str(right.get("title") or ""))
        )
        score = sum(idf[term] for term in distinctive) + 2.0 * sum(
            idf.get(term, 0.0) for term in title_shared & distinctive
        )
        return score, distinctive

    proposals = []
    for outer in external:
        ranked = []
        for inner in korean:
            if host(outer["url"]) == host(inner["url"]):
                continue
            score, shared = pair_score(outer, inner)
            if score >= args.min_pair_score:
                ranked.append((score, inner, shared))
        ranked.sort(key=lambda item: (-item[0], item[1]["source_id"]))
        if len(ranked) < 2:
            continue
        # Choose a coherent clique, not merely three documents that each have
        # a weak lexical relation to the foreign seed.  Every Korean document
        # must also share a body-level rare-term core with every other member.
        selected = None
        selected_coherence = 0.0
        pool = ranked[:14]
        for size in (3, 2):
            best = None
            for combination in itertools.combinations(pool, size):
                rows = [value[1] for value in combination]
                hosts = {host(outer["url"]), *(host(row["url"]) for row in rows)}
                if len(hosts) != size + 1 or not any(is_discourse(row) for row in rows):
                    continue
                coherence_scores = []
                for left, right in itertools.combinations(rows, 2):
                    score, _ = pair_score(left, right)
                    coherence_scores.append(score)
                coherence = min(coherence_scores, default=args.min_coherence_score)
                if coherence < args.min_coherence_score:
                    continue
                value = min(item[0] for item in combination) + coherence
                if best is None or value > best[0]:
                    best = (value, combination, coherence)
            if best is not None:
                selected = list(best[1])
                selected_coherence = best[2]
                break
        if selected is None:
            continue
        values = [outer] + [row for _, row, _ in selected]
        signature_terms = set.intersection(*(shared for _, _, shared in selected))
        if len(signature_terms) < 2:
            signature_terms = set.union(*(shared for _, _, shared in selected))
        signature = tuple(sorted(signature_terms, key=lambda term: (-idf.get(term, 0.0), term))[:8])
        if len(signature) < 2:
            continue
        score = (
            min(value[0] for value in selected)
            + selected_coherence
            + sum(idf.get(term, 0.0) for term in signature[:4])
        )
        proposals.append({
            "score": score,
            "sources": values,
            "signature": signature,
            "lane": infer_lane(values),
        })

    proposals.sort(key=lambda row: (-row["score"], row["signature"]))
    selected_clusters = []
    source_reuse = Counter()
    accepted_signatures: list[set[str]] = []
    for proposal in proposals:
        ids = [row["source_id"] for row in proposal["sources"]]
        if any(source_reuse[source_id] >= args.max_source_reuse for source_id in ids):
            continue
        signature_set = set(proposal["signature"])
        if any(
            len(signature_set & prior) / max(1, len(signature_set | prior)) >= 0.75
            for prior in accepted_signatures
        ):
            continue
        cluster_id = "KSSFC-" + hashlib.sha256(
            json.dumps(ids, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        selected_clusters.append({**proposal, "cluster_id": cluster_id})
        accepted_signatures.append(signature_set)
        source_reuse.update(ids)
        if len(selected_clusters) >= args.max_clusters:
            break

    packets: dict[str, dict[str, Any]] = {}
    queries = []
    cluster_rows = []
    for cluster in selected_clusters:
        query = "source_first_body_cluster " + " ".join(cluster["signature"])
        cluster_rows.append({
            "cluster_id": cluster["cluster_id"],
            "lane": cluster["lane"],
            "score": round(cluster["score"], 6),
            "signature_terms": list(cluster["signature"]),
            "source_ids": [row["source_id"] for row in cluster["sources"]],
            "source_titles": [row["title"] for row in cluster["sources"]],
            "source_jurisdictions": [row["jurisdiction"] for row in cluster["sources"]],
        })
        queries.append({
            "lane": cluster["lane"],
            "language": "mixed",
            "query": query,
            "engines": ["source_first_registry_graph"],
            "candidate_id": cluster["cluster_id"],
            "source_role": "source_first_body_cluster",
            "source_variant": "cross_jurisdiction_rare_term_cluster",
            "jurisdiction": "mixed",
        })
        for rank, item in enumerate(cluster["sources"], 1):
            packet = packets.setdefault(item["source_id"], {
                "schema_version": "ksova-v2-trusted-discovery-source-1.0",
                "source_lead_id": item["source_id"],
                "lead_ids": [],
                "discoveries": [],
                "resolution_method": "source_first_registry_graph",
                "source": item,
            })
            packet["discoveries"].append({
                "lane": cluster["lane"],
                "language": "mixed",
                "query": query,
                "engine": "source_first_registry_graph",
                "rank": rank,
                "candidate_id": cluster["cluster_id"],
                "source_role": (
                    "external_body" if item.get("jurisdiction") != "KR"
                    else "korean_discourse_body" if is_discourse(item)
                    else "korean_official_or_institutional_body"
                ),
                "jurisdiction": item.get("jurisdiction"),
                "source_variant": "cross_jurisdiction_rare_term_cluster",
                "retrieval_score": round(cluster["score"], 6),
            })

    source_output = sorted(packets.values(), key=lambda row: row["source_lead_id"])
    write_jsonl(args.output_sources, source_output)
    write_jsonl(args.output_clusters, cluster_rows)
    write_json(args.output_query_config, {
        "schema_version": "ksova-source-first-query-config-1.0",
        "provenance_route": "source_first_body_cluster",
        "evidence_eligibility": "none_until_material_generation_and_independent_audit",
        "queries": queries,
    })
    report = {
        "schema_version": "ksova-source-first-cluster-report-1.0",
        "registry_sources": len(registry),
        "eligible_nonbenchmark_sources": len(sources),
        "korean_sources": len(korean),
        "external_sources": len(external),
        "proposals": len(proposals),
        "selected_clusters": len(selected_clusters),
        "unique_cluster_sources": len(source_output),
        "cluster_source_links": sum(len(row["source_ids"]) for row in cluster_rows),
        "by_lane": dict(sorted(Counter(row["lane"] for row in cluster_rows).items())),
        "excluded": dict(sorted(excluded.items())),
        "explicit_excluded_source_ids": len(excluded_source_ids),
        "min_shared_terms": args.min_shared_terms,
        "min_pair_score": args.min_pair_score,
        "min_coherence_score": args.min_coherence_score,
        "max_source_reuse": args.max_source_reuse,
        "output_sources": str(args.output_sources),
        "output_query_config": str(args.output_query_config),
        "output_clusters": str(args.output_clusters),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
