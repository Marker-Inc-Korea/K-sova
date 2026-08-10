#!/usr/bin/env python3
"""Retrieve existing non-benchmark source bodies for new issue candidates.

This is a deterministic lexical retrieval stage, not a relevance judgment.
Every selected body remains subject to the same GLM body-role validation as
newly searched sources.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from discover_calibration_sources import base_host, trusted_jurisdiction
from training_source_expansion import normalized_url


TOKEN = re.compile(r"[0-9A-Za-z가-힣一-龥ぁ-んァ-ン]{2,}")
STOPWORDS = {
    "대한민국", "한국", "정부", "국가", "사회", "국제", "외국", "관련", "대한",
    "정책", "입장", "주장", "문제", "쟁점", "판단", "결정", "기준", "범위", "대응",
    "필요", "여부", "권리", "이익", "가치", "주권", "논쟁", "자료", "문서", "보고서",
    "korea", "korean", "south", "republic", "government", "policy", "issue", "report",
    "official", "international", "국제법", "한국의", "대해", "따른", "위한", "에서",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def tokens(value: str) -> list[str]:
    output = []
    for raw in TOKEN.findall(value.casefold()):
        term = re.sub(
            r"(?:으로부터|에서는|에게서|으로서|으로써|에서의|까지|부터|에게|에서|으로|"
            r"라는|이라는|들의|대한|위한|통한|이나|에는|에도|만의|의|을|를|이|가|은|는|에|로|와|과)$",
            "", raw,
        )
        if len(term) >= 2 and term not in STOPWORDS:
            output.append(term)
    return output


def benchmark_urls(registry: list[dict[str, Any]], issues: list[dict[str, Any]]) -> set[str]:
    by_id = {row["source_id"]: row for row in registry}
    used = {
        source_id
        for issue in issues
        for source_id in issue.get("source_ids") or []
    }
    return {
        normalized_url(str(by_id[value].get("url") or ""))
        for value in used if value in by_id and by_id[value].get("url")
    }


def source_text(row: dict[str, Any]) -> str:
    return " ".join((
        (str(row.get("title") or "") + " ") * 3,
        (str(row.get("evidence_summary_ko") or "") + " ") * 2,
        str(row.get("content_excerpt") or "")[:12000],
        " ".join(str(value) for value in row.get("evidence_claims") or [])[:3000],
    ))


def issue_text(row: dict[str, Any]) -> str:
    plan = row.get("source_search_plan") or {}
    anchors = " ".join(
        str(value)
        for role_values in plan.values() if isinstance(role_values, list)
        for value in role_values
    )
    return " ".join(str(row.get(key) or "") for key in (
        "issue_ko", "external_actor_or_frame", "competing_frame_ko",
        "korean_stake_ko", "decision_object_ko", "korean_discourse_needed_ko",
    )) + " " + anchors


def role_query_texts(row: dict[str, Any]) -> dict[str, str]:
    """Build provenance-preserving retrieval queries for each evidence role."""
    plan = row.get("source_search_plan") or {}

    def anchors(role: str) -> str:
        values = plan.get(role)
        if not isinstance(values, list):
            return ""
        return " ".join(str(value) for value in values)

    issue = str(row.get("issue_ko") or "")
    external = str(row.get("external_actor_or_frame") or "")
    competing = str(row.get("competing_frame_ko") or "")
    stake = str(row.get("korean_stake_ko") or "")
    decision = str(row.get("decision_object_ko") or "")
    discourse = str(row.get("korean_discourse_needed_ko") or "")
    return {
        "external_position": " ".join((
            issue, external, competing, anchors("external_position"),
        )),
        "korean_position": " ".join((
            issue, stake, decision, anchors("korean_position"),
        )),
        "korean_counterargument_or_critique": " ".join((
            issue, stake, discourse,
            anchors("korean_counterargument_or_critique"),
            "비판 우려 반론 한계 대안 비용 권리",
        )),
        "common_fact_or_legal_anchor": " ".join((
            issue, external, decision, anchors("common_fact_or_legal_anchor"),
        )),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--benchmark-issues", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-body-chars", type=int, default=500)
    parser.add_argument("--min-shared-terms", type=int, default=2)
    parser.add_argument(
        "--role-aware",
        action="store_true",
        help="Retrieve top-k candidates separately for each required evidence role.",
    )
    args = parser.parse_args()

    issues = read_jsonl(args.issues)
    registry = read_jsonl(args.registry)
    collision_urls = benchmark_urls(registry, read_jsonl(args.benchmark_issues))

    sources = []
    source_term_counts: dict[str, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()
    excluded = Counter()
    for row in registry:
        url = normalized_url(str(row.get("url") or ""))
        if not url or url in collision_urls:
            excluded["benchmark_url_collision" if url in collision_urls else "empty_url"] += 1
            continue
        body = str(row.get("content_excerpt") or "")
        if len(body) < args.min_body_chars:
            excluded["short_body"] += 1
            continue
        jurisdiction, tier = trusted_jurisdiction(base_host(url))
        if tier not in {"A", "B"}:
            excluded["untrusted_host"] += 1
            continue
        source_id = str(row["source_id"])
        counts = Counter(tokens(source_text(row)))
        if len(counts) < args.min_shared_terms:
            excluded["too_few_terms"] += 1
            continue
        enriched = {
            **row,
            "jurisdiction": row.get("jurisdiction") or jurisdiction,
            "reliability_tier": tier,
            "normalized_url": url,
        }
        sources.append(enriched)
        source_term_counts[source_id] = counts
        document_frequency.update(counts)

    average_length = sum(sum(value.values()) for value in source_term_counts.values()) / max(1, len(sources))
    total_documents = max(1, len(sources))
    source_by_id = {row["source_id"]: row for row in sources}
    packets: dict[str, dict[str, Any]] = {}
    issue_coverage = Counter()
    for issue in issues:
        query_specs = (
            role_query_texts(issue).items()
            if args.role_aware
            else [("body_role_unverified", issue_text(issue))]
        )
        issue_has_candidates = False
        for source_role, query_text in query_specs:
            query_counts = Counter(tokens(query_text))
            scored = []
            for source_id, term_counts in source_term_counts.items():
                shared = set(query_counts) & set(term_counts)
                if len(shared) < args.min_shared_terms:
                    continue
                length = sum(term_counts.values())
                score = 0.0
                for term in shared:
                    frequency = term_counts[term]
                    inverse_frequency = math.log(
                        1.0 + (total_documents - document_frequency[term] + 0.5)
                        / (document_frequency[term] + 0.5)
                    )
                    denominator = frequency + 1.2 * (
                        1 - 0.75 + 0.75 * length / max(1.0, average_length)
                    )
                    score += inverse_frequency * frequency * 2.2 / denominator
                # Exact numbers, acronyms, and long named terms are especially useful
                # for distinguishing the event/document from generic Korea policy text.
                distinctive = sum(
                    1 for term in shared if any(char.isdigit() for char in term)
                    or (term.isascii() and len(term) <= 12)
                    or len(term) >= 5
                )
                if distinctive == 0:
                    continue
                scored.append((score + distinctive * 0.7, len(shared), source_id))
            scored.sort(key=lambda value: (-value[0], -value[1], value[2]))
            selected = scored[: args.top_k]
            issue_has_candidates = issue_has_candidates or bool(selected)
            for rank, (score, shared_count, source_id) in enumerate(selected, 1):
                row = source_by_id[source_id]
                packet = packets.setdefault(source_id, {
                    "schema_version": "ksova-v2-trusted-discovery-source-1.0",
                    "source_lead_id": source_id,
                    "lead_ids": [],
                    "discoveries": [],
                    "resolution_method": "existing_registry_bm25_candidate",
                    "source": {
                        "schema_version": row.get("schema_version"),
                        "source_id": source_id,
                        "title": row.get("title"),
                        "url": row.get("url"),
                        "jurisdiction": row.get("jurisdiction"),
                        "authority_type": row.get("authority_type"),
                        "issuing_authority": row.get("publisher"),
                        "reliability_tier": row.get("reliability_tier"),
                        "allowed_uses": ["issue_discovery", "position_evidence", "fact_check", "generation_grounding"],
                        "evidence_policy": "existing_registry_body_requires_role_validation",
                        "content_sha256": row.get("content_sha256"),
                        "retrieval": {
                            "status": "reused_verified_snapshot",
                            "text_chars": len(str(row.get("content_excerpt") or "")),
                            "text_excerpt": str(row.get("content_excerpt") or ""),
                            "relevance_pass": True,
                            "final_url": row.get("url"),
                        },
                    },
                })
                packet["discoveries"].append({
                    "lane": issue["lane"],
                    "language": "mixed",
                    "query": f"deterministic_registry_bm25__{source_role}",
                    "engine": "registry_bm25",
                    "rank": rank,
                    "candidate_id": issue["issue_id"],
                    "source_role": source_role,
                    "jurisdiction": row.get("jurisdiction"),
                    "source_variant": "existing_nonbenchmark_registry_role_aware" if args.role_aware else "existing_nonbenchmark_registry",
                    "retrieval_score": round(score, 6),
                    "shared_terms": shared_count,
                })
        if issue_has_candidates:
            issue_coverage[issue["lane"]] += 1

    output = sorted(packets.values(), key=lambda row: row["source_lead_id"])
    write_jsonl(args.output, output)
    report = {
        "schema_version": "ksova-registry-source-retrieval-report-1.0",
        "issues": len(issues),
        "registry_sources": len(registry),
        "eligible_nonbenchmark_sources": len(sources),
        "unique_selected_sources": len(output),
        "issue_source_links": sum(len(row["discoveries"]) for row in output),
        "issues_with_candidates": sum(issue_coverage.values()),
        "coverage_by_lane": dict(sorted(issue_coverage.items())),
        "excluded": dict(sorted(excluded.items())),
        "top_k": args.top_k,
        "role_aware": args.role_aware,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
