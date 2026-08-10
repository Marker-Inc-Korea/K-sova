#!/usr/bin/env python3
"""Prepare fail-closed source packets for K-SOVA training-data expansion.

This module deliberately does not repair rejected material cards.  A rejected
card's source request may be used only as a discovery query; any later material
must be generated afresh from fetched source bodies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TARGET_LANES = (
    "democracy_rights_values",
    "energy_environment_resources",
    "finance_currency_tax",
    "health_bio_social_protection",
    "migration_citizenship_labor",
    "technology_data_science",
)

LANE_KEYWORDS = {
    "democracy_rights_values": (
        "민주", "인권", "기본권", "표현", "언론", "집회", "선거", "정당",
        "사생활", "감시", "차별", "종교", "양심", "헌법재판", "human rights",
        "freedom", "privacy", "democracy",
    ),
    "energy_environment_resources": (
        "에너지", "원전", "원자력", "핵연료", "기후", "탄소", "배출", "대기",
        "환경", "광물", "자원", "lng", "수소", "오염", "전력", "재생에너지",
        "climate", "emission", "nuclear", "mineral",
    ),
    "finance_currency_tax": (
        "환율", "통화", "외환", "조세", "세금", "금융", "은행", "연금", "재정",
        "디지털세", "자본", "과세", "currency", "exchange rate", "tax", "bank",
        "financial",
    ),
    "health_bio_social_protection": (
        "의약", "약가", "보건", "건강", "감염", "백신", "바이오", "식품", "검역",
        "병원체", "공중보건", "의료", "유전자", "생명", "건강보험", "pandemic",
        "pathogen", "vaccine", "medicine", "public health",
    ),
    "migration_citizenship_labor": (
        "이민", "난민", "국적", "재외동포", "이주", "노동", "근로자", "송환",
        "비자", "영사", "복수국적", "고용", "migration", "refugee", "nationality",
        "worker", "labor", "labour",
    ),
    "technology_data_science": (
        "인공지능", " ai ", "데이터", "반도체", "플랫폼", "디지털", "사이버", "기술",
        "우주", "연구보안", "클라우드", "통신", "알고리즘", "개인정보", "특허",
        "semiconductor", "technology", "cyber", "cloud", "artificial intelligence",
    ),
    "culture_language_identity": (
        "문화", "언어", "한복", "김치", "문화유산", "유네스코", "표기", "지명",
        "콘텐츠", "미디어", "cultural", "heritage", "language",
    ),
    "history_responsibility_memory": (
        "역사", "식민", "강제동원", "위안부", "배상", "사과", "기억", "독재",
        "민주화운동", "과거사", "historical", "colonial", "reparations",
    ),
    "international_order_development": (
        "개발협력", "공적개발원조", "oda", "국제질서", "다자주의", "유엔 평화유지",
        "국제개발", "원조", "multilateral", "development cooperation",
    ),
    "north_korea_diplomacy_human_rights": (
        "북한", "비핵화", "대북", "한반도", "국군포로", "납북", "탈북", "북핵",
        "dprk", "north korea", "denuclearization",
    ),
    "security_alliance_defense": (
        "안보", "국방", "방위", "동맹", "주한미군", "군사", "무기", "억지", "전작권",
        "사이버공격", "defense", "defence", "security", "alliance", "military",
    ),
    "territory_maritime_airspace": (
        "영토", "영유권", "독도", "이어도", "해양", "영공", "대륙붕", "관할권", "경계",
        "territory", "maritime", "airspace", "continental shelf",
    ),
    "trade_industry_supply_chain": (
        "통상", "무역", "관세", "수출", "수입", "공급망", "산업", "보조금", "무역흑자",
        "301조", "section 301", "trade", "tariff", "supply chain", "export",
    ),
}

DOMAIN_LANE_BIAS = {
    "D02": "democracy_rights_values",
    "D08": "technology_data_science",
}

SITE_HINTS = (
    (re.compile(r"헌법재판소|헌재"), "ccourt.go.kr", "KR"),
    (re.compile(r"대법원|법원"), "scourt.go.kr", "KR"),
    (re.compile(r"국회|의원실|입법조사처"), "assembly.go.kr", "KR"),
    (re.compile(r"외교부"), "mofa.go.kr", "KR"),
    (re.compile(r"방위사업청"), "dapa.go.kr", "KR"),
    (re.compile(r"국방부"), "mnd.go.kr", "KR"),
    (re.compile(r"산업통상자원부|산업부|산업통상부"), "motie.go.kr", "KR"),
    (re.compile(r"환경부|기후에너지환경부"), "me.go.kr", "KR"),
    (re.compile(r"보건복지부"), "mohw.go.kr", "KR"),
    (re.compile(r"질병관리청"), "kdca.go.kr", "KR"),
    (re.compile(r"식품의약품안전처|식약처"), "mfds.go.kr", "KR"),
    (re.compile(r"개인정보보호위원회"), "pipc.go.kr", "KR"),
    (re.compile(r"과학기술정보통신부|과기정통부"), "msit.go.kr", "KR"),
    (re.compile(r"한국은행"), "bok.or.kr", "KR"),
    (re.compile(r"기획재정부|재정경제부"), "moef.go.kr", "KR"),
    (re.compile(r"금융위원회"), "fsc.go.kr", "KR"),
    (re.compile(r"고용노동부"), "moel.go.kr", "KR"),
    (re.compile(r"법무부"), "moj.go.kr", "KR"),
    (re.compile(r"국가인권위원회"), "humanrights.go.kr", "KR"),
    (re.compile(r"\bWTO\b|세계무역기구", re.I), "wto.org", "INTL"),
    (re.compile(r"\bWHO\b|세계보건기구", re.I), "who.int", "INTL"),
    (re.compile(r"\bILO\b|국제노동기구", re.I), "ilo.org", "INTL"),
    (re.compile(r"\bOECD\b", re.I), "oecd.org", "INTL"),
    (re.compile(r"\bIMF\b|국제통화기금", re.I), "imf.org", "INTL"),
    (re.compile(r"\bUSTR\b|미국 무역대표부", re.I), "ustr.gov", "US"),
    (re.compile(r"미국 재무부"), "home.treasury.gov", "US"),
    (re.compile(r"\bOHCHR\b|유엔 인권", re.I), "ohchr.org", "INTL"),
    (re.compile(r"\bUNFCCC\b|유엔기후변화", re.I), "unfccc.int", "INTL"),
    (re.compile(r"\bIAEA\b|국제원자력기구", re.I), "iaea.org", "INTL"),
)

ACCEPTED_RETRIEVAL = {
    "fetched", "reused_verified_snapshot", "reachable",
    "dossier_snapshot_verified", "current_v2_card",
}

BINDING_STOPWORDS = {
    "대한민국", "한국", "정부", "공식", "원문", "문서", "자료", "보고서", "관련",
    "확인", "검증", "필요", "필요함", "보강", "구체", "내용", "입장", "정책", "근거",
    "평가", "사안", "여부", "해당", "발표", "분석", "국가", "기관", "사회", "국제",
    "korea", "korean", "republic", "official", "document", "report", "policy",
}

LISTING_PATH = re.compile(
    r"/(?:news|news-statements|newshq-ko|high-commissioner-reports-ko|"
    r"reports?|issues?|topics?|archive|search|index)/?$",
    re.IGNORECASE,
)

TRACKING_QUERY_KEYS = {"fbclid", "gclid"}
DOCUMENT_ID_QUERY_KEYS = {
    "admrulseq", "arti_id", "articleid", "artiid", "artclno", "bcidx", "bbsid",
    "bbsseq", "bbsseqno", "boardid", "boardmasterid", "brdbltno", "brdscnbltno",
    "cbidx", "contentid", "detcseq", "docseq", "enc", "eventno", "fileid", "fileseq", "id",
    "idx", "idxno", "jono", "levelid", "list_no", "listno", "lsid", "lsiseq",
    "mst", "ncd", "newsid", "newsitemid", "no", "nttid", "nttseqno", "precseq",
    "seq", "serearticlesearchbean.artiid", "sno",
}
PATH_SESSION = re.compile(r";jsessionid=[^/?;]+", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
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


def normalized_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value.strip())
    if not parsed.netloc:
        return ""
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    path = PATH_SESSION.sub("", parsed.path).rstrip("/")
    query = [
        (key, val)
        for key, val in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if val
        and not key.casefold().startswith("utm_")
        and key.casefold() not in TRACKING_QUERY_KEYS
        and key.casefold() not in {"jsessionid", "sessionid"}
    ]
    # Korean government CMS detail pages often share one path and use a query
    # identifier (boardId, nttId, bcIdx, ...).  Retain those identifiers rather
    # than collapsing every document on the endpoint, while discarding menu,
    # pagination, and search-state parameters when a document ID is present.
    identity_query = [
        (key, val) for key, val in query
        if key.casefold() in DOCUMENT_ID_QUERY_KEYS
    ]
    if identity_query:
        query = identity_query
    return urllib.parse.urlunsplit((
        parsed.scheme.casefold(), host, path,
        urllib.parse.urlencode(sorted(query)), "",
    ))


def source_row(wrapper: dict[str, Any]) -> dict[str, Any]:
    nested = wrapper.get("source")
    return nested if isinstance(nested, dict) else wrapper


def content_digest(source: dict[str, Any]) -> str:
    direct = str(source.get("content_sha256") or "")
    if direct:
        return direct
    retrieval = source.get("retrieval") if isinstance(source.get("retrieval"), dict) else {}
    text = str(retrieval.get("text_excerpt") or source.get("content_excerpt") or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def source_keys(source: dict[str, Any]) -> set[str]:
    keys = set()
    if source_id := str(source.get("source_id") or ""):
        keys.add("id:" + source_id)
    if url := normalized_url(str(source.get("url") or "")):
        keys.add("url:" + url)
    if digest := content_digest(source):
        keys.add("sha256:" + digest)
    return keys


def registry_index(rows: list[dict[str, Any]]) -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]]
]:
    by_id: dict[str, dict[str, Any]] = {}
    by_url: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in by_id:
            raise ValueError(f"missing or duplicate canonical source_id: {source_id!r}")
        by_id[source_id] = row
        url = normalized_url(str(row.get("url") or ""))
        if url and url not in by_url:
            by_url[url] = row
    return by_id, by_url


def benchmark_source_keys(
    issues: list[dict[str, Any]], registry: dict[str, dict[str, Any]]
) -> tuple[set[str], set[str]]:
    source_ids = {
        str(source_id)
        for issue in issues
        for source_id in issue.get("source_ids") or []
        if source_id
    }
    missing = source_ids - set(registry)
    if missing:
        raise ValueError(f"benchmark sources missing from registry: {sorted(missing)[:5]}")
    keys = set()
    for source_id in source_ids:
        keys.update(source_keys(registry[source_id]))
    return source_ids, keys


def canonical_tier(source: dict[str, Any]) -> str:
    provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
    tier = str(provenance.get("reliability_tier") or source.get("reliability_tier") or "")
    if tier in {"A", "B"}:
        return tier
    authority = str(source.get("authority_type") or "").casefold()
    secondary_markers = ("secondary", "news", "media", "academic", "research")
    return "B" if any(marker in authority for marker in secondary_markers) else "A"


def canonical_packet(source: dict[str, Any], discoveries: list[dict[str, Any]]) -> dict[str, Any]:
    provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
    retrieval = provenance.get("retrieval") if isinstance(provenance.get("retrieval"), dict) else {}
    text = str(retrieval.get("text_excerpt") or source.get("content_excerpt") or "")
    status = str(retrieval.get("status") or source.get("retrieval_status") or "")
    if status not in ACCEPTED_RETRIEVAL:
        status = "fetched"
    normalized_retrieval = {
        **retrieval,
        "status": "fetched" if status in ACCEPTED_RETRIEVAL else status,
        "text_chars": len(text),
        "text_excerpt": text,
        "relevance_pass": True,
        "final_url": str(retrieval.get("final_url") or source.get("url") or ""),
    }
    nested = {
        "schema_version": "ksova-v2-calibration-source-1.0",
        "source_id": source["source_id"],
        "title": source.get("title"),
        "url": source.get("url"),
        "jurisdiction": source.get("jurisdiction"),
        "authority_type": source.get("authority_type"),
        "issuing_authority": source.get("publisher") or source.get("authority_type"),
        "evidence_scope_ko": source.get("evidence_summary_ko") or "",
        "content_sha256": source.get("content_sha256"),
        "availability": "canonical_registry_reuse",
        "origins": ["canonical_registry_training_expansion"],
        "reliability_tier": canonical_tier(source),
        "allowed_uses": ["issue_discovery", "position_evidence", "fact_check", "generation_grounding"],
        "retrieval_context": [str(source.get("title") or "")],
        "retrieval": normalized_retrieval,
        "raw_snapshot_files": source.get("raw_snapshot_files") or [],
    }
    return {
        "schema_version": "ksova-v2-trusted-discovery-source-1.0",
        "source_lead_id": source["source_id"],
        "lead_ids": [],
        "discoveries": discoveries,
        "resolution_method": "canonical_registry_reuse",
        "source": nested,
    }


def query_text(issue: str, request: str, site: str | None) -> str:
    request = re.sub(r"\s+", " ", request).strip()
    request = re.sub(
        r"(?:원문\s*)?(?:확인|검증|보강)(?:이|가)?\s*필요(?:함|하다)?[.!]?\s*$",
        "",
        request,
    ).strip()
    issue = re.sub(r"\s+", " ", issue).strip()

    def search_tokens(value: str) -> list[str]:
        values = re.findall(
            r"[A-Za-z][A-Za-z0-9./-]{1,30}|[0-9]{2,}[A-Za-z가-힣0-9./-]*|[가-힣]{2,20}",
            value,
        )
        output = []
        seen = set()
        for token in values:
            folded = token.casefold().strip(".,:;()[]{}'")
            if not folded or folded in BINDING_STOPWORDS or folded in seen:
                continue
            if folded.endswith(("필요", "원문", "확인", "검증")) and len(folded) <= 6:
                continue
            seen.add(folded)
            output.append(token.strip(".,:;()[]{}'"))
        return output

    # Search engines retrieve exact records more reliably from a compact list of
    # institutions, document identifiers and topic nouns than from validator prose.
    tokens = search_tokens(request)[:14]
    for token in search_tokens(issue):
        if token.casefold() not in {value.casefold() for value in tokens}:
            tokens.append(token)
        if len(tokens) >= 18:
            break
    combined = " ".join(tokens)
    if site:
        combined = f"site:{site} {combined}"
    return combined[:180].strip()


def site_hint(value: str) -> tuple[str | None, str | None]:
    for pattern, site, jurisdiction in SITE_HINTS:
        if pattern.search(value):
            return site, jurisdiction
    return None, None


def material_requests(material: dict[str, Any]) -> list[str]:
    audit = material.get("pipeline_material_audit")
    glm = audit.get("glm") if isinstance(audit, dict) and isinstance(audit.get("glm"), dict) else {}
    requests = [
        str(value).strip()
        for value in glm.get("source_requests_ko") or []
        if str(value).strip()
    ]
    if not requests and str(material.get("source_gap_ko") or "").strip():
        requests = [str(material["source_gap_ko"]).strip()]
    return requests


def infer_lane(material: dict[str, Any]) -> tuple[str, dict[str, int]]:
    text = " " + " ".join((
        str(material.get("specific_issue_ko") or ""),
        str(material.get("decision_object_ko") or ""),
        str(material.get("korean_stake_ko") or ""),
    )).casefold() + " "
    scores = {
        lane: sum(
            2 if len(keyword.strip()) >= 5 else 1
            for keyword in keywords if keyword in text
        )
        for lane, keywords in LANE_KEYWORDS.items()
    }
    declared = str(material.get("lane") or "")
    inferred = max(scores, key=lambda lane: (scores[lane], lane))
    # Preserve the declared lane across close or genuinely cross-domain calls.
    if scores.get(inferred, 0) < scores.get(declared, 0) + 2:
        inferred = declared
    return inferred, scores


def command_prepare_gap_search(args: argparse.Namespace) -> int:
    registry_rows = read_jsonl(args.source_registry)
    registry, _ = registry_index(registry_rows)
    issues = read_jsonl(args.benchmark_issues)
    _, benchmark_keys = benchmark_source_keys(issues, registry)
    target_lanes = tuple(args.target_lane or DEFAULT_TARGET_LANES)
    needs = [row for row in read_jsonl(args.needs_source) if row.get("lane") in target_lanes]

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    by_lane_count: Counter[str] = Counter()
    for material in sorted(needs, key=lambda row: (str(row.get("lane")), str(row.get("material_id")))):
        lane = str(material.get("lane"))
        if args.limit_per_lane and by_lane_count[lane] >= args.limit_per_lane:
            continue
        ids = [str(value) for value in material.get("source_ids") or []]
        missing = [source_id for source_id in ids if source_id not in registry]
        keys = {key for source_id in ids if source_id in registry for key in source_keys(registry[source_id])}
        requests = material_requests(material)
        reasons = []
        if missing:
            reasons.append("missing_source_registry_rows")
        if keys & benchmark_keys:
            reasons.append("benchmark_source_collision")
        if not ids:
            reasons.append("missing_existing_source")
        if not requests:
            reasons.append("missing_validator_source_request")
        if reasons:
            excluded.append({"material_id": material.get("material_id"), "reasons": reasons})
            continue
        selected.append(material)
        by_lane_count[lane] += 1

    queries: list[dict[str, Any]] = []
    query_keys = set()
    packet_discoveries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reclassified = []
    for material in selected:
        candidate_id = str(material["material_id"])
        declared_lane = str(material["lane"])
        lane, lane_scores = infer_lane(material)
        if lane != declared_lane:
            reclassified.append({
                "material_id": candidate_id,
                "declared_lane": declared_lane,
                "discovery_lane": lane,
                "declared_score": lane_scores.get(declared_lane, 0),
                "inferred_score": lane_scores.get(lane, 0),
            })
        issue = str(material.get("specific_issue_ko") or material.get("decision_object_ko") or "")
        evidence_status = str(material.get("evidence_status") or "")
        role = {
            "needs_korean_source": "kr_grounding",
            "needs_counter_source": "external_or_counter",
            "context_only": "independent_fact",
        }.get(evidence_status, "supplemental_grounding")
        candidate_queries = []
        for request_index, request in enumerate(
            material_requests(material)[: args.max_queries_per_material], 1
        ):
            site, jurisdiction = site_hint(request)
            value = query_text(issue, request, site)
            key = (lane, "ko", value)
            if key in query_keys:
                value = query_text(issue + " " + str(material.get("decision_object_ko") or ""), request, site)
                key = (lane, "ko", value)
            if key in query_keys:
                value = (value[:220] + " " + candidate_id[-8:])[:245]
                key = (lane, "ko", value)
            query_keys.add(key)
            entry = {
                "lane": lane,
                "language": "ko",
                "query": value,
                "candidate_id": candidate_id,
                "source_role": role,
                "jurisdiction": jurisdiction or "ANY",
                "source_variant": "official_site_query" if site else "open_web_query",
                "engines": (
                    ["bing_web", "duckduckgo_web", "naver_web"]
                    if site else
                    ["bing_web", "duckduckgo_web", "naver_web", "google_news"]
                ),
            }
            queries.append(entry)
            candidate_queries.append(entry)
        anchor = candidate_queries[0]
        for source_id in material.get("source_ids") or []:
            packet_discoveries[str(source_id)].append({
                "lane": lane,
                "language": "ko",
                "query": anchor["query"],
                "engine": "canonical_registry",
                "rank": 0,
                "candidate_id": candidate_id,
                "source_role": "existing_context",
                "jurisdiction": str(registry[str(source_id)].get("jurisdiction") or "ANY"),
                "source_variant": "existing_material_source",
            })

    packets = [
        canonical_packet(registry[source_id], discoveries)
        for source_id, discoveries in sorted(packet_discoveries.items())
    ]
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    config = {
        "schema_version": "ksova-training-source-gap-query-1.0",
        "engines": ["google_news", "bing_web", "duckduckgo_web", "naver_web"],
        "discovery_only_hosts": base_config.get("discovery_only_hosts", []),
        "blocked_hosts": base_config.get("blocked_hosts", []),
        "queries": queries,
    }
    write_json(args.query_config, config)
    write_jsonl(args.base_packets, packets)
    write_jsonl(args.selected_materials, selected)
    report = {
        "schema_version": "ksova-training-source-gap-plan-1.0",
        "policy": "rejected_material_is_query_seed_only_no_repair",
        "target_lanes": list(target_lanes),
        "input_needs_source": len(needs),
        "selected_query_seeds": len(selected),
        "excluded": len(excluded),
        "excluded_details": excluded,
        "reclassified": len(reclassified),
        "reclassified_details": reclassified,
        "queries": len(queries),
        "base_source_packets": len(packets),
        "by_lane": dict(sorted(Counter(str(row["lane"]) for row in selected).items())),
        "query_config": str(args.query_config.resolve()),
        "base_packets": str(args.base_packets.resolve()),
        "selected_materials": str(args.selected_materials.resolve()),
    }
    write_json(args.report, report)
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in {"excluded_details", "reclassified_details"}
    }, ensure_ascii=False))
    return 0 if queries and packets else 2


def lane_score(source: dict[str, Any], lane: str) -> int:
    provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
    screen = provenance.get("screen") if isinstance(provenance.get("screen"), dict) else {}
    curation = provenance.get("curation") if isinstance(provenance.get("curation"), dict) else {}
    text = " ".join((
        str(source.get("title") or ""),
        str(source.get("evidence_summary_ko") or ""),
        " ".join(str(value) for value in screen.get("material_tags") or []),
        str(screen.get("pairing_need_ko") or ""),
        str(curation.get("issue_claim_ko") or ""),
        str(source.get("content_excerpt") or "")[:1600],
    )).casefold()
    padded = " " + text + " "
    score = sum(2 if len(keyword.strip()) >= 5 else 1 for keyword in LANE_KEYWORDS[lane] if keyword in padded)
    evaluation_domain = str(curation.get("evaluation_domain") or "")
    if DOMAIN_LANE_BIAS.get(evaluation_domain) == lane:
        score += 5
    return score


def command_prepare_canonical(args: argparse.Namespace) -> int:
    sources = read_jsonl(args.source_registry)
    registry, _ = registry_index(sources)
    issues = read_jsonl(args.benchmark_issues)
    _, benchmark_keys = benchmark_source_keys(issues, registry)
    material_rows = [row for path in args.existing_materials for row in read_jsonl(path)]
    used_ids = {
        str(source_id)
        for material in material_rows
        for source_id in material.get("source_ids") or []
        if source_id
    }
    used_keys = {
        key for source_id in used_ids if source_id in registry for key in source_keys(registry[source_id])
    }
    target_lanes = tuple(args.target_lane or DEFAULT_TARGET_LANES)
    pools: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    for source in sources:
        provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
        curation = provenance.get("curation") if isinstance(provenance.get("curation"), dict) else {}
        evidence_quality = str(curation.get("evidence_quality") or "")
        disposition = str(curation.get("disposition") or "")
        text = str(source.get("content_excerpt") or "")
        keys = source_keys(source)
        reasons = []
        if keys & benchmark_keys:
            reasons.append("benchmark_collision")
        if keys & used_keys or str(source.get("source_id")) in used_ids:
            reasons.append("already_used_by_material")
        if evidence_quality not in {"direct", "context"}:
            reasons.append("weak_evidence_quality")
        if disposition not in {"retain", "context_only"}:
            reasons.append("not_curated_for_reuse")
        if len(text) < args.minimum_text_chars:
            reasons.append("short_body")
        if str(source.get("retrieval_status") or "") not in ACCEPTED_RETRIEVAL:
            reasons.append("unusable_retrieval")
        if reasons:
            rejection_counts.update(reasons)
            continue
        scored = [(lane_score(source, lane), lane) for lane in target_lanes]
        score, lane = max(scored, key=lambda item: (item[0], item[1]))
        if score < args.minimum_lane_score:
            rejection_counts["no_target_lane_signal"] += 1
            continue
        pools[lane].append((score, source))

    selected: list[tuple[str, int, dict[str, Any]]] = []
    for lane in target_lanes:
        ranked = sorted(
            pools.get(lane, []),
            key=lambda item: (
                -int(str((item[1].get("provenance", {}).get("curation") or {}).get("evidence_quality")) == "direct"),
                -int(str((item[1].get("provenance", {}).get("curation") or {}).get("disposition")) == "retain"),
                -item[0],
                -len(str(item[1].get("content_excerpt") or "")),
                str(item[1]["source_id"]),
            ),
        )
        for score, source in ranked[: args.limit_per_lane]:
            selected.append((lane, score, source))

    packets = []
    query_entries = []
    for lane, score, source in sorted(selected, key=lambda item: (item[0], str(item[2]["source_id"]))):
        candidate_id = "KCANON-" + str(source["source_id"]).replace("SRCV2-", "")
        query = re.sub(r"\s+", " ", str(source.get("title") or source["source_id"])).strip()[:220]
        discovery = {
            "lane": lane,
            "language": "ko",
            "query": query,
            "engine": "canonical_registry",
            "rank": 0,
            "candidate_id": candidate_id,
            "source_role": "canonical_unused_source",
            "jurisdiction": str(source.get("jurisdiction") or "ANY"),
            "source_variant": "curated_unused_canonical",
            "lane_score": score,
        }
        packets.append(canonical_packet(source, [discovery]))
        query_entries.append({key: discovery[key] for key in (
            "lane", "language", "query", "candidate_id", "source_role", "jurisdiction", "source_variant"
        )})

    config = {
        "schema_version": "ksova-training-canonical-packet-config-1.0",
        "engines": ["bing_web"],
        "discovery_only_hosts": [],
        "blocked_hosts": ["kp"],
        "queries": query_entries,
    }
    write_jsonl(args.output, packets)
    write_json(args.query_config, config)
    report = {
        "schema_version": "ksova-training-canonical-packet-plan-1.0",
        "policy": "curated_unused_canonical_sources_only",
        "source_registry": len(sources),
        "existing_materials": len(material_rows),
        "selected_sources": len(packets),
        "by_lane": dict(sorted(Counter(lane for lane, _, _ in selected).items())),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "minimum_text_chars": args.minimum_text_chars,
        "minimum_lane_score": args.minimum_lane_score,
        "limit_per_lane": args.limit_per_lane,
        "output": str(args.output.resolve()),
        "query_config": str(args.query_config.resolve()),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if packets else 2


def merge_discoveries(target: dict[str, Any], additions: Iterable[dict[str, Any]]) -> None:
    rows = list(target.get("discoveries") or []) + list(additions)
    seen = set()
    unique = []
    for row in rows:
        identity = tuple(
            str(row.get(key) or "")
            for key in (
                "lane", "language", "query", "engine", "rank", "candidate_id",
                "source_role", "jurisdiction", "source_variant",
            )
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(row)
    target["discoveries"] = unique


def binding_terms(value: str) -> set[str]:
    value = re.sub(r"site:[^\s]+", " ", value, flags=re.IGNORECASE)
    return {
        token.casefold()
        for token in re.findall(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9-]{2,}", value)
        if token.casefold() not in BINDING_STOPWORDS
    }


def binding_relevant(query: str, title: str, text: str) -> tuple[bool, dict[str, Any]]:
    query_terms = binding_terms(query)
    title_terms = binding_terms(title)
    title_matches = query_terms & title_terms
    folded = text.casefold()
    occurrences: list[tuple[int, str]] = []
    for term in query_terms:
        start = 0
        for _ in range(5):
            index = folded.find(term, start)
            if index < 0:
                break
            occurrences.append((index, term))
            start = index + max(1, len(term))
    occurrences.sort()
    best_cluster: set[str] = set()
    left = 0
    counts: Counter[str] = Counter()
    for right, (position, term) in enumerate(occurrences):
        counts[term] += 1
        while left <= right and position - occurrences[left][0] > 1200:
            counts[occurrences[left][1]] -= 1
            left += 1
        cluster = {value for value, count in counts.items() if count > 0}
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    distinctive = {
        term for term in best_cluster
        if len(term) >= (6 if term.isascii() else 4)
    }
    passed = (
        len(title_matches) >= 2
        or (len(best_cluster) >= 4 and len(distinctive) >= 2)
    )
    return passed, {
        "query_terms": sorted(query_terms),
        "title_matches": sorted(title_matches),
        "best_cluster": sorted(best_cluster),
    }


def obvious_listing_page(source: dict[str, Any]) -> bool:
    retrieval = source.get("retrieval") if isinstance(source.get("retrieval"), dict) else {}
    url = str(retrieval.get("final_url") or source.get("url") or "")
    path = urllib.parse.urlsplit(url).path.rstrip("/") + "/"
    return bool(LISTING_PATH.search(path))


def command_filter_results(args: argparse.Namespace) -> int:
    registry_rows = read_jsonl(args.source_registry)
    registry, registry_by_url = registry_index(registry_rows)
    _, benchmark_keys = benchmark_source_keys(read_jsonl(args.benchmark_issues), registry)
    output_by_id: dict[str, dict[str, Any]] = {}
    outcomes: Counter[str] = Counter()
    rejected_bindings = []
    for wrapper in read_jsonl(args.resolved):
        source = source_row(wrapper)
        retrieval = source.get("retrieval") if isinstance(source.get("retrieval"), dict) else {}
        text = str(retrieval.get("text_excerpt") or source.get("content_excerpt") or "")
        if (
            str(source.get("reliability_tier") or "") not in {"A", "B"}
            or str(retrieval.get("status") or "") not in {"fetched", "reused_verified_snapshot"}
            or len(text) < args.minimum_text_chars
        ):
            outcomes["unusable"] += 1
            continue
        if obvious_listing_page(source):
            outcomes["listing_page"] += 1
            continue
        accepted_discoveries = []
        for discovery in wrapper.get("discoveries") or []:
            passed, evidence = binding_relevant(
                str(discovery.get("query") or ""), str(source.get("title") or ""), text
            )
            if passed:
                accepted_discoveries.append(discovery)
            else:
                rejected_bindings.append({
                    "source_id": source.get("source_id"),
                    "candidate_id": discovery.get("candidate_id"),
                    "query": discovery.get("query"),
                    "evidence": evidence,
                })
        if not accepted_discoveries:
            outcomes["no_relevant_candidate_binding"] += 1
            continue
        wrapper = {**wrapper, "discoveries": accepted_discoveries}
        canonical = registry.get(str(source.get("source_id") or ""))
        canonical = canonical or registry_by_url.get(normalized_url(str(source.get("url") or "")))
        candidate = (
            canonical_packet(canonical, list(wrapper.get("discoveries") or []))
            if canonical is not None
            else wrapper
        )
        candidate_source = source_row(candidate)
        if source_keys(candidate_source) & benchmark_keys:
            outcomes["benchmark_collision"] += 1
            continue
        source_id = str(candidate_source.get("source_id") or "")
        if not source_id:
            outcomes["missing_source_id"] += 1
            continue
        if source_id in output_by_id:
            previous = output_by_id[source_id]
            if normalized_url(str(source_row(previous).get("url") or "")) != normalized_url(
                str(candidate_source.get("url") or "")
            ):
                raise ValueError(f"conflicting source URLs for {source_id}")
            merge_discoveries(previous, candidate.get("discoveries") or [])
            outcomes["duplicate_merged"] += 1
            continue
        output_by_id[source_id] = candidate
        outcomes["registry_reused" if canonical is not None else "new_fetched"] += 1

    output = sorted(output_by_id.values(), key=lambda row: str(source_row(row).get("source_id")))
    write_jsonl(args.output, output)
    report = {
        "schema_version": "ksova-training-filtered-source-report-1.0",
        "input": len(read_jsonl(args.resolved)),
        "accepted": len(output),
        "outcomes": dict(sorted(outcomes.items())),
        "accepted_bindings": sum(len(row.get("discoveries") or []) for row in output),
        "rejected_bindings": len(rejected_bindings),
        "rejected_binding_details": rejected_bindings,
        "by_tier": dict(sorted(Counter(
            str(source_row(row).get("reliability_tier")) for row in output
        ).items())),
        "by_lane": dict(sorted(Counter(
            str(discovery.get("lane") or "unknown")
            for row in output for discovery in row.get("discoveries") or []
        ).items())),
        "output": str(args.output.resolve()),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_merge_packets(args: argparse.Namespace) -> int:
    output_by_id: dict[str, dict[str, Any]] = {}
    input_rows = 0
    duplicates = 0
    for path in args.input:
        for wrapper in read_jsonl(path):
            input_rows += 1
            source = source_row(wrapper)
            source_id = str(source.get("source_id") or "")
            if not source_id:
                raise ValueError(f"missing source_id in {path}")
            if source_id not in output_by_id:
                output_by_id[source_id] = wrapper
                continue
            prior = output_by_id[source_id]
            if normalized_url(str(source_row(prior).get("url") or "")) != normalized_url(
                str(source.get("url") or "")
            ):
                raise ValueError(f"conflicting source URLs for {source_id}")
            merge_discoveries(prior, wrapper.get("discoveries") or [])
            duplicates += 1
    output = sorted(output_by_id.values(), key=lambda row: str(source_row(row).get("source_id")))
    write_jsonl(args.output, output)
    report = {
        "schema_version": "ksova-training-source-packet-merge-1.0",
        "inputs": [str(path.resolve()) for path in args.input],
        "input_rows": input_rows,
        "output_rows": len(output),
        "duplicates_merged": duplicates,
        "output": str(args.output.resolve()),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    gap = commands.add_parser("prepare-gap-search")
    gap.add_argument("--needs-source", type=Path, required=True)
    gap.add_argument("--source-registry", type=Path, required=True)
    gap.add_argument("--benchmark-issues", type=Path, required=True)
    gap.add_argument("--base-config", type=Path, required=True)
    gap.add_argument("--target-lane", action="append", choices=sorted(LANE_KEYWORDS))
    gap.add_argument("--limit-per-lane", type=int, default=0)
    gap.add_argument("--max-queries-per-material", type=int, default=2)
    gap.add_argument("--query-config", type=Path, required=True)
    gap.add_argument("--base-packets", type=Path, required=True)
    gap.add_argument("--selected-materials", type=Path, required=True)
    gap.add_argument("--report", type=Path, required=True)
    gap.set_defaults(func=command_prepare_gap_search)

    canonical = commands.add_parser("prepare-canonical")
    canonical.add_argument("--source-registry", type=Path, required=True)
    canonical.add_argument("--benchmark-issues", type=Path, required=True)
    canonical.add_argument("--existing-materials", type=Path, action="append", required=True)
    canonical.add_argument("--target-lane", action="append", choices=sorted(LANE_KEYWORDS))
    canonical.add_argument("--minimum-text-chars", type=int, default=500)
    canonical.add_argument("--minimum-lane-score", type=int, default=2)
    canonical.add_argument("--limit-per-lane", type=int, default=20)
    canonical.add_argument("--output", type=Path, required=True)
    canonical.add_argument("--query-config", type=Path, required=True)
    canonical.add_argument("--report", type=Path, required=True)
    canonical.set_defaults(func=command_prepare_canonical)

    filtered = commands.add_parser("filter-results")
    filtered.add_argument("--resolved", type=Path, required=True)
    filtered.add_argument("--source-registry", type=Path, required=True)
    filtered.add_argument("--benchmark-issues", type=Path, required=True)
    filtered.add_argument("--minimum-text-chars", type=int, default=500)
    filtered.add_argument("--output", type=Path, required=True)
    filtered.add_argument("--report", type=Path, required=True)
    filtered.set_defaults(func=command_filter_results)

    merge = commands.add_parser("merge-packets")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--report", type=Path, required=True)
    merge.set_defaults(func=command_merge_packets)
    return root


def main() -> int:
    args = parser().parse_args()
    if hasattr(args, "limit_per_lane") and args.limit_per_lane < 0:
        raise ValueError("limit_per_lane must be non-negative")
    if hasattr(args, "max_queries_per_material") and args.max_queries_per_material < 1:
        raise ValueError("max_queries_per_material must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
