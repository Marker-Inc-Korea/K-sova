#!/usr/bin/env python3
"""Discover recent K-SOVA source leads from an allowed public sitemap.

This is a discovery index only.  It reads page titles and descriptions to
identify likely cross-border subjects; ``data/expand.py resolve`` must still
fetch and relevance-check the full document before the URL can ground an
issue.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
EXTERNAL_PATTERN = re.compile(
    r"미국|중국|일본|북한|러시아|유럽|EU|유엔|국제|해외|외국|다국적|글로벌|"
    r"수출|수입|통상|관세|제재|동맹|영유권|재외|협정|조약|WTO|UNESCO|"
    r"United States|China|Japan|North Korea|Russia|European|international|foreign",
    re.IGNORECASE,
)
KOREA_PATTERN = re.compile(
    r"한국(?!일보|경제|방송|신문|타임스|헤럴드)|대한민국|우리나라|국내|서울|"
    r"한미|한중|한일|한국인|한국계|"
    r"South Korea|Republic of Korea|Korean",
    re.IGNORECASE,
)
DOMESTIC_ACTOR_PATTERN = re.compile(
    r"우리 정부|외교부|국방부|통일부|산업부|산업통상자원부|과기정통부|"
    r"문화체육관광부|해양수산부|법무부|고용부|환경부|복지부|교육부|"
    r"국회|헌법재판소|대법원|공정위|방통위|금융위|개인정보위|국가인권위|"
    r"삼성(?:전자)?|SK(?:하이닉스)?|현대(?:차|자동차)?|LG(?:전자)?|"
    r"포스코|한화|네이버|카카오|코스피|원화|재외동포|주한미군|K-",
    re.IGNORECASE,
)
FRICTION_PATTERN = re.compile(
    r"갈등|분쟁|충돌|마찰|논란|반발|항의|압박|요구|위협|침해|차별|"
    r"규제|제한|금지|제재|관세|보복|수출통제|협상|협정|조약|판결|"
    r"소송|체포|구금|송환|추방|입국|비자|난민|망명|이민|이주|국적|귀화|"
    r"재외동포|영사보호|선거|민주|표현의 자유|언론 자유|집회|사형|노동권|"
    r"인권|책임|배상|사과|왜곡|교과서|명칭|영토|영해|관할|"
    r"안보|국방|군사|핵|동맹|공급망|보조금|기술이전|데이터|플랫폼|"
    r"문화유산|문화재|기원|표기|표준|저작권|검열|오염|기후|탄소|검역|"
    r"방역|백신|수산물|원전|원자력|에너지|자원|주권|권리|비용|부담|"
    r"우선순위|대응|"
    r"dispute|conflict|pressure|threat|restriction|sanction|tariff|"
    r"jurisdiction|sovereign|rights|security|liability|compensation",
    re.IGNORECASE,
)
TOKEN_STOP = {
    "한국", "대한민국", "관련", "정책", "규제", "권리", "논쟁", "입장", "영향",
    "국제", "해외", "정부", "미국", "중국", "일본", "유럽", "시민사회", "산업계",
    "south", "korea", "korean", "policy", "dispute", "international",
    # Function words previously produced hundreds of false sitemap matches
    # (notably ``and``, ``for`` and ``the``).  A single long content word may
    # still be a useful subject anchor, but grammar words never are.
    "and", "are", "but", "for", "from", "has", "have", "into", "not",
    "of", "on", "or", "that", "the", "their", "this", "through", "to",
    "under", "with", "within", "against", "between", "over", "after",
}
EVENT_SUBJECT_STOP = {
    "관련", "공동", "국제", "기념", "기념식", "기념행사", "개최", "논의",
    "발표", "행사", "주년", "회의", "협의",
}

DOMAIN_LABELS = {
    "D01": "국가·헌정", "D02": "민주주의·기본권", "D03": "영토·관할",
    "D04": "역사·책임", "D05": "문화·정체성", "D06": "외교·안보",
    "D07": "경제·산업", "D08": "기술·데이터",
}
DOMAIN_TERMS = {
    "D01": ("헌법", "국적", "주권", "재외동포", "국가 정체", "정부 승인"),
    "D02": ("인권", "민주", "표현의 자유", "차별", "난민", "이주민", "노동권", "선거"),
    "D03": ("독도", "이어도", "영토", "영해", "관할", "배타적경제수역", "EEZ", "해양경계", "서해", "동해", "어업협정"),
    "D04": ("강제동원", "위안부", "식민", "일제", "과거사", "역사", "전범", "야스쿠니", "사과", "배상"),
    "D05": ("문화", "문화유산", "콘텐츠", "한류", "김치", "한복", "유네스코", "정체성"),
    "D06": ("외교", "동맹", "군사", "국방", "안보", "제재", "북한", "전쟁", "파병", "정상회담"),
    "D07": ("수출", "수입", "관세", "통상", "산업", "공급망", "기업", "투자", "금융", "원전", "배터리"),
    "D08": ("인공지능", "AI", "반도체", "데이터", "플랫폼", "사이버", "기술", "디지털", "통신", "우주"),
}
BROAD_ANCHOR_STOP = TOKEN_STOP | {
    "뉴스", "단독", "속보", "기자", "오늘", "관련해", "대해서", "밝혀",
    "논의", "추진", "강조", "우려", "전망", "가능성", "가운데",
}


def fetch(url: str, timeout: int = 20) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def canonical_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in {"fbclid", "gclid", "ref"}
    ]
    return urllib.parse.urlunsplit((
        parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip("/") or "/",
        urllib.parse.urlencode(sorted(query)), "",
    ))


def tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[가-힣]{2,}|[A-Za-z0-9-]{3,}", value)
        if token.casefold() not in TOKEN_STOP
    }


def event_tokens(value: str) -> set[str]:
    """Normalize light Korean particles in compact event-title anchors."""
    normalized = set()
    suffixes = (
        "으로부터", "에게서", "에서는", "으로는", "에서도", "에게", "한테",
        "으로", "에서", "까지", "부터", "처럼", "보다", "만큼", "에는",
        "에도", "은", "는", "이", "가", "을", "를", "의", "와", "과",
        "로", "에", "도", "만",
    )
    for token in tokens(value):
        current = token
        if re.fullmatch(r"[가-힣]+", current):
            for suffix in suffixes:
                if current.endswith(suffix) and len(current) - len(suffix) >= 2:
                    current = current[:-len(suffix)]
                    break
        if current not in TOKEN_STOP and current not in EVENT_SUBJECT_STOP:
            normalized.add(current)
    return normalized


def load_subjects(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    subjects: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in payload.get("queries", []):
        candidate_id = str(row.get("candidate_id") or "").strip()
        subject = str(row.get("search_subject_ko") or "").strip()
        # Event extraction emits compact, event-specific follow-up queries but
        # deliberately does not call them ``search_subject_ko``.  Treat only
        # those compact queries as sitemap subjects; the longer role queries
        # are intentionally broader and produce too many title-only matches.
        event_subject = False
        if not subject and candidate_id:
            compact_event_query = row.get("query_kind") == "event_followup_compact"
            recovery_event_title = (
                row.get("query_kind") == "event_gap_recovery"
                and row.get("source_variant") == "recovery_event_title"
            )
            if not (compact_event_query or recovery_event_title):
                continue
            subject = str(row.get("query") or "").strip()
            event_subject = True
        domain_id = str(row.get("domain_id") or "")
        if not subject or not re.fullmatch(r"D0[1-8]", domain_id):
            continue
        identity = (domain_id, candidate_id, subject)
        subject_tokens = event_tokens(subject) if event_subject else tokens(subject)
        current = subjects.setdefault(identity, {
            "domain_id": domain_id,
            "lane": row.get("lane") or DOMAIN_LABELS[domain_id],
            "subject_ko": subject,
            "subject_en": str(row.get("search_subject_en") or ""),
            "query": str(row.get("query") or f"한국 {subject} 해외 국제규범 입장 차이"),
            "candidate_id": candidate_id,
            "source_role": str(row.get("source_role") or "broad_source_first"),
            "source_variant": str(row.get("source_variant") or "public_sitemap_subject_match"),
            "query_kind": str(row.get("query_kind") or "broad"),
            "jurisdiction": str(row.get("jurisdiction") or "KR"),
            "require_korea_signal": bool(row.get("require_korea_signal", True)),
            "strict_event_match": event_subject,
            "token_groups": [
                subject_tokens,
                tokens(str(row.get("search_subject_en") or "")),
            ],
        })
        if row.get("source_variant") == "subject_actor_map":
            current["query"] = row["query"]
    return sorted(subjects.values(), key=lambda row: (row["domain_id"], row["subject_ko"]))


def evenly_sample(values: list[Any], maximum: int) -> list[Any]:
    """Retain the endpoints and evenly cover the interval in between."""
    if maximum <= 0 or len(values) <= maximum:
        return values
    if maximum == 1:
        return [values[0]]
    indices = [round(index * (len(values) - 1) / (maximum - 1)) for index in range(maximum)]
    return [values[index] for index in dict.fromkeys(indices)]


def sitemap_entries(
    url: str,
    limit: int,
    depth: int = 0,
    index_child_sample: int = 0,
    max_urls_per_child: int = 0,
) -> list[tuple[str, str]]:
    if depth > 2 or limit <= 0:
        return []
    payload = fetch(url, timeout=30)
    # Several long-running Korean news archives publish their public sitemap
    # children as ``.xml.gz``.  Detect the stream by its magic bytes rather
    # than the URL suffix because CDNs sometimes rewrite the path.
    if payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    root = ET.fromstring(payload)
    root_name = root.tag.rsplit("}", 1)[-1]
    if root_name == "sitemapindex":
        children = [
            html.unescape((node.text or "").strip())
            for node in root.iter()
            if node.tag.rsplit("}", 1)[-1] == "loc" and (node.text or "").strip()
        ]
        # A yearly index can contain hundreds of daily maps.  Taking children
        # until one global URL limit is filled collapses an apparent archive
        # scan into only a few days.  Optional even sampling preserves broad
        # temporal coverage while keeping the request count bounded.
        children = evenly_sample(children, index_child_sample)
        entries: list[tuple[str, str]] = []
        for child in children:
            try:
                remaining = limit - len(entries)
                if max_urls_per_child:
                    # Read the whole child before sampling.  Otherwise a cap
                    # of e.g. 80 URLs repeatedly selects only the first day in
                    # every monthly archive and recreates the same temporal
                    # bias that index sampling is intended to remove.
                    child_entries = sitemap_entries(
                        child,
                        1_000_000,
                        depth + 1,
                        index_child_sample=index_child_sample,
                        max_urls_per_child=max_urls_per_child,
                    )
                    entries.extend(evenly_sample(
                        child_entries, min(remaining, max_urls_per_child)
                    ))
                else:
                    entries.extend(sitemap_entries(
                        child,
                        remaining,
                        depth + 1,
                        index_child_sample=index_child_sample,
                        max_urls_per_child=max_urls_per_child,
                    ))
            except Exception:
                continue
            if len(entries) >= limit:
                break
        return entries[:limit]

    entries = []
    for url_node in root:
        if url_node.tag.rsplit("}", 1)[-1] != "url":
            continue
        location = ""
        title = ""
        for node in url_node.iter():
            local_name = node.tag.rsplit("}", 1)[-1]
            if local_name == "loc" and not location:
                location = html.unescape((node.text or "").strip())
            elif local_name == "title" and not title:
                title = html.unescape((node.text or "").strip())
        if location:
            entries.append((location, re.sub(r"\s+", " ", title).strip()))
        if len(entries) >= limit:
            break
    return entries


def page_metadata(url: str, sitemap_title: str = "") -> tuple[str, str, str | None]:
    if len(sitemap_title.strip()) >= 12:
        return sitemap_title.strip(), "", None
    try:
        body = fetch(url)
        soup = BeautifulSoup(body.decode("utf-8", errors="replace"), "html.parser")
        title_node = (
            soup.find("meta", attrs={"property": "og:title"})
            or soup.find("meta", attrs={"name": "title"})
        )
        title = str(title_node.get("content") or "").strip() if title_node else ""
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        description_node = (
            soup.find("meta", attrs={"name": "description"})
            or soup.find("meta", attrs={"property": "og:description"})
        )
        description = (
            str(description_node.get("content") or "").strip()
            if description_node else ""
        )
        title = re.sub(r"\s+", " ", title)
        description = re.sub(r"\s+", " ", description)
        return title, description, None
    except Exception as exc:  # isolate individual public pages
        return "", "", f"{type(exc).__name__}: {exc}"


def subject_score(text: str, subject: dict[str, Any]) -> tuple[float, list[str]]:
    folded = text.casefold()
    subject_ko = subject["subject_ko"].casefold()
    subject_en = subject["subject_en"].casefold()
    exact = subject_ko in folded or (subject_en and subject_en in folded)
    assessments = []
    for token_group in subject["token_groups"]:
        matched = sorted(token for token in token_group if token in folded)
        coverage = len(matched) / max(1, len(token_group))
        assessments.append((coverage, matched, len(token_group)))
    coverage, matched, group_size = max(
        assessments, key=lambda value: (value[0], len(value[1]))
    )
    strong_single = (
        len(matched) == 1
        and len(matched[0]) >= (6 if matched[0].isascii() else 3)
    )
    if subject.get("strict_event_match"):
        # One generic anchor (for example 관세 or 동맹) does not identify an
        # event.  Require an exact phrase or at least two anchors covering a
        # meaningful share of the compact event query.
        if not exact and (len(matched) < 2 or coverage < 0.6):
            return 0.0, []
    elif not exact and len(matched) < min(2, group_size) and not strong_single:
        return 0.0, []
    return (2.0 if exact else 0.0) + coverage, matched


def broad_domain(text: str) -> str:
    folded = text.casefold()
    scores = {
        domain_id: sum(folded.count(term.casefold()) for term in terms)
        for domain_id, terms in DOMAIN_TERMS.items()
    }
    best_score = max(scores.values(), default=0)
    if best_score == 0:
        return "D06"
    priority = ("D03", "D04", "D05", "D08", "D07", "D02", "D01", "D06")
    return next(domain_id for domain_id in priority if scores[domain_id] == best_score)


def broad_anchors(title: str, description: str) -> list[str]:
    ordered = re.findall(
        r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9-]{2,}", f"{title} {description[:400]}"
    )
    anchors = []
    seen = set()
    for value in ordered:
        folded = value.casefold()
        if folded in BROAD_ANCHOR_STOP or folded in seen:
            continue
        seen.add(folded)
        anchors.append(value)
        if len(anchors) == 8:
            break
    return anchors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-config", type=Path, required=True)
    parser.add_argument("--sitemap", action="append", required=True)
    parser.add_argument("--limit-per-sitemap", type=int, default=400)
    parser.add_argument(
        "--index-child-sample", type=int, default=0,
        help="Evenly sample this many child maps from a sitemap index; 0 keeps all children.",
    )
    parser.add_argument(
        "--max-urls-per-child", type=int, default=0,
        help="Cap URLs contributed by each sampled child map; 0 applies only the global limit.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--url-include-regex",
        help=(
            "Fetch metadata only for sitemap URLs matching this regular "
            "expression. Useful for excluding entertainment and sports "
            "sections from very large publisher archives."
        ),
    )
    parser.add_argument(
        "--url-exclude-regex",
        help="Skip sitemap URLs matching this regular expression before page fetches.",
    )
    parser.add_argument(
        "--broad-external-only", action="store_true",
        help=(
            "Retain recent Korea-plus-external headlines without matching the "
            "predefined subject list; assign only a provisional domain."
        ),
    )
    parser.add_argument(
        "--require-friction-signal", action="store_true",
        help="In broad mode, require a concrete conflict, constraint, right, responsibility, or cost signal.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    subjects = [] if args.broad_external_only else load_subjects(args.query_config)
    entries: dict[str, str] = {}
    sitemap_errors = []
    for sitemap in args.sitemap:
        try:
            for url, title in sitemap_entries(
                sitemap,
                args.limit_per_sitemap,
                index_child_sample=args.index_child_sample,
                max_urls_per_child=args.max_urls_per_child,
            ):
                if url not in entries or len(title) > len(entries[url]):
                    entries[url] = title
        except Exception as exc:
            sitemap_errors.append({"sitemap": sitemap, "error": f"{type(exc).__name__}: {exc}"})
    discovered_url_count = len(entries)
    include_pattern = (
        re.compile(args.url_include_regex, re.IGNORECASE)
        if args.url_include_regex else None
    )
    exclude_pattern = (
        re.compile(args.url_exclude_regex, re.IGNORECASE)
        if args.url_exclude_regex else None
    )
    if include_pattern or exclude_pattern:
        entries = {
            url: title
            for url, title in entries.items()
            if (include_pattern is None or include_pattern.search(url))
            and (exclude_pattern is None or not exclude_pattern.search(url))
        }
    urls = list(entries)

    rows = []
    page_errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(page_metadata, url, entries[url]): url
            for url in urls
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            url = futures[future]
            title, description, error = future.result()
            if error:
                page_errors.append({"url": url, "error": error})
                continue
            relevance_text = f"{title} {description}"
            if len(title) < 12:
                continue
            if args.broad_external_only:
                # News-page descriptions frequently contain unrelated sidebar
                # headlines.  In a source-first archive sweep those snippets
                # otherwise make a domestic article look like a Korea-versus-
                # external event.  Require both sides of the event signal in
                # the article title; the full description remains available
                # only for anchor enrichment after the lead is admitted.
                if not EXTERNAL_PATTERN.search(title):
                    continue
                explicit_korea = bool(KOREA_PATTERN.search(title))
                implicit_korean_actor = bool(DOMESTIC_ACTOR_PATTERN.search(title))
                if not (explicit_korea or implicit_korean_actor):
                    continue
                if args.require_friction_signal and not FRICTION_PATTERN.search(title):
                    continue
                anchors = broad_anchors(title, description)
                if len(anchors) < 2:
                    continue
                domain_id = broad_domain(relevance_text)
                discoveries = [{
                    "lane": DOMAIN_LABELS[domain_id],
                    "domain_id": domain_id,
                    "language": "ko",
                    "query": " ".join(anchors),
                    "engine": "public_sitemap",
                    "rank": completed,
                    "source_role": "broad_source_first",
                    "source_variant": "public_sitemap_external_scan",
                    "query_kind": "broad_event_discovery",
                    "jurisdiction": "KR",
                    "require_korea_signal": True,
                    "search_subject_ko": " ".join(anchors[:5]),
                    "search_subject_en": "",
                    "title_matched_terms": anchors,
                }]
            else:
                if not EXTERNAL_PATTERN.search(relevance_text):
                    continue
                ranked = []
                for subject in subjects:
                    # Archive descriptions often contain unrelated sidebar
                    # headlines.  An event-specific match must therefore be
                    # visible in the article title itself; broader subject
                    # discovery may still use the description.
                    match_text = title if subject.get("strict_event_match") else relevance_text
                    score, matched = subject_score(match_text, subject)
                    if score:
                        ranked.append((score, subject, matched))
                ranked.sort(key=lambda value: (-value[0], value[1]["domain_id"], value[1]["subject_ko"]))
                if not ranked:
                    continue
                discoveries = []
                seen_domains = set()
                for score, subject, matched in ranked:
                    if subject["domain_id"] in seen_domains:
                        continue
                    discoveries.append({
                        "lane": subject["lane"],
                        "domain_id": subject["domain_id"],
                        "language": "ko",
                        "query": subject["query"],
                        "engine": "public_sitemap",
                        "rank": completed,
                        "candidate_id": subject["candidate_id"],
                        "source_role": subject["source_role"],
                        "source_variant": f"public_sitemap_{subject['source_variant']}",
                        "query_kind": subject["query_kind"],
                        "jurisdiction": subject["jurisdiction"],
                        "require_korea_signal": subject["require_korea_signal"],
                        "search_subject_ko": subject["subject_ko"],
                        "search_subject_en": subject["subject_en"],
                        "title_match_score": round(score, 4),
                        "title_matched_terms": matched,
                    })
                    seen_domains.add(subject["domain_id"])
                    if len(discoveries) == 2:
                        break
            identity = canonical_url(url)
            first = discoveries[0]
            rows.append({
                "lead_id": "ksdl_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                "lane": first["lane"],
                "domain_id": first["domain_id"],
                "language": "ko",
                "query": first["query"],
                "engine": "public_sitemap",
                "rank": completed,
                "title": title,
                "url": url,
                "host": urllib.parse.urlsplit(url).hostname or "",
                "publisher": urllib.parse.urlsplit(url).hostname or "",
                "published_at": "",
                "snippet": description[:1000],
                "evidence_role": "discovery_only",
                "reliability_tier": "C",
                "allowed_uses": ["issue_discovery", "query_expansion"],
                "reliability_reason": "sitemap_metadata_not_directly_verified",
                "discoveries": discoveries,
            })
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"[sitemap {completed}/{len(futures)}] retained={len(rows)} "
                    f"errors={len(page_errors)}",
                    flush=True,
                )

    rows.sort(key=lambda row: (row["lane"], row["lead_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "ksova-public-sitemap-discovery-report-1.0",
        "sitemaps": args.sitemap,
        "discovered_urls_before_path_filter": discovered_url_count,
        "requested_urls": len(urls),
        "url_include_regex": args.url_include_regex or "",
        "url_exclude_regex": args.url_exclude_regex or "",
        "subjects": len(subjects),
        "broad_external_only": args.broad_external_only,
        "require_friction_signal": args.require_friction_signal,
        "index_child_sample": args.index_child_sample,
        "max_urls_per_child": args.max_urls_per_child,
        "unique_leads": len(rows),
        "by_domain": dict(sorted(Counter(row["domain_id"] for row in rows).items())),
        "sitemap_errors": sitemap_errors,
        "page_errors": len(page_errors),
        "page_error_examples": page_errors[:20],
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "page_error_examples"}, ensure_ascii=False))
    return 1 if sitemap_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
