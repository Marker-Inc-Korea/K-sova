#!/usr/bin/env python3
"""Resolve trusted discovery headlines to fetched A/B source documents."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

from discover_calibration_sources import (
    base_host,
    bing_rss_results,
    duckduckgo_lite_results,
    fetch_source,
    host_matches,
    likely_homepage,
    normalized,
    search_results,
    text_score,
    trusted_jurisdiction,
)


PUBLISHER_DOMAINS = (
    ("대한민국 정책브리핑", "korea.kr"),
    ("정책브리핑", "korea.kr"),
    ("국회입법조사처", "nars.go.kr"),
    ("국가인권위원회", "humanrights.go.kr"),
    ("한국무역협회", "kita.net"),
    ("kita.net", "kita.net"),
    ("대외경제정책연구원", "kiep.go.kr"),
    ("kiep", "kiep.go.kr"),
    ("정보통신정책연구원", "kisdi.re.kr"),
    ("kisdi", "kisdi.re.kr"),
    ("한국인터넷진흥원", "kisa.or.kr"),
    ("국가안보전략연구원", "inss.re.kr"),
    ("통일연구원", "kinu.or.kr"),
    ("kinu", "kinu.or.kr"),
    ("한국해양수산개발원", "kmi.re.kr"),
    ("kmi.re.kr", "kmi.re.kr"),
    ("한국국방연구원", "kida.re.kr"),
    ("kida", "kida.re.kr"),
    ("한국군사문제연구원", "kima.re.kr"),
    ("kima", "kima.re.kr"),
    ("한국사데이터베이스", "db.history.go.kr"),
    ("우리역사넷", "contents.history.go.kr"),
    ("문화체육관광부", "mcst.go.kr"),
    ("국립국어원", "korean.go.kr"),
    ("대한민국 법원", "scourt.go.kr"),
    ("헌법재판소", "ccourt.go.kr"),
    ("외교부", "mofa.go.kr"),
    ("mofa.go.kr", "mofa.go.kr"),
    ("congress.gov", "congress.gov"),
    ("european parliament", "europarl.europa.eu"),
    ("european commission", "ec.europa.eu"),
    ("voa 한국어", "voakorea.com"),
    ("연합뉴스tv", "yonhapnewstv.co.kr"),
    ("한국일보", "hankookilbo.com"),
    ("경향신문", "khan.co.kr"),
    ("동아일보", "donga.com"),
    ("서울신문", "seoul.co.kr"),
    ("매일경제", "mk.co.kr"),
    ("한국경제", "hankyung.com"),
    ("머니투데이", "mt.co.kr"),
    ("서울경제", "sedaily.com"),
    ("이데일리", "edaily.co.kr"),
    ("세계일보", "segye.com"),
    ("국민일보", "kmib.co.kr"),
    ("문화일보", "munhwa.com"),
    ("헤럴드경제", "heraldcorp.com"),
    ("노컷뉴스", "nocutnews.co.kr"),
    ("전자신문", "etnews.com"),
    ("ZDNET Korea", "zdnet.co.kr"),
    ("지디넷코리아", "zdnet.co.kr"),
    ("법률신문", "lawtimes.co.kr"),
    ("조세일보", "joseilbo.com"),
    ("한국농어민신문", "agrinet.co.kr"),
    ("부산일보", "busan.com"),
    ("뉴시스", "newsis.com"),
    ("뉴스1", "news1.kr"),
    ("ytn", "ytn.co.kr"),
    ("sbs", "sbs.co.kr"),
    ("mbc", "imbc.com"),
    ("mbn", "mbn.co.kr"),
    ("채널a", "ichannela.com"),
    ("channel a", "ichannela.com"),
    ("the korea times", "koreatimes.co.kr"),
    ("korea times", "koreatimes.co.kr"),
    ("the diplomat", "thediplomat.com"),
    ("carnegie endowment", "carnegieendowment.org"),
    ("bbc", "bbc.com"),
    ("the korea herald", "koreaherald.com"),
    ("reuters", "reuters.com"),
    ("human rights watch", "hrw.org"),
    ("chosunbiz", "chosun.com"),
    ("조선일보", "chosun.com"),
    ("38 north", "38north.org"),
    ("hankyoreh", "hani.co.kr"),
    ("한겨레", "hani.co.kr"),
    ("council on foreign relations", "cfr.org"),
    ("korea joongang daily", "joongang.co.kr"),
    ("중앙일보", "joongang.co.kr"),
    ("csis", "csis.org"),
    ("brookings", "brookings.edu"),
    ("ap news", "apnews.com"),
    ("nhk", "nhk.or.jp"),
    ("asahi", "asahi.com"),
    ("mainichi", "mainichi.jp"),
    ("yomiuri", "yomiuri.co.jp"),
    ("japan times", "japantimes.co.jp"),
    ("xinhua", "xinhuanet.com"),
    ("people's daily", "people.com.cn"),
    ("peoples daily", "people.com.cn"),
    ("yonhap", "yna.co.kr"),
    ("연합뉴스", "yna.co.kr"),
)

USER_AGENT = "Mozilla/5.0 (compatible; K-SOVA-SourceDiscovery/1.0)"
DISCOVERY_METADATA_KEYS = (
    "candidate_id", "source_role", "jurisdiction", "source_variant",
)
SECONDARY_OPINION_PATH = re.compile(
    r"/(?:opinion|editorials?|columns?|letters?|commentary)(?:/|$)", re.IGNORECASE
)
SECONDARY_EDITORIAL_TITLE = re.compile(
    r"^\s*(?:"
    r"\[(?:analysis|opinion|editorial|commentary|column)\]"
    r"|\[[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,3}\]"
    r"|\[?(?:사설|칼럼|시론|기고|논단)\]?(?:\s|:|$)"
    r")",
    re.IGNORECASE,
)
LISTING_PATH_ENDING = re.compile(
    r"/(?:all|index(?:\.s?html?)?|latest|archive|search|overview)/?$",
    re.IGNORECASE,
)
GENERIC_LISTING_TITLE = re.compile(
    r"^\s*(?:전체\s*[|｜]|all(?:\s+news)?(?:\s*[|｜-]|$)|latest(?:\s+news)?(?:\s*[|｜-]|$))",
    re.IGNORECASE,
)
SECONDARY_OPINION_MARKERS = (
    "essay contest",
    "op-ed",
    "opinion contributor",
    "the views expressed in this article",
    "외부 필진의 견해",
    "본 칼럼은",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def publisher_domain(value: str) -> str | None:
    folded = re.sub(r"\s+", " ", value.casefold()).strip()
    host = folded.removeprefix("www.")
    if "." in host and trusted_jurisdiction(host)[1] in {"A", "B"}:
        return host
    for marker, domain in PUBLISHER_DOMAINS:
        if marker in folded:
            return domain
    return None


def clean_title(title: str, publisher: str) -> str:
    value = re.sub(r"\s+", " ", title).strip()
    suffixes = [publisher, publisher.split(" - ")[0]]
    for suffix in filter(None, suffixes):
        value = re.sub(rf"\s+[-|–—]\s+{re.escape(suffix)}\s*$", "", value, flags=re.IGNORECASE)
    return value[:240]


def secondary_unsuitable_for_position(source: dict[str, Any]) -> bool:
    retrieval = source.get("retrieval") or {}
    text = " ".join((
        str(source.get("title") or ""),
        str(source.get("url") or ""),
        str(retrieval.get("text_excerpt") or ""),
    )).casefold()
    title = str(source.get("title") or "")
    path = urllib.parse.urlsplit(str(source.get("url") or "")).path
    return (
        bool(SECONDARY_OPINION_PATH.search(path))
        or bool(SECONDARY_EDITORIAL_TITLE.search(title))
        or any(marker in text for marker in SECONDARY_OPINION_MARKERS)
    )


def likely_listing_page(url: str, title: str = "") -> bool:
    path = urllib.parse.urlsplit(url).path
    return bool(LISTING_PATH_ENDING.search(path)) or bool(GENERIC_LISTING_TITLE.search(title))


def decode_google_news_url(url: str, retries: int = 2) -> str | None:
    """Resolve an RSS article token without trusting or fetching its target host."""
    parsed = urllib.parse.urlsplit(url)
    if not host_matches((parsed.hostname or "").casefold(), "news.google.com"):
        return None
    article_id = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,500}", article_id):
        return None

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            proxy = os.environ.get("KSOVA_GOOGLE_NEWS_PROXY", "").strip()
            common = [
                "curl", "--location", "--fail", "--silent", "--show-error",
                "--max-time", "20", "--user-agent", USER_AGENT,
            ]
            if proxy:
                common.extend(("--proxy", proxy))
            # Google News RSS tokens currently expose their decode metadata on
            # the original ``/rss/articles/`` route.  Rewriting those links to
            # the older ``/articles/`` route can return a bot-check page with
            # no signature fields, making every otherwise valid lead look
            # unresolved.  Preserve the route family supplied by the feed.
            article_route = (
                "/rss/articles/" if "/rss/articles/" in parsed.path
                else "/articles/"
            )
            parameter_url = (
                f"https://news.google.com{article_route}{article_id}"
                "?hl=en-US&gl=US&ceid=US:en"
            )
            parameter_process = subprocess.run(
                [*common, parameter_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=25,
            )
            if parameter_process.returncode != 0:
                raise RuntimeError(
                    parameter_process.stderr.decode("utf-8", "replace").strip()
                )
            html = parameter_process.stdout.decode("utf-8", "replace")
            timestamp_match = re.search(r'data-n-a-ts="([0-9]+)"', html)
            signature_match = re.search(r'data-n-a-sg="([A-Za-z0-9_-]+)"', html)
            if not timestamp_match or not signature_match:
                return None

            inner = json.dumps(
                [
                    "garturlreq",
                    [
                        [
                            "X", "X", ["X", "X"], None, None, 1, 1,
                            "US:en", None, 1, None, None, None, None, None, 0, 1,
                        ],
                        "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0,
                    ],
                    article_id,
                    int(timestamp_match.group(1)),
                    signature_match.group(1),
                ],
                separators=(",", ":"),
            )
            batch = [[["Fbv4je", inner, None, "generic"]]]
            body = urllib.parse.urlencode({
                "f.req": json.dumps(batch, separators=(",", ":")),
            }).encode("utf-8")
            decode_process = subprocess.run(
                [
                    *common,
                    "--header", "Content-Type: application/x-www-form-urlencoded;charset=UTF-8",
                    "--data-binary", "@-",
                    "https://news.google.com/_/DotsSplashUi/data/batchexecute",
                ],
                input=body,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=25,
            )
            if decode_process.returncode != 0:
                raise RuntimeError(
                    decode_process.stderr.decode("utf-8", "replace").strip()
                )
            payload_text = decode_process.stdout.decode("utf-8", "replace")
            for chunk in payload_text.split("\n\n"):
                try:
                    payload = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, list):
                    continue
                for entry in payload:
                    if not isinstance(entry, list) or len(entry) < 3 or entry[0] != "wrb.fr":
                        continue
                    decoded = json.loads(entry[2])
                    if (
                        isinstance(decoded, list)
                        and len(decoded) >= 2
                        and decoded[0] == "garturlres"
                        and isinstance(decoded[1], str)
                    ):
                        target = urllib.parse.urlsplit(decoded[1])
                        if target.scheme in {"http", "https"} and target.hostname:
                            return decoded[1]
            return None
        except Exception as exc:  # isolate transient Google token-resolution failures
            last_error = exc
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
    if last_error:
        return None
    return None


def resolve_url(record: dict[str, Any], domain: str) -> tuple[str | None, str | None, str | None]:
    direct_url = str(record.get("url") or "")
    direct_host = base_host(direct_url)
    record_title = str(record.get("title") or "")
    if (
        host_matches(direct_host, domain)
        and not likely_homepage(direct_url)
        and not likely_listing_page(direct_url, record_title)
    ):
        return direct_url, str(record.get("title") or ""), "direct"
    decode_outcome = None
    if host_matches(direct_host, "news.google.com"):
        decoded_url = decode_google_news_url(direct_url)
        if (
            decoded_url
            and host_matches(base_host(decoded_url), domain)
            and not likely_homepage(decoded_url)
            and not likely_listing_page(decoded_url, record_title)
        ):
            return decoded_url, str(record.get("title") or ""), "google_news_decode"
        decode_outcome = "decoded_host_mismatch" if decoded_url else "google_news_decode_failed"
    title = clean_title(str(record.get("title") or ""), str(record.get("publisher") or ""))
    query = f"site:{domain} {title}"
    # Yahoo Japan resolves Korean quoted titles reliably and returns ordinary
    # publisher URLs. DDG and Bing remain fallbacks for gaps in its index.
    results = search_results(query, retries=1)
    if not any(
        not result.get("error")
        and host_matches(base_host(str(result.get("url") or "")), domain)
        for result in results
    ):
        results.extend(duckduckgo_lite_results(query, retries=1))
    if not any(
        not result.get("error")
        and host_matches(base_host(str(result.get("url") or "")), domain)
        for result in results
    ):
        results.extend(bing_rss_results(query))
    ranked = []
    for result in results:
        if result.get("error"):
            continue
        url = str(result.get("url") or "")
        if (
            not host_matches(base_host(url), domain)
            or likely_homepage(url)
            or likely_listing_page(url, str(result.get("title") or ""))
        ):
            continue
        score = text_score(title, f"{result.get('title', '')} {result.get('snippet', '')}")
        ranked.append((score, result))
    if not ranked:
        return None, None, decode_outcome or "not_found"
    score, result = max(ranked, key=lambda row: row[0])
    if score < 0.24:
        return None, None, "weak_title_match"
    return str(result["url"]), str(result.get("title") or title), "title_search"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leads", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--discourse-mode", action="store_true",
        help=(
            "Accept attributed Korean research, constituency, and opinion documents "
            "as position evidence only; never promote them to factual anchors."
        ),
    )
    args = parser.parse_args()

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    input_count = 0
    for path in args.leads:
        for lead in read_jsonl(path):
            input_count += 1
            domain = (
                publisher_domain(str(lead.get("publisher") or ""))
                or publisher_domain(str(lead.get("host") or ""))
            )
            if not domain:
                continue
            title_key = normalized(clean_title(
                str(lead.get("title") or ""), str(lead.get("publisher") or "")
            ))
            if not title_key:
                continue
            key = (domain, title_key)
            if key not in grouped:
                grouped[key] = {
                    "record": lead,
                    "domain": domain,
                    "discoveries": [],
                    "lead_ids": [],
                }
            grouped[key]["lead_ids"].append(str(lead["lead_id"]))
            grouped[key]["discoveries"].extend(lead.get("discoveries") or [{
                key: lead.get(key)
                for key in ("lane", "language", "query", "engine", "rank", *DISCOVERY_METADATA_KEYS)
                if lead.get(key) is not None
            }])

    def resolve(group: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        record, domain = group["record"], group["domain"]
        url, title, method = resolve_url(record, domain)
        if not url:
            return None, str(method)
        jurisdiction, tier = trusted_jurisdiction(base_host(url))
        if tier not in {"A", "B"}:
            return None, "source_policy"
        if args.discourse_mode and jurisdiction != "KR":
            return None, "non_korean_discourse"
        discoveries = []
        seen = set()
        for discovery in group["discoveries"]:
            identity = tuple(
                discovery.get(key)
                for key in ("lane", "language", "query", "engine", "rank", *DISCOVERY_METADATA_KEYS)
            )
            if identity not in seen:
                discoveries.append(discovery)
                seen.add(identity)
        contexts = list(dict.fromkeys(filter(None, (
            str(record.get("title") or ""),
            *(str(row.get("query") or "") for row in discoveries),
        ))))
        source_id = "v2lead_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        discourse_categories = sorted({
            str(row.get("source_variant") or "")
            for row in discoveries if row.get("source_variant")
        })
        source = {
            "schema_version": "ksova-v2-calibration-source-1.0",
            "source_id": source_id,
            "title": title,
            "url": url,
            "jurisdiction": jurisdiction,
            "authority_type": (
                "domestic_public_discourse"
                if args.discourse_mode else
                ("official_web_document" if tier == "A" else "trusted_secondary_document")
            ),
            "issuing_authority": str(record.get("publisher") or domain),
            "section": None,
            "evidence_scope_ko": str(record.get("snippet") or "")[:1000],
            "evidence_claims": [],
            "snapshot_sha256": None,
            "availability": "search_discovered",
            "origins": ["trusted_discovery_headline_resolution"],
            "reliability_tier": tier,
            "allowed_uses": (
                ["issue_discovery", "position_evidence"]
                if args.discourse_mode else
                ["issue_discovery", "position_evidence", "fact_check", "generation_grounding"]
            ),
            "evidence_policy": (
                "attributed_position_only_no_unattributed_fact"
                if args.discourse_mode else "trusted_document"
            ),
            "discourse_categories": discourse_categories,
            "required_final_host_suffix": domain,
            "retrieval_context": contexts,
        }
        fetched = fetch_source(source, args.snapshot_dir)
        if args.discourse_mode:
            # fetch_source reapplies the generic A/B policy.  Narrow it again so
            # a discourse document can ground an attributed viewpoint but can
            # never silently become a factual or legal anchor downstream.
            fetched = {
                **fetched,
                "authority_type": "domestic_public_discourse",
                "allowed_uses": ["issue_discovery", "position_evidence"],
                "evidence_policy": "attributed_position_only_no_unattributed_fact",
                "discourse_categories": discourse_categories,
            }
        retrieval = fetched.get("retrieval") or {}
        if retrieval.get("status") not in {"fetched", "reused_verified_snapshot"}:
            return None, "fetch_failed"
        if likely_listing_page(
            str(retrieval.get("final_url") or fetched.get("url") or ""),
            str(fetched.get("title") or ""),
        ):
            return None, "listing_or_section_page"
        if not retrieval.get("relevance_pass"):
            return None, "relevance_failed"
        if tier == "B" and secondary_unsuitable_for_position(fetched) and not args.discourse_mode:
            return None, "secondary_opinion_or_essay"
        return {
            "schema_version": "ksova-v2-trusted-discovery-source-1.0",
            "source_lead_id": source_id,
            "lead_ids": sorted(set(group["lead_ids"])),
            "discoveries": discoveries,
            "resolution_method": method,
            "source": fetched,
        }, "fetched"

    output = []
    outcomes = Counter()
    groups = list(grouped.values())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(resolve, group) for group in groups]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row, outcome = future.result()
            outcomes[outcome] += 1
            if row:
                output.append(row)
            if index % 50 == 0:
                print(f"[resolve {index}/{len(groups)}] {dict(outcomes)}", flush=True)

    output.sort(key=lambda row: row["source_lead_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    report = {
        "schema_version": "ksova-v2-trusted-discovery-source-report-1.0",
        "input_leads": input_count,
        "trusted_headline_groups": len(groups),
        "fetched_sources": len(output),
        "outcomes": dict(sorted(outcomes.items())),
        "by_tier": dict(sorted(Counter(row["source"]["reliability_tier"] for row in output).items())),
        "by_jurisdiction": dict(sorted(Counter(row["source"]["jurisdiction"] for row in output).items())),
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
