#!/usr/bin/env python3
"""Collect broad, non-gold issue leads from news and web search feeds."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import subprocess
import time
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from discover_calibration_sources import (
    brave_results,
    duckduckgo_lite_results,
    naver_results,
    search_results,
)


USER_AGENT = "Mozilla/5.0 (compatible; KSOVABenchSourceDiscovery/1.0)"
KOREA_PATTERN = re.compile(
    r"한국|대한민국|북한|한반도|한미|한중|한일|조선|고려|고구려|발해|독도|이어도|"
    r"korea|korean|seoul|pyongyang|dprk|\brok\b|"
    r"韓国|朝鮮|韓半島|韩国|朝鲜|韩美|韩中|韩日",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣一-龥ぁ-んァ-ン]+", "", value.casefold())


def canonical_host(url: str) -> str:
    host = (urllib.parse.urlsplit(url).hostname or "").casefold()
    return host[4:] if host.startswith("www.") else host


def host_blocked(host: str, blocked: list[str]) -> bool:
    return any(host == value or host.endswith("." + value) for value in blocked)


def host_matches(host: str, suffixes: list[str]) -> bool:
    return any(host == value or host.endswith("." + value) for value in suffixes)


def site_restrictions(query: str) -> list[str]:
    """Return normalized hosts from explicit ``site:`` query operators."""
    return [
        canonical_host("https://" + value.strip("()[]{}<>,;\"'"))
        for value in re.findall(r"(?:^|\s)site:([^\s]+)", query, re.IGNORECASE)
        if value.strip("()[]{}<>,;\"'")
    ]


def fetch_xml(url: str, retries: int) -> ET.Element:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            proxy = os.environ.get("KSOVA_SEARCH_PROXY", "").strip()
            if proxy:
                process = subprocess.run(
                    [
                        "curl", "--location", "--fail", "--silent", "--show-error",
                        "--max-time", "25", "--user-agent", USER_AGENT,
                        "--proxy", proxy, url,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                if process.returncode != 0:
                    raise RuntimeError(
                        process.stderr.decode("utf-8", "replace").strip()
                    )
                return ET.fromstring(process.stdout)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=30) as response:
                return ET.fromstring(response.read())
        except Exception as exc:  # isolate transient search-feed failures
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def fetch_json(url: str, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError("JSON response is not an object")
            return payload
        except Exception as exc:  # isolate transient search API failures
            last_error = exc
            if attempt < retries:
                retry_after = 0.0
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                    try:
                        retry_after = float(exc.headers.get("Retry-After") or 0)
                    except (TypeError, ValueError):
                        retry_after = 0.0
                time.sleep(max(1.5 * (attempt + 1), min(retry_after, 60.0)))
    assert last_error is not None
    raise last_error


def text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    return re.sub(r"\s+", " ", child.text or "").strip() if child is not None else ""


def google_news(query: str, language: str, retries: int) -> list[dict[str, str]]:
    locale = {
        "ko": ("ko", "KR", "KR:ko"),
        "en": ("en-US", "US", "US:en"),
        "ja": ("ja", "JP", "JP:ja"),
        "zh": ("zh-CN", "CN", "CN:zh-Hans"),
    }.get(language, ("en-US", "US", "US:en"))
    params = urllib.parse.urlencode({"q": query, "hl": locale[0], "gl": locale[1], "ceid": locale[2]})
    root = fetch_xml("https://news.google.com/rss/search?" + params, retries)
    rows = []
    for item in root.findall("./channel/item"):
        source = item.find("source")
        rows.append({
            "title": text(item, "title"),
            "url": text(item, "link"),
            "snippet": text(item, "description"),
            "publisher": re.sub(r"\s+", " ", source.text or "").strip() if source is not None else "",
            "published_at": text(item, "pubDate"),
        })
    return rows


def bing_web(query: str, _language: str, retries: int) -> list[dict[str, str]]:
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"format": "rss", "q": query})
    root = fetch_xml(url, retries)
    return [{
        "title": text(item, "title"),
        "url": text(item, "link"),
        "snippet": text(item, "description"),
        "publisher": canonical_host(text(item, "link")),
        "published_at": text(item, "pubDate"),
    } for item in root.findall("./channel/item")]


def wikipedia_search(query: str, language: str, retries: int) -> list[dict[str, str]]:
    language_code = language if language in {"ko", "en", "ja", "zh"} else "en"
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 0,
        "srlimit": 20,
        "format": "json",
        "utf8": 1,
    })
    payload = fetch_json(f"https://{language_code}.wikipedia.org/w/api.php?{params}", retries)
    rows = []
    for result in payload.get("query", {}).get("search", []):
        title_value = str(result.get("title") or "").strip()
        if not title_value:
            continue
        page_name = urllib.parse.quote(title_value.replace(" ", "_"), safe="()_-")
        rows.append({
            "title": title_value,
            "url": f"https://{language_code}.wikipedia.org/wiki/{page_name}",
            "snippet": re.sub(r"<[^>]+>", " ", str(result.get("snippet") or "")),
            "publisher": f"{language_code}.wikipedia.org",
            "published_at": str(result.get("timestamp") or ""),
        })
    return rows


def duckduckgo_web(query: str, _language: str, retries: int) -> list[dict[str, str]]:
    return [
        {
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
            "snippet": str(row.get("snippet") or ""),
            "publisher": canonical_host(str(row.get("url") or "")),
            "published_at": "",
        }
        for row in duckduckgo_lite_results(query, max(1, retries + 1))
        if not row.get("error")
    ]


def brave_web(query: str, _language: str, retries: int) -> list[dict[str, str]]:
    return [
        {
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
            "snippet": str(row.get("snippet") or ""),
            "publisher": canonical_host(str(row.get("url") or "")),
            "published_at": "",
        }
        for row in brave_results(query, max(1, retries + 1))
        if not row.get("error")
    ]


NAVER_STARTS = (1,)


def naver_web(query: str, _language: str, _retries: int) -> list[dict[str, str]]:
    rows = [
        {
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
            "snippet": str(row.get("snippet") or ""),
            "publisher": canonical_host(str(row.get("url") or "")),
            "published_at": "",
        }
        for start in NAVER_STARTS
        for row in naver_results(query, start=start)
        if not row.get("error")
    ]
    deduped = {}
    for row in rows:
        deduped.setdefault(canonical_url(row["url"]), row)
    return list(deduped.values())


def naver_news(query: str, _language: str, _retries: int) -> list[dict[str, str]]:
    rows = [
        {
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
            "snippet": str(row.get("snippet") or ""),
            "publisher": canonical_host(str(row.get("url") or "")),
            "published_at": "",
        }
        for start in NAVER_STARTS
        for row in naver_results(query, start=start, where="news")
        if not row.get("error")
    ]
    deduped = {}
    for row in rows:
        deduped.setdefault(canonical_url(row["url"]), row)
    return list(deduped.values())


def yahoo_web(query: str, _language: str, retries: int) -> list[dict[str, str]]:
    return [
        {
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
            "snippet": str(row.get("snippet") or ""),
            "publisher": canonical_host(str(row.get("url") or "")),
            "published_at": "",
        }
        for row in search_results(query, max(1, retries + 1))
        if not row.get("error")
    ]


ENGINES = {
    "brave_web": brave_web,
    "google_news": google_news,
    "bing_web": bing_web,
    "duckduckgo_web": duckduckgo_web,
    "naver_web": naver_web,
    "naver_news": naver_news,
    "yahoo_web": yahoo_web,
    "wikipedia_search": wikipedia_search,
}

DISCOVERY_METADATA_KEYS = (
    "candidate_id", "domain_id", "source_role", "jurisdiction",
    "source_variant", "query_kind", "require_korea_signal",
    "search_subject_ko", "search_subject_en",
)


def canonical_url(url: str) -> str:
    """Return a conservative URL identity for discovery deduplication."""
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return url.strip()
    query = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in {"fbclid", "gclid", "ref"}
    ]
    path = parts.path.rstrip("/") or "/"
    return urllib.parse.urlunsplit((
        parts.scheme.casefold(), parts.netloc.casefold(), path,
        urllib.parse.urlencode(sorted(query)), "",
    ))


def discovery_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = ("lane", "language", "query", "engine", "rank", *DISCOVERY_METADATA_KEYS)
    return {key: record.get(key) for key in keys if record.get(key) is not None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--existing",
        type=Path,
        help=(
            "Prior discovery-lead JSONL to merge (rows must contain lead_id). "
            "This is not a canonical sources.jsonl input."
        ),
    )
    parser.add_argument("--retry-report", type=Path)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-results-per-query", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument(
        "--naver-start", action="append", type=int,
        choices=range(1, 92, 10),
        help=(
            "Naver result offsets to collect (1, 11, 21, ...). Repeat for "
            "multiple pages; defaults to the first page only."
        ),
    )
    parser.add_argument(
        "--domain-id",
        action="append",
        choices=[f"D{index:02d}" for index in range(1, 9)],
        help=(
            "Run only queries assigned to this K-SOVA domain. Repeat the "
            "option to select more than one domain."
        ),
    )
    parser.add_argument(
        "--source-variant",
        action="append",
        help=(
            "Run only queries with this source_variant. Repeat the option to "
            "select multiple variants."
        ),
    )
    parser.add_argument(
        "--candidate-id", action="append",
        help="Run only queries linked to the named topic/event id; repeatable.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help="Deterministically shuffle selected queries before an optional limit.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        help="Run at most this many selected queries after filtering and shuffling.",
    )
    parser.add_argument(
        "--query-offset", type=int, default=0,
        help="Skip this many selected queries before applying query-limit.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=sorted(ENGINES),
        help="Override the engines configured in the query file.",
    )
    parser.add_argument(
        "--exclude-engines",
        nargs="+",
        choices=sorted(ENGINES),
        default=[],
        help="Skip temporarily unavailable engines without rewriting the query plan.",
    )
    args = parser.parse_args()

    # Validate resumable discovery input before issuing any network requests.
    # Canonical source rows and discovery leads are both JSONL and were easy to
    # confuse when this option had no schema guard; failing only after all
    # searches wastes the run and encourages ad-hoc recovery files.
    existing_rows: list[dict[str, Any]] = []
    if args.existing:
        if not args.existing.exists():
            raise FileNotFoundError(f"existing discovery leads not found: {args.existing}")
        existing_rows = [
            json.loads(line)
            for line in args.existing.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        invalid_existing = [
            index for index, row in enumerate(existing_rows, 1)
            if not isinstance(row, dict) or not str(row.get("lead_id") or "").strip()
        ]
        if invalid_existing:
            raise ValueError(
                "--existing accepts discovery leads only; missing lead_id at "
                f"rows {invalid_existing[:10]}"
            )

    global NAVER_STARTS
    NAVER_STARTS = tuple(dict.fromkeys(args.naver_start or [1]))

    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected_queries = list(enumerate(config["queries"], 1))
    if args.domain_id:
        selected_domains = set(args.domain_id)
        selected_queries = [
            (index, entry)
            for index, entry in selected_queries
            if entry.get("domain_id") in selected_domains
        ]
    if args.source_variant:
        selected_variants = set(args.source_variant)
        selected_queries = [
            (index, entry)
            for index, entry in selected_queries
            if entry.get("source_variant") in selected_variants
        ]
    if args.candidate_id:
        selected_candidates = set(args.candidate_id)
        selected_queries = [
            (index, entry)
            for index, entry in selected_queries
            if entry.get("candidate_id") in selected_candidates
        ]
    if args.retry_report:
        retry_payload = json.loads(args.retry_report.read_text(encoding="utf-8"))
        failed_indices = {
            int(row["query_index"])
            for row in retry_payload.get("errors", [])
            if isinstance(row, dict) and str(row.get("query_index", "")).isdigit()
        }
        selected_queries = [
            (index, entry) for index, entry in selected_queries if index in failed_indices
        ]
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(selected_queries)
    if args.query_offset < 0:
        raise ValueError("query-offset must not be negative")
    if args.query_offset:
        selected_queries = selected_queries[args.query_offset:]
    if args.query_limit is not None:
        if args.query_limit < 1:
            raise ValueError("query-limit must be positive")
        selected_queries = selected_queries[:args.query_limit]
    blocked = [str(value).casefold() for value in config.get("blocked_hosts", [])]
    discovery_only = [
        str(value).casefold() for value in config.get("discovery_only_hosts", [])
    ]
    excluded_engines = set(args.exclude_engines)
    configured_engines = [
        engine
        for engine in (args.engines or config.get("engines") or list(ENGINES))
        if engine not in excluded_engines
    ]
    worker_count = args.max_workers
    if set(configured_engines) == {"wikipedia_search"}:
        worker_count = min(worker_count, 2)
    records: list[dict[str, Any]] = []
    errors = []
    raw_count = 0

    tasks: list[tuple[int, dict[str, Any], str]] = []
    for query_index, entry in selected_queries:
        entry_engines = args.engines or entry.get("engines") or configured_engines
        for engine_name in entry_engines:
            if engine_name in excluded_engines:
                continue
            if engine_name not in ENGINES:
                raise ValueError(f"unsupported engine: {engine_name}")
            tasks.append((query_index, entry, engine_name))

    def run_task(task: tuple[int, dict[str, Any], str]) -> tuple[
        int, dict[str, Any], str, list[dict[str, str]], str | None
    ]:
        query_index, entry, engine_name = task
        try:
            results = ENGINES[engine_name](entry["query"], entry["language"], args.retries)
            if args.delay:
                time.sleep(args.delay)
            return query_index, entry, engine_name, results, None
        except Exception as exc:
            return query_index, entry, engine_name, [], f"{type(exc).__name__}: {exc}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        progress_interval = max(100, len(futures) // 20)
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), 1
        ):
            query_index, entry, engine_name, results, error = future.result()
            if error:
                errors.append({
                    "query_index": query_index,
                    "engine": engine_name,
                    "query": entry["query"],
                    "error": error,
                })
            else:
                for rank, result in enumerate(results[:args.max_results_per_query], 1):
                    raw_count += 1
                    title = re.sub(r"\s+", " ", result["title"]).strip()
                    url = result["url"].strip()
                    if len(title) < 12 or not url:
                        continue
                    relevance_text = " ".join((
                        title,
                        result.get("snippet", ""),
                        result.get("publisher", ""),
                    ))
                    if entry.get("require_korea_signal", True) and not KOREA_PATTERN.search(
                        relevance_text
                    ):
                        continue
                    host = canonical_host(url)
                    if host_blocked(host, blocked):
                        continue
                    # Public search engines occasionally ignore an explicit
                    # site operator and return a high-ranking mirror or wiki.
                    # Those results would contaminate an institutional census,
                    # so enforce the requested host after URL resolution.
                    requested_sites = site_restrictions(entry["query"])
                    if requested_sites and not host_matches(host, requested_sites):
                        continue
                    identity = canonical_url(url)
                    if not identity:
                        continue
                    record = {
                        "lead_id": "ksdl_" + hashlib.sha256(
                            identity.encode("utf-8")
                        ).hexdigest()[:16],
                        "lane": entry["lane"],
                        "language": entry["language"],
                        "query": entry["query"],
                        "engine": engine_name,
                        "rank": rank,
                        "title": title,
                        "url": url,
                        "host": host,
                        "publisher": result.get("publisher", ""),
                        "published_at": result.get("published_at", ""),
                        "snippet": re.sub(
                            r"<[^>]+>", " ", result.get("snippet", "")
                        )[:1000].strip(),
                        "evidence_role": "discovery_only",
                        "reliability_tier": "C",
                        "allowed_uses": ["issue_discovery", "query_expansion"],
                        "reliability_reason": (
                            "editable_tertiary_index"
                            if host_matches(host, discovery_only)
                            else "search_result_not_directly_verified"
                        ),
                    }
                    for key in DISCOVERY_METADATA_KEYS:
                        if entry.get(key) is not None:
                            record[key] = entry[key]
                    records.append(record)
            if completed % progress_interval == 0 or completed == len(futures):
                print(
                    f"[search {completed}/{len(futures)}] "
                    f"raw={raw_count} retained={len(records)} errors={len(errors)}",
                    flush=True,
                )

    # The same URL may be returned by multiple languages and engines. Preserve
    # every discovery query as provenance while emitting one compact URL lead.
    merged: dict[str, dict[str, Any]] = {}
    existing_count = 0
    if existing_rows:
        existing_count = len(existing_rows)
        for record in existing_rows:
            record["evidence_role"] = "discovery_only"
            record["reliability_tier"] = "C"
            record["allowed_uses"] = ["issue_discovery", "query_expansion"]
            record.setdefault("reliability_reason", "search_result_not_directly_verified")
            record.setdefault("discoveries", [])
            merged[record["lead_id"]] = record
    for record in records:
        lead_id = record["lead_id"]
        if lead_id not in merged:
            record["discoveries"] = [discovery_record(record)]
            merged[lead_id] = record
            continue
        prior = merged[lead_id]
        prior["discoveries"].append(discovery_record(record))
        if len(record["snippet"]) > len(prior["snippet"]):
            prior["snippet"] = record["snippet"]
        if prior["host"] == "news.google.com" and record["host"] != "news.google.com":
            for key in ("url", "host", "publisher", "published_at"):
                prior[key] = record[key]

    for record in merged.values():
        unique_discoveries = []
        seen_discoveries = set()
        for discovery in record.get("discoveries", []):
            identity = tuple(
                discovery.get(key)
                for key in ("lane", "language", "query", "engine", "rank", *DISCOVERY_METADATA_KEYS)
            )
            if identity in seen_discoveries:
                continue
            seen_discoveries.add(identity)
            unique_discoveries.append(discovery)
        record["discoveries"] = unique_discoveries

    rows = sorted(merged.values(), key=lambda row: (row["lane"], row["lead_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "ksova-v2-source-discovery-report-1.0",
        "queries": len(selected_queries),
        "config_queries": len(config["queries"]),
        "engine_requests": len(tasks),
        "workers": worker_count,
        "naver_starts": list(NAVER_STARTS),
        "query_offset": args.query_offset,
        "excluded_engines": sorted(excluded_engines),
        "raw_results": raw_count,
        "existing_leads": existing_count,
        "unique_leads": len(rows),
        "by_lane": dict(sorted(Counter(row["lane"] for row in rows).items())),
        "by_engine": dict(sorted(Counter(row["engine"] for row in rows).items())),
        "by_language": dict(sorted(Counter(row["language"] for row in rows).items())),
        "top_publishers": Counter(row["publisher"] or row["host"] for row in rows).most_common(30),
        "errors": errors,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, ensure_ascii=False))
    return 1 if len(errors) > max(3, report["engine_requests"] // 10) else 0


if __name__ == "__main__":
    raise SystemExit(main())
