#!/usr/bin/env python3
"""Collect topic-independent leads from Korean public policy archives.

Only archive list pages are read here.  Titles and document URLs are discovery
hints; ``data/expand.py resolve`` must still fetch a substantive body, and the
ordinary event/generation/validation gates remain mandatory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from collect_sitemap_discovery_leads import (
    DOMAIN_LABELS,
    DOMESTIC_ACTOR_PATTERN,
    EXTERNAL_PATTERN,
    FRICTION_PATTERN,
    KOREA_PATTERN,
    broad_anchors,
    broad_domain,
    evenly_sample,
)


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
ARCHIVE_SIGNAL_PATTERN = re.compile(
    r"국익|주권|한반도|북한|한미|한중|한일|재외|국제기준|국제규범|"
    r"주요국|해외 사례|비교법|시사점|정책과제|대응과제|대응전략|"
    r"공급망|수출통제|통상|동맹|방위|제재|소버린|K-",
    re.IGNORECASE,
)

ARCHIVES: dict[str, dict[str, Any]] = {
    "nars": {
        "publisher": "국회입법조사처",
        "host": "www.nars.go.kr",
        "max_page": 469,
        "list_url": "https://www.nars.go.kr/report/list.do?page={page}&cmsCode=CM0043",
    },
    "inss": {
        "publisher": "국가안보전략연구원",
        "host": "www.inss.re.kr",
        "max_page": 89,
        "list_url": "https://www.inss.re.kr/publication/bbs/ib_list.do?page={page}",
    },
    "kiep": {
        "publisher": "대외경제정책연구원",
        "host": "www.kiep.go.kr",
        "max_page": 145,
        "list_url": (
            "https://www.kiep.go.kr/gallery.es?mid=a10101010000&bid=0001&nPage={page}"
        ),
    },
    "kisdi": {
        "publisher": "정보통신정책연구원",
        "host": "www.kisdi.re.kr",
        "max_page": 21,
        "default_domain": "D08",
        "list_url": (
            "https://www.kisdi.re.kr/report/list.do?"
            "arrMasterId=3934550&key=m2101113025536&pageIndex={page}"
        ),
    },
    "kmi": {
        "publisher": "한국해양수산개발원",
        "host": "www.kmi.re.kr",
        "max_page": 45,
        "default_domain": "D03",
        "list_url": (
            "https://www.kmi.re.kr/web/board/web/trebook/list.do?"
            "page={page}&rbsIdx=273"
        ),
    },
    "khs": {
        "publisher": "국가유산청",
        "host": "www.khs.go.kr",
        "max_page": 1172,
        "default_domain": "D05",
        "list_url": (
            "https://www.khs.go.kr/newsBbz/selectNewsBbzList.do?"
            "mn=NS_01_02_02&sectionId=all_sec_1&pageIndex={page}"
        ),
    },
    "humanrights": {
        "publisher": "국가인권위원회",
        "host": "www.humanrights.go.kr",
        "max_page": 503,
        "default_domain": "D02",
        "list_url": (
            "https://www.humanrights.go.kr/base/board/list?"
            "boardManagementNo=24&menuLevel=3&menuNo=91&page={page}"
        ),
    },
}


def fetch_text(url: str, timeout: int = 25) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def clean_title(value: str) -> str:
    value = BeautifulSoup(html.unescape(value), "html.parser").get_text(" ", strip=True)
    value = re.sub(r"^\[[^\]]{1,30}\]\s*", "", value)
    return re.sub(r"\s+", " ", value).strip(" -")


def parse_nars(body: str, page: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    rows = []
    for node in soup.find_all("a"):
        match = re.search(r"javascript:view\(['\"]([0-9]+)['\"]\)", str(node.get("href") or ""))
        title = clean_title(node.get_text(" ", strip=True))
        if not match or len(title) < 8:
            continue
        rows.append({
            "title": title,
            "url": (
                "https://www.nars.go.kr/report/view.do?"
                f"brdSeq={match.group(1)}&cmsCode=CM0043"
            ),
            "page": page,
        })
    return rows


def parse_inss(body: str, page: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    rows = []
    for node in soup.find_all("a"):
        match = re.search(
            r"setView\(['\"]([0-9]+)['\"],['\"]ib['\"]\)",
            str(node.get("onclick") or ""),
        )
        title = clean_title(node.get_text(" ", strip=True))
        if not match or len(title) < 8:
            continue
        rows.append({
            "title": title,
            "url": (
                "https://www.inss.re.kr/publication/bbs/ib_view.do?"
                f"bbsId=ib&nttId={match.group(1)}&page={page}"
            ),
            "page": page,
        })
    return rows


def parse_kiep(body: str, page: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    rows = []
    seen = set()
    for node in soup.find_all("img"):
        source = str(node.get("src") or "")
        match = re.search(r"galleryThumbView\.es\?[^\"']*\blist_no=([0-9]+)", source)
        title = clean_title(str(node.get("alt") or ""))
        if not match or len(title) < 8 or match.group(1) in seen:
            continue
        seen.add(match.group(1))
        rows.append({
            "title": title,
            "url": (
                "https://www.kiep.go.kr/gallery.es?act=view&bid=0001&"
                f"list_no={match.group(1)}&mid=a10101010000"
            ),
            "page": page,
        })
    return rows


def parse_kisdi(body: str, page: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    rows = []
    seen = set()
    for node in soup.find_all("a"):
        match = re.search(
            r"goView\(['\"]([0-9]+)['\"],\s*['\"]([0-9]+)['\"]\)",
            str(node.get("onclick") or ""),
        )
        title_node = node.find("strong")
        title = clean_title(title_node.get_text(" ", strip=True) if title_node else "")
        if not match or len(title) < 8 or match.group(2) in seen:
            continue
        seen.add(match.group(2))
        rows.append({
            "title": title,
            "url": (
                "https://www.kisdi.re.kr/report/view.do?"
                "key=m2101113025536&"
                f"masterId={match.group(1)}&arrMasterId=3934550&artId={match.group(2)}"
            ),
            "page": page,
        })
    return rows


def parse_kmi(body: str, page: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    rows = []
    seen = set()
    for node in soup.find_all("a"):
        href = html.unescape(str(node.get("href") or ""))
        match = re.search(r"(?:^|/)view\.do\?[^#]*\brbsIdx=273[^#]*\bidx=([0-9]+)", href)
        title = clean_title(str(node.get("title") or node.get_text(" ", strip=True)))
        title = re.sub(r"\s+보기$", "", title).strip()
        if not match or len(title) < 8 or match.group(1) in seen:
            continue
        seen.add(match.group(1))
        rows.append({
            "title": title,
            "url": (
                "https://www.kmi.re.kr/web/trebook/view.do?"
                f"rbsIdx=273&idx={match.group(1)}"
            ),
            "page": page,
        })
    return rows


def parse_khs(body: str, page: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    rows = []
    seen = set()
    for node in soup.select("a.b_tit"):
        href = html.unescape(str(node.get("href") or ""))
        match = re.search(r"[?&]newsItemId=([0-9]+)", href)
        title = clean_title(node.get_text(" ", strip=True))
        if not match or len(title) < 8 or match.group(1) in seen:
            continue
        seen.add(match.group(1))
        rows.append({
            "title": title,
            "url": (
                "https://www.khs.go.kr/newsBbz/selectNewsBbzView.do?"
                f"newsItemId={match.group(1)}&sectionId=b_sec_1&mn=NS_01_02_02"
            ),
            "page": page,
        })
    return rows


def parse_humanrights(body: str, page: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    rows = []
    seen = set()
    for node in soup.select("a.td-title"):
        href = html.unescape(str(node.get("href") or ""))
        match = re.search(r"[?&]boardNo=([0-9]+)", href)
        title = clean_title(node.get_text(" ", strip=True))
        if not match or len(title) < 8 or match.group(1) in seen:
            continue
        seen.add(match.group(1))
        rows.append({
            "title": title,
            "url": (
                "https://www.humanrights.go.kr/base/board/read?"
                f"boardManagementNo=24&boardNo={match.group(1)}&"
                "menuLevel=3&menuNo=91"
            ),
            "page": page,
        })
    return rows


PARSERS = {
    "nars": parse_nars,
    "inss": parse_inss,
    "kiep": parse_kiep,
    "kisdi": parse_kisdi,
    "kmi": parse_kmi,
    "khs": parse_khs,
    "humanrights": parse_humanrights,
}


def retain(title: str, broad_external_policy: bool = False) -> bool:
    external = bool(EXTERNAL_PATTERN.search(title) or re.search(
        r"주요국|국제기준|국제규범|해외 사례|비교법", title, re.IGNORECASE
    ))
    korea_or_policy = bool(
        KOREA_PATTERN.search(title)
        or DOMESTIC_ACTOR_PATTERN.search(title)
        or ARCHIVE_SIGNAL_PATTERN.search(title)
    )
    decision_signal = bool(FRICTION_PATTERN.search(title) or ARCHIVE_SIGNAL_PATTERN.search(title))
    # A Korean policy archive is itself a useful discovery prior.  Requiring
    # every list-page title to repeat ``한국`` or ``시사점`` drops documents
    # such as an EU regulation, foreign judgment, or sanctions analysis before
    # its fetched body can reveal the concrete Korean stake.  The broad mode
    # remains bounded by an external actor/rule plus a decision/friction
    # signal; body retrieval and the ordinary source-closed event gates still
    # decide whether the document is usable for K-SOVA.
    if broad_external_policy:
        return external and decision_signal
    return external and korea_or_policy and decision_signal


def lead(archive: str, row: dict[str, Any], rank: int) -> dict[str, Any]:
    config = ARCHIVES[archive]
    title = row["title"]
    domain_id = str(config.get("default_domain") or broad_domain(title))
    anchors = broad_anchors(title, "")
    identity = row["url"]
    return {
        "lead_id": "ksdl_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        "lane": DOMAIN_LABELS[domain_id],
        "domain_id": domain_id,
        "language": "ko",
        "query": " ".join(anchors),
        "engine": "public_policy_archive",
        "rank": rank,
        "title": title,
        "url": row["url"],
        "host": config["host"],
        "publisher": config["publisher"],
        "published_at": "",
        "snippet": "",
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "policy_archive_metadata_not_directly_verified",
        "discoveries": [{
            "lane": DOMAIN_LABELS[domain_id],
            "domain_id": domain_id,
            "language": "ko",
            "query": " ".join(anchors),
            "engine": "public_policy_archive",
            "rank": rank,
            "source_role": "korean_discourse",
            "source_variant": f"public_policy_archive_{archive}",
            "query_kind": "broad_policy_archive",
            "jurisdiction": "KR",
            "require_korea_signal": True,
            "search_subject_ko": " ".join(anchors[:5]),
            "search_subject_en": "",
            "title_matched_terms": anchors,
        }],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", choices=sorted(ARCHIVES), required=True)
    parser.add_argument("--sample-pages", type=int, default=60)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--broad-external-policy",
        action="store_true",
        help=(
            "Within a Korean policy archive, retain titles with an external "
            "actor/rule and a concrete decision or friction signal even when "
            "the title itself omits an explicit Korea/시사점 marker. Fetched-body "
            "Korean-stake gates remain mandatory downstream."
        ),
    )
    parser.add_argument(
        "--existing",
        action="append",
        type=Path,
        default=[],
        help=(
            "Prior policy-archive discovery JSONL whose URLs should be excluded; "
            "repeatable so deeper scans emit only genuinely new leads."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    existing_urls: set[str] = set()
    existing_rows = 0
    for path in args.existing:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                url = str(row.get("url") or "").strip()
                if not url:
                    raise ValueError(f"{path}:{line_number}: missing url")
                existing_rows += 1
                existing_urls.add(url)

    jobs = []
    for archive in dict.fromkeys(args.archive):
        config = ARCHIVES[archive]
        pages = evenly_sample(list(range(1, config["max_page"] + 1)), args.sample_pages)
        jobs.extend((archive, page, config["list_url"].format(page=page)) for page in pages)

    parsed_rows = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(fetch_text, url): (archive, page, url)
            for archive, page, url in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            archive, page, url = futures[future]
            try:
                body = future.result()
                parsed_rows.extend(
                    {**row, "archive": archive}
                    for row in PARSERS[archive](body, page)
                )
            except Exception as exc:
                errors.append({"archive": archive, "page": page, "url": url, "error": f"{type(exc).__name__}: {exc}"})

    retained = []
    seen_urls = set()
    skipped_existing = 0
    for rank, row in enumerate(sorted(
        parsed_rows, key=lambda value: (value["archive"], value["page"], value["url"])
    ), 1):
        if row["url"] in seen_urls or not retain(
            row["title"], args.broad_external_policy
        ):
            continue
        seen_urls.add(row["url"])
        if row["url"] in existing_urls:
            skipped_existing += 1
            continue
        retained.append(lead(row["archive"], row, rank))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in retained),
        encoding="utf-8",
    )
    report = {
        "schema_version": "ksova-policy-archive-discovery-report-1.0",
        "archives": list(dict.fromkeys(args.archive)),
        "list_pages_requested": len(jobs),
        "list_page_errors": len(errors),
        "raw_entries": len(parsed_rows),
        "existing_files": [str(path) for path in args.existing],
        "existing_rows": existing_rows,
        "existing_unique_urls": len(existing_urls),
        "skipped_existing_urls": skipped_existing,
        "broad_external_policy": args.broad_external_policy,
        "unique_leads": len(retained),
        "by_archive": dict(sorted(Counter(
            discovery["source_variant"].removeprefix("public_policy_archive_")
            for row in retained for discovery in row["discoveries"][:1]
        ).items())),
        "by_domain": dict(sorted(Counter(row["domain_id"] for row in retained).items())),
        "error_examples": errors[:20],
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "error_examples"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
