#!/usr/bin/env python3
"""Collect topic-independent leads from authoritative issue registries.

Registry rows are discovery hints only.  ``data/expand.py resolve`` must fetch
the substantive record body, and the normal event, independent-source,
novelty, generation, and final-validation gates still apply.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import http.cookiejar
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
WTO_CASE_ARRAY_URL = (
    "https://www.wto.org/library/disputes/dispute_case_arrays_e.js"
)
WTO_CASE_URL = (
    "https://www.wto.org/english/tratop_e/dispu_e/cases_e/{case_lower}_e.htm"
)
OHCHR_KOREA_REGISTRY_URL = (
    "https://tbinternet.ohchr.org/_layouts/15/TreatyBodyExternal/"
    "Countries.aspx?CountryCode=KOR&Lang=EN"
)
OHCHR_BASE_URL = "https://tbinternet.ohchr.org"
OHCHR_SP_BASE_URL = "https://spcommreports.ohchr.org"
OHCHR_SP_SEARCH_URL = OHCHR_SP_BASE_URL + "/TmSearch/Results"
OHCHR_SP_KOREA_STATE_ID = "141"
OHCHR_SP_START_DATE = "01 Dec 2010"
ILO_NORMLEX_BASE_URL = "https://normlex.ilo.org/dyn/nrmlx_en/"
ILO_KOREA_CFA_LIST_URLS = {
    "active": (
        "f?p=1000:20060:0:FIND:NO:20060:"
        "P20060_COUNTRY_ID,P20060_COMPLAINT_STATU_ID:103123,1495810"
    ),
    "follow_up": (
        "f?p=1000:20060:0:FIND:NO:20060:"
        "P20060_COUNTRY_ID,P20060_COMPLAINT_STATU_ID:103123,1495811"
    ),
    "closed": (
        "f?p=1000:20060:0:FIND:NO:20060:"
        "P20060_COUNTRY_ID,P20060_COMPLAINT_STATU_ID:103123,1495812"
    ),
}
MOJ_ISDS_ARCHIVE_URL = "https://www.moj.go.kr/bbs/moj/182/artclList.do"
MOJ_BASE_URL = "https://www.moj.go.kr"
USTR_NTE_REPORTS = (
    (
        2025,
        "https://ustr.gov/sites/default/files/files/Press/Reports/2025NTE.pdf",
    ),
    (
        2024,
        "https://ustr.gov/sites/default/files/2024%20NTE%20Report.pdf",
    ),
    (
        2023,
        "https://ustr.gov/sites/default/files/2023-03/2023%20NTE%20Report.pdf",
    ),
)
WTO_TPR_KOREA_2021_RECORDS = (
    {
        "record_type": "secretariat_report",
        "title": "2021 WTO Trade Policy Review — Republic of Korea: Secretariat report",
        "url": "https://www.wto.org/english/tratop_e/tpr_e/s414_rev1_e.pdf",
        "publisher": "World Trade Organization Secretariat",
        "jurisdiction": "INTL",
        "source_role": "external_position",
    },
    {
        "record_type": "government_report",
        "title": "2021 WTO Trade Policy Review — report by the Republic of Korea",
        "url": "https://www.wto.org/english/tratop_e/tpr_e/g414_rev1_e.pdf",
        "publisher": "Republic of Korea via World Trade Organization",
        "jurisdiction": "KR",
        "source_role": "korean_record",
    },
    {
        "record_type": "questions_and_answers",
        "title": "2021 WTO Trade Policy Review — member questions and Korea's answers",
        "url": (
            "https://docs.wto.org/dol2fe/Pages/SS/directdoc.aspx?"
            "filename=q:/WT/TPR/M414A1.pdf&Open=True"
        ),
        "publisher": "World Trade Organization Trade Policy Review Body",
        "jurisdiction": "INTL",
        "source_role": "common_fact",
    },
    {
        "record_type": "meeting_minutes",
        "title": "2021 WTO Trade Policy Review — meeting minutes",
        "url": (
            "https://docs.wto.org/dol2fe/Pages/SS/directdoc.aspx?"
            "filename=q:/WT/TPR/M414.pdf&Open=True"
        ),
        "publisher": "World Trade Organization Trade Policy Review Body",
        "jurisdiction": "INTL",
        "source_role": "external_position",
    },
)
OHCHR_UPR_KOREA_2023_RECORDS = (
    {
        "symbol": "A/HRC/WG.6/42/KOR/1",
        "record_type": "national_report",
        "title": "Republic of Korea fourth-cycle UPR national report",
        "jurisdiction": "KR",
        "source_role": "korean_record",
    },
    {
        "symbol": "A/HRC/WG.6/42/KOR/2",
        "record_type": "un_compilation",
        "title": "Republic of Korea fourth-cycle UPR United Nations compilation",
        "jurisdiction": "INTL",
        "source_role": "external_position",
    },
    {
        "symbol": "A/HRC/WG.6/42/KOR/3",
        "record_type": "stakeholder_summary",
        "title": "Republic of Korea fourth-cycle UPR stakeholder submissions summary",
        "jurisdiction": "KR",
        "source_role": "korean_discourse",
    },
    {
        "symbol": "A/HRC/53/11",
        "record_type": "working_group_report",
        "title": "Republic of Korea fourth-cycle UPR Working Group report",
        "jurisdiction": "INTL",
        "source_role": "external_position",
    },
    {
        "symbol": "A/HRC/53/11/Add.1",
        "record_type": "government_positions",
        "title": "Republic of Korea positions on fourth-cycle UPR recommendations",
        "jurisdiction": "KR",
        "source_role": "korean_record",
    },
)
UN_CLCS_KOREA_RECORDS = (
    {
        "record_type": "korea_partial_submission",
        "title": "CLCS submission by the Republic of Korea and Japan–Korea communications",
        "url": (
            "https://www.un.org/depts/los/clcs_new/submissions_files/"
            "submission_kor_65_2012.htm"
        ),
    },
    {
        "record_type": "korea_reactions_to_japan_submission",
        "title": "CLCS submission by Japan and reactions of the Republic of Korea",
        "url": (
            "https://www.un.org/depts/los/clcs_new/submissions_files/"
            "submission_jpn.htm"
        ),
    },
)
WTO_STC_API_BASE = (
    "https://tradeconcerns.wto.org/api/specificTradeConcerns/v1/stcs"
)
WTO_KOREA_MEMBER_CODE = "C410"
WTO_STC_START_DATE = "2020-01-01"


def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    with opener.open(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str, timeout: int = 30) -> Any:
    return json.loads(fetch_bytes(url, timeout=timeout).decode("utf-8"))


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip(" -")


def parse_js_object(block: str) -> dict[str, str]:
    return dict(
        re.findall(r'^\s*([A-Za-z_][A-Za-z0-9_]*):\s*"(.*?)",?\s*$', block, re.MULTILINE)
    )


def collect_wto_disputes() -> list[dict[str, Any]]:
    # The WTO currently serves this legacy JavaScript file as Windows-1252.
    body = fetch_bytes(WTO_CASE_ARRAY_URL).decode("cp1252", errors="replace")
    blocks = re.findall(
        r"ds_array\[[0-9]+\]\s*=\s*\{(.*?)\r?\n\};",
        body,
        re.DOTALL,
    )
    rows = []
    for block in blocks:
        fields = parse_js_object(block)
        complainant = fields.get("origin", "")
        respondent = fields.get("destination", "")
        if complainant != "KOR" and respondent != "KOR":
            continue
        case_id = fields.get("code", "").upper()
        title = clean_text(fields.get("title", ""))
        if not re.fullmatch(r"DS[0-9]+", case_id) or not title:
            continue
        rows.append(
            {
                "case_id": case_id,
                "title": title,
                "url": WTO_CASE_URL.format(case_lower=case_id.lower()),
                "complainant_code": complainant,
                "respondent_code": respondent,
                "start_date": fields.get("start_date", ""),
                "status": fields.get("status", ""),
            }
        )
    return sorted(rows, key=lambda row: int(row["case_id"][2:]))


def make_wto_lead(row: dict[str, Any], rank: int) -> dict[str, Any]:
    url = row["url"]
    case_id = row["case_id"]
    role = "complainant" if row["complainant_code"] == "KOR" else "respondent"
    query = f'{case_id} Korea WTO {row["title"]}'
    return {
        "lead_id": "ksdl_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "lane": "경제·산업",
        "domain_id": "D07",
        "language": "en",
        "query": query,
        "engine": "official_issue_registry",
        "rank": rank,
        "title": f'WTO dispute {case_id}: {row["title"]}',
        "url": url,
        "host": "www.wto.org",
        "publisher": "World Trade Organization",
        "published_at": "",
        "snippet": (
            f'Republic of Korea role: {role}; consultations requested: '
            f'{row["start_date"] or "unknown"}'
        ),
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "official_registry_metadata_not_body_verified",
        "discoveries": [
            {
                "lane": "경제·산업",
                "domain_id": "D07",
                "language": "en",
                "query": query,
                "engine": "official_issue_registry",
                "rank": rank,
                "source_role": "external_position",
                "source_variant": "official_issue_registry_wto_disputes",
                "query_kind": "structured_issue_registry",
                "jurisdiction": "INTL",
                "require_korea_signal": True,
                "search_subject_ko": f"한국 WTO 분쟁 {case_id}",
                "search_subject_en": query,
                "title_matched_terms": [case_id, "Korea", "WTO"],
            }
        ],
    }


def parse_date(value: str) -> datetime:
    for pattern in ("%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), pattern)
        except ValueError:
            continue
    return datetime.min


def treaty_body_for_symbol(symbol: str) -> str:
    upper = symbol.upper()
    if upper.startswith("E/C.12/"):
        return "CESCR"
    return upper.split("/", 1)[0]


def resolve_ohchr_english_body_url(landing_url: str) -> str | None:
    body = fetch_bytes(landing_url, timeout=45).decode("utf-8", errors="replace")
    links: dict[str, str] = {}
    for tag in re.findall(r"<a\b[^>]*>", body, re.IGNORECASE):
        attributes = dict(
            (name.casefold(), html.unescape(value))
            for name, _, value in re.findall(
                r'''([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(["'])(.*?)\2''',
                tag,
                re.DOTALL,
            )
        )
        title = attributes.get("title", "").casefold()
        href = attributes.get("href", "")
        if title and href:
            links[title] = urllib.parse.urljoin(landing_url, href)
    for title in ("english pdf", "english html", "english docx", "english doc"):
        if title in links:
            return links[title]
    return None


def collect_ohchr_korea() -> list[dict[str, Any]]:
    """Collect Korea jurisprudence and the latest observations per treaty.

    Historical concluding observations largely repeat the same review lanes.
    Keeping only the latest record for each treaty avoids treating review
    cycles as distinct issues, while jurisprudence rows represent separate
    communications and are therefore retained individually.
    """

    body = fetch_bytes(OHCHR_KOREA_REGISTRY_URL, timeout=90).decode(
        "utf-8", errors="replace"
    )
    rows: list[dict[str, Any]] = []
    for block in re.findall(r"<tr\b[^>]*>(.*?)</tr>", body, re.DOTALL | re.IGNORECASE):
        cells = [
            clean_text(cell)
            for cell in re.findall(
                r"<td\b[^>]*>(.*?)</td>", block, re.DOTALL | re.IGNORECASE
            )
        ]
        if len(cells) < 2 or cells[0] not in {"Jurisprudence", "Concluding observations"}:
            continue
        link_match = re.search(
            r'href=["\']([^"\']*Download\.aspx\?[^"\']+)["\']',
            block,
            re.IGNORECASE,
        )
        if not link_match:
            continue
        symbol = cells[1].strip()
        if not symbol:
            continue
        record_type = (
            "jurisprudence" if cells[0] == "Jurisprudence" else "concluding_observations"
        )
        # Publication dates occur in different columns across the two grids.
        dates = [value for value in cells[2:5] if parse_date(value) != datetime.min]
        published_at = max(dates, key=parse_date) if dates else ""
        url = urllib.parse.urljoin(
            OHCHR_BASE_URL,
            html.unescape(link_match.group(1)),
        )
        treaty_body = treaty_body_for_symbol(symbol)
        rows.append(
            {
                "record_type": record_type,
                "symbol": symbol,
                "treaty_body": treaty_body,
                "published_at": published_at,
                "url": url,
            }
        )

    # Registry markup can repeat observation rows in nested grids.  URLs are
    # stable official identifiers, so deduplicate before selecting cycles.
    deduplicated = {row["url"]: row for row in rows}
    jurisprudence = [
        row for row in deduplicated.values() if row["record_type"] == "jurisprudence"
    ]
    latest_observations: dict[str, dict[str, Any]] = {}
    for row in deduplicated.values():
        if row["record_type"] != "concluding_observations":
            continue
        # Modern country-specific records can be assigned unambiguously to a
        # treaty.  Generic General Assembly supplements are intentionally
        # excluded because a single PDF can contain many countries.
        if "/KOR/CO/" not in row["symbol"].upper():
            continue
        previous = latest_observations.get(row["treaty_body"])
        if previous is None or parse_date(row["published_at"]) > parse_date(
            previous["published_at"]
        ):
            latest_observations[row["treaty_body"]] = row

    selected = sorted(
        [*jurisprudence, *latest_observations.values()],
        key=lambda row: (
            row["record_type"],
            row["treaty_body"],
            row["symbol"],
        ),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        body_urls = list(pool.map(
            resolve_ohchr_english_body_url,
            (row["url"] for row in selected),
        ))
    resolved = []
    for row, body_url in zip(selected, body_urls):
        if not body_url:
            continue
        resolved.append({**row, "registry_record_url": row["url"], "url": body_url})
    return resolved


def make_ohchr_lead(row: dict[str, Any], rank: int) -> dict[str, Any]:
    url = row["url"]
    symbol = row["symbol"]
    record_label = (
        "individual communication decision"
        if row["record_type"] == "jurisprudence"
        else "latest concluding observations"
    )
    query = f'"{symbol}" Republic of Korea UN {record_label}'
    return {
        "lead_id": "ksdl_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "lane": "민주주의·기본권",
        "domain_id": "D02",
        "language": "en",
        "query": query,
        "engine": "official_issue_registry",
        "rank": rank,
        "title": f'UN treaty-body {record_label}: {symbol}',
        "url": url,
        "host": "tbinternet.ohchr.org",
        "publisher": "United Nations human rights treaty bodies",
        "published_at": row["published_at"],
        "snippet": (
            f'Official Republic of Korea record; treaty body: '
            f'{row["treaty_body"]}; record type: {row["record_type"]}'
        ),
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "official_registry_metadata_not_body_verified",
        "discoveries": [
            {
                "lane": "민주주의·기본권",
                "domain_id": "D02",
                "language": "en",
                "query": query,
                "engine": "official_issue_registry",
                "rank": rank,
                "source_role": "external_position",
                "source_variant": "official_issue_registry_ohchr_korea",
                "query_kind": "structured_issue_registry",
                "jurisdiction": "INTL",
                "require_korea_signal": True,
                "search_subject_ko": f"유엔 인권조약기구 한국 {symbol}",
                "search_subject_en": query,
                "title_matched_terms": [symbol, "Republic of Korea", "UN"],
                "registry_record_url": row["registry_record_url"],
            }
        ],
    }


def infer_ohchr_sp_domain(mandates: list[str], summary: str) -> tuple[str, str]:
    """Route registry records without treating the route as an issue verdict."""

    combined = " ".join([*mandates, summary]).casefold()
    if any(term in combined for term in (
        "telecommunication", "privacy", "personal data", "surveillance",
        "artificial intelligence", "digital",
    )):
        return "D08", "기술·데이터"
    if any(term in combined for term in (
        "comfort women", "japanese imperial", "colonial", "wartime",
    )):
        return "D04", "역사·책임"
    if "business" in {mandate.casefold() for mandate in mandates} and any(
        term in combined for term in (
            "company", "corporat", "investor", "project", "operation",
            "samsung", "enterprise",
        )
    ):
        return "D07", "경제·산업"
    return "D02", "민주주의·기본권"


def collect_ohchr_special_procedures_korea() -> list[dict[str, Any]]:
    """Collect paired Special Procedures letters and Korean replies.

    The search result is session-backed: the first POST establishes the
    result set and the subsequent page requests must reuse its cookie jar.
    A communication and a government reply are retained as separate source
    bodies but linked by the stable KOR communication code.  They remain
    discovery leads until the ordinary fetched-body and issue gates pass.
    """

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookie_jar)
    )
    form = urllib.parse.urlencode({
        "SessionID": "0",
        "TypeOfGovReply": "0",
        "Countries": OHCHR_SP_KOREA_STATE_ID,
        "LetterDateStart": OHCHR_SP_START_DATE,
        "LetterDateEnd": "31 Dec 2099",
        "Keyword": "",
        "command": "Search",
    }).encode("utf-8")
    request = urllib.request.Request(
        OHCHR_SP_SEARCH_URL,
        data=form,
        headers={"User-Agent": USER_AGENT},
    )
    with opener.open(request, timeout=90) as response:
        first_page = response.read().decode("utf-8", errors="replace")

    page_numbers = [
        int(value)
        for value in re.findall(r"/TmSearch/Results\?page=([0-9]+)", first_page)
    ]
    page_count = max(page_numbers, default=1)
    pages = [first_page]
    for page in range(2, page_count + 1):
        page_request = urllib.request.Request(
            f"{OHCHR_SP_SEARCH_URL}?page={page}",
            headers={"User-Agent": USER_AGENT},
        )
        with opener.open(page_request, timeout=90) as response:
            pages.append(response.read().decode("utf-8", errors="replace"))

    records_by_url: dict[str, dict[str, Any]] = {}
    for body in pages:
        for block in re.findall(
            r"<tr\b[^>]*>(.*?)</tr>", body, re.DOTALL | re.IGNORECASE
        ):
            communication_match = re.search(
                r'href=["\']([^"\']*DownLoadPublicCommunicationFile\?gId=([0-9]+))'
                r'["\'][^>]*>(.*?)</a>',
                block,
                re.DOTALL | re.IGNORECASE,
            )
            if not communication_match:
                continue
            relative_url, communication_gid, code_html = communication_match.groups()
            communication_code = clean_text(code_html)
            if not re.fullmatch(r"KOR\s+[0-9]+/[0-9]{4}", communication_code):
                continue
            date_match = re.search(
                r"\b([0-9]{2}\s+[A-Z][a-z]{2}\s+[0-9]{4})\b", block
            )
            sent_at = date_match.group(1) if date_match else ""
            mandates = [
                clean_text(value)
                for value in re.findall(
                    r"<li\b[^>]*>(.*?)</li>", block, re.DOTALL | re.IGNORECASE
                )
                if clean_text(value)
            ]
            summary_match = re.search(
                r'<td\b[^>]*style=["\'][^"\']*word-wrap[^"\']*["\'][^>]*>'
                r"(.*?)(?:<a\b|</td>)",
                block,
                re.DOTALL | re.IGNORECASE,
            )
            summary = clean_text(summary_match.group(1)) if summary_match else ""
            domain_id, lane = infer_ohchr_sp_domain(mandates, summary)
            communication_url = urllib.parse.urljoin(
                OHCHR_SP_BASE_URL, html.unescape(relative_url)
            )
            detail_url = (
                OHCHR_SP_BASE_URL + "/TmSearch/RelCom?" +
                urllib.parse.urlencode({"code": communication_code})
            )
            base = {
                "communication_code": communication_code,
                "communication_gid": communication_gid,
                "sent_at": sent_at,
                "mandates": mandates,
                "summary": summary,
                "domain_id": domain_id,
                "lane": lane,
                "registry_record_url": detail_url,
            }
            records_by_url[communication_url] = {
                **base,
                "record_type": "communication",
                "url": communication_url,
                "published_at": sent_at,
                "jurisdiction": "INTL",
                "source_role": "external_position",
            }

            for reply_relative, reply_gid, reply_label in re.findall(
                r'href=["\']([^"\']*DownLoadFile\?gId=([0-9]+))["\']'
                r'[^>]*>(.*?)</a>',
                block,
                re.DOTALL | re.IGNORECASE,
            ):
                reply_url = urllib.parse.urljoin(
                    OHCHR_SP_BASE_URL, html.unescape(reply_relative)
                )
                records_by_url[reply_url] = {
                    **base,
                    "record_type": "government_reply",
                    "reply_gid": reply_gid,
                    "url": reply_url,
                    "published_at": clean_text(reply_label),
                    "jurisdiction": "KR",
                    "source_role": "korean_record",
                    "corresponding_communication_url": communication_url,
                }

    return sorted(
        records_by_url.values(),
        key=lambda row: (
            parse_date(str(row.get("sent_at") or "")),
            str(row["communication_code"]),
            row["record_type"] == "communication",
        ),
        reverse=True,
    )


def make_ohchr_special_procedures_lead(
    row: dict[str, Any], rank: int
) -> dict[str, Any]:
    url = str(row["url"])
    communication_code = str(row["communication_code"])
    record_type = str(row["record_type"])
    mandates = ", ".join(row.get("mandates") or [])
    summary = str(row.get("summary") or "")
    subject = clean_text(
        f'"{communication_code}" Republic of Korea OHCHR Special Procedures '
        f'{mandates} {summary[:500]}'
    )
    title_suffix = (
        "special-procedures communication"
        if record_type == "communication"
        else "Republic of Korea government reply"
    )
    return {
        "lead_id": "ksdl_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "lane": row["lane"],
        "domain_id": row["domain_id"],
        "language": "en",
        "query": subject,
        "engine": "official_issue_registry",
        "rank": rank,
        "title": f"OHCHR {communication_code}: {title_suffix}",
        "url": url,
        "host": "spcommreports.ohchr.org",
        "publisher": (
            "United Nations human rights special procedures"
            if record_type == "communication"
            else "Republic of Korea via United Nations special procedures"
        ),
        "published_at": row.get("published_at") or "",
        "snippet": clean_text(
            f"Official paired record; type: {record_type}; mandates: {mandates}; "
            f"summary: {summary[:700]}"
        ),
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "official_special_procedures_body_not_yet_verified",
        "discoveries": [
            {
                "lane": row["lane"],
                "domain_id": row["domain_id"],
                "language": "en",
                "query": subject,
                "engine": "official_issue_registry",
                "rank": rank,
                "source_role": row["source_role"],
                "source_variant": (
                    "official_issue_registry_ohchr_special_procedures_"
                    + record_type
                ),
                "query_kind": "structured_special_procedures_communication",
                "jurisdiction": row["jurisdiction"],
                "require_korea_signal": True,
                "search_subject_ko": (
                    f"유엔 특별절차 한국 {communication_code} "
                    + ("정부 답변" if record_type == "government_reply" else "서한")
                ),
                "search_subject_en": subject,
                "title_matched_terms": [
                    communication_code, "Republic of Korea", "OHCHR"
                ],
                "registry_record_url": row["registry_record_url"],
                "communication_code": communication_code,
                "record_type": record_type,
                "corresponding_communication_url": row.get(
                    "corresponding_communication_url", ""
                ),
            }
        ],
    }


def collect_ilo_korea_cfa() -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for status, relative_url in ILO_KOREA_CFA_LIST_URLS.items():
        list_url = urllib.parse.urljoin(ILO_NORMLEX_BASE_URL, relative_url)
        body = fetch_bytes(list_url, timeout=45).decode("utf-8", errors="replace")
        for tag, text in re.findall(
            r"(<a\b[^>]*href=[\"'][^\"']+[\"'][^>]*>)(.*?)</a>",
            body,
            re.DOTALL | re.IGNORECASE,
        ):
            label = clean_text(text)
            match = re.fullmatch(
                r"Case No\s+([0-9]+)\s+\(Republic of Korea\)\s+-\s+"
                r"Complaint date:\s+([0-9]{2}-[A-Z]{3}-[0-9]{2})",
                label,
                re.IGNORECASE,
            )
            href_match = re.search(r'href=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            if not match or not href_match:
                continue
            case_number, complaint_date = match.groups()
            complaint_date = datetime.strptime(
                complaint_date.upper(), "%d-%b-%y"
            ).date().isoformat()
            rows[case_number] = {
                "case_number": case_number,
                "complaint_date": complaint_date,
                "status": status,
                "url": urllib.parse.urljoin(
                    ILO_NORMLEX_BASE_URL,
                    html.unescape(href_match.group(1)),
                ),
            }
    return sorted(rows.values(), key=lambda row: int(row["case_number"]), reverse=True)


def make_ilo_lead(row: dict[str, Any], rank: int) -> dict[str, Any]:
    url = row["url"]
    case_number = row["case_number"]
    query = f'ILO freedom of association Case {case_number} Republic of Korea'
    return {
        "lead_id": "ksdl_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "lane": "민주주의·기본권",
        "domain_id": "D02",
        "language": "en",
        "query": query,
        "engine": "official_issue_registry",
        "rank": rank,
        "title": f'ILO freedom-of-association Case No. {case_number} (Republic of Korea)',
        "url": url,
        "host": "normlex.ilo.org",
        "publisher": "International Labour Organization",
        "published_at": row["complaint_date"],
        "snippet": (
            f'Official case registry; status: {row["status"]}; '
            f'complaint date: {row["complaint_date"]}'
        ),
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "official_registry_metadata_not_body_verified",
        "discoveries": [
            {
                "lane": "민주주의·기본권",
                "domain_id": "D02",
                "language": "en",
                "query": query,
                "engine": "official_issue_registry",
                "rank": rank,
                "source_role": "external_position",
                "source_variant": "official_issue_registry_ilo_korea_cfa",
                "query_kind": "structured_issue_registry",
                "jurisdiction": "INTL",
                "require_korea_signal": True,
                "search_subject_ko": f"ILO 결사의 자유 한국 사건 {case_number}",
                "search_subject_en": query,
                "title_matched_terms": [case_number, "Republic of Korea", "ILO"],
            }
        ],
    }


def collect_moj_isds_archive() -> list[dict[str, Any]]:
    archive_queries = ("국제투자분쟁", "ISDS")
    rows: dict[str, dict[str, Any]] = {}
    archive_pages: dict[str, int] = {}
    for query in archive_queries:
        first_url = MOJ_ISDS_ARCHIVE_URL + "?" + urllib.parse.urlencode(
            {"page": 1, "srchColumn": "sj", "srchWrd": query}
        )
        first_body = fetch_bytes(first_url, timeout=45).decode(
            "utf-8", errors="replace"
        )
        page_matches = re.findall(
            r"javascript:page_link\(['\"]([0-9]+)['\"]\)",
            first_body,
            re.IGNORECASE,
        )
        total_pages = max((int(value) for value in page_matches), default=1)
        archive_pages[query] = total_pages
        page_bodies = [first_body]
        for page in range(2, total_pages + 1):
            url = MOJ_ISDS_ARCHIVE_URL + "?" + urllib.parse.urlencode(
                {"page": page, "srchColumn": "sj", "srchWrd": query}
            )
            page_bodies.append(
                fetch_bytes(url, timeout=45).decode("utf-8", errors="replace")
            )

        for body in page_bodies:
            for block in re.findall(
                r"<tr\b[^>]*>(.*?)</tr>", body, re.DOTALL | re.IGNORECASE
            ):
                link_match = re.search(
                    r'href=["\'](/bbs/moj/182/([0-9]+)/artclView\.do)["\']',
                    block,
                    re.IGNORECASE,
                )
                if not link_match:
                    continue
                title_match = re.search(
                    r"<strong\b[^>]*>(.*?)</strong>",
                    block,
                    re.DOTALL | re.IGNORECASE,
                )
                date_match = re.search(
                    r'class=["\'][^"\']*_artclTdRdate[^"\']*["\'][^>]*>\s*'
                    r"([0-9]{4}\.[0-9]{2}\.[0-9]{2})",
                    block,
                    re.DOTALL | re.IGNORECASE,
                )
                title = clean_text(title_match.group(1)) if title_match else ""
                if not title:
                    continue
                article_id = link_match.group(2)
                previous = rows.get(article_id, {})
                matched_queries = sorted({
                    *previous.get("archive_queries", []),
                    query,
                })
                rows[article_id] = {
                    "article_id": article_id,
                    "title": title,
                    "published_at": (
                        date_match.group(1).replace(".", "-") if date_match else ""
                    ),
                    "url": urllib.parse.urljoin(MOJ_BASE_URL, link_match.group(1)),
                    "archive_queries": matched_queries,
                    "archive_pages": archive_pages,
                }
    selected = sorted(
        rows.values(), key=lambda row: int(row["article_id"]), reverse=True
    )

    def resolve_article_body(row: dict[str, Any]) -> dict[str, Any]:
        body = fetch_bytes(row["url"], timeout=45).decode("utf-8", errors="replace")
        attachments = []
        for tag, label_html in re.findall(
            r"(<a\b[^>]*href=[\"'][^\"']+[\"'][^>]*>)(.*?)</a>",
            body,
            re.DOTALL | re.IGNORECASE,
        ):
            href_match = re.search(r'href=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            if not href_match:
                continue
            label = clean_text(label_html)
            if not label.casefold().endswith(".pdf"):
                continue
            attachments.append(
                urllib.parse.urljoin(MOJ_BASE_URL, html.unescape(href_match.group(1)))
            )
        if not attachments:
            return row
        return {
            **row,
            "registry_record_url": row["url"],
            "url": attachments[0],
            "body_format": "pdf_attachment",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(resolve_article_body, selected))


def make_moj_isds_lead(row: dict[str, Any], rank: int) -> dict[str, Any]:
    url = row["url"]
    title = row["title"]
    query = f'법무부 국제투자분쟁 {title}'
    return {
        "lead_id": "ksdl_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "lane": "경제·산업",
        "domain_id": "D07",
        "language": "ko",
        "query": query,
        "engine": "official_issue_registry",
        "rank": rank,
        "title": title,
        "url": url,
        "host": "www.moj.go.kr",
        "publisher": "대한민국 법무부",
        "published_at": row["published_at"],
        "snippet": (
            f'법무부 보도자료 제목 일반검색 {row["archive_queries"]} 결과; '
            f'article id: {row["article_id"]}'
        ),
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "official_archive_metadata_not_body_verified",
        "discoveries": [
            {
                "lane": "경제·산업",
                "domain_id": "D07",
                "language": "ko",
                "query": query,
                "engine": "official_issue_registry",
                "rank": rank,
                "source_role": "korean_record",
                "source_variant": "official_issue_registry_moj_isds_archive",
                "query_kind": "structured_issue_registry",
                "jurisdiction": "KR",
                "require_korea_signal": True,
                "search_subject_ko": title,
                "search_subject_en": "Republic of Korea ISDS " + title,
                "title_matched_terms": ["국제투자분쟁", "ISDS", title],
                "registry_record_url": row.get("registry_record_url", url),
            }
        ],
    }


def collect_ustr_nte_korea() -> list[dict[str, Any]]:
    rows = []
    for year, url in USTR_NTE_REPORTS:
        # Reject stale URL guesses at collection time.  The fetch stage still
        # performs its own body, type, and relevance checks.
        body = fetch_bytes(url, timeout=90)
        if not body.startswith(b"%PDF"):
            continue
        rows.append({"year": year, "url": url})
    return rows


def make_ustr_nte_lead(row: dict[str, Any], rank: int) -> dict[str, Any]:
    year = row["year"]
    url = row["url"]
    # These are the report's standard barrier categories rather than a
    # hand-written list of presumed Korean controversies.  They guide the
    # long-document excerpt selector toward the Korea chapter and let body
    # extraction discover the concrete measures.
    subject = (
        f"{year} National Trade Estimate KOREA trade agreements technical "
        "barriers sanitary phytosanitary government procurement intellectual "
        "property services investment data localization competition policy"
    )
    return {
        "lead_id": "ksdl_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "lane": "경제·산업",
        "domain_id": "D07",
        "language": "en",
        "query": subject,
        "engine": "official_issue_registry",
        "rank": rank,
        "title": f"{year} National Trade Estimate Report — Korea chapter",
        "url": url,
        "host": "ustr.gov",
        "publisher": "Office of the United States Trade Representative",
        "published_at": "",
        "snippet": (
            "Official annual country-by-country registry of foreign trade "
            "barriers; Korea chapter selected for body extraction"
        ),
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "official_annual_report_body_not_yet_verified",
        "discoveries": [
            {
                "lane": "경제·산업",
                "domain_id": "D07",
                "language": "en",
                "query": subject,
                "engine": "official_issue_registry",
                "rank": rank,
                "source_role": "external_position",
                "source_variant": "official_issue_registry_ustr_nte_korea",
                "query_kind": "structured_country_report",
                "jurisdiction": "US",
                "require_korea_signal": True,
                "search_subject_ko": f"{year} 미국 무역장벽보고서 한국 장",
                "search_subject_en": subject,
                "title_matched_terms": ["KOREA", "National Trade Estimate", str(year)],
                "body_section_start": (
                    "KOREA TRADE AGREEMENTS United States–Korea Free Trade Agreement"
                ),
                "body_section_end_regex": (
                    "FOREIGN TRADE BARRIERS [A-Z][A-Z ]{2,40} TRADE AGREEMENTS"
                ),
            }
        ],
    }


def collect_wto_tpr_korea() -> list[dict[str, Any]]:
    """Verify the four complementary records from Korea's latest full TPR.

    The set is registry-driven rather than keyword-driven: an independent
    Secretariat report, Korea's own report, member questions with Korea's
    answers, and the meeting minutes.  Concrete issues are still extracted
    from fetched bodies and require independent follow-up provenance.
    """

    rows = []
    for record in WTO_TPR_KOREA_2021_RECORDS:
        body = fetch_bytes(str(record["url"]), timeout=120)
        if record["url"].endswith(".pdf") or "directdoc.aspx" in record["url"]:
            if not body.startswith(b"%PDF"):
                continue
        elif b"Trade Policy Review" not in body:
            continue
        rows.append(dict(record))
    return rows


def make_wto_tpr_lead(row: dict[str, Any], rank: int) -> dict[str, Any]:
    url = str(row["url"])
    subject = (
        "WT/TPR/414 Republic of Korea tariff agriculture trade remedies "
        "subsidies government procurement services investment competition "
        "shipbuilding fisheries standards SPS intellectual property digital "
        "economy data cloud energy member questions answers"
    )
    return {
        "lead_id": "ksdl_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "lane": "경제·산업",
        "domain_id": "D07",
        "language": "en",
        "query": subject,
        "engine": "official_issue_registry",
        "rank": rank,
        "title": row["title"],
        "url": url,
        "host": urllib.parse.urlparse(url).netloc,
        "publisher": row["publisher"],
        "published_at": "2021-12-21",
        "snippet": (
            "Latest complete WTO Trade Policy Review registry for the "
            f"Republic of Korea; record type: {row['record_type']}"
        ),
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "official_tpr_record_body_not_yet_verified",
        "discoveries": [
            {
                "lane": "경제·산업",
                "domain_id": "D07",
                "language": "en",
                "query": subject,
                "engine": "official_issue_registry",
                "rank": rank,
                "source_role": row["source_role"],
                "source_variant": (
                    "official_issue_registry_wto_tpr_korea_"
                    + str(row["record_type"])
                ),
                "query_kind": "structured_country_review",
                "jurisdiction": row["jurisdiction"],
                "require_korea_signal": True,
                "search_subject_ko": (
                    "2021 WTO 한국 무역정책검토 " + str(row["record_type"])
                ),
                "search_subject_en": subject,
                "title_matched_terms": [
                    "Republic of Korea", "Trade Policy Review", "WT/TPR/414"
                ],
            }
        ],
    }


def un_symbol_pdf_url(symbol: str) -> str:
    return "https://documents.un.org/api/symbol/access?" + urllib.parse.urlencode(
        {"s": symbol, "l": "en", "t": "pdf"}
    )


def collect_ohchr_upr_korea() -> list[dict[str, Any]]:
    rows = []
    for record in OHCHR_UPR_KOREA_2023_RECORDS:
        url = un_symbol_pdf_url(str(record["symbol"]))
        body = fetch_bytes(url, timeout=120)
        if not body.startswith(b"%PDF"):
            continue
        rows.append({**record, "url": url})
    return rows


def make_ohchr_upr_lead(row: dict[str, Any], rank: int) -> dict[str, Any]:
    symbol = str(row["symbol"])
    url = str(row["url"])
    subject = (
        f'"{symbol}" Republic of Korea Universal Periodic Review '
        "recommendation supported noted government position"
    )
    return {
        "lead_id": "ksdl_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "lane": "민주주의·기본권",
        "domain_id": "D02",
        "language": "en",
        "query": subject,
        "engine": "official_issue_registry",
        "rank": rank,
        "title": row["title"],
        "url": url,
        "host": "documents.un.org",
        "publisher": "United Nations Human Rights Council",
        "published_at": "2023",
        "snippet": (
            "Official fourth-cycle Universal Periodic Review record for the "
            f"Republic of Korea; record type: {row['record_type']}"
        ),
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "official_upr_record_body_not_yet_verified",
        "discoveries": [
            {
                "lane": "민주주의·기본권",
                "domain_id": "D02",
                "language": "en",
                "query": subject,
                "engine": "official_issue_registry",
                "rank": rank,
                "source_role": row["source_role"],
                "source_variant": (
                    "official_issue_registry_ohchr_upr_korea_"
                    + str(row["record_type"])
                ),
                "query_kind": "structured_country_review",
                "jurisdiction": row["jurisdiction"],
                "require_korea_signal": True,
                "search_subject_ko": f"유엔 UPR 한국 {symbol}",
                "search_subject_en": subject,
                "title_matched_terms": [
                    symbol, "Republic of Korea", "Universal Periodic Review"
                ],
            }
        ],
    }


def collect_un_clcs_korea() -> list[dict[str, Any]]:
    rows = []
    for record in UN_CLCS_KOREA_RECORDS:
        body = fetch_bytes(str(record["url"]), timeout=90)
        if b"Commission on the Limits of the Continental Shelf" not in body:
            continue
        if b"Republic of Korea" not in body:
            continue
        rows.append(dict(record))
    return rows


def make_un_clcs_lead(row: dict[str, Any], rank: int) -> dict[str, Any]:
    url = str(row["url"])
    subject = (
        "Republic of Korea Japan Commission on the Limits of the Continental "
        "Shelf East China Sea submission communication UNCLOS Article 76"
    )
    return {
        "lead_id": "ksdl_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "lane": "영토·관할",
        "domain_id": "D03",
        "language": "en",
        "query": subject,
        "engine": "official_issue_registry",
        "rank": rank,
        "title": row["title"],
        "url": url,
        "host": "www.un.org",
        "publisher": "United Nations Division for Ocean Affairs and the Law of the Sea",
        "published_at": "",
        "snippet": (
            "Official CLCS submission registry and inter-State communications; "
            f"record type: {row['record_type']}"
        ),
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "official_clcs_registry_body_not_yet_verified",
        "discoveries": [
            {
                "lane": "영토·관할",
                "domain_id": "D03",
                "language": "en",
                "query": subject,
                "engine": "official_issue_registry",
                "rank": rank,
                "source_role": "external_position",
                "source_variant": (
                    "official_issue_registry_un_clcs_" + str(row["record_type"])
                ),
                "query_kind": "structured_maritime_claim_registry",
                "jurisdiction": "INTL",
                "require_korea_signal": True,
                "search_subject_ko": "한일 대륙붕 한계위원회 국가통신문",
                "search_subject_en": subject,
                "title_matched_terms": [
                    "Republic of Korea", "Japan", "continental shelf"
                ],
            }
        ],
    }


def collect_wto_stc_korea() -> list[dict[str, Any]]:
    """Collect recent WTO committee trade concerns in which Korea is a party.

    Search each formal role independently, then deduplicate on the WTO area's
    IMS identifier.  A concern can recur across meetings, but those repetitions
    are evidence about one regulatory issue rather than separate issue leads.
    The detail endpoint contains the attributed statements of the member(s)
    raising the concern and the responding member, so it is fetched later as
    the source body instead of treating list metadata as evidence.
    """

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    role_parameters = {
        "responding": "searchParameterSubjectMembers",
        "raising": "searchParameterRaisingMembers",
        "supporting": "searchParameterSupportingMembers",
    }
    for korea_role, parameter in role_parameters.items():
        query = urllib.parse.urlencode({
            "language": "English",
            "page": 1,
            "pageSize": 500,
            parameter: WTO_KOREA_MEMBER_CODE,
        })
        payload = fetch_json(f"{WTO_STC_API_BASE}/search?{query}", timeout=90)
        for item in payload.get("items") or []:
            last_raised = str(item.get("lastTimeRaised") or "")
            if last_raised[:10] < WTO_STC_START_DATE:
                continue
            area = str(item.get("domainId") or "").strip().upper()
            ims_id = item.get("imsId")
            title = clean_text(str(item.get("title") or ""))
            if not area or not isinstance(ims_id, int) or not title:
                continue
            key = (area, ims_id)
            row = by_key.setdefault(key, {
                "area": area,
                "ims_id": ims_id,
                "title": title,
                "subject_name": str(item.get("subjectName") or ""),
                "raising_name": str(item.get("raisingName") or ""),
                "supporting_name": str(item.get("supportingName") or ""),
                "first_time_raised": str(item.get("firstTimeRaised") or ""),
                "last_time_raised": last_raised,
                "number_of_times_raised": item.get("numberOfTimesRaised"),
                "symbols": item.get("symbols") or [],
                "korea_roles": [],
            })
            row["korea_roles"] = sorted({*row["korea_roles"], korea_role})
            if last_raised > row["last_time_raised"]:
                row["last_time_raised"] = last_raised

    rows = []
    for row in by_key.values():
        detail_query = urllib.parse.urlencode({
            "imsId": row["ims_id"],
            "domainId": row["area"],
            "language": "English",
        })
        rows.append({
            **row,
            "url": f"{WTO_STC_API_BASE}/getDetails?{detail_query}",
        })
    return sorted(
        rows,
        key=lambda row: (row["last_time_raised"], row["area"], row["ims_id"]),
        reverse=True,
    )


def make_wto_stc_lead(row: dict[str, Any], rank: int) -> dict[str, Any]:
    url = str(row["url"])
    area = str(row["area"])
    ims_id = int(row["ims_id"])
    actors = "; ".join(filter(None, [
        str(row.get("raising_name") or ""),
        str(row.get("supporting_name") or ""),
        str(row.get("subject_name") or ""),
    ]))
    subject = f'WTO {area} specific trade concern {ims_id} Korea {row["title"]}'
    return {
        "lead_id": "ksdl_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        "lane": "경제·산업",
        "domain_id": "D07",
        "language": "en",
        "query": subject,
        "engine": "official_issue_registry",
        "rank": rank,
        "title": str(row["title"]),
        "url": url,
        "host": "tradeconcerns.wto.org",
        "publisher": "World Trade Organization",
        "published_at": str(row["last_time_raised"])[:10],
        "snippet": (
            f"Official WTO {area} specific trade concern; Korea role(s): "
            f"{', '.join(row['korea_roles'])}; parties: {actors}; repeated in "
            f"{row.get('number_of_times_raised') or 1} committee meeting(s)"
        ),
        "evidence_role": "discovery_only",
        "reliability_tier": "C",
        "allowed_uses": ["issue_discovery", "query_expansion"],
        "reliability_reason": "official_stc_registry_body_not_yet_verified",
        "discoveries": [
            {
                "lane": "경제·산업",
                "domain_id": "D07",
                "language": "en",
                "query": subject,
                "engine": "official_issue_registry",
                "rank": rank,
                "source_role": "common_fact",
                "source_variant": "official_issue_registry_wto_stc_korea",
                "query_kind": "structured_specific_trade_concern_registry",
                "jurisdiction": "INTL",
                "require_korea_signal": True,
                "search_subject_ko": f"WTO {area} 한국 구체적 무역현안 {ims_id}",
                "search_subject_en": subject,
                "title_matched_terms": [
                    "Korea", area, str(ims_id), str(row["title"])
                ],
                "wto_area": area,
                "wto_ims_id": ims_id,
                "korea_roles": row["korea_roles"],
            }
        ],
    }


def read_existing_urls(paths: list[Path]) -> tuple[set[str], int]:
    urls: set[str] = set()
    rows = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                url = str(row.get("url") or "").strip()
                if not url:
                    raise ValueError(f"{path}:{line_number}: missing url")
                rows += 1
                urls.add(url)
    return urls, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        choices=[
            "wto_disputes", "ohchr_korea", "ilo_korea_cfa", "moj_isds_archive",
            "ustr_nte_korea", "wto_tpr_korea", "ohchr_upr_korea",
            "un_clcs_korea", "wto_stc_korea",
            "ohchr_special_procedures_korea",
        ],
        required=True,
    )
    parser.add_argument("--existing", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    existing_urls, existing_rows = read_existing_urls(args.existing)
    if args.registry == "wto_disputes":
        registry_rows = collect_wto_disputes()
        make_lead = make_wto_lead
        registry_url = WTO_CASE_ARRAY_URL
    elif args.registry == "ohchr_korea":
        registry_rows = collect_ohchr_korea()
        make_lead = make_ohchr_lead
        registry_url = OHCHR_KOREA_REGISTRY_URL
    elif args.registry == "ilo_korea_cfa":
        registry_rows = collect_ilo_korea_cfa()
        make_lead = make_ilo_lead
        registry_url = ILO_NORMLEX_BASE_URL + ILO_KOREA_CFA_LIST_URLS["active"]
    elif args.registry == "moj_isds_archive":
        registry_rows = collect_moj_isds_archive()
        make_lead = make_moj_isds_lead
        registry_url = MOJ_ISDS_ARCHIVE_URL
    elif args.registry == "ustr_nte_korea":
        registry_rows = collect_ustr_nte_korea()
        make_lead = make_ustr_nte_lead
        registry_url = "https://ustr.gov/issue-areas/enforcement/foreign-trade-barriers"
    elif args.registry == "wto_tpr_korea":
        registry_rows = collect_wto_tpr_korea()
        make_lead = make_wto_tpr_lead
        registry_url = "https://www.wto.org/english/tratop_e/tpr_e/tp514_e.htm"
    elif args.registry == "ohchr_upr_korea":
        registry_rows = collect_ohchr_upr_korea()
        make_lead = make_ohchr_upr_lead
        registry_url = "https://www.ohchr.org/en/hr-bodies/upr/kr-index"
    elif args.registry == "un_clcs_korea":
        registry_rows = collect_un_clcs_korea()
        make_lead = make_un_clcs_lead
        registry_url = (
            "https://www.un.org/depts/los/clcs_new/commission_submissions.htm"
        )
    elif args.registry == "wto_stc_korea":
        registry_rows = collect_wto_stc_korea()
        make_lead = make_wto_stc_lead
        registry_url = "https://tradeconcerns.wto.org/en/stcs"
    else:
        registry_rows = collect_ohchr_special_procedures_korea()
        make_lead = make_ohchr_special_procedures_lead
        registry_url = OHCHR_SP_BASE_URL + "/Tmsearch/TMDocuments"
    leads = [
        make_lead(row, rank)
        for rank, row in enumerate(registry_rows, 1)
        if row["url"] not in existing_urls
        and row.get("registry_record_url") not in existing_urls
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in leads
        ),
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "schema_version": "ksova-issue-registry-discovery-report-1.0",
        "registry": args.registry,
        "registry_url": registry_url,
        "registry_rows": len(registry_rows),
        "existing_files": [str(path) for path in args.existing],
        "existing_rows": existing_rows,
        "existing_unique_urls": len(existing_urls),
        "skipped_existing_urls": len(registry_rows) - len(leads),
        "unique_leads": len(leads),
        "by_domain": dict(sorted(Counter(row["domain_id"] for row in leads).items())),
        "output": str(args.output),
    }
    if args.registry == "wto_disputes":
        report.update(
            {
                "direct_korea_complainant": sum(
                    row["complainant_code"] == "KOR" for row in registry_rows
                ),
                "direct_korea_respondent": sum(
                    row["respondent_code"] == "KOR" for row in registry_rows
                ),
            }
        )
    elif args.registry == "ohchr_korea":
        report.update(
            {
                "by_record_type": dict(
                    sorted(Counter(row["record_type"] for row in registry_rows).items())
                ),
                "by_treaty_body": dict(
                    sorted(Counter(row["treaty_body"] for row in registry_rows).items())
                ),
            }
        )
    elif args.registry == "ilo_korea_cfa":
        report.update(
            {
                "by_case_status": dict(
                    sorted(Counter(row["status"] for row in registry_rows).items())
                ),
                "registry_list_urls": {
                    status: urllib.parse.urljoin(ILO_NORMLEX_BASE_URL, relative_url)
                    for status, relative_url in ILO_KOREA_CFA_LIST_URLS.items()
                },
            }
        )
    elif args.registry == "moj_isds_archive":
        report.update(
            {
                "archive_queries": ["국제투자분쟁", "ISDS"],
                "archive_pages": (
                    registry_rows[0]["archive_pages"] if registry_rows else {}
                ),
            }
        )
    elif args.registry == "ustr_nte_korea":
        report.update(
            {
                "report_years": [row["year"] for row in registry_rows],
                "report_urls": [row["url"] for row in registry_rows],
            }
        )
    elif args.registry == "wto_tpr_korea":
        report.update(
            {
                "record_types": [row["record_type"] for row in registry_rows],
                "record_urls": [row["url"] for row in registry_rows],
            }
        )
    elif args.registry == "ohchr_upr_korea":
        report.update(
            {
                "document_symbols": [row["symbol"] for row in registry_rows],
                "record_types": [row["record_type"] for row in registry_rows],
                "record_urls": [row["url"] for row in registry_rows],
            }
        )
    elif args.registry == "un_clcs_korea":
        report.update(
            {
                "record_types": [row["record_type"] for row in registry_rows],
                "record_urls": [row["url"] for row in registry_rows],
            }
        )
    elif args.registry == "wto_stc_korea":
        report.update(
            {
                "start_date": WTO_STC_START_DATE,
                "by_wto_area": dict(sorted(
                    Counter(row["area"] for row in registry_rows).items()
                )),
                "by_korea_role": dict(sorted(Counter(
                    role
                    for row in registry_rows
                    for role in row["korea_roles"]
                ).items())),
            }
        )
    else:
        report.update(
            {
                "start_date": OHCHR_SP_START_DATE,
                "communications": sum(
                    row["record_type"] == "communication" for row in registry_rows
                ),
                "government_replies": sum(
                    row["record_type"] == "government_reply"
                    for row in registry_rows
                ),
                "communication_codes": len({
                    row["communication_code"] for row in registry_rows
                }),
                "by_record_type": dict(sorted(Counter(
                    row["record_type"] for row in registry_rows
                ).items())),
            }
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
