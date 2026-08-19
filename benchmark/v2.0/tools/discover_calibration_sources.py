#!/usr/bin/env python3
"""Match source plans to the catalog, search official sites, and fetch evidence text."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import http.client
import http.cookiejar
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
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
SEARCH_USER_AGENTS = tuple(
    f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    f"(KHTML, like Gecko) Chrome/{version}.0.0.0 Safari/537.36"
    for version in range(110, 142)
)
ROLE_ORDER = (
    "kr_primary", "comparator_primary", "foreign_or_international_primary",
    "international_or_independent_official", "independent_support",
)
INTERNATIONAL_HOSTS = {
    "un.org", "ohchr.org", "unesco.org", "who.int", "wto.org", "oecd.org",
    "imf.org", "worldbank.org", "bis.org", "iea.org", "ilo.org", "nato.int",
    "icj-cij.org", "coe.int", "europa.eu", "iaea.org", "iho.int", "unfccc.int",
    "wipo.int", "icao.int", "itu.int", "fao.org", "unhcr.org", "unodc.org",
    "unep.org", "icrc.org",
}
BLOCKED_HOST_SUFFIXES = {"kp"}
ALLOWED_CONTENT_TYPES = {
    "text/html", "application/xhtml+xml", "application/pdf", "text/plain",
    "application/xml", "text/xml", "application/json",
}
KR_PUBLIC_HOSTS = {
    "history.go.kr", "nahf.or.kr", "korean.go.kr", "bok.or.kr", "kdi.re.kr",
    "kistep.re.kr", "kisa.or.kr", "nier.go.kr", "heritage.go.kr", "0404.go.kr",
    "hira.or.kr",
    "korea.kr", "assembly.go.kr", "scourt.go.kr", "ccourt.go.kr", "nec.go.kr",
    "humanrights.go.kr", "archives.go.kr", "unikorea.go.kr", "mofa.go.kr",
    "i815.or.kr", "aks.ac.kr", "keei.re.kr", "kisdi.re.kr", "kiep.go.kr",
    "krivet.re.kr", "kwdi.re.kr", "kihasa.re.kr", "kead.or.kr",
    "kida.re.kr", "inss.re.kr", "krei.re.kr", "klri.re.kr", "etri.re.kr",
    "kari.re.kr", "kmi.re.kr", "kiet.re.kr", "kcmi.re.kr", "kif.re.kr",
    "kli.re.kr", "kipf.re.kr", "krihs.re.kr", "stepi.re.kr", "nrc.re.kr",
    "khidi.or.kr", "kims.or.kr", "nars.go.kr",
}
KR_SECONDARY_HOSTS = {
    "riss.kr", "kci.go.kr", "koreascience.kr", "scienceon.kisti.re.kr",
    "yna.co.kr", "koreatimes.co.kr", "koreaherald.com", "kbs.co.kr", "ytn.co.kr",
    "hani.co.kr", "chosun.com", "joongang.co.kr", "donga.com", "newsis.com",
    "news1.kr", "asaninst.org", "sejong.org", "kinu.or.kr", "kima.re.kr",
    "khan.co.kr", "hankookilbo.com", "jtbc.co.kr", "sbs.co.kr", "imbc.com",
    "mbn.co.kr", "ichannela.com",
    "ohmynews.com", "pressian.com", "sisain.co.kr", "mk.co.kr", "hankyung.com",
    "mt.co.kr", "sedaily.com", "edaily.co.kr", "seoul.co.kr", "segye.com",
    "kmib.co.kr", "munhwa.com", "heraldcorp.com", "nocutnews.co.kr",
    "etnews.com", "zdnet.co.kr", "lawtimes.co.kr", "joseilbo.com",
    "agrinet.co.kr", "busan.com",
    "peoplepower21.org", "minbyun.or.kr", "kctu.org", "inochong.org",
    "korcham.net", "kita.net", "fki.or.kr", "kbiz.or.kr", "kfem.or.kr",
    "greenkorea.org", "amnesty.or.kr",
}
SECONDARY_HOST_JURISDICTIONS = {
    "ac.kr": "KR", "edu.kr": "KR",
    "ac.jp": "JP", "nhk.or.jp": "JP", "asahi.com": "JP",
    "mainichi.jp": "JP", "yomiuri.co.jp": "JP", "japantimes.co.jp": "JP",
    "edu.cn": "CN", "xinhuanet.com": "CN", "people.com.cn": "CN",
    "reuters.com": "INTL", "apnews.com": "INTL", "bbc.com": "INTL",
    "rand.org": "INTL", "sipri.org": "INTL", "brookings.edu": "INTL",
    "csis.org": "INTL", "carnegieendowment.org": "INTL", "cfr.org": "INTL",
    "iiss.org": "INTL", "chathamhouse.org": "INTL", "lowyinstitute.org": "INTL",
    "thediplomat.com": "INTL", "foreignaffairs.com": "INTL", "hrw.org": "INTL",
    "amnesty.org": "INTL", "atlanticcouncil.org": "INTL", "38north.org": "INTL",
    "tandfonline.com": "INTL", "cambridge.org": "INTL", "oxfordacademic.com": "INTL",
    "sagepub.com": "INTL", "springer.com": "INTL", "sciencedirect.com": "INTL",
    "edu": "INTL",
}
SOURCE_RELEVANCE_STOP = {
    "대한민국", "한국", "정부", "공식", "자료", "문서", "기록", "보고서",
    "관련", "대한", "쟁점", "문제", "정당성", "여부", "범위", "입장", "정책",
    "국가", "국제", "근거", "주장", "검증", "official", "government", "report",
    "policy", "korea", "korean", "republic", "문화", "등재", "갈등", "분쟁",
    "구조", "경쟁", "해석", "기원", "책임", "권리", "비용", "안보", "국익",
    "and", "the", "for", "from", "with", "between",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣一-龥ぁ-んァ-ン]+", "", str(value).casefold())


def tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[0-9a-z가-힣一-龥ぁ-んァ-ン]{2,}", str(value).casefold())
        if token not in {"대한민국", "정부", "공식", "자료", "문서", "report", "official"}
    }


def text_score(left: str, right: str) -> float:
    import difflib
    seq = difflib.SequenceMatcher(None, normalized(left), normalized(right)).ratio()
    lt, rt = tokens(left), tokens(right)
    jac = len(lt & rt) / len(lt | rt) if lt and rt else 0.0
    return max(seq, 0.55 * seq + 0.45 * jac)


def source_relevance(text: str, contexts: list[str]) -> dict[str, Any]:
    clean_contexts = list(dict.fromkeys(
        re.sub(r"\s+", " ", str(value)).strip()
        for value in contexts if str(value).strip()
    ))
    if len(clean_contexts) > 1:
        assessments = [source_relevance(text, [value]) for value in clean_contexts]
        selected_index, best = max(
            enumerate(assessments),
            key=lambda item: (
                bool(item[1]["passed"]),
                len(item[1]["best_proximity_cluster"]),
                len(item[1]["matched_terms"]),
                -item[0],
            ),
        )
        return {
            **best,
            "evaluated_context_count": len(clean_contexts),
            "selected_context_index": selected_index,
        }
    folded = text.casefold()
    anchor_contexts = clean_contexts
    raw_terms = tokens(" ".join(anchor_contexts))
    terms = set()
    for raw in raw_terms:
        term = re.sub(r"(?:으로|에서|에게|까지|부터|처럼|보다|의|을|를|이|가|은|는|에|로|와|과)$", "", raw)
        minimum = 3 if term.isascii() else 2
        if len(term) >= minimum and term not in SOURCE_RELEVANCE_STOP:
            terms.add(term)
    for code in re.findall(r"\b[A-Za-z]{1,12}[-/][A-Za-z0-9-]{1,16}\b", " ".join(anchor_contexts)):
        terms.add(code.casefold())
    occurrences = []
    for term in terms:
        start = 0
        for _ in range(8):
            index = folded.find(term.casefold(), start)
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
        while position - occurrences[left][0] > 1200:
            counts[occurrences[left][1]] -= 1
            left += 1
        cluster = {value for value, count in counts.items() if count > 0}
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    matched = sorted({term for _, term in occurrences}, key=lambda value: (-len(value), value))
    distinctive = [
        term for term in best_cluster
        if len(term) >= (7 if term.isascii() else 4)
    ]
    passed = (len(best_cluster) >= 2 and bool(distinctive)) or len(best_cluster) >= 4
    return {
        "passed": passed,
        "candidate_terms": sorted(terms, key=lambda value: (-len(value), value))[:40],
        "matched_terms": matched[:20],
        "best_proximity_cluster": sorted(best_cluster, key=lambda value: (-len(value), value))[:20],
    }


def base_host(url: str) -> str:
    host = (urllib.parse.urlparse(url).hostname or "").lower().strip(".")
    return host[4:] if host.startswith("www.") else host


def host_matches(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith("." + suffix)


def host_blocked(host: str) -> bool:
    return any(host_matches(host, suffix) for suffix in BLOCKED_HOST_SUFFIXES)


def likely_homepage(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/").casefold()
    homepage_paths = {
        "", "/index.do", "/index.html", "/index.htm", "/main.do", "/main.html",
        "/www/index.do", "/eng/index.do", "/en/index.do", "/portal/main.do",
    }
    generic_leaf = re.search(r"/(?:index|main|home)(?:_[a-z]{2})?\.(?:do|html?|aspx?)$", path)
    return (path in homepage_paths or bool(generic_leaf)) and not parsed.query


def official_jurisdiction(host: str) -> str | None:
    if host_blocked(host):
        return None
    if host.endswith(".go.kr") or host.endswith(".gov.kr") or any(
        host_matches(host, suffix) for suffix in KR_PUBLIC_HOSTS
    ):
        return "KR"
    if host.endswith(".go.jp") or host.endswith(".lg.jp") or host_matches(host, "ndl.go.jp"):
        return "JP"
    if host.endswith(".gov.cn") or any(host_matches(host, suffix) for suffix in (
        "npc.gov.cn", "court.gov.cn", "fmprc.gov.cn", "mod.gov.cn", "cass.cn",
        "cssn.cn",
    )):
        return "CN"
    if host.endswith(".gov.ru") or any(host_matches(host, suffix) for suffix in (
        "kremlin.ru", "mid.ru", "mil.ru",
    )):
        return "RU"
    if host.endswith(".gov") or host.endswith(".mil"):
        return "US"
    if host_matches(host, "gov.uk") or any(host_matches(host, suffix) for suffix in (
        "parliament.uk", "supremecourt.uk", "judiciary.uk"
    )):
        return "GB"
    if host.endswith(".gc.ca") or host.endswith(".canada.ca"):
        return "CA"
    if host.endswith(".gov.au"):
        return "AU"
    if host.endswith(".gov.sg"):
        return "SG"
    if host.endswith(".gov.tw"):
        return "TW"
    if host.endswith(".gov.hk"):
        return "HK"
    if host.endswith(".gov.in") or host.endswith(".nic.in"):
        return "IN"
    if host.endswith(".gov.mn"):
        return "MN"
    if host.endswith(".gov.my"):
        return "MY"
    if host.endswith(".gov.pk"):
        return "PK"
    if host.endswith(".gov.bd"):
        return "BD"
    if host.endswith(".gov.lk"):
        return "LK"
    if host.endswith(".gouv.fr"):
        return "FR"
    if host.endswith(".bund.de") or host_matches(host, "gesetze-im-internet.de"):
        return "DE"
    if host.endswith(".go.id"):
        return "ID"
    if host.endswith(".gov.ph"):
        return "PH"
    if host.endswith(".go.th"):
        return "TH"
    if host.endswith(".gov.vn"):
        return "VN"
    if any(host_matches(host, suffix) for suffix in ("regjeringen.no", "government.no")):
        return "NO"
    if any(host_matches(host, suffix) for suffix in ("admin.ch", "eda.admin.ch")):
        return "CH"
    if any(host_matches(host, suffix) for suffix in ("regeringen.se", "government.se")):
        return "SE"
    if any(host_matches(host, suffix) for suffix in ("valtioneuvosto.fi", "finlex.fi")):
        return "FI"
    if any(host_matches(host, suffix) for suffix in ("gov.dk", "retsinformation.dk")):
        return "DK"
    if any(host_matches(host, suffix) for suffix in ("government.nl", "overheid.nl")):
        return "NL"
    if host.endswith(".gov.pl") or host_matches(host, "gov.pl"):
        return "PL"
    if host.endswith(".gob.es") or host_matches(host, "lamoncloa.gob.es"):
        return "ES"
    if host.endswith(".gob.mx") or host_matches(host, "gob.mx"):
        return "MX"
    if host.endswith(".gov.br") or host_matches(host, "gov.br"):
        return "BR"
    if host.endswith(".gob.ar") or host_matches(host, "argentina.gob.ar"):
        return "AR"
    if host.endswith(".gob.cl") or host_matches(host, "gob.cl"):
        return "CL"
    if host.endswith(".gov.co"):
        return "CO"
    if host.endswith(".gov.il"):
        return "IL"
    if host.endswith(".gov.ae"):
        return "AE"
    if host.endswith(".gov.za"):
        return "ZA"
    if host.endswith(".go.ke"):
        return "KE"
    if host.endswith(".gov.ng"):
        return "NG"
    if host.endswith(".gov.ma"):
        return "MA"
    if host.endswith(".govt.nz"):
        return "NZ"
    if any(host_matches(host, suffix) for suffix in INTERNATIONAL_HOSTS):
        return "INTL"
    return None


def secondary_jurisdiction(host: str) -> str | None:
    if host_blocked(host):
        return None
    if any(host_matches(host, suffix) for suffix in KR_SECONDARY_HOSTS):
        return "KR"
    for suffix, jurisdiction in SECONDARY_HOST_JURISDICTIONS.items():
        if host_matches(host, suffix):
            return jurisdiction
    return None


def trusted_jurisdiction(host: str) -> tuple[str | None, str | None]:
    observed = official_jurisdiction(host)
    if observed is not None:
        return observed, "A"
    observed = secondary_jurisdiction(host)
    if observed is not None:
        return observed, "B"
    return None, None


def with_source_policy(source: dict[str, Any]) -> dict[str, Any]:
    _, tier = trusted_jurisdiction(base_host(source.get("url", "")))
    # Existing verified catalog snapshots predate explicit tier metadata. They
    # remain usable only when their host passes the same A/B allow-list.
    return {
        **source,
        "reliability_tier": tier,
        "allowed_uses": (
            ["position_evidence", "fact_check", "generation_grounding"]
            if tier in {"A", "B"} else []
        ),
    }


def jurisdiction_compatible(expected: str, observed: str | None, role: str) -> bool:
    expected = {"UK": "GB", "EL": "GR"}.get(expected, expected)
    if role == "kr_primary":
        return observed == "KR"
    if observed is None or observed == "KR":
        return False
    if expected in {"INTL", "EU"}:
        return observed in {"INTL", "EU"}
    return observed in {expected, "INTL"}


def catalog_match(plan: dict[str, Any], catalog: list[dict[str, Any]], used: set[str]) -> tuple[dict[str, Any] | None, float]:
    role = plan["role"]
    expected = plan["jurisdiction"]
    target = f"{plan['document_title_or_record']} {plan['issuing_authority']}"
    native_query = str(plan.get("search_query") or "")
    best, best_score = None, 0.0
    for source in catalog:
        if source["url"] in used:
            continue
        if likely_homepage(source["url"]):
            continue
        observed, tier = trusted_jurisdiction(base_host(source["url"]))
        if tier not in {"A", "B"}:
            continue
        observed = source.get("jurisdiction") or observed
        if not jurisdiction_compatible(expected, observed, role):
            continue
        source_text = f"{source.get('title', '')} {source.get('issuing_authority', '')}"
        score = max(text_score(target, source_text), text_score(native_query, source_text))
        if score > best_score:
            best, best_score = source, score
    return best, best_score


def bing_rss_results(query: str, retries: int = 1) -> list[dict[str, str]]:
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"format": "rss", "q": query})
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=15) as response:
                body = response.read().decode("utf-8", errors="replace")
            found = []
            root = ET.fromstring(body)
            for item in root.findall("./channel/item"):
                found.append({
                    "title": html.unescape(item.findtext("title") or ""),
                    "url": html.unescape(item.findtext("link") or ""),
                    "snippet": html.unescape(item.findtext("description") or ""),
                    "engine": "bing_rss",
                })
            return found
        except (
            urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            http.client.HTTPException, ConnectionError, OSError, ET.ParseError,
        ) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    return [{"error": f"{type(last_error).__name__}: {last_error}"}] if last_error else []


def duckduckgo_lite_results(query: str, retries: int = 1) -> list[dict[str, str]]:
    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
    last_error = None
    for attempt in range(retries):
        try:
            proxy = os.environ.get("KSOVA_SEARCH_PROXY", "").strip()
            if proxy:
                process = subprocess.run(
                    [
                        "curl", "--location", "--fail", "--silent", "--show-error",
                        "--max-time", "15", "--user-agent", USER_AGENT,
                        "--proxy", proxy, url,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=20,
                )
                if process.returncode != 0:
                    raise RuntimeError(
                        process.stderr.decode("utf-8", errors="replace").strip()
                    )
                body = process.stdout.decode("utf-8", errors="replace")
            else:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=15) as response:
                    body = response.read().decode("utf-8", errors="replace")
            soup = BeautifulSoup(body, "html.parser")
            found = []
            seen = set()
            for anchor in soup.select("a.result-link"):
                target = html.unescape(str(anchor.get("href") or "")).strip()
                if target.startswith("//"):
                    target = "https:" + target
                parsed = urllib.parse.urlparse(target)
                if host_matches(base_host(target), "duckduckgo.com") and parsed.path == "/l/":
                    target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
                title = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
                if not target.startswith(("http://", "https://")) or not title or target in seen:
                    continue
                row = anchor.find_parent("tr")
                snippet_row = row.find_next_sibling("tr") if row else None
                snippet = re.sub(
                    r"\s+", " ", snippet_row.get_text(" ", strip=True) if snippet_row else ""
                ).strip()
                found.append({
                    "title": title,
                    "url": target,
                    "snippet": snippet[:1000],
                    "engine": "duckduckgo_lite",
                })
                seen.add(target)
                if len(found) == 15:
                    break
            if found:
                return found
            return []
        except (
            urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            http.client.HTTPException, ConnectionError, OSError, RuntimeError,
        ) as exc:
            last_error = exc
        time.sleep(1.2 * (attempt + 1))
    return [{"error": f"{type(last_error).__name__}: {last_error}"}] if last_error else []


def naver_results(
    query: str, start: int = 1, where: str = "web"
) -> list[dict[str, str]]:
    if where not in {"web", "news"}:
        raise ValueError(f"unsupported Naver search vertical: {where}")
    params = {"where": where, "query": query}
    if start > 1:
        params["start"] = str(start)
    url = "https://search.naver.com/search.naver?" + urllib.parse.urlencode(params)
    try:
        curl_args = [
            "curl", "--location", "--fail", "--silent", "--show-error",
            "--max-time", "15", "--user-agent", USER_AGENT,
        ]
        proxy = os.environ.get("KSOVA_SEARCH_PROXY", "").strip()
        if proxy:
            curl_args.extend(("--proxy", proxy))
        curl_args.append(url)
        process = subprocess.run(
            curl_args,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr.decode("utf-8", errors="replace").strip())
        soup = BeautifulSoup(process.stdout.decode("utf-8", errors="replace"), "html.parser")
        found = []
        seen = set()
        for anchor in soup.find_all("a"):
            target = html.unescape(str(anchor.get("href") or "")).strip()
            title = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
            if not target.startswith(("http://", "https://")) or not title:
                continue
            host = base_host(target)
            if host_matches(host, "naver.com") or host_matches(host, "naver.net"):
                continue
            if target in seen:
                continue
            parent_text = re.sub(r"\s+", " ", anchor.parent.get_text(" ", strip=True)).strip()
            found.append({
                "title": title,
                "url": target,
                "snippet": parent_text[:1000],
                "engine": f"naver_{where}",
            })
            seen.add(target)
            if len(found) == 30:
                break
        return found
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def brave_results(query: str, retries: int = 1) -> list[dict[str, str]]:
    """Return ordinary web-result URLs from Brave's public result page."""
    url = "https://search.brave.com/search?" + urllib.parse.urlencode({
        "q": query,
        "source": "web",
    })
    last_error = None
    for attempt in range(max(1, retries)):
        try:
            agent_index = (
                int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16) + attempt
            ) % len(SEARCH_USER_AGENTS)
            process = subprocess.run(
                [
                    "curl", "--location", "--fail", "--silent", "--show-error",
                    "--max-time", "20", "--user-agent", SEARCH_USER_AGENTS[agent_index], url,
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25,
            )
            if process.returncode != 0:
                raise RuntimeError(process.stderr.decode("utf-8", errors="replace").strip())
            soup = BeautifulSoup(process.stdout.decode("utf-8", errors="replace"), "html.parser")
            found = []
            seen = set()
            for container in soup.select('div.snippet[data-type="web"]'):
                title_node = container.select_one(".search-snippet-title")
                anchor = title_node.find_parent("a", href=True) if title_node else None
                if anchor is None:
                    continue
                target = html.unescape(str(anchor.get("href") or "")).strip()
                if not target.startswith(("http://", "https://")) or target in seen:
                    continue
                host = base_host(target)
                if host_matches(host, "brave.com") or host_matches(host, "search.brave.com"):
                    continue
                title = re.sub(
                    r"\s+", " ",
                    str(title_node.get("title") or title_node.get_text(" ", strip=True)),
                ).strip()
                snippet_node = container.select_one(".generic-snippet .content")
                snippet = re.sub(
                    r"\s+", " ",
                    snippet_node.get_text(" ", strip=True) if snippet_node else "",
                ).strip()
                if not title:
                    continue
                found.append({
                    "title": title,
                    "url": target,
                    "snippet": snippet[:1000],
                    "engine": "brave_web",
                })
                seen.add(target)
                if len(found) == 30:
                    break
            return found
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            last_error = exc
        time.sleep(1.2 * (attempt + 1))
    return [{"error": f"{type(last_error).__name__}: {last_error}"}] if last_error else []


def search_results(query: str, retries: int = 1) -> list[dict[str, str]]:
    # Search output is only a URL-discovery index. Evidence always comes from a
    # separately fetched official document and never from result snippets.
    url = "https://search.yahoo.co.jp/search?" + urllib.parse.urlencode({"p": query})
    last_error = None
    for attempt in range(retries):
        try:
            agent_index = (
                int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16) + attempt
            ) % len(SEARCH_USER_AGENTS)
            process = subprocess.run(
                [
                    "curl", "--location", "--fail", "--silent", "--show-error",
                    "--max-time", "15", "--user-agent", SEARCH_USER_AGENTS[agent_index], url,
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
            )
            if process.returncode != 0:
                raise RuntimeError(process.stderr.decode("utf-8", errors="replace").strip())
            body = process.stdout.decode("utf-8", errors="replace")
            soup = BeautifulSoup(body, "html.parser")
            found = []
            seen = set()
            for anchor in soup.find_all("a"):
                target = html.unescape(str(anchor.get("href") or "")).strip()
                title = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
                if not target.startswith(("http://", "https://")) or not title:
                    continue
                host = base_host(target)
                if host_matches(host, "yahoo.co.jp") or host_matches(host, "lycorp.co.jp"):
                    continue
                if target in seen:
                    continue
                parent_text = re.sub(r"\s+", " ", anchor.parent.get_text(" ", strip=True)).strip()
                found.append({
                    "title": title,
                    "url": target,
                    "snippet": parent_text[:1000],
                    "engine": "yahoo_jp",
                })
                seen.add(target)
                if len(found) == 15:
                    break
            if found:
                return found
            return []
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            last_error = exc
        time.sleep(2.0 * (attempt + 1))
    return [{"error": f"{type(last_error).__name__}: {last_error}"}] if last_error else []


def jurisdiction_scope(jurisdiction: str) -> str:
    return {
        "KR": "site:go.kr",
        "JP": "site:go.jp",
        "CN": "site:gov.cn",
        "US": "site:gov",
        "GB": "site:gov.uk",
        "UK": "site:gov.uk",
        "CA": "site:canada.ca",
        "AU": "site:gov.au",
        "SG": "site:gov.sg",
        "TW": "site:gov.tw",
        "IN": "site:gov.in",
        "MN": "site:gov.mn",
        "MY": "site:gov.my",
        "ID": "site:go.id",
        "PH": "site:gov.ph",
        "TH": "site:go.th",
        "VN": "site:gov.vn",
        "PK": "site:gov.pk",
        "BD": "site:gov.bd",
        "LK": "site:gov.lk",
        "FR": "site:gouv.fr",
        "DE": "site:bund.de",
        "RU": "site:gov.ru",
        "NO": "site:regjeringen.no",
        "CH": "site:admin.ch",
        "SE": "site:regeringen.se",
        "FI": "site:valtioneuvosto.fi",
        "DK": "site:gov.dk",
        "NL": "site:government.nl",
        "PL": "site:gov.pl",
        "ES": "site:gob.es",
        "MX": "site:gob.mx",
        "BR": "site:gov.br",
        "AR": "site:argentina.gob.ar",
        "CL": "site:gob.cl",
        "CO": "site:gov.co",
        "IL": "site:gov.il",
        "AE": "site:gov.ae",
        "ZA": "site:gov.za",
        "KE": "site:go.ke",
        "NG": "site:gov.ng",
        "MA": "site:gov.ma",
        "NZ": "site:govt.nz",
        "EU": "site:europa.eu",
    }.get(jurisdiction, "")


def web_match(plan: dict[str, Any], used: set[str], candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    search_query = str(plan.get("search_query") or "").strip()
    original_query = str(plan.get("original_search_query") or "").strip()
    scope = jurisdiction_scope(str(plan.get("jurisdiction") or ""))
    core_query = original_query or search_query or str(plan["document_title_or_record"])
    scoped_query = f"{scope} {core_query}".strip() if scope else core_query
    queries = list(dict.fromkeys(filter(None, (scoped_query, core_query))))
    candidates = []
    result_count = 0
    query_errors = []

    def collect_results(results: list[dict[str, str]], query: str,
                        query_index: int) -> None:
        nonlocal result_count
        result_count += len(results)
        for rank, result in enumerate(results, 1):
            if result.get("error"):
                query_errors.append(result["error"])
                continue
            url = result["url"]
            if url in used or not url.startswith(("http://", "https://")):
                continue
            if likely_homepage(url):
                continue
            host = base_host(url)
            if host_blocked(host):
                continue
            observed, tier = trusted_jurisdiction(host)
            if not jurisdiction_compatible(plan["jurisdiction"], observed, plan["role"]):
                continue
            searchable_result = f"{result['title']} {result.get('snippet', '')}"
            score = (
                0.25 * text_score(plan["document_title_or_record"], searchable_result)
                + 0.55 * text_score(plan.get("search_query", ""), searchable_result)
                + 0.20 * text_score(candidate["title_ko"], searchable_result)
            )
            if observed == plan["jurisdiction"]:
                score += 0.18
            score += max(0, 0.10 - rank * 0.008 - query_index * 0.01)
            candidates.append((score, result, observed, tier, host, rank, query))

    jurisdiction = str(plan.get("jurisdiction") or "")
    if jurisdiction == "KR":
        engines = (naver_results, bing_rss_results, duckduckgo_lite_results, search_results)
    elif jurisdiction == "JP":
        engines = (search_results, bing_rss_results, duckduckgo_lite_results)
    else:
        engines = (bing_rss_results, duckduckgo_lite_results, search_results, naver_results)
    # One jurisdiction-preferred search plus one unscoped recovery search keeps
    # the cost bounded at two engine calls per source plan.
    engine_queries = [(engines[0], queries[0])]
    if len(queries) > 1:
        engine_queries.append((engines[1], queries[1]))
    for query_index, (engine, query) in enumerate(engine_queries, 1):
        collect_results(engine(query), query, query_index)
    if not candidates:
        return None, {
            "queries": queries,
            "result_count": result_count,
            "official_matches": 0,
            "query_errors": query_errors,
        }
    ranked = sorted(candidates, key=lambda row: row[0], reverse=True)
    unique_ranked = []
    seen_urls = set()
    for row in ranked:
        if row[1]["url"] in seen_urls:
            continue
        unique_ranked.append(row)
        seen_urls.add(row[1]["url"])
        if len(unique_ranked) == 5:
            break

    def as_source(row: tuple[Any, ...]) -> dict[str, Any]:
        _, result, observed, tier, _, _, _ = row
        return {
            "schema_version": "ksova-v2-calibration-source-1.0",
            "source_id": "v2cal_web_" + hashlib.sha256(result["url"].encode()).hexdigest()[:16],
            "title": result["title"],
            "url": result["url"],
            "jurisdiction": observed,
            "authority_type": (
                "official_web_document" if tier == "A" else "trusted_secondary_document"
            ),
            "issuing_authority": plan["issuing_authority"],
            "section": None,
            "evidence_scope_ko": result.get("snippet", ""),
            "evidence_claims": [],
            "snapshot_sha256": None,
            "availability": "search_discovered",
            "origins": ["v2_calibration_web_discovery"],
            "reliability_tier": tier,
            "allowed_uses": ["position_evidence", "fact_check", "generation_grounding"],
        }

    score, result, observed, tier, host, rank, selected_query = unique_ranked[0]
    source = as_source(unique_ranked[0])
    source["retrieval_alternatives"] = [as_source(row) for row in unique_ranked[1:]]
    return source, {
        "queries": queries,
        "selected_query": selected_query,
        "result_count": result_count,
        "official_matches": len(candidates),
        "selected_rank": rank,
        "title_match_score": round(score, 3),
        "host": host,
        "reliability_tier": tier,
        "search_engine": result.get("engine"),
        "retrieval_alternatives": len(unique_ranked) - 1,
        "query_errors": query_errors,
    }


def extract_html_text(body: bytes) -> str:
    decoded = body.decode("utf-8", errors="replace")
    # A few legacy publishers leave an earlier ``script`` tag malformed.
    # HTML parsers then treat the remainder, including the visible article,
    # as JavaScript.  Recover an explicitly marked article region from the
    # raw document before asking BeautifulSoup to parse the whole page.
    raw_article = re.search(
        r'''(?is)<(?:div|article)\b[^>]*(?:
            itemprop\s*=\s*["']articleBody["']|
            id\s*=\s*["']newsViewArea["']
        )[^>]*>''',
        decoded,
        flags=re.VERBOSE,
    )
    if raw_article:
        end_candidates = [
            decoded.find(marker, raw_article.end())
            for marker in (
                "<!--// 기사 본문 내용 -->",
                "<!-- // 기사 본문 내용 -->",
                "<!-- 기자 정보 -->",
                "</article>",
            )
        ]
        end_candidates = [value for value in end_candidates if value >= 0]
        end = min(end_candidates) if end_candidates else min(
            len(decoded), raw_article.end() + 60_000
        )
        fragment = BeautifulSoup(decoded[raw_article.end():end], "html.parser")
        for element in fragment(["script", "style", "nav", "footer", "noscript", "svg"]):
            element.decompose()
        raw_text = re.sub(r"\s+", " ", fragment.get_text(" ", strip=True)).strip()
        if len(raw_text) >= 200:
            return raw_text
    soup = BeautifulSoup(body, "html.parser")
    # Some legacy news pages contain malformed navigation markup that wraps
    # the actual article node.  Removing every ``nav`` first can therefore
    # erase a perfectly usable body.  Prefer explicit article containers and
    # clean a detached copy before falling back to whole-page extraction.
    article_selectors = (
        '[itemprop="articleBody"]',
        "#newsViewArea",
        "article",
        ".article-body",
        ".article_body",
        ".articleBody",
        ".news_view",
        ".view_con",
        ".article_txt",
        ".news_end",
    )
    article_candidates = []
    for selector in article_selectors:
        for node in soup.select(selector):
            fragment = BeautifulSoup(str(node), "html.parser")
            for element in fragment(["script", "style", "nav", "footer", "noscript", "svg"]):
                element.decompose()
            text = re.sub(r"\s+", " ", fragment.get_text(" ", strip=True)).strip()
            if len(text) >= 200:
                article_candidates.append(text)
    if article_candidates:
        return max(article_candidates, key=len)
    for element in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
        element.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def extract_html_published_at(body: bytes) -> str:
    """Return an article publication timestamp from page metadata when present.

    Body text often omits the year while sidebars contain newer dates.  Keeping
    the publisher supplied metadata separate prevents a repeated event from
    being mistaken for the concrete event named by a recovery query.
    """
    soup = BeautifulSoup(body, "html.parser")
    selectors = (
        ("meta", {"property": "article:published_time"}, "content"),
        ("meta", {"name": "published_date"}, "content"),
        ("meta", {"name": "pubdate"}, "content"),
        ("meta", {"itemprop": "datePublished"}, "content"),
        ("time", {"itemprop": "datePublished"}, "datetime"),
    )
    for tag_name, attrs, value_attr in selectors:
        element = soup.find(tag_name, attrs=attrs)
        value = str(element.get(value_attr) or "").strip() if element else ""
        if re.search(r"(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}", value):
            return value
    # Some publishers expose datePublished only inside JSON-LD.
    match = re.search(
        rb'"datePublished"\s*:\s*"([^"\\]{8,64})"', body, flags=re.I,
    )
    if match:
        value = match.group(1).decode("utf-8", errors="replace").strip()
        if re.search(r"(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}", value):
            return value
    return ""


def relevant_excerpt(text: str, contexts: list[str], limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    context_text = " ".join(contexts)
    anchors: list[str] = []
    for value in re.findall(r"제\s*\d+\s*조(?:의\s*\d+)?", context_text):
        anchors.append(re.sub(r"\s+", "", value))
    for value in re.findall(r"(?i)article\s+\d+(?:\.\d+)?", context_text):
        anchors.append(re.sub(r"\s+", " ", value))
    for match in re.finditer(
        r"(?i)sections?\s+(\d+(?:\.\d+)?)(?:\s+(?:and|to)\s+(\d+(?:\.\d+)?))?",
        context_text,
    ):
        anchors.extend(f"Section {number}" for number in match.groups() if number)
    token_terms = sorted(tokens(context_text), key=lambda value: (-len(value), value))[:64]
    terms = list(dict.fromkeys(anchors + token_terms))
    folded = text.casefold()
    frequencies = {
        term: folded.count(term.casefold())
        for term in terms
        if term and term.casefold() in folded
    }
    if not frequencies:
        # Cross-language validation contexts may not share literal tokens with
        # an English source.  In that case retain the requested amount rather
        # than silently collapsing to a short lead paragraph.
        return text[:limit]

    # Rank overlapping chunks by distinctive context coverage instead of
    # sorting every hit by document position.  The previous implementation
    # could spend the entire excerpt budget on a long introduction containing
    # a frequent word such as "사할린", cutting off the later paragraph that
    # contained the concrete event, quotation, and figures under audit.
    chunk_size = min(1400, max(700, limit // 2))
    stride = max(350, chunk_size // 2)
    anchor_set = set(anchors)
    candidates: list[tuple[float, int, int]] = []
    for start in range(0, len(text), stride):
        end = min(len(text), start + chunk_size)
        chunk = folded[start:end]
        score = 0.0
        matched = 0
        for term, frequency in frequencies.items():
            if term.casefold() not in chunk:
                continue
            matched += 1
            rarity = 1.0 / max(1.0, frequency ** 0.5)
            score += (len(term) + 3.0) * rarity
            if term in anchor_set:
                score += 20.0
        if matched:
            # Coverage keeps a passage with several mutually supporting
            # anchors ahead of one that merely repeats a single frequent term.
            score += matched * matched * 0.75
            candidates.append((score, start, end))
        if end == len(text):
            break
    if not candidates:
        return text[:limit]

    selected: list[tuple[int, int]] = []
    used = 0
    for _score, start, end in sorted(candidates, key=lambda row: (-row[0], row[1])):
        if any(start < prior_end and end > prior_start for prior_start, prior_end in selected):
            continue
        remaining = limit - used - (5 if selected else 0)
        if remaining <= 0:
            break
        if end - start > remaining:
            end = start + remaining
        selected.append((start, end))
        used += end - start + (5 if len(selected) > 1 else 0)
    selected.sort()
    return "\n...\n".join(text[start:end].strip() for start, end in selected)[:limit]


def fetch_source_once(source: dict[str, Any], snapshot_dir: Path) -> dict[str, Any]:
    if source.get("availability") == "snapshot_verified" and source.get("evidence_claims"):
        claim_chars = sum(
            len(str(claim.get("claim_ko") or ""))
            for claim in source["evidence_claims"]
            if isinstance(claim, dict)
        )
        relevance_text = " ".join([
            str(source.get("title") or ""), str(source.get("evidence_scope_ko") or ""),
            *(
                str(claim.get("claim_ko") or "")
                for claim in source["evidence_claims"] if isinstance(claim, dict)
            ),
        ])
        relevance = source_relevance(relevance_text, source.get("retrieval_context", []))
        return {
            **source,
            "retrieval": {
                "status": "reused_verified_snapshot",
                "text_chars": max(200, claim_chars),
                "relevance_pass": relevance["passed"],
                "relevance": relevance,
            },
        }
    source_id = str(source.get("source_id") or "")
    for suffix in (".pdf", ".html"):
        snapshot = snapshot_dir / f"{source_id}{suffix}"
        if not source_id or not snapshot.is_file():
            continue
        body = snapshot.read_bytes()
        if suffix == ".pdf":
            text_path = snapshot.with_suffix(".txt")
            text = (
                text_path.read_text(encoding="utf-8", errors="replace")
                if text_path.is_file() else ""
            )
            content_type = "application/pdf"
        else:
            text = extract_html_text(body)
            content_type = "text/html"
        published_at = (
            extract_html_published_at(body) if suffix == ".html" else ""
        )
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 200:
            continue
        contexts = source.get("retrieval_context", [])
        relevance = source_relevance(text, contexts)
        return {
            **source,
            "published_at": published_at or source.get("published_at") or "",
            "snapshot_sha256": hashlib.sha256(body).hexdigest(),
            "availability": "fetched",
            "retrieval": {
                "status": "reused_run_snapshot",
                "source_url": source["url"],
                "final_url": source["url"],
                "content_type": content_type,
                "size_bytes": len(body),
                "text_chars": len(text),
                "text_excerpt": relevant_excerpt(text, contexts),
                "relevance_pass": relevance["passed"],
                "relevance": relevance,
            },
        }
    url = source.get("retrieval_url") or source["url"]
    parsed_url = urllib.parse.urlsplit(url)
    if host_blocked((parsed_url.hostname or "").casefold()):
        return {
            **source,
            "availability": "fetch_blocked",
            "retrieval": {"status": "blocked", "error": "blocked host suffix"},
        }
    request_url = urllib.parse.urlunsplit((
        parsed_url.scheme,
        parsed_url.netloc,
        urllib.parse.quote(parsed_url.path, safe="/%:@"),
        urllib.parse.quote(parsed_url.query, safe="=&%:+,()"),
        parsed_url.fragment,
    ))
    request_user_agent = "Mozilla/5.0" if parsed_url.hostname == "r.jina.ai" else USER_AGENT
    max_size = 40 * 1024 * 1024
    try:
        request = urllib.request.Request(request_url, headers={"User-Agent": request_user_agent})
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        with opener.open(request, timeout=45) as response:
            final_url = response.geturl()
            status = getattr(response, "status", 200)
            content_type = response.headers.get_content_type()
            final_host = base_host(final_url)
            if host_blocked(final_host):
                raise ValueError("redirected to blocked host suffix")
            required_final_host = source.get("required_final_host_suffix")
            if required_final_host and not host_matches(final_host, required_final_host):
                raise ValueError(
                    f"redirected outside required publisher host: {final_host}"
                )
            generic_download_types = {
                "application/octet-stream",
                "application/x-download",
                "application/x-msdownload",
                # Several Korean public archives serve valid PDF bytes with a
                # generic download header.  These values are accepted only
                # provisionally: the PDF magic-byte gate below still rejects
                # every non-PDF payload.
                "application/unknown",
                "application/download",
                "application/force-download",
                "binary/octet-stream",
            }
            if (
                content_type not in ALLOWED_CONTENT_TYPES
                and content_type not in generic_download_types
            ):
                raise ValueError(f"unsupported content type: {content_type}")
            body = response.read(max_size + 1)
        if len(body) > max_size:
            raise ValueError("response exceeds 40 MiB")
        if (
            content_type not in ALLOWED_CONTENT_TYPES
            and not body.startswith(b"%PDF")
        ):
            raise ValueError(
                f"generic download is not a PDF: {content_type}"
            )
        digest = hashlib.sha256(body).hexdigest()
        suffix = ".pdf" if content_type == "application/pdf" or body[:4] == b"%PDF" else ".html"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"{source['source_id']}{suffix}"
        snapshot.write_bytes(body)
        if suffix == ".pdf":
            text_path = snapshot.with_suffix(".txt")
            proc = subprocess.run(
                ["pdftotext", "-layout", str(snapshot), str(text_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90,
            )
            text = text_path.read_text(encoding="utf-8", errors="replace") if proc.returncode == 0 and text_path.exists() else ""
        else:
            text = extract_html_text(body)
        published_at = (
            extract_html_published_at(body) if suffix == ".html" else ""
        )
        text = re.sub(r"\s+", " ", text).strip()
        contexts = source.get("retrieval_context", [])
        excerpt = relevant_excerpt(text, contexts)
        relevance = source_relevance(text, contexts)
        return {
            **source,
            "published_at": published_at or source.get("published_at") or "",
            "snapshot_sha256": digest,
            "availability": "fetched",
            "retrieval": {
                "status": "fetched",
                "http_status": status,
                "source_url": source["url"],
                "transport_url": url if url != source["url"] else None,
                "final_url": final_url,
                "content_type": content_type,
                "size_bytes": len(body),
                "text_chars": len(text),
                "text_excerpt": excerpt,
                "relevance_pass": relevance["passed"],
                "relevance": relevance,
            },
        }
    except Exception as exc:  # network and parser failures are data, not pipeline crashes
        return {
            **source,
            "availability": "fetch_failed",
            "retrieval": {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        }


def fetch_source(source: dict[str, Any], snapshot_dir: Path) -> dict[str, Any]:
    alternatives = source.get("retrieval_alternatives", [])
    base = with_source_policy({
        key: value for key, value in source.items() if key != "retrieval_alternatives"
    })
    options = [base]
    for alternative in alternatives:
        options.append(with_source_policy({
            **alternative,
            "retrieval_context": source.get("retrieval_context", []),
        }))
    attempts = []
    for rank, option in enumerate(options, 1):
        fetched = fetch_source_once(option, snapshot_dir)
        retrieval = fetched.get("retrieval", {})
        usable = (
            retrieval.get("status") in {
                "fetched", "reused_verified_snapshot", "reused_run_snapshot"
            }
            and retrieval.get("text_chars", 1) >= 200
            and retrieval.get("relevance_pass") is True
        )
        attempts.append({
            "rank": rank,
            "url": option["url"],
            "status": retrieval.get("status"),
            "text_chars": retrieval.get("text_chars", 0),
            "relevance_pass": retrieval.get("relevance_pass"),
            "error": retrieval.get("error"),
        })
        if usable:
            fetched["retrieval"] = {
                **retrieval,
                "selected_search_rank": rank,
                "attempts": attempts,
            }
            return fetched
    return {
        **base,
        "availability": "fetch_failed",
        "retrieval": {
            "status": "failed",
            "error": "no search candidate yielded at least 200 characters",
            "attempts": attempts,
        },
    }


def discover_candidate_sources(candidate: dict[str, Any], catalog: list[dict[str, Any]],
                               search_delay: float) -> dict[str, Any]:
    used: set[str] = set()
    selected = []
    attempts = []
    local_stats: Counter[str] = Counter()
    source_plans = candidate.get("source_search_plan") or candidate.get(
        "official_source_search_plan", []
    )
    plans = sorted(
        (plan for plan in source_plans
         if plan["role"] in {
             "kr_primary", "comparator_primary", "foreign_or_international_primary",
             "international_or_independent_official",
         }),
        key=lambda row: ROLE_ORDER.index(row["role"]),
    )
    for plan in plans:
        retrieval_context = list(dict.fromkeys(filter(None, (
            candidate.get("title_ko"), candidate.get("question_ko"),
            candidate.get("conflict_axis_ko"), plan.get("search_query"),
            plan.get("document_title_or_record"), plan.get("claim_to_verify_ko"),
        ))))
        source, score = catalog_match(plan, catalog, used)
        # Reuse a catalog source directly only on a strong title/authority match.
        # Moderate matches are often a different decision from the same broad topic;
        # in that case search the planned official record first.
        if source is not None and score >= 0.72:
            source = {**with_source_policy(source), "retrieval_context": retrieval_context}
            recovery_detail = None
            if source.get("availability") != "snapshot_verified":
                recovery, recovery_detail = web_match(plan, used | {source["url"]}, candidate)
                time.sleep(search_delay)
                if recovery is not None:
                    source["retrieval_alternatives"] = [
                        {key: value for key, value in recovery.items()
                         if key != "retrieval_alternatives"},
                        *recovery.get("retrieval_alternatives", []),
                    ]
            selected.append({"role": plan["role"], "plan": plan, "source": source})
            used.add(source["url"])
            attempts.append({
                "role": plan["role"], "method": "catalog_match",
                "score": round(score, 3), "source_id": source["source_id"],
                "web_recovery": recovery_detail,
            })
            local_stats["catalog_match"] += 1
            if source.get("retrieval_alternatives"):
                local_stats["catalog_web_recovery"] += 1
            continue
        web_source, detail = web_match(plan, used, candidate)
        time.sleep(search_delay)
        if web_source is not None:
            web_source["retrieval_context"] = retrieval_context
            if source is not None and score >= 0.45:
                web_source.setdefault("retrieval_alternatives", []).append(source)
            selected.append({"role": plan["role"], "plan": plan, "source": web_source})
            used.add(web_source["url"])
            attempts.append({"role": plan["role"], "method": "web_search",
                             "source_id": web_source["source_id"],
                             "catalog_fallback_score": round(score, 3), **detail})
            local_stats["web_match"] += 1
        elif source is not None and score >= 0.45:
            source = {**with_source_policy(source), "retrieval_context": retrieval_context}
            selected.append({"role": plan["role"], "plan": plan, "source": source})
            used.add(source["url"])
            attempts.append({
                "role": plan["role"], "method": "catalog_fallback",
                "score": round(score, 3), "source_id": source["source_id"],
                "web_search": detail,
            })
            local_stats["catalog_fallback"] += 1
        else:
            attempts.append({
                "role": plan["role"], "method": "not_found",
                "best_catalog_score": round(score, 3), **detail,
            })
            local_stats["not_found"] += 1
    return {
        "candidate": candidate,
        "selected": selected,
        "search_attempts": attempts,
        "search_stats": dict(local_stats),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--search-delay", type=float, default=0.2)
    parser.add_argument("--search-workers", type=int, default=6)
    parser.add_argument("--fetch-workers", type=int, default=12)
    args = parser.parse_args()
    shortlist = read_jsonl(args.shortlist)
    catalog = read_jsonl(args.catalog)
    search_stats = Counter()
    all_sources: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.search_workers) as pool:
        futures = {
            pool.submit(discover_candidate_sources, candidate, catalog, args.search_delay): index
            for index, candidate in enumerate(shortlist)
        }
        drafts_by_index = {}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            candidate_index = futures[future]
            try:
                draft = future.result()
            except Exception as exc:  # isolate one failed search from the full collection run
                draft = {
                    "candidate": shortlist[candidate_index],
                    "selected": [],
                    "search_attempts": [{
                        "role": "candidate",
                        "method": "candidate_search_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }],
                    "search_stats": {"candidate_search_error": 1},
                }
            drafts_by_index[candidate_index] = draft
            search_stats.update(draft.pop("search_stats"))
            if completed % 25 == 0:
                print(f"[search {completed}/{len(shortlist)}] {dict(search_stats)}", flush=True)
    packet_drafts = [drafts_by_index[index] for index in range(len(shortlist))]
    for draft in packet_drafts:
        for binding in draft["selected"]:
            source = binding["source"]
            existing = all_sources.get(source["source_id"])
            if existing:
                source["retrieval_context"] = list(dict.fromkeys(
                    existing.get("retrieval_context", []) + source.get("retrieval_context", [])
                ))
            all_sources[source["source_id"]] = source

    to_fetch = [source for source in all_sources.values()]
    fetched_by_id: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.fetch_workers) as pool:
        futures = {
            pool.submit(fetch_source, source, args.snapshot_dir): source["source_id"]
            for source in to_fetch
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            fetched = future.result()
            fetched_by_id[futures[future]] = fetched
            if index % 50 == 0:
                print(f"[fetch {index}/{len(to_fetch)}]", flush=True)

    packets = []
    complete = 0
    binding_relevance_rejected = 0
    for draft in packet_drafts:
        bindings = []
        failed_bindings = []
        for binding in draft["selected"]:
            fetched_source = fetched_by_id[binding["source"]["source_id"]]
            retrieval = fetched_source.get("retrieval") or {}
            binding_context = binding["source"].get("retrieval_context", [])
            claim_text = " ".join(
                str(claim.get("claim_ko") or "")
                for claim in fetched_source.get("evidence_claims", [])
                if isinstance(claim, dict)
            )
            binding_text = " ".join(filter(None, (
                str(fetched_source.get("title") or ""),
                str(fetched_source.get("evidence_scope_ko") or ""),
                claim_text,
                str(retrieval.get("text_excerpt") or ""),
            )))
            binding_relevance = source_relevance(binding_text, binding_context)
            source = {
                **fetched_source,
                "retrieval_context": binding_context,
                "retrieval": {
                    **retrieval,
                    "collection_relevance": retrieval.get("relevance"),
                    "relevance": binding_relevance,
                    "relevance_pass": binding_relevance["passed"],
                },
            }
            resolved = {"role": binding["role"], "plan": binding["plan"], "source": source}
            retrieval = source["retrieval"]
            usable = (
                retrieval.get("status") in {"fetched", "reused_verified_snapshot"}
                and retrieval.get("text_chars", 0) >= 200
                and retrieval.get("relevance_pass") is True
            )
            if usable:
                bindings.append(resolved)
            else:
                failed_bindings.append(resolved)
                if retrieval.get("status") in {"fetched", "reused_verified_snapshot"}:
                    binding_relevance_rejected += 1
        kr_ok = any(
            binding["role"] == "kr_primary"
            and binding["source"]["retrieval"]["status"] in {"fetched", "reused_verified_snapshot"}
            and (binding["source"]["retrieval"].get("text_chars", 1) >= 200)
            and binding["source"]["retrieval"].get("relevance_pass") is True
            for binding in bindings
        )
        foreign_ok = any(
            binding["role"] in {"comparator_primary", "foreign_or_international_primary"}
            and binding["source"]["retrieval"]["status"] in {"fetched", "reused_verified_snapshot"}
            and (binding["source"]["retrieval"].get("text_chars", 1) >= 200)
            and binding["source"]["retrieval"].get("relevance_pass") is True
            for binding in bindings
        )
        status = "retrieval_complete" if kr_ok and foreign_ok else "retrieval_incomplete"
        complete += int(status == "retrieval_complete")
        packets.append({
            "schema_version": "ksova-v2-source-packet-draft-1.0",
            "candidate_id": draft["candidate"]["candidate_id"],
            "type_id": draft["candidate"]["type_id"],
            "domain_id": draft["candidate"]["domain_id"],
            "candidate": draft["candidate"],
            "source_status": status,
            "source_bindings": bindings,
            "failed_source_bindings": failed_bindings,
            "search_attempts": draft["search_attempts"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packets),
        encoding="utf-8",
    )
    fetch_counts = Counter(source["retrieval"]["status"] for source in fetched_by_id.values())
    report = {
        "schema_version": "ksova-v2-source-discovery-report-1.0",
        "shortlist": len(shortlist),
        "packets": len(packets),
        "retrieval_complete": complete,
        "complete_by_type": dict(sorted(Counter(
            row["type_id"] for row in packets if row["source_status"] == "retrieval_complete"
        ).items())),
        "unique_sources": len(fetched_by_id),
        "search": dict(search_stats),
        "retrieval": dict(fetch_counts),
        "binding_relevance_rejected": binding_relevance_rejected,
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
