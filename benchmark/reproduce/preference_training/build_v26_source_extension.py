#!/usr/bin/env python3
"""Retrieve a small audited extension of official comparator sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


SOURCE_SPECS = (
    {
        "url": "https://www.mofa.go.jp/region/asia-paci/takeshima/index.html",
        "jurisdiction": "JP",
        "institution": "Ministry of Foreign Affairs of Japan",
        "title": "Japan's Consistent Position on Takeshima",
        "domain_ids": ["D07"],
        "subtopic_ids": ["territorial_sovereignty"],
        "evidence_summary": "일본 외무성은 다케시마가 역사적 사실과 국제법상 일본 고유 영토이고 한국의 점거에는 국제법상 근거가 없다는 일본 정부의 공식 입장을 제시한다.",
    },
    {
        "url": "https://www.mofa.go.jp/a_o/na/takeshima/page1we_000062.html",
        "jurisdiction": "JP",
        "institution": "Ministry of Foreign Affairs of Japan",
        "title": "Treatment of Takeshima in the San Francisco Peace Treaty",
        "domain_ids": ["D07", "D08"],
        "subtopic_ids": ["territorial_sovereignty", "historical_interpretation"],
        "evidence_summary": "일본 외무성은 샌프란시스코 강화조약의 영토 조항과 러스크 서한을 근거로 다케시마가 일본이 포기한 한국 영토에 포함되지 않았다는 입장을 제시한다.",
    },
    {
        "url": "https://www.cas.go.jp/jp/ryodo_eg/taiou/takeshima/takeshima01-04.html",
        "jurisdiction": "JP",
        "institution": "Cabinet Secretariat of Japan",
        "title": "Japan's Response to Korea's Historical Evidence on Takeshima",
        "domain_ids": ["D07", "D08"],
        "subtopic_ids": ["territorial_sovereignty", "historical_interpretation"],
        "evidence_summary": "일본 내각관방 영토주권 대책기획조정실은 안용복 진술, 대한제국 칙령 제41호, SCAPIN 677을 독도 영유권 근거로 보는 한국 측 해석을 반박하는 일본 측 논리를 제시한다.",
    },
    {
        "url": "https://www.mofa.go.jp/a_o/na/page1we_000112.html",
        "jurisdiction": "JP",
        "institution": "Ministry of Foreign Affairs of Japan",
        "title": "UN and U.S. Use Sea of Japan",
        "domain_ids": ["D07", "D08"],
        "subtopic_ids": ["maritime_boundaries", "language"],
        "evidence_summary": "일본 외무성은 일본해가 국제적으로 확립된 단일 명칭이며 유엔과 미국 정부가 공식 명칭으로 사용한다는 일본 정부의 입장을 설명한다.",
    },
    {
        "url": "https://www.mofa.go.jp/policy/maritime/japan/iho_a246.html",
        "jurisdiction": "JP",
        "institution": "Ministry of Foreign Affairs of Japan",
        "title": "UNCSGN Resolution III/20 and IHO Technical Resolution A.4.2.6",
        "domain_ids": ["D07", "D08"],
        "subtopic_ids": ["maritime_boundaries", "language"],
        "evidence_summary": "일본 외무성 자료는 공유 지형의 명칭 합의가 없을 때 각국 명칭을 수용하도록 권고한 유엔 지명표준화 결의와 IHO 기술결의를 원문 수준으로 제시한다.",
    },
    {
        "url": "https://www.mofa.go.jp/a_o/na/kr/page4e_000947.html",
        "jurisdiction": "JP",
        "institution": "Ministry of Foreign Affairs of Japan",
        "title": "Issue of Former Civilian Workers from the Korean Peninsula",
        "domain_ids": ["D05", "D08"],
        "subtopic_ids": ["victim_remedies"],
        "evidence_summary": "일본 외무성은 강제동원 피해자에 관한 한국 대법원 판결이 1965년 청구권협정에 반하고 한국이 국제법 위반 상태를 시정해야 한다는 일본 정부의 입장을 모아 제시한다.",
    },
    {
        "url": "https://www.mofa.go.jp/press/release/press4e_001920.html",
        "jurisdiction": "JP",
        "institution": "Ministry of Foreign Affairs of Japan",
        "title": "Japan's Position on the 2015 Comfort Women Agreement",
        "domain_ids": ["D05", "D08"],
        "subtopic_ids": ["victim_remedies"],
        "evidence_summary": "일본 외무성은 2015년 한일 합의로 위안부 문제가 최종적이고 불가역적으로 해결됐으며 양국이 국제사회에서 상호 비난을 자제하기로 했다는 입장을 제시한다.",
    },
    {
        "url": "https://www.mofa.go.jp/policy/q_a/faq3.html",
        "jurisdiction": "JP",
        "institution": "Ministry of Foreign Affairs of Japan",
        "title": "Recent Policy of Japan on the Comfort Women Issue",
        "domain_ids": ["D08"],
        "subtopic_ids": ["victim_remedies"],
        "evidence_summary": "일본 외무성은 고노 담화의 사과와 아시아여성기금의 국민 모금 및 정부 예산 지원을 일본의 도의적 책임 이행 방식으로 설명한다.",
    },
    {
        "url": "https://www.mofa.go.jp/press/kaiken/kaikenite_000001_00074.html",
        "jurisdiction": "JP",
        "institution": "Ministry of Foreign Affairs of Japan",
        "title": "Japan's Position on the Korea-Japan Continental Shelf Joint Development Agreement",
        "domain_ids": ["D07"],
        "subtopic_ids": ["maritime_boundaries"],
        "evidence_summary": "일본 외무상은 한일 대륙붕 공동개발협정의 유효기간과 종료 조항을 고려해 남부 대륙붕의 향후 지위를 한국 정부와 계속 협의하는 것이 중요하다는 입장을 밝혔다.",
    },
    {
        "url": "https://eu.china-mission.gov.cn/eng/more/dhfksbq13/201312/t20131217_8304113.htm",
        "jurisdiction": "CN",
        "institution": "Ministry of National Defense of the People's Republic of China",
        "title": "China Defense Ministry Statement on the East China Sea ADIZ",
        "domain_ids": ["D06", "D07"],
        "subtopic_ids": ["airspace"],
        "evidence_summary": "중국 국방부는 동중국해 방공식별구역이 영공 밖 식별·경보 구역이며 중국의 주권과 영토·영공 안전을 지키기 위한 조치로 국제법과 관행에 부합한다는 입장을 제시한다.",
    },
    {
        "url": "https://www.jilinja.gov.cn/zwgk/yw/202509/t20250925_752604.html",
        "jurisdiction": "CN",
        "institution": "Ji'an Municipal People's Government",
        "title": "Protection and Public Interpretation of Goguryeo Heritage in Ji'an",
        "domain_ids": ["D08"],
        "subtopic_ids": ["historical_interpretation"],
        "evidence_summary": "중국 지안시 정부는 고구려를 한당 시기 중국 동북지역에서 활동한 중요한 민족정권이자 중화문명 다원일체의 역사적 증거로 서술한다.",
    },
    {
        "url": "https://www.cssn.cn/skgz/bwyc/202604/t20260422_5981745.shtml",
        "jurisdiction": "CN",
        "institution": "Chinese Academy of Social Sciences",
        "title": "Archaeology and Public Understanding of the Bohai Kingdom",
        "domain_ids": ["D08"],
        "subtopic_ids": ["historical_interpretation"],
        "evidence_summary": "중국사회과학원 계열 자료는 발해를 말갈족을 주체로 중국 동북지역에 세워진 당대의 변방 민족정권으로 서술한다.",
    },
    {
        "url": "https://ustr.gov/about-us/policy-offices/press-office/fact-sheets/2018/march/new-us-trade-policy-and-national",
        "jurisdiction": "US",
        "institution": "Office of the United States Trade Representative",
        "title": "New U.S. Trade Policy and National Security Outcomes with the Republic of Korea",
        "domain_ids": ["E02", "E03"],
        "subtopic_ids": ["tariffs", "steel_trade", "industrial_policy"],
        "evidence_summary": "미국 무역대표부는 한국산 철강의 무관세 예외를 미국 국가안보 우려를 해소하기 위한 2015~2017년 평균 수입량의 70% 쿼터와 결합하고, 이를 한국산 철강 반입의 상당한 감축으로 설명한다.",
    },
    {
        "url": "https://ustr.gov/about-us/policy-offices/press-office/press-releases/2018/march/joint-statement-united-states-trade",
        "jurisdiction": "US",
        "institution": "Office of the United States Trade Representative",
        "title": "Joint Statement on KORUS FTA Amendments and Korea's Section 232 Steel Arrangement",
        "domain_ids": ["E03"],
        "subtopic_ids": ["tariffs", "trade_negotiation", "steel_trade"],
        "evidence_summary": "미국 무역대표부와 한국 산업통상자원부의 공동성명은 한미 FTA 개정과 자동차·관세·무역구제 조정, 한국산 철강의 무역확장법 232조 관세 국가면제 조건을 양국 협상 결과로 제시한다.",
    },
    {
        "url": "https://www.fmprc.gov.cn/mfa_eng/xw/fyrbt/lxjzh/202405/t20240530_11347345.html",
        "jurisdiction": "CN",
        "institution": "Ministry of Foreign Affairs of the People's Republic of China",
        "title": "China's Position on THAAD Deployment in the ROK",
        "domain_ids": ["S08"],
        "subtopic_ids": ["missile_defense", "alliance_security"],
        "evidence_summary": "중국 외교부는 한국의 주권과 안보 우려를 존중한다고 밝히면서도 주한미군 사드 체계가 중국의 전략안보를 훼손하므로 한국이 양국 간 기존 이해에 따라 사안을 처리해야 한다는 입장을 제시한다.",
    },
    {
        "url": "https://www.bundesverfassungsgericht.de/SharedDocs/Pressemitteilungen/EN/2017/bvg17-004.html",
        "jurisdiction": "DE",
        "institution": "Federal Constitutional Court of Germany",
        "title": "No Prohibition of the NPD Without Indications of Successful Anti-Constitutional Action",
        "domain_ids": ["P03"],
        "subtopic_ids": ["party_dissolution", "militant_democracy"],
        "evidence_summary": "독일 연방헌법재판소는 NPD의 목표와 활동이 자유민주적 기본질서에 반하지만 그 목표를 성공시킬 구체적이고 중대한 가능성이 부족하다는 이유로 정당금지 청구를 기각했다.",
    },
    {
        "url": "https://www.bundesregierung.de/breg-en/service/archive/compensation-for-nuclear-phaseout-1881422",
        "jurisdiction": "DE",
        "institution": "Federal Government of Germany",
        "title": "Compensation Agreement for Germany's Nuclear Phase-Out",
        "domain_ids": ["N06"],
        "subtopic_ids": ["nuclear_energy", "energy_transition"],
        "evidence_summary": "독일 연방정부는 상업적 원자력 이용을 종료하는 탈원전 결정을 전제로 잔여 발전량 손실에 대한 전력기업 보상 합의를 설명하며 탈원전 정책의 법적·재정적 이행을 공식화한다.",
    },
    {
        "url": "https://www.motie.go.kr/kor/article/ATCL3f49a5a8c/170183/view",
        "jurisdiction": "KR",
        "institution": "Ministry of Trade, Industry and Energy of the Republic of Korea",
        "title": "The 11th Basic Plan for Long-Term Electricity Supply and Demand",
        "domain_ids": ["N06"],
        "subtopic_ids": ["nuclear_energy", "electricity_supply", "energy_security"],
        "evidence_summary": "산업통상자원부가 2025년 2월 확정한 제11차 전력수급기본계획은 설계수명 만료 원전의 계속운전을 전제로 하고 신규 대형원전 2기와 SMR 1기를 반영해 원전을 무탄소 전원으로 계속 활용한다.",
    },
    {
        "url": "https://english.mofcom.gov.cn/News/SpokesmansRemarks/art/2025/art_c202dcc0433d476db52b1e7f7fe53926.html",
        "jurisdiction": "CN",
        "institution": "Ministry of Commerce of the People's Republic of China",
        "title": "MOFCOM Remarks on China's Rare Earth Export Controls",
        "domain_ids": ["E11"],
        "subtopic_ids": ["export_controls", "critical_minerals", "supply_chain_security"],
        "evidence_summary": "중국 상무부는 희토류 수출통제가 국가안보와 비확산 의무를 위한 합법적 조치이며 수출금지가 아니고 적격 민간용 신청에는 허가를 부여한다는 공식 입장을 제시한다.",
    },
    {
        "url": "https://english.www.gov.cn/news/202403/23/content_WS65fe0f84c6d0868f4e8e5612.html",
        "jurisdiction": "CN",
        "institution": "State Council of the People's Republic of China",
        "title": "China's Regulations on Cross-Border Data Flows",
        "domain_ids": ["Q07"],
        "subtopic_ids": ["data_localization", "cross_border_data", "data_security"],
        "evidence_summary": "중국 국무원 자료는 중요데이터와 일정 규모 이상의 개인정보를 국외로 제공하는 사업자 및 핵심정보기반시설 운영자에게 국외이전 안전평가 신고를 요구하는 중국의 데이터 안보 체계를 설명한다.",
    },
    {
        "url": "https://un.china-mission.gov.cn/eng/fyrth/202401/t20240129_11235342.htm",
        "jurisdiction": "CN",
        "institution": "Permanent Mission of the People's Republic of China to the United Nations",
        "title": "China's Position on Repatriation of DPRK Border Crossers",
        "domain_ids": ["S20", "D08"],
        "subtopic_ids": ["non_refoulement", "dprk_human_rights", "border_control"],
        "evidence_summary": "중국 외교부는 중국에 경제적 이유로 불법 입국한 북한 주민을 난민이 아닌 불법 입국자로 규정하고 국내법·국제법과 인도주의 원칙에 따라 처리한다는 공식 입장을 제시한다.",
    },
    {
        "url": "https://un.china-mission.gov.cn/eng/hyyfy/202205/t20220527_10693323.htm",
        "jurisdiction": "CN",
        "institution": "Permanent Mission of the People's Republic of China to the United Nations",
        "title": "China's Explanation of Vote on Additional DPRK Sanctions",
        "domain_ids": ["D05", "D06"],
        "subtopic_ids": ["dprk_sanctions", "humanitarian_exemptions", "diplomacy"],
        "evidence_summary": "중국 유엔대표부는 대북 제재만을 일방적으로 강조해서는 안 되며 인도주의·민생 피해를 줄이고 정치적 해결을 촉진하도록 제재를 적절히 완화해야 한다는 입장을 제시한다.",
    },
)


OFFICIAL_DOMAINS = {
    "mofa.go.jp",
    "www.mofa.go.jp",
    "www.cas.go.jp",
    "www.gov.cn",
    "eu.china-mission.gov.cn",
    "www.jilinja.gov.cn",
    "www.cssn.cn",
    "ustr.gov",
    "www.ustr.gov",
    "dokdo.mofa.go.kr",
    "www.fmprc.gov.cn",
    "www.bundesverfassungsgericht.de",
    "www.bundesregierung.de",
    "www.motie.go.kr",
    "www.motir.go.kr",
    "english.mofcom.gov.cn",
    "english.www.gov.cn",
    "un.china-mission.gov.cn",
    "www.mct.gov.cn",
    "www.neac.gov.cn",
}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.suppressed = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"}:
            self.suppressed += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"} and self.suppressed:
            self.suppressed -= 1

    def handle_data(self, data: str) -> None:
        if not self.suppressed and data.strip():
            self.parts.append(data.strip())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value.strip())
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query = [(key, item) for key, item in query if not key.casefold().startswith("utm_")]
    return urllib.parse.urlunsplit(
        (parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path, urllib.parse.urlencode(query), "")
    ).rstrip("/")


def benchmark_urls(paths: list[Path]) -> set[str]:
    return {
        canonical_url(str(source.get("url") or ""))
        for path in paths
        for row in load_jsonl(path)
        for source in row.get("sources") or []
        if isinstance(source, dict) and str(source.get("url") or "").strip()
    }


def retrieve(spec: dict[str, Any], timeout: int) -> dict[str, Any]:
    url = str(spec["url"])
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc.casefold() not in OFFICIAL_DOMAINS:
        raise ValueError(f"non-official URL: {url}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "K-SOVA-official-source-audit/1.0",
            "Accept-Language": "en,ja,zh-CN,ko;q=0.8",
        },
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        body = response.read(2_000_000)
        status = int(response.status)
        final_url = response.geturl()
        content_type = str(response.headers.get("Content-Type") or "")
    charset_match = re.search(r"charset=([^;\s]+)", content_type, re.IGNORECASE)
    candidates = [charset_match.group(1) if charset_match else "", "utf-8", "shift_jis", "gb18030"]
    decoded = ""
    for encoding in candidates:
        if not encoding:
            continue
        try:
            decoded = body.decode(encoding)
            break
        except (LookupError, UnicodeDecodeError):
            continue
    if not decoded:
        decoded = body.decode("utf-8", errors="replace")
    extractor = TextExtractor()
    extractor.feed(decoded)
    text = re.sub(r"\s+", " ", " ".join(extractor.parts)).strip()
    if status != 200 or len(text) < 300:
        raise ValueError(f"retrieval failed status={status} text_chars={len(text)} url={url}")
    normalized_url = canonical_url(final_url)
    return {
        "authority_type": "government_position_or_public_institution",
        "content_excerpt": text[:6000],
        "content_sha256_prefix_1m": hashlib.sha256(body[:1_000_000]).hexdigest(),
        "content_type": content_type,
        "domain_ids": spec["domain_ids"],
        "evidence_summary": spec["evidence_summary"],
        "final_url": final_url,
        "http_status": status,
        "institution": spec["institution"],
        "jurisdiction": spec["jurisdiction"],
        "official": True,
        "retrieval_status": "reachable",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "retrieved_bytes": len(body),
        "source_id": "src_" + hashlib.sha256(normalized_url.encode()).hexdigest()[:14],
        "subtopic_ids": spec["subtopic_ids"],
        "title": spec["title"],
        "url": url,
        "canonical_url": normalized_url,
        "authority_tier": "T2_government_policy",
        "authority_tier_rule": "official_government_or_public_institution_position",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, action="append", required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    base = load_jsonl(args.base)
    base_urls = {canonical_url(str(row.get("canonical_url") or row.get("url") or "")) for row in base}
    blocked_urls = benchmark_urls(args.benchmark)
    extension = []
    failures = []
    for spec in SOURCE_SPECS:
        url = canonical_url(str(spec["url"]))
        if url in base_urls:
            failures.append({"url": url, "reason": "already_in_base"})
            continue
        if url in blocked_urls:
            failures.append({"url": url, "reason": "benchmark_url_overlap"})
            continue
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                extension.append(retrieve(spec, args.timeout))
                last_error = None
                break
            except Exception as exc:  # Failures are reported and the final gate remains fail closed.
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        if last_error is not None:
            failures.append({
                "url": url,
                "reason": f"retrieval_error:{type(last_error).__name__}:{last_error}",
            })

    by_url = {
        canonical_url(str(row.get("canonical_url") or row.get("url") or "")): row
        for row in [*base, *extension]
    }
    combined = sorted(by_url.values(), key=lambda row: str(row.get("source_id") or ""))
    write_jsonl(args.extension, extension)
    write_jsonl(args.combined, combined)
    report = {
        "requested": len(SOURCE_SPECS),
        "retrieved": len(extension),
        "failures": failures,
        "base_sources": len(base),
        "combined_sources": len(combined),
        "benchmark_url_overlap": sum(
            canonical_url(str(row.get("canonical_url") or row.get("url") or "")) in blocked_urls
            for row in extension
        ),
        "official_only": all(row.get("official") is True for row in extension),
        "wikipedia_allowed": False,
        "passed": bool(extension)
        and not any(row["reason"] == "benchmark_url_overlap" for row in failures)
        and all(row.get("official") is True for row in extension),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
