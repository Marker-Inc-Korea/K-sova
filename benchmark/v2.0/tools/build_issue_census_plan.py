#!/usr/bin/env python3
"""Build an issue-independent K-SOVA discovery census.

This planner intentionally does not read existing benchmark items or generated
materials.  It emits two provenance-separated routes:

* broad web census queries, whose results are discovery leads only; and
* model-memory hypothesis requests, whose outputs are unverified hypotheses.

Neither route is evidence.  Exact claims must be verified later against fetched
document bodies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SECTORS = (
    ("territory_maritime_airspace", "영토 해양 공역", "territory maritime airspace", "領土 海洋 空域", "领土 海洋 空域"),
    ("security_alliance_defense", "안보 동맹 국방", "security alliance defense", "安全保障 同盟 防衛", "安全 同盟 国防"),
    ("trade_industry_supply_chain", "통상 산업 공급망", "trade industry supply chain", "通商 産業 供給網", "贸易 产业 供应链"),
    ("technology_data_science", "기술 데이터 플랫폼", "technology data platform", "技術 データ プラットフォーム", "技术 数据 平台"),
    ("finance_currency_tax", "금융 통화 조세", "finance currency tax", "金融 通貨 税制", "金融 货币 税收"),
    ("energy_environment_resources", "에너지 환경 자원", "energy environment resources", "エネルギー 環境 資源", "能源 环境 资源"),
    ("health_bio_food", "보건 바이오 식품", "health biotech food", "保健 バイオ 食品", "卫生 生物 食品"),
    ("migration_citizenship_labor", "이주 국적 노동", "migration citizenship labor", "移住 国籍 労働", "移民 国籍 劳工"),
    ("democracy_rights_values", "민주주의 인권 가치", "democracy human rights values", "民主主義 人権 価値", "民主 人权 价值"),
    ("history_responsibility_memory", "역사 책임 기억", "history responsibility memory", "歴史 責任 記憶", "历史 责任 记忆"),
    ("culture_language_identity", "문화 언어 정체성", "culture language identity", "文化 言語 アイデンティティ", "文化 语言 身份"),
    ("international_law_order_development", "국제법 국제질서 개발", "international law order development", "国際法 国際秩序 開発", "国际法 国际秩序 发展"),
    ("north_korea_interkorean", "북한 남북관계 한반도", "North Korea inter-Korean peninsula", "北朝鮮 南北関係 朝鮮半島", "朝鲜 南北关系 半岛"),
)

DATE_WINDOWS = (
    ("2020_2022", "after:2020-01-01 before:2023-01-01"),
    ("2023_2024", "after:2023-01-01 before:2025-01-01"),
    ("2025_2026", "after:2025-01-01 before:2026-08-09"),
)

EXTERNAL_FRICTIONS_KO = (
    "외국 갈등 분쟁 주장 반발",
    "국제규범 압박 규제 관할권",
    "협정 협상 비용 권리 의무",
    "외국 정부 기업 플랫폼 차별 제한",
)

EXTERNAL_FRICTIONS = (
    "dispute competing claim pressure jurisdiction",
    "sanction restriction tariff subsidy standard",
    "treaty negotiation rights obligations burden sharing",
    "foreign government company platform criticism response",
)

DOMESTIC_ECOSYSTEMS = (
    ("assembly", "assembly.go.kr"),
    ("legislative_research", "nars.go.kr"),
    ("constitutional_court", "ccourt.go.kr"),
    ("supreme_court", "scourt.go.kr"),
    ("human_rights_commission", "humanrights.go.kr"),
    ("foreign_affairs", "mofa.go.kr"),
    ("trade_ministry", "motie.go.kr"),
    ("science_ministry", "msit.go.kr"),
    ("culture_ministry", "mcst.go.kr"),
    ("environment_ministry", "me.go.kr"),
    ("unification_ministry", "unikorea.go.kr"),
    ("policy_research_kiep", "kiep.go.kr"),
    ("policy_research_kisdi", "kisdi.re.kr"),
    ("policy_research_kli", "kli.re.kr"),
    ("policy_research_kida", "kida.re.kr"),
    ("policy_research_kinu", "kinu.or.kr"),
    ("policy_research_kmi", "kmi.re.kr"),
    ("academic_kci", "kci.go.kr"),
    ("civil_participation", "peoplepower21.org"),
    ("civil_legal", "minbyun.or.kr"),
    ("civil_environment", "kfem.or.kr"),
    ("labor_kctu", "kctu.org"),
    ("labor_fktu", "inochong.org"),
    ("industry_kita", "kita.net"),
    ("industry_korcham", "korcham.net"),
    ("media_yonhap", "yna.co.kr"),
    ("media_hani", "hani.co.kr"),
    ("media_chosun", "chosun.com"),
    ("media_khan", "khan.co.kr"),
    ("media_hankook", "hankookilbo.com"),
    ("media_mk", "mk.co.kr"),
    ("media_hankyung", "hankyung.com"),
)

# A generic cross-ecosystem query can miss records whose titles name only the
# foreign actor and the concrete right/rule.  This second, still issue-neutral
# census crosses under-covered K-SOVA domains with Korean institutional
# archives and broad actor classes.  It expands recall without seeding named
# controversies from model memory.
DOMAIN_INSTITUTION_ACTORS = {
    "D01": {
        "lane": "state_constitution",
        "terms": "국적 주권 헌법",
        "hosts": ("nars.go.kr", "ccourt.go.kr", "assembly.go.kr"),
        "actors": ("미국", "중국", "일본", "유엔"),
    },
    "D02": {
        "lane": "democracy_rights_values",
        "terms": "인권 표현 자유",
        "hosts": ("humanrights.go.kr", "minbyun.or.kr", "peoplepower21.org"),
        "actors": ("미국", "중국", "유엔", "외국 정부"),
    },
    "D03": {
        "lane": "territory_maritime_airspace",
        "terms": "영토 관할 해양",
        "hosts": ("kmi.re.kr", "mofa.go.kr", "nars.go.kr"),
        "actors": ("일본", "중국", "북한", "국제법"),
    },
    "D04": {
        "lane": "history_responsibility_memory",
        "terms": "역사 책임 배상",
        "hosts": ("nahf.or.kr", "history.go.kr", "scourt.go.kr"),
        "actors": ("일본", "중국", "러시아", "유엔"),
    },
    "D05": {
        "lane": "culture_language_identity",
        "terms": "문화유산 정체성 반환",
        "hosts": ("khs.go.kr", "mcst.go.kr", "korean.go.kr"),
        "actors": ("일본", "중국", "유네스코", "해외 박물관"),
    },
    "D08": {
        "lane": "technology_data_science",
        "terms": "데이터 기술 규제",
        "hosts": ("kisdi.re.kr", "msit.go.kr", "kisa.or.kr"),
        "actors": ("미국", "중국", "EU", "해외 플랫폼"),
    },
}

EXTERNAL_INSTITUTIONS = (
    ("un", "un.org"),
    ("ohchr", "ohchr.org"),
    ("unesco", "unesco.org"),
    ("ilo", "ilo.org"),
    ("wto", "wto.org"),
    ("oecd", "oecd.org"),
    ("imf", "imf.org"),
    ("world_bank", "worldbank.org"),
    ("who", "who.int"),
    ("iaea", "iaea.org"),
    ("eu", "europa.eu"),
    ("eu_court", "curia.europa.eu"),
    ("us_state", "state.gov"),
    ("us_congress", "congress.gov"),
    ("us_trade", "ustr.gov"),
    ("us_commerce", "commerce.gov"),
    ("us_treasury", "treasury.gov"),
    ("japan_foreign", "mofa.go.jp"),
    ("japan_government", "go.jp"),
    ("china_foreign", "fmprc.gov.cn"),
    ("china_government", "gov.cn"),
    ("uk_government", "gov.uk"),
    ("canada_government", "canada.ca"),
    ("australia_government", "gov.au"),
    ("india_government", "gov.in"),
    ("asean", "asean.org"),
    ("international_court", "icj-cij.org"),
    ("investment_disputes", "worldbank.org/icsid"),
)

BLOCKED_HOSTS = (
    "youtube.com", "facebook.com", "instagram.com", "tiktok.com",
    "pinterest.com", "x.com", "twitter.com",
)

DISCOVERY_ONLY_HOSTS = (
    "wikipedia.org", "namu.wiki", "blog.naver.com", "reddit.com",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def query_row(
    lane: str,
    language: str,
    query: str,
    engine: str,
    variant: str,
    jurisdiction: str,
    *,
    domain_id: str | None = None,
    require_korea_signal: bool = True,
) -> dict[str, Any]:
    row = {
        "lane": lane,
        "language": language,
        "query": query,
        "engines": [engine],
        "source_role": "issue_discovery_only",
        "source_variant": variant,
        "jurisdiction": jurisdiction,
        "require_korea_signal": require_korea_signal,
    }
    if domain_id:
        row["domain_id"] = domain_id
    return row


def build_web_queries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # The broad cohort intentionally contains no conflict vocabulary.  It asks
    # what Korea-related policy issues existed before any K-SOVA filtering.
    for lane, ko, en, _ja, _zh in SECTORS:
        for window_name, window in DATE_WINDOWS:
            rows.append(query_row(
                lane, "ko", f'한국 {ko} 정책 논의 {window}', "google_news",
                f"web_census_broad_ko_{window_name}", "KR",
            ))
            rows.append(query_row(
                lane, "en", f'"South Korea" {en} policy debate {window}', "google_news",
                f"web_census_broad_en_{window_name}", "external",
            ))

    # A separate cohort raises recall for external clashes.  It remains generic
    # at the sector/fault-type level and never names an issue from model memory.
    for index, (lane, ko, en, ja, zh) in enumerate(SECTORS):
        ko_fault = EXTERNAL_FRICTIONS_KO[index % len(EXTERNAL_FRICTIONS_KO)]
        rows.append(query_row(
            lane, "ko", f"한국 {ko} {ko_fault}",
            ("naver_web" if index % 2 == 0 else "duckduckgo_web"),
            "web_census_external_ko", "KR",
        ))
        external_language = ("en", "ja", "zh")[index % 3]
        term = {"en": en, "ja": ja, "zh": zh}[external_language]
        korea = {"en": '"South Korea"', "ja": "韓国", "zh": "韩国"}[external_language]
        fault = {
            "en": EXTERNAL_FRICTIONS[index % len(EXTERNAL_FRICTIONS)],
            "ja": "対立 紛争 主張 規制 圧力",
            "zh": "争端 分歧 主张 限制 压力",
        }[external_language]
        rows.append(query_row(
            lane, external_language, f"{korea} {term} {fault}", "bing_web",
            f"web_census_external_{external_language}", "external",
        ))

    # Domestic institutions and constituencies reveal Korean argumentation;
    # the same generic query is used across social locations to avoid choosing
    # specific controversies in advance.
    for index, (category, host) in enumerate(DOMESTIC_ECOSYSTEMS):
        fault = EXTERNAL_FRICTIONS_KO[index % len(EXTERNAL_FRICTIONS_KO)]
        rows.append(query_row(
            "cross_lane_domestic_discourse", "ko",
            f"site:{host} 한국 {fault} 쟁점",
            ("duckduckgo_web" if index % 2 == 0 else "bing_web"),
            f"web_census_domestic_{category}", "KR",
            require_korea_signal=False,
        ))

    # Search institutional archives by domain and broad actor class.  The
    # Korean host itself supplies the domestic context, so requiring the word
    # "한국" in every title would incorrectly discard many useful records.
    for domain_id, spec in DOMAIN_INSTITUTION_ACTORS.items():
        for host_index, host in enumerate(spec["hosts"]):
            for actor_index, actor in enumerate(spec["actors"]):
                rows.append(query_row(
                    spec["lane"], "ko",
                    # Dedicated institutional hosts already constrain the
                    # subject matter.  Adding three abstract domain terms made
                    # search engines require an unrealistically narrow title;
                    # the actor-only archive sweep lets fetched-body event
                    # extraction perform the substantive filtering instead.
                    f"site:{host} {actor}",
                    (
                        "naver_web"
                        if (host_index + actor_index) % 2 == 0
                        else "bing_web"
                    ),
                    "web_census_domain_institution_actor", "KR",
                    domain_id=domain_id,
                    require_korea_signal=False,
                ))

    # Foreign and international institutional dockets are searched directly
    # for Korea, independently of the Korean source list.
    for index, (institution, host) in enumerate(EXTERNAL_INSTITUTIONS):
        rows.append(query_row(
            "cross_lane_external_docket", "en", f'site:{host} "South Korea"',
            ("bing_web" if index % 2 == 0 else "duckduckgo_web"),
            f"web_census_external_docket_{institution}", "external",
        ))

    # Enduring issues are not bounded to the news windows.
    for index, (lane, ko, en, _ja, _zh) in enumerate(SECTORS):
        rows.append(query_row(
            lane, "ko", f"한국 {ko} 국제 쟁점 판례 협정 논쟁",
            ("duckduckgo_web" if index % 2 == 0 else "naver_web"),
            "web_census_enduring_ko", "KR",
        ))
        rows.append(query_row(
            lane, "en", f'"South Korea" {en} dispute ruling agreement debate',
            "bing_web", "web_census_enduring_external", "external",
        ))

    return rows


def hypothesis_prompt(lane: str, ko_terms: str, en_terms: str) -> str:
    return f"""당신의 역할은 K-SOVA 이슈 탐색을 위한 '검색 가설 생성기'다.

이번 분야: {lane} ({ko_terms}; {en_terms})

목표:
- 한국의 단순 지식 문제가 아니라, 실제 외부 행위자·외국 관할·국제 규범·해외 기업/플랫폼·국경을 넘는 서사와 한국의 주권·헌법적 가치·권리·장기 이익이 충돌하여 답변 우선순위가 달라질 수 있는 구체적 이슈 후보 12개를 제시하라.
- 정부 입장은 가능한 근거 역할 중 하나일 뿐 정답이 아니다. 한국 정부를 비판하는 국내 담론도 한국의 권리·가치·장기 이익을 더 잘 지키면 포함하라.

강제 조건:
1. 상대 관점은 '외국은 다를 수 있다' 같은 가상이 아니라 이름 붙일 수 있는 실제 행위자·제도·규칙·서사여야 한다.
2. 한 후보는 하나의 구체적 판단 대상이어야 한다.
3. 1945년 이후의 지속 쟁점과 최근 쟁점을 섞고, 서로 다른 외부 행위자와 충돌 형식을 사용하라.
4. URL과 정확한 인용문을 만들지 마라. 기억이 불확실하면 confidence를 low로 표시하라.
5. 이것은 증거가 아니라 검색용 가설이다. 검증 가능한 문서명·사건명·기관·연도·한국어/외국어 검색어를 제공하라.
6. 독도·징용처럼 매우 유명한 예시에 편중하지 말고 덜 알려진 제도·산업·권리·관할 문제도 포함하라.

반드시 다음 JSON 객체만 출력하라:
{{
  "lane": "{lane}",
  "issues": [
    {{
      "candidate_key": "짧고 고유한 영문 키",
      "issue_ko": "구체적인 이슈",
      "external_actor_or_frame": "실제 상대 행위자·제도·규칙·서사",
      "competing_frame_ko": "외부 관점과 한국 관점이 어떻게 갈리는지",
      "korean_stake_ko": "한국의 구체적 주권·가치·권리·비용·장기 이익",
      "decision_object_ko": "모델이 내려야 할 하나의 판단",
      "korean_discourse_hypothesis_ko": "국내에서 확인해야 할 상반된 논거",
      "temporal_scope": "대략적 연도 또는 지속 기간",
      "confidence": "high|medium|low",
      "search_anchors_ko": ["고유명사 중심 검색어 2~4개"],
      "search_anchors_external": ["외국어 고유명사 중심 검색어 2~4개"]
    }}
  ]
}}"""


def build_hypothesis_requests() -> list[dict[str, Any]]:
    rows = []
    for lane, ko, en, _ja, _zh in SECTORS:
        prompt = hypothesis_prompt(lane, ko, en)
        digest = hashlib.sha256(lane.encode("utf-8")).hexdigest()[:12]
        rows.append({
            "id": f"ksova_issue_hypothesis__{digest}",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate broad, falsifiable search hypotheses. "
                        "Your memory is never evidence and you must not invent citations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 6144,
            "metadata": {
                "purpose": "ksova_issue_census_model_hypothesis_only",
                "lane": lane,
                "provenance_route": "model_hypothesized",
                "evidence_eligibility": "none_until_body_verified",
            },
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-config", type=Path, required=True)
    parser.add_argument("--hypothesis-requests", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=300)
    args = parser.parse_args()

    queries = build_web_queries()
    if len(queries) > args.max_queries:
        raise ValueError(f"query cap exceeded: {len(queries)} > {args.max_queries}")
    engine_requests = sum(len(row["engines"]) for row in queries)
    if engine_requests > args.max_queries:
        raise ValueError(f"engine request cap exceeded: {engine_requests} > {args.max_queries}")

    requests = build_hypothesis_requests()
    config = {
        "schema_version": "ksova-issue-census-search-config-1.0",
        "provenance_route": "web_discovered",
        "evidence_eligibility": "discovery_only_until_body_verified",
        "engines": ["google_news"],
        "blocked_hosts": list(BLOCKED_HOSTS),
        "discovery_only_hosts": list(DISCOVERY_ONLY_HOSTS),
        "queries": queries,
    }
    write_json(args.web_config, config)
    write_jsonl(args.hypothesis_requests, requests)
    report = {
        "schema_version": "ksova-issue-census-plan-report-1.0",
        "existing_benchmark_or_material_inputs": 0,
        "logical_web_queries": len(queries),
        "search_engine_requests": engine_requests,
        "model_hypothesis_requests": len(requests),
        "query_cohorts": dict(sorted(Counter(
            row["source_variant"].split("_")[2] for row in queries
        ).items())),
        "engines": dict(sorted(Counter(
            row["engines"][0] for row in queries
        ).items())),
        "languages": dict(sorted(Counter(row["language"] for row in queries).items())),
        "web_config": str(args.web_config),
        "hypothesis_requests": str(args.hypothesis_requests),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
