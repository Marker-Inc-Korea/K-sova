#!/usr/bin/env python3
"""Summarize official-source diversity and per-domain evidence coverage."""

from __future__ import annotations

import argparse
import collections
import json
import urllib.parse
from pathlib import Path

from common import ROOT, load_json, write_json


def jurisdiction_parts(source: dict) -> set[str]:
    return {part for part in str(source.get("jurisdiction", "")).split("-") if part}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "config/case_blueprints.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/source_coverage_v0.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "reports/source_coverage_v0.md")
    args = parser.parse_args()
    cases = load_json(args.input)
    sources = [(case, source) for case in cases for source in case["sources"]]
    host_counts = collections.Counter(
        urllib.parse.urlsplit(source["url"]).netloc.lower().removeprefix("www.")
        for _, source in sources
    )
    kind_counts = collections.Counter(source["kind"] for _, source in sources)
    jurisdiction_counts = collections.Counter(source["jurisdiction"] for _, source in sources)
    domain_rows = {}
    for domain_id in sorted({case["domain_id"] for case in cases}):
        domain_cases = [case for case in cases if case["domain_id"] == domain_id]
        domain_sources = [source for case in domain_cases for source in case["sources"]]
        domain_rows[domain_id] = {
            "cases": len(domain_cases),
            "sources": len(domain_sources),
            "korean_sources": sum("KR" in jurisdiction_parts(source) for source in domain_sources),
            "comparison_sources": sum(bool(jurisdiction_parts(source) - {"KR"}) for source in domain_sources),
            "unique_hosts": len({urllib.parse.urlsplit(source["url"]).netloc.lower().removeprefix("www.") for source in domain_sources}),
        }
    wiki_sources = [
        {"case_id": case["id"], "source_id": source["id"], "url": source["url"]}
        for case, source in sources
        if "wikipedia.org" in source["url"] or "namu.wiki" in source["url"]
    ]
    report = {
        "cases": len(cases),
        "sources": len(sources),
        "sources_per_case": round(len(sources) / max(len(cases), 1), 3),
        "unique_hosts": len(host_counts),
        "host_counts": dict(sorted(host_counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "jurisdiction_counts": dict(sorted(jurisdiction_counts.items())),
        "domains": domain_rows,
        "wiki_sources": wiki_sources,
        "gates": {
            "all_cases_have_two_korean_sources": all(
                sum("KR" in jurisdiction_parts(source) for source in case["sources"]) >= 2
                for case in cases
            ),
            "all_cases_have_comparison_source": all(
                any(jurisdiction_parts(source) - {"KR"} for source in case["sources"])
                for case in cases
            ),
            "at_least_ten_official_hosts": len(host_counts) >= 10,
            "no_wiki_sources": not wiki_sources,
        },
    }
    write_json(args.output, report)
    lines = [
        "# K-SOVA v3 원천 다양성 보고서",
        "",
        f"- 사례: {report['cases']}개",
        f"- 공식 소스: {report['sources']}개(사례당 평균 {report['sources_per_case']:.2f}개)",
        f"- 고유 기관 도메인: {report['unique_hosts']}개",
        f"- 위키 소스: {len(wiki_sources)}개",
        "",
        "## 관할별",
        "",
    ]
    lines.extend(f"- {key}: {value}개" for key, value in report["jurisdiction_counts"].items())
    lines.extend(["", "## 자료 유형", ""])
    lines.extend(f"- {key}: {value}개" for key, value in report["kind_counts"].items())
    lines.extend(["", "## 기관 도메인", ""])
    lines.extend(f"- `{key}`: {value}개" for key, value in sorted(host_counts.items(), key=lambda pair: (-pair[1], pair[0])))
    lines.extend(["", "## 게이트", ""])
    lines.extend(f"- {'통과' if value else '실패'} `{key}`" for key, value in report["gates"].items())
    lines.append("")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"cases": report["cases"], "sources": report["sources"], "unique_hosts": report["unique_hosts"], "wiki_sources": len(wiki_sources)}, ensure_ascii=False))
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
