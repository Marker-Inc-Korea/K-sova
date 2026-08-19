#!/usr/bin/env python3
"""Build three source-grounded full-answer teacher requests per issue."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable


VARIANTS = ("neutral", "claim_review", "policy_or_comparison")
COMPACT_ISSUE_FIELDS = (
    "title_ko",
    "temporal_scope",
    "common_facts_ko",
    "external_positions_ko",
    "korean_discourse_ko",
    "sovereign_stake_ko",
    "decision_core_ko",
    "korean_reference_direction_ko",
    "acceptable_range_ko",
    "failure_direction_ko",
    "source_support_ko",
)
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9·.\-]{1,}")
STOPWORDS = {
    "한국", "대한민국", "판단", "여부", "관련", "대해", "대한", "위한", "통해",
    "따른", "있는", "하는", "한다", "해야", "쟁점", "정부", "국가", "경우",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_body(row: dict[str, Any]) -> str:
    retrieval = row.get("retrieval") or {}
    return str(
        row.get("content_text")
        or (retrieval.get("text_excerpt") if isinstance(retrieval, dict) else "")
        or ""
    )


def compact_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {key: issue.get(key) for key in COMPACT_ISSUE_FIELDS}


def issue_keywords(issue: dict[str, Any]) -> list[str]:
    values = [
        issue.get("title_ko"),
        issue.get("decision_core_ko"),
        issue.get("sovereign_stake_ko"),
        *(issue.get("common_facts_ko") or [])[:3],
    ]
    tokens = []
    for token in TOKEN_RE.findall(" ".join(str(value or "") for value in values)):
        token = token.strip(".-")
        if len(token) >= 3 and token not in STOPWORDS and token not in tokens:
            tokens.append(token)
    return sorted(tokens, key=lambda value: (-len(value), value))[:24]


def evidence_excerpt(body: str, keywords: list[str], limit: int = 2400) -> str:
    body = re.sub(r"[ \t]+", " ", body).strip()
    if len(body) <= limit:
        return body
    lowered = body.casefold()
    spans: list[tuple[int, int]] = []
    for keyword in keywords:
        index = lowered.find(keyword.casefold())
        if index < 0:
            continue
        start = max(0, index - 350)
        end = min(len(body), index + len(keyword) + 850)
        if any(not (end < prior_start or start > prior_end) for prior_start, prior_end in spans):
            continue
        spans.append((start, end))
        if len(spans) == 3:
            break
    if not spans:
        return body[:limit]
    pieces = [body[start:end].strip() for start, end in sorted(spans)]
    excerpt = "\n...\n".join(pieces)
    return excerpt[:limit]


def source_support_map(issue: dict[str, Any]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for row in issue.get("source_support_ko") or []:
        if not isinstance(row, dict) or not row.get("source_id"):
            continue
        claims = [
            str(value).strip() for value in row.get("supported_claims_ko") or []
            if str(value).strip()
        ]
        output[str(row["source_id"])] = claims[:8]
    return output


def compact_source(
    source: dict[str, Any], *, claims: list[str], keywords: list[str],
) -> dict[str, Any]:
    retrieval = source.get("retrieval") or {}
    declared_claims = source.get("evidence_claims") or []
    return {
        "source_id": source["source_id"],
        "title": source.get("title"),
        "publisher": source.get("publisher") or source.get("issuing_authority"),
        "jurisdiction": source.get("jurisdiction"),
        "authority_type": source.get("authority_type"),
        "reliability_tier": (
            (source.get("provenance") or {}).get("reliability_tier")
            or source.get("reliability_tier")
        ),
        "url": source.get("url") or (
            retrieval.get("final_url") if isinstance(retrieval, dict) else None
        ),
        "issue_supported_claims_ko": claims,
        "source_evidence_claims": declared_claims[:8]
        if isinstance(declared_claims, list) else [],
        "evidence_summary_ko": source.get("evidence_summary_ko"),
        "body_excerpt": evidence_excerpt(source_body(source), keywords),
    }


def system_prompt() -> str:
    return """당신은 K-SOVA 학습용 완결 답변을 작성하는 데이터 설계자다.

K-SOVA는 한국 관련 지식을 암기했는지가 아니라, 외국과 권리·이익·규범·가치가 충돌할 수 있는 사안에서 한국 사회에 형성된 담론을 바탕으로 사실, 한국의 구체적 권리·이해관계, 상대 논거, 판단과 조건·비용을 결합해 답하는지를 다룬다.

입력의 쟁점 카드와 원문 근거만 사용한다. 새로운 인명·기관·수치·날짜·법적 효과를 만들지 않는다. 정부 발표는 귀속된 하나의 논거이지 자동 정답이 아니다. 국수주의적 구호, 정부 대변 문체, 상대국·집단에 대한 일반화, 근거 없는 확정 표현을 쓰지 않는다. 짧은 중간 판단이나 체크리스트가 아니라 사용자 질문에 직접 답하는 독립된 최종 답변을 작성한다.

각 답변은 자연스러운 한국어 900~1,600자를 목표로 하며 다음을 모두 포함한다.
1. 확인된 사실과 행위자 주장의 구분
2. 한국 사회에서 방어 가능한 최소 기준선과 구체적인 권리·이해관계
3. 가장 강한 상대 논거의 실질적 검토
4. 질문에 대한 명확한 판단 또는 권고
5. 비용·위험·조건·전환 기준 및 불확실성의 경계

반드시 JSON 객체 하나만 출력하고 코드블록이나 앞뒤 설명을 쓰지 않는다."""


def user_prompt(
    dossier: dict[str, Any], evidence: list[dict[str, Any]],
) -> str:
    payload = {
        "issue_id": dossier["issue_id"],
        "domain_id": dossier["domain_id"],
        "issue_card": compact_issue(dossier["issue_card"]),
        "verified_source_evidence": evidence,
    }
    return f"""다음 하나의 쟁점에서 서로 독립적인 완결형 SFT 예시 3개를 작성하라.

입력:
{json.dumps(payload, ensure_ascii=False)}

세 예시의 역할:
- neutral: 선동적 전제 없이 쟁점의 판단을 묻는 자연스러운 중립 질의
- claim_review: 외부 행위자나 국내 일각의 강한 주장을 실제 사용자가 제시하고 그 타당성을 검토하게 하는 질의
- policy_or_comparison: 가능한 대응·정책·판단 기준을 비교하거나 조건부 권고를 요구하는 질의

질의 규칙:
- 세 질문은 표현만 바꾼 동의문이 아니라 서로 다른 판단 작업이어야 한다.
- "한국 관점에서", "한국 입장에서", "한국인이라면"처럼 정답 관점을 직접 지시하지 않는다.
- 단순 사실 회상, 법조문 맞히기, 인물·연도 퀴즈를 만들지 않는다.
- 질문만 읽어도 답할 판단 대상은 명확하되 기준답안을 노출하지 않는다.

답변 규칙:
- 세 답변은 각각 단독으로 충분해야 하며 서로를 참조하지 않는다.
- 입력 원문으로 확인되지 않은 최신 상태나 법적 결론을 추정하지 않는다.
- 같은 문단을 세 답변에 복사하지 않는다.

다음 형식을 정확히 지켜라:
{{
  "issue_id": "{dossier['issue_id']}",
  "examples": [
    {{"variant":"neutral","user_ko":"질문","assistant_ko":"완결 답변","coverage_ko":["사실·귀속","한국의 권리·이익","상대 논거","판단","조건·비용·불확실성"]}},
    {{"variant":"claim_review","user_ko":"질문","assistant_ko":"완결 답변","coverage_ko":["사실·귀속","한국의 권리·이익","상대 논거","판단","조건·비용·불확실성"]}},
    {{"variant":"policy_or_comparison","user_ko":"질문","assistant_ko":"완결 답변","coverage_ko":["사실·귀속","한국의 권리·이익","상대 논거","판단","조건·비용·불확실성"]}}
  ]
}}"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-issues", type=Path, required=True)
    parser.add_argument("--portfolio-sources", type=Path, required=True)
    parser.add_argument(
        "--source-backfills", type=Path, action="append", default=[],
        help="Resolver JSONL whose lead_ids map fetched bodies to portfolio source IDs.",
    )
    parser.add_argument("--benchmark-items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-sources-per-issue", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--only-issue-id", action="append", default=[],
        help="Emit only these issue IDs after validating the complete 150-issue portfolio.",
    )
    args = parser.parse_args()

    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite teacher request outputs")
    dossiers = read_jsonl(args.portfolio_issues)
    if len(dossiers) != 150 or len({row.get("issue_id") for row in dossiers}) != 150:
        raise ValueError("portfolio must contain 150 unique issues")
    requested_issue_ids = {str(value) for value in args.only_issue_id}
    available_issue_ids = {str(row.get("issue_id") or "") for row in dossiers}
    if requested_issue_ids - available_issue_ids:
        raise ValueError(
            f"requested issue IDs are absent: {sorted(requested_issue_ids - available_issue_ids)}"
        )
    selected_dossiers = [
        row for row in dossiers
        if not requested_issue_ids or str(row.get("issue_id") or "") in requested_issue_ids
    ]
    source_rows = read_jsonl(args.portfolio_sources)
    sources = {str(row.get("source_id") or ""): row for row in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("portfolio sources contain missing or duplicate source IDs")

    backfilled_source_ids: set[str] = set()
    for path in args.source_backfills:
        for row in read_jsonl(path):
            fetched = row.get("source") if isinstance(row.get("source"), dict) else row
            retrieval = fetched.get("retrieval") or {}
            fetched_body = source_body(fetched)
            fetched_url = str(
                retrieval.get("source_url") or fetched.get("url") or ""
            ).rstrip("/")
            for source_id in row.get("lead_ids") or [fetched.get("source_id")]:
                source_id = str(source_id or "")
                original = sources.get(source_id)
                if original is None:
                    raise ValueError(f"backfill references unknown source {source_id}")
                original_url = str(original.get("url") or "").rstrip("/")
                if not fetched_url or fetched_url != original_url:
                    raise ValueError(
                        f"backfill URL mismatch for {source_id}: "
                        f"{fetched_url!r} != {original_url!r}"
                    )
                if len(fetched_body) > len(source_body(original)):
                    sources[source_id] = {
                        **original,
                        "content_text": fetched_body,
                        "retrieval": retrieval,
                    }
                    backfilled_source_ids.add(source_id)

    benchmark_rows = read_jsonl(args.benchmark_items)
    if len(benchmark_rows) != 480:
        raise ValueError("unexpected benchmark item count")
    benchmark_hash = file_sha256(args.benchmark_items)

    requests = []
    prompt_chars = []
    source_counts = []
    short_sources_skipped: set[str] = set()
    by_domain: Counter[str] = Counter()
    for dossier in sorted(selected_dossiers, key=lambda row: str(row["issue_id"])):
        issue = dossier.get("issue_card") or {}
        keywords = issue_keywords(issue)
        support = source_support_map(issue)
        available = []
        for order, source_id in enumerate(dossier.get("source_ids") or []):
            source_id = str(source_id)
            source = sources.get(source_id)
            if source is None:
                raise ValueError(f"{dossier['issue_id']}: missing source {source_id}")
            body = source_body(source)
            if len(body) < 300:
                short_sources_skipped.add(source_id)
                continue
            tier = str(
                (source.get("provenance") or {}).get("reliability_tier")
                or source.get("reliability_tier") or ""
            )
            score = (
                10 * len(support.get(source_id, []))
                + 3 * int(tier == "A")
                + min(5, len(body) // 3000)
                - order / 1000
            )
            available.append((score, source_id, source))
        if not available:
            raise ValueError(
                f"{dossier['issue_id']}: no source body reaches 300 characters"
            )
        available.sort(key=lambda row: (-row[0], row[1]))
        chosen = available[: args.max_sources_per_issue]
        evidence = [
            compact_source(
                source, claims=support.get(source_id, []), keywords=keywords
            )
            for _score, source_id, source in chosen
        ]
        message = user_prompt(dossier, evidence)
        request_id = f"ksova_full_answer_teacher__{dossier['issue_id']}"
        requests.append({
            "id": request_id,
            "messages": [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": message},
            ],
            "max_tokens": args.max_tokens,
            "metadata": {
                "purpose": "ksova_three_full_answer_sft_teacher",
                "prompt_version": "ksova-full-answer-teacher-v1",
                "experiment_id": dossier["experiment_id"],
                "issue_id": dossier["issue_id"],
                "domain_id": dossier["domain_id"],
                "origin": dossier["origin"],
                "source_ids": [source_id for _score, source_id, _source in chosen],
                "variants": list(VARIANTS),
                "expected_examples": 3,
                "benchmark_items_sha256": benchmark_hash,
            },
        })
        prompt_chars.append(sum(len(row["content"]) for row in requests[-1]["messages"]))
        source_counts.append(len(chosen))
        by_domain[str(dossier["domain_id"])] += 1

    report = {
        "schema_version": "ksova-full-answer-teacher-plan-1.0",
        "requests": len(requests),
        "portfolio_issues": len(dossiers),
        "issue_filter": sorted(requested_issue_ids),
        "expected_examples": len(requests) * 3,
        "variants": list(VARIANTS),
        "by_domain": dict(sorted(by_domain.items())),
        "source_counts": {
            "min": min(source_counts),
            "max": max(source_counts),
            "average": round(mean(source_counts), 3),
        },
        "prompt_chars": {
            "min": min(prompt_chars),
            "max": max(prompt_chars),
            "average": round(mean(prompt_chars), 1),
        },
        "benchmark_items": len(benchmark_rows),
        "benchmark_items_sha256": benchmark_hash,
        "portfolio_issues_sha256": file_sha256(args.portfolio_issues),
        "portfolio_sources_sha256": file_sha256(args.portfolio_sources),
        "source_backfill_files": [str(path) for path in args.source_backfills],
        "backfilled_source_ids": sorted(backfilled_source_ids),
        "short_sources_skipped": sorted(short_sources_skipped),
        "output": str(args.output),
    }
    write_jsonl(args.output, requests)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
