#!/usr/bin/env python3
"""Prepare and parse independent LLM audits of full-answer K-SOVA SFT rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


ISSUE_FIELDS = (
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
)
CHECKS = (
    "source_grounding_and_attribution",
    "ksova_construct_fit",
    "korean_rights_and_interests",
    "strongest_counterargument",
    "clear_judgment_or_recommendation",
    "conditions_costs_risks_uncertainty",
    "complete_natural_answer",
    "no_nationalist_or_government_mimic",
    "question_has_no_answer_cue",
    "three_tasks_are_substantively_distinct",
)


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


def parse_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("validator response is not a JSON object")
    return payload


def source_body(row: dict[str, Any]) -> str:
    retrieval = row.get("retrieval") or {}
    return str(
        row.get("content_text")
        or (retrieval.get("text_excerpt") if isinstance(retrieval, dict) else "")
        or ""
    )


def overlay_backfills(
    sources: dict[str, dict[str, Any]], paths: list[Path],
) -> set[str]:
    used: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            fetched = row.get("source") if isinstance(row.get("source"), dict) else row
            retrieval = fetched.get("retrieval") or {}
            body = source_body(fetched)
            fetched_url = str(
                retrieval.get("source_url") or fetched.get("url") or ""
            ).rstrip("/")
            for source_id in row.get("lead_ids") or [fetched.get("source_id")]:
                source_id = str(source_id or "")
                original = sources.get(source_id)
                if original is None:
                    raise ValueError(f"backfill references unknown source {source_id}")
                if fetched_url != str(original.get("url") or "").rstrip("/"):
                    raise ValueError(f"backfill URL mismatch for {source_id}")
                if len(body) > len(source_body(original)):
                    sources[source_id] = {
                        **original,
                        "content_text": body,
                        "retrieval": retrieval,
                    }
                    used.add(source_id)
    return used


def evidence_excerpt(value: str, limit: int = 1600) -> str:
    compact = re.sub(r"[ \t]+", " ", value).strip()
    return compact[:limit]


def validator_system_prompt() -> str:
    return """당신은 K-SOVA 완결형 학습데이터의 독립 감사자다. 후보를 고치거나 선의로 보완하지 말고 입력된 쟁점 카드와 검증 원천 범위에서 각 예시를 통과 또는 탈락시킨다.

K-SOVA는 단순 한국 지식이 아니라 외국과 권리·이익·규범·가치가 충돌할 수 있는 사안에서 한국 사회에 형성된 담론을 바탕으로 사실, 한국의 구체적 권리·이해관계, 상대 논거, 판단, 조건·비용을 완결된 답변으로 구성하는 능력을 다룬다. 정부 입장을 흉내 내거나 민족주의적 구호를 쓰는 것은 통과 사유가 아니다.

문체 취향이나 사소한 표현 때문에 탈락시키지 않는다. 그러나 근거 없는 구체 사실, 행위자 주장과 사실의 혼동, 한국의 권리·이익 누락, 가장 강한 상대 논거 회피, 판단 회피, 조건·비용·불확실성 누락, 관점을 정답처럼 지시하는 질문, 사실 암기형 질문, 짧은 중간 판단 형식은 탈락시킨다. 세 질문이 표현만 바꾼 동의문이면 관련 예시를 탈락시킨다.

반드시 JSON 객체 하나만 출력하고 코드블록이나 앞뒤 설명을 쓰지 않는다."""


def validator_user_prompt(
    issue_id: str,
    issue_card: dict[str, Any],
    evidence: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    candidates = []
    for example in examples:
        messages = example.get("messages") or []
        if len(messages) != 2:
            raise ValueError(f"{example.get('example_id')}: expected two messages")
        candidates.append({
            "example_id": example["example_id"],
            "variant": example["variant"],
            "user_ko": messages[0]["content"],
            "assistant_ko": messages[1]["content"],
        })
    payload = {
        "issue_id": issue_id,
        "issue_card": {key: issue_card.get(key) for key in ISSUE_FIELDS},
        "verified_source_evidence": evidence,
        "candidate_examples": candidates,
    }
    checks = json.dumps(list(CHECKS), ensure_ascii=False)
    return f"""다음 후보들을 서로 비교하고 각각 감사하라.

입력:
{json.dumps(payload, ensure_ascii=False)}

허용되는 failed_checks 값은 다음뿐이다:
{checks}

모든 기준을 실질적으로 충족할 때만 pass로 판정한다. pass이면 failed_checks는 빈 배열이어야 한다. 다음 형식을 정확히 지켜라:
{{
  "issue_id": "{issue_id}",
  "judgments": [
    {{"example_id":"후보 ID","verdict":"pass 또는 reject","failed_checks":[],"reason_ko":"한두 문장 근거"}}
  ]
}}"""


def prepare(args: argparse.Namespace) -> int:
    outputs = (args.output, args.report)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")
    examples = read_jsonl(args.examples)
    dossiers = read_jsonl(args.portfolio_issues)
    source_rows = read_jsonl(args.portfolio_sources)
    if len({row.get("example_id") for row in examples}) != len(examples):
        raise ValueError("duplicate or missing example IDs")
    issue_lookup = {str(row.get("issue_id") or ""): row for row in dossiers}
    sources = {str(row.get("source_id") or ""): row for row in source_rows}
    if len(issue_lookup) != len(dossiers) or len(sources) != len(source_rows):
        raise ValueError("duplicate portfolio issue or source IDs")
    backfilled = overlay_backfills(sources, args.source_backfills)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        issue_id = str(example.get("issue_id") or "")
        if issue_id not in issue_lookup:
            raise ValueError(f"example references unknown issue {issue_id}")
        grouped[issue_id].append(example)

    requested_issue_ids = {str(value) for value in args.only_issue_id}
    missing_issue_ids = requested_issue_ids - set(grouped)
    if missing_issue_ids:
        raise ValueError(
            f"requested issue IDs are absent: {sorted(missing_issue_ids)}"
        )
    if requested_issue_ids:
        grouped = defaultdict(
            list,
            {
                issue_id: rows
                for issue_id, rows in grouped.items()
                if issue_id in requested_issue_ids
            },
        )

    requests = []
    prompt_chars = []
    for issue_id in sorted(grouped):
        dossier = issue_lookup[issue_id]
        rows = sorted(grouped[issue_id], key=lambda row: str(row["variant"]))
        source_ids = list(dict.fromkeys(
            source_id
            for row in rows
            for source_id in row.get("source_ids") or []
        ))
        evidence = []
        for source_id in source_ids[:4]:
            source = sources.get(str(source_id))
            if source is None or len(source_body(source)) < 300:
                continue
            evidence.append({
                "source_id": source_id,
                "title": source.get("title"),
                "publisher": source.get("publisher") or source.get("issuing_authority"),
                "url": source.get("url"),
                "excerpt": evidence_excerpt(source_body(source)),
            })
        if not evidence:
            raise ValueError(f"{issue_id}: no usable validation evidence")
        user = validator_user_prompt(
            issue_id, dossier.get("issue_card") or {}, evidence, rows
        )
        request = {
            "id": f"ksova_full_answer_validator__{issue_id}",
            "messages": [
                {"role": "system", "content": validator_system_prompt()},
                {"role": "user", "content": user},
            ],
            "max_tokens": args.max_tokens,
            "metadata": {
                "purpose": "ksova_full_answer_sft_independent_validation",
                "prompt_version": "ksova-full-answer-validator-v1",
                "issue_id": issue_id,
                "domain_id": dossier.get("domain_id"),
                "example_ids": [row["example_id"] for row in rows],
                "source_ids": [row["source_id"] for row in evidence],
            },
        }
        requests.append(request)
        prompt_chars.append(sum(len(row["content"]) for row in request["messages"]))
    report = {
        "schema_version": "ksova-full-answer-validation-plan-1.0",
        "requests": len(requests),
        "examples": sum(len(rows) for rows in grouped.values()),
        "issues": len(grouped),
        "backfilled_sources": sorted(backfilled),
        "prompt_chars": {
            "min": min(prompt_chars) if prompt_chars else 0,
            "max": max(prompt_chars) if prompt_chars else 0,
            "average": round(sum(prompt_chars) / len(prompt_chars), 1) if prompt_chars else 0,
        },
        "hashes": {
            "examples_sha256": file_sha256(args.examples),
            "portfolio_issues_sha256": file_sha256(args.portfolio_issues),
            "portfolio_sources_sha256": file_sha256(args.portfolio_sources),
        },
        "output": str(args.output),
    }
    write_jsonl(args.output, requests)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def parse(args: argparse.Namespace) -> int:
    outputs = (args.accepted, args.rejected, args.report)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")
    examples = read_jsonl(args.examples)
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    example_lookup = {str(row.get("example_id") or ""): row for row in examples}
    request_lookup = {str(row.get("id") or ""): row for row in requests}
    response_lookup = {str(row.get("id") or ""): row for row in responses}
    if (
        len(example_lookup) != len(examples)
        or len(request_lookup) != len(requests)
        or len(response_lookup) != len(responses)
    ):
        raise ValueError("duplicate example, request, or response IDs")
    requested_example_ids = [
        str(example_id_value)
        for request in requests
        for example_id_value in (request.get("metadata") or {}).get("example_ids") or []
    ]
    if (
        len(requested_example_ids) != len(set(requested_example_ids))
        or set(requested_example_ids) != set(example_lookup)
    ):
        raise ValueError("validation requests do not partition the example set")
    accepted_by_id: dict[str, dict[str, Any]] = {}
    rejected_rows: list[dict[str, Any]] = []
    invalid_response_counts: Counter[str] = Counter()
    failed_checks: Counter[str] = Counter()
    for request in requests:
        request_id = str(request["id"])
        metadata = request.get("metadata") or {}
        issue_id = str(metadata.get("issue_id") or "")
        expected_ids = {str(value) for value in metadata.get("example_ids") or []}
        response = response_lookup.get(request_id)
        failure = None
        payload = None
        if response is None:
            failure = "missing_response"
        elif response.get("error"):
            failure = "response_error"
        elif str(response.get("finish_reason") or "").casefold() != "stop":
            failure = "finish_reason_not_stop"
        else:
            try:
                payload = parse_object(str(response.get("response") or ""))
            except (json.JSONDecodeError, ValueError):
                failure = "json_parse_failure"
        judgments: dict[str, dict[str, Any]] = {}
        if not failure:
            if str(payload.get("issue_id") or "") != issue_id:
                failure = "issue_id_mismatch"
            elif not isinstance(payload.get("judgments"), list):
                failure = "judgments_not_list"
            else:
                judgments = {
                    str(row.get("example_id") or ""): row
                    for row in payload["judgments"] if isinstance(row, dict)
                }
                if set(judgments) != expected_ids or len(judgments) != len(payload["judgments"]):
                    failure = "judgment_id_set_mismatch"
        if failure:
            invalid_response_counts[failure] += 1
            for example_id_value in sorted(expected_ids):
                rejected_rows.append({
                    "example": example_lookup[example_id_value],
                    "validation": {
                        "verdict": "reject",
                        "failed_checks": ["validator_response_invalid"],
                        "reason_ko": failure,
                        "validator_request_id": request_id,
                    },
                })
            continue
        for example_id_value in sorted(expected_ids):
            judgment = judgments[example_id_value]
            verdict = str(judgment.get("verdict") or "").casefold()
            checks = judgment.get("failed_checks")
            checks = [str(value) for value in checks] if isinstance(checks, list) else []
            invalid_checks = sorted(set(checks) - set(CHECKS))
            valid_pass = verdict == "pass" and not checks and not invalid_checks
            valid_reject = verdict == "reject" and bool(checks) and not invalid_checks
            if valid_pass:
                accepted_by_id[example_id_value] = {
                    **example_lookup[example_id_value],
                    "validation": {
                        "verdict": "pass",
                        "failed_checks": [],
                        "reason_ko": str(judgment.get("reason_ko") or ""),
                        "validator_request_id": request_id,
                        "validator_model": response.get("model"),
                        "prompt_version": metadata.get("prompt_version"),
                    },
                }
                continue
            if not valid_reject:
                checks = ["validator_schema_invalid"]
                invalid_response_counts["judgment_schema_invalid"] += 1
            failed_checks.update(checks)
            rejected_rows.append({
                "example": example_lookup[example_id_value],
                "validation": {
                    "verdict": "reject",
                    "failed_checks": checks,
                    "reason_ko": str(judgment.get("reason_ko") or ""),
                    "validator_request_id": request_id,
                    "validator_model": response.get("model"),
                },
            })
    accepted = [
        accepted_by_id[str(row.get("example_id") or "")]
        for row in examples
        if str(row.get("example_id") or "") in accepted_by_id
    ]
    judged_ids = set(accepted_by_id) | {
        str(row["example"].get("example_id") or "") for row in rejected_rows
    }
    unjudged_ids = sorted(set(example_lookup) - judged_ids)
    if unjudged_ids:
        raise ValueError(f"examples missing validation judgment: {unjudged_ids[:10]}")
    report = {
        "schema_version": "ksova-full-answer-validation-report-1.0",
        "examples": len(examples),
        "requests": len(requests),
        "responses": len(responses),
        "accepted": len(accepted),
        "rejected": len(rejected_rows),
        "accepted_by_domain": dict(sorted(Counter(
            str(row.get("domain_id") or "") for row in accepted
        ).items())),
        "accepted_by_variant": dict(sorted(Counter(
            str(row.get("variant") or "") for row in accepted
        ).items())),
        "invalid_response_counts": dict(sorted(invalid_response_counts.items())),
        "failed_checks": dict(sorted(failed_checks.items())),
        "hashes": {
            "examples_sha256": file_sha256(args.examples),
            "requests_sha256": file_sha256(args.requests),
            "responses_sha256": file_sha256(args.responses),
        },
    }
    write_jsonl(args.accepted, accepted)
    write_jsonl(args.rejected, rejected_rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--examples", type=Path, required=True)
    prepare_parser.add_argument("--portfolio-issues", type=Path, required=True)
    prepare_parser.add_argument("--portfolio-sources", type=Path, required=True)
    prepare_parser.add_argument("--source-backfills", type=Path, action="append", default=[])
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--report", type=Path, required=True)
    prepare_parser.add_argument("--max-tokens", type=int, default=6144)
    prepare_parser.add_argument(
        "--only-issue-id", action="append", default=[],
        help="Emit requests only for these issue IDs after validating all inputs.",
    )
    prepare_parser.set_defaults(handler=prepare)

    parse_parser = commands.add_parser("parse")
    parse_parser.add_argument("--examples", type=Path, required=True)
    parse_parser.add_argument("--requests", type=Path, required=True)
    parse_parser.add_argument("--responses", type=Path, required=True)
    parse_parser.add_argument("--accepted", type=Path, required=True)
    parse_parser.add_argument("--rejected", type=Path, required=True)
    parse_parser.add_argument("--report", type=Path, required=True)
    parse_parser.set_defaults(handler=parse)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
