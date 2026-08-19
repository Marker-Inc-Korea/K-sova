#!/usr/bin/env python3
"""Prepare and parse focused one-variant K-SOVA answer repairs."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from parse_full_answer_teacher_responses import (
    INPUT_REFERENCE_CUES,
    PERSPECTIVE_CUES,
    VARIANTS,
    example_id,
    korean_ratio,
    normalized_text,
    parse_object,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def combo(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("issue_id") or ""), str(row.get("variant") or "")


def focused_system_prompt() -> str:
    return """당신은 K-SOVA 완결형 학습 예시 한 개를 집중 교정하는 데이터 설계자다.

원래 쟁점 카드와 검증 원천만 사실 근거로 사용한다. 감사 의견은 누락 요소를 찾는 데만 사용한다. 새로운 인명·기관·수치·날짜·법적 효과를 만들지 않는다. 정부 발표는 귀속된 하나의 논거이지 자동 정답이 아니다. 국수주의적 구호, 정부 대변 문체, 상대국·집단 일반화, 근거 없는 확정 표현을 쓰지 않는다.

질문은 지정된 역할을 지키되 "한국 관점에서", "한국 입장에서", "한국인이라면"처럼 정답 관점을 직접 지시하거나 답을 노출하지 않는다. 답변은 자연스러운 한국어 최종 답변으로 1,100~1,600자를 목표로 하며, 확인 사실과 주장 구분, 한국 사회에서 방어 가능한 최소 기준선과 구체적 권리·이익, 가장 강한 상대 논거와 그 한계, 명확한 판단·권고, 판단을 바꿀 조건·전환 기준과 비용 부담 주체·위험·불확실성을 모두 실질적으로 포함한다. 체크리스트나 짧은 중간 산출물로 대신하지 않는다.

반드시 요청된 변형 한 개만 JSON 객체로 출력하고 코드블록이나 앞뒤 설명을 쓰지 않는다."""


def prepare(args: argparse.Namespace) -> int:
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite variant-repair outputs")
    source_requests = read_jsonl(args.source_requests)
    current_examples = read_jsonl(args.current_examples)
    current_rejections = read_jsonl(args.current_rejections)
    if len(source_requests) != 150 or len({row.get("id") for row in source_requests}) != 150:
        raise ValueError("source request set must contain 150 unique requests")
    if len(current_examples) != 450 or len({row.get("example_id") for row in current_examples}) != 450:
        raise ValueError("current example set must contain 450 unique examples")

    current_by_combo = {combo(row): row for row in current_examples}
    expected = {
        (str(row.get("issue_id") or ""), variant)
        for row in current_examples for variant in VARIANTS
    }
    if len(current_by_combo) != 450 or len(expected) != 450:
        raise ValueError("current examples do not form 150x3 issue-variant pairs")

    validated: set[tuple[str, str]] = set()
    for path in args.accepted:
        for row in read_jsonl(path):
            validation = row.get("validation") or {}
            if validation.get("verdict") != "pass":
                raise ValueError(f"{path}: accepted input contains a non-pass row")
            key = combo(row)
            if key not in expected:
                raise ValueError(f"{path}: unexpected issue-variant pair {key}")
            validated.add(key)
    missing = sorted(expected - validated)

    rejection_by_combo: dict[tuple[str, str], dict[str, Any]] = {}
    for row in current_rejections:
        example = row.get("example")
        validation = row.get("validation")
        if not isinstance(example, dict) or not isinstance(validation, dict):
            raise ValueError("current rejection rows must embed example and validation")
        key = combo(example)
        if key in rejection_by_combo:
            raise ValueError(f"duplicate current rejection pair {key}")
        rejection_by_combo[key] = validation
    absent_feedback = [key for key in missing if key not in rejection_by_combo]
    if absent_feedback:
        raise ValueError(f"missing current rejection feedback: {absent_feedback}")

    source_by_issue: dict[str, dict[str, Any]] = {}
    for request in source_requests:
        issue_id = str((request.get("metadata") or {}).get("issue_id") or "")
        source_by_issue[issue_id] = request
    if {issue_id for issue_id, _variant in expected} != set(source_by_issue):
        raise ValueError("source request and current example issue sets differ")

    output_rows = []
    prompt_chars = []
    failure_counts: Counter[str] = Counter()
    for issue_id, variant in missing:
        source = source_by_issue[issue_id]
        source_messages = source.get("messages") or []
        if len(source_messages) != 2 or source_messages[1].get("role") != "user":
            raise ValueError(f"{issue_id}: malformed source request")
        current = current_by_combo[(issue_id, variant)]
        messages = current.get("messages") or []
        if len(messages) != 2:
            raise ValueError(f"{issue_id}/{variant}: malformed current example")
        audit = rejection_by_combo[(issue_id, variant)]
        failed_checks = [str(value) for value in audit.get("failed_checks") or []]
        failure_counts.update(failed_checks)
        role_ko = {
            "neutral": "선동적 전제 없이 판단을 묻는 중립 질의",
            "claim_review": "사용자가 제시한 강한 주장의 타당성을 검토하는 질의",
            "policy_or_comparison": "대응·정책·판단 기준을 비교하거나 조건부 권고를 요구하는 질의",
        }[variant]
        repair_payload = {
            "issue_id": issue_id,
            "variant": variant,
            "required_role_ko": role_ko,
            "previous_candidate": {
                "user_ko": messages[0]["content"],
                "assistant_ko": messages[1]["content"],
            },
            "independent_audit": {
                "failed_checks": failed_checks,
                "reason_ko": audit.get("reason_ko") or "",
            },
        }
        user = f"""아래 원래 요청에는 쟁점 카드와 검증 원천이 들어 있다. 그중 세 예시를 만들라는 출력 지시는 무시하고, 뒤에 지정한 변형 한 개만 집중 교정하라.

원래 요청:
{source_messages[1]['content']}

집중 교정 대상:
{json.dumps(repair_payload, ensure_ascii=False)}

이전 문장을 늘려 쓰는 데 그치지 말고 감사 누락을 실제 논증으로 보완하라. 다음 형식을 정확히 지켜라:
{{"issue_id":"{issue_id}","variant":"{variant}","user_ko":"질문","assistant_ko":"완결 답변","coverage_ko":["사실·귀속","한국의 권리·이익","상대 논거","판단","조건·비용·불확실성"]}}"""
        request_messages = [
            {"role": "system", "content": focused_system_prompt()},
            {"role": "user", "content": user},
        ]
        source_metadata = source.get("metadata") or {}
        output_rows.append({
            "id": f"ksova_full_answer_variant_repair__{issue_id}__{variant}",
            "messages": request_messages,
            "max_tokens": args.max_tokens,
            "metadata": {
                "purpose": "ksova_single_full_answer_variant_repair",
                "prompt_version": "ksova-full-answer-variant-repair-v1",
                "issue_id": issue_id,
                "domain_id": current.get("domain_id"),
                "variant": variant,
                "source_ids": list(source_metadata.get("source_ids") or []),
                "source_teacher_request_id": source.get("id"),
                "previous_example_id": current.get("example_id"),
            },
        })
        prompt_chars.append(sum(len(message["content"]) for message in request_messages))

    report = {
        "schema_version": "ksova-full-answer-variant-repair-plan-1.0",
        "validated_pairs": len(validated),
        "repair_pairs": len(missing),
        "repair_issues": len({issue_id for issue_id, _variant in missing}),
        "missing_pairs": [f"{issue_id}::{variant}" for issue_id, variant in missing],
        "failed_checks": dict(sorted(failure_counts.items())),
        "prompt_chars": {
            "min": min(prompt_chars) if prompt_chars else 0,
            "max": max(prompt_chars) if prompt_chars else 0,
            "average": round(sum(prompt_chars) / len(prompt_chars), 1) if prompt_chars else 0,
        },
        "hashes": {
            "source_requests_sha256": file_sha256(args.source_requests),
            "current_examples_sha256": file_sha256(args.current_examples),
            "current_rejections_sha256": file_sha256(args.current_rejections),
            "accepted_sha256": {str(path): file_sha256(path) for path in args.accepted},
        },
        "output": str(args.output),
    }
    write_jsonl(args.output, output_rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def structural_reasons(user: str, assistant: str, benchmark_questions: set[str]) -> list[str]:
    reasons = []
    if not 24 <= len(user) <= 400:
        reasons.append("question_length")
    if not 600 <= len(assistant) <= 2200:
        reasons.append("answer_length")
    if korean_ratio(user) < 0.25 or korean_ratio(assistant) < 0.25:
        reasons.append("insufficient_korean")
    if "```" in user or "```" in assistant:
        reasons.append("code_fence")
    if any(cue in user for cue in PERSPECTIVE_CUES):
        reasons.append("perspective_cue_in_question")
    if any(cue in user for cue in INPUT_REFERENCE_CUES):
        reasons.append("input_reference_in_question")
    if normalized_text(user) in benchmark_questions:
        reasons.append("exact_benchmark_question_overlap")
    if not normalized_text(user) or not normalized_text(assistant):
        reasons.append("empty_normalized_text")
    return sorted(set(reasons))


def parse(args: argparse.Namespace) -> int:
    outputs = (args.accepted, args.rejected, args.report, args.retry_requests)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite variant-repair parse outputs")
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    benchmarks = read_jsonl(args.benchmark_items)
    request_by_id = {str(row.get("id") or ""): row for row in requests}
    response_by_id = {str(row.get("id") or ""): row for row in responses}
    if len(request_by_id) != len(requests) or len(response_by_id) != len(responses):
        raise ValueError("duplicate or missing request/response IDs")
    if set(request_by_id) != set(response_by_id):
        raise ValueError("variant-repair request and response ID sets differ")
    if len(benchmarks) != 480:
        raise ValueError("expected frozen 480-item benchmark")
    benchmark_questions = {
        normalized_text(str(row.get("question_ko") or "")) for row in benchmarks
    }
    accepted = []
    rejected = []
    retry_rows = []
    reason_counts: Counter[str] = Counter()
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    for request in requests:
        request_id = str(request["id"])
        metadata = request.get("metadata") or {}
        issue_id = str(metadata.get("issue_id") or "")
        variant = str(metadata.get("variant") or "")
        response = response_by_id[request_id]
        reasons = []
        payload = None
        if response.get("error"):
            reasons.append("response_error")
        if str(response.get("finish_reason") or "").casefold() != "stop":
            reasons.append("finish_reason_not_stop")
        if not str(response.get("response") or "").strip():
            reasons.append("empty_response")
        if not reasons:
            try:
                payload = parse_object(str(response["response"]))
            except (json.JSONDecodeError, ValueError):
                reasons.append("json_parse_failure")
        if payload is not None:
            if str(payload.get("issue_id") or "") != issue_id:
                reasons.append("issue_id_mismatch")
            if str(payload.get("variant") or "") != variant:
                reasons.append("variant_mismatch")
        user = str((payload or {}).get("user_ko") or "").strip()
        assistant = str((payload or {}).get("assistant_ko") or "").strip()
        if payload is not None:
            reasons.extend(structural_reasons(user, assistant, benchmark_questions))
            if normalized_text(user) in seen_questions:
                reasons.append("duplicate_repair_question")
            if normalized_text(assistant) in seen_answers:
                reasons.append("duplicate_repair_answer")
        reasons = sorted(set(reasons))
        if reasons:
            reason_counts.update(reasons)
            rejected.append({
                "request_id": request_id,
                "issue_id": issue_id,
                "variant": variant,
                "reasons": reasons,
                "user_ko": user,
                "assistant_ko": assistant,
            })
            retry = copy.deepcopy(request)
            retry["messages"][0]["content"] += (
                "\n\n재시도에서는 다음 구조 실패를 모두 고쳐라: " + ", ".join(reasons)
            )
            retry["metadata"] = {
                **metadata,
                "retry_of_request_id": request_id,
                "retry_reasons": reasons,
            }
            retry_rows.append(retry)
            continue
        eid = example_id(issue_id, variant, user, assistant)
        accepted.append({
            "schema_version": "ksova-full-answer-sft-1.0",
            "example_id": eid,
            "issue_id": issue_id,
            "domain_id": metadata.get("domain_id"),
            "variant": variant,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "source_ids": list(metadata.get("source_ids") or []),
            "metadata": {
                "training_id": eid,
                "issue_id": issue_id,
                "example_kind": f"full_answer_{variant}",
                "coverage_priority": variant == "neutral",
                "sample_weight": 1.0,
            },
            "provenance": {
                "teacher_model": response.get("model"),
                "teacher_request_id": request_id,
                "teacher_messages_sha256": response.get("messages_sha256"),
                "teacher_finish_reason": response.get("finish_reason"),
                "prompt_version": metadata.get("prompt_version"),
                "previous_example_id": metadata.get("previous_example_id"),
            },
        })
        seen_questions.add(normalized_text(user))
        seen_answers.add(normalized_text(assistant))
    report = {
        "schema_version": "ksova-full-answer-variant-repair-parse-1.0",
        "requests": len(requests),
        "responses": len(responses),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "failure_counts": dict(sorted(reason_counts.items())),
        "hashes": {
            "requests_sha256": file_sha256(args.requests),
            "responses_sha256": file_sha256(args.responses),
            "benchmark_sha256": file_sha256(args.benchmark_items),
        },
    }
    write_jsonl(args.accepted, accepted)
    write_jsonl(args.rejected, rejected)
    write_jsonl(args.retry_requests, retry_rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-requests", type=Path, required=True)
    prepare_parser.add_argument("--current-examples", type=Path, required=True)
    prepare_parser.add_argument("--current-rejections", type=Path, required=True)
    prepare_parser.add_argument("--accepted", type=Path, action="append", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--report", type=Path, required=True)
    prepare_parser.add_argument("--max-tokens", type=int, default=4096)
    prepare_parser.set_defaults(handler=prepare)
    parse_parser = commands.add_parser("parse")
    parse_parser.add_argument("--requests", type=Path, required=True)
    parse_parser.add_argument("--responses", type=Path, required=True)
    parse_parser.add_argument("--benchmark-items", type=Path, required=True)
    parse_parser.add_argument("--accepted", type=Path, required=True)
    parse_parser.add_argument("--rejected", type=Path, required=True)
    parse_parser.add_argument("--retry-requests", type=Path, required=True)
    parse_parser.add_argument("--report", type=Path, required=True)
    parse_parser.set_defaults(handler=parse)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
