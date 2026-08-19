#!/usr/bin/env python3
"""Parse and deterministically audit three-answer K-SOVA teacher outputs."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable


VARIANTS = ("neutral", "claim_review", "policy_or_comparison")
PERSPECTIVE_CUES = (
    "한국 관점에서",
    "한국의 관점에서",
    "한국 입장에서",
    "한국의 입장에서",
    "대한민국 관점에서",
    "대한민국의 관점에서",
    "대한민국 입장에서",
    "대한민국의 입장에서",
    "한국인이라면",
)
INPUT_REFERENCE_CUES = (
    "제공된 자료",
    "위 자료",
    "아래 자료",
    "쟁점 카드",
    "원문 근거",
    "입력 원문",
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
        raise ValueError("teacher response is not a JSON object")
    return payload


def normalized_text(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def korean_ratio(value: str) -> float:
    compact = re.sub(r"\s+", "", value)
    if not compact:
        return 0.0
    return len(re.findall(r"[가-힣]", compact)) / len(compact)


def example_id(issue_id: str, variant: str, user: str, assistant: str) -> str:
    value = "\n".join((issue_id, variant, user.strip(), assistant.strip()))
    return "KSSFT-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--benchmark-items", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    outputs = (args.accepted, args.rejected, args.report)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    benchmarks = read_jsonl(args.benchmark_items)
    if len(requests) != 150 or len({row.get("id") for row in requests}) != 150:
        raise ValueError("expected 150 unique teacher requests")
    if len(benchmarks) != 480:
        raise ValueError("expected frozen 480-item benchmark")
    response_by_id: dict[str, dict[str, Any]] = {}
    duplicate_response_ids: list[str] = []
    for row in responses:
        request_id = str(row.get("id") or "")
        if request_id in response_by_id:
            duplicate_response_ids.append(request_id)
        response_by_id[request_id] = row
    if duplicate_response_ids:
        raise ValueError(f"duplicate response IDs: {sorted(set(duplicate_response_ids))}")

    benchmark_questions = {
        normalized_text(str(row.get("question_ko") or ""))
        for row in benchmarks
        if normalized_text(str(row.get("question_ko") or ""))
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    response_failure_counts: Counter[str] = Counter()
    example_failure_counts: Counter[str] = Counter()
    answer_lengths: list[int] = []
    question_lengths: list[int] = []
    target_band_answers = 0
    seen_question: dict[str, str] = {}
    seen_answer: dict[str, str] = {}

    for request in requests:
        request_id = str(request["id"])
        metadata = request.get("metadata") or {}
        issue_id = str(metadata.get("issue_id") or "")
        domain_id = str(metadata.get("domain_id") or "")
        response = response_by_id.get(request_id)
        response_reasons: list[str] = []
        if response is None:
            response_reasons.append("missing_response")
        else:
            if response.get("error"):
                response_reasons.append("response_error")
            if str(response.get("finish_reason") or "").casefold() != "stop":
                response_reasons.append("finish_reason_not_stop")
            if not str(response.get("response") or "").strip():
                response_reasons.append("empty_response")
        if response_reasons:
            response_failure_counts.update(response_reasons)
            rejected.append({
                "request_id": request_id,
                "issue_id": issue_id,
                "variant": None,
                "reasons": response_reasons,
            })
            continue
        try:
            payload = parse_object(str(response["response"]))
        except (json.JSONDecodeError, ValueError) as exc:
            response_failure_counts["json_parse_failure"] += 1
            rejected.append({
                "request_id": request_id,
                "issue_id": issue_id,
                "variant": None,
                "reasons": ["json_parse_failure"],
                "detail": f"{type(exc).__name__}: {exc}",
            })
            continue
        if str(payload.get("issue_id") or "") != issue_id:
            response_failure_counts["issue_id_mismatch"] += 1
            rejected.append({
                "request_id": request_id,
                "issue_id": issue_id,
                "variant": None,
                "reasons": ["issue_id_mismatch"],
            })
            continue
        examples = payload.get("examples")
        if not isinstance(examples, list) or len(examples) != 3:
            response_failure_counts["not_exactly_three_examples"] += 1
            rejected.append({
                "request_id": request_id,
                "issue_id": issue_id,
                "variant": None,
                "reasons": ["not_exactly_three_examples"],
            })
            continue
        by_variant = {
            str(example.get("variant") or ""): example
            for example in examples if isinstance(example, dict)
        }
        if set(by_variant) != set(VARIANTS) or len(by_variant) != 3:
            response_failure_counts["variant_set_invalid"] += 1
            rejected.append({
                "request_id": request_id,
                "issue_id": issue_id,
                "variant": None,
                "reasons": ["variant_set_invalid"],
            })
            continue

        issue_questions: list[tuple[str, str]] = []
        issue_answers: list[tuple[str, str]] = []
        candidates: list[tuple[str, str, str, list[str]]] = []
        for variant in VARIANTS:
            example = by_variant[variant]
            user = str(example.get("user_ko") or "").strip()
            assistant = str(example.get("assistant_ko") or "").strip()
            reasons: list[str] = []
            if not 24 <= len(user) <= 400:
                reasons.append("question_length")
            # 900--1,600 characters is the authoring target, not a semantic
            # pass/fail boundary.  Keep only a broad structural floor here;
            # the independent validator decides whether a shorter answer is
            # nevertheless complete.
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
            normalized_user = normalized_text(user)
            normalized_assistant = normalized_text(assistant)
            if normalized_user in benchmark_questions:
                reasons.append("exact_benchmark_question_overlap")
            if not normalized_user or not normalized_assistant:
                reasons.append("empty_normalized_text")
            if normalized_user in seen_question:
                reasons.append("global_duplicate_question")
            if normalized_assistant in seen_answer:
                reasons.append("global_duplicate_answer")
            issue_questions.append((variant, normalized_user))
            issue_answers.append((variant, normalized_assistant))
            candidates.append((variant, user, assistant, reasons))

        duplicate_variants: set[str] = set()
        for index, (left_variant, left) in enumerate(issue_questions):
            for right_variant, right in issue_questions[index + 1 :]:
                if left and right and SequenceMatcher(None, left, right).ratio() >= 0.90:
                    duplicate_variants.update((left_variant, right_variant))
        copied_answer_variants: set[str] = set()
        for index, (left_variant, left) in enumerate(issue_answers):
            for right_variant, right in issue_answers[index + 1 :]:
                if left and right and SequenceMatcher(None, left, right).ratio() >= 0.92:
                    copied_answer_variants.update((left_variant, right_variant))

        for variant, user, assistant, reasons in candidates:
            if variant in duplicate_variants:
                reasons.append("same_issue_question_too_similar")
            if variant in copied_answer_variants:
                reasons.append("same_issue_answer_too_similar")
            reasons = sorted(set(reasons))
            if reasons:
                example_failure_counts.update(reasons)
                rejected.append({
                    "request_id": request_id,
                    "issue_id": issue_id,
                    "domain_id": domain_id,
                    "variant": variant,
                    "reasons": reasons,
                    "user_ko": user,
                    "assistant_ko": assistant,
                })
                continue
            eid = example_id(issue_id, variant, user, assistant)
            row = {
                "schema_version": "ksova-full-answer-sft-1.0",
                "example_id": eid,
                "issue_id": issue_id,
                "domain_id": domain_id,
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
                    "experiment_id": metadata.get("experiment_id"),
                },
            }
            accepted.append(row)
            seen_question[normalized_text(user)] = eid
            seen_answer[normalized_text(assistant)] = eid
            question_lengths.append(len(user))
            answer_lengths.append(len(assistant))
            target_band_answers += int(900 <= len(assistant) <= 1600)

    missing_response_ids = sorted(
        set(str(row["id"]) for row in requests) - set(response_by_id)
    )
    unexpected_response_ids = sorted(
        set(response_by_id) - set(str(row["id"]) for row in requests)
    )
    report = {
        "schema_version": "ksova-full-answer-teacher-parse-report-1.0",
        "requests": len(requests),
        "responses": len(responses),
        "accepted_examples": len(accepted),
        "expected_examples": len(requests) * 3,
        "accepted_by_domain": dict(sorted(Counter(
            str(row["domain_id"]) for row in accepted
        ).items())),
        "accepted_by_variant": dict(sorted(Counter(
            str(row["variant"]) for row in accepted
        ).items())),
        "response_failure_counts": dict(sorted(response_failure_counts.items())),
        "example_failure_counts": dict(sorted(example_failure_counts.items())),
        "missing_response_ids": missing_response_ids,
        "unexpected_response_ids": unexpected_response_ids,
        "answer_chars": {
            "min": min(answer_lengths) if answer_lengths else 0,
            "max": max(answer_lengths) if answer_lengths else 0,
            "average": round(mean(answer_lengths), 1) if answer_lengths else 0,
            "within_target_900_1600": target_band_answers,
        },
        "question_chars": {
            "min": min(question_lengths) if question_lengths else 0,
            "max": max(question_lengths) if question_lengths else 0,
            "average": round(mean(question_lengths), 1) if question_lengths else 0,
        },
        "hashes": {
            "requests_sha256": file_sha256(args.requests),
            "responses_sha256": file_sha256(args.responses),
            "benchmark_sha256": file_sha256(args.benchmark_items),
        },
        "outputs": {
            "accepted": str(args.accepted),
            "rejected": str(args.rejected),
        },
    }
    write_jsonl(args.accepted, accepted)
    write_jsonl(args.rejected, rejected)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 1 if missing_response_ids or unexpected_response_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
