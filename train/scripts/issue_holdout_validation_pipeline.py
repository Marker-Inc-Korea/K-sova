#!/usr/bin/env python3
"""Prepare and parse an LLM-assisted semantic holdout audit for training issues."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


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


def sha256(path: Path) -> str:
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


def compact(text: Any, limit: int = 1200) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()[:limit]


def system_prompt() -> str:
    return """당신은 학습-평가 분리 감사자다. 학습 쟁점 하나와 의미상 가까운 벤치마크 쟁점 후보들을 비교한다.

overlap은 단순히 같은 대영역·국가·산업·역사 사건을 다룬다는 뜻이 아니다. 학습 쟁점의 답을 알면 별도의 판단 없이 벤치마크의 결론·핵심 근거·조건을 사실상 재현할 수 있을 정도로 다음이 겹칠 때만 overlap이다: 구체적 분쟁 또는 결정 대상, 요구되는 규범적 판단, 결정적 법리·사실, 실질적 귀결. 동일 사건이라도 서로 다른 권리, 행위, 절차 또는 정책 선택을 독립적으로 판단해야 하면 distinct다. 서로 다른 사건에 같은 추상 원리를 적용하는 것도 distinct다. 표현 유사도 점수 자체를 판정 근거로 삼지 않는다.

후보 목록에 실제 중복이 없으면 distinct로 판정한다. 애매해서 추가 검토가 반드시 필요하면 review로 판정한다. 반드시 JSON 객체 하나만 출력하고 코드블록이나 앞뒤 설명을 쓰지 않는다."""


def prepare(args: argparse.Namespace) -> int:
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    records = read_jsonl(args.records)
    embeddings = read_jsonl(args.embeddings)
    record_by_id = {str(row.get("record_id") or ""): row for row in records}
    embedding_by_id = {str(row.get("record_id") or ""): row for row in embeddings}
    if (
        len(record_by_id) != len(records)
        or len(embedding_by_id) != len(embeddings)
        or set(record_by_id) != set(embedding_by_id)
    ):
        raise ValueError("record and embedding IDs must be unique and identical")

    import numpy as np

    benchmark = [row for row in records if row.get("role") == "benchmark_holdout"]
    training = [row for row in records if row.get("role") == "training_material_candidate"]
    if not benchmark or not training:
        raise ValueError("both benchmark and training records are required")
    benchmark_matrix = np.asarray([
        embedding_by_id[str(row["record_id"])]["embedding"] for row in benchmark
    ], dtype=np.float32)
    requests: list[dict[str, Any]] = []
    prompt_chars: list[int] = []
    for row in training:
        vector = np.asarray(
            embedding_by_id[str(row["record_id"])]["embedding"], dtype=np.float32,
        )
        similarities = benchmark_matrix @ vector
        order = np.argsort(-similarities)[: args.candidates]
        candidates = [{
            "issue_id": benchmark[int(index)]["entity_id"],
            "title_ko": benchmark[int(index)].get("title_ko"),
            "decision_summary_ko": compact(benchmark[int(index)].get("text")),
        } for index in order]
        training_issue_id = str(row["entity_id"])
        payload = {
            "training_issue": {
                "issue_id": training_issue_id,
                "title_ko": row.get("title_ko"),
                "decision_summary_ko": compact(row.get("text")),
            },
            "nearest_benchmark_candidates": candidates,
        }
        user = f"""다음 입력을 판정하라.
{json.dumps(payload, ensure_ascii=False)}

다음 형식을 정확히 지켜라:
{{
  "training_issue_id": "{training_issue_id}",
  "verdict": "distinct 또는 overlap 또는 review",
  "overlap_benchmark_issue_id": "overlap/review 대상 ID 또는 null",
  "reason_ko": "판정 대상과 핵심 차이 또는 중복을 두세 문장으로 설명"
}}"""
        request = {
            "id": "ksova_issue_holdout_validator__" + training_issue_id,
            "messages": [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user},
            ],
            "max_tokens": args.max_tokens,
            "metadata": {
                "purpose": "ksova_training_issue_semantic_holdout_validation",
                "prompt_version": "ksova-issue-semantic-holdout-validator-v1",
                "training_issue_id": training_issue_id,
                "candidate_issue_ids": [value["issue_id"] for value in candidates],
            },
        }
        requests.append(request)
        prompt_chars.append(sum(len(message["content"]) for message in request["messages"]))
    report = {
        "schema_version": "ksova-issue-semantic-holdout-validation-plan-1.0",
        "benchmark_issues": len(benchmark),
        "training_issues": len(training),
        "requests": len(requests),
        "nearest_candidates_per_request": args.candidates,
        "prompt_chars": {
            "min": min(prompt_chars),
            "average": round(sum(prompt_chars) / len(prompt_chars), 1),
            "max": max(prompt_chars),
        },
        "hashes": {
            "records_sha256": sha256(args.records),
            "embeddings_sha256": sha256(args.embeddings),
        },
    }
    write_jsonl(args.output, requests)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def parse(args: argparse.Namespace) -> int:
    for path in (args.judgments, args.flags, args.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    request_by_id = {str(row.get("id") or ""): row for row in requests}
    response_by_id = {str(row.get("id") or ""): row for row in responses}
    if len(request_by_id) != len(requests) or len(response_by_id) != len(responses):
        raise ValueError("duplicate request or response ID")
    unexpected_response_ids = sorted(set(response_by_id) - set(request_by_id))
    if unexpected_response_ids:
        raise ValueError(f"unexpected validation response IDs: {unexpected_response_ids[:5]}")
    judgments: list[dict[str, Any]] = []
    verdict_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    for request in requests:
        request_id = str(request["id"])
        metadata = request.get("metadata") or {}
        issue_id = str(metadata.get("training_issue_id") or "")
        candidate_ids = {str(value) for value in metadata.get("candidate_issue_ids") or []}
        response = response_by_id.get(request_id)
        payload: dict[str, Any] | None = None
        error = ""
        if response is None:
            error = "missing_response"
        elif response.get("error"):
            error = "response_error"
        elif str(response.get("finish_reason") or "").casefold() != "stop":
            error = "finish_reason_not_stop"
        else:
            try:
                payload = parse_object(str(response.get("response") or ""))
            except (json.JSONDecodeError, ValueError):
                error = "json_parse_failure"
        verdict = "invalid"
        overlap_id: str | None = None
        reason = ""
        if not error and payload is not None:
            verdict = str(payload.get("verdict") or "").casefold()
            raw_overlap = payload.get("overlap_benchmark_issue_id")
            overlap_id = str(raw_overlap) if raw_overlap is not None else None
            reason = str(payload.get("reason_ko") or "").strip()
            if str(payload.get("training_issue_id") or "") != issue_id:
                error = "training_issue_id_mismatch"
            elif verdict not in {"distinct", "overlap", "review"}:
                error = "invalid_verdict"
            elif verdict == "distinct" and overlap_id is not None:
                error = "distinct_with_overlap_id"
            elif verdict in {"overlap", "review"} and overlap_id not in candidate_ids:
                error = "flag_without_valid_candidate_id"
            elif not reason:
                error = "missing_reason"
        if error:
            verdict = "invalid"
            invalid_counts[error] += 1
        verdict_counts[verdict] += 1
        judgments.append({
            "schema_version": "ksova-issue-semantic-holdout-judgment-1.0",
            "training_issue_id": issue_id,
            "verdict": verdict,
            "overlap_benchmark_issue_id": overlap_id,
            "reason_ko": reason or error,
            "validator_request_id": request_id,
            "validator_model": response.get("model") if response else None,
            "candidate_issue_ids": sorted(candidate_ids),
        })
    flags = [row for row in judgments if row["verdict"] != "distinct"]
    report = {
        "schema_version": "ksova-issue-semantic-holdout-validation-report-1.0",
        "requests": len(requests),
        "responses": len(responses),
        "judgments": len(judgments),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "invalid_response_counts": dict(sorted(invalid_counts.items())),
        "flagged_for_manual_decision": len(flags),
        "hashes": {
            "requests_sha256": sha256(args.requests),
            "responses_sha256": sha256(args.responses),
        },
    }
    write_jsonl(args.judgments, judgments)
    write_jsonl(args.flags, flags)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 1 if invalid_counts else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--records", type=Path, required=True)
    prepare_parser.add_argument("--embeddings", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--report", type=Path, required=True)
    prepare_parser.add_argument("--candidates", type=int, default=8)
    prepare_parser.add_argument("--max-tokens", type=int, default=1024)
    prepare_parser.set_defaults(handler=prepare)
    parse_parser = commands.add_parser("parse")
    parse_parser.add_argument("--requests", type=Path, required=True)
    parse_parser.add_argument("--responses", type=Path, required=True)
    parse_parser.add_argument("--judgments", type=Path, required=True)
    parse_parser.add_argument("--flags", type=Path, required=True)
    parse_parser.add_argument("--report", type=Path, required=True)
    parse_parser.set_defaults(handler=parse)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
