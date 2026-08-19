#!/usr/bin/env python3
"""Build fail-closed teacher repair requests from independent audit feedback."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


VARIANTS = ("neutral", "claim_review", "policy_or_comparison")


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


def repair_system_prompt() -> str:
    return """당신은 독립 감사에서 탈락한 K-SOVA 완결형 학습 예시를 교정하는 데이터 설계자다.

입력에 다시 제시된 원래 쟁점 카드와 검증 원천만 사실 근거로 사용한다. 감사 의견은 누락 요소를 찾는 데만 사용하고, 감사 의견 속 표현을 사실 근거로 삼지 않는다. 새로운 인명·기관·수치·날짜·법적 효과를 만들지 않는다. 정부 발표는 귀속된 하나의 논거이지 자동 정답이 아니다. 국수주의적 구호, 정부 대변 문체, 상대국·집단 일반화, 근거 없는 확정 표현을 쓰지 않는다.

세 질문은 neutral, claim_review, policy_or_comparison이라는 서로 다른 판단 작업이어야 한다. 질문에 "한국 관점에서", "한국 입장에서", "한국인이라면"처럼 정답 관점을 직접 지시하거나 기준답안의 결론을 노출하지 않는다.

각 답변은 자연스러운 한국어 최종 답변으로 1,100~1,600자를 목표로 한다. 체크리스트나 짧은 중간 산출물을 쓰지 말고, 각각 다음 내용을 구체적으로 완결한다.
1. 원천으로 확인되는 사실과 행위자 주장·전망을 구분하고 필요한 경우 출처 주체를 귀속한다.
2. 한국 사회에서 방어 가능한 최소 기준선과 한국의 구체적 권리·이익을 밝힌다.
3. 상대가 제시할 수 있는 가장 강한 논거를 약화하지 않고 설명한 뒤 그 한계나 수용 범위를 검토한다.
4. 질문에 대한 선택·판단·권고를 모호한 균형론으로 끝내지 않고 명시한다.
5. 판단을 유지하거나 바꿀 구체적 조건·전환 기준, 누가 부담하는 비용과 위험, 확인되지 않은 불확실성을 밝힌다.

이전 답변을 문장만 늘려 복제하지 말고 감사에서 지적된 누락을 실질적으로 고친다. 세 답변 사이의 공통 사실은 사용할 수 있지만 같은 문단을 복사하지 않는다. 반드시 JSON 객체 하나만 출력하고 코드블록이나 앞뒤 설명을 쓰지 않는다."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-requests", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite repair request outputs")

    base_requests = read_jsonl(args.base_requests)
    examples = read_jsonl(args.examples)
    rejected_rows = read_jsonl(args.rejections)
    if len(base_requests) != 150 or len({row.get("id") for row in base_requests}) != 150:
        raise ValueError("base request set must contain exactly 150 unique requests")
    if len(examples) != 450 or len({row.get("example_id") for row in examples}) != 450:
        raise ValueError("example set must contain exactly 450 unique examples")

    examples_by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    example_by_id: dict[str, dict[str, Any]] = {}
    for example in examples:
        issue_id = str(example.get("issue_id") or "")
        example_id = str(example.get("example_id") or "")
        examples_by_issue[issue_id].append(example)
        example_by_id[example_id] = example
    for issue_id, rows in examples_by_issue.items():
        variants = {str(row.get("variant") or "") for row in rows}
        if len(rows) != 3 or variants != set(VARIANTS):
            raise ValueError(f"{issue_id}: expected exactly three variants")

    rejection_by_id: dict[str, dict[str, Any]] = {}
    failed_checks: Counter[str] = Counter()
    for row in rejected_rows:
        example = row.get("example")
        validation = row.get("validation")
        if not isinstance(example, dict) or not isinstance(validation, dict):
            raise ValueError("rejection rows must contain example and validation objects")
        example_id = str(example.get("example_id") or "")
        if example_id not in example_by_id:
            raise ValueError(f"rejection references unknown example {example_id}")
        if example != example_by_id[example_id]:
            raise ValueError(f"rejection embeds a changed example {example_id}")
        if validation.get("verdict") != "reject":
            raise ValueError(f"non-rejection verdict for {example_id}")
        if example_id in rejection_by_id:
            raise ValueError(f"duplicate rejection for {example_id}")
        rejection_by_id[example_id] = validation
        failed_checks.update(str(value) for value in validation.get("failed_checks") or [])

    repair_issue_ids = {
        str(example_by_id[example_id]["issue_id"])
        for example_id in rejection_by_id
    }
    request_by_issue: dict[str, dict[str, Any]] = {}
    for request in base_requests:
        metadata = request.get("metadata") or {}
        issue_id = str(metadata.get("issue_id") or "")
        if issue_id in request_by_issue:
            raise ValueError(f"duplicate base request issue {issue_id}")
        request_by_issue[issue_id] = request
    if set(examples_by_issue) != set(request_by_issue):
        raise ValueError("base request and example issue sets differ")

    output_rows = []
    prompt_chars = []
    for issue_id in sorted(repair_issue_ids):
        base = request_by_issue[issue_id]
        messages = base.get("messages") or []
        if len(messages) != 2 or messages[1].get("role") != "user":
            raise ValueError(f"{issue_id}: unexpected base request messages")
        candidates = []
        for example in sorted(
            examples_by_issue[issue_id], key=lambda row: VARIANTS.index(row["variant"])
        ):
            example_id = str(example["example_id"])
            example_messages = example.get("messages") or []
            if len(example_messages) != 2:
                raise ValueError(f"{example_id}: expected two example messages")
            validation = rejection_by_id.get(example_id)
            audit = (
                {
                    "verdict": "reject",
                    "failed_checks": validation.get("failed_checks") or [],
                    "reason_ko": validation.get("reason_ko") or "",
                }
                if validation is not None
                else {"verdict": "pass", "failed_checks": [], "reason_ko": ""}
            )
            candidates.append({
                "example_id": example_id,
                "variant": example["variant"],
                "user_ko": example_messages[0]["content"],
                "assistant_ko": example_messages[1]["content"],
                "independent_audit": audit,
            })
        repair_instruction = f"""

이전 후보와 독립 감사 결과:
{json.dumps(candidates, ensure_ascii=False)}

위 감사에서 하나라도 탈락했으므로 이 쟁점의 세 예시를 모두 다시 작성하라. 통과했던 예시의 장점은 유지할 수 있지만 세 과제의 구분과 답변 간 정합성을 함께 점검한다. failed_checks의 문구를 답변에 나열하는 것으로 대신하지 말고 실제 논증으로 충족한다. 원래 요청과 동일한 issue_id 및 examples JSON 형식을 정확히 사용한다."""
        new_messages = [
            {"role": "system", "content": repair_system_prompt()},
            {"role": "user", "content": str(messages[1]["content"]) + repair_instruction},
        ]
        metadata = dict(base.get("metadata") or {})
        metadata.update({
            "purpose": "ksova_three_full_answer_sft_teacher_repair",
            "prompt_version": "ksova-full-answer-teacher-repair-v2",
            "repair_rejected_example_ids": sorted(
                example["example_id"]
                for example in examples_by_issue[issue_id]
                if example["example_id"] in rejection_by_id
            ),
        })
        output_rows.append({
            **base,
            "messages": new_messages,
            "max_tokens": args.max_tokens,
            "metadata": metadata,
        })
        prompt_chars.append(sum(len(message["content"]) for message in new_messages))

    report = {
        "schema_version": "ksova-full-answer-repair-plan-1.0",
        "base_requests": len(base_requests),
        "examples": len(examples),
        "rejections": len(rejection_by_id),
        "repair_issues": len(repair_issue_ids),
        "preserved_all_pass_issues": len(examples_by_issue) - len(repair_issue_ids),
        "failed_checks": dict(sorted(failed_checks.items())),
        "prompt_chars": {
            "min": min(prompt_chars) if prompt_chars else 0,
            "max": max(prompt_chars) if prompt_chars else 0,
            "average": round(sum(prompt_chars) / len(prompt_chars), 1) if prompt_chars else 0,
        },
        "hashes": {
            "base_requests_sha256": file_sha256(args.base_requests),
            "examples_sha256": file_sha256(args.examples),
            "rejections_sha256": file_sha256(args.rejections),
        },
        "output": str(args.output),
    }
    write_jsonl(args.output, output_rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
