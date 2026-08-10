#!/usr/bin/env python3
"""Build source-isolated, capability-staged K-SOVA SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROMPT_VERSION = "ksova-staged-capability-generation-v1"
TASKS = (
    "fact_claim_actor_separation",
    "korean_stake_extraction",
    "baseline_and_range",
    "response_discrimination",
)
SYSTEM = """당신은 K-SOVA 전이학습 자료 편집자다. 제공된 검증 완료 문항의 사실·경계·금지오류만 사용해, 정답 암기가 아니라 판단 절차를 학습하는 짧고 자립적인 예시를 만든다. 장문의 숨은 사고과정은 쓰지 않고 검증 가능한 구조화 결과만 제시한다. JSON 객체 하나만 출력한다."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_example(row: dict[str, Any], origin: str) -> dict[str, Any]:
    if origin == "accepted_benchmark_candidate":
        return {
            "example_id": row["item_id"],
            "prompt_form": row.get("prompt_form"),
            "question_ko": row["question_ko"],
            "answer_ko": row["reference_answer_ko"],
            "required_elements_ko": row["required_elements_ko"],
            "factual_boundaries_ko": row["factual_boundaries_ko"],
            "prohibited_errors_ko": row["prohibited_errors_ko"],
        }
    return {
        "example_id": row["example_id"],
        "prompt_form": row.get("form"),
        "question_ko": row["question_ko"],
        "answer_ko": row["answer_ko"],
        "required_elements_ko": row["required_elements_ko"],
        "factual_boundaries_ko": row["factual_boundaries_ko"],
        "prohibited_errors_ko": row["prohibited_errors_ko"],
    }


def choose_two(rows: list[dict[str, Any]], form_key: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (str(row.get(form_key)), str(
        row.get("item_id") or row.get("example_id")
    )))
    selected: list[dict[str, Any]] = []
    forms: set[str] = set()
    for row in ordered:
        form = str(row.get(form_key))
        if form not in forms:
            selected.append(row)
            forms.add(form)
        if len(selected) == 2:
            return selected
    for row in ordered:
        if row not in selected:
            selected.append(row)
        if len(selected) == 2:
            break
    return selected


def support_dossiers(
    core_items_path: Path,
    recommendations_path: Path,
    accepted_items_path: Path,
    safe_sft_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core_items = read_jsonl(core_items_path)
    recommendations = {
        str(row["issue_id"]): row for row in read_jsonl(recommendations_path)
    }
    accepted_items = read_jsonl(accepted_items_path)
    safe_sft = read_jsonl(safe_sft_path)
    core_issues = {str(row["issue_id"]) for row in core_items}
    core_sources = {str(value) for row in core_items for value in row.get("source_ids", [])}
    supporting = {
        issue_id for issue_id, row in recommendations.items()
        if row.get("final_verdict") == "supporting"
    }

    candidate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted_items:
        issue_id = str(row.get("issue_id") or "")
        if issue_id in supporting and row.get("item_role") == "sovereign_judgment":
            candidate_groups[issue_id].append(row)

    dossiers: list[dict[str, Any]] = []
    for issue_id, rows in sorted(candidate_groups.items()):
        sources = {str(value) for row in rows for value in row.get("source_ids", [])}
        if not sources or sources & core_sources:
            continue
        selected = choose_two(rows, "prompt_form")
        recommendation = recommendations[issue_id]
        dossier = {
            "issue_id": issue_id,
            "origin": "accepted_benchmark_candidate",
            "category_id": recommendation["category_id"],
            "title_ko": recommendation["title_ko"],
            "source_ids": sorted(sources),
            "validated_examples": [
                normalized_example(row, "accepted_benchmark_candidate")
                for row in selected
            ],
        }
        dossier["input_sha256"] = canonical_hash(dossier)
        dossiers.append(dossier)

    sft_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in safe_sft:
        material_id = str(row.get("material_id") or "")
        if material_id.startswith("KSMAT-") and material_id in supporting:
            sft_groups[material_id].append(row)
    for material_id, rows in sorted(sft_groups.items()):
        sources = {str(value) for row in rows for value in row.get("source_ids", [])}
        if not sources or sources & core_sources:
            raise RuntimeError(f"unsafe supporting material: {material_id}")
        selected = choose_two(rows, "form")
        recommendation = recommendations[material_id]
        dossier = {
            "issue_id": material_id,
            "origin": "safe_final_sft",
            "category_id": recommendation["category_id"],
            "title_ko": recommendation["title_ko"],
            "source_ids": sorted(sources),
            "validated_examples": [
                normalized_example(row, "safe_final_sft") for row in selected
            ],
        }
        dossier["input_sha256"] = canonical_hash(dossier)
        dossiers.append(dossier)

    dossiers.sort(key=lambda row: str(row["issue_id"]))
    issue_ids = [str(row["issue_id"]) for row in dossiers]
    all_sources = {
        source for row in dossiers for source in row.get("source_ids", [])
    }
    if len(dossiers) != 32 or len(set(issue_ids)) != 32:
        raise RuntimeError(f"expected 32 unique safe issues, got {len(dossiers)}")
    if set(issue_ids) & core_issues:
        raise RuntimeError("training issue overlaps core benchmark")
    if all_sources & core_sources:
        raise RuntimeError("training source overlaps core benchmark")
    report = {
        "schema_version": "ksova-staged-training-pool-report-1.0",
        "safe_issues": len(dossiers),
        "benchmark_origin_issues": sum(
            row["origin"] == "accepted_benchmark_candidate" for row in dossiers
        ),
        "sft_origin_issues": sum(row["origin"] == "safe_final_sft" for row in dossiers),
        "safe_sources": len(all_sources),
        "benchmark_issue_overlap": 0,
        "benchmark_source_overlap": 0,
        "category_counts": dict(sorted(Counter(
            row["category_id"] for row in dossiers
        ).items())),
        "input_hashes": {
            "core_items": file_hash(core_items_path),
            "recommendations": file_hash(recommendations_path),
            "accepted_items": file_hash(accepted_items_path),
            "safe_sft": file_hash(safe_sft_path),
        },
    }
    return dossiers, report


def preferred_label(issue_id: str) -> str:
    return "A" if int(hashlib.sha256(issue_id.encode()).hexdigest()[-1], 16) % 2 == 0 else "B"


def prompt_for(dossier: dict[str, Any], label: str) -> str:
    shape = {
        "issue_id": dossier["issue_id"],
        "examples": [
            {"task": task, "user_ko": "자립적인 학습 입력", "assistant_ko": "짧고 검증 가능한 정답"}
            for task in TASKS
        ],
    }
    return f"""다음 검증 완료 보조 쟁점을 사용해 네 가지 전이학습 예시를 각각 하나씩 작성하라. 새 사실을 추가하지 말고 validated_examples의 공통 사실·경계만 사용한다.

[과제]
1. fact_claim_actor_separation
   - user_ko에 쟁점 상황과 한 개 이상의 주장을 제시하고 사실·주장·행위자/관점의 구분을 요구한다.
   - assistant_ko는 '확인된 사실:', '주장·평가:', '행위자·관점:' 세 항목으로 450자 이내에 답한다.
2. korean_stake_extraction
   - user_ko는 관점 정답을 미리 말하지 않은 자립적 상황 질문이어야 한다.
   - assistant_ko는 한국에 걸린 권리·비용·위험·가치 중 실제 해당 항목만 구체적으로 추출하고, 곧바로 특정 정책을 정답화하지 않는다. 350자 이내다.
3. baseline_and_range
   - user_ko는 최소 기준과 조건에 따라 허용되는 선택 범위를 구분하도록 요구한다.
   - assistant_ko는 '최소 기준:', '허용 범위:', '실패 방향:' 세 항목으로 500자 이내에 답한다.
4. response_discrimination
   - user_ko에 동일 질문에 대한 후보 A와 후보 B를 각각 1~3문장으로 제시하고 더 적절한 답을 고르게 한다.
   - {label}가 최소 기준과 필요한 유보를 지킨 답이 되게 하고, 다른 후보는 실제 failure/prohibited 방향을 채택하게 한다.
   - assistant_ko는 반드시 '판정: {label}'로 시작하고 핵심 위반 기준만 2문장 이내로 설명한다.

[공통 규칙]
- 네 user_ko 모두 별도의 숨은 문맥 없이 이해 가능해야 하며 '위 자료', '앞의 내용', source_id를 쓰지 않는다.
- 한국인 역할극, 애국적 수사, 외국 국민 일반화, 현 정부 입장 복창을 학습시키지 않는다.
- 장문의 사고과정이나 단계별 내적 추론을 출력하지 않는다.
- validated_examples의 질문을 그대로 복제하지 말고 판단 기술을 분리한 새 과제로 만든다.
- JSON 객체 하나만 출력하고 정확히 아래 키를 쓴다.

[검증 완료 보조 쟁점]
{json.dumps(dossier, ensure_ascii=False, indent=2)}

[출력 형식]
{json.dumps(shape, ensure_ascii=False, indent=2)}"""


def command_prepare(args: argparse.Namespace) -> int:
    dossiers, pool_report = support_dossiers(
        args.core_items, args.recommendations, args.accepted_items, args.safe_sft
    )
    requests: list[dict[str, Any]] = []
    for dossier in dossiers:
        label = preferred_label(dossier["issue_id"])
        requests.append({
            "id": f"staged_training__{dossier['issue_id']}",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt_for(dossier, label)},
            ],
            "max_tokens": args.max_tokens,
            "metadata": {
                "purpose": "staged_training_generation",
                "prompt_version": PROMPT_VERSION,
                "issue_id": dossier["issue_id"],
                "origin": dossier["origin"],
                "category_id": dossier["category_id"],
                "source_ids": dossier["source_ids"],
                "input_sha256": dossier["input_sha256"],
                "preferred_label": label,
                "synthesis_example": dossier["validated_examples"][0],
            },
        })
    atomic_jsonl(args.dossiers, dossiers)
    atomic_jsonl(args.output, requests)
    pool_report.update({
        "prompt_version": PROMPT_VERSION,
        "generation_requests": len(requests),
        "dossiers_sha256": file_hash(args.dossiers),
        "requests_sha256": file_hash(args.output),
    })
    atomic_json(args.report, pool_report)
    print(json.dumps(pool_report, ensure_ascii=False, indent=2))
    return 0


def parse_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response is not object")
    return parsed


def validate_payload(
    payload: dict[str, Any], request: dict[str, Any], response: dict[str, Any],
) -> list[str]:
    metadata = request["metadata"]
    errors: list[str] = []
    if set(payload) != {"issue_id", "examples"}:
        errors.append("invalid_top_keys")
    if payload.get("issue_id") != metadata["issue_id"]:
        errors.append("issue_id_mismatch")
    if response.get("metadata") != metadata:
        errors.append("metadata_mismatch")
    if response.get("error"):
        errors.append("runtime_error")
    if str(response.get("finish_reason") or "").lower() != "stop":
        errors.append("finish_reason_not_stop")
    examples = payload.get("examples")
    if not isinstance(examples, list) or len(examples) != 4:
        return sorted(set(errors + ["invalid_examples_count"]))
    tasks = [row.get("task") for row in examples if isinstance(row, dict)]
    if tuple(tasks) != TASKS:
        errors.append("tasks_or_order_mismatch")
    for index, row in enumerate(examples):
        if not isinstance(row, dict) or set(row) != {"task", "user_ko", "assistant_ko"}:
            errors.append(f"invalid_example_shape:{index}")
            continue
        user = row.get("user_ko")
        assistant = row.get("assistant_ko")
        if not isinstance(user, str) or not 40 <= len(user.strip()) <= 1800:
            errors.append(f"invalid_user:{index}")
        if not isinstance(assistant, str) or not 20 <= len(assistant.strip()) <= 1200:
            errors.append(f"invalid_assistant:{index}")
        if isinstance(user, str) and any(term in user for term in ("source_id", "위 자료", "앞의 내용")):
            errors.append(f"hidden_context_reference:{index}")
    if len(examples) == 4 and isinstance(examples[3], dict):
        answer = str(examples[3].get("assistant_ko") or "").lstrip()
        if not answer.startswith(f"판정: {metadata['preferred_label']}"):
            errors.append("preferred_label_mismatch")
        user = str(examples[3].get("user_ko") or "")
        if "후보 A" not in user or "후보 B" not in user:
            errors.append("candidate_labels_missing")
    return sorted(set(errors))


def command_parse(args: argparse.Namespace) -> int:
    requests = read_jsonl(args.requests)
    request_by_id = {str(row["id"]): row for row in requests}
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in args.responses:
        for row in read_jsonl(path):
            attempts[str(row.get("id"))].append(row)
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for request_id, request in request_by_id.items():
        selected: tuple[dict[str, Any], dict[str, Any]] | None = None
        failures: list[list[str]] = []
        for response in reversed(attempts.get(request_id, [])):
            try:
                payload = parse_object(str(response.get("response") or ""))
                errors = validate_payload(payload, request, response)
            except (ValueError, json.JSONDecodeError) as exc:
                payload = {}
                errors = [f"parse:{type(exc).__name__}"]
            if not errors:
                selected = (response, payload)
                break
            failures.append(errors)
        if selected is None:
            invalid.append({
                "request_id": request_id,
                "issue_id": request["metadata"]["issue_id"],
                "attempts": len(attempts.get(request_id, [])),
                "errors": sorted({error for values in failures for error in values})
                or ["missing_response"],
            })
            continue
        response, payload = selected
        valid.append({
            "schema_version": "ksova-staged-capability-bundle-1.0",
            **payload,
            "metadata": request["metadata"],
            "generation": {
                "model": response.get("model"),
                "messages_sha256": response.get("messages_sha256"),
                "prompt_version": PROMPT_VERSION,
            },
        })
    valid.sort(key=lambda row: str(row["issue_id"]))
    atomic_jsonl(args.output, valid)
    atomic_jsonl(args.invalid, invalid)
    report = {
        "schema_version": "ksova-staged-capability-parse-report-1.0",
        "requests": len(requests),
        "valid_bundles": len(valid),
        "invalid_bundles": len(invalid),
        "generated_examples": sum(len(row["examples"]) for row in valid),
        "complete": len(valid) == len(requests) and not invalid,
        "output_sha256": file_hash(args.output),
    }
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 1


def command_retry(args: argparse.Namespace) -> int:
    invalid_rows = read_jsonl(args.invalid)
    invalid_by_id = {
        str(row["request_id"]): row.get("errors", []) for row in invalid_rows
    }
    invalid_ids = set(invalid_by_id)
    rows = []
    for original in read_jsonl(args.requests):
        request_id = str(original["id"])
        if request_id not in invalid_ids:
            continue
        row = json.loads(json.dumps(original, ensure_ascii=False))
        errors = ", ".join(str(value) for value in invalid_by_id[request_id])
        row["messages"][-1]["content"] += f"""

[형식 검증 재시도]
이전 출력의 검증 오류: {errors}
- 최상위 키는 issue_id와 examples만 사용한다.
- examples는 정확히 4개이며 지정된 과제 순서를 지킨다.
- 각 예시는 task, user_ko, assistant_ko 세 키만 사용한다.
- response_discrimination의 user_ko에는 '후보 A'와 '후보 B'를 문자 그대로 모두 넣는다.
- 설명이나 코드펜스 없이 완전한 JSON 객체 하나만 출력한다.
"""
        rows.append(row)
    if len(rows) != len(invalid_ids):
        raise ValueError("invalid ids do not resolve exactly")
    atomic_jsonl(args.output, rows)
    print(json.dumps({"retry_requests": len(rows), "output": str(args.output)}))
    return 0


def dev_issues(bundles: list[dict[str, Any]], count: int = 4) -> set[str]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for row in bundles:
        by_category[str(row["metadata"]["category_id"])].append(str(row["issue_id"]))
    candidates: list[tuple[str, str]] = []
    for category, values in sorted(by_category.items()):
        selected = min(values, key=lambda issue: canonical_hash({"dev": issue}))
        candidates.append((canonical_hash({"category": category}), selected))
    return {issue for _, issue in sorted(candidates)[:count]}


def training_row(
    issue_id: str, task: str, user: str, assistant: str,
    metadata: dict[str, Any], split: str,
) -> dict[str, Any]:
    identifier = hashlib.sha256(
        f"{issue_id}\0{task}\0{user}\0{assistant}".encode("utf-8")
    ).hexdigest()[:16]
    phase = {
        "fact_claim_actor_separation": "phase1_mapping",
        "korean_stake_extraction": "phase1_mapping",
        "baseline_and_range": "phase2_boundary",
        "response_discrimination": "phase2_boundary",
        "final_synthesis": "phase3_synthesis",
    }[task]
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "metadata": {
            "training_id": f"KSSTAGE-{identifier}",
            "issue_id": issue_id,
            "source_ids": metadata["source_ids"],
            "category_id": metadata["category_id"],
            "origin": metadata["origin"],
            "task": task,
            "curriculum_phase": phase,
            "split": split,
            "source_isolated_from_core250": True,
            "input_sha256": metadata["input_sha256"],
        },
    }


def command_assemble(args: argparse.Namespace) -> int:
    core_items = read_jsonl(args.core_items)
    core_issues = {str(row["issue_id"]) for row in core_items}
    core_sources = {str(value) for row in core_items for value in row.get("source_ids", [])}
    bundles = read_jsonl(args.bundles)
    if len(bundles) != 32 or len({row["issue_id"] for row in bundles}) != 32:
        raise RuntimeError("requires exactly 32 valid, unique bundles")
    dev = dev_issues(bundles, args.dev_issues)
    all_rows: list[dict[str, Any]] = []
    per_issue: dict[str, dict[str, dict[str, Any]]] = {}
    for bundle in bundles:
        issue_id = str(bundle["issue_id"])
        metadata = bundle["metadata"]
        split = "dev" if issue_id in dev else "train"
        task_rows: dict[str, dict[str, Any]] = {}
        for example in bundle["examples"]:
            row = training_row(
                issue_id, example["task"], example["user_ko"],
                example["assistant_ko"], metadata, split,
            )
            task_rows[example["task"]] = row
            all_rows.append(row)
        synthesis = metadata["synthesis_example"]
        row = training_row(
            issue_id, "final_synthesis", synthesis["question_ko"],
            synthesis["answer_ko"], metadata, split,
        )
        task_rows["final_synthesis"] = row
        all_rows.append(row)
        per_issue[issue_id] = task_rows
    all_rows.sort(key=lambda row: (
        row["metadata"]["issue_id"], row["metadata"]["task"]
    ))
    train_rows = [row for row in all_rows if row["metadata"]["split"] == "train"]
    dev_rows = [row for row in all_rows if row["metadata"]["split"] == "dev"]

    def phase_rows(split: str, phase: int) -> list[dict[str, Any]]:
        issue_ids = sorted(
            issue for issue in per_issue
            if per_issue[issue]["final_synthesis"]["metadata"]["split"] == split
        )
        rows: list[dict[str, Any]] = []
        for index, issue in enumerate(issue_ids):
            tasks = per_issue[issue]
            if phase == 1:
                rows.extend((
                    tasks["fact_claim_actor_separation"],
                    tasks["korean_stake_extraction"],
                ))
            elif phase == 2:
                rows.extend((tasks["baseline_and_range"], tasks["response_discrimination"]))
                replay_task = (
                    "fact_claim_actor_separation" if index % 2 == 0
                    else "korean_stake_extraction"
                )
                rows.append(tasks[replay_task])
            else:
                rows.extend((
                    tasks["final_synthesis"],
                    tasks["baseline_and_range"] if index % 2 == 0
                    else tasks["response_discrimination"],
                ))
                if index % 2 == 0:
                    rows.append(tasks["korean_stake_extraction"])
        return rows

    outputs = {
        "all_examples.jsonl": all_rows,
        "train_all.jsonl": train_rows,
        "dev_all.jsonl": dev_rows,
        "phase1_train.jsonl": phase_rows("train", 1),
        "phase1_dev.jsonl": phase_rows("dev", 1),
        "phase2_train.jsonl": phase_rows("train", 2),
        "phase2_dev.jsonl": phase_rows("dev", 2),
        "phase3_train.jsonl": phase_rows("train", 3),
        "phase3_dev.jsonl": phase_rows("dev", 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        atomic_jsonl(args.output_dir / name, rows)
    training_issues = {row["metadata"]["issue_id"] for row in all_rows}
    training_sources = {
        value for row in all_rows for value in row["metadata"]["source_ids"]
    }
    ids = [row["metadata"]["training_id"] for row in all_rows]
    errors: list[str] = []
    if len(all_rows) != 160 or len(ids) != len(set(ids)):
        errors.append("expected_160_unique_examples")
    if training_issues & core_issues:
        errors.append("benchmark_issue_overlap")
    if training_sources & core_sources:
        errors.append("benchmark_source_overlap")
    if len(train_rows) != (32 - args.dev_issues) * 5:
        errors.append("train_split_count")
    if len(dev_rows) != args.dev_issues * 5:
        errors.append("dev_split_count")
    report = {
        "schema_version": "ksova-staged-training-assembly-report-1.0",
        "pass": not errors,
        "errors": errors,
        "safe_issues": len(training_issues),
        "safe_sources": len(training_sources),
        "benchmark_issue_overlap": len(training_issues & core_issues),
        "benchmark_source_overlap": len(training_sources & core_sources),
        "examples": len(all_rows),
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "train_issues": len(training_issues - dev),
        "dev_issues": len(dev),
        "dev_issue_ids": sorted(dev),
        "task_counts": dict(sorted(Counter(
            row["metadata"]["task"] for row in all_rows
        ).items())),
        "phase_file_rows": {name: len(rows) for name, rows in outputs.items()},
        "artifact_sha256": {
            name: file_hash(args.output_dir / name) for name in outputs
        },
    }
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--core-items", type=Path, required=True)
    prepare.add_argument("--recommendations", type=Path, required=True)
    prepare.add_argument("--accepted-items", type=Path, required=True)
    prepare.add_argument("--safe-sft", type=Path, required=True)
    prepare.add_argument("--dossiers", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--report", type=Path, required=True)
    prepare.add_argument("--max-tokens", type=int, default=3072)

    parse = sub.add_parser("parse")
    parse.add_argument("--requests", type=Path, required=True)
    parse.add_argument("--responses", type=Path, nargs="+", required=True)
    parse.add_argument("--output", type=Path, required=True)
    parse.add_argument("--invalid", type=Path, required=True)
    parse.add_argument("--report", type=Path, required=True)

    retry = sub.add_parser("retry")
    retry.add_argument("--requests", type=Path, required=True)
    retry.add_argument("--invalid", type=Path, required=True)
    retry.add_argument("--output", type=Path, required=True)

    assemble = sub.add_parser("assemble")
    assemble.add_argument("--core-items", type=Path, required=True)
    assemble.add_argument("--bundles", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--report", type=Path, required=True)
    assemble.add_argument("--dev-issues", type=int, default=4)

    args = parser.parse_args()
    return {
        "prepare": command_prepare,
        "parse": command_parse,
        "retry": command_retry,
        "assemble": command_assemble,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
