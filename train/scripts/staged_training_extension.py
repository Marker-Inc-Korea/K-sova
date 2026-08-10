#!/usr/bin/env python3
"""Build 9-capability and expanded K-SOVA SFT candidate sets.

This tool treats the frozen 160-example curriculum as immutable input.  It
reuses eligible earlier GLM-validated examples, generates only missing slots,
can validate generated candidates in separate stateless requests, and fails
closed unless model A has 288 and model B has 416 unique examples.  Whether a
particular assembled dataset completed independent LLM validation is recorded
in its assembly report and must not be inferred from the row count alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
FIRST_RELEASE = ROOT / "benchmark/v2.0/releases/first_pass_20260810_v1"
FIRST_WORK = ROOT / "benchmark/v2.0/work/runs/first_pass_20260810_v1/training"
PRIOR_ACCEPTED = (
    ROOT
    / "benchmark/v2.0/work/runs/training_review_reconciliation_250_20260809_v1"
    / "examples_accepted_final.jsonl"
)
DEFAULT_WORK = ROOT / "benchmark/v2.0/work/runs/second_pass_20260810_v1/training/data_extension"

STANDARD_MISSING = (
    "frame_adversarial_resistance",
    "competing_claim_resolution",
    "actor_decision_application",
    "uncertainty_consequence_boundary",
)
EXPANDED = (
    "hard_adversarial_stress",
    "hard_near_miss_discrimination",
    "hard_condition_consequence_shift",
    "hard_actor_shift_application",
)
CORE_TASKS = (
    "fact_claim_actor_separation",
    "korean_stake_extraction",
    "baseline_and_range",
    *STANDARD_MISSING,
    "response_discrimination",
    "final_synthesis",
)
STAGE = {
    "fact_claim_actor_separation": 1,
    "korean_stake_extraction": 2,
    "baseline_and_range": 3,
    "frame_adversarial_resistance": 4,
    "competing_claim_resolution": 5,
    "actor_decision_application": 6,
    "uncertainty_consequence_boundary": 7,
    "response_discrimination": 8,
    "final_synthesis": 9,
    "hard_adversarial_stress": 4,
    "hard_near_miss_discrimination": 8,
    "hard_condition_consequence_shift": 7,
    "hard_actor_shift_application": 6,
}
PHASE = {
    task: (
        "phase1_foundation" if STAGE[task] <= 3
        else "phase2_application" if STAGE[task] <= 7
        else "phase3_synthesis"
    )
    for task in (*CORE_TASKS, *EXPANDED)
}
PRIOR_SKILL_MAP = {
    "frame_resistance": "frame_adversarial_resistance",
    "competing_claim_resolution": "competing_claim_resolution",
    "actor_decision": "actor_decision_application",
    "boundary_control": "uncertainty_consequence_boundary",
}
GENERATION_PROMPT_VERSION = "ksova-9cap-expanded-generation-v1"
VALIDATION_PROMPT_VERSION = "ksova-9cap-expanded-validation-v1"
SYSTEM_GENERATE = """당신은 K-SOVA 전이학습 데이터 편집자다. 제공된 검증 완료 보조 쟁점의 사실·경계·금지오류만 사용한다. 관점 표어 암기가 아니라 사실 구분, 이해관계, 최소선, 반대전제 저항, 비교판단, 행위자 적용, 불확실성 경계, 답변 판별, 종합 답변으로 이어지는 판단 능력을 학습시킨다. 숨은 사고과정은 쓰지 말고 자립적인 사용자 입력과 검증 가능한 정답만 JSON 객체로 출력한다."""
SYSTEM_VALIDATE = """당신은 K-SOVA 학습데이터의 독립 검수자다. 생성 요청과 분리된 새 요청에서 후보를 원 쟁점에 대조한다. 문체 취향이 아니라 원천 정합성, 과제 적합성, 자립성, 한국의 주권·가치·이익 관련 판단 능력, 정당한 다원성과 경계 보존 여부를 검수한다. 실제 중대 오류만 탈락시키며 JSON 객체 하나만 출력한다."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(value[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("response is not a JSON object")
    return result


def candidate_id(issue_id: str, task: str) -> str:
    suffix = hashlib.sha256(f"{issue_id}\0{task}".encode()).hexdigest()[:16]
    return f"KSEXT-{suffix}"


def task_spec(task: str, issue_id: str) -> dict[str, Any]:
    requirements = {
        "frame_adversarial_resistance": (
            "사실 일부는 맞아 보이는 강한 반대 주장을 user_ko에 제시한다. assistant_ko는 "
            "타당한 부분과 잘못된 도약을 명시적으로 나누고, 쟁점의 최소선과 필요한 유보를 지킨다."
        ),
        "competing_claim_resolution": (
            "서로 충돌하는 가장 강한 두 주장을 공정하게 제시한다. assistant_ko는 사실, 권리, "
            "한국에 귀속되는 비용·위험, 조건을 기준으로 비교한 뒤 허위 대등성 없이 결론을 낸다."
        ),
        "actor_decision_application": (
            "쟁점에 실제로 관련된 한국의 구체적 행위자 하나가 취할 결정을 묻는다. assistant_ko는 "
            "그 행위자의 권한·이익·비용·제약 범위 안에서 행동과 조건을 제시하고 다른 행위자 권한을 혼동하지 않는다."
        ),
        "uncertainty_consequence_boundary": (
            "확인된 사실과 불확실한 요소, 결론이 바뀌는 조건을 묻는다. assistant_ko는 '확인:', "
            "'불확실:', '전환 조건:', '결과:'를 구분하고 원자료 밖의 수치·효과를 만들지 않는다."
        ),
        "hard_adversarial_stress": (
            "원자료의 참인 사실을 포함해 매우 그럴듯하지만 결론이 과도한 반대 전제를 제시한다. "
            "assistant_ko는 참인 전제를 인정하면서도 잘못된 결론 도약을 정확히 차단한다."
        ),
        "hard_near_miss_discrimination": (
            "후보 A와 B가 모두 그럴듯하고 대부분 맞지만 하나만 미묘한 금지 오류·유보 누락을 갖게 한다. "
            "assistant_ko는 더 나은 후보를 판정하고 결정적 차이만 설명한다."
        ),
        "hard_condition_consequence_shift": (
            "원 쟁점의 핵심 조건 하나가 달라지는 가정을 제시한다. assistant_ko는 무엇이 유지되고 "
            "어떤 임계조건에서 결론·비용·위험 평가가 바뀌는지 한정한다. 새 사실을 단정하지 않는다."
        ),
        "hard_actor_shift_application": (
            "원 쟁점의 원칙을 다른 한국 행위자에게 옮겨 적용하게 한다. assistant_ko는 공통 원칙은 "
            "유지하되 행위자별 권한·책임·비용 차이 때문에 달라지는 행동을 구분한다."
        ),
    }[task]
    return {
        "candidate_id": candidate_id(issue_id, task),
        "task": task,
        "capability_stage": STAGE[task],
        "variant": "standard_core" if task in STANDARD_MISSING else "expanded_hard",
        "requirements_ko": requirements,
    }


def eligible_prior(
    dossiers: list[dict[str, Any]], current: list[dict[str, Any]], prior: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dossier_by_id = {str(row["issue_id"]): row for row in dossiers}
    current_pairs = {
        (row["messages"][0]["content"].strip(), row["messages"][1]["content"].strip())
        for row in current
    }
    by_slot: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prior:
        issue_id = str(row.get("material_id") or "")
        task = PRIOR_SKILL_MAP.get(str(row.get("learning_skill") or ""))
        if issue_id not in dossier_by_id or task is None:
            continue
        if not (row.get("benchmark_holdout_gate") or {}).get("effective_pass"):
            continue
        if not (row.get("final_training_gate") or {}).get("effective_pass"):
            continue
        training_audit = row.get("training_example_audit") or {}
        judgment = training_audit.get("judgment") or {}
        source_item_audit = training_audit.get("source_item_audit") or {}
        if not (
            judgment.get("verdict") == "pass"
            or source_item_audit.get("effective_pass") is True
        ):
            continue
        pair = (str(row.get("question_ko") or "").strip(), str(row.get("answer_ko") or "").strip())
        if not all(pair) or pair in current_pairs:
            continue
        by_slot[(issue_id, task)].append(row)
    selected: list[dict[str, Any]] = []
    for slot, rows in sorted(by_slot.items()):
        selected.append(
            sorted(
                rows,
                key=lambda row: (
                    -int(((row.get("training_example_audit") or {}).get("judgment") or {}).get("quality_score_0_100_integer") or 0),
                    str(row.get("example_id")),
                ),
            )[0]
        )
    return selected


def generation_prompt(
    dossier: dict[str, Any], requested: list[dict[str, Any]],
) -> str:
    shape = {
        "issue_id": dossier["issue_id"],
        "examples": [
            {
                "candidate_id": row["candidate_id"],
                "task": row["task"],
                "capability_stage": row["capability_stage"],
                "variant": row["variant"],
                "user_ko": "자립적인 사용자 입력",
                "assistant_ko": "검증 가능한 모범 답변",
            }
            for row in requested
        ],
    }
    return f"""다음 검증 완료 보조 쟁점에서 요청된 학습 예시를 정확히 하나씩 작성하라.

[핵심 규칙]
1. validated_examples에 있는 사실, 최소선, 유보, 경계, 금지 오류만 사용하고 최신 사실이나 법적 효과를 추가하지 않는다.
2. user_ko는 숨은 문맥 없이 이해되어야 하며 '위 자료', '앞의 내용', source_id를 쓰지 않는다.
3. 질문에 '한국 관점에서 답하라', '한국의 입장으로 답하라'처럼 정답 관점을 직접 지시하지 않는다. 상황 자체의 한국 행위자·권리·비용은 구체적으로 적어도 된다.
4. 애국적 수사, 외국 국민 일반화, 현 정부 입장 복창, 무조건적 국익 우선, 장문의 사고과정을 쓰지 않는다.
5. 기존 validated_examples의 질문을 복제하지 않고 판단 능력을 분리해 훈련한다.
6. user_ko는 60~1,500자, assistant_ko는 40~1,200자로 간결하게 쓴다.
7. hard_near_miss_discrimination의 user_ko에는 '후보 A'와 '후보 B'를 모두 넣고 assistant_ko는 '판정: A' 또는 '판정: B'로 시작한다. 정답 위치는 candidate_id 마지막 16진수 값이 짝수면 A, 홀수면 B로 한다.
8. requested_examples 순서와 모든 고정 식별자를 그대로 유지하고 JSON 객체 외에는 출력하지 않는다.

[요청 과제]
{json.dumps(requested, ensure_ascii=False, indent=2)}

[검증 완료 보조 쟁점]
{json.dumps(dossier, ensure_ascii=False, indent=2)}

[출력 형식]
{json.dumps(shape, ensure_ascii=False, indent=2)}"""


def command_prepare(args: argparse.Namespace) -> int:
    dossiers = read_jsonl(args.dossiers)
    current = read_jsonl(args.current)
    prior = read_jsonl(args.prior)
    if len(dossiers) != 32 or len({row["issue_id"] for row in dossiers}) != 32:
        raise ValueError("requires exactly 32 unique frozen dossiers")
    if len(current) != 160 or len({row["metadata"]["issue_id"] for row in current}) != 32:
        raise ValueError("requires the frozen 160-example curriculum")
    prior_selected = eligible_prior(dossiers, current, prior)
    if len(prior_selected) != 17:
        raise ValueError(f"expected 17 reusable prior examples, got {len(prior_selected)}")
    occupied = {
        (str(row["material_id"]), PRIOR_SKILL_MAP[str(row["learning_skill"])])
        for row in prior_selected
    }
    requests: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    for dossier in sorted(dossiers, key=lambda row: str(row["issue_id"])):
        issue_id = str(dossier["issue_id"])
        requested = [
            task_spec(task, issue_id)
            for task in STANDARD_MISSING
            if (issue_id, task) not in occupied
        ] + [task_spec(task, issue_id) for task in EXPANDED]
        plans.extend({"issue_id": issue_id, **row} for row in requested)
        for specification in requested:
            requests.append(
                {
                    "id": f"training_extension_generate__{specification['candidate_id']}",
                    "messages": [
                        {"role": "system", "content": SYSTEM_GENERATE},
                        {"role": "user", "content": generation_prompt(dossier, [specification])},
                    ],
                    "max_tokens": args.max_tokens,
                    "metadata": {
                        "purpose": "training_extension_generation",
                        "prompt_version": GENERATION_PROMPT_VERSION,
                        "issue_id": issue_id,
                        "source_ids": dossier["source_ids"],
                        "category_id": dossier["category_id"],
                        "origin": dossier["origin"],
                        "input_sha256": dossier["input_sha256"],
                        "candidate_ids": [specification["candidate_id"]],
                    },
                }
            )
    standard_new = sum(row["variant"] == "standard_core" for row in plans)
    expanded_new = sum(row["variant"] == "expanded_hard" for row in plans)
    if standard_new != 111 or expanded_new != 128 or len(plans) != 239:
        raise ValueError(
            f"unexpected plan: standard={standard_new}, expanded={expanded_new}, all={len(plans)}"
        )
    args.work.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.work / "prior_reused_core17.jsonl", prior_selected)
    write_jsonl(args.work / "generation_plan.jsonl", plans)
    write_jsonl(args.work / "generation_requests.jsonl", requests)
    report = {
        "schema_version": "ksova-training-extension-plan-1.0",
        "issues": len(dossiers),
        "generation_requests": len(requests),
        "generation_request_unit": "one_candidate",
        "current_examples_preserved": len(current),
        "prior_validated_reused": len(prior_selected),
        "new_standard_slots": standard_new,
        "new_expanded_slots": expanded_new,
        "new_generation_slots": len(plans),
        "target_standard_a": 288,
        "target_expanded_b": 416,
        "input_hashes": {
            "dossiers": file_hash(args.dossiers),
            "current": file_hash(args.current),
            "prior": file_hash(args.prior),
        },
    }
    write_json(args.work / "plan_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def validate_generated_example(
    row: dict[str, Any], planned: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    required = {"candidate_id", "task", "capability_stage", "variant", "user_ko", "assistant_ko"}
    if set(row) != required:
        errors.append("invalid_keys")
    for key in ("candidate_id", "task", "capability_stage", "variant"):
        if row.get(key) != planned.get(key):
            errors.append(f"{key}_mismatch")
    user = row.get("user_ko")
    assistant = row.get("assistant_ko")
    if not isinstance(user, str) or not 40 <= len(user.strip()) <= 1800:
        errors.append("invalid_user")
    if not isinstance(assistant, str) or not 20 <= len(assistant.strip()) <= 1500:
        errors.append("invalid_assistant")
    if isinstance(user, str) and any(term in user for term in ("source_id", "위 자료", "앞의 내용")):
        errors.append("hidden_context_reference")
    if row.get("task") == "hard_near_miss_discrimination":
        if not isinstance(user, str) or "후보 A" not in user or "후보 B" not in user:
            errors.append("candidate_labels_missing")
        expected = "A" if int(str(row.get("candidate_id"))[-1], 16) % 2 == 0 else "B"
        if not isinstance(assistant, str) or not assistant.lstrip().startswith(f"판정: {expected}"):
            errors.append("near_miss_label_mismatch")
    return sorted(set(errors))


def command_parse_generation(args: argparse.Namespace) -> int:
    requests = {str(row["id"]): row for row in read_jsonl(args.requests)}
    plans = {str(row["candidate_id"]): row for row in read_jsonl(args.plan)}
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in args.responses:
        for row in read_jsonl(path):
            attempts[str(row.get("id"))].append(row)
    valid_candidates: list[dict[str, Any]] = []
    invalid_requests: list[dict[str, Any]] = []
    for request_id, request in requests.items():
        expected_ids = list(request["metadata"]["candidate_ids"])
        selected: list[dict[str, Any]] | None = None
        attempt_errors: list[str] = []
        for response in reversed(attempts.get(request_id, [])):
            try:
                payload = parse_object(str(response.get("response") or ""))
                rows = payload.get("examples")
                errors: list[str] = []
                if payload.get("issue_id") != request["metadata"]["issue_id"]:
                    errors.append("issue_id_mismatch")
                if response.get("error") or str(response.get("finish_reason") or "").lower() != "stop":
                    errors.append("runtime_or_finish_error")
                if not isinstance(rows, list):
                    errors.append("examples_not_list")
                    rows = []
                ids = [str(row.get("candidate_id")) for row in rows if isinstance(row, dict)]
                if len(rows) != len(expected_ids) or ids != expected_ids:
                    errors.append("candidate_order_or_coverage")
                for row in rows:
                    if not isinstance(row, dict) or str(row.get("candidate_id")) not in plans:
                        errors.append("unknown_candidate")
                        continue
                    errors.extend(validate_generated_example(row, plans[str(row["candidate_id"])]))
                if not errors:
                    selected = [
                        {
                            **row,
                            "issue_id": request["metadata"]["issue_id"],
                            "source_ids": request["metadata"]["source_ids"],
                            "category_id": request["metadata"]["category_id"],
                            "origin": request["metadata"]["origin"],
                            "input_sha256": request["metadata"]["input_sha256"],
                            "generation": {
                                "model": response.get("model"),
                                "messages_sha256": response.get("messages_sha256"),
                                "prompt_version": GENERATION_PROMPT_VERSION,
                            },
                        }
                        for row in rows
                    ]
                    break
                attempt_errors.extend(errors)
            except (ValueError, json.JSONDecodeError) as error:
                attempt_errors.append(f"parse:{type(error).__name__}")
        if selected is None:
            invalid_requests.append(
                {
                    "request_id": request_id,
                    "issue_id": request["metadata"]["issue_id"],
                    "errors": sorted(set(attempt_errors)) or ["missing_response"],
                }
            )
        else:
            valid_candidates.extend(selected)
    write_jsonl(args.output, valid_candidates)
    write_jsonl(args.invalid, invalid_requests)
    report = {
        "schema_version": "ksova-training-extension-generation-parse-1.0",
        "requests": len(requests),
        "valid_requests": len(requests) - len(invalid_requests),
        "invalid_requests": len(invalid_requests),
        "valid_candidates": len(valid_candidates),
        "expected_candidates": len(plans),
        "complete": not invalid_requests and len(valid_candidates) == len(plans),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 1


def command_prepare_retry(args: argparse.Namespace) -> int:
    """Materialize only failed generation requests as a reproducible retry set."""
    requests = read_jsonl(args.requests)
    invalid = read_jsonl(args.invalid)
    request_by_id = {str(row.get("id") or ""): row for row in requests}
    invalid_ids = [str(row.get("request_id") or "") for row in invalid]
    if len(request_by_id) != len(requests):
        raise ValueError("duplicate original generation request id")
    if not invalid_ids or len(invalid_ids) != len(set(invalid_ids)):
        raise ValueError("retry set is empty or contains duplicate request ids")
    missing = sorted(set(invalid_ids) - set(request_by_id))
    if missing:
        raise ValueError(f"retry ids missing from original requests:{missing}")
    retry_ids = set(invalid_ids)
    retry = [row for row in requests if str(row["id"]) in retry_ids]
    if len(retry) != len(invalid_ids):
        raise ValueError("retry request cardinality mismatch")
    write_jsonl(args.output, retry)
    report = {
        "schema_version": "ksova-generation-retry-set-1.0",
        "pass": True,
        "original_requests": len(requests),
        "failed_requests": len(invalid),
        "retry_requests": len(retry),
        "retry_requests_sha256": file_hash(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def validation_prompt(dossier: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    shape = {
        "issue_id": dossier["issue_id"],
        "judgments": [
            {
                "candidate_id": row["candidate_id"],
                "checks": {
                    "source_grounded": True,
                    "task_matches": True,
                    "self_contained": True,
                    "no_answer_perspective_cue": True,
                    "preserves_limits_and_pluralism": True,
                    "no_government_mimicry_or_nationalist_overreach": True,
                    "no_unsupported_fact_or_legal_claim": True,
                    "teaches_target_decision_skill": True,
                    "nonduplicate_within_issue": True,
                    "variant_difficulty_valid": True,
                },
                "verdict": "pass|fail",
                "fatal_issues_ko": [],
                "reason_ko": "판정 이유",
            }
            for row in candidates
        ],
    }
    compact_candidates = [
        {key: row[key] for key in (
            "candidate_id", "task", "capability_stage", "variant", "user_ko", "assistant_ko"
        )}
        for row in candidates
    ]
    return f"""아래 생성 후보를 검증 완료 보조 쟁점에 대조해 각각 독립 검수하라.

[검수 규칙]
1. 모든 checks는 실제로 충족할 때만 true다. 하나라도 false면 verdict=fail이며 fatal_issues_ko에 구체적 문제를 쓴다.
2. source_grounded: validated_examples의 사실·기준·경계만 사용한다.
3. task_matches와 teaches_target_decision_skill: 이름만 바꾼 종합답변이 아니라 지정 능력을 실제로 훈련한다.
4. self_contained와 no_answer_perspective_cue: 숨은 문맥이 없고, 질문이 '한국 관점으로 답하라'처럼 정답 관점을 직접 지시하지 않는다. 한국 행위자나 한국에 귀속되는 비용을 상황에 적는 것은 허용한다.
5. preserves_limits_and_pluralism: 정당한 조건부 선택과 필요한 유보를 지우지 않는다.
6. 사소한 문체 선호로 탈락시키지 않는다. 다만 근거 없는 사실·법적 효과·권한, 정부 입장 복창, 애국적 과잉, 외국인 일반화는 중대 오류다.
7. nonduplicate_within_issue: 후보끼리 질문과 훈련기술이 실질적으로 구별된다.
8. variant_difficulty_valid: standard_core는 해당 핵심능력을 명확히 훈련하면 true다. expanded_hard는 그럴듯한 오답, 조건·결과 변화, 행위자 이동 등 명시된 난점이 실제로 있어야 true다.
9. 입력 순서와 candidate_id를 그대로 유지하고 JSON 객체 외 텍스트를 출력하지 않는다.

[검증 완료 보조 쟁점]
{json.dumps(dossier, ensure_ascii=False, indent=2)}

[생성 후보]
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

[출력 형식]
{json.dumps(shape, ensure_ascii=False, indent=2)}"""


def command_prepare_validation(args: argparse.Namespace) -> int:
    dossiers = {str(row["issue_id"]): row for row in read_jsonl(args.dossiers)}
    candidates = read_jsonl(args.candidates)
    candidates_by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        candidates_by_issue[str(row["issue_id"])].append(row)
    if set(candidates_by_issue) != set(dossiers):
        raise ValueError("candidate issues do not exactly cover dossiers")
    requests: list[dict[str, Any]] = []
    for issue_id in sorted(dossiers):
        issue_candidates = sorted(
            candidates_by_issue[issue_id],
            key=lambda row: [*STANDARD_MISSING, *EXPANDED].index(row["task"]),
        )
        for candidate in issue_candidates:
            requests.append(
                {
                    "id": f"training_extension_validate__{candidate['candidate_id']}",
                    "messages": [
                        {"role": "system", "content": SYSTEM_VALIDATE},
                        {"role": "user", "content": validation_prompt(dossiers[issue_id], [candidate])},
                    ],
                    "max_tokens": args.max_tokens,
                    "metadata": {
                        "purpose": "training_extension_validation",
                        "prompt_version": VALIDATION_PROMPT_VERSION,
                        "issue_id": issue_id,
                        "candidate_ids": [candidate["candidate_id"]],
                    },
                }
            )
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-training-extension-validation-request-1.0",
        "requests": len(requests),
        "request_unit": "one_candidate",
        "candidates": len(candidates),
        "generation_candidates_sha256": file_hash(args.candidates),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


CHECK_KEYS = {
    "source_grounded", "task_matches", "self_contained",
    "no_answer_perspective_cue", "preserves_limits_and_pluralism",
    "no_government_mimicry_or_nationalist_overreach",
    "no_unsupported_fact_or_legal_claim", "teaches_target_decision_skill",
    "nonduplicate_within_issue", "variant_difficulty_valid",
}


def parse_validation(
    requests_path: Path, response_paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    requests = {str(row["id"]): row for row in read_jsonl(requests_path)}
    attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in response_paths:
        for row in read_jsonl(path):
            attempts[str(row.get("id"))].append(row)
    judgments: dict[str, dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    for request_id, request in requests.items():
        expected = list(request["metadata"]["candidate_ids"])
        selected: list[dict[str, Any]] | None = None
        errors_all: list[str] = []
        for response in reversed(attempts.get(request_id, [])):
            try:
                payload = parse_object(str(response.get("response") or ""))
                rows = payload.get("judgments")
                errors: list[str] = []
                if payload.get("issue_id") != request["metadata"]["issue_id"]:
                    errors.append("issue_id_mismatch")
                if response.get("error") or str(response.get("finish_reason") or "").lower() != "stop":
                    errors.append("runtime_or_finish_error")
                if not isinstance(rows, list):
                    rows = []
                    errors.append("judgments_not_list")
                ids = [str(row.get("candidate_id")) for row in rows if isinstance(row, dict)]
                if len(rows) != len(expected) or ids != expected:
                    errors.append("judgment_order_or_coverage")
                for row in rows:
                    if not isinstance(row, dict):
                        errors.append("judgment_not_object")
                        continue
                    checks = row.get("checks")
                    if not isinstance(checks, dict) or set(checks) != CHECK_KEYS or any(
                        not isinstance(value, bool) for value in (checks or {}).values()
                    ):
                        errors.append("invalid_checks")
                    verdict = row.get("verdict")
                    if verdict not in {"pass", "fail"}:
                        errors.append("invalid_verdict")
                    if isinstance(checks, dict) and verdict == "pass" and not all(checks.values()):
                        errors.append("pass_with_failed_check")
                    if isinstance(checks, dict) and verdict == "fail" and all(checks.values()):
                        errors.append("fail_without_failed_check")
                    if not isinstance(row.get("fatal_issues_ko"), list):
                        errors.append("invalid_fatal_issues")
                    if not isinstance(row.get("reason_ko"), str) or len(row["reason_ko"].strip()) < 8:
                        errors.append("invalid_reason")
                if not errors:
                    selected = rows
                    break
                errors_all.extend(errors)
            except (ValueError, json.JSONDecodeError) as error:
                errors_all.append(f"parse:{type(error).__name__}")
        if selected is None:
            invalid.append(
                {"request_id": request_id, "errors": sorted(set(errors_all)) or ["missing_response"]}
            )
            continue
        for row in selected:
            judgment = {
                **row,
                "issue_id": request["metadata"]["issue_id"],
                "validator_model": attempts[request_id][-1].get("model"),
                "prompt_version": VALIDATION_PROMPT_VERSION,
            }
            judgments[str(row["candidate_id"])] = judgment
    return judgments, invalid


def training_id(issue_id: str, task: str, user: str, assistant: str) -> str:
    suffix = hashlib.sha256(f"{issue_id}\0{task}\0{user}\0{assistant}".encode()).hexdigest()[:16]
    return f"KSSTAGE-{suffix}"


def make_row(
    issue_id: str, task: str, user: str, assistant: str, split: str,
    source_ids: list[str], category_id: str, origin: str, input_sha256: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "training_id": training_id(issue_id, task, user, assistant),
        "issue_id": issue_id,
        "source_ids": source_ids,
        "category_id": category_id,
        "origin": origin,
        "task": task,
        "capability_stage": STAGE[task],
        "curriculum_phase": PHASE[task],
        "split": split,
        "source_isolated_from_core250": True,
        "input_sha256": input_sha256,
    }
    if extra:
        metadata.update(extra)
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "metadata": metadata,
    }


def normalize_frozen_current(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich frozen rows for the expanded curriculum without editing the source.

    The first-pass release predates ``capability_stage`` and uses the legacy
    three-phase labels.  Expanded training needs a single phase convention for
    every row, so preserve the legacy label explicitly and derive the new
    stage/phase from the already-frozen task.
    """
    normalized: list[dict[str, Any]] = []
    for source in rows:
        row = json.loads(json.dumps(source, ensure_ascii=False))
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("frozen row is missing metadata")
        task = str(metadata.get("task") or "")
        if task not in CORE_TASKS:
            raise ValueError(f"unknown frozen task:{task}")
        legacy_phase = metadata.get("curriculum_phase")
        if legacy_phase != PHASE[task]:
            metadata["legacy_curriculum_phase"] = legacy_phase
        metadata["capability_stage"] = STAGE[task]
        metadata["curriculum_phase"] = PHASE[task]
        normalized.append(row)
    return normalized


def write_dataset(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (
        row["metadata"]["issue_id"], row["metadata"]["capability_stage"],
        row["metadata"]["task"], row["metadata"]["training_id"],
    ))
    train = [row for row in rows if row["metadata"]["split"] == "train"]
    dev = [row for row in rows if row["metadata"]["split"] == "dev"]
    outputs: dict[str, list[dict[str, Any]]] = {
        "all_examples.jsonl": rows,
        "train_all.jsonl": train,
        "dev_all.jsonl": dev,
    }
    for phase in range(1, 4):
        phase_name = ("phase1_foundation", "phase2_application", "phase3_synthesis")[phase - 1]
        outputs[f"phase{phase}_train.jsonl"] = [
            row for row in train if row["metadata"]["curriculum_phase"] == phase_name
        ]
        outputs[f"phase{phase}_dev.jsonl"] = [
            row for row in dev if row["metadata"]["curriculum_phase"] == phase_name
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in outputs.items():
        write_jsonl(output_dir / name, values)
    return {
        "rows": len(rows),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "task_counts": dict(sorted(Counter(row["metadata"]["task"] for row in rows).items())),
        "phase_rows": {name: len(values) for name, values in outputs.items()},
        "hashes": {name: file_hash(output_dir / name) for name in outputs},
    }


def command_assemble_provisional(args: argparse.Namespace) -> int:
    """Assemble a clearly labeled mechanical-only dataset under time pressure.

    This path never represents generated examples as independently validated and
    never writes into the exact-count 288/416 release directories.
    """
    current = normalize_frozen_current(read_jsonl(args.current))
    dossiers = {str(row["issue_id"]): row for row in read_jsonl(args.dossiers)}
    prior = read_jsonl(args.prior_reused)
    generated = read_jsonl(args.candidates)
    plans = {
        str(row["candidate_id"]): row for row in read_jsonl(args.plan)
    }
    split_by_issue = {
        str(row["metadata"]["issue_id"]): str(row["metadata"]["split"])
        for row in current
    }
    errors: list[str] = []
    if len(current) != 160:
        errors.append(f"frozen_current_count:{len(current)}")
    if len(prior) != 17:
        errors.append(f"prior_validated_count:{len(prior)}")
    if len(split_by_issue) != 32:
        errors.append(f"frozen_issue_count:{len(split_by_issue)}")

    generated_ids = [str(row.get("candidate_id") or "") for row in generated]
    if len(generated_ids) != len(set(generated_ids)):
        errors.append("duplicate_generated_candidate_id")
    for row in generated:
        candidate = str(row.get("candidate_id") or "")
        planned = plans.get(candidate)
        if planned is None:
            errors.append(f"unplanned_candidate:{candidate}")
            continue
        compact = {
            key: row.get(key) for key in (
                "candidate_id", "task", "capability_stage", "variant",
                "user_ko", "assistant_ko",
            )
        }
        errors.extend(
            f"{candidate}:{value}"
            for value in validate_generated_example(compact, planned)
        )
        if str(row.get("issue_id") or "") not in split_by_issue:
            errors.append(f"unknown_generated_issue:{candidate}")
    if errors:
        raise ValueError(sorted(set(errors)))

    standard_additions: list[dict[str, Any]] = []
    for row in prior:
        issue_id = str(row["material_id"])
        task = PRIOR_SKILL_MAP[str(row["learning_skill"])]
        dossier = dossiers[issue_id]
        standard_additions.append(
            make_row(
                issue_id, task, row["question_ko"], row["answer_ko"],
                split_by_issue[issue_id], list(row["source_ids"]),
                dossier["category_id"], "prior_glm_validated_holdout_safe",
                dossier["input_sha256"],
                {
                    "prior_example_id": row["example_id"],
                    "validator_reused": True,
                    "validation_status": "prior_independently_validated",
                },
            )
        )

    expanded_additions: list[dict[str, Any]] = []
    for row in generated:
        issue_id = str(row["issue_id"])
        value = make_row(
            issue_id, row["task"], row["user_ko"], row["assistant_ko"],
            split_by_issue[issue_id], list(row["source_ids"]), row["category_id"],
            "glm_generated_mechanical_only_unreviewed", row["input_sha256"],
            {
                "candidate_id": row["candidate_id"],
                "variant": row["variant"],
                "validation_status": "mechanical_only_not_llm_reviewed",
                "provisional": True,
            },
        )
        if row["variant"] == "standard_core":
            standard_additions.append(value)
        else:
            expanded_additions.append(value)

    standard = current + standard_additions
    expanded = standard + expanded_additions
    for name, rows in (("standard", standard), ("expanded", expanded)):
        ids = [row["metadata"]["training_id"] for row in rows]
        pairs = [
            tuple(message["content"].strip() for message in row["messages"])
            for row in rows
        ]
        if len(ids) != len(set(ids)):
            errors.append(f"{name}_duplicate_training_id")
        if len(pairs) != len(set(pairs)):
            errors.append(f"{name}_duplicate_message_pair")
        if any(
            not row["metadata"].get("source_isolated_from_core250") for row in rows
        ):
            errors.append(f"{name}_holdout_flag")
    if errors:
        raise ValueError(sorted(set(errors)))

    standard_name = f"standard_a_partial_{len(standard)}"
    expanded_name = f"expanded_b_partial_{len(expanded)}"
    standard_report = write_dataset(args.output / standard_name, standard)
    expanded_report = write_dataset(args.output / expanded_name, expanded)
    report_value = {
        "schema_version": "ksova-9cap-expanded-provisional-assembly-1.0",
        "pass": True,
        "release_eligible": False,
        "provisional": True,
        "independent_llm_validation": "skipped_due_time",
        "mechanical_validation": "passed",
        "frozen_current_examples": len(current),
        "prior_validated_reused": len(prior),
        "new_generated_mechanical_only": len(generated),
        "new_standard_mechanical_only": len(standard_additions) - len(prior),
        "new_expanded_mechanical_only": len(expanded_additions),
        "failed_generation_requests_excluded": len(plans) - len(generated),
        "standard_dataset": {"directory": standard_name, **standard_report},
        "expanded_dataset": {"directory": expanded_name, **expanded_report},
    }
    write_json(args.output / "provisional_assembly_report.json", report_value)
    write_json(args.work / "provisional_assembly_report.json", report_value)
    print(json.dumps(report_value, ensure_ascii=False, indent=2))
    return 0


def command_assemble(args: argparse.Namespace) -> int:
    current = normalize_frozen_current(read_jsonl(args.current))
    dossiers = {str(row["issue_id"]): row for row in read_jsonl(args.dossiers)}
    prior = read_jsonl(args.prior_reused)
    generated = read_jsonl(args.candidates)
    judgments, invalid_validation = parse_validation(args.validation_requests, args.responses)
    candidate_ids = {str(row["candidate_id"]) for row in generated}
    if invalid_validation or set(judgments) != candidate_ids:
        write_jsonl(args.work / "validation_invalid_requests.jsonl", invalid_validation)
        raise ValueError(
            f"validation not structurally complete: invalid={len(invalid_validation)} "
            f"judgments={len(judgments)}/{len(candidate_ids)}"
        )
    failed = [judgments[value] for value in sorted(candidate_ids) if judgments[value]["verdict"] != "pass"]
    write_jsonl(args.work / "validation_judgments.jsonl", [judgments[value] for value in sorted(judgments)])
    write_jsonl(args.work / "validation_failed_candidates.jsonl", failed)
    if failed:
        write_json(
            args.work / "validation_report.json",
            {"complete": True, "passed": len(judgments) - len(failed), "failed": len(failed)},
        )
        print(json.dumps({"validation_failed": len(failed)}, ensure_ascii=False))
        return 2

    split_by_issue = {
        str(row["metadata"]["issue_id"]): str(row["metadata"]["split"])
        for row in current
    }
    if len(split_by_issue) != 32 or Counter(split_by_issue.values()) != Counter({"train": 28, "dev": 4}):
        raise ValueError("frozen split is not 28 train / 4 dev issues")
    standard_additions: list[dict[str, Any]] = []
    for row in prior:
        issue_id = str(row["material_id"])
        task = PRIOR_SKILL_MAP[str(row["learning_skill"])]
        dossier = dossiers[issue_id]
        standard_additions.append(
            make_row(
                issue_id, task, row["question_ko"], row["answer_ko"], split_by_issue[issue_id],
                list(row["source_ids"]), dossier["category_id"],
                "prior_glm_validated_holdout_safe", dossier["input_sha256"],
                {"prior_example_id": row["example_id"], "validator_reused": True},
            )
        )
    expanded_additions: list[dict[str, Any]] = []
    for row in generated:
        issue_id = str(row["issue_id"])
        value = make_row(
            issue_id, row["task"], row["user_ko"], row["assistant_ko"], split_by_issue[issue_id],
            list(row["source_ids"]), row["category_id"],
            "glm_generated_separately_validated", row["input_sha256"],
            {
                "candidate_id": row["candidate_id"],
                "variant": row["variant"],
                "validation": judgments[row["candidate_id"]],
            },
        )
        if row["variant"] == "standard_core":
            standard_additions.append(value)
        else:
            expanded_additions.append(value)
    standard = current + standard_additions
    expanded = standard + expanded_additions

    errors: list[str] = []
    if len(standard) != 288:
        errors.append(f"standard_count:{len(standard)}")
    if len(expanded) != 416:
        errors.append(f"expanded_count:{len(expanded)}")
    for name, rows, tasks in (
        ("standard", standard, CORE_TASKS),
        ("expanded", expanded, (*CORE_TASKS, *EXPANDED)),
    ):
        ids = [row["metadata"]["training_id"] for row in rows]
        pairs = [tuple(message["content"].strip() for message in row["messages"]) for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{name}_duplicate_training_id")
        if len(pairs) != len(set(pairs)):
            errors.append(f"{name}_duplicate_message_pair")
        counts = Counter(row["metadata"]["task"] for row in rows)
        if counts != Counter({task: 32 for task in tasks}):
            errors.append(f"{name}_task_counts:{dict(counts)}")
        if any(not row["metadata"].get("source_isolated_from_core250") for row in rows):
            errors.append(f"{name}_holdout_flag")
    if errors:
        raise ValueError(errors)

    standard_report = write_dataset(args.output / "standard_a_288", standard)
    expanded_report = write_dataset(args.output / "expanded_b_416", expanded)
    common_dev = [row for row in standard if row["metadata"]["split"] == "dev"]
    write_jsonl(args.output / "common_selection_dev.jsonl", common_dev)
    report_value = {
        "schema_version": "ksova-9cap-expanded-assembly-1.0",
        "pass": True,
        "frozen_current_examples": len(current),
        "prior_reused_examples": len(prior),
        "new_generated_validated": len(generated),
        "validator_passed": len(judgments),
        "validator_failed": 0,
        "standard_a": standard_report,
        "expanded_b": expanded_report,
        "common_selection_dev_rows": len(common_dev),
        "common_selection_dev_sha256": file_hash(args.output / "common_selection_dev.jsonl"),
        "dev_issue_ids": sorted(issue for issue, split in split_by_issue.items() if split == "dev"),
    }
    write_json(args.output / "assembly_report.json", report_value)
    write_json(args.work / "validation_report.json", {
        "complete": True, "passed": len(judgments), "failed": 0,
        "judgments_sha256": file_hash(args.work / "validation_judgments.jsonl"),
    })
    print(json.dumps(report_value, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "prepare", "prepare-retry", "parse-generation",
            "prepare-validation", "assemble", "assemble-provisional",
        ],
    )
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--dossiers", type=Path, default=FIRST_WORK / "dossiers.jsonl")
    parser.add_argument("--current", type=Path, default=FIRST_RELEASE / "training/all_examples.jsonl")
    parser.add_argument("--prior", type=Path, default=PRIOR_ACCEPTED)
    parser.add_argument("--max-tokens", type=int, default=7168)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--responses", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--invalid", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--prior-reused", type=Path)
    parser.add_argument("--validation-requests", type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "prepare-retry":
        args.requests = args.requests or args.work / "generation_requests.jsonl"
        args.invalid = args.invalid or args.work / "generation_invalid_requests.jsonl"
        args.output = args.output or args.work / "generation_retry_requests.jsonl"
        args.report = args.report or args.work / "generation_retry_report.json"
        return command_prepare_retry(args)
    if args.command == "parse-generation":
        args.requests = args.requests or args.work / "generation_requests.jsonl"
        args.plan = args.plan or args.work / "generation_plan.jsonl"
        args.output = args.output or args.work / "generated_candidates.jsonl"
        args.invalid = args.invalid or args.work / "generation_invalid_requests.jsonl"
        args.report = args.report or args.work / "generation_parse_report.json"
        if not args.responses:
            raise ValueError("--responses is required")
        return command_parse_generation(args)
    if args.command == "prepare-validation":
        args.candidates = args.candidates or args.work / "generated_candidates.jsonl"
        args.output = args.output or args.work / "validation_requests.jsonl"
        args.report = args.report or args.work / "validation_request_report.json"
        return command_prepare_validation(args)
    if args.command == "assemble-provisional":
        args.candidates = args.candidates or args.work / "generated_candidates.jsonl"
        args.plan = args.plan or args.work / "generation_plan.jsonl"
        args.prior_reused = args.prior_reused or args.work / "prior_reused_core17.jsonl"
        args.output = args.output or args.work.parent / "datasets"
        return command_assemble_provisional(args)
    args.candidates = args.candidates or args.work / "generated_candidates.jsonl"
    args.prior_reused = args.prior_reused or args.work / "prior_reused_core17.jsonl"
    args.validation_requests = args.validation_requests or args.work / "validation_requests.jsonl"
    args.output = args.output or args.work.parent / "datasets"
    if not args.responses:
        raise ValueError("--responses is required")
    return command_assemble(args)


if __name__ == "__main__":
    raise SystemExit(main())
