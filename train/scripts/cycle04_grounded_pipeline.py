#!/usr/bin/env python3
"""Build strictly grounded complete-answer SFT data for K-SOVA cycle 4.

The script is intentionally stage-oriented: every model call is represented by
an immutable request JSONL, and every parser refuses duplicate or unexpected
prompt IDs.  This keeps generation, validation, one-shot repair, and final
materialization independently auditable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


EXPERIMENT_ID = "method_search_20260816_cycle04_grounded_sft_dpo"
EXPECTED_HASHES = {
    "issues": "97dfdb057d0967fe267ac92a409dc85dc09e129299e223a33baad78d84e24e13",
    "sources": "d77c9a5edd7035461f1a88615daecc4829688672e47aed0eb3c06ff34ec671bd",
    "train_examples": "164e419f67ace0febce3a9d60b7b3666388908534b0964d7c86a38e2bd62c01c",
    "benchmark": "0ca49e7bc94178300c0309919d63a51c0496d84d6e352501d350f0deac494032",
    "selected_issues": "c46a404421795e6f52fab378075e4ac7fa251b2de2b9dc547e5597de65ca410b",
    "prompts": "15489e2c8d5fd4dc2daa60c5113b51eb56c8b71ca0cb8ff88003876540148358",
}
VALIDATION_FLAGS = (
    "factually_bounded",
    "answers_question_directly",
    "korean_stake_clear",
    "counterargument_addressed",
    "conditions_and_costs_present",
    "clear_judgment",
    "natural_final_answer",
    "no_government_mimicry_or_nationalist_overreach",
)
NEGATIVE_VALIDATION_FLAGS = (
    "rejected_factually_bounded",
    "assigned_defect_present",
    "other_capabilities_preserved",
    "single_material_defect",
    "length_and_style_balanced",
    "preference_direction_clear",
    "no_artificial_meta_language",
)
DEFECT_TYPES = (
    "korean_stake_omission",
    "counterargument_omission",
    "condition_cost_omission",
    "conclusion_evasion",
)
LEGACY_TASK_PRIORITY = (
    "final_synthesis",
    "competing_claim_resolution",
    "frame_adversarial_resistance",
    "uncertainty_consequence_boundary",
    "baseline_and_range",
    "korean_stake_extraction",
)
MIN_ANSWER_CHARS = 350
MAX_ANSWER_CHARS = 2400


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def has_unexplained_foreign_script(text: str) -> bool:
    """Allow short parenthetical Hanja glosses, reject other Han/Kana leakage."""
    without_hanja_glosses = re.sub(
        r"\([^()]{0,30}[一-龥][^()]{0,30}\)", "", text
    )
    return bool(re.search(r"[一-龥ぁ-ゟァ-ヿ]", without_hanja_glosses))


def unique_nonempty(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalized(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    attempts = [candidate]
    left, right = candidate.find("{"), candidate.rfind("}")
    if left >= 0 and right > left:
        attempts.append(candidate[left : right + 1])
    errors: list[str] = []
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
            errors.append("not_object")
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
    raise ValueError("could not parse JSON object: " + " | ".join(errors))


def assert_hash(path: Path, key: str) -> None:
    actual = sha256(path)
    expected = EXPECTED_HASHES[key]
    if actual != expected:
        raise ValueError(f"{key} hash mismatch: {actual} != {expected}")


def source_url(row: dict[str, Any]) -> str:
    return normalized(row.get("normalized_url") or row.get("url"))


def answer_generation_system() -> str:
    return """당신은 K-SOVA 학습용 완성답변 작성자다.

K-SOVA는 한국 관련 단순 지식이나 정부 입장 암기를 평가하지 않는다. 외국과 권리·규범·이해관계가 충돌할 수 있는 사안에서 한국 사회에 형성된 담론을 바탕으로 사실, 권리와 이해관계, 상대 논거, 판단, 비용과 조건을 함께 구성하는 능력을 다룬다.

입력의 evidence_packet은 사용할 수 있는 사실과 담론의 완전한 경계다. verified_facts_ko 또는 validated_reference_answers_ko가 뒷받침하지 않는 인명·기관·수치·날짜·법적 효과를 새로 단정하지 않는다. documented_positions_ko는 행위자의 주장으로만 취급하고 객관적 사실처럼 확대하지 않는다. 사용자 질문에 포함된 전제가 근거 패킷보다 강하면 그 전제를 그대로 확정하지 말고 필요한 한계를 밝힌다.

답변은 질문에 직접 답하면서 다음을 자연스럽게 통합한다: 확인 가능한 사실과 주장 구분, 한국의 구체적 권리·책임·이해관계, 가장 강한 반대 논거, 명확한 종합 판단, 그 판단이 성립하는 조건·비용·불확실성. 공식 대변인처럼 말하거나 국수주의적으로 과장하지 않는다. 카드 항목을 나열하는 메타 답변이나 단계 학습용 축약문이 아니라 실제 사용자에게 줄 독립적인 최종 답변이어야 한다.

한국어 700~1400자를 목표로 하고, 필요하면 350~2400자 안에서 조정한다. 답변 본문만 출력한다. JSON, 근거 패킷, 작성 지침, 메타 해설은 출력하지 않는다."""


def answer_validation_system() -> str:
    return """당신은 K-SOVA 완성답변 데이터의 독립 검수자다. 답변을 수정하지 말고 질문, evidence_packet, candidate_answer만 대조한다. evidence_packet 밖의 지식을 보태 판정하지 않는다.

각 검사는 다음 뜻이다.
- factually_bounded: 검증 근거 밖의 인명·기관·수치·날짜·법적 효과를 새로 단정하지 않고, 행위자 주장과 사실을 구분한다.
- answers_question_directly: 질문의 핵심 판단에 직접 답한다.
- korean_stake_clear: 한국 또는 한국 사회 당사자의 구체적 권리·책임·비용·선택권 중 관련 요소가 분명하다.
- counterargument_addressed: 가장 강한 상대 논거나 충돌 가치를 공정하게 제시하고 응답한다.
- conditions_and_costs_present: 판단의 조건, 비용 또는 불확실성 경계를 구체적으로 제시한다.
- clear_judgment: 양쪽 소개만 하고 회피하지 않으며 방어 가능한 결론을 낸다.
- natural_final_answer: 축약된 단계 과제, 채점표, 카드 복사물이 아니라 자연스럽고 독립적인 최종 답변이다.
- no_government_mimicry_or_nationalist_overreach: 정부의 공식 대변인처럼 가장하거나 국수주의적으로 과장하지 않는다.

불확실하면 false다. 모든 불리언이 true일 때만 verdict는 pass이며, 이때 reasons_ko는 정확히 "모든 검사 통과."라고 쓴다. 하나라도 false이면 verdict는 fail이고 failed_checks에 false인 키를 모두 넣으며 reasons_ko에 근거를 간단히 쓴다. JSON 객체 하나만 출력한다.

출력 키: prompt_id, factually_bounded, answers_question_directly, korean_stake_clear, counterargument_addressed, conditions_and_costs_present, clear_judgment, natural_final_answer, no_government_mimicry_or_nationalist_overreach, failed_checks, verdict, reasons_ko"""


def repair_system() -> str:
    return """당신은 검수에 실패한 K-SOVA 완성답변을 한 번만 다시 작성하는 복구 편집자다.

질문과 evidence_packet의 경계 안에서 previous_answer와 failure_reasons를 반영해 독립적인 최종 답변을 다시 쓴다. 근거 밖의 인명·기관·수치·날짜·법적 효과를 새로 단정하지 않는다. 사실과 행위자 주장을 구분하고, 한국의 구체적 권리·책임·이해관계, 가장 강한 상대 논거, 명확한 판단, 조건·비용·불확실성 경계를 질문에 필요한 범위에서 자연스럽게 통합한다. 공식 대변인처럼 말하거나 국수주의적으로 과장하지 않는다. 이전 답변이 비어 있거나 길이 제한을 어겼으면 처음부터 완성답변을 작성한다.

한국어 700~1400자를 목표로 하고 반드시 350~2400자 안에 둔다. 답변 본문만 출력한다. JSON이나 메타 해설은 출력하지 않는다."""


def negative_generation_system() -> str:
    return """당신은 K-SOVA DPO용 hard negative 편집자다.

입력의 chosen_response는 독립 검수를 통과한 완성답변이다. evidence_packet의 사실 경계와 chosen의 자연스러운 어조·구조·길이를 보존하면서 assigned_defect 하나만 실질적으로 만든 rejected_response를 작성한다. 새 사실 오류, 근거 밖의 인명·기관·수치·날짜·법적 효과, 공격적 표현, 정부 대변인 흉내, 메타 해설을 넣지 않는다. 단순히 답변 전체를 짧게 줄이거나 문장을 망가뜨려서는 안 된다.

결함별 편집 방식:
- korean_stake_omission: 일반 원칙과 상대 논거는 유지하되 한국 또는 한국 사회 당사자의 구체적 권리·책임·비용·선택권을 모호한 일반론으로 약화한다.
- counterargument_omission: 사실·한국의 이해관계·결론·조건은 유지하되 가장 강한 상대 논거를 피상적 언급으로 약화한다.
- condition_cost_omission: 사실·한국의 이해관계·상대 논거·결론은 유지하되 결론의 성립 조건, 비용 또는 불확실성 경계를 제거해 무조건적 판단으로 만든다.
- conclusion_evasion: 사실·한국의 이해관계·상대 논거·조건은 유지하되 최종 판단을 양쪽 소개나 추가 검토 필요라는 회피로 바꾼다.

rejected는 chosen 문자 길이의 75~115%를 목표로 하고 최소 350자, 최대 2400자다. JSON 객체 하나만 출력한다. 키는 prompt_id, defect_type, rejected_response, edit_summary_ko이며 defect_type은 입력 assigned_defect와 정확히 같아야 한다."""


def negative_validation_system() -> str:
    return """당신은 K-SOVA DPO hard negative의 독립 검수자다. 질문, evidence_packet, chosen_response, assigned_defect, rejected_response를 대조하며 답변을 고치지 않는다.

검사 기준:
- rejected_factually_bounded: rejected가 chosen과 동일한 사실 경계를 지키고 새 오류나 근거 밖 구체성을 만들지 않는다.
- assigned_defect_present: assigned_defect가 실제로 명확히 존재한다.
- other_capabilities_preserved: 지정 결함 이외의 사실성, 질문 직접성, 나머지 K-SOVA 구성 요소와 답변 완결성이 유지된다.
- single_material_defect: 우열을 만드는 실질 결함이 하나뿐이다.
- length_and_style_balanced: 길이·문체·유창성 차이 자체가 선호 신호가 아니다.
- preference_direction_clear: chosen이 rejected보다 분명히 낫다.
- no_artificial_meta_language: 어느 답변에도 카드·채점·결함을 언급하는 인위적 메타 표현이 없다.

불확실하면 false다. 모든 불리언이 true일 때만 verdict는 pass이며 reasons_ko는 정확히 "모든 검사 통과."다. 하나라도 false이면 verdict는 fail이고 failed_checks에 false인 키를 모두 넣는다. JSON 객체 하나만 출력한다.

출력 키: prompt_id, rejected_factually_bounded, assigned_defect_present, other_capabilities_preserved, single_material_defect, length_and_style_balanced, preference_direction_clear, no_artificial_meta_language, failed_checks, verdict, reasons_ko"""


def build_evidence_packet(
    issue: dict[str, Any],
    examples_by_issue: dict[str, list[dict[str, Any]]],
    source_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    issue_id = str(issue["issue_id"])
    card = issue.get("issue_card") or {}
    assessments = [
        row for row in card.get("validated_source_assessments") or []
        if row.get("usable") is not False
    ]
    verified_facts: list[dict[str, str]] = []
    for assessment in assessments:
        source_id = str(assessment.get("source_id") or "")
        for claim in assessment.get("supported_claims_ko") or []:
            text = normalized(claim)
            if text:
                verified_facts.append({"claim_ko": text, "source_id": source_id})

    packet: dict[str, Any] = {
        "issue_id": issue_id,
        "title_ko": normalized(card.get("title_ko")),
        "temporal_scope": normalized(card.get("temporal_scope")) or None,
        "decision_reference_ko": {
            key: card.get(key)
            for key in (
                "sovereign_stake_ko",
                "decision_core_ko",
                "korean_reference_direction_ko",
                "acceptable_range_ko",
                "failure_direction_ko",
            )
            if card.get(key)
        },
    }
    documented_positions: list[dict[str, Any]] = []
    for kind, rows in (
        ("external_position", card.get("external_positions_ko") or []),
        ("korean_discourse", card.get("korean_discourse_ko") or []),
    ):
        for row in rows:
            if not isinstance(row, dict) or not normalized(row.get("claim_ko")):
                continue
            documented_positions.append({
                "kind": kind,
                "actor": normalized(row.get("actor")),
                "role": normalized(row.get("role")) or None,
                "claim_ko": normalized(row.get("claim_ko")),
                "source_ids": [str(value) for value in row.get("source_ids") or []],
            })
    if documented_positions:
        packet["documented_positions_ko"] = documented_positions

    if verified_facts:
        deduplicated: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in verified_facts:
            key = (row["claim_ko"], row["source_id"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(row)
        packet["verified_facts_ko"] = deduplicated
        evidence_mode = "validated_source_assessments"
    else:
        examples = examples_by_issue.get(issue_id) or []
        by_task = {
            str((row.get("metadata") or {}).get("task")): row for row in examples
        }
        references: list[dict[str, str]] = []
        for task in LEGACY_TASK_PRIORITY:
            row = by_task.get(task)
            messages = (row or {}).get("messages") or []
            if len(messages) != 2 or messages[1].get("role") != "assistant":
                continue
            answer = normalized(messages[1].get("content"))
            if answer:
                references.append({"task": task, "answer_ko": answer})
        if not references:
            raise ValueError(f"legacy issue has no validated reference answers: {issue_id}")
        packet["validated_reference_answers_ko"] = references
        source_notes: list[dict[str, Any]] = []
        for source_id in issue.get("source_ids") or []:
            source = source_by_id.get(str(source_id))
            if source is None:
                raise ValueError(f"missing source {source_id} for {issue_id}")
            claims = unique_nonempty(source.get("evidence_claims") or [])
            summary = normalized(source.get("evidence_summary_ko"))
            if claims or summary:
                source_notes.append({
                    "source_id": str(source_id),
                    "evidence_claims_ko": claims,
                    "evidence_summary_ko": summary or None,
                })
        if source_notes:
            packet["source_registry_notes_ko"] = source_notes
        evidence_mode = "validated_training_references"

    if not packet.get("decision_reference_ko"):
        packet.pop("decision_reference_ko", None)
    return packet, evidence_mode


def prepare(args: argparse.Namespace) -> int:
    for key, path in (
        ("issues", args.issues),
        ("sources", args.sources),
        ("train_examples", args.train_examples),
        ("benchmark", args.benchmark),
        ("selected_issues", args.selected_issues),
        ("prompts", args.prompts),
    ):
        assert_hash(path, key)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")

    issues = read_jsonl(args.issues)
    sources = read_jsonl(args.sources)
    examples = read_jsonl(args.train_examples)
    benchmark = read_jsonl(args.benchmark)
    selected = read_jsonl(args.selected_issues)
    prompts = read_jsonl(args.prompts)
    if len(issues) != 500 or len(sources) != 1973 or len(examples) != 806:
        raise ValueError("canonical input row count mismatch")
    if len(benchmark) != 480 or len(selected) != 150 or len(prompts) != 600:
        raise ValueError("frozen experiment input row count mismatch")

    source_by_id = {str(row["source_id"]): row for row in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("duplicate canonical source IDs")
    selected_by_id = {str(row["issue_id"]): row for row in selected}
    prompt_by_id = {str(row["prompt_id"]): row for row in prompts}
    if len(selected_by_id) != 150 or len(prompt_by_id) != 600:
        raise ValueError("duplicate issue or prompt IDs")
    examples_by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        examples_by_issue[str((row.get("metadata") or {}).get("issue_id"))].append(row)

    benchmark_issue_ids = {str(row["issue_id"]) for row in benchmark}
    benchmark_source_ids = {
        str(source_id) for row in benchmark for source_id in row.get("source_ids") or []
    }
    benchmark_urls = {
        source_url(source_by_id[source_id]) for source_id in benchmark_source_ids
        if source_id in source_by_id and source_url(source_by_id[source_id])
    }
    benchmark_content_hashes = {
        normalized(source_by_id[source_id].get("content_sha256"))
        for source_id in benchmark_source_ids if source_id in source_by_id
        and normalized(source_by_id[source_id].get("content_sha256"))
    }
    for issue in selected:
        issue_id = str(issue["issue_id"])
        if issue_id in benchmark_issue_ids:
            raise ValueError(f"benchmark issue leaked into training pool: {issue_id}")
        issue_source_ids = {str(value) for value in issue.get("source_ids") or []}
        if issue_source_ids - set(source_by_id):
            raise ValueError(f"missing selected source for {issue_id}")
        if issue_source_ids & benchmark_source_ids:
            raise ValueError(f"benchmark source ID overlap for {issue_id}")
        issue_urls = {source_url(source_by_id[value]) for value in issue_source_ids}
        issue_hashes = {
            normalized(source_by_id[value].get("content_sha256"))
            for value in issue_source_ids
            if normalized(source_by_id[value].get("content_sha256"))
        }
        if issue_urls & benchmark_urls or issue_hashes & benchmark_content_hashes:
            raise ValueError(f"benchmark source content overlap for {issue_id}")

    prompts_per_issue = Counter(str(row["issue_id"]) for row in prompts)
    if set(prompts_per_issue) != set(selected_by_id) or set(prompts_per_issue.values()) != {4}:
        raise ValueError("expected exactly four prompts for every selected issue")
    evidence_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    for prompt in sorted(prompts, key=lambda row: str(row["prompt_id"])):
        prompt_id = str(prompt["prompt_id"])
        issue_id = str(prompt["issue_id"])
        issue = selected_by_id.get(issue_id)
        if issue is None or prompt.get("split") != issue.get("split"):
            raise ValueError(f"prompt issue/split mismatch: {prompt_id}")
        packet, evidence_mode = build_evidence_packet(
            issue, examples_by_issue, source_by_id
        )
        mode_counts[evidence_mode] += 1
        packet_hash = hashlib.sha256(
            json.dumps(packet, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        evidence_rows.append({
            "prompt_id": prompt_id,
            "issue_id": issue_id,
            "domain_id": prompt["domain_id"],
            "split": prompt["split"],
            "prompt_form": prompt["prompt_form"],
            "question_ko": prompt["question_ko"],
            "evidence_mode": evidence_mode,
            "evidence_packet_sha256": packet_hash,
            "evidence_packet": packet,
        })
        payload = {
            "prompt_id": prompt_id,
            "question_ko": prompt["question_ko"],
            "evidence_packet": packet,
        }
        requests.append({
            "id": f"cycle04_grounded_answer__{prompt_id}",
            "messages": [
                {"role": "system", "content": answer_generation_system()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 2048,
            "metadata": {
                "purpose": "ksova_cycle04_grounded_complete_answer",
                "experiment_id": EXPERIMENT_ID,
                "prompt_id": prompt_id,
                "issue_id": issue_id,
                "domain_id": prompt["domain_id"],
                "split": prompt["split"],
                "prompt_form": prompt["prompt_form"],
                "evidence_mode": evidence_mode,
                "evidence_packet_sha256": packet_hash,
            },
        })

    evidence_path = args.output_dir / "evidence_packets.jsonl"
    requests_path = args.output_dir / "answer_generation_requests.jsonl"
    atomic_jsonl(evidence_path, evidence_rows)
    atomic_jsonl(requests_path, requests)
    report = {
        "schema_version": "ksova-cycle04-grounded-preparation-1.0",
        "experiment_id": EXPERIMENT_ID,
        "issues": len(selected),
        "prompts": len(prompts),
        "train_prompts": sum(row["split"] == "train" for row in prompts),
        "dev_prompts": sum(row["split"] == "dev" for row in prompts),
        "evidence_mode_counts": dict(mode_counts),
        "domain_prompt_counts": dict(Counter(row["domain_id"] for row in prompts)),
        "prompt_form_counts": dict(Counter(row["prompt_form"] for row in prompts)),
        "isolation": {
            "issue_id_overlap": 0,
            "source_id_overlap": 0,
            "normalized_url_overlap": 0,
            "content_sha256_overlap": 0,
        },
        "outputs": {
            "evidence_packets": str(evidence_path),
            "evidence_packets_sha256": sha256(evidence_path),
            "answer_generation_requests": str(requests_path),
            "answer_generation_requests_sha256": sha256(requests_path),
        },
    }
    atomic_json(args.output_dir / "preparation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def evidence_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    index = {str(row["prompt_id"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate evidence prompt IDs")
    return index


def response_prompt_id(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("prompt_id") or "")


def build_validation(args: argparse.Namespace) -> int:
    evidence = evidence_index(args.evidence_packets)
    responses = read_jsonl(args.responses)
    prompt_ids = [response_prompt_id(row) for row in responses]
    if not all(prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("response prompt IDs are empty or duplicated")
    unexpected = sorted(set(prompt_ids) - set(evidence))
    if unexpected:
        raise ValueError(f"responses contain unexpected prompt IDs: {unexpected[:5]}")

    candidates: list[dict[str, Any]] = []
    structural_rejections: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for response in sorted(responses, key=response_prompt_id):
        prompt_id = response_prompt_id(response)
        context = evidence[prompt_id]
        answer = str(response.get("response") or "").strip()
        reasons: list[str] = []
        if response.get("error"):
            reasons.append("generation_error")
        if str(response.get("finish_reason") or "").casefold() != "stop":
            reasons.append("finish_reason_not_stop")
        if len(answer) < MIN_ANSWER_CHARS:
            reasons.append(f"answer_too_short:{len(answer)}")
        if len(answer) > MAX_ANSWER_CHARS:
            reasons.append(f"answer_too_long:{len(answer)}")
        if has_unexplained_foreign_script(answer):
            reasons.append("foreign_script_outside_parenthetical_gloss")
        candidate = {
            "prompt_id": prompt_id,
            "issue_id": context["issue_id"],
            "domain_id": context["domain_id"],
            "split": context["split"],
            "prompt_form": context["prompt_form"],
            "question_ko": context["question_ko"],
            "answer_ko": answer,
            "answer_origin": args.candidate_origin,
            "answer_model": response.get("model"),
            "answer_characters": len(answer),
            "evidence_mode": context["evidence_mode"],
            "evidence_packet_sha256": context["evidence_packet_sha256"],
            "generation_messages_sha256": response.get("messages_sha256"),
        }
        if reasons:
            candidate["failure_reasons"] = reasons
            structural_rejections.append(candidate)
            continue
        candidates.append(candidate)
        payload = {
            "prompt_id": prompt_id,
            "question_ko": context["question_ko"],
            "evidence_packet": context["evidence_packet"],
            "candidate_answer": answer,
        }
        requests.append({
            "id": f"{args.validation_id_prefix}{prompt_id}",
            "messages": [
                {"role": "system", "content": answer_validation_system()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 768,
            "metadata": {
                "purpose": "ksova_cycle04_grounded_answer_validation",
                "experiment_id": EXPERIMENT_ID,
                "prompt_id": prompt_id,
                "issue_id": context["issue_id"],
                "domain_id": context["domain_id"],
                "split": context["split"],
                "candidate_origin": args.candidate_origin,
                "evidence_packet_sha256": context["evidence_packet_sha256"],
            },
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidates.jsonl"
    rejected_path = args.output_dir / "structural_rejections.jsonl"
    request_path = args.output_dir / "validation_requests.jsonl"
    atomic_jsonl(candidate_path, candidates)
    atomic_jsonl(rejected_path, structural_rejections)
    atomic_jsonl(request_path, requests)
    report = {
        "schema_version": "ksova-cycle04-validation-build-1.0",
        "responses": len(responses),
        "structurally_valid": len(candidates),
        "structurally_rejected": len(structural_rejections),
        "validation_requests": len(requests),
        "answer_character_summary": {
            "minimum": min((row["answer_characters"] for row in candidates), default=None),
            "maximum": max((row["answer_characters"] for row in candidates), default=None),
            "mean": round(sum(row["answer_characters"] for row in candidates) / len(candidates), 2) if candidates else None,
        },
        "structural_failure_counts": dict(Counter(
            reason.split(":", 1)[0]
            for row in structural_rejections for reason in row["failure_reasons"]
        )),
        "outputs": {
            "candidates_sha256": sha256(candidate_path),
            "structural_rejections_sha256": sha256(rejected_path),
            "validation_requests_sha256": sha256(request_path),
        },
    }
    atomic_json(args.output_dir / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_validation(args: argparse.Namespace) -> int:
    candidates = read_jsonl(args.candidates)
    responses = read_jsonl(args.responses)
    candidate_by_id = {str(row["prompt_id"]): row for row in candidates}
    response_by_prompt_id = {response_prompt_id(row): row for row in responses}
    if len(candidate_by_id) != len(candidates) or len(response_by_prompt_id) != len(responses):
        raise ValueError("duplicate candidate or validation response prompt IDs")
    if set(response_by_prompt_id) - set(candidate_by_id):
        raise ValueError("validation responses contain unexpected prompt IDs")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    false_counts: Counter[str] = Counter()
    for prompt_id, candidate in sorted(candidate_by_id.items()):
        response = response_by_prompt_id.get(prompt_id)
        failures: list[str] = []
        parsed: dict[str, Any] | None = None
        if response is None:
            failures.append("missing_validation_response")
        elif response.get("error"):
            failures.append("validation_response_error")
        elif str(response.get("finish_reason") or "").casefold() != "stop":
            failures.append("validation_finish_reason_not_stop")
        else:
            try:
                parsed = parse_json_object(str(response.get("response") or ""))
            except ValueError as exc:
                failures.append(f"validation_json_parse:{exc}")
        if parsed is not None:
            if str(parsed.get("prompt_id")) != prompt_id:
                failures.append("validation_prompt_id_mismatch")
            for flag in VALIDATION_FLAGS:
                if type(parsed.get(flag)) is not bool:
                    failures.append(f"validation_nonboolean:{flag}")
                elif parsed[flag] is not True:
                    failures.append(f"validator_false:{flag}")
                    false_counts[flag] += 1
            failed_checks = parsed.get("failed_checks")
            if not isinstance(failed_checks, list):
                failures.append("failed_checks_not_array")
            elif failed_checks:
                failures.append("failed_checks_nonempty")
            if parsed.get("verdict") != "pass":
                failures.append("verdict_not_pass")
            if parsed.get("reasons_ko") != "모든 검사 통과.":
                failures.append("pass_reason_not_canonical")
        enriched = dict(candidate)
        enriched["validator_model"] = response.get("model") if response else None
        enriched["validation"] = parsed
        enriched["validation_failures"] = failures
        if not failures:
            accepted.append(enriched)
        else:
            rejected.append(enriched)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.output_dir / "accepted.jsonl"
    rejected_path = args.output_dir / "rejected.jsonl"
    atomic_jsonl(accepted_path, accepted)
    atomic_jsonl(rejected_path, rejected)
    report = {
        "schema_version": "ksova-cycle04-validation-parse-1.0",
        "candidates": len(candidates),
        "responses": len(responses),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": round(len(accepted) / len(candidates), 6) if candidates else 0,
        "validator_false_counts": dict(false_counts),
        "parse_failure_counts": dict(Counter(
            failure.split(":", 1)[0]
            for row in rejected for failure in row["validation_failures"]
        )),
        "outputs": {
            "accepted_sha256": sha256(accepted_path),
            "rejected_sha256": sha256(rejected_path),
        },
    }
    atomic_json(args.output_dir / "parse_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_repair(args: argparse.Namespace) -> int:
    evidence = evidence_index(args.evidence_packets)
    structural = read_jsonl(args.structural_rejections)
    validation = read_jsonl(args.validation_rejections)
    combined = structural + validation
    by_id = {str(row["prompt_id"]): row for row in combined}
    if len(by_id) != len(combined):
        raise ValueError("repair inputs overlap or contain duplicate prompt IDs")
    requests: list[dict[str, Any]] = []
    for prompt_id, row in sorted(by_id.items()):
        context = evidence[prompt_id]
        reasons = list(row.get("failure_reasons") or []) + list(
            row.get("validation_failures") or []
        )
        validation_object = row.get("validation") or {}
        if validation_object.get("reasons_ko"):
            reasons.append(str(validation_object["reasons_ko"]))
        payload = {
            "prompt_id": prompt_id,
            "question_ko": context["question_ko"],
            "evidence_packet": context["evidence_packet"],
            "previous_answer": row.get("answer_ko") or "",
            "failure_reasons": reasons,
        }
        requests.append({
            "id": f"cycle04_grounded_repair__{prompt_id}",
            "messages": [
                {"role": "system", "content": repair_system()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 2048,
            "metadata": {
                "purpose": "ksova_cycle04_grounded_answer_repair",
                "experiment_id": EXPERIMENT_ID,
                "prompt_id": prompt_id,
                "issue_id": context["issue_id"],
                "domain_id": context["domain_id"],
                "split": context["split"],
                "evidence_packet_sha256": context["evidence_packet_sha256"],
                "repair_attempt": 1,
            },
        })
    atomic_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-cycle04-repair-build-1.0",
        "structural_rejections": len(structural),
        "validation_rejections": len(validation),
        "repair_requests": len(requests),
        "requests_sha256": sha256(args.output),
    }
    atomic_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def materialize_sft(args: argparse.Namespace) -> int:
    prompts = read_jsonl(args.prompts)
    initial = read_jsonl(args.initial_accepted)
    repaired = read_jsonl(args.repair_accepted)
    combined = initial + repaired
    accepted_by_id = {str(row["prompt_id"]): row for row in combined}
    if len(accepted_by_id) != len(combined):
        raise ValueError("initial and repair accepted sets overlap")
    prompt_by_id = {str(row["prompt_id"]): row for row in prompts}
    if set(accepted_by_id) - set(prompt_by_id):
        raise ValueError("accepted response has unexpected prompt ID")

    output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for prompt_id, accepted in sorted(accepted_by_id.items()):
        prompt = prompt_by_id[prompt_id]
        split = str(prompt["split"])
        if split not in output_rows or split != accepted["split"]:
            raise ValueError(f"split mismatch for {prompt_id}")
        training_id = "KSC4SFT-" + hashlib.sha256(
            f"{prompt_id}\0{accepted['answer_ko']}".encode()
        ).hexdigest()[:20]
        output_rows[split].append({
            "schema_version": "ksova-cycle04-sft-example-1.0",
            "messages": [
                {"role": "user", "content": prompt["question_ko"]},
                {"role": "assistant", "content": accepted["answer_ko"]},
            ],
            "metadata": {
                "training_id": training_id,
                "experiment_id": EXPERIMENT_ID,
                "prompt_id": prompt_id,
                "issue_id": prompt["issue_id"],
                "domain_id": prompt["domain_id"],
                "split": split,
                "prompt_form": prompt["prompt_form"],
                "answer_origin": accepted["answer_origin"],
                "answer_model": accepted["answer_model"],
                "validator_model": accepted["validator_model"],
                "evidence_mode": accepted["evidence_mode"],
                "evidence_packet_sha256": accepted["evidence_packet_sha256"],
            },
        })

    counts = {split: len(rows) for split, rows in output_rows.items()}
    issue_counts = {
        split: len({row["metadata"]["issue_id"] for row in rows})
        for split, rows in output_rows.items()
    }
    gates = {
        "train_rows": counts["train"] >= 405,
        "train_issues": issue_counts["train"] >= 110,
        "dev_rows": counts["dev"] >= 45,
        "dev_issues": issue_counts["dev"] >= 12,
    }
    if not all(gates.values()):
        raise ValueError(
            f"SFT validation gate failed: counts={counts}, issues={issue_counts}, gates={gates}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "sft_train.jsonl"
    dev_path = args.output_dir / "sft_dev.jsonl"
    atomic_jsonl(train_path, output_rows["train"])
    atomic_jsonl(dev_path, output_rows["dev"])
    report = {
        "schema_version": "ksova-cycle04-sft-materialization-1.0",
        "accepted_initial": len(initial),
        "accepted_after_one_repair": len(repaired),
        "rows": counts,
        "issues": issue_counts,
        "domain_counts": {
            split: dict(Counter(row["metadata"]["domain_id"] for row in rows))
            for split, rows in output_rows.items()
        },
        "prompt_form_counts": {
            split: dict(Counter(row["metadata"]["prompt_form"] for row in rows))
            for split, rows in output_rows.items()
        },
        "answer_origin_counts": dict(Counter(
            row["metadata"]["answer_origin"]
            for rows in output_rows.values() for row in rows
        )),
        "gates": gates,
        "outputs": {
            "sft_train": str(train_path),
            "sft_train_sha256": sha256(train_path),
            "sft_dev": str(dev_path),
            "sft_dev_sha256": sha256(dev_path),
        },
    }
    atomic_json(args.output_dir / "materialization_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def load_sft_rows(train_path: Path, dev_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(train_path) + read_jsonl(dev_path)
    prompt_ids = [str((row.get("metadata") or {}).get("prompt_id") or "") for row in rows]
    if not all(prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("SFT rows have empty or duplicate prompt IDs")
    for row in rows:
        messages = row.get("messages") or []
        if (
            len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
        ):
            raise ValueError("SFT row must contain one user and one assistant message")
    return rows


def assigned_defect(prompt_id: str) -> str:
    value = int(hashlib.sha256(f"cycle04-defect\0{prompt_id}".encode()).hexdigest()[:16], 16)
    return DEFECT_TYPES[value % len(DEFECT_TYPES)]


def build_negative_requests(args: argparse.Namespace) -> int:
    evidence = evidence_index(args.evidence_packets)
    rows = load_sft_rows(args.sft_train, args.sft_dev)
    requests: list[dict[str, Any]] = []
    defect_counts: Counter[str] = Counter()
    for row in sorted(rows, key=lambda value: str(value["metadata"]["prompt_id"])):
        metadata = row["metadata"]
        prompt_id = str(metadata["prompt_id"])
        context = evidence.get(prompt_id)
        if context is None:
            raise ValueError(f"missing evidence for SFT prompt {prompt_id}")
        defect = assigned_defect(prompt_id)
        defect_counts[defect] += 1
        chosen = str(row["messages"][1]["content"] or "").strip()
        payload = {
            "prompt_id": prompt_id,
            "question_ko": context["question_ko"],
            "evidence_packet": context["evidence_packet"],
            "assigned_defect": defect,
            "chosen_response": chosen,
        }
        requests.append({
            "id": f"cycle04_hard_negative__{prompt_id}",
            "messages": [
                {"role": "system", "content": negative_generation_system()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 2048,
            "metadata": {
                "purpose": "ksova_cycle04_hard_negative_generation",
                "experiment_id": EXPERIMENT_ID,
                "prompt_id": prompt_id,
                "issue_id": metadata["issue_id"],
                "domain_id": metadata["domain_id"],
                "split": metadata["split"],
                "prompt_form": metadata["prompt_form"],
                "assigned_defect": defect,
                "chosen_sha256": hashlib.sha256(chosen.encode()).hexdigest(),
                "evidence_packet_sha256": context["evidence_packet_sha256"],
            },
        })
    atomic_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-cycle04-negative-request-build-1.0",
        "requests": len(requests),
        "train_requests": sum(row["metadata"]["split"] == "train" for row in requests),
        "dev_requests": sum(row["metadata"]["split"] == "dev" for row in requests),
        "assigned_defect_counts": dict(defect_counts),
        "requests_sha256": sha256(args.output),
    }
    atomic_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def ngram_dice(left: str, right: str, size: int = 5) -> float:
    def grams(text: str) -> Counter[str]:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) < size:
            return Counter({compact: 1}) if compact else Counter()
        return Counter(compact[index : index + size] for index in range(len(compact) - size + 1))

    left_grams, right_grams = grams(left), grams(right)
    total = sum(left_grams.values()) + sum(right_grams.values())
    if total == 0:
        return 1.0
    overlap = sum((left_grams & right_grams).values())
    return 2.0 * overlap / total


def build_negative_validation(args: argparse.Namespace) -> int:
    evidence = evidence_index(args.evidence_packets)
    sft_rows = load_sft_rows(args.sft_train, args.sft_dev)
    sft_by_id = {
        str(row["metadata"]["prompt_id"]): row for row in sft_rows
    }
    responses = read_jsonl(args.responses)
    response_ids = [response_prompt_id(row) for row in responses]
    if not all(response_ids) or len(set(response_ids)) != len(response_ids):
        raise ValueError("negative responses have empty or duplicate prompt IDs")
    if set(response_ids) - set(sft_by_id):
        raise ValueError("negative responses contain unexpected prompt IDs")

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for response in sorted(responses, key=response_prompt_id):
        prompt_id = response_prompt_id(response)
        sft = sft_by_id[prompt_id]
        metadata = sft["metadata"]
        context = evidence[prompt_id]
        chosen = str(sft["messages"][1]["content"] or "").strip()
        defect = assigned_defect(prompt_id)
        failures: list[str] = []
        parsed: dict[str, Any] | None = None
        if response.get("error"):
            failures.append("generation_error")
        if str(response.get("finish_reason") or "").casefold() != "stop":
            failures.append("finish_reason_not_stop")
        try:
            parsed = parse_json_object(str(response.get("response") or ""))
        except ValueError as exc:
            failures.append(f"json_parse:{exc}")
        rejected_answer = ""
        edit_summary = ""
        if parsed is not None:
            if str(parsed.get("prompt_id")) != prompt_id:
                failures.append("prompt_id_mismatch")
            if parsed.get("defect_type") != defect:
                failures.append("defect_type_mismatch")
            rejected_answer = str(parsed.get("rejected_response") or "").strip()
            edit_summary = normalized(parsed.get("edit_summary_ko"))
        ratio = len(rejected_answer) / len(chosen) if chosen else 0.0
        similarity = ngram_dice(chosen, rejected_answer)
        if len(rejected_answer) < MIN_ANSWER_CHARS:
            failures.append(f"rejected_too_short:{len(rejected_answer)}")
        if len(rejected_answer) > MAX_ANSWER_CHARS:
            failures.append(f"rejected_too_long:{len(rejected_answer)}")
        if not 0.70 <= ratio <= 1.20:
            failures.append(f"length_ratio:{ratio:.4f}")
        if not 0.45 <= similarity < 0.995:
            failures.append(f"ngram_similarity:{similarity:.4f}")
        candidate = {
            "prompt_id": prompt_id,
            "issue_id": metadata["issue_id"],
            "domain_id": metadata["domain_id"],
            "split": metadata["split"],
            "prompt_form": metadata["prompt_form"],
            "question_ko": context["question_ko"],
            "chosen_response": chosen,
            "rejected_response": rejected_answer,
            "defect_type": defect,
            "edit_summary_ko": edit_summary,
            "length_ratio": round(ratio, 6),
            "ngram_dice": round(similarity, 6),
            "negative_model": response.get("model"),
            "evidence_packet_sha256": context["evidence_packet_sha256"],
        }
        if failures:
            candidate["validation_failures"] = failures
            rejected.append(candidate)
            continue
        candidates.append(candidate)
        payload = {
            "prompt_id": prompt_id,
            "question_ko": context["question_ko"],
            "evidence_packet": context["evidence_packet"],
            "assigned_defect": defect,
            "chosen_response": chosen,
            "rejected_response": rejected_answer,
        }
        requests.append({
            "id": f"cycle04_hard_negative_validate__{prompt_id}",
            "messages": [
                {"role": "system", "content": negative_validation_system()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 768,
            "metadata": {
                "purpose": "ksova_cycle04_hard_negative_validation",
                "experiment_id": EXPERIMENT_ID,
                "prompt_id": prompt_id,
                "issue_id": metadata["issue_id"],
                "domain_id": metadata["domain_id"],
                "split": metadata["split"],
                "assigned_defect": defect,
                "evidence_packet_sha256": context["evidence_packet_sha256"],
            },
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "candidates.jsonl"
    rejected_path = args.output_dir / "structural_rejections.jsonl"
    requests_path = args.output_dir / "validation_requests.jsonl"
    atomic_jsonl(candidates_path, candidates)
    atomic_jsonl(rejected_path, rejected)
    atomic_jsonl(requests_path, requests)
    report = {
        "schema_version": "ksova-cycle04-negative-validation-build-1.0",
        "responses": len(responses),
        "structurally_valid": len(candidates),
        "structurally_rejected": len(rejected),
        "validation_requests": len(requests),
        "structural_failure_counts": dict(Counter(
            failure.split(":", 1)[0]
            for row in rejected for failure in row["validation_failures"]
        )),
        "outputs": {
            "candidates_sha256": sha256(candidates_path),
            "structural_rejections_sha256": sha256(rejected_path),
            "validation_requests_sha256": sha256(requests_path),
        },
    }
    atomic_json(args.output_dir / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_negative_validation(args: argparse.Namespace) -> int:
    candidates = read_jsonl(args.candidates)
    responses = read_jsonl(args.responses)
    candidate_by_id = {str(row["prompt_id"]): row for row in candidates}
    response_by_id = {response_prompt_id(row): row for row in responses}
    if len(candidate_by_id) != len(candidates) or len(response_by_id) != len(responses):
        raise ValueError("duplicate hard-negative candidate or response IDs")
    if set(response_by_id) - set(candidate_by_id):
        raise ValueError("hard-negative validation contains unexpected prompt IDs")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    false_counts: Counter[str] = Counter()
    for prompt_id, candidate in sorted(candidate_by_id.items()):
        response = response_by_id.get(prompt_id)
        failures: list[str] = []
        parsed: dict[str, Any] | None = None
        if response is None:
            failures.append("missing_validation_response")
        elif response.get("error"):
            failures.append("validation_response_error")
        elif str(response.get("finish_reason") or "").casefold() != "stop":
            failures.append("validation_finish_reason_not_stop")
        else:
            try:
                parsed = parse_json_object(str(response.get("response") or ""))
            except ValueError as exc:
                failures.append(f"validation_json_parse:{exc}")
        if parsed is not None:
            if str(parsed.get("prompt_id")) != prompt_id:
                failures.append("validation_prompt_id_mismatch")
            for flag in NEGATIVE_VALIDATION_FLAGS:
                if type(parsed.get(flag)) is not bool:
                    failures.append(f"validation_nonboolean:{flag}")
                elif parsed[flag] is not True:
                    failures.append(f"validator_false:{flag}")
                    false_counts[flag] += 1
            if not isinstance(parsed.get("failed_checks"), list):
                failures.append("failed_checks_not_array")
            elif parsed["failed_checks"]:
                failures.append("failed_checks_nonempty")
            if parsed.get("verdict") != "pass":
                failures.append("verdict_not_pass")
            if parsed.get("reasons_ko") != "모든 검사 통과.":
                failures.append("pass_reason_not_canonical")
        enriched = dict(candidate)
        enriched["validator_model"] = response.get("model") if response else None
        enriched["validation"] = parsed
        enriched["validation_failures"] = failures
        (accepted if not failures else rejected).append(enriched)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.output_dir / "accepted.jsonl"
    rejected_path = args.output_dir / "rejected.jsonl"
    atomic_jsonl(accepted_path, accepted)
    atomic_jsonl(rejected_path, rejected)
    report = {
        "schema_version": "ksova-cycle04-negative-validation-parse-1.0",
        "candidates": len(candidates),
        "responses": len(responses),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": round(len(accepted) / len(candidates), 6) if candidates else 0,
        "validator_false_counts": dict(false_counts),
        "outputs": {
            "accepted_sha256": sha256(accepted_path),
            "rejected_sha256": sha256(rejected_path),
        },
    }
    atomic_json(args.output_dir / "parse_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def materialize_dpo(args: argparse.Namespace) -> int:
    accepted = read_jsonl(args.accepted_pairs)
    prompt_ids = [str(row["prompt_id"]) for row in accepted]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("accepted DPO pairs contain duplicate prompt IDs")
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for row in sorted(accepted, key=lambda value: str(value["prompt_id"])):
        split = str(row["split"])
        if split not in rows_by_split:
            raise ValueError(f"unexpected split in DPO pair: {split}")
        pair_id = "KSC4DPO-" + hashlib.sha256(
            f"{row['prompt_id']}\0{row['chosen_response']}\0{row['rejected_response']}".encode()
        ).hexdigest()[:20]
        rows_by_split[split].append({
            "schema_version": "ksova-cycle04-dpo-pair-1.0",
            "prompt_messages": [{"role": "user", "content": row["question_ko"]}],
            "chosen": row["chosen_response"],
            "rejected": row["rejected_response"],
            "metadata": {
                "training_id": pair_id,
                "pair_id": pair_id,
                "experiment_id": EXPERIMENT_ID,
                "prompt_id": row["prompt_id"],
                "issue_id": row["issue_id"],
                "domain_id": row["domain_id"],
                "split": split,
                "prompt_form": row["prompt_form"],
                "defect_type": row["defect_type"],
                "sample_weight": 1.0,
                "negative_model": row["negative_model"],
                "validator_model": row["validator_model"],
                "evidence_packet_sha256": row["evidence_packet_sha256"],
            },
        })
    counts = {split: len(rows) for split, rows in rows_by_split.items()}
    issue_counts = {
        split: len({row["metadata"]["issue_id"] for row in rows})
        for split, rows in rows_by_split.items()
    }
    gates = {
        "train_pairs": counts["train"] >= 300,
        "train_issues": issue_counts["train"] >= 100,
        "dev_pairs": counts["dev"] >= 30,
        "dev_issues": issue_counts["dev"] >= 10,
    }
    if not all(gates.values()):
        raise ValueError(
            f"DPO validation gate failed: counts={counts}, issues={issue_counts}, gates={gates}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "dpo_train.jsonl"
    dev_path = args.output_dir / "dpo_dev.jsonl"
    atomic_jsonl(train_path, rows_by_split["train"])
    atomic_jsonl(dev_path, rows_by_split["dev"])
    report = {
        "schema_version": "ksova-cycle04-dpo-materialization-1.0",
        "rows": counts,
        "issues": issue_counts,
        "gates": gates,
        "defect_counts": {
            split: dict(Counter(row["metadata"]["defect_type"] for row in rows))
            for split, rows in rows_by_split.items()
        },
        "domain_counts": {
            split: dict(Counter(row["metadata"]["domain_id"] for row in rows))
            for split, rows in rows_by_split.items()
        },
        "outputs": {
            "dpo_train": str(train_path),
            "dpo_train_sha256": sha256(train_path),
            "dpo_dev": str(dev_path),
            "dpo_dev_sha256": sha256(dev_path),
        },
    }
    atomic_json(args.output_dir / "materialization_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evidence-packets", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("prepare")
    command.add_argument("--issues", type=Path, required=True)
    command.add_argument("--sources", type=Path, required=True)
    command.add_argument("--train-examples", type=Path, required=True)
    command.add_argument("--benchmark", type=Path, required=True)
    command.add_argument("--selected-issues", type=Path, required=True)
    command.add_argument("--prompts", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(function=prepare)

    command = subparsers.add_parser("build-validation")
    add_shared_paths(command)
    command.add_argument("--responses", type=Path, required=True)
    command.add_argument("--candidate-origin", required=True)
    command.add_argument("--validation-id-prefix", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(function=build_validation)

    command = subparsers.add_parser("parse-validation")
    command.add_argument("--candidates", type=Path, required=True)
    command.add_argument("--responses", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(function=parse_validation)

    command = subparsers.add_parser("build-repair")
    add_shared_paths(command)
    command.add_argument("--structural-rejections", type=Path, required=True)
    command.add_argument("--validation-rejections", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=build_repair)

    command = subparsers.add_parser("materialize-sft")
    command.add_argument("--prompts", type=Path, required=True)
    command.add_argument("--initial-accepted", type=Path, required=True)
    command.add_argument("--repair-accepted", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(function=materialize_sft)

    command = subparsers.add_parser("build-negative-requests")
    add_shared_paths(command)
    command.add_argument("--sft-train", type=Path, required=True)
    command.add_argument("--sft-dev", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=build_negative_requests)

    command = subparsers.add_parser("build-negative-validation")
    add_shared_paths(command)
    command.add_argument("--sft-train", type=Path, required=True)
    command.add_argument("--sft-dev", type=Path, required=True)
    command.add_argument("--responses", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(function=build_negative_validation)

    command = subparsers.add_parser("parse-negative-validation")
    command.add_argument("--candidates", type=Path, required=True)
    command.add_argument("--responses", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(function=parse_negative_validation)

    command = subparsers.add_parser("materialize-dpo")
    command.add_argument("--accepted-pairs", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(function=materialize_dpo)

    args = parser.parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
