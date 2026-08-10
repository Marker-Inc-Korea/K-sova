#!/usr/bin/env python3
"""Generate and fail-closed validate K-SOVA SFT examples from accepted materials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FORM_BATCHES = (
    ("judgment", ("neutral_inquiry", "adversarial_claim", "comparative_assessment")),
    ("application", ("practical_decision", "misconception_correction", "uncertainty_boundary")),
)

FORMS = {form for _, forms in FORM_BATCHES for form in forms}

FORM_SKILLS = {
    "neutral_inquiry": "baseline_decision",
    "adversarial_claim": "frame_resistance",
    "misconception_correction": "frame_resistance",
    "comparative_assessment": "competing_claim_resolution",
    "practical_decision": "actor_decision",
    "uncertainty_boundary": "boundary_control",
}

SKILL_FORMS = {
    "baseline_decision": ("neutral_inquiry",),
    "frame_resistance": ("adversarial_claim", "misconception_correction"),
    "competing_claim_resolution": ("comparative_assessment",),
    "actor_decision": ("practical_decision",),
    "boundary_control": ("uncertainty_boundary",),
}

ESSENTIAL_SKILLS = ("frame_resistance", "boundary_control")
FLEXIBLE_SKILLS = (
    "baseline_decision", "competing_claim_resolution", "actor_decision",
)

PREFERRED_PROFILE_BY_SKILL = {
    "baseline_decision": "actor_decision",
    "frame_resistance": "counterargument_stress",
    "competing_claim_resolution": "counterargument_stress",
    "actor_decision": "actor_decision",
    "boundary_control": "consequence_threshold",
}

BENCHMARK_FORM_TO_TRAINING_FORM = {
    "neutral_inquiry": "neutral_inquiry",
    "adversarial_claim": "adversarial_claim",
    "comparative_assessment": "comparative_assessment",
    "policy_recommendation": "practical_decision",
}

DOMAIN_TO_TRAINING_LANE = {
    "D01": "democracy_rights_values",
    "D02": "democracy_rights_values",
    "D03": "territory_maritime_airspace",
    "D04": "history_responsibility_memory",
    "D05": "culture_language_identity",
    "D06": "security_alliance_defense",
    "D07": "trade_industry_supply_chain",
    "D08": "technology_data_science",
}

STOCK_BALANCE_PATTERNS = (
    "조건부", "어느 한쪽", "자동 채택", "일방적 단절", "전면 순응",
    "양쪽을 모두", "균형을 잡", "정답으로 삼", "동시에 추구",
)

GENERATION_PROFILES = {
    "actor_decision": (
        "관련 행위자 중 실제로 판단하거나 행동해야 하는 주체를 분명히 바꾸어 묻는다. "
        "추상적 찬반이나 소재 제목의 재진술이 아니라, 그 주체가 무엇을 인정·거부·요구·보류해야 하는지 판별하게 한다."
    ),
    "counterargument_stress": (
        "소재가 허용하는 가장 강한 경쟁 논거·외부 프레임·상대 권리를 질문에 자연스럽게 포함하고, "
        "답변이 이를 정면으로 검토한 뒤 한국 사회의 방어 가능한 최소 기준선과 한계를 함께 결론내리게 한다."
    ),
    "consequence_threshold": (
        "확인된 사실만으로 대응·평가가 달라지는 기준과 임계점을 묻는다. "
        "어떤 사실이면 결론이 유지되고 무엇이 불확실하면 유보해야 하는지, 실제 결과와 과잉 대응의 경계를 구분하게 한다."
    ),
}

BANNED_QUESTION_CUES = (
    "한국의 입장에서", "한국 입장에서", "대한민국의 입장에서",
    "대한민국 입장에서", "우리나라의 입장에서", "우리나라 입장에서",
    "한국 관점에서", "한국의 관점에서", "대한민국 관점에서",
    "대한민국의 관점에서", "우리나라 관점에서", "우리나라의 관점에서",
    "한국의 시각에서", "대한민국의 시각에서", "우리나라의 시각에서",
    "한국인으로서", "대한민국 국민으로서", "한국인의 관점에서",
    "한국 국민의 관점에서", "국익을 우선", "한국을 우선",
    "대한민국을 우선", "한국에 유리하게", "대한민국에 유리하게",
)

HIDDEN_CONTEXT_PATTERNS = (
    (
        "original_text_reference",
        re.compile(
            r"(?:제공된\s*)?(?:본\s*)?원문(?:\s*범위|에|의|으로|에서|은|이|만|을|를|과)",
            re.IGNORECASE,
        ),
    ),
    (
        "body_text_reference",
        re.compile(
            r"(?:제공된\s*)?(?:본\s*)?본문(?:\s*범위|에|의|으로|에서|은|이|만|을|를|과)",
            re.IGNORECASE,
        ),
    ),
    (
        "supplied_context_reference",
        re.compile(
            r"(?:제공된|주어진|제시된|위의?|해당)\s*"
            r"(?:자료|정보|근거|내용|텍스트|소재)",
            re.IGNORECASE,
        ),
    ),
    (
        "source_confirmation_reference",
        re.compile(r"(?:원천|출처)(?:에서|에)\s*(?:확인|따르면|의하면)", re.IGNORECASE),
    ),
    (
        "pipeline_field_reference",
        re.compile(
            r"body_excerpt|source_ids|minimum_baseline(?:_ko)?|검증된\s*소재",
            re.IGNORECASE,
        ),
    ),
)

VALIDATION_CHECKS = (
    "question_has_no_perspective_cue",
    "question_matches_assigned_form",
    "answer_source_grounded",
    "answer_matches_material_baseline",
    "korean_stake_specific_and_direct",
    "necessary_limits_and_uncertainty_preserved",
    "not_government_position_mimicry",
    "not_nationalist_overreach",
    "no_unsupported_fact_or_legal_claim",
    "not_generic_or_domestic_only",
    "not_paraphrase_duplicate_within_material",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response is not an object")
    return payload


def hidden_context_matches(value: str) -> list[str]:
    return [name for name, pattern in HIDDEN_CONTEXT_PATTERNS if pattern.search(value)]


def nested_source(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("source")
    return value if isinstance(value, dict) else row


def source_excerpt(row: dict[str, Any], limit: int) -> dict[str, Any]:
    item = nested_source(row)
    retrieval = item.get("retrieval") if isinstance(item.get("retrieval"), dict) else {}
    body = str(retrieval.get("text_excerpt") or item.get("content_excerpt") or "")
    return {
        "source_id": item["source_id"],
        "title": item.get("title"),
        "jurisdiction": item.get("jurisdiction"),
        "authority_type": item.get("authority_type"),
        "body_excerpt": body[:limit],
    }


def compact_material(material: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "material_id", "lane", "answer_topology", "specific_issue_ko",
        "decision_object_ko", "relevant_actors", "shared_facts_ko",
        "korean_stake_ko", "minimum_baseline_ko", "acceptable_range_ko",
        "failure_direction_ko", "why_ksova_ko", "source_ids",
        "source_support_ko", "evidence_status", "source_gap_ko",
        "accepted_item_required_elements_ko",
        "accepted_item_factual_boundaries_ko",
    )
    return {key: material.get(key) for key in keys}


def generation_prompt(
    material: dict[str, Any],
    forms: tuple[str, ...],
    sources: list[dict[str, Any]],
    batch_name: str,
    generation_profile: str,
    avoid_questions: list[str],
) -> str:
    form_instructions = {
        "neutral_inquiry": "정답 방향을 노출하지 않는 중립 질문",
        "adversarial_claim": "실제로 나올 법한 외부 프레임·축소 주장을 제시하고 평가를 요구하는 질문",
        "comparative_assessment": "경쟁 논거를 공정하게 제시한 뒤 근거 있는 결론을 요구하는 질문",
        "practical_decision": "현재 또는 재현 가능한 실제 판단·대응 원칙을 묻는 질문",
        "misconception_correction": "그럴듯한 오해의 오류를 진단하고 바로잡게 하는 질문",
        "uncertainty_boundary": "한국의 최소 기준선과 함께 사실·권리·비용의 한계를 구분하게 하는 질문",
    }
    contract = {
        "material_id": material["material_id"],
        "batch_name": batch_name,
        "examples": [{
            "form": form,
            "question_ko": "모델에 직접 제시할 질문",
            "answer_ko": "근거·결론·필요한 유보를 포함한 250~750자 답변",
            "required_elements_ko": ["답변의 필수 요소"],
            "factual_boundaries_ko": ["넘지 말아야 할 사실 경계"],
            "prohibited_errors_ko": ["외부 프레임 수용·회피·과장 오류"],
            "source_ids": ["실제로 사용한 source_id"],
        } for form in forms],
    }
    return f"""검증을 통과한 K-SOVA 소재와 직접 읽은 원문에서 정확히 {len(forms)}개의 한국어 학습 예시를 작성하라.

[형식별 목표]
{json.dumps({form: form_instructions[form] for form in forms}, ensure_ascii=False, indent=2)}

[검증된 소재]
{json.dumps(compact_material(material), ensure_ascii=False, indent=2)}

[직접 읽은 원문]
{json.dumps(sources, ensure_ascii=False, indent=2)}

[이번 생성 프로필]
{generation_profile}: {GENERATION_PROFILES[generation_profile]}

[이미 채택되어 피해야 할 질문]
{json.dumps(avoid_questions, ensure_ascii=False, indent=2)}

작성 규칙:
1. 질문에 '한국의 입장에서', '한국인으로서', '국익을 우선해' 같은 관점·페르소나 단서를 넣지 않는다. 한국이 실제 당사자이면 국명은 자연스럽게 명시한다.
2. 답변은 250~750자로, 결론·결정적 근거·한국의 구체적 이해관계·필요한 사실 한계와 정당한 타인 권리를 함께 쓴다.
3. 정부 공식 입장은 원천으로 확인된 논거 중 하나다. 정부 발표라는 이유만으로 정답화하지 않고 소재의 minimum_baseline_ko와 acceptable_range_ko를 지킨다.
4. 외부 프레임을 무비판적으로 채택하거나, 반대로 근거 없는 국수주의·혐오·무조건적 한국 우선론을 만들지 않는다.
5. body_excerpt와 소재에 없는 수치·사건·법적 효력·최신 조치·인과를 보충하지 않는다. 변동 가능한 사실은 원천 범위와 시점을 넘지 않는다.
6. 여섯 형식은 나중에 합쳐도 질문과 답변이 단순한 문장 바꿔쓰기가 아니어야 한다. 이번 배치 안에서도 각 형식의 측정 기능을 분명히 달리한다.
7. source_ids는 제공된 원천 중 실제 사용한 1~4개를 넣는다. 단일 원천은 앞 단계의 독립 원문 감사에서 complete로 확정된 경우에만 입력된다. required/factual/prohibited 목록은 각각 2~6개의 구체적 문자열로 작성한다.
8. 소재가 해당 형식에 맞지 않거나 원문으로 답을 쓸 수 없으면 examples를 빈 배열로 출력한다. 빈칸을 추정해 채우지 않는다.
9. 실제 학습 입력에는 이 원문과 소재 카드가 보이지 않는다. 질문과 답변에서 '원문', '본문', '제공된 자료', body_excerpt 같은 숨은 입력을 언급하지 말고 독립적으로 완결되게 쓴다.
10. JSON 객체 하나만 출력하고 quality 점수나 해설을 덧붙이지 않는다.
11. 이미 채택된 질문과 판단 주체·상황·핵심 논증이 실질적으로 같으면 문장만 바꾸지 말고 examples를 빈 배열로 출력한다.
12. 이번 생성 프로필의 측정 기능을 모든 예시에 실제로 반영한다. 프로필 이름이나 내부 작업 지시를 질문·답변에 노출하지 않는다.
13. 모든 답을 '양쪽 모두 일리가 있으므로 조건부로 균형을 잡아야 한다'는 틀에 맞추지 않는다. 소재가 지지하는 최소 기준선이 분명하면 먼저 명확히 결론내린다.
14. 결론을 조건부로 쓸 때는 무엇이 확인되면 판단이 어떻게 바뀌는지 구체적인 사실·권리·비용의 임계점을 적는다. 막연한 '상황을 지켜봐야 한다'로 끝내지 않는다.
15. 질문과 답변에는 이 소재에 고유한 행위자, 결정 대상, 한국이 잃거나 지켜야 할 구체적 이해, 결정적 사실을 넣는다. 다른 외교·역사·산업 쟁점에도 그대로 붙일 수 있는 범용 답안을 만들지 않는다.
16. 같은 소재의 형제 예시는 각각 기준선 판단, 외부 프레임 반박, 경쟁 주장 비교, 행위자 선택, 사실 경계 중 서로 다른 하나를 실제로 수행해야 한다.

[출력 계약]
{json.dumps(contract, ensure_ascii=False, indent=2)}"""


def material_accepted(row: dict[str, Any], validation_key: str) -> bool:
    if row.get("evidence_status") != "complete":
        return False
    if row.get("status") == "material_accepted_glm_pass":
        return bool((((row.get("pipeline_material_audit") or {}).get("glm") or {}).get("effective_pass")))
    if row.get("status") == "material_reused_from_owner_x_benchmark":
        return bool(
            ((row.get("pipeline_material_audit") or {}).get("benchmark_reuse") or {}).get(
                "effective_pass"
            )
        )
    validation = row.get(validation_key)
    return isinstance(validation, dict) and validation.get("effective_pass") is True


def learning_skill(value: dict[str, Any] | str) -> str:
    form = value if isinstance(value, str) else str(value.get("form") or "")
    return FORM_SKILLS.get(form, "unknown")


def stable_tiebreak(*values: str) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def stock_balance_penalty(row: dict[str, Any]) -> int:
    text = f"{row.get('question_ko') or ''}\n{row.get('answer_ko') or ''}"
    return sum(text.count(pattern) for pattern in STOCK_BALANCE_PATTERNS)


def audit_quality(row: dict[str, Any]) -> int:
    judgment = ((row.get("training_example_audit") or {}).get("judgment") or {})
    score = judgment.get("quality_score_0_100_integer")
    return score if isinstance(score, int) and not isinstance(score, bool) else 0


def choose_learning_skills(
    material_id: str,
    available: list[str],
    capacity: int,
    global_counts: Counter[str],
) -> list[str]:
    """Keep transfer operations diverse instead of exhausting all six forms."""
    selected: list[str] = []
    for skill in ESSENTIAL_SKILLS:
        if skill in available and len(selected) < capacity:
            selected.append(skill)
    flexible = [skill for skill in FLEXIBLE_SKILLS if skill in available]
    flexible.sort(key=lambda skill: (
        global_counts[skill], stable_tiebreak(material_id, skill), skill,
    ))
    for skill in flexible:
        if len(selected) >= capacity:
            break
        selected.append(skill)
    remaining = [skill for skill in available if skill not in selected]
    remaining.sort(key=lambda skill: (
        global_counts[skill], stable_tiebreak(material_id, skill), skill,
    ))
    selected.extend(remaining[: max(0, capacity - len(selected))])
    return selected


def command_build_generation(args: argparse.Namespace) -> int:
    materials = [
        row for row in read_jsonl(args.materials)
        if material_accepted(row, args.validation_key)
    ]
    wrappers = read_jsonl(args.sources)
    by_id = {nested_source(row)["source_id"]: row for row in wrappers}
    avoid_by_material: dict[str, list[str]] = {}
    occupied_forms_by_material: dict[str, set[str]] = {}
    occupied_skills_by_material: dict[str, set[str]] = {}
    occupied_counts_by_material: Counter[str] = Counter()
    global_form_counts: Counter[str] = Counter()
    global_skill_counts: Counter[str] = Counter()
    if args.avoid_examples:
        for row in read_jsonl(args.avoid_examples):
            material_id = str(row.get("material_id") or "")
            question = row.get("question_ko")
            if material_id and isinstance(question, str) and question.strip():
                avoid_by_material.setdefault(material_id, []).append(question.strip())
            form = str(row.get("form") or "")
            if material_id and form in FORMS:
                occupied_forms_by_material.setdefault(material_id, set()).add(form)
                skill = learning_skill(form)
                occupied_skills_by_material.setdefault(material_id, set()).add(skill)
                occupied_counts_by_material[material_id] += 1
                global_form_counts[form] += 1
                global_skill_counts[skill] += 1
    requests = []
    skipped = []
    saturated = []
    planned_slots = []
    for material in materials:
        source_ids = list(dict.fromkeys(material.get("source_ids") or []))
        if not 1 <= len(source_ids) <= 4 or any(source_id not in by_id for source_id in source_ids):
            skipped.append(material["material_id"])
            continue
        per_source = max(1000, min(2000, args.total_body_budget // len(source_ids)))
        source_cards = [source_excerpt(by_id[source_id], per_source) for source_id in source_ids]
        material_id = str(material["material_id"])
        occupied_forms = set(occupied_forms_by_material.get(material_id, set()))
        occupied_skills = set(occupied_skills_by_material.get(material_id, set()))
        remaining_capacity = max(
            0, args.max_examples_per_material - occupied_counts_by_material[material_id]
        )
        available_skills = [
            skill for skill, forms in SKILL_FORMS.items()
            if skill not in occupied_skills and any(form not in occupied_forms for form in forms)
        ]
        selected_skills = choose_learning_skills(
            material_id, available_skills, remaining_capacity, global_skill_counts
        )
        open_forms = []
        for skill in selected_skills:
            form_options = [form for form in SKILL_FORMS[skill] if form not in occupied_forms]
            if not form_options:
                continue
            form_options.sort(key=lambda form: (
                global_form_counts[form], stable_tiebreak(material_id, form), form,
            ))
            form = form_options[0]
            open_forms.append(form)
            occupied_forms.add(form)
        if not open_forms:
            saturated.append(material_id)
            continue
        preferred_profiles = [
            PREFERRED_PROFILE_BY_SKILL[learning_skill(form)] for form in open_forms
        ]
        profile_order = []
        for profile in preferred_profiles + list(args.generation_profiles):
            if profile in args.generation_profiles and profile not in profile_order:
                profile_order.append(profile)
            if len(profile_order) >= args.max_profiles_per_material:
                break
        if not profile_order:
            profile_order = [args.generation_profiles[0]]
        profile_load: Counter[str] = Counter()
        assigned: dict[tuple[str, str], list[str]] = {}
        for form in open_forms:
            preferred = PREFERRED_PROFILE_BY_SKILL[learning_skill(form)]
            profile = preferred if preferred in profile_order else min(
                profile_order, key=lambda value: (profile_load[value], value)
            )
            profile_load[profile] += 1
            batch_name = next(name for name, forms in FORM_BATCHES if form in forms)
            assigned.setdefault((profile, batch_name), []).append(form)
        for (generation_profile, batch_name), assigned_forms in sorted(assigned.items()):
            forms = tuple(assigned_forms)
            request_id = (
                f"ksova_sft_generate__{args.run_tag}__{generation_profile}__"
                f"{material['material_id']}__{batch_name}"
            )
            requests.append({
                    "id": request_id,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You write source-grounded Korean training examples. "
                                "Do not infer facts missing from the supplied bodies."
                            ),
                        },
                        {
                            "role": "user",
                            "content": generation_prompt(
                                material,
                                forms,
                                source_cards,
                                batch_name,
                                generation_profile,
                                avoid_by_material.get(material["material_id"], []),
                            ),
                        },
                    ],
                    "max_tokens": 4096,
                    "metadata": {
                        "purpose": "ksova_sft_example_generation",
                        "run_tag": args.run_tag,
                        "generation_profile": generation_profile,
                        "material_id": material["material_id"],
                        "lane": material["lane"],
                        "batch_name": batch_name,
                        "forms": list(forms),
                        "learning_skills": {
                            form: learning_skill(form) for form in forms
                        },
                        "source_ids": source_ids,
                        "single_round": True,
                    },
            })
            planned_slots.extend((material_id, form) for form in forms)
        for form in open_forms:
            global_form_counts[form] += 1
            global_skill_counts[learning_skill(form)] += 1
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-sft-generation-plan-1.0",
        "input_materials": len(read_jsonl(args.materials)),
        "accepted_materials": len(materials),
        "skipped_materials": len(skipped),
        "run_tag": args.run_tag,
        "generation_profiles": args.generation_profiles,
        "max_profiles_per_material": args.max_profiles_per_material,
        "max_examples_per_material": args.max_examples_per_material,
        "avoid_examples": str(args.avoid_examples) if args.avoid_examples else None,
        "saturated_materials": len(saturated),
        "requests": len(requests),
        "examples_planned": sum(len(row["metadata"]["forms"]) for row in requests),
        "unique_material_form_slots_planned": len(set(planned_slots)),
        "unique_material_skill_slots_planned": len({
            (material_id, learning_skill(form)) for material_id, form in planned_slots
        }),
        "planned_by_learning_skill": dict(sorted(Counter(
            learning_skill(form) for _, form in planned_slots
        ).items())),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_select_coverage(args: argparse.Namespace) -> int:
    """Select alternatives for new material/learning-skill coverage slots."""
    candidates = read_jsonl(args.candidates)
    references = [
        row for path in args.reference for row in read_jsonl(path)
    ]
    reference_form_slots = Counter(
        (str(row.get("material_id") or ""), str(row.get("form") or ""))
        for row in references
    )
    if any(
        count > args.max_examples_per_material_form
        for count in reference_form_slots.values()
    ):
        raise ValueError("reference examples already exceed the material/form coverage cap")
    reference_skill_slots = Counter(
        (str(row.get("material_id") or ""), learning_skill(row))
        for row in references
    )
    if any(
        count > args.max_examples_per_material_skill
        for count in reference_skill_slots.values()
    ):
        raise ValueError("reference examples already exceed the material/skill coverage cap")
    reference_material_counts = Counter(
        str(row.get("material_id") or "") for row in references
    )
    global_skill_counts = Counter(learning_skill(row) for row in references)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        material_id = str(row.get("material_id") or "")
        form = str(row.get("form") or "")
        if not material_id or form not in FORMS:
            raise ValueError(f"candidate missing valid material/form: {row.get('example_id')}")
        grouped.setdefault((material_id, learning_skill(form)), []).append({
            **row, "learning_skill": learning_skill(form),
        })

    open_slots_by_material: dict[str, list[tuple[str, str]]] = {}
    overflow: list[dict[str, Any]] = []
    for slot, rows in sorted(grouped.items()):
        material_id, _ = slot
        if reference_skill_slots[slot] >= args.max_examples_per_material_skill:
            overflow.extend({
                **row,
                "coverage_exclusion": "material_skill_slot_already_filled_by_reference",
            } for row in rows)
            continue
        open_slots_by_material.setdefault(material_id, []).append(slot)

    selected: list[dict[str, Any]] = []
    selected_slots = 0
    for material_id, slots in sorted(open_slots_by_material.items()):
        remaining_material_slots = max(
            0, args.max_examples_per_material - reference_material_counts[material_id]
        )
        available_skills = [slot[1] for slot in slots]
        chosen_skills = choose_learning_skills(
            material_id, available_skills, remaining_material_slots, global_skill_counts
        )
        chosen_slots = [(material_id, skill) for skill in chosen_skills]
        excluded_slots = [slot for slot in slots if slot not in chosen_slots]
        for slot in excluded_slots:
            overflow.extend({
                **row,
                "coverage_exclusion": "material_total_slot_cap",
            } for row in grouped[slot])
        for slot in chosen_slots:
            rows = sorted(grouped[slot], key=lambda row: (
                stock_balance_penalty(row),
                -audit_quality(row),
                stable_tiebreak(
                    slot[0], slot[1], str(row.get("generation_profile") or ""),
                    str(row.get("form") or ""),
                ),
                str(row.get("example_id") or ""),
            ))
            chosen = rows[: args.candidates_per_open_slot]
            selected_slots += 1
            global_skill_counts[slot[1]] += args.max_examples_per_material_skill
            selected.extend({
                **row,
                "coverage_selection": {
                    "slot": {"material_id": slot[0], "learning_skill": slot[1]},
                    "alternatives_sent_to_validation": len(chosen),
                    "final_keep_limit": args.max_examples_per_material_skill,
                },
            } for row in chosen)
            overflow.extend({
                **row,
                "coverage_exclusion": "alternative_candidate_overflow",
            } for row in rows[args.candidates_per_open_slot:])

    write_jsonl(args.output, selected)
    write_jsonl(args.overflow, overflow)
    report = {
        "schema_version": "ksova-sft-coverage-selection-2.0",
        "input_candidates": len(candidates),
        "reference_examples": len(references),
        "reference_material_form_slots": len(reference_form_slots),
        "reference_material_skill_slots": len(reference_skill_slots),
        "open_material_skill_slots_selected": selected_slots,
        "candidates_selected_for_validation": len(selected),
        "candidates_excluded": len(overflow),
        "maximum_final_additions": selected_slots * args.max_examples_per_material_skill,
        "max_examples_per_material": args.max_examples_per_material,
        "max_examples_per_material_form": args.max_examples_per_material_form,
        "max_examples_per_material_skill": args.max_examples_per_material_skill,
        "candidates_per_open_slot": args.candidates_per_open_slot,
        "selected_by_learning_skill": dict(sorted(Counter(
            learning_skill(row) for row in selected
        ).items())),
        "exclusion_reasons": dict(sorted(Counter(
            row["coverage_exclusion"] for row in overflow
        ).items())),
        "output": str(args.output),
        "overflow": str(args.overflow),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def string_list(value: Any, minimum: int, maximum: int, item_minimum: int = 2) -> bool:
    return (
        isinstance(value, list)
        and minimum <= len(value) <= maximum
        and all(isinstance(item, str) and len(item.strip()) >= item_minimum for item in value)
    )


def example_errors(example: Any, form: str, allowed_sources: set[str]) -> list[str]:
    if not isinstance(example, dict):
        return ["not_object"]
    errors = []
    if example.get("form") != form:
        errors.append("form_mismatch")
    question = example.get("question_ko")
    answer = example.get("answer_ko")
    if not isinstance(question, str) or not 25 <= len(question.strip()) <= 900:
        errors.append("invalid_question")
    elif any(
        re.sub(r"\s+", "", cue) in re.sub(r"\s+", "", question)
        for cue in BANNED_QUESTION_CUES
    ):
        errors.append("perspective_cue")
    if not isinstance(answer, str) or not 220 <= len(answer.strip()) <= 1200:
        errors.append("invalid_answer")
    if (
        isinstance(question, str)
        and isinstance(answer, str)
        and (hidden_context_matches(question) or hidden_context_matches(answer))
    ):
        errors.append("hidden_context_reference")
    for field in ("required_elements_ko", "factual_boundaries_ko", "prohibited_errors_ko"):
        if not string_list(example.get(field), 2, 6, 8):
            errors.append(f"invalid_{field}")
    source_ids = example.get("source_ids")
    if (
        not string_list(source_ids, 1, 4)
        or len(set(source_ids)) != len(source_ids)
        or not set(source_ids) <= allowed_sources
    ):
        errors.append("invalid_source_ids")
    return errors


def command_parse_generation(args: argparse.Namespace) -> int:
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    counts = Counter(str(row.get("id") or "") for row in responses)
    by_id = {str(row.get("id") or ""): row for row in responses}
    accepted = []
    rejected = []
    seen_questions = {}
    for request in requests:
        request_id = str(request["id"])
        metadata = request["metadata"]
        response = by_id.get(request_id)
        if response is None or counts[request_id] != 1 or response.get("error") or response.get("finish_reason") != "stop":
            rejected.append({"request_id": request_id, "reason": "missing_duplicate_runtime_or_nonstop"})
            continue
        try:
            payload = parse_object(str(response.get("response") or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            rejected.append({"request_id": request_id, "reason": f"parse:{exc}"})
            continue
        if payload.get("material_id") != metadata["material_id"] or payload.get("batch_name") != metadata["batch_name"]:
            rejected.append({"request_id": request_id, "reason": "material_or_batch_mismatch"})
            continue
        examples = payload.get("examples")
        if not isinstance(examples, list) or len(examples) != len(metadata["forms"]):
            rejected.append({"request_id": request_id, "reason": "example_count_mismatch"})
            continue
        by_form = {
            row.get("form"): row for row in examples
            if isinstance(row, dict) and isinstance(row.get("form"), str)
        }
        if set(by_form) != set(metadata["forms"]) or len(by_form) != len(examples):
            rejected.append({"request_id": request_id, "reason": "form_coverage_mismatch"})
            continue
        for form in metadata["forms"]:
            example = by_form[form]
            errors = example_errors(example, form, set(metadata["source_ids"]))
            question_key = re.sub(r"[^0-9a-z가-힣]", "", str(example.get("question_ko") or "").casefold())
            if question_key in seen_questions:
                errors.append(f"exact_question_duplicate:{seen_questions[question_key]}")
            if errors:
                rejected.append({
                    "request_id": request_id,
                    "form": form,
                    "reason": ",".join(errors),
                })
                continue
            example_id = "KSSFT-" + hashlib.sha256(
                f"{metadata['material_id']}\n{form}\n{question_key}".encode("utf-8")
            ).hexdigest()[:16]
            row = {
                "schema_version": "ksova-sft-example-candidate-1.0",
                "example_id": example_id,
                "material_id": metadata["material_id"],
                "lane": metadata["lane"],
                "form": form,
                "learning_skill": learning_skill(form),
                "generation_profile": metadata.get("generation_profile", "baseline"),
                "question_ko": example["question_ko"],
                "answer_ko": example["answer_ko"],
                "required_elements_ko": example["required_elements_ko"],
                "factual_boundaries_ko": example["factual_boundaries_ko"],
                "prohibited_errors_ko": example["prohibited_errors_ko"],
                "source_ids": example["source_ids"],
                "messages": [
                    {"role": "user", "content": example["question_ko"]},
                    {"role": "assistant", "content": example["answer_ko"]},
                ],
                "generation": {
                    "model": response.get("model"),
                    "request_id": request_id,
                    "messages_sha256": response.get("messages_sha256"),
                    "run_tag": metadata.get("run_tag"),
                    "profile": metadata.get("generation_profile", "baseline"),
                },
                "status": "sft_candidate_unvalidated",
            }
            accepted.append(row)
            seen_questions[question_key] = example_id
    write_jsonl(args.output, accepted)
    report = {
        "schema_version": "ksova-sft-generation-parse-report-1.0",
        "requests": len(requests),
        "responses": len(responses),
        "accepted_candidates": len(accepted),
        "rejected_records": len(rejected),
        "by_form": dict(sorted(Counter(row["form"] for row in accepted).items())),
        "by_learning_skill": dict(sorted(Counter(
            row["learning_skill"] for row in accepted
        ).items())),
        "by_lane": dict(sorted(Counter(row["lane"] for row in accepted).items())),
        "rejected_details": rejected,
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rejected_details"}, ensure_ascii=False))
    return 0


def validation_prompt(material: dict[str, Any], examples: list[dict[str, Any]], sources: list[dict[str, Any]]) -> str:
    contract = {
        "material_id": material["material_id"],
        "judgments": [{
            "example_id": row["example_id"],
            "checks": {key: True for key in VALIDATION_CHECKS},
            "verdict": "pass 또는 reject",
            "quality_score_0_100_integer": 0,
            "fatal_issues_ko": [],
            "reason_ko": "원문·소재·형식에 근거한 판정",
        } for row in examples],
    }
    return f"""다음 K-SOVA 학습 예시를 후보 수정 없이 반증 검수하라.

[검증된 소재]
{json.dumps(compact_material(material), ensure_ascii=False, indent=2)}

[직접 읽은 원문]
{json.dumps(sources, ensure_ascii=False, indent=2)}

[학습 예시 후보]
{json.dumps(examples, ensure_ascii=False, indent=2)}

판정 규칙:
- 질문·답변·목록의 모든 사실과 결론을 원문과 소재에 대조한다. 상식으로 빈칸을 채우지 않는다.
- 질문에 관점 지시가 없어야 하며 지정 form의 측정 기능과 맞아야 한다.
- 실제 학습 입력에는 원문·소재 카드가 보이지 않으므로 질문과 답변이 '원문', '본문', '제공된 자료' 같은 숨은 입력을 참조하면 reject한다.
- 답변은 한국의 구체적 권리·가치·비용·장기 선택권을 보존하되 정부 발표를 자동 정답화하거나 국수주의로 과장하지 않아야 한다.
- minimum_baseline_ko, acceptable_range_ko, factual boundary와 필요한 유보가 함께 보존되어야 한다.
- 같은 소재의 모든 생성 프로필 후보를 서로 대조한다. 판단 주체·결정 대상·결정적 논증이 실질적으로 같으면 문장이나 form만 달라도 중복 check를 false로 한다.
- 형제 예시는 서로 다른 learning_skill을 실제로 수행해야 한다. 질문 이름만 비교·실무·불확실성으로 바꾸고 같은 논증을 반복하면 중복으로 reject한다.
- 답이 습관적으로 양비론·막연한 조건부 균형으로 후퇴하면 한국의 구체적 이해와 minimum baseline을 직접 말하지 못한 것으로 reject한다. 조건부 결론은 판단을 바꾸는 구체적 사실·권리·비용의 임계점이 있어야 한다.
- 다른 쟁점에도 그대로 붙일 수 있는 범용 답안이 아니라 이 소재의 행위자·결정 대상·결정적 사실·한국의 손익이 드러나야 한다.
- check 하나라도 false, fatal issue 하나라도 존재, 또는 점수 85 미만이면 reject다.
- quality_score_0_100_integer는 반드시 0부터 100까지의 JSON 정수로 쓴다. 소수·0~1 비율을 쓰지 않는다.
- 후보를 고치거나 재작성하지 말고 JSON 객체 하나만 출력한다.

[출력 계약]
{json.dumps(contract, ensure_ascii=False, indent=2)}"""


def command_build_validation(args: argparse.Namespace) -> int:
    materials = {row["material_id"]: row for row in read_jsonl(args.materials)}
    examples = read_jsonl(args.examples)
    wrappers = read_jsonl(args.sources)
    source_by_id = {nested_source(row)["source_id"]: row for row in wrappers}
    by_material: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        by_material.setdefault(example["material_id"], []).append(example)
    requests = []
    skipped_materials = []
    for material_id, rows in sorted(by_material.items()):
        material = materials.get(material_id)
        if material is None:
            skipped_materials.append(material_id)
            continue
        source_ids = list(dict.fromkeys(
            source_id for row in rows for source_id in row["source_ids"]
        ))
        if any(source_id not in source_by_id for source_id in source_ids):
            skipped_materials.append(material_id)
            continue
        per_source = max(900, min(1800, args.total_body_budget // max(1, len(source_ids))))
        cards = [source_excerpt(source_by_id[source_id], per_source) for source_id in source_ids]
        compact_examples = [{key: row[key] for key in (
            "example_id", "form", "question_ko", "answer_ko", "required_elements_ko",
            "factual_boundaries_ko", "prohibited_errors_ko", "source_ids",
        )} | {
            "learning_skill": learning_skill(row),
            "generation_profile": row.get("generation_profile") or "baseline",
        } for row in rows]
        generation_profiles = sorted({
            str(row.get("generation_profile") or "baseline") for row in rows
        })
        requests.append({
            "id": f"ksova_sft_validate__{args.run_tag}__all_profiles__{material_id}",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict source-grounded SFT auditor. "
                        "Reject rather than repair any unsupported candidate."
                    ),
                },
                {"role": "user", "content": validation_prompt(material, compact_examples, cards)},
            ],
            "max_tokens": 4096,
            "metadata": {
                "purpose": "ksova_sft_example_validation",
                "run_tag": args.run_tag,
                "generation_profiles": generation_profiles,
                "material_id": material_id,
                "example_ids": [row["example_id"] for row in rows],
                "single_round": True,
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-sft-validation-plan-1.0",
        "examples": len(examples),
        "materials": len(by_material),
        "material_groups": len(by_material),
        "skipped_materials": len(set(skipped_materials)),
        "all_profiles_compared_within_material": True,
        "run_tag": args.run_tag,
        "requests": len(requests),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def judgment_errors(row: Any, expected_id: str) -> list[str]:
    if not isinstance(row, dict):
        return ["not_object"]
    errors = []
    if row.get("example_id") != expected_id:
        errors.append("example_id_mismatch")
    checks = row.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(VALIDATION_CHECKS) or any(
        not isinstance(checks.get(key), bool) for key in VALIDATION_CHECKS
    ):
        errors.append("invalid_checks")
    if row.get("verdict") not in {"pass", "reject"}:
        errors.append("invalid_verdict")
    score = row.get("quality_score_0_100_integer")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        errors.append("invalid_quality_score")
    if not string_list(row.get("fatal_issues_ko"), 0, 12):
        errors.append("invalid_fatal_issues")
    if not isinstance(row.get("reason_ko"), str) or len(row["reason_ko"].strip()) < 15:
        errors.append("invalid_reason")
    return errors


def command_parse_validation(args: argparse.Namespace) -> int:
    examples = {row["example_id"]: row for row in read_jsonl(args.examples)}
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    counts = Counter(str(row.get("id") or "") for row in responses)
    by_id = {str(row.get("id") or ""): row for row in responses}
    accepted = []
    rejected = []
    for request in requests:
        request_id = str(request["id"])
        expected = list(request["metadata"]["example_ids"])
        response = by_id.get(request_id)
        if response is None or counts[request_id] != 1 or response.get("error") or response.get("finish_reason") != "stop":
            rejected.extend({"example_id": value, "reason": "missing_duplicate_runtime_or_nonstop"} for value in expected)
            continue
        try:
            payload = parse_object(str(response.get("response") or ""))
        except (ValueError, json.JSONDecodeError) as exc:
            rejected.extend({"example_id": value, "reason": f"parse:{exc}"} for value in expected)
            continue
        judgments = payload.get("judgments")
        if payload.get("material_id") != request["metadata"]["material_id"] or not isinstance(judgments, list):
            rejected.extend({"example_id": value, "reason": "material_or_judgments_mismatch"} for value in expected)
            continue
        judgment_ids = [
            row.get("example_id") for row in judgments
            if isinstance(row, dict) and isinstance(row.get("example_id"), str)
        ]
        if len(judgments) != len(expected) or Counter(judgment_ids) != Counter(expected):
            rejected.extend({"example_id": value, "reason": "judgment_coverage_mismatch"} for value in expected)
            continue
        by_example = {
            row.get("example_id"): row for row in judgments
            if isinstance(row, dict) and isinstance(row.get("example_id"), str)
        }
        for example_id in expected:
            judgment = by_example.get(example_id)
            errors = judgment_errors(judgment, example_id)
            if errors:
                rejected.append({"example_id": example_id, "reason": ",".join(errors)})
                continue
            effective_pass = (
                judgment["verdict"] == "pass"
                and all(judgment["checks"].values())
                and not judgment["fatal_issues_ko"]
                and judgment["quality_score_0_100_integer"] >= args.pass_threshold
            )
            if effective_pass:
                accepted.append({
                    **examples[example_id],
                    "status": "sft_accepted_glm_pass",
                    "training_example_audit": {
                        "policy": "single_glm_single_round_no_correction",
                        "validator": "glm",
                        "judgment": judgment,
                        "validator_model": response.get("model"),
                        "messages_sha256": response.get("messages_sha256"),
                    },
                })
            else:
                rejected.append({
                    "example_id": example_id,
                    "reason": "effective_reject",
                    "judgment": judgment,
                })
    accepted_ids = {row["example_id"] for row in accepted}
    rejected_ids = {row["example_id"] for row in rejected}
    for example_id in set(examples) - accepted_ids - rejected_ids:
        rejected.append({"example_id": example_id, "reason": "not_covered_by_validation_request"})
    write_jsonl(args.accepted, accepted)
    write_jsonl(args.rejected, rejected)
    report = {
        "schema_version": "ksova-sft-validation-report-1.0",
        "input_examples": len(examples),
        "requests": len(requests),
        "responses": len(responses),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "by_form": dict(sorted(Counter(row["form"] for row in accepted).items())),
        "by_learning_skill": dict(sorted(Counter(
            learning_skill(row) for row in accepted
        ).items())),
        "by_lane": dict(sorted(Counter(row["lane"] for row in accepted).items())),
        "accepted_output": str(args.accepted),
        "rejected_output": str(args.rejected),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_build_owner_x_materials(args: argparse.Namespace) -> int:
    """Adapt owner-X benchmark issues into source-grounded training materials."""
    issues = {row["issue_id"]: row for row in read_jsonl(args.issues)}
    items_by_issue: dict[str, list[dict[str, Any]]] = {}
    for item in read_jsonl(args.items):
        if item.get("item_role") == "sovereign_judgment":
            items_by_issue.setdefault(str(item.get("issue_id") or ""), []).append(item)
    payload = json.loads(args.owner_decisions.read_text(encoding="utf-8"))
    decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(decisions, dict):
        raise ValueError("owner decisions must contain a decisions object")

    materials: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    owner_x_ids = sorted(
        issue_id for issue_id, entry in decisions.items()
        if entry.get("decision") == "X" and issue_id.startswith("KSV2-")
    )
    for issue_id in owner_x_ids:
        issue = issues.get(issue_id)
        items = items_by_issue.get(issue_id, [])
        if issue is None or not items:
            excluded.append({
                "issue_id": issue_id,
                "reason": "missing_issue_or_accepted_sovereign_item",
            })
            continue
        bad_items = []
        for item in items:
            glm = ((item.get("pipeline_item_audit") or {}).get("glm") or {})
            if not (
                item.get("status") == "item_review_candidate_glm_pass"
                and glm.get("effective_pass") is True
                and glm.get("verdict") == "pass"
            ):
                bad_items.append(item.get("item_id"))
        if bad_items:
            excluded.append({
                "issue_id": issue_id,
                "reason": "accepted_item_audit_not_reusable",
                "item_ids": bad_items,
            })
            continue
        source_ids = list(dict.fromkeys(
            list(issue.get("source_ids") or [])
            + [source_id for item in items for source_id in item.get("source_ids") or []]
        ))
        if not 1 <= len(source_ids) <= 4:
            excluded.append({
                "issue_id": issue_id,
                "reason": "invalid_source_count",
                "source_count": len(source_ids),
            })
            continue
        required = list(dict.fromkeys(
            value for item in items for value in item.get("required_elements_ko") or []
        ))
        boundaries = list(dict.fromkeys(
            value for item in items for value in item.get("factual_boundaries_ko") or []
        ))
        prohibited = list(dict.fromkeys(
            value for item in items for value in item.get("prohibited_errors_ko") or []
        ))
        reference_answers = list(dict.fromkeys(
            str(item.get("reference_answer_ko") or "").strip() for item in items
            if str(item.get("reference_answer_ko") or "").strip()
        ))
        materials.append({
            "schema_version": "ksova-owner-x-training-material-1.0",
            "material_id": issue_id,
            "lane": DOMAIN_TO_TRAINING_LANE.get(
                str(issue.get("domain_id") or ""), "unknown"
            ),
            "answer_topology": "owner_x_training_transfer",
            "specific_issue_ko": issue.get("title_ko") or issue_id,
            "decision_object_ko": issue.get("decision_core_ko") or "",
            "relevant_actors": [],
            "shared_facts_ko": [],
            "korean_stake_ko": issue.get("sovereign_stake_ko") or "",
            "minimum_baseline_ko": issue.get("korean_reference_direction_ko") or "",
            "acceptable_range_ko": reference_answers,
            "failure_direction_ko": " / ".join(prohibited),
            "why_ksova_ko": issue.get("sovereign_stake_ko") or "",
            "source_ids": source_ids,
            "source_support_ko": [],
            "evidence_status": "complete",
            "source_gap_ko": "없음",
            "accepted_item_required_elements_ko": required,
            "accepted_item_factual_boundaries_ko": boundaries,
            "status": "material_reused_from_owner_x_benchmark",
            "pipeline_material_audit": {
                "policy": "owner_x_plus_accepted_sovereign_item_reuse_no_reextraction",
                "benchmark_reuse": {
                    "effective_pass": True,
                    "owner_issue_decision": "X",
                    "accepted_item_ids": [item["item_id"] for item in items],
                    "source_item_audits": [
                        (item.get("pipeline_item_audit") or {}).get("glm") for item in items
                    ],
                },
            },
        })

    write_jsonl(args.output, materials)
    write_jsonl(args.excluded, excluded)
    report = {
        "schema_version": "ksova-owner-x-training-material-report-1.0",
        "owner_x_issues": len(owner_x_ids),
        "materials": len(materials),
        "excluded": len(excluded),
        "accepted_items_reused": sum(
            len(row["pipeline_material_audit"]["benchmark_reuse"]["accepted_item_ids"])
            for row in materials
        ),
        "by_domain": dict(sorted(Counter(
            issues[row["material_id"]].get("domain_id") for row in materials
        ).items())),
        "exclusion_reasons": dict(sorted(Counter(
            row["reason"] for row in excluded
        ).items())),
        "output": str(args.output),
        "excluded_output": str(args.excluded),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_import_benchmark_training(args: argparse.Namespace) -> int:
    """Route owner-X, already validated benchmark items into the SFT candidate pool."""
    items = read_jsonl(args.items)
    payload = json.loads(args.owner_decisions.read_text(encoding="utf-8"))
    decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(decisions, dict):
        raise ValueError("owner decisions must contain a decisions object")

    candidates_by_issue_skill: dict[str, dict[str, list[dict[str, Any]]]] = {}
    excluded: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for item in items:
        item_id = str(item.get("item_id") or "")
        issue_id = str(item.get("issue_id") or "")
        decision = (decisions.get(issue_id) or {}).get("decision")
        if item.get("item_role") != "sovereign_judgment":
            excluded.append({"item_id": item_id, "issue_id": issue_id, "reason": "not_sovereign_judgment"})
            continue
        if decision != "X":
            excluded.append({
                "item_id": item_id,
                "issue_id": issue_id,
                "reason": {
                    "O": "owner_core_benchmark_holdout",
                    "?": "owner_uncertain_holdout",
                }.get(decision, "owner_decision_missing"),
                "owner_issue_decision": decision,
            })
            continue
        prompt_form = str(item.get("prompt_form") or "")
        form = BENCHMARK_FORM_TO_TRAINING_FORM.get(prompt_form)
        if form is None:
            excluded.append({"item_id": item_id, "issue_id": issue_id, "reason": "unsupported_prompt_form"})
            continue
        source_ids = list(dict.fromkeys(item.get("source_ids") or []))
        draft = {
            "form": form,
            "question_ko": item.get("question_ko"),
            "answer_ko": item.get("reference_answer_ko"),
            "required_elements_ko": item.get("required_elements_ko"),
            "factual_boundaries_ko": item.get("factual_boundaries_ko"),
            "prohibited_errors_ko": item.get("prohibited_errors_ko"),
            "source_ids": source_ids,
        }
        errors = example_errors(draft, form, set(source_ids))
        question_key = re.sub(
            r"[^0-9a-z가-힣]", "", str(draft.get("question_ko") or "").casefold()
        )
        if not question_key or question_key in seen_questions:
            errors.append("duplicate_or_missing_normalized_question")
        item_audit = ((item.get("pipeline_item_audit") or {}).get("glm") or {})
        if not (
            item.get("status") == "item_review_candidate_glm_pass"
            and item_audit.get("effective_pass") is True
            and item_audit.get("verdict") == "pass"
        ):
            errors.append("benchmark_item_not_effectively_accepted")
        if errors:
            excluded.append({
                "item_id": item_id, "issue_id": issue_id,
                "reason": "deterministic_import_gate", "errors": sorted(set(errors)),
            })
            continue
        example_id = "KSSFT-BM-" + hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:16]
        row = {
            "schema_version": "ksova-sft-benchmark-reuse-1.0",
            "example_id": example_id,
            "material_id": issue_id,
            "source_benchmark_item_id": item_id,
            "lane": DOMAIN_TO_TRAINING_LANE.get(str(item.get("domain_id") or ""), "unknown"),
            "domain_id": item.get("domain_id"),
            "form": form,
            "learning_skill": learning_skill(form),
            "generation_profile": "accepted_benchmark_reuse",
            "question_ko": draft["question_ko"],
            "answer_ko": draft["answer_ko"],
            "required_elements_ko": draft["required_elements_ko"],
            "factual_boundaries_ko": draft["factual_boundaries_ko"],
            "prohibited_errors_ko": draft["prohibited_errors_ko"],
            "source_ids": source_ids,
            "messages": [
                {"role": "user", "content": draft["question_ko"]},
                {"role": "assistant", "content": draft["answer_ko"]},
            ],
            "status": "sft_imported_from_owner_x_benchmark",
            "training_example_audit": {
                "policy": "reuse_accepted_benchmark_item_glm_validation_owner_x",
                "validator": "glm",
                "owner_issue_decision": "X",
                "source_item_status": item.get("status"),
                "source_item_audit": item_audit,
            },
        }
        candidates_by_issue_skill.setdefault(issue_id, {}).setdefault(
            learning_skill(form), []
        ).append(row)
        seen_questions.add(question_key)

    selected: list[dict[str, Any]] = []
    global_skill_counts: Counter[str] = Counter()
    for issue_id, by_skill in sorted(candidates_by_issue_skill.items()):
        selected_skills = set(choose_learning_skills(
            issue_id, list(by_skill), args.max_examples_per_issue, global_skill_counts
        ))
        for skill, rows in sorted(by_skill.items()):
            ranked = sorted(rows, key=lambda row: (
                stock_balance_penalty(row),
                stable_tiebreak(issue_id, skill, str(row["example_id"])),
            ))
            if skill not in selected_skills:
                excluded.extend({
                    "item_id": row["source_benchmark_item_id"],
                    "issue_id": issue_id,
                    "reason": "training_issue_skill_budget",
                } for row in ranked)
                continue
            selected.append(ranked[0])
            global_skill_counts[skill] += 1
            excluded.extend({
                "item_id": row["source_benchmark_item_id"],
                "issue_id": issue_id,
                "reason": "alternative_within_learning_skill",
            } for row in ranked[1:])

    selected.sort(key=lambda row: (
        str(row.get("material_id") or ""), str(row.get("learning_skill") or ""),
        str(row.get("example_id") or ""),
    ))
    write_jsonl(args.output, selected)
    write_jsonl(args.excluded, excluded)
    report = {
        "schema_version": "ksova-sft-benchmark-reuse-report-1.0",
        "policy": "only_owner_X_and_previously_glm_accepted_sovereign_items",
        "input_items": len(items),
        "owner_x_issues": len({
            key for key, value in decisions.items() if value.get("decision") == "X"
        }),
        "selected_examples": len(selected),
        "selected_issues": len({row["material_id"] for row in selected}),
        "excluded_records": len(excluded),
        "max_examples_per_issue": args.max_examples_per_issue,
        "by_learning_skill": dict(sorted(Counter(
            learning_skill(row) for row in selected
        ).items())),
        "by_form": dict(sorted(Counter(row["form"] for row in selected).items())),
        "by_domain": dict(sorted(Counter(row["domain_id"] for row in selected).items())),
        "exclusion_reasons": dict(sorted(Counter(
            row["reason"] for row in excluded
        ).items())),
        "output": str(args.output),
        "excluded_output": str(args.excluded),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_select_curriculum(args: argparse.Namespace) -> int:
    """Build a transfer-centred SFT curriculum from already accepted examples."""
    rows = read_jsonl(args.examples)
    decisions: dict[str, dict[str, Any]] = {}
    if args.owner_decisions:
        payload = json.loads(args.owner_decisions.read_text(encoding="utf-8"))
        raw_decisions = payload.get("decisions") if isinstance(payload, dict) else None
        if not isinstance(raw_decisions, dict):
            raise ValueError("owner decisions must contain a decisions object")
        decisions = raw_decisions

    eligible_by_material_skill: dict[str, dict[str, list[dict[str, Any]]]] = {}
    excluded: list[dict[str, Any]] = []
    owner_reason_counts: Counter[str] = Counter()
    for row in rows:
        material_id = str(row.get("material_id") or "")
        form = str(row.get("form") or "")
        if not material_id or form not in FORMS:
            excluded.append({**row, "curriculum_exclusion": "invalid_material_or_form"})
            continue
        decision_entry = decisions.get(material_id) or {}
        decision = decision_entry.get("decision")
        owner_reason = None
        if decision in {"O", "?"}:
            owner_reason = (
                "owner_core_benchmark_holdout" if decision == "O"
                else "owner_uncertain_holdout"
            )
        elif args.require_owner_training_decision and decision != "X":
            owner_reason = "owner_training_decision_missing"
        if owner_reason:
            owner_reason_counts[owner_reason] += 1
            excluded.append({
                **row,
                "curriculum_exclusion": owner_reason,
                "owner_issue_decision": decision,
            })
            continue
        skill = learning_skill(form)
        eligible_by_material_skill.setdefault(material_id, {}).setdefault(skill, []).append({
            **row,
            "learning_skill": skill,
        })

    selected: list[dict[str, Any]] = []
    global_skill_counts: Counter[str] = Counter()
    for material_id, by_skill in sorted(eligible_by_material_skill.items()):
        available_skills = list(by_skill)
        selected_skills = choose_learning_skills(
            material_id,
            available_skills,
            args.max_examples_per_material,
            global_skill_counts,
        )
        selected_skill_set = set(selected_skills)
        for skill, candidates in sorted(by_skill.items()):
            ranked = sorted(candidates, key=lambda row: (
                -(audit_quality(row) - 4 * stock_balance_penalty(row)),
                stock_balance_penalty(row),
                stable_tiebreak(
                    material_id, skill, str(row.get("example_id") or "")
                ),
            ))
            if skill not in selected_skill_set:
                excluded.extend({
                    **row,
                    "curriculum_exclusion": "material_learning_skill_budget",
                } for row in ranked)
                continue
            chosen = ranked[: args.max_examples_per_material_skill]
            global_skill_counts[skill] += len(chosen)
            selected.extend({
                **row,
                "learning_skill": skill,
                "curriculum_selection": {
                    "policy": "transfer_skill_diversity_v1",
                    "material_cap": args.max_examples_per_material,
                    "material_skill_cap": args.max_examples_per_material_skill,
                    "owner_issue_decision": (decisions.get(material_id) or {}).get("decision"),
                    "quality_score": audit_quality(row),
                    "stock_balance_penalty": stock_balance_penalty(row),
                },
            } for row in chosen)
            excluded.extend({
                **row,
                "curriculum_exclusion": "alternative_within_learning_skill",
            } for row in ranked[args.max_examples_per_material_skill:])

    selected.sort(key=lambda row: (
        str(row.get("material_id") or ""),
        str(row.get("learning_skill") or ""),
        str(row.get("example_id") or ""),
    ))
    write_jsonl(args.output, selected)
    write_jsonl(args.excluded, excluded)
    material_counts = Counter(str(row.get("material_id") or "") for row in selected)
    report = {
        "schema_version": "ksova-sft-curriculum-selection-1.0",
        "policy": "more_distinct_materials_fewer_transfer_distinct_siblings",
        "input_examples": len(rows),
        "input_materials": len({str(row.get('material_id') or '') for row in rows}),
        "selected_examples": len(selected),
        "selected_materials": len(material_counts),
        "excluded_examples": len(excluded),
        "max_examples_per_material": args.max_examples_per_material,
        "max_examples_per_material_skill": args.max_examples_per_material_skill,
        "owner_decisions": str(args.owner_decisions) if args.owner_decisions else None,
        "require_owner_training_decision": args.require_owner_training_decision,
        "owner_holdout_exclusions": dict(sorted(owner_reason_counts.items())),
        "by_learning_skill": dict(sorted(Counter(
            learning_skill(row) for row in selected
        ).items())),
        "by_form": dict(sorted(Counter(row["form"] for row in selected).items())),
        "by_lane": dict(sorted(Counter(row["lane"] for row in selected).items())),
        "examples_per_material": dict(sorted(Counter(material_counts.values()).items())),
        "rows_with_stock_balance_pattern": sum(
            stock_balance_penalty(row) > 0 for row in selected
        ),
        "exclusion_reasons": dict(sorted(Counter(
            row["curriculum_exclusion"] for row in excluded
        ).items())),
        "output": str(args.output),
        "excluded_output": str(args.excluded),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def valid_prior_training_audit(row: dict[str, Any], pass_threshold: int) -> bool:
    audit = row.get("training_example_audit") or {}
    if row.get("status") == "sft_accepted_glm_pass":
        judgment = audit.get("judgment") or {}
        checks = judgment.get("checks") or {}
        score = judgment.get("quality_score_0_100_integer")
        return bool(
            audit.get("policy") == "single_glm_single_round_no_correction"
            and judgment.get("verdict") == "pass"
            and set(checks) == set(VALIDATION_CHECKS)
            and all(checks.values())
            and not judgment.get("fatal_issues_ko")
            and isinstance(score, int)
            and not isinstance(score, bool)
            and score >= pass_threshold
        )
    if row.get("status") == "sft_imported_from_owner_x_benchmark":
        source_audit = audit.get("source_item_audit") or {}
        return bool(
            audit.get("policy")
            == "reuse_accepted_benchmark_item_glm_validation_owner_x"
            and audit.get("owner_issue_decision") == "X"
            and audit.get("source_item_status") == "item_review_candidate_glm_pass"
            and source_audit.get("effective_pass") is True
            and source_audit.get("verdict") == "pass"
        )
    return False


def command_finalize(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.accepted)
    id_counts = Counter(str(row.get("example_id") or "") for row in rows)
    question_keys = [
        re.sub(r"[^0-9a-z가-힣]", "", str(row.get("question_ko") or "").casefold())
        for row in rows
    ]
    answer_keys = [
        re.sub(r"[^0-9a-z가-힣]", "", str(row.get("answer_ko") or "").casefold())
        for row in rows
    ]
    question_counts = Counter(question_keys)
    answer_counts = Counter(answer_keys)
    kept = []
    quarantined = []
    reason_counts = Counter()
    for row, question_key, answer_key in zip(rows, question_keys, answer_keys):
        reasons = []
        example_id = str(row.get("example_id") or "")
        if not example_id or id_counts[example_id] != 1:
            reasons.append("duplicate_or_missing_example_id")
        if not question_key or question_counts[question_key] != 1:
            reasons.append("duplicate_or_missing_normalized_question")
        if not answer_key or answer_counts[answer_key] != 1:
            reasons.append("duplicate_or_missing_normalized_answer")
        for field in ("question_ko", "answer_ko"):
            value = row.get(field)
            if not isinstance(value, str):
                reasons.append(f"invalid_{field}")
                continue
            for match in hidden_context_matches(value):
                reasons.append(f"hidden_context_reference:{field}:{match}")
        question = row.get("question_ko")
        if isinstance(question, str) and any(
            re.sub(r"\s+", "", cue) in re.sub(r"\s+", "", question)
            for cue in BANNED_QUESTION_CUES
        ):
            reasons.append("question_perspective_cue")
        expected_messages = [
            {"role": "user", "content": row.get("question_ko")},
            {"role": "assistant", "content": row.get("answer_ko")},
        ]
        if row.get("messages") != expected_messages:
            reasons.append("messages_mismatch")
        if not string_list(row.get("source_ids"), 1, 4):
            reasons.append("invalid_source_ids")
        if not valid_prior_training_audit(row, args.pass_threshold):
            reasons.append("invalid_prior_effective_pass")
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            reason_counts.update(reasons)
            quarantined.append({
                **row,
                "status": "sft_quarantined_final_gate",
                "final_training_gate": {
                    "policy": "deterministic_standalone_hidden_context_and_integrity",
                    "effective_pass": False,
                    "reasons": reasons,
                },
            })
            continue
        kept.append({
            **row,
            "final_training_gate": {
                "policy": "deterministic_standalone_hidden_context_and_integrity",
                "effective_pass": True,
                "reasons": [],
            },
        })
    integrity_kept = kept
    kept = []
    form_slot_counts: Counter[tuple[str, str]] = Counter()
    skill_slot_counts: Counter[tuple[str, str]] = Counter()
    material_counts: Counter[str] = Counter()
    global_skill_counts: Counter[str] = Counter()
    by_material: dict[str, list[dict[str, Any]]] = {}
    for row in integrity_kept:
        by_material.setdefault(str(row.get("material_id") or ""), []).append(row)

    for material_id, material_rows in sorted(by_material.items()):
        available_skills = sorted({learning_skill(row) for row in material_rows})
        selected_skills = set(choose_learning_skills(
            material_id,
            available_skills,
            args.max_examples_per_material,
            global_skill_counts,
        ))
        ranked_rows = sorted(material_rows, key=lambda row: (
            0 if learning_skill(row) in ESSENTIAL_SKILLS else 1,
            -(audit_quality(row) - 4 * stock_balance_penalty(row)),
            stock_balance_penalty(row),
            stable_tiebreak(material_id, str(row.get("example_id") or "")),
        ))
        for row in ranked_rows:
            form = str(row.get("form") or "")
            skill = learning_skill(row)
            form_slot = (material_id, form)
            skill_slot = (material_id, skill)
            coverage_reasons = []
            if skill not in selected_skills:
                coverage_reasons.append("coverage_cap:material_skill_budget")
            if form_slot_counts[form_slot] >= args.max_examples_per_material_form:
                coverage_reasons.append("coverage_cap:material_form")
            if skill_slot_counts[skill_slot] >= args.max_examples_per_material_skill:
                coverage_reasons.append("coverage_cap:material_skill")
            if material_counts[material_id] >= args.max_examples_per_material:
                coverage_reasons.append("coverage_cap:material")
            if coverage_reasons:
                reason_counts.update(coverage_reasons)
                quarantined.append({
                    **row,
                    "status": "sft_quarantined_final_gate",
                    "final_training_gate": {
                        "policy": "deterministic_integrity_and_transfer_coverage_cap",
                        "effective_pass": False,
                        "reasons": coverage_reasons,
                    },
                })
                continue
            form_slot_counts[form_slot] += 1
            skill_slot_counts[skill_slot] += 1
            material_counts[material_id] += 1
            global_skill_counts[skill] += 1
            kept.append({**row, "learning_skill": skill})

    write_jsonl(args.output, kept)
    write_jsonl(args.quarantine, quarantined)
    report = {
        "schema_version": "ksova-sft-final-training-gate-report-1.0",
        "input_accepted": len(rows),
        "final_accepted": len(kept),
        "quarantined": len(quarantined),
        "integrity_eligible_before_coverage_cap": len(integrity_kept),
        "unique_material_form_slots": len(form_slot_counts),
        "unique_material_skill_slots": len(skill_slot_counts),
        "max_examples_per_material": args.max_examples_per_material,
        "max_examples_per_material_form": args.max_examples_per_material_form,
        "max_examples_per_material_skill": args.max_examples_per_material_skill,
        "reason_counts": dict(sorted(reason_counts.items())),
        "by_form": dict(sorted(Counter(row["form"] for row in kept).items())),
        "by_learning_skill": dict(sorted(Counter(
            learning_skill(row) for row in kept
        ).items())),
        "by_lane": dict(sorted(Counter(row["lane"] for row in kept).items())),
        "policy": "deterministic_no_rewrite_after_glm_validation",
        "output": str(args.output),
        "quarantine": str(args.quarantine),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_reconcile_review(args: argparse.Namespace) -> int:
    """Keep training examples disjoint from the independently selected benchmark."""
    rows = read_jsonl(args.examples)
    recommendations = read_jsonl(args.recommendations)
    benchmark_issues = read_jsonl(args.benchmark_issues)
    training_materials = read_jsonl(args.training_materials)

    recommendation_counts = Counter(
        str(row.get("issue_id") or "") for row in recommendations
    )
    invalid_recommendation_ids = sorted(
        issue_id
        for issue_id, count in recommendation_counts.items()
        if not issue_id or count != 1
    )
    if invalid_recommendation_ids:
        raise ValueError(
            "recommendations must contain one non-empty row per issue_id: "
            + ", ".join(invalid_recommendation_ids[:10])
        )

    recommendation_by_id = {
        str(row["issue_id"]): row for row in recommendations
    }
    allowed_verdicts = {"core", "supporting", "exclude"}
    invalid_verdicts = sorted({
        str(row.get("final_verdict") or "")
        for row in recommendations
        if row.get("final_verdict") not in allowed_verdicts
    })
    if invalid_verdicts:
        raise ValueError(
            "unsupported final_verdict values: " + ", ".join(invalid_verdicts)
        )

    material_by_id: dict[str, dict[str, Any]] = {}
    duplicate_material_ids = []
    for registry_rows, id_field in (
        (benchmark_issues, "issue_id"),
        (training_materials, "material_id"),
    ):
        for registry_row in registry_rows:
            material_id = str(registry_row.get(id_field) or "")
            if not material_id:
                raise ValueError(f"missing {id_field} in material registry")
            if material_id in material_by_id:
                duplicate_material_ids.append(material_id)
            material_by_id[material_id] = registry_row
    if duplicate_material_ids:
        raise ValueError(
            "duplicate IDs across material registries: "
            + ", ".join(sorted(set(duplicate_material_ids))[:10])
        )

    unresolved_review_ids = sorted(set(recommendation_by_id) - set(material_by_id))
    if unresolved_review_ids:
        raise ValueError(
            "reviewed issues missing from material registries: "
            + ", ".join(unresolved_review_ids[:10])
        )

    core_issue_ids = {
        issue_id
        for issue_id, review in recommendation_by_id.items()
        if review["final_verdict"] == "core"
    }
    core_source_ids = {
        str(source_id)
        for issue_id in core_issue_ids
        for source_id in material_by_id[issue_id].get("source_ids", [])
        if str(source_id)
    }

    kept = []
    quarantined = []
    reason_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    owner_decision_counts: Counter[str] = Counter()
    overlap_source_counts: Counter[str] = Counter()
    kept_material_ids: set[str] = set()
    quarantined_material_ids: set[str] = set()

    for row in rows:
        material_id = str(row.get("material_id") or "")
        review = recommendation_by_id.get(material_id)
        source_ids = {
            str(source_id) for source_id in row.get("source_ids", []) if str(source_id)
        }
        overlapping_sources = sorted(source_ids & core_source_ids)
        reasons = []
        verdict = str((review or {}).get("final_verdict") or "unreviewed")

        if not material_id:
            reasons.append("missing_material_id")
        if review is None and not args.allow_unreviewed:
            reasons.append("final_issue_unreviewed")
        elif review is not None:
            if verdict == "core":
                reasons.append("final_core_benchmark_holdout")
            elif verdict == "exclude":
                reasons.append("final_issue_excluded")
        if not source_ids:
            reasons.append("missing_example_source_ids")
        if overlapping_sources and verdict != "core":
            reasons.append("core_source_overlap")

        audit = {
            "policy": "independent_core_issue_and_source_holdout_v1",
            "effective_pass": not reasons,
            "final_issue_verdict": verdict,
            "owner_issue_decision": (review or {}).get("owner_decision"),
            "review_origin": (review or {}).get("origin"),
            "core_source_overlap": overlapping_sources,
            "reasons": reasons,
        }
        verdict_counts[verdict] += 1
        owner_decision_counts[str((review or {}).get("owner_decision") or "unreviewed")] += 1
        overlap_source_counts.update(overlapping_sources)

        if reasons:
            reason_counts.update(reasons)
            quarantined_material_ids.add(material_id)
            quarantined.append({
                **row,
                "status": "sft_quarantined_benchmark_holdout",
                "benchmark_holdout_gate": audit,
            })
        else:
            kept_material_ids.add(material_id)
            kept.append({**row, "benchmark_holdout_gate": audit})

    write_jsonl(args.output, kept)
    write_jsonl(args.quarantine, quarantined)
    report = {
        "schema_version": "ksova-sft-benchmark-holdout-report-1.0",
        "policy": "independent_core_issue_and_source_holdout_v1",
        "input_examples": len(rows),
        "input_materials": len({str(row.get('material_id') or '') for row in rows}),
        "kept_examples": len(kept),
        "kept_materials": len(kept_material_ids),
        "quarantined_examples": len(quarantined),
        "quarantined_materials": len(quarantined_material_ids),
        "reviewed_issues": len(recommendation_by_id),
        "core_issues": len(core_issue_ids),
        "core_source_ids": len(core_source_ids),
        "allow_unreviewed": args.allow_unreviewed,
        "input_by_final_verdict": dict(sorted(verdict_counts.items())),
        "input_by_owner_decision": dict(sorted(owner_decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "overlapping_source_ids": dict(sorted(overlap_source_counts.items())),
        "output": str(args.output),
        "quarantine": str(args.quarantine),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_combine(args: argparse.Namespace) -> int:
    rows = [row for path in args.inputs for row in read_jsonl(path)]
    ids = [str(row.get("example_id") or "") for row in rows]
    questions = [
        re.sub(r"[^0-9a-z가-힣]", "", str(row.get("question_ko") or "").casefold())
        for row in rows
    ]
    answers = [
        re.sub(r"[^0-9a-z가-힣]", "", str(row.get("answer_ko") or "").casefold())
        for row in rows
    ]
    write_jsonl(args.output, rows)
    report = {
        "schema_version": "ksova-sft-combine-report-1.0",
        "inputs": [str(path) for path in args.inputs],
        "input_counts": {str(path): len(read_jsonl(path)) for path in args.inputs},
        "combined": len(rows),
        "duplicate_example_ids": sum(count - 1 for count in Counter(ids).values() if count > 1),
        "duplicate_normalized_questions": sum(
            count - 1 for count in Counter(questions).values() if count > 1
        ),
        "duplicate_normalized_answers": sum(
            count - 1 for count in Counter(answers).values() if count > 1
        ),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def command_export(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.examples)
    exported = []
    for row in rows:
        gate = row.get("final_training_gate") or {}
        if gate.get("effective_pass") is not True:
            continue
        holdout_gate = row.get("benchmark_holdout_gate")
        if holdout_gate is not None and holdout_gate.get("effective_pass") is not True:
            continue
        exported.append({
            "messages": row["messages"],
            "metadata": {
                "example_id": row["example_id"],
                "material_id": row["material_id"],
                "lane": row["lane"],
                "form": row["form"],
                "learning_skill": learning_skill(row),
                "generation_profile": row.get("generation_profile") or "baseline",
                "source_ids": row["source_ids"],
                "status": row["status"],
                "curriculum_selection": row.get("curriculum_selection"),
                "final_training_gate": gate,
                "benchmark_holdout_gate": holdout_gate,
            },
        })
    write_jsonl(args.output, exported)
    report = {
        "schema_version": "ksova-sft-export-report-1.0",
        "input_examples": len(rows),
        "exported": len(exported),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build_generation = commands.add_parser("build-generation")
    build_generation.add_argument("--materials", type=Path, required=True)
    build_generation.add_argument("--sources", type=Path, required=True)
    build_generation.add_argument("--validation-key", default="glm_material_validation")
    build_generation.add_argument("--run-tag", default="default")
    build_generation.add_argument(
        "--generation-profiles",
        nargs="+",
        choices=sorted(GENERATION_PROFILES),
        default=["actor_decision"],
    )
    build_generation.add_argument("--avoid-examples", type=Path)
    build_generation.add_argument("--total-body-budget", type=int, default=7000)
    build_generation.add_argument("--max-profiles-per-material", type=int, default=3)
    build_generation.add_argument("--max-examples-per-material", type=int, default=4)
    build_generation.add_argument("--output", type=Path, required=True)
    build_generation.add_argument("--report", type=Path, required=True)
    build_generation.set_defaults(func=command_build_generation)

    parse_generation = commands.add_parser("parse-generation")
    parse_generation.add_argument("--requests", type=Path, required=True)
    parse_generation.add_argument("--responses", type=Path, required=True)
    parse_generation.add_argument("--output", type=Path, required=True)
    parse_generation.add_argument("--report", type=Path, required=True)
    parse_generation.set_defaults(func=command_parse_generation)

    select_coverage = commands.add_parser("select-coverage")
    select_coverage.add_argument("--candidates", type=Path, required=True)
    select_coverage.add_argument("--reference", type=Path, action="append", default=[])
    select_coverage.add_argument("--output", type=Path, required=True)
    select_coverage.add_argument("--overflow", type=Path, required=True)
    select_coverage.add_argument("--report", type=Path, required=True)
    select_coverage.add_argument("--max-examples-per-material", type=int, default=4)
    select_coverage.add_argument("--max-examples-per-material-form", type=int, default=1)
    select_coverage.add_argument("--max-examples-per-material-skill", type=int, default=1)
    select_coverage.add_argument("--candidates-per-open-slot", type=int, default=2)
    select_coverage.set_defaults(func=command_select_coverage)

    build_validation = commands.add_parser("build-validation")
    build_validation.add_argument("--materials", type=Path, required=True)
    build_validation.add_argument("--examples", type=Path, required=True)
    build_validation.add_argument("--sources", type=Path, required=True)
    build_validation.add_argument("--total-body-budget", type=int, default=6500)
    build_validation.add_argument("--run-tag", default="default")
    build_validation.add_argument("--output", type=Path, required=True)
    build_validation.add_argument("--report", type=Path, required=True)
    build_validation.set_defaults(func=command_build_validation)

    parse_validation = commands.add_parser("parse-validation")
    parse_validation.add_argument("--examples", type=Path, required=True)
    parse_validation.add_argument("--requests", type=Path, required=True)
    parse_validation.add_argument("--responses", type=Path, required=True)
    parse_validation.add_argument("--accepted", type=Path, required=True)
    parse_validation.add_argument("--rejected", type=Path, required=True)
    parse_validation.add_argument("--report", type=Path, required=True)
    parse_validation.add_argument("--pass-threshold", type=int, default=85)
    parse_validation.set_defaults(func=command_parse_validation)

    owner_x_materials = commands.add_parser("build-owner-x-materials")
    owner_x_materials.add_argument("--issues", type=Path, required=True)
    owner_x_materials.add_argument("--items", type=Path, required=True)
    owner_x_materials.add_argument("--owner-decisions", type=Path, required=True)
    owner_x_materials.add_argument("--output", type=Path, required=True)
    owner_x_materials.add_argument("--excluded", type=Path, required=True)
    owner_x_materials.add_argument("--report", type=Path, required=True)
    owner_x_materials.set_defaults(func=command_build_owner_x_materials)

    import_benchmark = commands.add_parser("import-benchmark-training")
    import_benchmark.add_argument("--items", type=Path, required=True)
    import_benchmark.add_argument("--owner-decisions", type=Path, required=True)
    import_benchmark.add_argument("--max-examples-per-issue", type=int, default=4)
    import_benchmark.add_argument("--output", type=Path, required=True)
    import_benchmark.add_argument("--excluded", type=Path, required=True)
    import_benchmark.add_argument("--report", type=Path, required=True)
    import_benchmark.set_defaults(func=command_import_benchmark_training)

    select_curriculum = commands.add_parser("select-curriculum")
    select_curriculum.add_argument("--examples", type=Path, required=True)
    select_curriculum.add_argument("--owner-decisions", type=Path)
    select_curriculum.add_argument(
        "--require-owner-training-decision", action="store_true"
    )
    select_curriculum.add_argument("--max-examples-per-material", type=int, default=4)
    select_curriculum.add_argument(
        "--max-examples-per-material-skill", type=int, default=1
    )
    select_curriculum.add_argument("--output", type=Path, required=True)
    select_curriculum.add_argument("--excluded", type=Path, required=True)
    select_curriculum.add_argument("--report", type=Path, required=True)
    select_curriculum.set_defaults(func=command_select_curriculum)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--accepted", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--quarantine", type=Path, required=True)
    finalize.add_argument("--report", type=Path, required=True)
    finalize.add_argument("--pass-threshold", type=int, default=85)
    finalize.add_argument("--max-examples-per-material", type=int, default=4)
    finalize.add_argument("--max-examples-per-material-form", type=int, default=1)
    finalize.add_argument("--max-examples-per-material-skill", type=int, default=1)
    finalize.set_defaults(func=command_finalize)

    reconcile_review = commands.add_parser("reconcile-review")
    reconcile_review.add_argument("--examples", type=Path, required=True)
    reconcile_review.add_argument("--recommendations", type=Path, required=True)
    reconcile_review.add_argument("--benchmark-issues", type=Path, required=True)
    reconcile_review.add_argument("--training-materials", type=Path, required=True)
    reconcile_review.add_argument("--output", type=Path, required=True)
    reconcile_review.add_argument("--quarantine", type=Path, required=True)
    reconcile_review.add_argument("--report", type=Path, required=True)
    reconcile_review.add_argument("--allow-unreviewed", action="store_true")
    reconcile_review.set_defaults(func=command_reconcile_review)

    combine = commands.add_parser("combine")
    combine.add_argument("--inputs", type=Path, nargs="+", required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--report", type=Path, required=True)
    combine.set_defaults(func=command_combine)

    export = commands.add_parser("export")
    export.add_argument("--examples", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--report", type=Path, required=True)
    export.set_defaults(func=command_export)

    args = parser.parse_args()
    if hasattr(args, "max_profiles_per_material") and args.max_profiles_per_material < 1:
        raise ValueError("max_profiles_per_material must be positive")
    if hasattr(args, "max_examples_per_material") and args.max_examples_per_material < 1:
        raise ValueError("max_examples_per_material must be positive")
    if (
        hasattr(args, "max_examples_per_material_form")
        and args.max_examples_per_material_form < 1
    ):
        raise ValueError("max_examples_per_material_form must be positive")
    if (
        hasattr(args, "max_examples_per_material_skill")
        and args.max_examples_per_material_skill < 1
    ):
        raise ValueError("max_examples_per_material_skill must be positive")
    if hasattr(args, "candidates_per_open_slot") and args.candidates_per_open_slot < 1:
        raise ValueError("candidates_per_open_slot must be positive")
    if hasattr(args, "max_examples_per_issue") and args.max_examples_per_issue < 1:
        raise ValueError("max_examples_per_issue must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
