#!/usr/bin/env python3
"""Build a benchmark-isolated, on-policy K-SOVA Direct-DPO experiment.

The pipeline is intentionally stage based.  It selects a reviewed and
domain-capped issue portfolio, asks a teacher only for user prompts, records
the base model's own answers, and then asks an editor to make a source-bounded
single-defect repair.  Deterministic gates and an independent model review are
applied before DPO files are materialized.  Canonical data and the frozen
benchmark are read-only inputs.
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


EXPECTED_BENCHMARK_SHA256 = (
    "0ca49e7bc94178300c0309919d63a51c0496d84d6e352501d350f0deac494032"
)
DOMAIN_MAP = {
    "T-history_responsibility_memory": "D04",
    "T-north_korea_interkorean": "D06",
    "T-trade_industry_supply_chain": "D07",
}
# Only 155 reviewed non-benchmark issues are fully source-isolated from the
# frozen benchmark.  These quotas retain 150 of them; domain imbalance is
# controlled at optimization time with inverse-frequency loss weights rather
# than by leaking benchmark-linked sources or duplicating scarce examples.
ISSUE_QUOTAS = {
    "D01": 5,
    "D02": 24,
    "D03": 10,
    "D04": 10,
    "D05": 5,
    "D06": 27,
    "D07": 63,
    "D08": 6,
}
DEV_QUOTAS = {
    "D01": 1,
    "D02": 2,
    "D03": 2,
    "D04": 2,
    "D05": 1,
    "D06": 3,
    "D07": 3,
    "D08": 1,
}
PROMPT_FORMS = (
    "neutral_synthesis",
    "claim_review",
    "policy_comparison",
    "policy_recommendation",
)
DEFECT_TYPES = {
    "korean_stake_omission",
    "counterargument_omission",
    "condition_cost_omission",
    "conclusion_evasion",
    "unsupported_specificity",
}
VALIDATION_KEYS = (
    "rejected_is_base_derived",
    "rejected_factually_bounded",
    "chosen_factually_bounded",
    "single_material_defect_in_rejected",
    "chosen_repairs_assigned_defect",
    "chosen_korean_stake_clear",
    "chosen_counterargument_addressed",
    "chosen_conditions_and_costs_present",
    "chosen_has_clear_judgment",
    "length_and_style_balanced",
    "no_government_mimicry_or_nationalist_overreach",
    "pair_useful_for_preference_learning",
)
META_PATTERNS = (
    r"(?:chosen|rejected)[ _-]?response",
    r"(?:선호|비선호)\s*답변",
    r"학습용\s*(?:정답|오답|답변)",
    r"(?:의도적으로|일부러).{0,20}(?:누락|생략)",
)
PERSPECTIVE_CUE_PATTERNS = (
    r"한국(?:의)?\s*관점(?:에서|으로)",
    r"한국(?:인의)?\s*입장(?:에서|으로)",
    r"한국\s*중심(?:으로|의)",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
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


def stable_hash(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()


def normalized(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_domain(value: Any) -> str:
    raw = str(value or "")
    return DOMAIN_MAP.get(raw, raw)


def remove_explicit_perspective_cues(text: str) -> tuple[str, list[str]]:
    repaired = text
    applied: list[str] = []
    replacements = (
        (r"한국(?:의)?\s*관점(?:에서|으로)", ""),
        (r"한국(?:인의)?\s*입장(?:에서|으로)", ""),
        (r"한국\s*중심(?:으로|의)", ""),
    )
    for pattern, replacement in replacements:
        updated, count = re.subn(pattern, replacement, repaired)
        if count:
            repaired = updated
            applied.append("removed_explicit_answer_perspective_cue")
    repaired = re.sub(r"\s+", " ", repaired)
    repaired = re.sub(r"\s+([,?.!])", r"\1", repaired).strip()
    return repaired, applied


def parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE
    )
    if fenced:
        value = fenced.group(1).strip()
    candidates = [value]
    first, last = value.find("{"), value.rfind("}")
    if first >= 0 and last > first:
        candidates.append(value[first : last + 1])
    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            errors.append("not_object")
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
    raise ValueError("could not parse JSON object: " + " | ".join(errors))


def source_url(row: dict[str, Any]) -> str:
    return normalized(row.get("normalized_url") or row.get("url"))


def compact_issue(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
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
    result = {key: row.get(key) for key in keys}
    validation = ((row.get("selection_review") or {}).get("validation") or {})
    source_assessments = validation.get("source_assessments") or []
    if source_assessments:
        result["validated_source_assessments"] = source_assessments
    return result


def quality_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    validation = ((row.get("selection_review") or {}).get("validation") or {})
    maximum_similarity = row.get("maximum_semantic_similarity")
    if maximum_similarity is None:
        maximum_similarity = row.get("maximum_existing_similarity")
    try:
        similarity = float(maximum_similarity)
    except (TypeError, ValueError):
        similarity = 1.0
    return (
        int(row.get("review_status") == "llm_body_validated"),
        int(validation.get("effective_pass") is True),
        int(row.get("lifecycle_status") == "train_selected"),
        float(validation.get("quality_score") or 0),
        len(row.get("source_ids") or []),
        -similarity,
        stable_hash("cycle02-quality-tie", str(row.get("issue_id"))),
    )


def prompt_generation_system() -> str:
    return """당신은 K-SOVA Direct-DPO 실험용 사용자 질의를 설계한다.

K-SOVA는 한국 관련 단순 지식이 아니라, 외국과 권리·규범·이해관계가 충돌할 수 있는 사안에서 한국 사회에 형성된 담론을 바탕으로 사실, 권리, 가치, 상대 논거, 비용과 조건을 결합하는 능력을 본다.

검증된 쟁점 카드 안의 정보만 사용하여 같은 핵심 쟁점에 대한 서로 다른 질의 4개를 만든다. 네 질의는 각각 neutral_synthesis, claim_review, policy_comparison, policy_recommendation이어야 한다. 단순 사실 확인형이나 서로의 바꿔쓰기가 되어서는 안 된다. 질문 자체가 정답 방향을 노출하는 '한국 관점에서', '한국인의 입장에서', '한국 중심으로' 같은 지시를 쓰지 않는다. 다만 사안의 당사자인 한국·한국인·한국 기업 등을 자연스럽게 명시하는 것은 허용한다. 카드 밖의 인명·기관·수치·날짜·법적 효과를 만들지 않는다.

질의는 각각 독립적으로 이해할 수 있어야 하고 한국어 60~350자를 목표로 한다. JSON 객체 하나만 출력한다. 키는 issue_id와 questions이며, questions는 prompt_form과 question_ko를 가진 객체 4개의 배열이다."""


def editor_system() -> str:
    return """당신은 K-SOVA on-policy 선호쌍을 만드는 엄격한 최소수정 편집자다.

입력에는 검증된 쟁점 카드, 사용자 질의, 그리고 Qwen3.5-9B 베이스 모델이 실제로 생성한 응답이 들어 있다. 다음 순서를 반드시 따른다.

1. 베이스 응답에서 쟁점 카드가 뒷받침하지 않는 인명·기관·수치·날짜·법적 효과가 있으면 삭제하거나 주장·불확실성으로 한정한다. 그 밖의 문장, 구조, 어조는 최대한 보존하여 rejected_response를 만든다.
2. rejected_response에서 가장 중요한 결함 하나만 고른다. 허용 결함은 korean_stake_omission, counterargument_omission, condition_cost_omission, conclusion_evasion, unsupported_specificity 중 하나다.
3. 그 결함만 최소한으로 고쳐 chosen_response를 만든다. 사실과 행위자 주장을 구분하고, 한국 사회에서 방어 가능한 권리·이해관계, 가장 강한 상대 논거, 명확한 판단, 비용·조건·불확실성 경계를 질문에 필요한 범위에서 통합한다.

두 응답은 모두 최종 사용자 답변이어야 하며 메타 해설을 포함하지 않는다. chosen과 rejected의 길이는 토큰 기준 ±10% 안에 맞추기 위해, chosen에 필요한 내용을 보태는 만큼 중복 표현을 줄여라. 전면 재작성하지 않는다. 베이스 응답이 이미 충분하여 실질적 결함 하나를 만들 수 없거나, 사실 경계를 바로잡는 것만으로도 구조를 크게 다시 써야 한다면 usable을 false로 한다. rejected에 고의로 새 오류를 넣지 않는다.

JSON 객체 하나만 출력하며 키는 prompt_id, usable, defect_type, rejected_response, chosen_response, base_factual_repairs_ko, chosen_delta_ko, evidence_used_ko다."""


def recovery_editor_system() -> str:
    return """당신은 K-SOVA on-policy 선호쌍의 실패 항목만 다시 만드는 복구 편집자다.

입력에는 검증된 쟁점 카드, 사용자 질의, Qwen3.5-9B 베이스 응답과 직전 자동검사 실패 사유가 있다. 카드 밖의 사실을 보태지 말고 다음을 정확히 수행한다.

1. 베이스 응답에서 카드가 뒷받침하지 않는 인명·기관·수치·날짜·법적 효과만 삭제하거나 불확실한 주장으로 한정하여 rejected_response를 만든다. 그 밖의 구조와 어조는 보존한다.
2. rejected_response의 가장 중요한 K-SOVA 결함 하나를 고른다. 허용값은 korean_stake_omission, counterargument_omission, condition_cost_omission, conclusion_evasion, unsupported_specificity뿐이다.
3. 그 결함 하나만 최소 수정한 chosen_response를 만든다. 질문에 필요한 한국의 구체적 권리·이해관계, 강한 상대 논거, 판단, 비용·조건·불확실성 경계를 카드가 허용하는 범위에서 반영한다.

rejected와 chosen은 각각 독립적인 최종 사용자 답변이어야 한다. 두 답변은 각각 300~2200 토큰이고 토큰 길이 차이는 ±10% 이내여야 한다. 전면 재작성, 카드에 없는 수치·날짜·고유명사, 메타 해설, 의도적으로 만든 새 오류는 금지한다. 직전 실패 사유가 지적한 형식·길이·사실성 문제를 반드시 해소한다.

JSON 객체 하나만 출력한다. 다음 키 8개를 아래 순서로 모두 포함하며 어떤 키도 생략하지 않는다: prompt_id, usable, defect_type, chosen_response, rejected_response, base_factual_repairs_ko, chosen_delta_ko, evidence_used_ko. usable이 true일 때 chosen_response와 rejected_response는 절대 빈 문자열이면 안 된다. 근거 없는 사실 수정이 없으면 base_factual_repairs_ko에는 '없음'을 쓴다."""


def validation_system() -> str:
    return """당신은 K-SOVA on-policy Direct-DPO 선호쌍의 독립 검수자다. 답변을 고치지 말고 검증된 쟁점 카드와 질의, 베이스 원응답, 두 후보를 엄격히 판정한다.

rejected는 베이스 원응답에서 근거 없는 구체 사실만 삭제·한정한 형태여야 하며, 새 오류를 넣거나 전면 재작성해서는 안 된다. chosen은 rejected의 지정 결함 하나만 실질적으로 고쳐야 한다. 두 답변 모두 쟁점 카드 밖의 인명·기관·수치·날짜·법적 효과를 확정해서는 안 된다. chosen은 질문에 필요한 범위에서 한국의 구체적 권리·이해관계, 강한 상대 논거, 비용·조건·불확실성 경계와 명확한 판단을 통합해야 한다. 정부의 공식 대변인처럼 단정하거나 국수주의적 과잉을 보이면 안 된다. 길이나 문체 차이만으로 우열이 생겨서도 안 된다.

불확실하면 false로 판정한다. 아래 12개 검사 키의 값은 설명문·결함명·문자열이 아니라 반드시 JSON 리터럴 true 또는 false 중 하나여야 한다. 지정 결함의 이름이나 설명도 검사 키에 쓰지 말고 모든 설명은 reasons_ko에만 쓴다. JSON 객체 하나만 출력하며 다음 키를 정확히 사용한다.
prompt_id, rejected_is_base_derived, rejected_factually_bounded, chosen_factually_bounded, single_material_defect_in_rejected, chosen_repairs_assigned_defect, chosen_korean_stake_clear, chosen_counterargument_addressed, chosen_conditions_and_costs_present, chosen_has_clear_judgment, length_and_style_balanced, no_government_mimicry_or_nationalist_overreach, pair_useful_for_preference_learning, verdict, reasons_ko

출력 형태는 다음과 같다. 값의 자료형을 바꾸지 않는다.
{"prompt_id":"입력값", "rejected_is_base_derived":true, "rejected_factually_bounded":true, "chosen_factually_bounded":true, "single_material_defect_in_rejected":true, "chosen_repairs_assigned_defect":true, "chosen_korean_stake_clear":true, "chosen_counterargument_addressed":true, "chosen_conditions_and_costs_present":true, "chosen_has_clear_judgment":true, "length_and_style_balanced":true, "no_government_mimicry_or_nationalist_overreach":true, "pair_useful_for_preference_learning":true, "verdict":"pass", "reasons_ko":"판정 설명"}

모든 불리언이 true일 때만 verdict를 pass로 한다."""


def prepare(args: argparse.Namespace) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    issues = read_jsonl(args.issues)
    sources = read_jsonl(args.sources)
    benchmark = read_jsonl(args.benchmark_items)
    if len(issues) != 500 or len({row["issue_id"] for row in issues}) != 500:
        raise ValueError("expected 500 unique canonical issues")
    if len(benchmark) != 480 or sha256(args.benchmark_items) != EXPECTED_BENCHMARK_SHA256:
        raise ValueError("unexpected frozen benchmark")
    source_by_id = {str(row["source_id"]): row for row in sources}
    benchmark_issue_ids = {str(row["issue_id"]) for row in benchmark}
    benchmark_source_ids = {
        str(source_id)
        for row in benchmark
        for source_id in row.get("source_ids") or []
    }
    benchmark_urls = {
        source_url(source_by_id[source_id])
        for source_id in benchmark_source_ids
        if source_id in source_by_id and source_url(source_by_id[source_id])
    }
    benchmark_hashes = {
        normalized(source_by_id[source_id].get("content_sha256"))
        for source_id in benchmark_source_ids
        if source_id in source_by_id
        and normalized(source_by_id[source_id].get("content_sha256"))
    }

    eligible_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []
    for row in issues:
        issue_id = str(row["issue_id"])
        reasons: list[str] = []
        domain_id = normalize_domain(row.get("domain_id"))
        if issue_id in benchmark_issue_ids or row.get("lifecycle_status") == "benchmark_selected":
            reasons.append("benchmark_issue")
        if row.get("review_status") == "not_reviewed":
            reasons.append("not_reviewed")
        if domain_id not in ISSUE_QUOTAS:
            reasons.append("unmapped_domain")
        issue_source_ids = {str(value) for value in row.get("source_ids") or []}
        missing = sorted(issue_source_ids - set(source_by_id))
        if missing:
            reasons.append("missing_sources")
        issue_urls = {
            source_url(source_by_id[source_id])
            for source_id in issue_source_ids
            if source_id in source_by_id and source_url(source_by_id[source_id])
        }
        issue_hashes = {
            normalized(source_by_id[source_id].get("content_sha256"))
            for source_id in issue_source_ids
            if source_id in source_by_id
            and normalized(source_by_id[source_id].get("content_sha256"))
        }
        if issue_source_ids & benchmark_source_ids:
            reasons.append("benchmark_source_id_overlap")
        if issue_urls & benchmark_urls:
            reasons.append("benchmark_url_overlap")
        if issue_hashes & benchmark_hashes:
            reasons.append("benchmark_content_hash_overlap")
        if reasons:
            excluded.append({"issue_id": issue_id, "reasons": reasons})
        else:
            eligible_by_domain[domain_id].append(row)

    selected: list[dict[str, Any]] = []
    for domain_id, quota in ISSUE_QUOTAS.items():
        candidates = sorted(eligible_by_domain[domain_id], key=quality_rank, reverse=True)
        if len(candidates) < quota:
            raise ValueError(
                f"domain {domain_id} has {len(candidates)} eligible issues, needs {quota}"
            )
        selected.extend(candidates[:quota])
    if len(selected) != 150 or len({row["issue_id"] for row in selected}) != 150:
        raise AssertionError("selection must contain 150 unique issues")

    dev_ids: set[str] = set()
    for domain_id, quota in DEV_QUOTAS.items():
        domain_rows = [row for row in selected if normalize_domain(row["domain_id"]) == domain_id]
        ordered = sorted(
            domain_rows,
            key=lambda row: stable_hash("cycle02-dev", str(row["issue_id"])),
        )
        dev_ids.update(str(row["issue_id"]) for row in ordered[:quota])
    if len(dev_ids) != 15:
        raise AssertionError("dev split must contain 15 issues")

    selected_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for issue in sorted(selected, key=lambda row: str(row["issue_id"])):
        issue_id = str(issue["issue_id"])
        domain_id = normalize_domain(issue["domain_id"])
        split = "dev" if issue_id in dev_ids else "train"
        card = compact_issue(issue)
        selected_row = {
            "issue_id": issue_id,
            "domain_id": domain_id,
            "split": split,
            "lifecycle_status": issue.get("lifecycle_status"),
            "review_status": issue.get("review_status"),
            "quality_rank": list(quality_rank(issue)[:-1]),
            "source_ids": issue.get("source_ids") or [],
            "issue_card": card,
        }
        selected_rows.append(selected_row)
        payload = {"issue_id": issue_id, "validated_issue_card": card}
        requests.append({
            "id": f"cycle02_prompt_generate__{issue_id}",
            "messages": [
                {"role": "system", "content": prompt_generation_system()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 2048,
            "metadata": {
                "purpose": "ksova_cycle02_prompt_generation",
                "experiment_id": args.experiment_id,
                "issue_id": issue_id,
                "domain_id": domain_id,
                "split": split,
            },
        })
    write_jsonl(args.output_dir / "selected_issues.jsonl", selected_rows)
    write_jsonl(args.output_dir / "prompt_generation_requests.jsonl", requests)
    report = {
        "schema_version": "ksova-onpolicy-dpo-preparation-1.0",
        "experiment_id": args.experiment_id,
        "selected_issues": 150,
        "train_issues": 135,
        "dev_issues": 15,
        "issue_quotas": ISSUE_QUOTAS,
        "dev_quotas": DEV_QUOTAS,
        "eligible_domain_counts": {
            domain: len(rows) for domain, rows in sorted(eligible_by_domain.items())
        },
        "selected_domain_counts": dict(Counter(
            normalize_domain(row["domain_id"]) for row in selected
        )),
        "train_domain_counts": dict(Counter(
            normalize_domain(row["domain_id"])
            for row in selected if str(row["issue_id"]) not in dev_ids
        )),
        "dev_domain_counts": dict(Counter(
            normalize_domain(row["domain_id"])
            for row in selected if str(row["issue_id"]) in dev_ids
        )),
        "excluded_reason_counts": dict(Counter(
            reason for row in excluded for reason in row["reasons"]
        )),
        "benchmark": {
            "items": 480,
            "issues": len(benchmark_issue_ids),
            "sha256": EXPECTED_BENCHMARK_SHA256,
        },
        "isolation": {
            "issue_id_overlap": 0,
            "source_id_overlap": 0,
            "normalized_url_overlap": 0,
            "content_sha256_overlap": 0,
        },
        "inputs": {
            "issues_sha256": sha256(args.issues),
            "sources_sha256": sha256(args.sources),
            "benchmark_sha256": sha256(args.benchmark_items),
        },
        "outputs": {
            "selected_issues_sha256": sha256(args.output_dir / "selected_issues.jsonl"),
            "prompt_requests_sha256": sha256(args.output_dir / "prompt_generation_requests.jsonl"),
        },
        "dev_issue_ids": sorted(dev_ids),
    }
    write_json(args.output_dir / "preparation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_prompts(args: argparse.Namespace) -> int:
    selected = read_jsonl(args.selected_issues)
    responses = read_jsonl(args.responses)
    response_by_id = {str(row.get("id")): row for row in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("duplicate prompt-generation response IDs")
    benchmark_prompts = {
        normalized(row.get("question_ko")) for row in read_jsonl(args.benchmark_items)
    }
    prompt_rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    repair_counts: Counter[str] = Counter()
    seen_prompts: set[str] = set()
    for issue in selected:
        issue_id = str(issue["issue_id"])
        response_id = f"cycle02_prompt_generate__{issue_id}"
        response = response_by_id.get(response_id)
        reasons: list[str] = []
        parsed: dict[str, Any] | None = None
        if response is None:
            reasons.append("missing_response")
        elif response.get("error"):
            reasons.append("response_error")
        elif str(response.get("finish_reason") or "").casefold() != "stop":
            reasons.append("response_not_stopped")
        else:
            try:
                parsed = parse_json_object(str(response.get("response") or ""))
            except ValueError as exc:
                reasons.append(f"json_parse:{exc}")
        questions: list[dict[str, Any]] = []
        if parsed is not None:
            if str(parsed.get("issue_id")) != issue_id:
                reasons.append("issue_id_mismatch")
            if not isinstance(parsed.get("questions"), list):
                reasons.append("questions_not_array")
            else:
                questions = parsed["questions"]
                forms = [str(row.get("prompt_form")) for row in questions if isinstance(row, dict)]
                if len(questions) != 4 or sorted(forms) != sorted(PROMPT_FORMS):
                    reasons.append("prompt_form_set_mismatch")
        local_rows: list[dict[str, Any]] = []
        for question in questions if not reasons else []:
            form = str(question.get("prompt_form"))
            structural_repairs: list[str] = []
            raw_text = question.get("question_ko")
            if not raw_text and question.get("prompt_ko"):
                raw_text = question.get("prompt_ko")
                structural_repairs.append("normalized_prompt_ko_alias_to_question_ko")
            text = normalized(raw_text)
            text, cue_repairs = remove_explicit_perspective_cues(text)
            structural_repairs.extend(cue_repairs)
            if not 40 <= len(text) <= 700:
                reasons.append(f"question_length:{form}:{len(text)}")
                continue
            if any(re.search(pattern, text) for pattern in PERSPECTIVE_CUE_PATTERNS):
                reasons.append(f"perspective_cue:{form}")
                continue
            normalized_text = normalized(text)
            if normalized_text in benchmark_prompts:
                reasons.append(f"benchmark_prompt_overlap:{form}")
                continue
            if normalized_text in seen_prompts:
                reasons.append(f"duplicate_prompt:{form}")
                continue
            prompt_id = "KSC2PROMPT-" + stable_hash(
                "cycle02-prompt", f"{issue_id}\0{form}\0{normalized_text}"
            )[:20]
            local_rows.append({
                "prompt_id": prompt_id,
                "issue_id": issue_id,
                "domain_id": issue["domain_id"],
                "split": issue["split"],
                "prompt_form": form,
                "question_ko": text,
                "issue_card": issue["issue_card"],
                "source_ids": issue["source_ids"],
                "prompt_generation_model": response.get("model") if response else None,
                "prompt_generation_messages_sha256": (
                    response.get("messages_sha256") if response else None
                ),
                "structural_repairs": structural_repairs,
            })
        if reasons:
            invalid.append({"issue_id": issue_id, "reasons": reasons})
            continue
        for row in local_rows:
            repair_counts.update(row["structural_repairs"])
            seen_prompts.add(normalized(row["question_ko"]))
            prompt_rows.append(row)
    prompt_rows.sort(key=lambda row: (row["split"], row["issue_id"], row["prompt_form"]))
    if invalid:
        write_jsonl(args.output_dir / "invalid_prompt_generations.jsonl", invalid)
        raise ValueError(f"prompt generation failed for {len(invalid)} issues")
    if len(prompt_rows) != 600:
        raise ValueError(f"expected 600 prompts, got {len(prompt_rows)}")
    base_requests = [{
        "id": f"cycle02_base_answer__{row['prompt_id']}",
        "messages": [{"role": "user", "content": row["question_ko"]}],
        "max_tokens": 2048,
        "metadata": {
            "purpose": "ksova_cycle02_onpolicy_base_generation",
            "experiment_id": args.experiment_id,
            "prompt_id": row["prompt_id"],
            "issue_id": row["issue_id"],
            "domain_id": row["domain_id"],
            "split": row["split"],
            "prompt_form": row["prompt_form"],
        },
    } for row in prompt_rows]
    write_jsonl(args.output_dir / "prompts.jsonl", prompt_rows)
    write_jsonl(args.output_dir / "base_generation_requests.jsonl", base_requests)
    report = {
        "schema_version": "ksova-onpolicy-prompt-parse-1.0",
        "experiment_id": args.experiment_id,
        "prompts": len(prompt_rows),
        "train_prompts": sum(row["split"] == "train" for row in prompt_rows),
        "dev_prompts": sum(row["split"] == "dev" for row in prompt_rows),
        "issues": len({row["issue_id"] for row in prompt_rows}),
        "form_counts": dict(Counter(row["prompt_form"] for row in prompt_rows)),
        "domain_counts": dict(Counter(row["domain_id"] for row in prompt_rows)),
        "structural_repair_counts": dict(repair_counts),
        "outputs": {
            "prompts_sha256": sha256(args.output_dir / "prompts.jsonl"),
            "base_requests_sha256": sha256(args.output_dir / "base_generation_requests.jsonl"),
        },
    }
    write_json(args.output_dir / "prompt_parse_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_editor(args: argparse.Namespace) -> int:
    prompts = read_jsonl(args.prompts)
    responses = read_jsonl(args.base_responses)
    response_by_id = {str(row.get("id")): row for row in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("duplicate base response IDs")
    requests: list[dict[str, Any]] = []
    valid_base = 0
    for row in prompts:
        response_id = f"cycle02_base_answer__{row['prompt_id']}"
        response = response_by_id.get(response_id)
        if response is None or response.get("error"):
            raise ValueError(f"missing/errored base response for {row['prompt_id']}")
        if str(response.get("finish_reason") or "").casefold() != "stop":
            raise ValueError(f"non-stopped base response for {row['prompt_id']}")
        answer = str(response.get("response") or "").strip()
        if not answer:
            raise ValueError(f"empty base response for {row['prompt_id']}")
        valid_base += 1
        payload = {
            "prompt_id": row["prompt_id"],
            "question_ko": row["question_ko"],
            "validated_issue_card": row["issue_card"],
            "base_response": answer,
        }
        requests.append({
            "id": f"cycle02_pair_edit__{row['prompt_id']}",
            "messages": [
                {"role": "system", "content": editor_system()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 4096,
            "metadata": {
                "purpose": "ksova_cycle02_onpolicy_pair_edit",
                "experiment_id": args.experiment_id,
                "prompt_id": row["prompt_id"],
                "issue_id": row["issue_id"],
                "domain_id": row["domain_id"],
                "split": row["split"],
                "prompt_form": row["prompt_form"],
                "base_response_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-onpolicy-editor-requests-1.0",
        "experiment_id": args.experiment_id,
        "prompts": len(prompts),
        "valid_base_responses": valid_base,
        "requests": len(requests),
        "base_responses_sha256": sha256(args.base_responses),
        "requests_sha256": sha256(args.output),
    }
    write_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_recovery_editor(args: argparse.Namespace) -> int:
    original_requests = read_jsonl(args.original_requests)
    invalid_pairs = read_jsonl(args.invalid_pairs)
    request_by_prompt_id = {
        str((row.get("metadata") or {}).get("prompt_id")): row
        for row in original_requests
    }
    if len(request_by_prompt_id) != len(original_requests):
        raise ValueError("duplicate or empty prompt IDs in original editor requests")
    invalid_by_prompt_id = {str(row["prompt_id"]): row for row in invalid_pairs}
    if len(invalid_by_prompt_id) != len(invalid_pairs):
        raise ValueError("duplicate prompt IDs in invalid pairs")
    missing = sorted(set(invalid_by_prompt_id) - set(request_by_prompt_id))
    if missing:
        raise ValueError(f"invalid pairs missing from original requests: {missing[:5]}")

    requests: list[dict[str, Any]] = []
    for prompt_id in sorted(invalid_by_prompt_id):
        original = request_by_prompt_id[prompt_id]
        original_payload = json.loads(str(original["messages"][1]["content"]))
        original_payload["previous_deterministic_failures"] = invalid_by_prompt_id[
            prompt_id
        ].get("deterministic_errors") or []
        metadata = dict(original.get("metadata") or {})
        metadata["purpose"] = "ksova_cycle02_onpolicy_pair_edit_recovery"
        metadata["recovery_of_model"] = args.primary_editor_model
        requests.append({
            "id": f"cycle02_pair_edit__{prompt_id}",
            "messages": [
                {"role": "system", "content": recovery_editor_system()},
                {"role": "user", "content": json.dumps(original_payload, ensure_ascii=False)},
            ],
            "max_tokens": 4096,
            "metadata": metadata,
        })

    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-onpolicy-editor-recovery-requests-1.0",
        "experiment_id": args.experiment_id,
        "primary_editor_model": args.primary_editor_model,
        "recovery_editor_model": args.recovery_editor_model,
        "invalid_pairs": len(invalid_pairs),
        "requests": len(requests),
        "invalid_pairs_sha256": sha256(args.invalid_pairs),
        "original_requests_sha256": sha256(args.original_requests),
        "requests_sha256": sha256(args.output),
    }
    write_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def merge_edit_responses(args: argparse.Namespace) -> int:
    prompts = read_jsonl(args.prompts)
    primary = read_jsonl(args.primary_responses)
    recovery = read_jsonl(args.recovery_responses)
    primary_by_id = {str(row.get("id")): row for row in primary}
    recovery_by_id = {str(row.get("id")): row for row in recovery}
    if len(primary_by_id) != len(primary):
        raise ValueError("duplicate IDs in primary editor responses")
    if len(recovery_by_id) != len(recovery):
        raise ValueError("duplicate IDs in recovery editor responses")
    expected_ids = [f"cycle02_pair_edit__{row['prompt_id']}" for row in prompts]
    unexpected = sorted((set(primary_by_id) | set(recovery_by_id)) - set(expected_ids))
    if unexpected:
        raise ValueError(f"unexpected editor response IDs: {unexpected[:5]}")
    missing = sorted(set(expected_ids) - (set(primary_by_id) | set(recovery_by_id)))
    if missing:
        raise ValueError(f"missing editor response IDs: {missing[:5]}")
    merged = [
        recovery_by_id.get(response_id, primary_by_id[response_id])
        for response_id in expected_ids
    ]
    write_jsonl(args.output, merged)
    report = {
        "schema_version": "ksova-onpolicy-editor-response-merge-1.0",
        "prompts": len(prompts),
        "primary_responses": len(primary),
        "recovery_responses": len(recovery),
        "recovery_overrides": len(set(primary_by_id) & set(recovery_by_id)),
        "merged_responses": len(merged),
        "primary_responses_sha256": sha256(args.primary_responses),
        "recovery_responses_sha256": sha256(args.recovery_responses),
        "merged_responses_sha256": sha256(args.output),
    }
    write_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def korean_signal(text: str) -> float:
    letters = re.findall(r"[A-Za-z가-힣]", text)
    if not letters:
        return 0.0
    return sum(bool(re.fullmatch(r"[가-힣]", char)) for char in letters) / len(letters)


def textual_similarity(left: str, right: str) -> float:
    """Return a linear-time multiset Dice score over normalized 5-grams.

    Character-level ``SequenceMatcher(..., autojunk=False)`` becomes quadratic
    on the long, highly repetitive Korean answers used here.  A multiset
    character n-gram score retains the two properties needed by the gate
    (identity is 1.0; local edits remain close) without making validation time
    depend quadratically on answer length.
    """
    left_normalized = normalized(left)
    right_normalized = normalized(right)
    if left_normalized == right_normalized:
        return 1.0
    width = 5
    if len(left_normalized) < width or len(right_normalized) < width:
        return 0.0
    left_grams = Counter(
        left_normalized[index:index + width]
        for index in range(len(left_normalized) - width + 1)
    )
    right_grams = Counter(
        right_normalized[index:index + width]
        for index in range(len(right_normalized) - width + 1)
    )
    shared = sum(
        min(count, right_grams.get(gram, 0))
        for gram, count in left_grams.items()
    )
    return 2 * shared / (sum(left_grams.values()) + sum(right_grams.values()))


def numeric_tokens(text: str) -> set[str]:
    values = set(re.findall(r"(?<![A-Za-z가-힣])\d[\d,./:%-]*", text))
    return {value.strip(".,:;/%-") for value in values if value.strip(".,:;/%-")}


def parse_edits(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    prompts = read_jsonl(args.prompts)
    base_responses = read_jsonl(args.base_responses)
    edit_responses = read_jsonl(args.edit_responses)
    prompt_by_id = {str(row["prompt_id"]): row for row in prompts}
    base_by_id = {str(row["id"]): row for row in base_responses}
    edit_by_id = {str(row["id"]): row for row in edit_responses}
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path), local_files_only=True, trust_remote_code=True
    )
    accepted: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for prompt_id, prompt in prompt_by_id.items():
        base = base_by_id.get(f"cycle02_base_answer__{prompt_id}")
        edit = edit_by_id.get(f"cycle02_pair_edit__{prompt_id}")
        reasons: list[str] = []
        parsed: dict[str, Any] | None = None
        if base is None or base.get("error") or not str(base.get("response") or "").strip():
            reasons.append("invalid_base_response")
            base_text = ""
        else:
            base_text = str(base["response"]).strip()
        if edit is None:
            reasons.append("missing_edit_response")
        elif edit.get("error"):
            reasons.append("edit_response_error")
        elif str(edit.get("finish_reason") or "").casefold() != "stop":
            reasons.append("edit_response_not_stopped")
        else:
            try:
                parsed = parse_json_object(str(edit.get("response") or ""))
            except ValueError as exc:
                reasons.append(f"edit_json_parse:{exc}")
        if parsed is not None:
            if str(parsed.get("prompt_id")) != prompt_id:
                reasons.append("prompt_id_mismatch")
            if parsed.get("usable") is not True:
                reasons.append("editor_marked_unusable")
            defect = str(parsed.get("defect_type") or "")
            if defect not in DEFECT_TYPES:
                reasons.append("invalid_defect_type")
            chosen = str(parsed.get("chosen_response") or "").strip()
            rejected = str(parsed.get("rejected_response") or "").strip()
            if not chosen:
                reasons.append("empty_chosen")
            if not rejected:
                reasons.append("empty_rejected")
            chosen_tokens = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
            rejected_tokens = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])
            if not 300 <= chosen_tokens <= 2200:
                reasons.append(f"chosen_token_length:{chosen_tokens}")
            if not 300 <= rejected_tokens <= 2200:
                reasons.append(f"rejected_token_length:{rejected_tokens}")
            ratio = chosen_tokens / rejected_tokens if rejected_tokens else 0.0
            if not 0.90 <= ratio <= 1.10:
                reasons.append(f"completion_token_ratio:{ratio:.5f}")
            base_similarity = textual_similarity(base_text, rejected) if base_text else 0.0
            pair_similarity = textual_similarity(chosen, rejected) if chosen and rejected else 0.0
            if base_similarity < 0.70:
                reasons.append(f"base_rejected_similarity:{base_similarity:.5f}")
            if not 0.55 <= pair_similarity < 0.995:
                reasons.append(f"chosen_rejected_similarity:{pair_similarity:.5f}")
            if korean_signal(chosen) < 0.55 or korean_signal(rejected) < 0.55:
                reasons.append("insufficient_korean_signal")
            for label, answer in (("chosen", chosen), ("rejected", rejected)):
                if any(re.search(pattern, answer, re.IGNORECASE | re.DOTALL) for pattern in META_PATTERNS):
                    reasons.append(f"{label}_meta_leakage")
            evidence_text = json.dumps(prompt["issue_card"], ensure_ascii=False)
            supported_numbers = numeric_tokens(evidence_text)
            for label, answer in (("chosen", chosen), ("rejected", rejected)):
                unsupported = {
                    value for value in numeric_tokens(answer)
                    if value not in supported_numbers and not (value.isdigit() and int(value) <= 9)
                }
                if unsupported:
                    reasons.append(f"{label}_unsupported_numbers:" + ",".join(sorted(unsupported)))
            for key in ("base_factual_repairs_ko", "chosen_delta_ko", "evidence_used_ko"):
                if not parsed.get(key):
                    reasons.append(f"empty_{key}")
        else:
            chosen = rejected = defect = ""
            chosen_tokens = rejected_tokens = 0
            ratio = base_similarity = pair_similarity = 0.0
        pair = {
            "pair_id": "KSC2PAIR-" + stable_hash("cycle02-pair", prompt_id)[:20],
            "prompt_id": prompt_id,
            "issue_id": prompt["issue_id"],
            "domain_id": prompt["domain_id"],
            "split": prompt["split"],
            "prompt_form": prompt["prompt_form"],
            "question_ko": prompt["question_ko"],
            "issue_card": prompt["issue_card"],
            "source_ids": prompt["source_ids"],
            "base_response": base_text,
            "rejected_response": rejected,
            "chosen_response": chosen,
            "defect_type": defect,
            "base_factual_repairs_ko": parsed.get("base_factual_repairs_ko") if parsed else None,
            "chosen_delta_ko": parsed.get("chosen_delta_ko") if parsed else None,
            "evidence_used_ko": parsed.get("evidence_used_ko") if parsed else None,
            "lengths": {
                "base_characters": len(base_text),
                "rejected_characters": len(rejected),
                "chosen_characters": len(chosen),
                "rejected_tokens": rejected_tokens,
                "chosen_tokens": chosen_tokens,
                "chosen_rejected_token_ratio": round(ratio, 6),
            },
            "similarities": {
                "base_rejected": round(base_similarity, 6),
                "chosen_rejected": round(pair_similarity, 6),
            },
            "generation": {
                "base_model": base.get("model") if base else None,
                "base_messages_sha256": base.get("messages_sha256") if base else None,
                "editor_model": edit.get("model") if edit else None,
                "editor_messages_sha256": edit.get("messages_sha256") if edit else None,
            },
            "deterministic_errors": reasons,
        }
        if reasons:
            invalid.append(pair)
        else:
            accepted.append(pair)
    accepted.sort(key=lambda row: (row["split"], row["issue_id"], row["prompt_form"]))
    invalid.sort(key=lambda row: (row["split"], row["issue_id"], row["prompt_form"]))
    write_jsonl(args.output_dir / "deterministic_pass_pairs.jsonl", accepted)
    write_jsonl(args.output_dir / "deterministic_invalid_pairs.jsonl", invalid)
    report = {
        "schema_version": "ksova-onpolicy-edit-parse-1.0",
        "input_prompts": len(prompts),
        "deterministic_pass": len(accepted),
        "deterministic_invalid": len(invalid),
        "pass_rate": round(len(accepted) / len(prompts), 6) if prompts else 0,
        "similarity_metric": "normalized_character_5gram_multiset_dice",
        "pass_split_counts": dict(Counter(row["split"] for row in accepted)),
        "pass_issue_counts": {
            split: len({row["issue_id"] for row in accepted if row["split"] == split})
            for split in ("train", "dev")
        },
        "pass_domain_counts": dict(Counter(row["domain_id"] for row in accepted)),
        "pass_defect_counts": dict(Counter(row["defect_type"] for row in accepted)),
        "invalid_reason_counts": dict(Counter(
            reason.split(":", 1)[0]
            for row in invalid for reason in row["deterministic_errors"]
        )),
        "outputs": {
            "pass_sha256": sha256(args.output_dir / "deterministic_pass_pairs.jsonl"),
            "invalid_sha256": sha256(args.output_dir / "deterministic_invalid_pairs.jsonl"),
        },
    }
    write_json(args.output_dir / "edit_parse_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_validation(args: argparse.Namespace) -> int:
    pairs = read_jsonl(args.pairs)
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate pair IDs")
    requests: list[dict[str, Any]] = []
    for pair in pairs:
        payload = {
            "prompt_id": pair["prompt_id"],
            "assigned_defect_type": pair["defect_type"],
            "question_ko": pair["question_ko"],
            "validated_issue_card": pair["issue_card"],
            "base_response": pair["base_response"],
            "rejected_response": pair["rejected_response"],
            "chosen_response": pair["chosen_response"],
        }
        requests.append({
            "id": f"cycle02_pair_validate__{pair['prompt_id']}",
            "messages": [
                {"role": "system", "content": validation_system()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 1536,
            "metadata": {
                "purpose": "ksova_cycle02_onpolicy_pair_validation",
                "experiment_id": args.experiment_id,
                "prompt_id": pair["prompt_id"],
                "pair_id": pair["pair_id"],
                "issue_id": pair["issue_id"],
                "domain_id": pair["domain_id"],
                "split": pair["split"],
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-onpolicy-validation-requests-1.0",
        "experiment_id": args.experiment_id,
        "pairs": len(pairs),
        "pairs_sha256": sha256(args.pairs),
        "requests_sha256": sha256(args.output),
    }
    write_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_dev_evaluation(args: argparse.Namespace) -> int:
    prompts = [row for row in read_jsonl(args.prompts) if row.get("split") == "dev"]
    if len(prompts) != 60 or len({row["issue_id"] for row in prompts}) != 15:
        raise ValueError("dev evaluation must contain 60 prompts from 15 issues")
    base_responses = read_jsonl(args.base_responses)
    base_by_id = {str(row.get("id")): row for row in base_responses}
    if len(base_by_id) != len(base_responses):
        raise ValueError("duplicate base response IDs")
    requests: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for prompt in sorted(prompts, key=lambda row: str(row["prompt_id"])):
        prompt_id = str(prompt["prompt_id"])
        source = base_by_id.get(f"cycle02_base_answer__{prompt_id}")
        if source is None or source.get("error") or not str(source.get("response") or "").strip():
            raise ValueError(f"invalid base dev response: {prompt_id}")
        request_id = f"cycle02_dev_answer__{prompt_id}"
        metadata = {
            "purpose": "model_evaluation",
            "experiment_id": args.experiment_id,
            "prompt_id": prompt_id,
            "issue_id": prompt["issue_id"],
            "domain_id": prompt["domain_id"],
            "prompt_form": prompt["prompt_form"],
            "split": "dev",
        }
        messages = [{"role": "user", "content": str(prompt["question_ko"])}]
        requests.append({
            "id": request_id,
            "messages": messages,
            "max_tokens": 2048,
            "metadata": metadata,
        })
        base_rows.append({
            "id": request_id,
            "model": args.base_model_name,
            "response": str(source["response"]).strip(),
            "reasoning_content": source.get("reasoning_content") or "",
            "usage": source.get("usage"),
            "finish_reason": source.get("finish_reason"),
            "requested_max_tokens": source.get("requested_max_tokens"),
            "generation_config": source.get("generation_config") or {},
            "messages_sha256": hashlib.sha256(
                json.dumps(
                    messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "metadata": metadata,
            "error": None,
            "elapsed_seconds": source.get("elapsed_seconds"),
        })
    write_jsonl(args.requests_output, requests)
    write_jsonl(args.base_output, base_rows)
    report = {
        "schema_version": "ksova-onpolicy-dev-evaluation-1.0",
        "experiment_id": args.experiment_id,
        "prompts": len(prompts),
        "issues": len({row["issue_id"] for row in prompts}),
        "base_model": args.base_model_name,
        "requests_sha256": sha256(args.requests_output),
        "base_output_sha256": sha256(args.base_output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_validation(args: argparse.Namespace) -> int:
    pairs = read_jsonl(args.pairs)
    responses = read_jsonl(args.responses)
    response_by_id = {str(row.get("id")): row for row in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("duplicate validation response IDs")
    accepted: list[dict[str, Any]] = []
    reviewed: list[dict[str, Any]] = []
    for pair in pairs:
        response_id = f"cycle02_pair_validate__{pair['prompt_id']}"
        response = response_by_id.get(response_id)
        reasons: list[str] = []
        parsed: dict[str, Any] | None = None
        if response is None:
            reasons.append("missing_validation_response")
        elif response.get("error"):
            reasons.append("validation_error")
        elif str(response.get("finish_reason") or "").casefold() != "stop":
            reasons.append("validation_not_stopped")
        else:
            try:
                parsed = parse_json_object(str(response.get("response") or ""))
            except ValueError as exc:
                reasons.append(f"validation_json_parse:{exc}")
        checks: dict[str, bool] = {}
        if parsed is not None:
            if str(parsed.get("prompt_id")) != str(pair["prompt_id"]):
                reasons.append("validation_prompt_id_mismatch")
            checks = {key: parsed.get(key) is True for key in VALIDATION_KEYS}
            reasons.extend(f"failed_{key}" for key, value in checks.items() if not value)
            if str(parsed.get("verdict") or "").casefold() != "pass":
                reasons.append("verdict_not_pass")
        else:
            reasons.extend(f"failed_{key}" for key in VALIDATION_KEYS)
            reasons.append("verdict_not_pass")
        reviewed_row = {
            **pair,
            "independent_validation": {
                "model": response.get("model") if response else None,
                "messages_sha256": response.get("messages_sha256") if response else None,
                "checks": checks,
                "verdict": parsed.get("verdict") if parsed else None,
                "reasons_ko": parsed.get("reasons_ko") if parsed else None,
            },
            "validation_errors": reasons,
        }
        reviewed.append(reviewed_row)
        if not reasons:
            accepted.append(reviewed_row)
    accepted.sort(key=lambda row: (row["split"], row["issue_id"], row["prompt_form"]))
    reviewed.sort(key=lambda row: (row["split"], row["issue_id"], row["prompt_form"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "reviewed_pairs.jsonl", reviewed)
    write_jsonl(args.output_dir / "accepted_pairs.jsonl", accepted)
    training_files: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev"):
        rows = [row for row in accepted if row["split"] == split]
        training_files[f"dpo_{split}.jsonl"] = [{
            "prompt_messages": [{"role": "user", "content": row["question_ko"]}],
            "chosen": row["chosen_response"],
            "rejected": row["rejected_response"],
            "metadata": {
                "training_id": "KSC2-DPO-" + row["pair_id"],
                "pair_id": row["pair_id"],
                "prompt_id": row["prompt_id"],
                "issue_id": row["issue_id"],
                "domain_id": row["domain_id"],
                "prompt_form": row["prompt_form"],
                "defect_type": row["defect_type"],
                "split": split,
            },
        } for row in rows]
    for name, rows in training_files.items():
        write_jsonl(args.output_dir / name, rows)
    train_rows = training_files["dpo_train.jsonl"]
    train_domain_counts = Counter(
        row["metadata"]["domain_id"] for row in train_rows
    )
    if train_domain_counts:
        domain_count = len(train_domain_counts)
        total_train_rows = len(train_rows)
        for row in train_rows:
            domain_id = row["metadata"]["domain_id"]
            row["metadata"]["sample_weight"] = round(
                total_train_rows / (domain_count * train_domain_counts[domain_id]),
                8,
            )
        write_jsonl(args.output_dir / "dpo_train.jsonl", train_rows)
    train_issue_pair_counts = Counter(
        row["metadata"]["issue_id"] for row in train_rows
    )
    report = {
        "schema_version": "ksova-onpolicy-validation-parse-1.0",
        "input_pairs": len(pairs),
        "accepted_pairs": len(accepted),
        "acceptance_rate": round(len(accepted) / len(pairs), 6) if pairs else 0,
        "accepted_split_counts": dict(Counter(row["split"] for row in accepted)),
        "accepted_issue_counts": {
            split: len({row["issue_id"] for row in accepted if row["split"] == split})
            for split in ("train", "dev")
        },
        "accepted_domain_counts": dict(Counter(row["domain_id"] for row in accepted)),
        "train_domain_counts": dict(sorted(train_domain_counts.items())),
        "train_domain_loss_weight": {
            domain_id: round(len(train_rows) / (len(train_domain_counts) * count), 8)
            for domain_id, count in sorted(train_domain_counts.items())
        } if train_domain_counts else {},
        "accepted_defect_counts": dict(Counter(row["defect_type"] for row in accepted)),
        "train_issues_with_at_least_3_pairs": sum(
            count >= 3 for count in train_issue_pair_counts.values()
        ),
        "train_pair_count_min_mean_max": {
            "min": min(train_issue_pair_counts.values()) if train_issue_pair_counts else 0,
            "mean": (
                round(sum(train_issue_pair_counts.values()) / len(train_issue_pair_counts), 6)
                if train_issue_pair_counts else 0
            ),
            "max": max(train_issue_pair_counts.values()) if train_issue_pair_counts else 0,
        },
        "rejection_reason_counts": dict(Counter(
            reason.split(":", 1)[0]
            for row in reviewed for reason in row["validation_errors"]
        )),
        "promotion_gates": {
            "train_pairs_at_least_405": len(train_rows) >= 405,
            "train_issues_at_least_120": len(train_issue_pair_counts) >= 120,
            "train_issues_with_3_pairs_at_least_105": sum(
                count >= 3 for count in train_issue_pair_counts.values()
            ) >= 105,
        },
        "outputs": {
            "accepted_pairs_sha256": sha256(args.output_dir / "accepted_pairs.jsonl"),
            "reviewed_pairs_sha256": sha256(args.output_dir / "reviewed_pairs.jsonl"),
            "dpo_train_sha256": sha256(args.output_dir / "dpo_train.jsonl"),
            "dpo_dev_sha256": sha256(args.output_dir / "dpo_dev.jsonl"),
        },
    }
    write_json(args.output_dir / "validation_parse_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("prepare")
    command.add_argument("--issues", type=Path, required=True)
    command.add_argument("--sources", type=Path, required=True)
    command.add_argument("--benchmark-items", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--experiment-id", required=True)
    command.set_defaults(function=prepare)

    command = subparsers.add_parser("parse-prompts")
    command.add_argument("--selected-issues", type=Path, required=True)
    command.add_argument("--responses", type=Path, required=True)
    command.add_argument("--benchmark-items", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--experiment-id", required=True)
    command.set_defaults(function=parse_prompts)

    command = subparsers.add_parser("build-editor")
    command.add_argument("--prompts", type=Path, required=True)
    command.add_argument("--base-responses", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--experiment-id", required=True)
    command.set_defaults(function=build_editor)

    command = subparsers.add_parser("build-recovery-editor")
    command.add_argument("--original-requests", type=Path, required=True)
    command.add_argument("--invalid-pairs", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--experiment-id", required=True)
    command.add_argument("--primary-editor-model", required=True)
    command.add_argument("--recovery-editor-model", required=True)
    command.set_defaults(function=build_recovery_editor)

    command = subparsers.add_parser("merge-edit-responses")
    command.add_argument("--prompts", type=Path, required=True)
    command.add_argument("--primary-responses", type=Path, required=True)
    command.add_argument("--recovery-responses", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=merge_edit_responses)

    command = subparsers.add_parser("parse-edits")
    command.add_argument("--prompts", type=Path, required=True)
    command.add_argument("--base-responses", type=Path, required=True)
    command.add_argument("--edit-responses", type=Path, required=True)
    command.add_argument("--tokenizer-path", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(function=parse_edits)

    command = subparsers.add_parser("build-validation")
    command.add_argument("--pairs", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--experiment-id", required=True)
    command.set_defaults(function=build_validation)

    command = subparsers.add_parser("build-dev-evaluation")
    command.add_argument("--prompts", type=Path, required=True)
    command.add_argument("--base-responses", type=Path, required=True)
    command.add_argument("--requests-output", type=Path, required=True)
    command.add_argument("--base-output", type=Path, required=True)
    command.add_argument("--report", type=Path, required=True)
    command.add_argument("--experiment-id", required=True)
    command.add_argument("--base-model-name", required=True)
    command.set_defaults(function=build_dev_evaluation)

    command = subparsers.add_parser("parse-validation")
    command.add_argument("--pairs", type=Path, required=True)
    command.add_argument("--responses", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(function=parse_validation)

    args = parser.parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
