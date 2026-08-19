#!/usr/bin/env python3
"""Parse, validate, and materialize K-SOVA preference pairs.

The pipeline deliberately separates deterministic structural checks from an
independent model review.  It never writes to canonical data or the frozen
benchmark; all outputs are explicit experiment artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


REQUIRED_PAIR_KEYS = {
    "pair_id",
    "chosen_response",
    "rejected_response",
    "rejection_type",
    "rejection_defects_ko",
    "chosen_coverage_ko",
    "shared_factual_basis_ko",
}
VALIDATION_BOOLEAN_KEYS = (
    "chosen_complete",
    "chosen_grounded",
    "chosen_korean_baseline_clear",
    "chosen_counterargument_addressed",
    "chosen_conditions_present",
    "rejected_plausible",
    "assigned_defect_present",
    "rejected_materially_worse",
    "both_factually_safe",
    "length_not_only_advantage",
    "no_government_mimicry",
    "pair_useful_for_preference_learning",
)
SFT_VALIDATION_BOOLEAN_KEYS = (
    "chosen_complete",
    "chosen_grounded",
    "chosen_korean_baseline_clear",
    "chosen_counterargument_addressed",
    "chosen_conditions_present",
    "both_factually_safe",
    "no_government_mimicry",
)
SFT_MIN_CHOSEN_CHARACTERS = 600
META_LEAKAGE_PATTERNS = (
    r"(?:의도적으로|일부러).{0,18}(?:누락|제외|생략)",
    r"(?:이 답변|본 답변)은.{0,18}(?:오답|탈락)",
    r"(?:chosen|rejected)[ _-]?response",
    r"(?:선호|비선호)\s*답변",
    r"학습용\s*(?:오답|답변)",
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


def normalized(value: Any) -> str:
    return " ".join(str(value or "").split())


def parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()
    attempts = [value]
    first, last = value.find("{"), value.rfind("}")
    if first >= 0 and last > first:
        attempts.append(value[first : last + 1])
    error: Exception | None = None
    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise ValueError("response JSON is not an object")
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            error = exc
    raise ValueError(f"could not parse one JSON object: {error}")


def korean_signal(text: str) -> float:
    letters = re.findall(r"[A-Za-z가-힣]", text)
    if not letters:
        return 0.0
    return sum(bool(re.fullmatch(r"[가-힣]", char)) for char in letters) / len(letters)


def deterministic_pair_errors(pair: dict[str, Any], seed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_PAIR_KEYS - set(pair))
    if missing:
        return ["missing_keys:" + ",".join(missing)]
    if str(pair.get("pair_id")) != str(seed["pair_id"]):
        errors.append("pair_id_mismatch")
    if str(pair.get("rejection_type")) != str(seed["assigned_rejection_type"]):
        errors.append("rejection_type_mismatch")
    chosen = str(pair.get("chosen_response") or "").strip()
    rejected = str(pair.get("rejected_response") or "").strip()
    chosen_length, rejected_length = len(chosen), len(rejected)
    if not 700 <= chosen_length <= 2200:
        errors.append(f"chosen_character_length:{chosen_length}")
    if not 650 <= rejected_length <= 2200:
        errors.append(f"rejected_character_length:{rejected_length}")
    if rejected_length:
        ratio = chosen_length / rejected_length
        if not 0.75 <= ratio <= 1.33:
            errors.append(f"length_ratio:{ratio:.3f}")
    if normalized(chosen) == normalized(rejected):
        errors.append("identical_answers")
    if korean_signal(chosen) < 0.55 or korean_signal(rejected) < 0.55:
        errors.append("insufficient_korean_signal")
    for label, answer in (("chosen", chosen), ("rejected", rejected)):
        if any(re.search(pattern, answer, re.IGNORECASE | re.DOTALL) for pattern in META_LEAKAGE_PATTERNS):
            errors.append(f"{label}_meta_leakage")
    for key in ("rejection_defects_ko", "chosen_coverage_ko", "shared_factual_basis_ko"):
        if not normalized(pair.get(key)):
            errors.append(f"empty_{key}")
    return errors


def parse_generation(args: argparse.Namespace) -> int:
    seeds = read_jsonl(args.seeds)
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    seed_by_pair = {str(row["pair_id"]): row for row in seeds}
    request_by_id = {str(row["id"]): row for row in requests}
    response_by_id = {str(row.get("id")): row for row in responses}
    if len(seed_by_pair) != len(seeds) or len(request_by_id) != len(requests):
        raise ValueError("duplicate seed pair or request ID")
    provisional: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request["id"])
        metadata = request.get("metadata") or {}
        pair_id = str(metadata.get("pair_id") or "")
        seed = seed_by_pair.get(pair_id)
        response = response_by_id.get(request_id)
        reasons: list[str] = []
        parsed: dict[str, Any] | None = None
        if seed is None:
            reasons.append("missing_seed")
        if response is None:
            reasons.append("missing_response")
        elif response.get("error"):
            reasons.append("generation_error")
        elif str(response.get("finish_reason") or "").casefold() != "stop":
            reasons.append("generation_not_stopped")
        else:
            try:
                parsed = parse_json_object(str(response.get("response") or ""))
            except ValueError as exc:
                reasons.append("json_parse_error:" + str(exc))
        deterministic_errors: list[str] = []
        if parsed is not None and seed is not None:
            deterministic_errors = deterministic_pair_errors(parsed, seed)
            reasons.extend(deterministic_errors)
            if not (REQUIRED_PAIR_KEYS - set(parsed)):
                chosen = str(parsed["chosen_response"]).strip()
                rejected = str(parsed["rejected_response"]).strip()
                candidate = {
                    **seed,
                    "chosen_response": chosen,
                    "rejected_response": rejected,
                    "rejection_type": str(parsed["rejection_type"]),
                    "rejection_defects_ko": parsed["rejection_defects_ko"],
                    "chosen_coverage_ko": parsed["chosen_coverage_ko"],
                    "shared_factual_basis_ko": parsed["shared_factual_basis_ko"],
                    "deterministic_pair_errors": deterministic_errors,
                    "generation_provenance": {
                        "request_id": request_id,
                        "model": response.get("model"),
                        "messages_sha256": response.get("messages_sha256"),
                        "generation_config": response.get("generation_config"),
                        "finish_reason": response.get("finish_reason"),
                        "usage": response.get("usage"),
                    },
                    "lengths": {
                        "chosen_characters": len(chosen),
                        "rejected_characters": len(rejected),
                        "chosen_to_rejected_ratio": (
                            round(len(chosen) / len(rejected), 4) if rejected else None
                        ),
                    },
                }
                candidates.append(candidate)
        if reasons or parsed is None or seed is None:
            invalid.append({
                "request_id": request_id,
                "pair_id": pair_id,
                "reasons": reasons or ["unknown_parse_failure"],
                "parsed_pair": parsed,
                "raw_response": response,
            })
            continue
        provisional.append(candidates[-1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provisional_path = args.output_dir / "provisional_pairs.jsonl"
    candidates_path = args.output_dir / "parsed_candidates.jsonl"
    invalid_path = args.output_dir / "generation_invalid.jsonl"
    write_jsonl(provisional_path, provisional)
    write_jsonl(candidates_path, candidates)
    write_jsonl(invalid_path, invalid)
    report = {
        "schema_version": "ksova-preference-generation-parse-1.0",
        "requests": len(requests),
        "responses": len(responses),
        "parsed_candidates": len(candidates),
        "provisional_pairs": len(provisional),
        "invalid_pairs": len(invalid),
        "invalid_reason_counts": dict(Counter(
            reason.split(":", 1)[0]
            for row in invalid
            for reason in row["reasons"]
        )),
        "split_counts": dict(Counter(row["split"] for row in provisional)),
        "domain_counts": dict(Counter(row["domain_id"] for row in provisional)),
        "mean_chosen_characters": round(
            sum(len(row["chosen_response"]) for row in provisional) / len(provisional), 2
        ) if provisional else 0,
        "mean_rejected_characters": round(
            sum(len(row["rejected_response"]) for row in provisional) / len(provisional), 2
        ) if provisional else 0,
        "inputs": {
            "seeds_sha256": sha256(args.seeds),
            "requests_sha256": sha256(args.requests),
            "responses_sha256": sha256(args.responses),
        },
        "outputs": {
            "provisional_pairs_sha256": sha256(provisional_path),
            "parsed_candidates_sha256": sha256(candidates_path),
            "generation_invalid_sha256": sha256(invalid_path),
        },
    }
    write_json(args.output_dir / "generation_parse_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def repair_system_prompt() -> str:
    return """당신은 K-SOVA 선호학습 답변 쌍의 길이 편향을 교정하는 데이터 편집자다.

입력에는 이미 생성된 답변 쌍과 자동검사 실패 이유가 있다. 질문·검증된 쟁점 카드·기존 답변의 사실 범위 밖 정보를 추가하지 않는다. chosen_response의 완결된 판단 구조와 지정된 rejection_type을 유지한다. rejected_response는 사실상 대체로 안전하고 실제 모델이 낼 법해야 하지만, 지정 결함은 반드시 그대로 남아 있어 chosen보다 실질적으로 낮아야 한다.

두 답변을 각각 한국어 800~1,600자로 만들고, 두 답변의 글자 수 차이를 긴 답변의 15% 이내로 맞춘다. rejected를 짧게 만드는 대신 이미 포함된 논거를 충분히 설명해 길이를 맞추며, 누락된 지정 결함을 보충해서는 안 된다. 답변 안에서 편집 과정, 오답, 누락 의도, 학습 데이터라고 밝히지 않는다. 정부의 공식 대변인처럼 쓰거나 새로운 인명·기관·수치·날짜·법적 효과를 단정하지 않는다.

마크다운 코드블록 없이 다음 키의 JSON 객체 하나만 출력한다.
pair_id, chosen_response, rejected_response, rejection_type, rejection_defects_ko, chosen_coverage_ko, shared_factual_basis_ko"""


def build_repair(args: argparse.Namespace) -> int:
    invalid = read_jsonl(args.invalid)
    seeds = {str(row["pair_id"]): row for row in read_jsonl(args.seeds)}
    requests: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in invalid:
        reasons = [str(value) for value in row.get("reasons") or []]
        reason_families = {value.split(":", 1)[0] for value in reasons}
        parsed = row.get("parsed_pair")
        pair_id = str(row.get("pair_id") or "")
        seed = seeds.get(pair_id)
        repairable = bool(
            parsed
            and seed
            and reason_families
            and reason_families <= {
                "chosen_character_length", "rejected_character_length", "length_ratio"
            }
        )
        if not repairable:
            skipped.append({"pair_id": pair_id, "reasons": reasons})
            continue
        payload = {
            "pair_id": pair_id,
            "assigned_rejection_type": seed["assigned_rejection_type"],
            "automatic_failure_reasons": reasons,
            "question_ko": seed["question_ko"],
            "validated_issue_card": seed["issue_card"],
            "existing_pair": parsed,
        }
        requests.append({
            "id": f"method_pair_repair__{pair_id}",
            "messages": [
                {"role": "system", "content": repair_system_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 4096,
            "metadata": {
                "purpose": "ksova_method_search_preference_pair_single_repair",
                "experiment_id": args.experiment_id,
                "pair_id": pair_id,
                "issue_id": seed["issue_id"],
                "domain_id": seed["domain_id"],
                "split": seed["split"],
                "origin": seed["origin"],
                "task": seed["task"],
                "assigned_rejection_type": seed["assigned_rejection_type"],
                "repair_attempt": 1,
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-preference-single-repair-requests-1.0",
        "experiment_id": args.experiment_id,
        "invalid_pairs": len(invalid),
        "repair_requests": len(requests),
        "nonrepairable_pairs": len(skipped),
        "invalid_sha256": sha256(args.invalid),
        "seeds_sha256": sha256(args.seeds),
        "requests_sha256": sha256(args.output),
        "nonrepairable": skipped,
    }
    write_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def merge_pairs(args: argparse.Namespace) -> int:
    primary = read_jsonl(args.primary)
    repair = read_jsonl(args.repair)
    if args.replace_existing:
        by_pair_id = {str(row["pair_id"]): row for row in primary}
        by_pair_id.update({str(row["pair_id"]): row for row in repair})
        combined = list(by_pair_id.values())
    else:
        combined = primary + repair
    pair_ids = [str(row["pair_id"]) for row in combined]
    if len(pair_ids) != len(set(pair_ids)):
        duplicates = sorted(
            pair_id for pair_id, count in Counter(pair_ids).items() if count > 1
        )
        raise ValueError(f"duplicate pair IDs across primary and repair: {duplicates[:5]}")
    combined.sort(key=lambda row: (row["issue_id"], row["task"], row["pair_id"]))
    write_jsonl(args.output, combined)
    report = {
        "schema_version": "ksova-preference-pair-merge-1.0",
        "primary_pairs": len(primary),
        "repaired_pairs": len(repair),
        "repair_replaced_existing": bool(args.replace_existing),
        "merged_pairs": len(combined),
        "issues": len({row["issue_id"] for row in combined}),
        "split_counts": dict(Counter(row["split"] for row in combined)),
        "domain_counts": dict(Counter(row["domain_id"] for row in combined)),
        "primary_sha256": sha256(args.primary),
        "repair_sha256": sha256(args.repair),
        "output_sha256": sha256(args.output),
    }
    write_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def validation_system_prompt() -> str:
    return """당신은 K-SOVA 선호학습 데이터의 독립 검수자다. 답변을 고쳐 쓰지 말고, 제공된 질문·검증된 쟁점 카드·답변 두 개만으로 엄격하게 판정한다.

chosen은 사실과 주장을 구분하고, 한국 사회에서 방어 가능한 최소 기준선과 구체적 권리·이해관계를 밝히며, 강한 상대 논거를 검토하고, 명확한 판단과 비용·위험·조건·불확실성의 경계를 함께 제시해야 한다. rejected는 터무니없는 답이 아니라 실제 모델이 낼 법하고 사실상 대체로 안전하지만, 지정된 한 가지 결함 때문에 실질적으로 더 나빠야 한다. 길이 차이만으로 우열이 생기거나 정부의 공식 대변인처럼 단정하면 안 된다.

쟁점 카드 밖의 구체적인 인명·기관·수치·날짜·법적 효과를 새로 단정하면 사실 안전성을 통과시키지 않는다. 모든 불리언은 엄격하게 true 또는 false로 출력한다. 하나라도 핵심 요건이 불확실하면 false다. JSON 객체 하나만 출력하며 키는 다음과 같다.
pair_id, chosen_complete, chosen_grounded, chosen_korean_baseline_clear, chosen_counterargument_addressed, chosen_conditions_present, rejected_plausible, assigned_defect_present, rejected_materially_worse, both_factually_safe, length_not_only_advantage, no_government_mimicry, pair_useful_for_preference_learning, verdict, reasons_ko

모든 불리언이 true일 때만 verdict를 pass로, 아니면 reject로 한다."""


def build_validation(args: argparse.Namespace) -> int:
    pairs = read_jsonl(args.pairs)
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate pair IDs")
    requests: list[dict[str, Any]] = []
    for row in pairs:
        pair_id = str(row["pair_id"])
        payload = {
            "pair_id": pair_id,
            "assigned_rejection_type": row["rejection_type"],
            "question_ko": row["question_ko"],
            "validated_issue_card": row["issue_card"],
            "chosen_response": row["chosen_response"],
            "rejected_response": row["rejected_response"],
        }
        requests.append({
            "id": f"method_pair_validate__{pair_id}",
            "messages": [
                {"role": "system", "content": validation_system_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 1024,
            "metadata": {
                "purpose": "ksova_method_search_preference_pair_validation",
                "experiment_id": args.experiment_id,
                "pair_id": pair_id,
                "issue_id": row["issue_id"],
                "domain_id": row["domain_id"],
                "split": row["split"],
            },
        })
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-preference-validation-requests-1.0",
        "experiment_id": args.experiment_id,
        "pairs": len(pairs),
        "pairs_sha256": sha256(args.pairs),
        "requests_sha256": sha256(args.output),
    }
    write_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_validation(args: argparse.Namespace) -> int:
    pairs = read_jsonl(args.pairs)
    responses = read_jsonl(args.responses)
    response_by_id = {str(row.get("id")): row for row in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("duplicate validation response IDs")
    accepted_pairs: list[dict[str, Any]] = []
    accepted_sft: list[dict[str, Any]] = []
    reviewed: list[dict[str, Any]] = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        response_id = f"method_pair_validate__{pair_id}"
        response = response_by_id.get(response_id)
        structural_reasons: list[str] = []
        parsed: dict[str, Any] | None = None
        if response is None:
            structural_reasons.append("missing_validation_response")
        elif response.get("error"):
            structural_reasons.append("validation_error")
        elif str(response.get("finish_reason") or "").casefold() != "stop":
            structural_reasons.append("validation_not_stopped")
        else:
            try:
                parsed = parse_json_object(str(response.get("response") or ""))
            except ValueError as exc:
                structural_reasons.append("validation_json_parse_error:" + str(exc))
        checks: dict[str, bool] = {}
        if parsed is not None:
            if str(parsed.get("pair_id")) != pair_id:
                structural_reasons.append("validation_pair_id_mismatch")
            for key in VALIDATION_BOOLEAN_KEYS:
                checks[key] = parsed.get(key) is True
        review = {
            "model": response.get("model") if response else None,
            "messages_sha256": response.get("messages_sha256") if response else None,
            "checks": checks,
            "verdict": parsed.get("verdict") if parsed else None,
            "reasons_ko": parsed.get("reasons_ko") if parsed else None,
        }
        output_row = {**pair, "independent_validation": review}
        deterministic_errors = [
            str(value) for value in pair.get("deterministic_pair_errors") or []
        ]
        pair_reasons = list(structural_reasons) + deterministic_errors
        pair_reasons.extend(
            "failed_" + key for key in VALIDATION_BOOLEAN_KEYS if not checks.get(key, False)
        )
        if parsed is None or str(parsed.get("verdict") or "").casefold() != "pass":
            pair_reasons.append("verdict_not_pass")

        chosen_error_families = {
            "missing_keys",
            "pair_id_mismatch",
            "rejection_type_mismatch",
            "chosen_character_length",
            "chosen_meta_leakage",
            "insufficient_korean_signal",
        }
        sft_reasons = list(structural_reasons)
        for value in deterministic_errors:
            family = value.split(":", 1)[0]
            if family not in chosen_error_families:
                continue
            if (
                family == "chosen_character_length"
                and int((pair.get("lengths") or {}).get("chosen_characters") or 0)
                >= SFT_MIN_CHOSEN_CHARACTERS
            ):
                continue
            sft_reasons.append(value)
        sft_reasons.extend(
            "failed_" + key
            for key in SFT_VALIDATION_BOOLEAN_KEYS
            if not checks.get(key, False)
        )
        reviewed_row = {
            **output_row,
            "pair_acceptance_reasons": pair_reasons,
            "sft_acceptance_reasons": sft_reasons,
        }
        reviewed.append(reviewed_row)
        if not pair_reasons:
            accepted_pairs.append(reviewed_row)
        if not sft_reasons:
            accepted_sft.append(reviewed_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.output_dir / "accepted_pairs.jsonl"
    accepted_sft_path = args.output_dir / "accepted_sft_candidates.jsonl"
    reviewed_path = args.output_dir / "reviewed_candidates.jsonl"
    write_jsonl(accepted_path, accepted_pairs)
    write_jsonl(accepted_sft_path, accepted_sft)
    write_jsonl(reviewed_path, reviewed)
    training_files: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev"):
        sft_split_rows = [row for row in accepted_sft if row["split"] == split]
        dpo_split_rows = [row for row in accepted_pairs if row["split"] == split]
        sft_rows = [{
            "messages": [
                {"role": "user", "content": row["question_ko"]},
                {"role": "assistant", "content": row["chosen_response"]},
            ],
            "metadata": {
                "training_id": "KSMS-SFT-" + row["pair_id"],
                "pair_id": row["pair_id"],
                "issue_id": row["issue_id"],
                "domain_id": row["domain_id"],
                "origin": row["origin"],
                "task": row["task"],
                "split": split,
            },
        } for row in sft_split_rows]
        dpo_rows = [{
            "prompt_messages": [{"role": "user", "content": row["question_ko"]}],
            "chosen": row["chosen_response"],
            "rejected": row["rejected_response"],
            "metadata": {
                "training_id": "KSMS-DPO-" + row["pair_id"],
                "pair_id": row["pair_id"],
                "issue_id": row["issue_id"],
                "domain_id": row["domain_id"],
                "origin": row["origin"],
                "task": row["task"],
                "rejection_type": row["rejection_type"],
                "split": split,
            },
        } for row in dpo_split_rows]
        training_files[f"sft_{split}.jsonl"] = sft_rows
        training_files[f"dpo_{split}.jsonl"] = dpo_rows
    for name, rows in training_files.items():
        write_jsonl(args.output_dir / name, rows)
    pair_issue_counts = {
        split: len({row["issue_id"] for row in accepted_pairs if row["split"] == split})
        for split in ("train", "dev")
    }
    sft_issue_counts = {
        split: len({row["issue_id"] for row in accepted_sft if row["split"] == split})
        for split in ("train", "dev")
    }
    report = {
        "schema_version": "ksova-preference-validation-parse-1.0",
        "input_candidates": len(pairs),
        "validation_responses": len(responses),
        "accepted_preference_pairs": len(accepted_pairs),
        "accepted_sft_candidates": len(accepted_sft),
        "sft_minimum_chosen_characters": SFT_MIN_CHOSEN_CHARACTERS,
        "preference_pair_acceptance_rate": (
            round(len(accepted_pairs) / len(pairs), 6) if pairs else 0
        ),
        "sft_candidate_acceptance_rate": (
            round(len(accepted_sft) / len(pairs), 6) if pairs else 0
        ),
        "accepted_pair_split_counts": dict(Counter(row["split"] for row in accepted_pairs)),
        "accepted_sft_split_counts": dict(Counter(row["split"] for row in accepted_sft)),
        "accepted_pair_issue_counts": pair_issue_counts,
        "accepted_sft_issue_counts": sft_issue_counts,
        "accepted_pair_domain_counts": dict(Counter(
            row["domain_id"] for row in accepted_pairs
        )),
        "accepted_sft_domain_counts": dict(Counter(
            row["domain_id"] for row in accepted_sft
        )),
        "pair_rejection_reason_counts": dict(Counter(
            reason.split(":", 1)[0]
            for row in reviewed
            for reason in row["pair_acceptance_reasons"]
        )),
        "sft_rejection_reason_counts": dict(Counter(
            reason.split(":", 1)[0]
            for row in reviewed
            for reason in row["sft_acceptance_reasons"]
        )),
        "inputs": {
            "pairs_sha256": sha256(args.pairs),
            "responses_sha256": sha256(args.responses),
        },
        "outputs": {
            "accepted_pairs_sha256": sha256(accepted_path),
            "accepted_sft_candidates_sha256": sha256(accepted_sft_path),
            "reviewed_candidates_sha256": sha256(reviewed_path),
            **{
                name.replace(".jsonl", "_sha256"): sha256(args.output_dir / name)
                for name in training_files
            },
        },
    }
    write_json(args.output_dir / "validation_parse_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generation = subparsers.add_parser("parse-generation")
    generation.add_argument("--seeds", type=Path, required=True)
    generation.add_argument("--requests", type=Path, required=True)
    generation.add_argument("--responses", type=Path, required=True)
    generation.add_argument("--output-dir", type=Path, required=True)
    generation.set_defaults(function=parse_generation)

    validation = subparsers.add_parser("build-validation")
    validation.add_argument("--pairs", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    validation.add_argument("--experiment-id", required=True)
    validation.set_defaults(function=build_validation)

    repair = subparsers.add_parser("build-repair")
    repair.add_argument("--invalid", type=Path, required=True)
    repair.add_argument("--seeds", type=Path, required=True)
    repair.add_argument("--output", type=Path, required=True)
    repair.add_argument("--experiment-id", required=True)
    repair.set_defaults(function=build_repair)

    merge = subparsers.add_parser("merge-pairs")
    merge.add_argument("--primary", type=Path, required=True)
    merge.add_argument("--repair", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--replace-existing", action="store_true")
    merge.set_defaults(function=merge_pairs)

    materialize = subparsers.add_parser("parse-validation")
    materialize.add_argument("--pairs", type=Path, required=True)
    materialize.add_argument("--responses", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.set_defaults(function=parse_validation)

    args = parser.parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
