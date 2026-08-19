#!/usr/bin/env python3
"""Prepare and materialize cycle-05 rejection-sampled K-SOVA training data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable


EXPERIMENT_ID = "method_search_20260816_cycle05_rejection_sft"
EXPECTED_HASHES = {
    "issues": "97dfdb057d0967fe267ac92a409dc85dc09e129299e223a33baad78d84e24e13",
    "sources": "d77c9a5edd7035461f1a88615daecc4829688672e47aed0eb3c06ff34ec671bd",
    "train_examples": "164e419f67ace0febce3a9d60b7b3666388908534b0964d7c86a38e2bd62c01c",
    "benchmark": "0ca49e7bc94178300c0309919d63a51c0496d84d6e352501d350f0deac494032",
    "selected_issues": "c46a404421795e6f52fab378075e4ac7fa251b2de2b9dc547e5597de65ca410b",
    "prompts": "15489e2c8d5fd4dc2daa60c5113b51eb56c8b71ca0cb8ff88003876540148358",
    "evidence_packets": "5cb1514b62f2b87c9bd011699f757518c955e6a04e256dc62fc8f4ece9b0f0d1",
}
MIN_CHARS = 350
MAX_CHARS = 2400


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
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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


def cleaned(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def has_unexplained_foreign_script(text: str) -> bool:
    without_hanja_glosses = re.sub(
        r"\([^()]{0,30}[一-龥][^()]{0,30}\)", "", text
    )
    return bool(re.search(r"[一-龥ぁ-ゟァ-ヿ]", without_hanja_glosses))


def assert_hash(path: Path, key: str) -> None:
    actual = sha256(path)
    if actual != EXPECTED_HASHES[key]:
        raise ValueError(f"{key} hash mismatch: {actual}")


def fact_first_system() -> str:
    return """당신은 K-SOVA 학습용 완성답변 작성자다.

K-SOVA는 외국과 권리·규범·이해관계가 충돌할 수 있는 사안에서 한국 사회에 형성된 담론을 바탕으로 사실, 한국의 구체적 권리와 이해관계, 강한 상대 논거, 판단, 조건과 비용을 함께 구성하는 능력을 다룬다.

입력의 evidence_packet이 사용할 수 있는 사실의 완전한 경계다. 답변을 쓰기 전에 내부적으로 질문에 꼭 필요한 사실만 고르고, verified_facts_ko 또는 validated_reference_answers_ko가 직접 뒷받침하지 않는 인명·기관·수치·날짜·법적 효과는 쓰지 않는다. documented_positions_ko는 해당 행위자의 주장으로만 표시한다. 근거가 부족한 세부사항은 일반화해 메우지 말고 판단의 조건이나 불확실성으로 남긴다.

첫 문단에서 질문의 핵심 판단에 직접 답하고, 이후 필요한 사실과 주장 구분, 한국의 구체적 권리·책임·비용·선택권, 가장 강한 상대 논거, 결론이 성립하는 조건과 비용을 연결한다. 정부의 공식 대변인처럼 말하거나 국수주의적으로 과장하지 않는다. 카드나 채점 기준을 나열하지 말고 실제 사용자에게 줄 자연스러운 독립 최종 답변을 쓴다.

정확성을 위해 불필요한 고유명사와 사례 확장을 피한다. 한국어 600~1100자를 목표로 하고 반드시 350~2400자 안에 둔다. 답변 본문만 출력한다."""


def repair_system() -> str:
    return """당신은 K-SOVA 학습 후보를 정확히 한 번 교정하는 편집자다.

evidence_packet 밖의 지식은 사용하지 않는다. best_previous_answer의 장점은 유지하되 judge_diagnosis에 적힌 사실 오류·누락·회피를 모두 고친다. 질문에 직접 답하고, 한국의 구체적 권리·책임·이해관계, 강한 상대 논거, 명확한 판단, 조건·비용·불확실성을 자연스러운 최종 답변으로 통합한다. 근거가 없는 인명·기관·수치·날짜·법적 효과를 추가하지 않고, 행위자의 주장은 주장으로 구분한다. 정부 대변인처럼 말하거나 국수주의적으로 과장하지 않는다.

한국어 600~1100자를 목표로 하고 반드시 350~2400자 안에 둔다. 답변 본문만 출력한다."""


def validate_fixed_inputs(args: argparse.Namespace) -> None:
    for key in (
        "issues", "sources", "train_examples", "benchmark",
        "selected_issues", "prompts", "evidence_packets",
    ):
        assert_hash(getattr(args, key), key)


def prepare(args: argparse.Namespace) -> int:
    validate_fixed_inputs(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    prompts = read_jsonl(args.prompts)
    packets = read_jsonl(args.evidence_packets)
    if len(prompts) != 600 or len(packets) != 600:
        raise ValueError("expected 600 prompts and evidence packets")
    prompt_by_id = {str(row["prompt_id"]): row for row in prompts}
    packet_by_id = {str(row["prompt_id"]): row for row in packets}
    if len(prompt_by_id) != 600 or set(prompt_by_id) != set(packet_by_id):
        raise ValueError("prompt/evidence coverage mismatch")

    teacher_requests: list[dict[str, Any]] = []
    base_requests: list[dict[str, Any]] = []
    for prompt_id in sorted(prompt_by_id):
        prompt = prompt_by_id[prompt_id]
        packet = packet_by_id[prompt_id]
        shared_metadata = {
            "experiment_id": EXPERIMENT_ID,
            "prompt_id": prompt_id,
            "issue_id": str(prompt["issue_id"]),
            "domain_id": str(prompt["domain_id"]),
            "split": str(prompt["split"]),
            "prompt_form": str(prompt["prompt_form"]),
            "evidence_packet_sha256": str(packet["evidence_packet_sha256"]),
        }
        teacher_payload = {
            "prompt_id": prompt_id,
            "question_ko": str(prompt["question_ko"]),
            "evidence_packet": packet["evidence_packet"],
        }
        teacher_requests.append({
            "id": f"cycle05_fact_first__{prompt_id}",
            "messages": [
                {"role": "system", "content": fact_first_system()},
                {
                    "role": "user",
                    "content": json.dumps(teacher_payload, ensure_ascii=False),
                },
            ],
            "max_tokens": 2048,
            "metadata": {**shared_metadata, "purpose": "cycle05_fact_first_teacher"},
        })
        base_requests.append({
            "id": f"cycle05_actual_base__{prompt_id}",
            "messages": [{"role": "user", "content": str(prompt["question_ko"])}],
            "max_tokens": 2048,
            "metadata": {**shared_metadata, "purpose": "cycle05_actual_base"},
        })

    teacher_path = args.output_dir / "teacher_requests.jsonl"
    base_path = args.output_dir / "base_requests.jsonl"
    atomic_jsonl(teacher_path, teacher_requests)
    atomic_jsonl(base_path, base_requests)
    report = {
        "schema_version": "ksova-cycle05-preparation-1.0",
        "prompts": 600,
        "issues": len({row["issue_id"] for row in prompts}),
        "splits": dict(Counter(str(row["split"]) for row in prompts)),
        "teacher_requests_sha256": sha256(teacher_path),
        "base_requests_sha256": sha256(base_path),
        "fixed_input_hashes": {
            key: sha256(getattr(args, key)) for key in EXPECTED_HASHES
        },
    }
    atomic_json(args.output_dir / "preparation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def outputs_by_prompt(path: Path, expected: set[str]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {
        str((row.get("metadata") or {}).get("prompt_id")): row for row in rows
    }
    if len(rows) != len(expected) or set(result) != expected:
        raise ValueError(f"output coverage mismatch: {path}")
    return result


def structural_state(answer: str, row: dict[str, Any] | None) -> tuple[bool, list[str]]:
    violations: list[str] = []
    if row is not None and row.get("error"):
        violations.append("runtime_error")
    if not answer:
        violations.append("empty_answer")
    if row is not None and row.get("finish_reason") != "stop":
        violations.append("finish_reason_not_stop")
    if len(answer) < MIN_CHARS:
        violations.append("below_minimum_characters")
    if len(answer) > MAX_CHARS:
        violations.append("above_maximum_characters")
    if has_unexplained_foreign_script(answer):
        violations.append("unexplained_foreign_script")
    return not violations, violations


def aggregate_candidates(args: argparse.Namespace) -> int:
    prompts = read_jsonl(args.prompts)
    prompt_by_id = {str(row["prompt_id"]): row for row in prompts}
    expected = set(prompt_by_id)
    if len(expected) != 600:
        raise ValueError("expected 600 unique prompts")
    qwen_fact = outputs_by_prompt(args.qwen_fact_outputs, expected)
    gemma = outputs_by_prompt(args.gemma_outputs, expected)
    base = outputs_by_prompt(args.base_outputs, expected)
    cycle04_raw = outputs_by_prompt(args.cycle04_raw_outputs, expected)
    cycle04_sft_rows = read_jsonl(args.cycle04_sft_train) + read_jsonl(args.cycle04_sft_dev)
    cycle04_selected = {
        str((row.get("metadata") or {}).get("prompt_id")): row
        for row in cycle04_sft_rows
    }
    if len(cycle04_selected) != len(cycle04_sft_rows):
        raise ValueError("duplicate cycle04 SFT prompt IDs")

    rows: list[dict[str, Any]] = []
    structural_counts: Counter[str] = Counter()
    for prompt_id in sorted(expected):
        prompt = prompt_by_id[prompt_id]
        candidates: list[dict[str, Any]] = []
        selected = cycle04_selected.get(prompt_id)
        if selected is not None:
            cycle04_answer = cleaned(selected["messages"][1]["content"])
            cycle04_row = None
            cycle04_origin = "cycle04_validated_or_repaired"
        else:
            cycle04_row = cycle04_raw[prompt_id]
            cycle04_answer = cleaned(cycle04_row.get("response"))
            cycle04_origin = "cycle04_raw_fallback"
        sources = (
            ("cycle04_qwen36", cycle04_answer, cycle04_row, cycle04_origin),
            ("qwen36_fact_first", cleaned(qwen_fact[prompt_id].get("response")), qwen_fact[prompt_id], "new_generation"),
            ("gemma4_31b", cleaned(gemma[prompt_id].get("response")), gemma[prompt_id], "new_generation"),
            ("qwen35_9b_base", cleaned(base[prompt_id].get("response")), base[prompt_id], "actual_base_generation"),
        )
        for source, answer, runtime_row, origin in sources:
            eligible, violations = structural_state(answer, runtime_row)
            structural_counts[f"{source}:eligible" if eligible else f"{source}:rejected"] += 1
            for violation in violations:
                structural_counts[f"{source}:{violation}"] += 1
            candidates.append({
                "candidate_id": f"{prompt_id}::{source}",
                "source": source,
                "origin": origin,
                "answer_ko": answer,
                "structural_eligible": eligible,
                "structural_violations": violations,
            })
        rows.append({
            "schema_version": "ksova-cycle05-candidate-set-1.0",
            "prompt_id": prompt_id,
            "issue_id": str(prompt["issue_id"]),
            "domain_id": str(prompt["domain_id"]),
            "split": str(prompt["split"]),
            "prompt_form": str(prompt["prompt_form"]),
            "question_ko": str(prompt["question_ko"]),
            "candidates": candidates,
        })
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    output = args.output_dir / "candidate_sets.jsonl"
    atomic_jsonl(output, rows)
    report = {
        "schema_version": "ksova-cycle05-candidate-aggregation-1.0",
        "prompts": len(rows),
        "candidates": sum(len(row["candidates"]) for row in rows),
        "structural_counts": dict(sorted(structural_counts.items())),
        "candidate_sets_sha256": sha256(output),
        "sources": {
            "qwen_fact": sha256(args.qwen_fact_outputs),
            "gemma": sha256(args.gemma_outputs),
            "base": sha256(args.base_outputs),
            "cycle04_raw": sha256(args.cycle04_raw_outputs),
            "cycle04_sft_train": sha256(args.cycle04_sft_train),
            "cycle04_sft_dev": sha256(args.cycle04_sft_dev),
        },
    }
    atomic_json(args.output_dir / "aggregation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_repair(args: argparse.Namespace) -> int:
    packets = {str(row["prompt_id"]): row for row in read_jsonl(args.evidence_packets)}
    candidate_sets = {
        str(row["prompt_id"]): row for row in read_jsonl(args.candidate_sets)
    }
    selections = read_jsonl(args.selection_rows)
    requests: list[dict[str, Any]] = []
    for selection in selections:
        if not selection.get("scorable", True):
            continue
        if selection.get("selected_source") is not None:
            continue
        prompt_id = str(selection["prompt_id"])
        candidate_set = candidate_sets[prompt_id]
        best_source = str(selection["best_source"])
        best = next(
            row for row in candidate_set["candidates"] if row["source"] == best_source
        )
        best_judgment = next(
            row for row in selection["candidate_judgments"]
            if row["source"] == best_source
        )
        payload = {
            "prompt_id": prompt_id,
            "question_ko": candidate_set["question_ko"],
            "evidence_packet": packets[prompt_id]["evidence_packet"],
            "best_previous_answer": best["answer_ko"],
            "judge_diagnosis": best_judgment,
        }
        requests.append({
            "id": f"cycle05_repair__{prompt_id}",
            "messages": [
                {"role": "system", "content": repair_system()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": 2048,
            "metadata": {
                "purpose": "cycle05_single_repair",
                "experiment_id": EXPERIMENT_ID,
                "prompt_id": prompt_id,
                "issue_id": candidate_set["issue_id"],
                "domain_id": candidate_set["domain_id"],
                "split": candidate_set["split"],
                "prompt_form": candidate_set["prompt_form"],
                "repair_source": best_source,
            },
        })
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    output = args.output_dir / "repair_requests.jsonl"
    atomic_jsonl(output, requests)
    report = {
        "schema_version": "ksova-cycle05-repair-build-1.0",
        "requests": len(requests),
        "request_sha256": sha256(output),
    }
    atomic_json(args.output_dir / "repair_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def aggregate_repair(args: argparse.Namespace) -> int:
    original_sets = {
        str(row["prompt_id"]): row for row in read_jsonl(args.candidate_sets)
    }
    selections = {
        str(row["prompt_id"]): row for row in read_jsonl(args.selection_rows)
    }
    responses = read_jsonl(args.repair_outputs)
    output_rows: list[dict[str, Any]] = []
    for response in responses:
        prompt_id = str((response.get("metadata") or {}).get("prompt_id"))
        if prompt_id not in selections or selections[prompt_id].get("selected_source") is not None:
            raise ValueError(f"unexpected repair response: {prompt_id}")
        original = original_sets[prompt_id]
        best_source = str(selections[prompt_id]["best_source"])
        best = next(row for row in original["candidates"] if row["source"] == best_source)
        repair_answer = cleaned(response.get("response"))
        eligible, violations = structural_state(repair_answer, response)
        output_rows.append({
            **{key: original[key] for key in (
                "schema_version", "prompt_id", "issue_id", "domain_id",
                "split", "prompt_form", "question_ko",
            )},
            "candidates": [
                {
                    "candidate_id": f"{prompt_id}::qwen36_repair",
                    "source": "qwen36_repair",
                    "origin": "single_repair",
                    "answer_ko": repair_answer,
                    "structural_eligible": eligible,
                    "structural_violations": violations,
                },
                best,
            ],
        })
    if len(output_rows) != len(responses):
        raise ValueError("repair aggregation count mismatch")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    output = args.output_dir / "repair_candidate_sets.jsonl"
    atomic_jsonl(output, output_rows)
    report = {
        "schema_version": "ksova-cycle05-repair-aggregation-1.0",
        "prompts": len(output_rows),
        "structurally_eligible_repairs": sum(
            row["candidates"][0]["structural_eligible"] for row in output_rows
        ),
        "candidate_sets_sha256": sha256(output),
    }
    atomic_json(args.output_dir / "aggregation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def stable_training_id(prompt_id: str, suffix: str) -> str:
    return "KSC5" + hashlib.sha256(f"{prompt_id}::{suffix}".encode()).hexdigest()[:20]


def materialize(args: argparse.Namespace) -> int:
    prompts = {str(row["prompt_id"]): row for row in read_jsonl(args.prompts)}
    candidate_sets = {
        str(row["prompt_id"]): row for row in read_jsonl(args.candidate_sets)
    }
    primary = {str(row["prompt_id"]): row for row in read_jsonl(args.selection_rows)}
    repair = (
        {str(row["prompt_id"]): row for row in read_jsonl(args.repair_selection_rows)}
        if args.repair_selection_rows and args.repair_selection_rows.is_file()
        else {}
    )
    if set(primary) != set(prompts) or set(candidate_sets) != set(prompts):
        raise ValueError("primary selection coverage mismatch")

    chosen_rows: list[dict[str, Any]] = []
    correction_pairs: list[dict[str, Any]] = []
    origins: Counter[str] = Counter()
    for prompt_id in sorted(prompts):
        prompt = prompts[prompt_id]
        selected = primary[prompt_id]
        chosen_source = selected.get("selected_source")
        chosen_answer = selected.get("selected_answer_ko")
        selection_attempt = "primary"
        if chosen_source is None and prompt_id in repair:
            repair_judgment = next(
                (
                    row for row in repair[prompt_id]["candidate_judgments"]
                    if row["source"] == "qwen36_repair"
                ),
                None,
            )
            if repair_judgment and repair_judgment.get("effective_strict"):
                chosen_source = "qwen36_repair"
                chosen_answer = repair_judgment["answer_ko"]
                selection_attempt = "repair1"
        if chosen_source is None or not normalized(chosen_answer):
            continue
        origins[str(chosen_source)] += 1
        metadata = {
            "training_id": stable_training_id(prompt_id, "sft"),
            "experiment_id": EXPERIMENT_ID,
            "prompt_id": prompt_id,
            "issue_id": str(prompt["issue_id"]),
            "domain_id": str(prompt["domain_id"]),
            "split": str(prompt["split"]),
            "prompt_form": str(prompt["prompt_form"]),
            "answer_origin": str(chosen_source),
            "selection_attempt": selection_attempt,
        }
        chosen_rows.append({
            "schema_version": "ksova-cycle05-sft-1.0",
            "messages": [
                {"role": "user", "content": str(prompt["question_ko"])},
                {"role": "assistant", "content": cleaned(chosen_answer)},
            ],
            "metadata": metadata,
        })

        base_candidate = next(
            row for row in candidate_sets[prompt_id]["candidates"]
            if row["source"] == "qwen35_9b_base"
        )
        base_judgment = next(
            row for row in selected["candidate_judgments"]
            if row["source"] == "qwen35_9b_base"
        )
        base_generation_complete = not any(
            violation in {
                "runtime_error", "empty_answer", "finish_reason_not_stop"
            }
            for violation in base_candidate.get("structural_violations", [])
        )
        if (
            not base_judgment.get("effective_strict")
            and base_generation_complete
            and normalized(base_candidate["answer_ko"])
            and normalized(base_candidate["answer_ko"]) != normalized(chosen_answer)
        ):
            correction_pairs.append({
                "schema_version": "ksova-cycle05-actual-base-pair-1.0",
                "prompt_messages": [
                    {"role": "user", "content": str(prompt["question_ko"])}
                ],
                "chosen": cleaned(chosen_answer),
                "rejected": cleaned(base_candidate["answer_ko"]),
                "metadata": {
                    **metadata,
                    "training_id": stable_training_id(prompt_id, "dpo"),
                    "pair_id": prompt_id,
                    "chosen_source": str(chosen_source),
                    "rejected_source": "qwen35_9b_actual_base",
                    "sample_weight": 1.0,
                },
            })

    train_rows = [row for row in chosen_rows if row["metadata"]["split"] == "train"]
    dev_rows = [row for row in chosen_rows if row["metadata"]["split"] == "dev"]
    domain_counts = Counter(str(row["metadata"]["domain_id"]) for row in train_rows)
    domain_count = len(domain_counts)
    for row in train_rows:
        domain = str(row["metadata"]["domain_id"])
        row["metadata"]["sample_weight"] = round(
            len(train_rows) / (domain_count * domain_counts[domain]), 8
        )
    train_issues = {row["metadata"]["issue_id"] for row in train_rows}
    dev_issues = {row["metadata"]["issue_id"] for row in dev_rows}
    pair_train = [
        row for row in correction_pairs if row["metadata"]["split"] == "train"
    ]
    pair_dev = [
        row for row in correction_pairs if row["metadata"]["split"] == "dev"
    ]
    pair_train_issues = {row["metadata"]["issue_id"] for row in pair_train}
    pair_dev_issues = {row["metadata"]["issue_id"] for row in pair_dev}
    gates = {
        "minimum_train_rows": len(train_rows) >= 300,
        "minimum_train_issues": len(train_issues) >= 100,
        "minimum_dev_rows": len(dev_rows) >= 30,
        "minimum_dev_issues": len(dev_issues) >= 10,
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    train_path = args.output_dir / "sft_train.jsonl"
    dev_path = args.output_dir / "sft_dev.jsonl"
    pair_train_path = args.output_dir / "actual_base_pairs_train.jsonl"
    pair_dev_path = args.output_dir / "actual_base_pairs_dev.jsonl"
    atomic_jsonl(train_path, train_rows)
    atomic_jsonl(dev_path, dev_rows)
    atomic_jsonl(pair_train_path, pair_train)
    atomic_jsonl(pair_dev_path, pair_dev)
    report = {
        "schema_version": "ksova-cycle05-materialization-1.0",
        "rows": {"train": len(train_rows), "dev": len(dev_rows)},
        "issues": {"train": len(train_issues), "dev": len(dev_issues)},
        "answer_origins": dict(sorted(origins.items())),
        "train_domain_counts": dict(sorted(domain_counts.items())),
        "sample_weight_range": [
            min((row["metadata"]["sample_weight"] for row in train_rows), default=0),
            max((row["metadata"]["sample_weight"] for row in train_rows), default=0),
        ],
        "actual_base_pairs": {"train": len(pair_train), "dev": len(pair_dev)},
        "actual_base_pair_issues": {
            "train": len(pair_train_issues), "dev": len(pair_dev_issues)
        },
        "gates": gates,
        "outputs": {
            "sft_train_sha256": sha256(train_path),
            "sft_dev_sha256": sha256(dev_path),
            "actual_base_pairs_train_sha256": sha256(pair_train_path),
            "actual_base_pairs_dev_sha256": sha256(pair_dev_path),
        },
    }
    atomic_json(args.output_dir / "materialization_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(gates.values()) else 1


def add_fixed_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--issues", type=Path, default=Path("data/current/issues.jsonl"))
    parser.add_argument("--sources", type=Path, default=Path("data/current/sources.jsonl"))
    parser.add_argument("--train-examples", type=Path, default=Path("data/current/train_examples.jsonl"))
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark/v2.0/releases/first_pass_20260810_v1/benchmark/items.jsonl"))
    parser.add_argument("--selected-issues", type=Path, default=Path("benchmark/v2.0/work/runs/method_search_20260815_cycle02_onpolicy/data/prepared/selected_issues.jsonl"))
    parser.add_argument("--prompts", type=Path, default=Path("benchmark/v2.0/work/runs/method_search_20260815_cycle02_onpolicy/data/prompts/prompts.jsonl"))
    parser.add_argument("--evidence-packets", type=Path, default=Path("benchmark/v2.0/work/runs/method_search_20260816_cycle04_grounded_sft_dpo/data/prepared/evidence_packets.jsonl"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("prepare")
    add_fixed_paths(command)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(func=prepare)

    command = sub.add_parser("aggregate-candidates")
    command.add_argument("--prompts", type=Path, required=True)
    command.add_argument("--qwen-fact-outputs", type=Path, required=True)
    command.add_argument("--gemma-outputs", type=Path, required=True)
    command.add_argument("--base-outputs", type=Path, required=True)
    command.add_argument("--cycle04-raw-outputs", type=Path, required=True)
    command.add_argument("--cycle04-sft-train", type=Path, required=True)
    command.add_argument("--cycle04-sft-dev", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(func=aggregate_candidates)

    command = sub.add_parser("build-repair")
    command.add_argument("--evidence-packets", type=Path, required=True)
    command.add_argument("--candidate-sets", type=Path, required=True)
    command.add_argument("--selection-rows", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(func=build_repair)

    command = sub.add_parser("aggregate-repair")
    command.add_argument("--candidate-sets", type=Path, required=True)
    command.add_argument("--selection-rows", type=Path, required=True)
    command.add_argument("--repair-outputs", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(func=aggregate_repair)

    command = sub.add_parser("materialize")
    command.add_argument("--prompts", type=Path, required=True)
    command.add_argument("--candidate-sets", type=Path, required=True)
    command.add_argument("--selection-rows", type=Path, required=True)
    command.add_argument("--repair-selection-rows", type=Path)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(func=materialize)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
