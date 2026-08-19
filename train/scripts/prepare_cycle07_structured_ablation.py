#!/usr/bin/env python3
"""Materialize the matched Cycle07 final-rehearsal vs structured-SFT ablation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:20]
    return f"KSC7-{digest}"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def select_facts(packet: dict[str, Any], limit: int = 3) -> list[str]:
    facts = unique_strings([
        str(row.get("claim_ko") or "")
        for row in packet.get("verified_facts_ko") or []
    ])
    if facts:
        return facts[:limit]
    references = {
        str(row.get("task") or ""): str(row.get("answer_ko") or "").strip()
        for row in packet.get("validated_reference_answers_ko") or []
    }
    uncertainty = references.get("uncertainty_consequence_boundary", "")
    if uncertainty:
        fact_part = re.split(r"불확실한 부분(?:은|:)", uncertainty, maxsplit=1)[0]
        fact_part = re.sub(r"^사실로 확인된 부분(?:은|:)?\s*", "", fact_part).strip()
        if fact_part:
            return [fact_part]
    for task in ("final_synthesis", "competing_claim_resolution"):
        if references.get(task):
            return [references[task]]
    return []


def select_positions(packet: dict[str, Any], limit: int = 4) -> list[dict[str, str]]:
    positions = list(packet.get("documented_positions_ko") or [])
    selected: list[dict[str, str]] = []
    used: set[tuple[str, str]] = set()

    def take(predicate: Any) -> None:
        for row in positions:
            actor = " ".join(str(row.get("actor") or "").split())
            claim = " ".join(str(row.get("claim_ko") or "").split())
            key = (actor, claim)
            if (
                len(selected) < limit
                and actor and claim and key not in used and predicate(row)
            ):
                selected.append({
                    "actor": actor,
                    "claim_ko": claim,
                    "kind": str(row.get("kind") or "position"),
                    "role": str(row.get("role") or "unspecified"),
                })
                used.add(key)

    take(lambda row: str(row.get("role") or "") == "counterargument")
    take(lambda row: str(row.get("role") or "") in {"official", "analysis"})
    take(lambda row: str(row.get("kind") or "") == "external_position")
    take(lambda row: True)
    return selected


def labeled_section(text: str, label: str, following: list[str]) -> str:
    pattern = rf"(?:^|\s){re.escape(label)}\s*:\s*"
    match = re.search(pattern, text)
    if not match:
        return ""
    end = len(text)
    for next_label in following:
        next_match = re.search(
            rf"(?:^|\s){re.escape(next_label)}\s*:\s*",
            text[match.end():],
        )
        if next_match:
            end = min(end, match.end() + next_match.start())
    return text[match.end():end].strip()


def legacy_decision_reference(packet: dict[str, Any]) -> dict[str, Any]:
    references = {
        str(row.get("task") or ""): str(row.get("answer_ko") or "").strip()
        for row in packet.get("validated_reference_answers_ko") or []
    }
    baseline = references.get("baseline_and_range", "")
    minimum = labeled_section(baseline, "최소 기준", ["허용 범위", "실패 방향"])
    acceptable = labeled_section(baseline, "허용 범위", ["실패 방향"])
    failure = labeled_section(baseline, "실패 방향", [])
    stake = references.get("korean_stake_extraction", "")
    if not all((minimum, acceptable, failure, stake)):
        raise ValueError(
            f"legacy reference packet cannot be deterministically parsed: "
            f"{packet.get('issue_id')}"
        )
    return {
        "decision_core_ko": str(packet.get("title_ko") or "").strip(),
        "sovereign_stake_ko": stake,
        "korean_reference_direction_ko": f"{minimum} {acceptable}".strip(),
        "acceptable_range_ko": [acceptable],
        "failure_direction_ko": failure,
    }


def compact_dossier(question: str, packet: dict[str, Any]) -> dict[str, Any]:
    primary_decision = packet.get("decision_reference_ko") or {}
    fallback_decision = (
        legacy_decision_reference(packet)
        if packet.get("validated_reference_answers_ko")
        else {}
    )
    decision = {
        key: primary_decision.get(key) or fallback_decision.get(key)
        for key in {
            "decision_core_ko", "sovereign_stake_ko",
            "korean_reference_direction_ko", "acceptable_range_ko",
            "failure_direction_ko",
        }
    }
    required = {
        "issue_id": str(packet.get("issue_id") or ""),
        "title_ko": str(packet.get("title_ko") or "").strip(),
        "temporal_scope": str(packet.get("temporal_scope") or "").strip(),
        "question_ko": question.strip(),
        "decision_core_ko": str(decision.get("decision_core_ko") or "").strip(),
        "sovereign_stake_ko": str(decision.get("sovereign_stake_ko") or "").strip(),
        "korean_reference_direction_ko": str(
            decision.get("korean_reference_direction_ko") or ""
        ).strip(),
        "acceptable_range_ko": unique_strings(
            [str(value) for value in decision.get("acceptable_range_ko") or []]
        ),
        "failure_direction_ko": str(
            decision.get("failure_direction_ko") or ""
        ).strip(),
        "verified_facts_ko": select_facts(packet),
        "documented_positions_ko": select_positions(packet),
    }
    missing = [
        key for key in (
            "issue_id", "title_ko", "question_ko", "decision_core_ko",
            "sovereign_stake_ko", "korean_reference_direction_ko",
            "failure_direction_ko",
        )
        if not required[key]
    ]
    if missing or not required["acceptable_range_ko"] or not required["verified_facts_ko"]:
        raise ValueError(f"incomplete evidence packet {required['issue_id']}: {missing}")
    return required


SPECIFICATION = """판단 명세
1. 검증된 사실과 각 행위자의 주장을 구분하고, 주장을 사실로 승격하지 않는다.
2. 한국의 구체적 권리·책임·비용·선택권을 확인하되 정부의 공식 입장만을 정답으로 삼지 않는다.
3. 한국 사회에 실제로 형성된 경쟁 논거 중 가장 강한 것을 공정하게 남긴다.
4. 하나의 구호가 아니라 허용 가능한 판단 범위와 결론이 달라지는 조건을 표시한다.
5. 자료의 시간·사실 경계를 넘는 세부사항은 만들지 않고 불확실성으로 남긴다."""


def structured_messages(question: str, packet: dict[str, Any]) -> list[dict[str, str]]:
    dossier = compact_dossier(question, packet)
    user_payload = {
        "질문": dossier["question_ko"],
        "쟁점": dossier["title_ko"],
        "시간범위": dossier["temporal_scope"],
        "판단핵심": dossier["decision_core_ko"],
        "한국의권리책임비용": dossier["sovereign_stake_ko"],
        "검증사실": dossier["verified_facts_ko"],
        "기록된입장": [
            {"행위자": row["actor"], "주장": row["claim_ko"]}
            for row in dossier["documented_positions_ko"]
        ],
        "한국사회기준선": dossier["korean_reference_direction_ko"],
        "허용범위": dossier["acceptable_range_ko"],
        "금지오류": dossier["failure_direction_ko"],
    }
    user = (
        "[작업 유형: K-SOVA 판단 준비 — 사용자용 최종답변이 아님]\n"
        f"{SPECIFICATION}\n\n"
        "아래 검증 자료에 명세를 적용해 판단 준비 기록을 작성하라. "
        "제공된 문구를 기계적으로 반복하지 말고 사실·귀속, 한국의 이해관계, "
        "경쟁 논거, 판단 범위, 조건·경계의 관계가 보이게 정리하라. "
        "자연스러운 사용자용 최종답변은 쓰지 않는다.\n\n"
        "검증 자료:\n"
        + json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
    )
    fact_lines = "\n".join(
        f"- {fact}" for fact in dossier["verified_facts_ko"]
    )
    position_lines = "\n".join(
        f"- {row['actor']}: {row['claim_ko']}"
        for row in dossier["documented_positions_ko"]
    ) or "- 검증 자료에 별도 경쟁 입장이 기록되지 않음"
    range_lines = "\n".join(
        f"- {value}" for value in dossier["acceptable_range_ko"]
    )
    temporal = dossier["temporal_scope"] or "제공 자료가 포괄하는 시점"
    assistant = f"""판단 핵심
{dossier['decision_core_ko']}

사실과 귀속
{fact_lines}
기록된 입장은 해당 행위자의 주장으로만 사용하며, 검증 사실과 혼합하지 않는다.

한국의 권리·책임·비용
{dossier['sovereign_stake_ko']}

경쟁 논거
{position_lines}

한국 사회의 판단 기준선
{dossier['korean_reference_direction_ko']}

허용 가능한 판단 범위
{range_lines}

조건·불확실성·경계
- 시간 경계는 {temporal}이다.
- 자료가 직접 뒷받침하지 않는 인명·기관·수치·날짜·법적 효과는 확정하지 않는다.
- 피해야 할 오류: {dossier['failure_direction_ko']}"""
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def parse_teacher_requests(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        metadata = row.get("metadata") or {}
        prompt_id = str(metadata.get("prompt_id") or "")
        messages = row.get("messages") or []
        if len(messages) != 2 or messages[1].get("role") != "user":
            raise ValueError(f"invalid teacher request messages: {prompt_id}")
        payload = json.loads(str(messages[1].get("content") or "{}"))
        if str(payload.get("prompt_id") or "") != prompt_id or not prompt_id:
            raise ValueError(f"teacher prompt ID mismatch: {prompt_id}")
        if prompt_id in result:
            raise ValueError(f"duplicate teacher prompt ID: {prompt_id}")
        result[prompt_id] = payload
    return result


def common_final_row(
    source: dict[str, Any], arm: str, split: str,
) -> dict[str, Any]:
    metadata = source.get("metadata") or {}
    source_id = str(metadata.get("training_id") or "")
    issue_id = str(metadata.get("issue_id") or "")
    prompt_id = str(metadata.get("prompt_id") or "")
    return {
        "schema_version": "ksova-cycle07-sft-1.0",
        "messages": source["messages"],
        "metadata": {
            "training_id": stable_id("cycle07", arm, "application", source_id),
            "source_training_id": source_id,
            "source_experiment_id": str(metadata.get("experiment_id") or ""),
            "experiment_id": "method_search_20260817_cycle07_structured_sft_ablation",
            "arm": arm,
            "split": split,
            "issue_id": issue_id,
            "domain_id": str(metadata.get("domain_id") or ""),
            "prompt_id": prompt_id,
            "prompt_form": str(metadata.get("prompt_form") or ""),
            "example_kind": "application_final",
            "curriculum_stage": 2,
            "coverage_priority": False,
            "answer_origin": str(metadata.get("answer_origin") or ""),
        },
    }


def coverage_row(
    canonical: dict[str, Any], teacher: dict[str, Any], arm: str, split: str,
) -> dict[str, Any]:
    metadata = canonical.get("metadata") or {}
    issue_id = str(metadata.get("issue_id") or "")
    prompt_id = str(metadata.get("prompt_id") or "")
    source_id = str(metadata.get("training_id") or "")
    if arm == "final_rehearsal_sft":
        messages = canonical["messages"]
        kind = "coverage_final_rehearsal"
    elif arm == "structured_curriculum_sft":
        payload_issue = str((teacher.get("evidence_packet") or {}).get("issue_id") or "")
        if payload_issue != issue_id:
            raise ValueError(f"evidence issue mismatch for {prompt_id}")
        messages = structured_messages(
            str(teacher.get("question_ko") or ""),
            teacher["evidence_packet"],
        )
        kind = "coverage_structured_dossier"
    else:
        raise ValueError(f"unknown arm: {arm}")
    return {
        "schema_version": "ksova-cycle07-sft-1.0",
        "messages": messages,
        "metadata": {
            "training_id": stable_id("cycle07", arm, "coverage", issue_id, source_id),
            "source_training_id": source_id,
            "experiment_id": "method_search_20260817_cycle07_structured_sft_ablation",
            "arm": arm,
            "split": split,
            "issue_id": issue_id,
            "domain_id": str(metadata.get("domain_id") or ""),
            "prompt_id": prompt_id,
            "prompt_form": str(metadata.get("prompt_form") or ""),
            "example_kind": kind,
            "curriculum_stage": 1,
            "coverage_priority": True,
            "source": "cycle05_validated_evidence_and_selected_answer",
        },
    }


def apply_template(tokenizer: Any, messages: list[dict[str, str]], generation: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def token_stats(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_lengths: list[int] = []
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    by_kind: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        messages = row["messages"]
        prompt = apply_template(tokenizer, messages[:1], True)
        full = apply_template(tokenizer, messages, False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("chat template prompt is not a full-sequence prefix")
        completion = len(full_ids) - len(prompt_ids)
        if completion <= 0:
            raise ValueError("empty completion after tokenization")
        full_lengths.append(len(full_ids))
        prompt_lengths.append(len(prompt_ids))
        completion_lengths.append(completion)
        by_kind[str(row["metadata"]["example_kind"])].append(len(full_ids))
    return {
        "rows": len(rows),
        "full_tokens": {
            "min": min(full_lengths),
            "mean": round(sum(full_lengths) / len(full_lengths), 3),
            "max": max(full_lengths),
            "sum": sum(full_lengths),
        },
        "prompt_tokens": {
            "min": min(prompt_lengths),
            "mean": round(sum(prompt_lengths) / len(prompt_lengths), 3),
            "max": max(prompt_lengths),
            "sum": sum(prompt_lengths),
        },
        "completion_tokens": {
            "min": min(completion_lengths),
            "mean": round(sum(completion_lengths) / len(completion_lengths), 3),
            "max": max(completion_lengths),
            "sum": sum(completion_lengths),
        },
        "maximum_full_tokens_by_kind": {
            key: max(values) for key, values in sorted(by_kind.items())
        },
    }


def message_pair_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row["messages"], ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def validate_arm(
    rows: list[dict[str, Any]], arm: str, split: str, expected_issues: set[str],
) -> None:
    ids = [str(row["metadata"]["training_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate training IDs in {arm}/{split}")
    if {str(row["metadata"]["arm"]) for row in rows} != {arm}:
        raise ValueError(f"arm label mismatch in {arm}/{split}")
    issues = {str(row["metadata"]["issue_id"]) for row in rows}
    if issues != expected_issues:
        raise ValueError(f"issue coverage mismatch in {arm}/{split}")
    priority = [row for row in rows if row["metadata"]["coverage_priority"] is True]
    if len(priority) != len(expected_issues):
        raise ValueError(f"priority row count mismatch in {arm}/{split}")
    priority_issues = [str(row["metadata"]["issue_id"]) for row in priority]
    if len(priority_issues) != len(set(priority_issues)) or set(priority_issues) != issues:
        raise ValueError(f"priority rows do not cover issues exactly once in {arm}/{split}")
    for row in rows:
        messages = row.get("messages") or []
        if (
            len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or not str(messages[0].get("content") or "").strip()
            or not str(messages[1].get("content") or "").strip()
        ):
            raise ValueError(f"invalid messages in {row['metadata']['training_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-requests", type=Path, required=True)
    parser.add_argument("--sft-train", type=Path, required=True)
    parser.add_argument("--sft-dev", type=Path, required=True)
    parser.add_argument("--benchmark-items", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=3072)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    teachers = parse_teacher_requests(args.teacher_requests)
    source_by_split = {
        "train": read_jsonl(args.sft_train),
        "dev": read_jsonl(args.sft_dev),
    }
    benchmark_rows = read_jsonl(args.benchmark_items)
    benchmark_issues = {str(row["issue_id"]) for row in benchmark_rows}
    benchmark_questions = {" ".join(str(row["question_ko"]).split()) for row in benchmark_rows}
    arms = ("final_rehearsal_sft", "structured_curriculum_sft")
    outputs: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {} for arm in arms
    }
    issue_manifest: list[dict[str, Any]] = []

    for split, source_rows in source_by_split.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in source_rows:
            metadata = row.get("metadata") or {}
            issue_id = str(metadata.get("issue_id") or "")
            prompt_id = str(metadata.get("prompt_id") or "")
            if str(metadata.get("split") or "") != split:
                raise ValueError(f"source split mismatch: {prompt_id}")
            if not issue_id or not prompt_id or prompt_id not in teachers:
                raise ValueError(f"source metadata/evidence missing: {prompt_id}")
            grouped[issue_id].append(row)
        issue_ids = set(grouped)
        overlap = issue_ids & benchmark_issues
        if overlap:
            raise ValueError(f"benchmark issue leakage in {split}: {sorted(overlap)}")
        exact_question_overlap = {
            " ".join(str(row["messages"][0]["content"]).split())
            for row in source_rows
        } & benchmark_questions
        if exact_question_overlap:
            raise ValueError(f"exact benchmark question leakage in {split}")

        canonical_by_issue = {
            issue_id: sorted(
                rows,
                key=lambda row: (
                    str((row.get("metadata") or {}).get("prompt_id") or ""),
                    str((row.get("metadata") or {}).get("training_id") or ""),
                ),
            )[0]
            for issue_id, rows in grouped.items()
        }
        for issue_id in sorted(issue_ids):
            canonical = canonical_by_issue[issue_id]
            meta = canonical["metadata"]
            issue_manifest.append({
                "issue_id": issue_id,
                "split": split,
                "domain_id": str(meta.get("domain_id") or ""),
                "canonical_prompt_id": str(meta.get("prompt_id") or ""),
                "source_final_rows": len(grouped[issue_id]),
            })

        sorted_finals = sorted(
            source_rows,
            key=lambda row: str((row.get("metadata") or {}).get("training_id") or ""),
        )
        for arm in arms:
            coverage = []
            for issue_id in sorted(issue_ids):
                canonical = canonical_by_issue[issue_id]
                prompt_id = str(canonical["metadata"]["prompt_id"])
                coverage.append(coverage_row(
                    canonical, teachers[prompt_id], arm, split
                ))
            application = [
                common_final_row(row, arm, split) for row in sorted_finals
            ]
            rows = coverage + application
            validate_arm(rows, arm, split, issue_ids)
            outputs[arm][split] = rows

    train_issues = {
        row["issue_id"] for row in issue_manifest if row["split"] == "train"
    }
    dev_issues = {
        row["issue_id"] for row in issue_manifest if row["split"] == "dev"
    }
    if train_issues & dev_issues:
        raise ValueError("train/dev issue leakage")
    if len(train_issues) != 133 or len(dev_issues) != 15:
        raise ValueError(
            f"unexpected issue split: train={len(train_issues)} dev={len(dev_issues)}"
        )
    for arm in arms:
        if len(outputs[arm]["train"]) != 556 or len(outputs[arm]["dev"]) != 69:
            raise ValueError(f"unexpected row count for {arm}")

    control_application = Counter(
        message_pair_hash(row)
        for row in outputs["final_rehearsal_sft"]["train"]
        if row["metadata"]["example_kind"] == "application_final"
    )
    structured_application = Counter(
        message_pair_hash(row)
        for row in outputs["structured_curriculum_sft"]["train"]
        if row["metadata"]["example_kind"] == "application_final"
    )
    if control_application != structured_application:
        raise ValueError("shared application-stage final answers differ between arms")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True, local_files_only=True
    )
    token_reports: dict[str, dict[str, Any]] = {}
    for arm in arms:
        token_reports[arm] = {}
        for split in ("train", "dev"):
            stats = token_stats(tokenizer, outputs[arm][split])
            if stats["full_tokens"]["max"] > args.max_length:
                raise ValueError(
                    f"{arm}/{split} exceeds max length: "
                    f"{stats['full_tokens']['max']} > {args.max_length}"
                )
            token_reports[arm][split] = stats

    paths: dict[str, dict[str, Any]] = {}
    for arm in arms:
        paths[arm] = {}
        for split in ("train", "dev"):
            path = args.output_dir / f"{arm}.{split}.jsonl"
            write_jsonl(path, outputs[arm][split])
            paths[arm][split] = {
                "path": str(path.resolve()),
                "rows": len(outputs[arm][split]),
                "issues": len({
                    row["metadata"]["issue_id"] for row in outputs[arm][split]
                }),
                "sha256": sha256(path),
            }
    issue_manifest_path = args.output_dir / "issue_split.jsonl"
    write_jsonl(issue_manifest_path, sorted(
        issue_manifest, key=lambda row: (row["split"], row["issue_id"])
    ))

    report = {
        "schema_version": "ksova-cycle07-materialization-1.0",
        "experiment_id": "method_search_20260817_cycle07_structured_sft_ablation",
        "inputs": {
            "teacher_requests": {
                "path": str(args.teacher_requests.resolve()),
                "sha256": sha256(args.teacher_requests),
            },
            "sft_train": {
                "path": str(args.sft_train.resolve()),
                "sha256": sha256(args.sft_train),
            },
            "sft_dev": {
                "path": str(args.sft_dev.resolve()),
                "sha256": sha256(args.sft_dev),
            },
            "benchmark_items": {
                "path": str(args.benchmark_items.resolve()),
                "sha256": sha256(args.benchmark_items),
            },
        },
        "issue_split": {
            "train": len(train_issues),
            "dev": len(dev_issues),
            "train_dev_overlap": 0,
            "benchmark_issue_overlap": 0,
            "exact_benchmark_question_overlap": 0,
            "path": str(issue_manifest_path.resolve()),
            "sha256": sha256(issue_manifest_path),
        },
        "arms": paths,
        "shared_application_stage": {
            "train_rows": sum(control_application.values()),
            "message_multiset_equal": True,
            "message_multiset_sha256": hashlib.sha256(
                "\n".join(
                    f"{key}:{value}" for key, value in sorted(control_application.items())
                ).encode()
            ).hexdigest(),
        },
        "coverage_policy": {
            "priority_rows_per_train_issue": 1,
            "priority_rows_per_dev_issue": 1,
            "replacement": False,
            "training_order_enforced_by": "--coverage-first-sft-sampling",
        },
        "tokenization": {
            "model_path": str(args.model_path.resolve()),
            "max_length": args.max_length,
            "arms": token_reports,
        },
        "example_kind_counts": {
            arm: {
                split: dict(sorted(Counter(
                    row["metadata"]["example_kind"] for row in outputs[arm][split]
                ).items()))
                for split in ("train", "dev")
            }
            for arm in arms
        },
        "generation_model_used": False,
        "pass": True,
    }
    write_json(args.output_dir / "materialization_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
