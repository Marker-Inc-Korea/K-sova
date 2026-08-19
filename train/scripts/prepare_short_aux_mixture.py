#!/usr/bin/env python3
"""Materialize a 70--80% final-answer / 20--30% short-aux SFT candidate.

This is a candidate dataset for the next experiment, not an automatically
promoted training release.  Each issue contributes one isolated fixed-field
auxiliary operation; the remaining rows are validated natural final answers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from prepare_same_issue_unseen_diagnostic import issue_card


TASKS = (
    "fact_attribution",
    "korean_stake_mapping",
    "competing_arguments",
    "decision_conditions",
)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: str) -> str:
    value = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"KSAUX-{value}"


def compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def eligible_tasks(card: dict[str, Any]) -> list[str]:
    tasks = ["korean_stake_mapping", "decision_conditions"]
    if card["verified_facts_ko"] and card["documented_positions_ko"]:
        tasks.append("fact_attribution")
    if len(card["documented_positions_ko"]) >= 2:
        tasks.append("competing_arguments")
    return [task for task in TASKS if task in tasks]


def choose_task(card: dict[str, Any], counts: Counter[str]) -> str:
    eligible = eligible_tasks(card)
    return min(eligible, key=lambda task: (counts[task], TASKS.index(task)))


def auxiliary_messages(
    task: str, card: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    title = card["title_ko"]
    system = (
        "당신은 K-SOVA의 짧은 판단 보조과제를 수행한다. 이 형식은 사용자용 "
        "최종답변이 아니며, 요청된 JSON 필드만 간결하게 채운다. 제공 자료 밖의 "
        "사실을 추가하지 않고 기록된 주장을 검증 사실로 바꾸지 않는다."
    )
    if task == "fact_attribution":
        facts = [row["claim_ko"] for row in card["verified_facts_ko"][:3]]
        positions = [
            {"actor": row["actor"], "claim": row["claim_ko"]}
            for row in card["documented_positions_ko"][:3]
        ]
        payload = {
            "쟁점": title,
            "검증사실후보": facts,
            "기록된입장후보": positions,
        }
        answer = {
            "verified_facts": facts,
            "attributed_claims": positions,
            "boundary": "기록된 입장은 해당 행위자의 주장으로만 사용한다.",
        }
        instruction = (
            "검증 사실과 행위자에게 귀속해야 하는 주장을 분리해 다음 필드로 "
            "반환하라: verified_facts, attributed_claims, boundary."
        )
    elif task == "korean_stake_mapping":
        payload = {
            "쟁점": title,
            "판단핵심": card["decision_core_ko"],
            "한국관련검증기준": card["sovereign_stake_ko"],
        }
        answer = {
            "rights_responsibilities_costs": card["sovereign_stake_ko"],
            "available_choice_direction": card["korean_reference_direction_ko"],
            "boundary": "정부의 공식 입장 자체를 유일한 정답으로 삼지 않는다.",
        }
        instruction = (
            "한국의 권리·책임·비용·선택권을 다음 필드로 정리하라: "
            "rights_responsibilities_costs, available_choice_direction, boundary."
        )
    elif task == "competing_arguments":
        positions = [
            {
                "actor": row["actor"],
                "role": row["role"],
                "claim": row["claim_ko"],
            }
            for row in card["documented_positions_ko"][:4]
        ]
        payload = {"쟁점": title, "기록된입장": positions}
        answer = {
            "documented_arguments": positions,
            "comparison_rule": (
                "각 논거를 행위자에게 귀속하고, 가장 강한 반대 논거도 삭제하지 않는다."
            ),
        }
        instruction = (
            "서로 경쟁하는 논거를 다음 필드로 반환하라: documented_arguments, "
            "comparison_rule."
        )
    elif task == "decision_conditions":
        payload = {
            "쟁점": title,
            "판단핵심": card["decision_core_ko"],
            "검증된판단범위": {
                "기준선": card["korean_reference_direction_ko"],
                "허용범위": card["acceptable_range_ko"],
                "실패경계": card["failure_direction_ko"],
            },
        }
        answer = {
            "defensible_direction": card["korean_reference_direction_ko"],
            "acceptable_range": card["acceptable_range_ko"][:2],
            "change_or_failure_boundary": card["failure_direction_ko"],
        }
        instruction = (
            "방어 가능한 판단과 조건 경계를 다음 필드로 반환하라: "
            "defensible_direction, acceptable_range, change_or_failure_boundary."
        )
    else:
        raise ValueError(f"unknown task: {task}")
    user = (
        f"[보조 판단 과제: {task}]\n{system}\n\n{instruction}\n\n검증 자료:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    messages = [
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
        },
    ]
    return messages, answer


def materialize_aux(
    split_rows: list[dict[str, Any]],
    teacher_by_issue: dict[str, list[dict[str, Any]]],
    split_name: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    task_counts: Counter[str] = Counter()
    result: list[dict[str, Any]] = []
    ordered = sorted(
        split_rows,
        key=lambda row: hashlib.sha256(
            f"20260817::short-aux::{row['issue_id']}".encode("utf-8")
        ).hexdigest(),
    )
    for split_row in ordered:
        issue_id = str(split_row["issue_id"])
        teacher_rows = teacher_by_issue[issue_id]
        if len(teacher_rows) != 4:
            raise ValueError(f"{issue_id}: expected four teacher requests")
        source = teacher_rows[0]
        packet = json.loads(str(source["messages"][1]["content"]))[
            "evidence_packet"
        ]
        card = issue_card(packet)
        task = choose_task(card, task_counts)
        task_counts[task] += 1
        messages, answer = auxiliary_messages(task, card)
        output_chars = len(messages[-1]["content"])
        if output_chars > 1800:
            raise ValueError(
                f"{issue_id}: auxiliary output is not short ({output_chars} chars)"
            )
        result.append(
            {
                "schema_version": "ksova-short-fixed-aux-sft-1.0",
                "messages": messages,
                "metadata": {
                    "training_id": stable_id(split_name, issue_id, task),
                    "split": split_name,
                    "issue_id": issue_id,
                    "domain_id": str(split_row["domain_id"]),
                    "example_kind": "fixed_field_auxiliary",
                    "task_family": task,
                    "output_format": "explicit_task_tag_json",
                    "natural_final_answer": False,
                    "answer_chars": output_chars,
                    "evidence_packet_sha256": str(
                        (source.get("metadata") or {}).get(
                            "evidence_packet_sha256"
                        )
                        or ""
                    ),
                    "answer_field_count": len(answer),
                },
            }
        )
    return result, task_counts


def normalize_final_rows(
    rows: list[dict[str, Any]], split_name: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "split": split_name,
                "example_kind": "natural_final_answer",
                "task_family": "final_answer",
                "natural_final_answer": True,
            }
        )
        result.append({**row, "metadata": metadata})
    return result


def interleave(
    finals: list[dict[str, Any]], auxiliaries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    final_by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in finals:
        final_by_issue[str(row["metadata"]["issue_id"])].append(row)
    aux_by_issue = {
        str(row["metadata"]["issue_id"]): row for row in auxiliaries
    }
    output: list[dict[str, Any]] = []
    issue_order = sorted(
        final_by_issue,
        key=lambda issue_id: hashlib.sha256(
            f"20260817::short-aux-order::{issue_id}".encode("utf-8")
        ).hexdigest(),
    )
    for issue_id in issue_order:
        rows = sorted(
            final_by_issue[issue_id],
            key=lambda row: str((row.get("metadata") or {}).get("prompt_id") or ""),
        )
        midpoint = min(1, len(rows))
        output.extend(rows[:midpoint])
        output.append(aux_by_issue[issue_id])
        output.extend(rows[midpoint:])
    if len(output) != len(finals) + len(auxiliaries):
        raise ValueError("interleaving lost rows")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-split", type=Path, required=True)
    parser.add_argument("--teacher-requests", type=Path, required=True)
    parser.add_argument("--final-train", type=Path, required=True)
    parser.add_argument("--final-dev", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    split_rows = read_jsonl(args.issue_split)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_rows:
        by_split[str(row["split"])].append(row)
    if len(by_split["train"]) != 133 or len(by_split["dev"]) != 15:
        raise ValueError("expected the fixed Cycle07 133/15 issue split")

    all_issues = {str(row["issue_id"]) for row in split_rows}
    teacher_by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.teacher_requests):
        issue_id = str((row.get("metadata") or {}).get("issue_id") or "")
        if issue_id in all_issues:
            teacher_by_issue[issue_id].append(row)

    final_train = normalize_final_rows(read_jsonl(args.final_train), "train")
    final_dev = normalize_final_rows(read_jsonl(args.final_dev), "dev")
    aux_train, train_tasks = materialize_aux(
        by_split["train"], teacher_by_issue, "train"
    )
    aux_dev, dev_tasks = materialize_aux(by_split["dev"], teacher_by_issue, "dev")
    mixed_train = interleave(final_train, aux_train)
    mixed_dev = interleave(final_dev, aux_dev)

    if len(final_train) != 423 or len(final_dev) != 54:
        raise ValueError("final-answer source counts changed")
    for label, finals, aux, mixed in (
        ("train", final_train, aux_train, mixed_train),
        ("dev", final_dev, aux_dev, mixed_dev),
    ):
        issues = {str(row["metadata"]["issue_id"]) for row in mixed}
        expected = {str(row["issue_id"]) for row in by_split[label]}
        if issues != expected:
            raise ValueError(f"{label}: issue coverage mismatch")
        ids = [str(row["metadata"]["training_id"]) for row in mixed]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{label}: duplicate training ids")
        ratio = len(finals) / len(mixed)
        if not 0.70 <= ratio <= 0.80:
            raise ValueError(f"{label}: final-answer ratio outside 70--80%: {ratio}")
        if len(aux) != len(expected):
            raise ValueError(f"{label}: auxiliary coverage is not one per issue")
        for row in mixed:
            messages = row.get("messages") or []
            if (
                len(messages) != 2
                or messages[0].get("role") != "user"
                or messages[1].get("role") != "assistant"
            ):
                raise ValueError(
                    f"{label}: row violates train_method_search SFT message contract"
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "short_aux_train": (args.out_dir / "short_aux.train.jsonl", aux_train),
        "short_aux_dev": (args.out_dir / "short_aux.dev.jsonl", aux_dev),
        "mixed_train": (args.out_dir / "mixed_final_short_aux.train.jsonl", mixed_train),
        "mixed_dev": (args.out_dir / "mixed_final_short_aux.dev.jsonl", mixed_dev),
    }
    for path, rows in outputs.values():
        write_jsonl(path, rows)

    report = {
        "schema_version": "ksova-short-aux-mixture-build-1.0",
        "status": "candidate_not_promoted",
        "design": {
            "natural_final_answer_target": "70--80%",
            "fixed_field_auxiliary_target": "20--30%",
            "auxiliary_policy": (
                "one isolated, explicitly tagged, short JSON operation per issue; "
                "no long free-form dossier"
            ),
            "trainer_message_contract": "exactly one user and one assistant message",
            "task_families": list(TASKS),
        },
        "train": {
            "rows": len(mixed_train),
            "issues": len(by_split["train"]),
            "natural_final_answers": len(final_train),
            "fixed_field_auxiliaries": len(aux_train),
            "natural_final_answer_percent": round(
                len(final_train) / len(mixed_train) * 100, 3
            ),
            "task_counts": dict(sorted(train_tasks.items())),
            "aux_answer_chars": {
                "min": min(row["metadata"]["answer_chars"] for row in aux_train),
                "max": max(row["metadata"]["answer_chars"] for row in aux_train),
                "mean": round(
                    sum(row["metadata"]["answer_chars"] for row in aux_train)
                    / len(aux_train),
                    3,
                ),
            },
        },
        "dev": {
            "rows": len(mixed_dev),
            "issues": len(by_split["dev"]),
            "natural_final_answers": len(final_dev),
            "fixed_field_auxiliaries": len(aux_dev),
            "natural_final_answer_percent": round(
                len(final_dev) / len(mixed_dev) * 100, 3
            ),
            "task_counts": dict(sorted(dev_tasks.items())),
        },
        "sources": {
            "issue_split": {"path": str(args.issue_split.resolve()), "sha256": sha256(args.issue_split)},
            "teacher_requests": {"path": str(args.teacher_requests.resolve()), "sha256": sha256(args.teacher_requests)},
            "final_train": {"path": str(args.final_train.resolve()), "sha256": sha256(args.final_train)},
            "final_dev": {"path": str(args.final_dev.resolve()), "sha256": sha256(args.final_dev)},
        },
        "outputs": {
            name: {"path": str(path.resolve()), "rows": len(rows), "sha256": sha256(path)}
            for name, (path, rows) in outputs.items()
        },
    }
    write_json(args.out_dir / "data_design_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
