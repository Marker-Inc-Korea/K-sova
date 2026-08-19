#!/usr/bin/env python3
"""Build the Cycle 08 150-issue SFT mix with base-response replay.

``prepare`` takes the 150 source-isolated issues frozen in Cycle 04, subtracts
the 133 issues in the Cycle 08 coverage arm, and emits exactly one request for
each of the remaining 17 issues. ``materialize`` validates the base-model
responses and combines them with the 133 coverage rows.
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


SEED = "20260818::cycle08::replay150"


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


def stable_hex(*parts: str, size: int = 20) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:size]


def normalize_question(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", value).casefold()


def row_issue_id(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("issue_id") or "")


def user_question(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def packet_source_ids(row: dict[str, Any]) -> set[str]:
    packet = row.get("evidence_packet") or {}
    values: set[str] = set()
    for fact in packet.get("verified_facts_ko") or []:
        value = str(fact.get("source_id") or "")
        if value:
            values.add(value)
    for position in packet.get("documented_positions_ko") or []:
        values.update(str(value) for value in position.get("source_ids") or [] if str(value))
    return values


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    candidate_rows = [
        row
        for path in args.candidate_sft
        for row in read_jsonl(path)
    ]
    packets = read_jsonl(args.evidence_packets)
    coverage_rows = read_jsonl(args.coverage_train)
    frozen_rows = read_jsonl(args.frozen_items)
    preparation_report = json.loads(args.preparation_report.read_text(encoding="utf-8"))

    coverage = [
        row
        for row in coverage_rows
        if (row.get("metadata") or {}).get("coverage_priority") is True
    ]
    coverage_issues = {row_issue_id(row) for row in coverage}
    if len(coverage) != 133 or len(coverage_issues) != 133 or "" in coverage_issues:
        raise ValueError("coverage train must contain one priority row for 133 issues")

    cycle04_issues = {row_issue_id(row) for row in candidate_rows}
    if len(cycle04_issues) != 150 or "" in cycle04_issues:
        raise ValueError(f"Cycle 04 candidate SFT must cover 150 issues, found {len(cycle04_issues)}")
    if not coverage_issues.issubset(cycle04_issues):
        missing = sorted(coverage_issues - cycle04_issues)
        raise ValueError(f"coverage issues are not a Cycle 04 subset: {missing}")
    replay_issues = cycle04_issues - coverage_issues
    if len(replay_issues) != 17:
        raise ValueError(f"expected exact 150 - 133 issue difference, found {len(replay_issues)}")

    isolation = preparation_report.get("isolation") or {}
    required_zero = (
        "issue_id_overlap",
        "source_id_overlap",
        "normalized_url_overlap",
        "content_sha256_overlap",
    )
    if any(isolation.get(key) != 0 for key in required_zero):
        raise ValueError(f"Cycle 04 isolation report did not pass: {isolation}")

    frozen_issues = {str(row.get("issue_id") or "") for row in frozen_rows}
    frozen_sources = {
        str(value)
        for row in frozen_rows
        for value in row.get("source_ids") or []
        if str(value)
    }
    if cycle04_issues.intersection(frozen_issues):
        raise ValueError("Cycle 04 candidate issues overlap the frozen benchmark")

    blocked_questions = {
        normalize_question(str(row.get("question_ko") or "")) for row in frozen_rows
    }
    blocked_questions.update(normalize_question(user_question(row)) for row in coverage_rows)
    blocked_questions.discard("")

    packets_by_prompt = {str(row.get("prompt_id") or ""): row for row in packets}
    candidates_by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        issue_id = row_issue_id(row)
        prompt_id = str((row.get("metadata") or {}).get("prompt_id") or "")
        question = user_question(row)
        if (
            issue_id in replay_issues
            and prompt_id in packets_by_prompt
            and question
            and normalize_question(question) not in blocked_questions
        ):
            candidates_by_issue[issue_id].append(row)

    missing_candidates = sorted(replay_issues - set(candidates_by_issue))
    if missing_candidates:
        raise ValueError(f"replay issues without isolated candidate prompts: {missing_candidates}")

    # Greedily balance the four question forms, with a seeded hash as the only
    # tie-break.  This selection is deterministic and never quota-relaxes.
    form_counts: Counter[str] = Counter()
    selected_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    selected_sources: set[str] = set()
    for issue_id in sorted(replay_issues):
        candidates = sorted(
            candidates_by_issue[issue_id],
            key=lambda row: (
                form_counts[str((row.get("metadata") or {}).get("prompt_form") or "")],
                stable_hex(
                    SEED,
                    "prompt",
                    str((row.get("metadata") or {}).get("prompt_id") or ""),
                    size=64,
                ),
            ),
        )
        candidate = candidates[0]
        metadata = candidate.get("metadata") or {}
        source_prompt_id = str(metadata.get("prompt_id") or "")
        packet = packets_by_prompt[source_prompt_id]
        sources = packet_source_ids(packet)
        form = str(metadata.get("prompt_form") or "")
        domain = str(metadata.get("domain_id") or packet.get("domain_id") or "")
        question = user_question(candidate)
        form_counts[form] += 1
        selected_sources.update(sources)
        replay_id = "KSC8REPLAY-" + stable_hex(issue_id, source_prompt_id, question)
        selected = {
            "schema_version": "ksova-cycle08-replay-selection-1.0",
            "replay_id": replay_id,
            "issue_id": issue_id,
            "domain_id": domain,
            "source_prompt_id": source_prompt_id,
            "prompt_form": form,
            "question_ko": question,
            "source_ids": sorted(sources),
            "evidence_packet_sha256": str(packet.get("evidence_packet_sha256") or ""),
        }
        selected_rows.append(selected)
        requests.append(
            {
                "id": f"replay__{replay_id}",
                "messages": [{"role": "user", "content": question}],
                "max_tokens": 4096,
                "metadata": {
                    "purpose": "training_preservation_replay_generation",
                    "evaluation_suite": "cycle08_replay150",
                    "prompt_id": replay_id,
                    "source_prompt_id": source_prompt_id,
                    "issue_id": issue_id,
                    "domain_id": domain,
                    "prompt_form": form,
                },
            }
        )

    if len(selected_rows) != 17 or len({row["issue_id"] for row in selected_rows}) != 17:
        raise ValueError("selection is not exactly 17 unique issues")
    if selected_sources.intersection(frozen_sources):
        raise ValueError("selected replay sources overlap the frozen benchmark")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.out_dir / "selected_replay_items.jsonl"
    requests_path = args.out_dir / "replay_requests.jsonl"
    write_jsonl(selected_path, selected_rows)
    write_jsonl(requests_path, requests)
    report = {
        "schema_version": "ksova-cycle08-replay-selection-report-1.0",
        "pass": True,
        "selection_rule": "exact set difference: 150 Cycle 04 isolated issues minus 133 Cycle 08 coverage issues",
        "cycle04_issues": len(cycle04_issues),
        "coverage_issues": len(coverage_issues),
        "selected_items": len(selected_rows),
        "selected_issues": len({row["issue_id"] for row in selected_rows}),
        "domain_counts": dict(sorted(Counter(row["domain_id"] for row in selected_rows).items())),
        "prompt_form_counts": dict(sorted(form_counts.items())),
        "coverage_issue_overlap": len({row["issue_id"] for row in selected_rows} & coverage_issues),
        "frozen_issue_overlap": len({row["issue_id"] for row in selected_rows} & frozen_issues),
        "frozen_source_overlap": len(selected_sources & frozen_sources),
        "normalized_question_overlap": sum(
            normalize_question(row["question_ko"]) in blocked_questions for row in selected_rows
        ),
        "cycle04_isolation_report": isolation,
        "inputs": {
            "candidate_sft": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in args.candidate_sft
            ],
            "evidence_packets": {
                "path": str(args.evidence_packets.resolve()),
                "sha256": sha256(args.evidence_packets),
            },
            "preparation_report": {
                "path": str(args.preparation_report.resolve()),
                "sha256": sha256(args.preparation_report),
            },
            "coverage_train": {
                "path": str(args.coverage_train.resolve()),
                "sha256": sha256(args.coverage_train),
            },
            "frozen_items": {
                "path": str(args.frozen_items.resolve()),
                "sha256": sha256(args.frozen_items),
            },
        },
        "outputs": {
            "selected": {"path": str(selected_path.resolve()), "sha256": sha256(selected_path)},
            "requests": {"path": str(requests_path.resolve()), "sha256": sha256(requests_path)},
        },
    }
    write_json(args.out_dir / "selection_report.json", report)
    return report


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    selected = read_jsonl(args.selected)
    coverage_rows = read_jsonl(args.coverage_train)
    frozen_rows = read_jsonl(args.frozen_items)
    responses = read_jsonl(args.base_responses)

    coverage = [
        row
        for row in coverage_rows
        if (row.get("metadata") or {}).get("coverage_priority") is True
    ]
    coverage_issues = {row_issue_id(row) for row in coverage}
    frozen_issues = {str(row.get("issue_id") or "") for row in frozen_rows}
    by_prompt = {
        str((row.get("metadata") or {}).get("prompt_id") or ""): row
        for row in responses
    }
    expected = {str(row["replay_id"]) for row in selected}
    if len(selected) != 17 or len(responses) != 17 or set(by_prompt) != expected:
        raise ValueError("base responses do not exactly cover 17 selected replay prompts")

    replay_rows: list[dict[str, Any]] = []
    for row in selected:
        replay_id = str(row["replay_id"])
        response = by_prompt[replay_id]
        if args.expected_model and response.get("model") != args.expected_model:
            raise ValueError(f"{replay_id}: unexpected model {response.get('model')}")
        if response.get("error"):
            raise ValueError(f"{replay_id}: runtime error {response.get('error')}")
        if str(response.get("finish_reason") or "").casefold() != "stop":
            raise ValueError(f"{replay_id}: incomplete generation")
        answer = str(response.get("response") or "").strip()
        if not answer:
            raise ValueError(f"{replay_id}: empty response")
        issue_id = str(row["issue_id"])
        replay_rows.append(
            {
                "schema_version": "ksova-cycle08-replay-sft-1.0",
                "messages": [
                    {"role": "user", "content": str(row["question_ko"])},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "training_id": "KSC8TRAIN-" + stable_hex(replay_id, answer),
                    "experiment_id": "method_search_20260817_cycle08_learning_ladder",
                    "arm": "replay150_sft",
                    "split": "train",
                    "issue_id": issue_id,
                    "domain_id": str(row["domain_id"]),
                    "prompt_id": replay_id,
                    "source_prompt_id": str(row["source_prompt_id"]),
                    "prompt_form": str(row["prompt_form"]),
                    "example_kind": "base_onpolicy_preservation_replay",
                    "coverage_priority": True,
                    "source": "Cycle 04 frozen-isolated prompt with base response",
                },
            }
        )

    combined = coverage + replay_rows
    combined_issues = {row_issue_id(row) for row in combined}
    training_ids = [
        str((row.get("metadata") or {}).get("training_id") or "") for row in combined
    ]
    if len(coverage) != 133 or len(coverage_issues) != 133:
        raise ValueError("coverage input is not exactly 133 unique issues")
    if len(combined) != 150 or len(combined_issues) != 150:
        raise ValueError("combined SFT is not exactly 150 rows and 150 issues")
    if "" in training_ids or len(set(training_ids)) != 150:
        raise ValueError("combined training identifiers are incomplete or duplicated")
    if combined_issues.intersection(frozen_issues):
        raise ValueError("combined training issues overlap frozen benchmark")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    replay_path = args.out_dir / "replay_sft.jsonl"
    combined_path = args.out_dir / "replay150_sft.train.jsonl"
    write_jsonl(replay_path, replay_rows)
    write_jsonl(combined_path, combined)
    token_counts = [
        int((by_prompt[row["replay_id"]].get("usage") or {}).get("completion_tokens") or 0)
        for row in selected
    ]
    report = {
        "schema_version": "ksova-cycle08-replay150-materialization-report-1.0",
        "pass": True,
        "coverage_rows": len(coverage),
        "coverage_issues": len(coverage_issues),
        "replay_rows": len(replay_rows),
        "replay_issues": len({row_issue_id(row) for row in replay_rows}),
        "combined_rows": len(combined),
        "combined_issues": len(combined_issues),
        "frozen_issue_overlap": len(combined_issues.intersection(frozen_issues)),
        "replay_generation": {
            "model": args.expected_model or None,
            "all_stop": True,
            "completion_tokens_mean": round(sum(token_counts) / len(token_counts), 3),
            "completion_tokens_min": min(token_counts),
            "completion_tokens_max": max(token_counts),
        },
        "inputs": {
            "selected": {"path": str(args.selected.resolve()), "sha256": sha256(args.selected)},
            "base_responses": {
                "path": str(args.base_responses.resolve()),
                "sha256": sha256(args.base_responses),
            },
            "coverage_train": {
                "path": str(args.coverage_train.resolve()),
                "sha256": sha256(args.coverage_train),
            },
            "frozen_items": {
                "path": str(args.frozen_items.resolve()),
                "sha256": sha256(args.frozen_items),
            },
        },
        "outputs": {
            "replay_sft": {"path": str(replay_path.resolve()), "sha256": sha256(replay_path)},
            "combined_sft": {"path": str(combined_path.resolve()), "sha256": sha256(combined_path)},
        },
    }
    write_json(args.out_dir / "materialization_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--candidate-sft", type=Path, nargs="+", required=True)
    prepare_parser.add_argument("--evidence-packets", type=Path, required=True)
    prepare_parser.add_argument("--preparation-report", type=Path, required=True)
    prepare_parser.add_argument("--coverage-train", type=Path, required=True)
    prepare_parser.add_argument("--frozen-items", type=Path, required=True)
    prepare_parser.add_argument("--out-dir", type=Path, required=True)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--selected", type=Path, required=True)
    materialize_parser.add_argument("--base-responses", type=Path, required=True)
    materialize_parser.add_argument("--coverage-train", type=Path, required=True)
    materialize_parser.add_argument("--frozen-items", type=Path, required=True)
    materialize_parser.add_argument("--expected-model", default="")
    materialize_parser.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args()
    result = prepare(args) if args.command == "prepare" else materialize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
