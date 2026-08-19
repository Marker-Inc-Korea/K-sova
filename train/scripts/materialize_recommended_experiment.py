#!/usr/bin/env python3
"""Freeze a routed, validated candidate pool into a staged LoRA experiment.

The materializer is intentionally read-only with respect to ``data/current``.
It selects one explicit portfolio route, keeps every accepted task even when an
issue has fewer than ten accepted tasks, makes an issue-disjoint train/dev
split, and audits isolation from the frozen benchmark before writing outputs.
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


EXPECTED_BENCHMARK_ITEMS_SHA256 = (
    "0ca49e7bc94178300c0309919d63a51c0496d84d6e352501d350f0deac494032"
)
PHASE_FILE_STEMS = {
    "phase1_foundation": "phase1",
    "phase2_application": "phase2",
    "phase3_synthesis": "phase3",
}
ANSWER_PERSPECTIVE_CUE_RE = re.compile(r"한국(?:의)?\s*(?:관점|입장)(?:에서|으로)")
REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_path(path: Path) -> str:
    """Use repository-relative paths in portable experiment manifests."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def source_url(row: dict[str, Any]) -> str:
    return normalized_text(row.get("normalized_url") or row.get("url"))


def source_content_hash(row: dict[str, Any]) -> str:
    return normalized_text(row.get("content_sha256"))


def select_proportional_dev_issues(
    issue_domains: dict[str, str], requested: int,
) -> set[str]:
    """Select a deterministic proportional dev set without emptying a domain."""
    if requested < 1 or requested >= len(issue_domains):
        raise ValueError("dev issue count must be between 1 and issue_count - 1")
    by_domain: dict[str, list[str]] = defaultdict(list)
    for issue_id, domain_id in issue_domains.items():
        by_domain[domain_id].append(issue_id)
    total = len(issue_domains)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for domain_id, values in sorted(by_domain.items()):
        exact = requested * len(values) / total
        maximum = max(0, len(values) - 1)
        allocation = min(int(exact), maximum)
        allocations[domain_id] = allocation
        remainders.append((exact - int(exact), domain_id))
    remaining = requested - sum(allocations.values())
    while remaining:
        eligible = [
            (remainder, domain_id)
            for remainder, domain_id in remainders
            if allocations[domain_id] < len(by_domain[domain_id]) - 1
        ]
        if not eligible:
            raise ValueError("cannot allocate dev issues without emptying a domain")
        ordered = sorted(
            eligible,
            key=lambda row: (
                row[0],
                len(by_domain[row[1]]) - allocations[row[1]],
                stable_key("dev-domain", row[1]),
            ),
            reverse=True,
        )
        for _, selected_domain in ordered:
            allocations[selected_domain] += 1
            remaining -= 1
            if not remaining:
                break
    selected: set[str] = set()
    for domain_id, count in sorted(allocations.items()):
        ordered = sorted(
            by_domain[domain_id], key=lambda value: stable_key("dev-issue", value)
        )
        selected.update(ordered[:count])
    if len(selected) != requested:
        raise AssertionError("dev allocation count mismatch")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--dossiers", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--benchmark-items", type=Path, required=True)
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route", default="training_candidate")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--expected-issues", type=int, required=True)
    parser.add_argument("--expected-examples", type=int, required=True)
    parser.add_argument("--dev-issues", type=int, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {args.output_dir}"
        )

    current_files = sorted(path for path in args.current_dir.iterdir() if path.is_file())
    current_hashes_before = {path.name: sha256(path) for path in current_files}
    benchmark_hash = sha256(args.benchmark_items)
    if benchmark_hash != EXPECTED_BENCHMARK_ITEMS_SHA256:
        raise ValueError(f"unexpected frozen benchmark hash: {benchmark_hash}")

    recommendations = read_jsonl(args.recommendations)
    selected_recommendations = {
        str(row["issue_id"]): row
        for row in recommendations
        if row.get("recommended_route") == args.route
    }
    if len(selected_recommendations) != args.expected_issues:
        raise ValueError(
            f"route issue count mismatch: {len(selected_recommendations)} "
            f"!= {args.expected_issues}"
        )
    dossiers = {str(row["issue_id"]): row for row in read_jsonl(args.dossiers)}
    missing_dossiers = sorted(set(selected_recommendations) - set(dossiers))
    if missing_dossiers:
        raise ValueError(f"missing dossiers: {missing_dossiers[:5]}")

    accepted = [
        row
        for row in read_jsonl(args.accepted)
        if str(row.get("issue_id")) in selected_recommendations
    ]
    if len(accepted) != args.expected_examples:
        raise ValueError(
            f"accepted example count mismatch: {len(accepted)} "
            f"!= {args.expected_examples}"
        )
    candidate_ids = [str(row.get("candidate_id") or "") for row in accepted]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs are missing or duplicated")

    issue_domains: dict[str, str] = {}
    for issue_id, recommendation in selected_recommendations.items():
        domain_id = str(recommendation.get("domain_id") or "")
        dossier_domain = str(
            dossiers[issue_id]["verified_issue_card"]["issue_identity"]["domain_id"]
        )
        if not domain_id or domain_id != dossier_domain:
            raise ValueError(f"domain mismatch for {issue_id}")
        issue_domains[issue_id] = domain_id
    dev_issue_ids = select_proportional_dev_issues(issue_domains, args.dev_issues)

    source_rows = {str(row["source_id"]): row for row in read_jsonl(args.sources)}
    benchmark_items = read_jsonl(args.benchmark_items)
    if len(benchmark_items) != 480:
        raise ValueError("frozen benchmark must contain exactly 480 items")
    benchmark_issue_ids = {str(row["issue_id"]) for row in benchmark_items}
    benchmark_source_ids = {
        str(source_id)
        for row in benchmark_items
        for source_id in row.get("source_ids") or []
    }
    training_source_ids = {
        str(source_id)
        for issue_id in selected_recommendations
        for source_id in dossiers[issue_id].get("source_ids") or []
    }
    missing_sources = sorted((benchmark_source_ids | training_source_ids) - set(source_rows))
    if missing_sources:
        raise ValueError(f"source registry is incomplete: {missing_sources[:5]}")

    benchmark_urls = {
        source_url(source_rows[source_id])
        for source_id in benchmark_source_ids
        if source_url(source_rows[source_id])
    }
    training_urls = {
        source_url(source_rows[source_id])
        for source_id in training_source_ids
        if source_url(source_rows[source_id])
    }
    benchmark_content_hashes = {
        source_content_hash(source_rows[source_id])
        for source_id in benchmark_source_ids
        if source_content_hash(source_rows[source_id])
    }
    training_content_hashes = {
        source_content_hash(source_rows[source_id])
        for source_id in training_source_ids
        if source_content_hash(source_rows[source_id])
    }
    isolation_overlaps = {
        "issue_ids": sorted(set(selected_recommendations) & benchmark_issue_ids),
        "source_ids": sorted(training_source_ids & benchmark_source_ids),
        "normalized_urls": sorted(training_urls & benchmark_urls),
        "content_sha256": sorted(training_content_hashes & benchmark_content_hashes),
    }
    if any(isolation_overlaps.values()):
        raise ValueError(f"training/benchmark isolation failure: {isolation_overlaps}")

    training_rows: list[dict[str, Any]] = []
    for candidate in accepted:
        issue_id = str(candidate["issue_id"])
        dossier = dossiers[issue_id]
        phase = str(candidate.get("curriculum_phase") or "")
        user = str(candidate.get("user_ko") or "").strip()
        assistant = str(candidate.get("assistant_ko") or "").strip()
        validation = candidate.get("validation") or {}
        checks = validation.get("checks") or {}
        if phase not in PHASE_FILE_STEMS:
            raise ValueError(f"invalid phase for {candidate['candidate_id']}: {phase}")
        if not user or not assistant:
            raise ValueError(f"empty message for {candidate['candidate_id']}")
        if ANSWER_PERSPECTIVE_CUE_RE.search(user):
            raise ValueError(f"answer-perspective cue for {candidate['candidate_id']}")
        if (
            validation.get("verdict") != "pass"
            or validation.get("llm_verdict") != "pass"
            or validation.get("deterministic_failures")
            or not checks
            or not all(value is True for value in checks.values())
        ):
            raise ValueError(f"unvalidated candidate: {candidate['candidate_id']}")
        if str(candidate.get("input_sha256")) != str(dossier.get("input_sha256")):
            raise ValueError(f"dossier hash mismatch for {candidate['candidate_id']}")
        training_rows.append({
            "schema_version": "ksova-data-train-example-1.0",
            "experiment_memberships": [args.experiment_id],
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": {
                "training_id": str(candidate["candidate_id"]),
                "issue_id": issue_id,
                "source_ids": list(dossier["source_ids"]),
                "category_id": issue_domains[issue_id],
                "origin": "validated_recommended_candidate_20260813",
                "task": str(candidate["task"]),
                "task_group": str(candidate["task_group"]),
                "capability_stage": int(candidate["capability_stage"]),
                "curriculum_phase": phase,
                "split": "dev" if issue_id in dev_issue_ids else "train",
                "source_isolated_from_frozen_benchmark_480": True,
                "input_sha256": str(dossier["input_sha256"]),
                "generation": candidate["generation"],
                "validation": validation,
            },
        })
    training_rows.sort(
        key=lambda row: (
            row["metadata"]["issue_id"],
            row["metadata"]["capability_stage"],
            row["metadata"]["task"],
        )
    )

    training_ids = [row["metadata"]["training_id"] for row in training_rows]
    users = [normalized_text(row["messages"][0]["content"]) for row in training_rows]
    pairs = [
        (normalized_text(row["messages"][0]["content"]),
         normalized_text(row["messages"][1]["content"]))
        for row in training_rows
    ]
    if len(training_ids) != len(set(training_ids)):
        raise ValueError("materialized training IDs are duplicated")
    if len(users) != len(set(users)):
        raise ValueError("materialized user prompts are duplicated")
    if len(pairs) != len(set(pairs)):
        raise ValueError("materialized message pairs are duplicated")
    benchmark_questions = {
        normalized_text(row.get("question_ko")) for row in benchmark_items
    }
    exact_prompt_overlap = sorted(set(users) & benchmark_questions)
    if exact_prompt_overlap:
        raise ValueError("exact training/benchmark prompt overlap")

    train_rows = [row for row in training_rows if row["metadata"]["split"] == "train"]
    dev_rows = [row for row in training_rows if row["metadata"]["split"] == "dev"]
    outputs: dict[str, list[dict[str, Any]]] = {
        "all_examples.jsonl": training_rows,
        "train_all.jsonl": train_rows,
        "dev_all.jsonl": dev_rows,
    }
    for phase, stem in PHASE_FILE_STEMS.items():
        outputs[f"{stem}_train.jsonl"] = [
            row for row in train_rows
            if row["metadata"]["curriculum_phase"] == phase
        ]
        outputs[f"{stem}_dev.jsonl"] = [
            row for row in dev_rows
            if row["metadata"]["curriculum_phase"] == phase
        ]
    for name, rows in outputs.items():
        if not rows:
            raise ValueError(f"empty required output: {name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_jsonl(args.output_dir / name, rows)

    current_hashes_after = {path.name: sha256(path) for path in current_files}
    if current_hashes_after != current_hashes_before:
        raise RuntimeError("data/current changed during materialization")
    split_issue_overlap = {
        row["metadata"]["issue_id"] for row in train_rows
    } & {
        row["metadata"]["issue_id"] for row in dev_rows
    }
    if split_issue_overlap:
        raise RuntimeError("train/dev issue overlap")

    report = {
        "schema_version": "ksova-materialized-training-experiment-1.0",
        "experiment_id": args.experiment_id,
        "portfolio_route": args.route,
        "candidate_issues": len(selected_recommendations),
        "examples": len(training_rows),
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "train_issues": len({row["metadata"]["issue_id"] for row in train_rows}),
        "dev_issues": len(dev_issue_ids),
        "dev_issue_ids": sorted(dev_issue_ids),
        "domain_issue_counts": dict(sorted(Counter(issue_domains.values()).items())),
        "dev_domain_issue_counts": dict(sorted(Counter(
            issue_domains[issue_id] for issue_id in dev_issue_ids
        ).items())),
        "task_counts": dict(sorted(Counter(
            row["metadata"]["task"] for row in training_rows
        ).items())),
        "phase_counts": dict(sorted(Counter(
            row["metadata"]["curriculum_phase"] for row in training_rows
        ).items())),
        "phase_split_counts": {
            phase: {
                split: sum(
                    row["metadata"]["curriculum_phase"] == phase
                    and row["metadata"]["split"] == split
                    for row in training_rows
                )
                for split in ("train", "dev")
            }
            for phase in PHASE_FILE_STEMS
        },
        "benchmark": {
            "items": len(benchmark_items),
            "issues": len(benchmark_issue_ids),
            "items_sha256": benchmark_hash,
        },
        "isolation_audit": {
            "issue_id_overlap": 0,
            "source_id_overlap": 0,
            "normalized_url_overlap": 0,
            "content_sha256_overlap": 0,
            "exact_prompt_overlap": 0,
            "train_dev_issue_overlap": 0,
        },
        "input_files": {
            "accepted": {"path": report_path(args.accepted), "sha256": sha256(args.accepted)},
            "recommendations": {
                "path": report_path(args.recommendations),
                "sha256": sha256(args.recommendations),
            },
            "dossiers": {"path": report_path(args.dossiers), "sha256": sha256(args.dossiers)},
            "sources": {"path": report_path(args.sources), "sha256": sha256(args.sources)},
            "benchmark_items": {
                "path": report_path(args.benchmark_items),
                "sha256": benchmark_hash,
            },
        },
        "output_files": {
            name: {"rows": len(rows), "sha256": sha256(args.output_dir / name)}
            for name, rows in sorted(outputs.items())
        },
        "data_current_hashes_before": current_hashes_before,
        "data_current_hashes_after": current_hashes_after,
        "data_current_mutated": False,
    }
    write_json(args.output_dir / "materialization_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
