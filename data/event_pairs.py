#!/usr/bin/env python3
"""Propose cross-source event pairs for a separate same-event audit.

Embedding similarity is only a retrieval signal.  This tool never merges
events or promotes issues; it emits compact pairs whose fetched source bodies
must be compared by a later source-closed validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def base_host(url: str) -> str:
    host = (urllib.parse.urlsplit(url).hostname or "").casefold()
    return host[4:] if host.startswith("www.") else host


def event_text(row: dict[str, Any]) -> str:
    return "passage: " + " ".join((
        (str(row.get("title_ko") or "") + " ") * 2,
        str(row.get("decision_object_ko") or ""),
        str(row.get("external_actor_or_rule") or ""),
        str(row.get("korean_stake_ko") or ""),
        " ".join(str(value) for value in row.get("anchor_terms") or []),
        str(row.get("temporal_scope") or ""),
    )).strip()


def view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "origin_run": row["origin_run"],
        "title_ko": row.get("title_ko"),
        "decision_object_ko": row.get("decision_object_ko"),
        "external_actor_or_rule": row.get("external_actor_or_rule"),
        "korean_stake_ko": row.get("korean_stake_ko"),
        "temporal_scope": row.get("temporal_scope"),
        "anchor_terms": row.get("anchor_terms") or [],
        "seed_source_ids": row.get("seed_source_ids") or [],
        "source_hosts": row["source_hosts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument(
        "--focus-run", action="append", default=[],
        help=(
            "emit nearest pairs for events from the named newly added run; "
            "repeatable and independent of source-run ordering"
        ),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-similarity", type=float, default=0.82)
    parser.add_argument("--top-k-per-event", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("event pairing requires torch and transformers") from exc

    events = []
    source_index: dict[tuple[str, str], dict[str, Any]] = {}
    for source_run_arg in args.source_run:
        source_run = source_run_arg.resolve()
        run_name = source_run.name
        for source in read_jsonl(source_run / "sources.jsonl"):
            source_index[(run_name, str(source.get("source_id") or ""))] = source
        for event in read_jsonl(source_run / "event_leads.jsonl"):
            row = dict(event)
            row["origin_run"] = run_name
            hosts = sorted({
                base_host(str(source_index[(run_name, source_id)].get("url") or ""))
                for source_id in row.get("seed_source_ids") or []
                if (run_name, source_id) in source_index
            } - {""})
            if not hosts:
                continue
            row["source_hosts"] = hosts
            events.append(row)

    if not events:
        raise RuntimeError("no events with resolvable seed sources")
    available_runs = {row["origin_run"] for row in events}
    focus_runs = set(args.focus_run)
    missing_focus_runs = focus_runs - available_runs
    if missing_focus_runs:
        raise RuntimeError(f"unknown focus runs: {sorted(missing_focus_runs)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model_path, local_files_only=True, dtype=torch.float32
    ).to(args.device).eval()
    texts = [event_text(row) for row in events]
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(
                texts[start:start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(args.device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"][..., None].to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            vectors.append(functional.normalize(pooled, p=2, dim=1).cpu())
    matrix = torch.cat(vectors)

    candidates = []
    if focus_runs:
        anchor_indices = [
            index for index, row in enumerate(events)
            if row["origin_run"] in focus_runs
        ]
    else:
        anchor_indices = list(range(len(events)))
    for left_index in anchor_indices:
        left = events[left_index]
        ranked = []
        left_hosts = set(left["source_hosts"])
        right_indices = (
            (index for index in range(len(events)) if index != left_index)
            if focus_runs
            else range(left_index + 1, len(events))
        )
        for right_index in right_indices:
            right = events[right_index]
            if left.get("domain_id") != right.get("domain_id"):
                continue
            if left_hosts & set(right["source_hosts"]):
                continue
            similarity = float(matrix[left_index] @ matrix[right_index])
            if similarity < args.min_similarity:
                continue
            ranked.append((similarity, right_index))
        ranked.sort(reverse=True)
        for similarity, right_index in ranked[: args.top_k_per_event]:
            if right_index < left_index:
                pair_left, pair_right = events[right_index], left
            else:
                pair_left, pair_right = left, events[right_index]
            identity = "\0".join(sorted((left["event_id"], right["event_id"])))
            candidates.append({
                "schema_version": "ksova-event-pair-candidate-1.0",
                "pair_id": "KSEPAIR-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                "domain_id": pair_left["domain_id"],
                "embedding_similarity": round(similarity, 4),
                "left": view(pair_left),
                "right": view(pair_right),
                "status": "requires_source_closed_same_event_validation",
            })

    unique = {}
    for row in sorted(
        candidates,
        key=lambda value: (-value["embedding_similarity"], value["pair_id"]),
    ):
        unique.setdefault(row["pair_id"], row)
    output = list(unique.values())[: args.max_pairs]
    write_jsonl(args.output, output)
    report = {
        "schema_version": "ksova-event-pair-proposal-report-1.0",
        "source_runs": [path.resolve().name for path in args.source_run],
        "focus_runs": sorted(focus_runs),
        "events": len(events),
        "minimum_similarity": args.min_similarity,
        "candidate_pairs": len(output),
        "by_domain": dict(sorted(Counter(row["domain_id"] for row in output).items())),
        "same_run_pairs": sum(
            row["left"]["origin_run"] == row["right"]["origin_run"] for row in output
        ),
        "cross_run_pairs": sum(
            row["left"]["origin_run"] != row["right"]["origin_run"] for row in output
        ),
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
