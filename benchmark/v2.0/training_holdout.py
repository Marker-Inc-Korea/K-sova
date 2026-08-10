#!/usr/bin/env python3
"""Build and audit benchmark-vs-training semantic holdout records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_QUERY_KEYS = {"fbclid", "gclid"}
DOCUMENT_ID_QUERY_KEYS = {
    "admrulseq", "arti_id", "articleid", "artiid", "artclno", "bcidx", "bbsid",
    "bbsseq", "bbsseqno", "boardid", "boardmasterid", "brdbltno", "brdscnbltno",
    "cbidx", "contentid", "detcseq", "docseq", "enc", "eventno", "fileid", "fileseq", "id",
    "idx", "idxno", "jono", "levelid", "list_no", "listno", "lsid", "lsiseq",
    "mst", "ncd", "newsid", "newsitemid", "no", "nttid", "nttseqno", "precseq",
    "seq", "serearticlesearchbean.artiid", "sno",
}
PATH_SESSION = re.compile(r";jsessionid=[^/?;]+", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_url(url: str) -> str:
    parsed = urlsplit(url or "")
    query = [
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if value
        and not key.casefold().startswith("utm_")
        and key.casefold() not in TRACKING_QUERY_KEYS
        and key.casefold() not in {"jsessionid", "sessionid"}
    ]
    identity_query = [
        (key, value) for key, value in query
        if key.casefold() in DOCUMENT_ID_QUERY_KEYS
    ]
    if identity_query:
        query = identity_query
    return urlunsplit((
        parsed.scheme.casefold(), parsed.netloc.casefold().removeprefix("www."),
        PATH_SESSION.sub("", parsed.path).rstrip("/"), urlencode(sorted(query)), "",
    ))


def source_keys(source_ids: list[str], registry: dict[str, dict[str, Any]]) -> list[str]:
    keys = []
    for source_id in source_ids:
        source = registry.get(source_id)
        if source is None:
            raise ValueError(f"unknown source_id: {source_id}")
        url = normalized_url(str(source.get("url") or ""))
        content_hash = str(source.get("content_sha256") or "")
        if url:
            keys.append("url:" + url)
        if content_hash:
            keys.append("sha256:" + content_hash)
    return sorted(set(keys))


def material_is_machine_accepted(material: dict[str, Any]) -> bool:
    audit = material.get("pipeline_material_audit") or {}
    policy = audit.get("policy")
    dual_pass = all((
        material.get("status") == "material_accepted",
        policy == "unanimous_two_model_single_round",
        (audit.get("qwen") or {}).get("effective_pass"),
        (audit.get("gpt_oss") or {}).get("effective_pass"),
    ))
    glm_pass = all((
        material.get("status") == "material_accepted_glm_pass",
        policy == "single_glm_single_round_no_correction",
        audit.get("validator") == "glm",
        (audit.get("glm") or {}).get("effective_pass"),
    ))
    return dual_pass or glm_pass


def command_build(args: argparse.Namespace) -> int:
    issues = read_jsonl(args.issues)
    materials = read_jsonl(args.materials)
    registry: dict[str, dict[str, Any]] = {}
    equivalent_source_duplicates = 0
    for path in args.sources:
        for wrapper in read_jsonl(path):
            row = wrapper.get("source") if isinstance(wrapper.get("source"), dict) else wrapper
            source_id = str(row.get("source_id") or "")
            if not source_id:
                raise ValueError(f"source registry row is missing source_id: {path}")
            if source_id in registry:
                previous = registry[source_id]
                previous_url = normalized_url(str(previous.get("url") or ""))
                current_url = normalized_url(str(row.get("url") or ""))
                if not previous_url or previous_url != current_url:
                    raise ValueError(f"conflicting source registry URL: {source_id}")
                previous_hash = str(previous.get("content_sha256") or "")
                current_hash = str(row.get("content_sha256") or "")
                if previous_hash and current_hash and previous_hash != current_hash:
                    raise ValueError(f"conflicting source registry content hash: {source_id}")
                # The canonical registry is richer than the nested discovery wrapper.
                # Preserve earlier canonical fields and fill only fields it lacks.
                registry[source_id] = {**row, **previous}
                equivalent_source_duplicates += 1
            else:
                registry[source_id] = row
    records = []
    for issue in issues:
        parts = [
            issue.get("title_ko"), issue.get("decision_core_ko"),
            issue.get("sovereign_stake_ko"), issue.get("korean_reference_direction_ko"),
            issue.get("novelty_key_ko"),
        ]
        records.append({
            "schema_version": "ksova-training-holdout-record-1.0",
            "record_id": "benchmark__" + str(issue["issue_id"]),
            "role": "benchmark_holdout",
            "entity_id": issue["issue_id"],
            "domain_or_lane": issue.get("domain_id"),
            "source_ids": issue.get("source_ids") or [],
            "source_keys": source_keys(issue.get("source_ids") or [], registry),
            "text": "\n".join(str(value).strip() for value in parts if value),
        })
    for material in materials:
        if not material_is_machine_accepted(material):
            raise ValueError(
                f"material input lacks a recognized passing audit: {material.get('material_id')}"
            )
        parts = [
            material.get("specific_issue_ko"), material.get("decision_object_ko"),
            material.get("korean_stake_ko"), material.get("minimum_baseline_ko"),
            material.get("why_ksova_ko"), material.get("novelty_from_existing_ko"),
        ]
        records.append({
            "schema_version": "ksova-training-holdout-record-1.0",
            "record_id": "material__" + str(material["material_id"]),
            "role": "training_material_candidate",
            "entity_id": material["material_id"],
            "domain_or_lane": material.get("lane"),
            "source_ids": material.get("source_ids") or [],
            "source_keys": source_keys(material.get("source_ids") or [], registry),
            "text": "\n".join(str(value).strip() for value in parts if value),
        })
    record_ids = [row["record_id"] for row in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate holdout record_id")
    write_jsonl(args.output, records)
    report = {
        "schema_version": "ksova-training-holdout-input-report-1.0",
        "benchmark_issues": len(issues),
        "accepted_materials": len(materials),
        "source_registry_records": len(registry),
        "equivalent_source_duplicates": equivalent_source_duplicates,
        "records": len(records),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def terms(text: str) -> set[str]:
    return {value.casefold() for value in re.findall(r"[0-9A-Za-z가-힣]{2,}", text)}


def percentile(values: Any, quantile: float) -> float | None:
    if len(values) == 0:
        return None
    import numpy as np
    return round(float(np.quantile(values, quantile)), 6)


def command_analyze(args: argparse.Namespace) -> int:
    import numpy as np

    records = read_jsonl(args.records)
    record_by_id = {str(row["record_id"]): row for row in records}
    embeddings = read_jsonl(args.embeddings)
    embedding_by_id = {str(row["record_id"]): row for row in embeddings}
    if set(embedding_by_id) != set(record_by_id):
        raise ValueError("embedding coverage does not match holdout records")
    benchmark = [row for row in records if row["role"] == "benchmark_holdout"]
    materials = [row for row in records if row["role"] == "training_material_candidate"]
    if not benchmark or not materials:
        raise ValueError("holdout analysis requires benchmark and material records")
    benchmark_matrix = np.asarray([
        embedding_by_id[row["record_id"]]["embedding"] for row in benchmark
    ], dtype=np.float32)
    material_matrix = np.asarray([
        embedding_by_id[row["record_id"]]["embedding"] for row in materials
    ], dtype=np.float32)
    if benchmark_matrix.ndim != 2 or material_matrix.shape[1] != benchmark_matrix.shape[1]:
        raise ValueError("embedding dimensions are inconsistent")
    similarities = material_matrix @ benchmark_matrix.T
    benchmark_terms = [terms(row["text"]) for row in benchmark]
    benchmark_source_keys = {key for row in benchmark for key in row["source_keys"]}
    pairs = []
    cosine_values = []
    lexical_values = []
    source_collisions = 0
    for material_index, material in enumerate(materials):
        cosine_index = int(np.argmax(similarities[material_index]))
        cosine_score = float(similarities[material_index, cosine_index])
        material_terms = terms(material["text"])
        lexical_scores = [
            len(material_terms & value) / max(1, len(material_terms | value))
            for value in benchmark_terms
        ]
        lexical_index = max(range(len(lexical_scores)), key=lexical_scores.__getitem__)
        lexical_score = lexical_scores[lexical_index]
        shared_sources = sorted(
            set(material["source_keys"]) & benchmark_source_keys
        )
        source_collisions += int(bool(shared_sources))
        cosine_values.append(cosine_score)
        lexical_values.append(lexical_score)
        pairs.append({
            "schema_version": "ksova-training-holdout-nearest-pair-1.0",
            "material_id": material["entity_id"],
            "material_lane": material.get("domain_or_lane"),
            "cosine_nearest_issue_id": benchmark[cosine_index]["entity_id"],
            "cosine_similarity": round(cosine_score, 6),
            "lexical_nearest_issue_id": benchmark[lexical_index]["entity_id"],
            "lexical_jaccard": round(lexical_score, 6),
            "shared_source_keys": shared_sources,
        })
    pairs.sort(key=lambda row: (
        -int(bool(row["shared_source_keys"])),
        -row["cosine_similarity"], -row["lexical_jaccard"], row["material_id"],
    ))
    write_jsonl(args.pairs, pairs)
    report = {
        "schema_version": "ksova-training-holdout-analysis-report-1.0",
        "benchmark_issues": len(benchmark),
        "material_candidates": len(materials),
        "embedding_dimensions": int(benchmark_matrix.shape[1]),
        "source_collision_materials": source_collisions,
        "cosine_nearest_distribution": {
            "min": round(min(cosine_values), 6),
            "p25": percentile(cosine_values, 0.25),
            "median": percentile(cosine_values, 0.5),
            "p75": percentile(cosine_values, 0.75),
            "p90": percentile(cosine_values, 0.9),
            "p95": percentile(cosine_values, 0.95),
            "max": round(max(cosine_values), 6),
        },
        "lexical_nearest_distribution": {
            "min": round(min(lexical_values), 6),
            "median": percentile(lexical_values, 0.5),
            "p90": percentile(lexical_values, 0.9),
            "p95": percentile(lexical_values, 0.95),
            "max": round(max(lexical_values), 6),
        },
        "thresholds": "not_set_distribution_audit_only",
        "records_sha256": sha256(args.records),
        "embeddings_sha256": sha256(args.embeddings),
        "pairs_path": str(args.pairs.resolve()),
        "pairs_sha256": sha256(args.pairs),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-input")
    build.add_argument("--issues", type=Path, required=True)
    build.add_argument("--materials", type=Path, required=True)
    build.add_argument("--sources", type=Path, action="append", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.set_defaults(func=command_build)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--records", type=Path, required=True)
    analyze.add_argument("--embeddings", type=Path, required=True)
    analyze.add_argument("--pairs", type=Path, required=True)
    analyze.add_argument("--report", type=Path, required=True)
    analyze.set_defaults(func=command_analyze)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
