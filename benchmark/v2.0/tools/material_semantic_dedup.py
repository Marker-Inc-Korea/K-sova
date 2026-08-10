#!/usr/bin/env python3
"""Route embedding-nearest material pairs to one fail-closed GLM dedup audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RELATIONS = {"semantic_duplicate", "related_but_distinct", "unrelated"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def compact_issue(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "benchmark_holdout",
        "entity_id": row["issue_id"],
        "issue_ko": row.get("title_ko"),
        "decision_ko": row.get("decision_core_ko"),
        "minimum_baseline_ko": row.get("korean_reference_direction_ko"),
        "stake_ko": row.get("sovereign_stake_ko"),
    }


def compact_material(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "training_material_candidate",
        "entity_id": row["material_id"],
        "issue_ko": row.get("specific_issue_ko"),
        "decision_ko": row.get("decision_object_ko"),
        "minimum_baseline_ko": row.get("minimum_baseline_ko"),
        "stake_ko": row.get("korean_stake_ko"),
    }


def prompt(pairs: list[dict[str, Any]]) -> str:
    contract = {
        "judgments": [{
            "pair_id": value["pair_id"],
            "relation": "semantic_duplicate | related_but_distinct | unrelated",
            "same_required_judgment": False,
            "same_minimum_baseline": False,
            "same_decisive_basis": False,
            "same_practical_consequence": False,
            "preferred_id": "semantic_duplicate면 보존할 entity_id, 아니면 both",
            "reason_ko": "구체적 판정 근거",
        } for value in pairs],
    }
    return f"""다음 K-SOVA 벤치마크-학습소재 또는 학습소재-학습소재 쌍의 의미 중복을 판정하라.

[후보 쌍]
{json.dumps(pairs, ensure_ascii=False, indent=2)}

[판정 규칙]
1. semantic_duplicate는 요구 판단, 한국의 최소 결론, 결정적 근거, 실제 결과가 모두 실질적으로 같아
   한 답변을 거의 그대로 다른 질문에도 쓸 수 있을 때만 선택한다. 이 경우 네 same_* 값이 모두 true여야 한다.
2. 같은 사건·조약·원천·분야라도 책임 주체, 권리·의무, 적용 시점, 비용 귀속, 정책 수단 또는 실제 결과가
   달라 별도 논증이 필요하면 related_but_distinct다. 넓은 분야만 같으면 unrelated다.
3. benchmark_holdout과 training_material_candidate가 중복이면 benchmark_holdout의 entity_id를 preferred_id로 쓴다.
4. 신규 소재끼리 중복이면 더 원자적이고, 판단 대상과 최소 결론이 구체적이며, 하나의 답으로 검증하기 쉬운
   소재 하나의 entity_id를 preferred_id로 쓴다. 중복이 아니면 preferred_id는 both다.
5. 표현이 비슷하다는 이유만으로 중복 처리하지 말고, 표현이 달라도 같은 판단이면 중복 처리한다.
6. 입력 쌍을 빠짐없이 정확히 한 번씩 반환하며 후보를 수정하거나 합치지 않는다. JSON 객체 하나만 출력한다.

[출력 계약]
{json.dumps(contract, ensure_ascii=False, indent=2)}"""


def command_build(args: argparse.Namespace) -> int:
    issues = read_jsonl(args.issues)
    materials = read_jsonl(args.materials)
    embeddings = {row["record_id"]: row["embedding"] for row in read_jsonl(args.embeddings)}
    issue_by_id = {row["issue_id"]: row for row in issues}
    material_by_id = {row["material_id"]: row for row in materials}
    issue_vectors = {
        issue_id: embeddings[f"benchmark__{issue_id}"] for issue_id in issue_by_id
    }
    material_vectors = {
        material_id: embeddings[f"material__{material_id}"] for material_id in material_by_id
    }

    pairs: dict[str, dict[str, Any]] = {}
    for material_id, vector in material_vectors.items():
        nearest = sorted(
            ((dot(vector, other), issue_id) for issue_id, other in issue_vectors.items()),
            reverse=True,
        )[: args.nearest_benchmark]
        for score, issue_id in nearest:
            pair_id = f"BM__{material_id}__{issue_id}"
            pairs[pair_id] = {
                "pair_id": pair_id,
                "routing": "embedding_nearest_benchmark",
                "cosine_similarity": round(score, 6),
                "entity_a": compact_material(material_by_id[material_id]),
                "entity_b": compact_issue(issue_by_id[issue_id]),
            }

    for material_id, vector in material_vectors.items():
        nearest = sorted(
            (
                (dot(vector, other), other_id)
                for other_id, other in material_vectors.items() if other_id != material_id
            ),
            reverse=True,
        )[: args.nearest_material]
        for score, other_id in nearest:
            left_id, right_id = sorted((material_id, other_id))
            pair_id = f"MM__{left_id}__{right_id}"
            pairs[pair_id] = {
                "pair_id": pair_id,
                "routing": "embedding_nearest_material",
                "cosine_similarity": round(score, 6),
                "entity_a": compact_material(material_by_id[left_id]),
                "entity_b": compact_material(material_by_id[right_id]),
            }

    pair_rows = sorted(pairs.values(), key=lambda row: row["pair_id"])
    requests = []
    for start in range(0, len(pair_rows), args.batch_size):
        batch = pair_rows[start:start + args.batch_size]
        requests.append({
            "id": f"material_semantic_dedup__{start // args.batch_size:04d}",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict K-SOVA semantic leakage auditor. "
                        "Judge duplicates without rewriting candidates."
                    ),
                },
                {"role": "user", "content": prompt(batch)},
            ],
            "max_tokens": max(1600, 480 * len(batch)),
            "metadata": {
                "purpose": "ksova_training_material_semantic_dedup",
                "pair_ids": [row["pair_id"] for row in batch],
                "single_round": True,
            },
        })
    write_jsonl(args.pairs, pair_rows)
    write_jsonl(args.output, requests)
    report = {
        "schema_version": "ksova-material-semantic-dedup-plan-1.0",
        "benchmark_issues": len(issues),
        "materials": len(materials),
        "nearest_benchmark_per_material": args.nearest_benchmark,
        "nearest_material_per_material": args.nearest_material,
        "similarity_threshold": "not_set_routing_only",
        "candidate_pairs": len(pair_rows),
        "benchmark_material_pairs": sum(row["pair_id"].startswith("BM__") for row in pair_rows),
        "material_material_pairs": sum(row["pair_id"].startswith("MM__") for row in pair_rows),
        "requests": len(requests),
        "pairs": str(args.pairs),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def parse_judgment(value: Any, pair: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, dict) or value.get("pair_id") != pair["pair_id"]:
        return None, "invalid_pair_id_or_object"
    relation = value.get("relation")
    if relation not in RELATIONS:
        return None, "invalid_relation"
    flags = (
        "same_required_judgment", "same_minimum_baseline",
        "same_decisive_basis", "same_practical_consequence",
    )
    if any(not isinstance(value.get(key), bool) for key in flags):
        return None, "invalid_same_flags"
    duplicate = relation == "semantic_duplicate"
    if duplicate != all(value[key] for key in flags):
        return None, "relation_flag_inconsistency"
    entity_ids = {pair["entity_a"]["entity_id"], pair["entity_b"]["entity_id"]}
    preferred = value.get("preferred_id")
    if duplicate and preferred not in entity_ids:
        return None, "invalid_duplicate_preferred_id"
    if not duplicate and preferred != "both":
        return None, "invalid_distinct_preferred_id"
    if not isinstance(value.get("reason_ko"), str) or len(value["reason_ko"].strip()) < 20:
        return None, "invalid_reason"
    return value, None


def command_parse(args: argparse.Namespace) -> int:
    materials = read_jsonl(args.materials)
    material_by_id = {row["material_id"]: row for row in materials}
    pairs = {row["pair_id"]: row for row in read_jsonl(args.pairs)}
    requests = read_jsonl(args.requests)
    responses = read_jsonl(args.responses)
    counts = Counter(str(row.get("id") or "") for row in responses)
    response_by_id = {str(row.get("id") or ""): row for row in responses}
    judgments = []
    quarantine_reasons: dict[str, list[dict[str, Any]]] = {}
    invalid_pairs = Counter()

    def quarantine(material_id: str, reason: dict[str, Any]) -> None:
        if material_id in material_by_id:
            quarantine_reasons.setdefault(material_id, []).append(reason)

    for request in requests:
        expected = list(request["metadata"]["pair_ids"])
        response = response_by_id.get(request["id"])
        if (
            response is None or counts[request["id"]] != 1 or response.get("error")
            or response.get("finish_reason") != "stop"
        ):
            for pair_id in expected:
                invalid_pairs["missing_duplicate_runtime_or_nonstop"] += 1
                for entity in (pairs[pair_id]["entity_a"], pairs[pair_id]["entity_b"]):
                    quarantine(entity["entity_id"], {"pair_id": pair_id, "reason": "audit_incomplete"})
            continue
        try:
            payload = json.loads(str(response.get("response") or ""))
        except json.JSONDecodeError:
            payload = None
        values = payload.get("judgments") if isinstance(payload, dict) else None
        value_ids = [
            row.get("pair_id") for row in values or []
            if isinstance(row, dict) and isinstance(row.get("pair_id"), str)
        ]
        if (
            not isinstance(values, list)
            or len(values) != len(expected)
            or Counter(value_ids) != Counter(expected)
        ):
            for pair_id in expected:
                invalid_pairs["judgment_coverage_mismatch"] += 1
                for entity in (pairs[pair_id]["entity_a"], pairs[pair_id]["entity_b"]):
                    quarantine(entity["entity_id"], {
                        "pair_id": pair_id,
                        "reason": "audit_incomplete",
                        "detail": "judgment_coverage_mismatch",
                    })
            continue
        by_pair = {
            row.get("pair_id"): row for row in values or []
            if isinstance(row, dict) and isinstance(row.get("pair_id"), str)
        }
        for pair_id in expected:
            judgment, error = parse_judgment(by_pair.get(pair_id), pairs[pair_id])
            if error:
                invalid_pairs[error] += 1
                for entity in (pairs[pair_id]["entity_a"], pairs[pair_id]["entity_b"]):
                    quarantine(entity["entity_id"], {"pair_id": pair_id, "reason": "audit_incomplete", "detail": error})
                continue
            judgments.append({
                **pairs[pair_id],
                "judgment": judgment,
                "validator_model": response.get("model"),
                "messages_sha256": response.get("messages_sha256"),
            })
            if judgment["relation"] != "semantic_duplicate":
                continue
            for entity in (pairs[pair_id]["entity_a"], pairs[pair_id]["entity_b"]):
                if entity["kind"] == "training_material_candidate" and entity["entity_id"] != judgment["preferred_id"]:
                    quarantine(entity["entity_id"], {
                        "pair_id": pair_id,
                        "reason": "semantic_duplicate",
                        "preferred_id": judgment["preferred_id"],
                    })

    kept = [row for row in materials if row["material_id"] not in quarantine_reasons]
    quarantined = [{
        **material_by_id[material_id],
        "status": "material_quarantined_semantic_dedup",
        "semantic_dedup_quarantine": reasons,
    } for material_id, reasons in sorted(quarantine_reasons.items())]
    write_jsonl(args.judgments, judgments)
    write_jsonl(args.kept, kept)
    write_jsonl(args.quarantine, quarantined)
    report = {
        "schema_version": "ksova-material-semantic-dedup-report-1.0",
        "materials": len(materials),
        "candidate_pairs": len(pairs),
        "requests": len(requests),
        "responses": len(responses),
        "valid_judgments": len(judgments),
        "semantic_duplicates": sum(
            row["judgment"]["relation"] == "semantic_duplicate" for row in judgments
        ),
        "invalid_pairs": sum(invalid_pairs.values()),
        "invalid_pair_reasons": dict(sorted(invalid_pairs.items())),
        "kept_materials": len(kept),
        "quarantined_materials": len(quarantined),
        "policy": "embedding_routes_without_threshold_then_single_glm_dedup_no_repair",
        "kept": str(args.kept),
        "quarantine": str(args.quarantine),
        "judgments": str(args.judgments),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--issues", type=Path, required=True)
    build.add_argument("--materials", type=Path, required=True)
    build.add_argument("--embeddings", type=Path, required=True)
    build.add_argument("--nearest-benchmark", type=int, default=3)
    build.add_argument("--nearest-material", type=int, default=3)
    build.add_argument("--batch-size", type=int, default=6)
    build.add_argument("--pairs", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.set_defaults(func=command_build)
    parse = commands.add_parser("parse")
    parse.add_argument("--materials", type=Path, required=True)
    parse.add_argument("--pairs", type=Path, required=True)
    parse.add_argument("--requests", type=Path, required=True)
    parse.add_argument("--responses", type=Path, required=True)
    parse.add_argument("--judgments", type=Path, required=True)
    parse.add_argument("--kept", type=Path, required=True)
    parse.add_argument("--quarantine", type=Path, required=True)
    parse.add_argument("--report", type=Path, required=True)
    parse.set_defaults(func=command_parse)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
