#!/usr/bin/env python3
"""Retrieve and adjudicate semantic duplicate issue candidates for K-SOVA v2."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM = """당신은 K-SOVA v2 원자 쟁점의 의미 중복 판정자다.
같은 사건·국가·법·조약·원천을 다룬다는 이유만으로 중복 처리하지 않는다.
책임 주체, 권리 소멸, 배상 의무, 사과, 기억·교육, 구제 방식, 위험 귀속,
정책 선택 또는 실제 결과 중 하나라도 달라 별도 논증이 필요하면 독립 쟁점이다.
표현만 다르고 같은 결론과 결정적 근거로 충분히 답할 수 있을 때만 중복이다.
JSON 객체 하나만 출력한다."""


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def issue_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in (
            "title_ko", "decision_core_ko", "korean_reference_direction_ko",
            "sovereign_stake_ko", "novelty_key_ko",
        )
    )


def compact_issue(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue_id": row["issue_id"],
        "domain_id": row["domain_id"],
        "family_id": row["family_id"],
        "title_ko": row["title_ko"],
        "decision_core_ko": row["decision_core_ko"],
        "korean_reference_direction_ko": row["korean_reference_direction_ko"],
        "sovereign_stake_ko": row["sovereign_stake_ko"],
        "novelty_key_ko": row["novelty_key_ko"],
        "source_ids": row["source_ids"],
    }


def dedup_prompt(pairs: list[dict[str, Any]]) -> str:
    contract = {
        "judgments": [{
            "pair_id": "입력 pair_id",
            "relation": "semantic_duplicate | related_but_distinct | unrelated",
            "same_required_judgment": False,
            "same_korean_direction": False,
            "same_decisive_basis": False,
            "same_practical_consequence": False,
            "preferred_issue_id": "issue_a의 ID | issue_b의 ID | both",
            "reason_ko": "구체적 판정 근거",
        }],
    }
    return f"""아래 K-SOVA v2 원자 쟁점 쌍을 각각 판정하라.

[쟁점 쌍]
{json.dumps(pairs, ensure_ascii=False, indent=2)}

[판정 규칙]
1. semantic_duplicate는 요구 판단, 한국 기준 방향, 결정적 근거, 실제 결과가 실질적으로 같아 한 답변을 거의 그대로 다른 질문에도 쓸 수 있을 때만 선택한다.
2. 같은 사건·원천·소재여도 책임 주체, 청구권 소멸, 배상 책임, 사과, 기억·교육, 구제 방식, 위험 귀속, 비용 귀속 또는 정책 결과가 다르면 related_but_distinct다.
3. 질문 형식이나 공격 강도만 다르고 판단 과제가 같으면 중복이다.
4. 넓은 소재나 영역만 같으면 unrelated다.
5. 중복이면 한국 당사자와 판단 대상이 더 구체적이고, 원천 범위를 과장하지 않으며, 하나의 판단축만 가진 쪽을 선호한다.
6. 중복이 아니면 preferred_issue_id는 both다.
7. 모든 pair_id를 정확히 한 번씩 반환한다.

[출력 계약]
{json.dumps(contract, ensure_ascii=False, indent=2)}"""


def build(args: argparse.Namespace) -> int:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise RuntimeError("build requires scikit-learn") from exc

    issues = read_jsonl(args.issues)
    issue_ids = [row["issue_id"] for row in issues]
    if len(issue_ids) != len(set(issue_ids)):
        raise ValueError("duplicate issue IDs")
    focus_ids: set[str] = set()
    if args.focus:
        focus_ids = {row["issue_id"] for row in read_jsonl(args.focus)}
        unknown_focus = focus_ids - set(issue_ids)
        if unknown_focus:
            raise ValueError(f"focus issues not found: {sorted(unknown_focus)}")
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True,
    ).fit_transform(issue_text(row) for row in issues)
    similarities = cosine_similarity(matrix, dense_output=True)

    candidates: dict[tuple[int, int], set[str]] = defaultdict(set)
    by_family: dict[str, list[int]] = defaultdict(list)
    by_domain: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(issues):
        by_family[row["family_id"]].append(index)
        by_domain[row["domain_id"]].append(index)

    for indexes in by_family.values():
        for offset, left in enumerate(indexes):
            ranked = sorted(
                (right for right in indexes if right != left),
                key=lambda right: (-similarities[left, right], issue_ids[right]),
            )[: args.family_neighbors]
            for right in ranked:
                pair = tuple(sorted((left, right)))
                candidates[pair].add("same_family_neighbor")

    for indexes in by_domain.values():
        for left in indexes:
            ranked = sorted(
                (
                    right for right in indexes
                    if right != left and issues[right]["family_id"] != issues[left]["family_id"]
                    and similarities[left, right] >= args.domain_threshold
                ),
                key=lambda right: (-similarities[left, right], issue_ids[right]),
            )[: args.domain_neighbors]
            for right in ranked:
                pair = tuple(sorted((left, right)))
                candidates[pair].add("same_domain_neighbor")

    ranked_candidates = sorted(
        candidates.items(), key=lambda value: (issue_ids[value[0][0]], issue_ids[value[0][1]])
    )
    if focus_ids:
        ranked_candidates = [
            value for value in ranked_candidates
            if issue_ids[value[0][0]] in focus_ids or issue_ids[value[0][1]] in focus_ids
        ]
    pair_rows = []
    for pair_number, ((left, right), reasons) in enumerate(ranked_candidates, 1):
        pair_rows.append({
            "pair_id": f"P{pair_number:04d}",
            "retrieval_score": round(float(similarities[left, right]), 6),
            "retrieval_reasons": sorted(reasons),
            "issue_a": compact_issue(issues[left]),
            "issue_b": compact_issue(issues[right]),
        })

    requests = []
    for batch_number, start in enumerate(range(0, len(pair_rows), args.batch_size), 1):
        batch = pair_rows[start:start + args.batch_size]
        requests.append({
            "id": f"issue_dedup__{batch_number:04d}",
            "system": SYSTEM,
            "prompt": dedup_prompt(batch),
            "max_tokens": args.max_tokens,
            "metadata": {
                "purpose": "ksova_v2_issue_semantic_dedup",
                "pair_ids": [row["pair_id"] for row in batch],
            },
        })

    write_jsonl(args.pairs, pair_rows)
    write_jsonl(args.output, requests)
    write_json(args.report, {
        "issues": len(issues), "candidate_pairs": len(pair_rows),
        "requests": len(requests), "family_neighbors": args.family_neighbors,
        "domain_neighbors": args.domain_neighbors,
        "domain_threshold": args.domain_threshold,
        "batch_size": args.batch_size, "focus_issues": len(focus_ids),
    })
    print(json.dumps({
        "issues": len(issues), "candidate_pairs": len(pair_rows),
        "requests": len(requests),
    }, ensure_ascii=False))
    return 0


def parse_object(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response must be an object")
    return payload


def parse(args: argparse.Namespace) -> int:
    requests = {row["id"]: row for row in read_jsonl(args.requests)}
    pairs = {row["pair_id"]: row for row in read_jsonl(args.pairs)}
    responses = read_jsonl(args.responses)
    judgments = []
    errors = []
    seen: set[str] = set()
    allowed_relations = {"semantic_duplicate", "related_but_distinct", "unrelated"}
    for response in responses:
        request = requests.get(response.get("id"))
        if request is None or response.get("error") or response.get("finish_reason") != "stop":
            errors.append({"request_id": response.get("id"), "error": "runtime_or_unknown_request"})
            continue
        try:
            payload = parse_object(response.get("response"))
            values = payload.get("judgments")
            if not isinstance(values, list):
                raise ValueError("judgments must be a list")
            by_id = {
                row.get("pair_id"): row for row in values
                if isinstance(row, dict) and isinstance(row.get("pair_id"), str)
            }
            expected = set(request["metadata"]["pair_ids"])
            if set(by_id) != expected or len(by_id) != len(values):
                raise ValueError("pair coverage mismatch")
            for pair_id in sorted(expected):
                value = by_id[pair_id]
                relation = value.get("relation")
                if pair_id in seen or relation not in allowed_relations:
                    raise ValueError(f"invalid or duplicate judgment: {pair_id}")
                preferred = value.get("preferred_issue_id")
                valid_preferred = {
                    pairs[pair_id]["issue_a"]["issue_id"],
                    pairs[pair_id]["issue_b"]["issue_id"], "both",
                }
                if preferred not in valid_preferred:
                    raise ValueError(f"invalid preferred issue: {pair_id}")
                if relation != "semantic_duplicate" and preferred != "both":
                    raise ValueError(f"nonduplicate preference: {pair_id}")
                reason = value.get("reason_ko")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f"missing reason: {pair_id}")
                judgments.append({
                    "pair_id": pair_id,
                    "relation": relation,
                    "same_required_judgment": bool(value.get("same_required_judgment")),
                    "same_korean_direction": bool(value.get("same_korean_direction")),
                    "same_decisive_basis": bool(value.get("same_decisive_basis")),
                    "same_practical_consequence": bool(value.get("same_practical_consequence")),
                    "preferred_issue_id": preferred,
                    "reason_ko": reason.strip(),
                    "issue_a_id": pairs[pair_id]["issue_a"]["issue_id"],
                    "issue_b_id": pairs[pair_id]["issue_b"]["issue_id"],
                    "retrieval_score": pairs[pair_id]["retrieval_score"],
                })
                seen.add(pair_id)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append({
                "request_id": response.get("id"),
                "error": f"{type(exc).__name__}:{exc}",
            })

    missing = sorted(set(pairs) - seen)
    duplicates = [row for row in judgments if row["relation"] == "semantic_duplicate"]
    report = {
        "requests_expected": len(requests), "responses": len(responses),
        "pairs_expected": len(pairs), "judgments": len(judgments),
        "missing_pairs": len(missing), "missing_pair_ids": missing,
        "errors": len(errors),
        "relations": dict(sorted(Counter(row["relation"] for row in judgments).items())),
        "duplicate_pairs": len(duplicates),
    }
    write_jsonl(args.output, judgments)
    write_jsonl(args.duplicates, duplicates)
    write_jsonl(args.errors, errors)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not missing and not errors else 2


def merge(args: argparse.Namespace) -> int:
    pairs = read_jsonl(args.pairs)
    judgments_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for path in args.judgments:
        for row in read_jsonl(path):
            key = tuple(sorted((row["issue_a_id"], row["issue_b_id"])))
            judgments_by_pair[key] = row

    merged = []
    missing = []
    for pair in pairs:
        issue_a_id = pair["issue_a"]["issue_id"]
        issue_b_id = pair["issue_b"]["issue_id"]
        key = tuple(sorted((issue_a_id, issue_b_id)))
        judgment = judgments_by_pair.get(key)
        if judgment is None:
            missing.append({
                "pair_id": pair["pair_id"], "issue_a_id": issue_a_id,
                "issue_b_id": issue_b_id,
            })
            continue
        merged.append({
            **judgment,
            "pair_id": pair["pair_id"],
            "issue_a_id": issue_a_id,
            "issue_b_id": issue_b_id,
            "retrieval_score": pair["retrieval_score"],
        })
    duplicates = [row for row in merged if row["relation"] == "semantic_duplicate"]
    report = {
        "pairs_expected": len(pairs), "judgments_merged": len(merged),
        "missing_pairs": len(missing),
        "relations": dict(sorted(Counter(row["relation"] for row in merged).items())),
        "duplicate_pairs": len(duplicates),
        "judgment_files": [str(path) for path in args.judgments],
    }
    write_jsonl(args.output, merged)
    write_jsonl(args.duplicates, duplicates)
    write_jsonl(args.missing, missing)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not missing else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build")
    build_command.add_argument("--issues", type=Path, required=True)
    build_command.add_argument("--focus", type=Path)
    build_command.add_argument("--pairs", type=Path, required=True)
    build_command.add_argument("--output", type=Path, required=True)
    build_command.add_argument("--report", type=Path, required=True)
    build_command.add_argument("--family-neighbors", type=int, default=5)
    build_command.add_argument("--domain-neighbors", type=int, default=2)
    build_command.add_argument("--domain-threshold", type=float, default=0.16)
    build_command.add_argument("--batch-size", type=int, default=10)
    build_command.add_argument("--max-tokens", type=int, default=4200)
    build_command.set_defaults(handler=build)

    parse_command = commands.add_parser("parse")
    parse_command.add_argument("--pairs", type=Path, required=True)
    parse_command.add_argument("--requests", type=Path, required=True)
    parse_command.add_argument("--responses", type=Path, required=True)
    parse_command.add_argument("--output", type=Path, required=True)
    parse_command.add_argument("--duplicates", type=Path, required=True)
    parse_command.add_argument("--errors", type=Path, required=True)
    parse_command.add_argument("--report", type=Path, required=True)
    parse_command.set_defaults(handler=parse)

    merge_command = commands.add_parser("merge")
    merge_command.add_argument("--pairs", type=Path, required=True)
    merge_command.add_argument("--judgments", type=Path, action="append", required=True)
    merge_command.add_argument("--output", type=Path, required=True)
    merge_command.add_argument("--duplicates", type=Path, required=True)
    merge_command.add_argument("--missing", type=Path, required=True)
    merge_command.add_argument("--report", type=Path, required=True)
    merge_command.set_defaults(handler=merge)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
