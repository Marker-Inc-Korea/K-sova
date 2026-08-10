#!/usr/bin/env python3
"""Merge blind K-SOVA issue reviews with explicit manual adjudications."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def command_run(args: argparse.Namespace) -> int:
    package = json.loads(args.issues.read_text(encoding="utf-8"))
    issues = package["issues"]
    issue_by_id = {row["issue_id"]: row for row in issues}
    owner_payload = json.loads(args.owner_decisions.read_text(encoding="utf-8"))
    owner = owner_payload["decisions"]
    advocate = {row["issue_id"]: row for row in read_jsonl(args.advocate_review)}
    strict = {row["issue_id"]: row for row in read_jsonl(args.strict_review)}
    override_payload = json.loads(args.manual_overrides.read_text(encoding="utf-8"))
    overrides = override_payload["overrides"]
    issue_ids = set(issue_by_id)
    for name, values in (("advocate", advocate), ("strict", strict)):
        if set(values) != issue_ids:
            raise ValueError(f"{name} review does not exactly cover issue package")
    if not set(overrides) <= issue_ids:
        raise ValueError("manual override contains unknown issue ids")
    valid_verdicts = {"core", "supporting", "exclude", "uncertain"}
    rows = []
    for issue in sorted(issues, key=lambda row: row["review_order"]):
        issue_id = issue["issue_id"]
        strict_row = strict[issue_id]
        override = overrides.get(issue_id)
        final_verdict = override["verdict"] if override else strict_row["verdict"]
        if final_verdict not in valid_verdicts:
            raise ValueError(f"invalid final verdict for {issue_id}: {final_verdict}")
        rows.append({
            "schema_version": "ksova-independent-issue-recommendation-1.0",
            "review_order": issue["review_order"],
            "issue_id": issue_id,
            "origin": issue["origin"],
            "category_id": issue["category_id"],
            "category_label": issue["category_label"],
            "title_ko": issue["title_ko"],
            "questions": issue["questions"],
            "question_count": issue["question_count"],
            "source_count": issue["source_count"],
            "owner_decision": (owner.get(issue_id) or {}).get("decision"),
            "advocate_verdict": advocate[issue_id]["verdict"],
            "advocate_reason_ko": advocate[issue_id]["reason_ko"],
            "strict_verdict": strict_row["verdict"],
            "strict_reason_ko": strict_row["reason_ko"],
            "strict_risk_flags": strict_row["risk_flags"],
            "final_verdict": final_verdict,
            "manual_override": override is not None,
            "final_reason_ko": override["reason_ko"] if override else strict_row["reason_ko"],
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "recommendations.jsonl", rows)
    csv_path = args.output_dir / "recommendations.csv"
    fieldnames = [
        "review_order", "issue_id", "origin", "category_id", "category_label",
        "title_ko", "owner_decision", "advocate_verdict", "strict_verdict",
        "final_verdict", "manual_override", "strict_risk_flags", "final_reason_ko",
        "questions",
    ]
    temporary_csv = csv_path.with_name(csv_path.name + f".tmp-{os.getpid()}")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    " | ".join(row[key]) if key in {"strict_risk_flags", "questions"}
                    else row[key]
                )
                for key in fieldnames
            })
    temporary_csv.replace(csv_path)
    final_counts = Counter(row["final_verdict"] for row in rows)
    owner_crosstab = Counter((row["owner_decision"], row["final_verdict"]) for row in rows)
    category_counts: dict[str, dict[str, int]] = {}
    for category_id in sorted({row["category_id"] for row in rows}):
        category_counts[category_id] = dict(sorted(Counter(
            row["final_verdict"] for row in rows if row["category_id"] == category_id
        ).items()))
    origin_counts: dict[str, dict[str, int]] = {}
    for origin in sorted({row["origin"] for row in rows}):
        origin_counts[origin] = dict(sorted(Counter(
            row["final_verdict"] for row in rows if row["origin"] == origin
        ).items()))
    report = {
        "schema_version": "ksova-independent-issue-recommendation-report-1.0",
        "issues": len(rows),
        "final_counts": dict(sorted(final_counts.items())),
        "strict_counts_before_manual": dict(sorted(Counter(
            row["strict_verdict"] for row in rows
        ).items())),
        "manual_overrides": len(overrides),
        "owner_final_crosstab": {
            f"{owner_value}->{verdict}": count
            for (owner_value, verdict), count in sorted(owner_crosstab.items())
        },
        "owner_O_not_core": sum(
            row["owner_decision"] == "O" and row["final_verdict"] != "core" for row in rows
        ),
        "owner_X_or_q_to_core": sum(
            row["owner_decision"] in {"X", "?"} and row["final_verdict"] == "core"
            for row in rows
        ),
        "by_category": category_counts,
        "by_origin": origin_counts,
        "owner_decisions_modified": False,
        "recommendations_jsonl": str((args.output_dir / "recommendations.jsonl").resolve()),
        "recommendations_csv": str(csv_path.resolve()),
    }
    write_json(args.output_dir / "report.json", report)
    promoted = [
        row for row in rows
        if row["owner_decision"] in {"X", "?"} and row["final_verdict"] == "core"
    ]
    owner_o_noncore = [
        row for row in rows
        if row["owner_decision"] == "O" and row["final_verdict"] != "core"
    ]
    excluded = [row for row in rows if row["final_verdict"] == "exclude"]
    md = [
        "# K-SOVA 독립 쟁점 재검토 결과",
        "",
        "사용자 O/X/?는 판정 입력에서 가린 채 353개를 두 번 검토했다. 첫 검토는 전체 쟁점 카드, 두 번째 검토는 실제 통과 문항 중심으로 수행했고, 두 결과와 원문이 충돌한 항목은 수동 재판정했다. 기존 사용자 결정 파일은 수정하지 않았다.",
        "",
        f"- 최종 권고: 핵심 {final_counts['core']} / 보조·학습 후보 {final_counts['supporting']} / 현 문항 제외 {final_counts['exclude']} / 유보 {final_counts['uncertain']}",
        f"- 사용자 O 중 비핵심 권고: {report['owner_O_not_core']}",
        f"- 사용자 X/? 중 핵심 복원 권고: {report['owner_X_or_q_to_core']}",
        f"- 수동 재판정: {len(overrides)}",
        "",
        "## 사용자 X/?에서 핵심으로 복원할 쟁점",
        "",
    ]
    md.extend(f"- `{row['issue_id']}` — {row['title_ko']}" for row in promoted)
    md.extend(["", "## 현 문항 제외", ""])
    md.extend(
        f"- `{row['issue_id']}` — {row['title_ko']}: {row['final_reason_ko']}"
        for row in excluded
    )
    md.extend(["", "## 사용자 O에서 보조군 또는 제외로 내릴 쟁점", ""])
    md.extend(
        f"- `{row['issue_id']}` — {row['title_ko']} ({row['final_verdict']})"
        for row in owner_o_noncore
    )
    md.extend(["", "보조군은 자동으로 학습에 투입한다는 뜻이 아니다. 지식회상·숨은 맥락·핵심 쟁점과의 의미 중복을 별도 게이트로 통과한 항목만 학습에 사용해야 한다.", ""])
    atomic_text(args.output_dir / "summary.md", "\n".join(md))
    print(json.dumps(report, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues", type=Path, required=True)
    parser.add_argument("--owner-decisions", type=Path, required=True)
    parser.add_argument("--advocate-review", type=Path, required=True)
    parser.add_argument("--strict-review", type=Path, required=True)
    parser.add_argument("--manual-overrides", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return command_run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
