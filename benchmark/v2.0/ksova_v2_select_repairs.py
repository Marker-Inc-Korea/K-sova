#!/usr/bin/env python3
"""Select targeted K-SOVA v2 item repairs from independent quality evidence."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MAJOR = {"major_fix", "reject"}


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


def compact(value: Any, limit: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("named paths must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError("named paths must use nonempty NAME=PATH")
    return name.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--manual-actions", type=Path, required=True)
    parser.add_argument("--manual-overrides", type=Path)
    parser.add_argument("--issue-repairs", type=Path)
    parser.add_argument("--control-review", action="append", default=[])
    parser.add_argument("--adjudication-review", type=Path)
    parser.add_argument("--text-flags", type=Path, required=True)
    parser.add_argument("--dedup-duplicates", type=Path)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    item_rows = read_jsonl(args.items)
    items = {row["item_id"]: row for row in item_rows}
    if len(items) != len(item_rows):
        raise ValueError("duplicate item IDs")
    by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in item_rows:
        by_issue[item["issue_id"]].append(item)

    reasons: dict[str, list[str]] = defaultdict(list)
    sources: dict[str, set[str]] = defaultdict(set)
    unresolved: list[dict[str, Any]] = []
    resolved_adjudications: set[str] = set()

    def add_target(item_id: str, source: str, reason: Any) -> None:
        if item_id not in items:
            raise ValueError(f"unknown target item: {item_id}")
        rendered = compact(reason)
        if rendered and rendered not in reasons[item_id]:
            reasons[item_id].append(rendered)
        sources[item_id].add(source)

    for row in read_jsonl(args.consensus):
        item_id = row["item_id"]
        feedback = []
        for name, review in sorted(row["reviews"].items()):
            if review["verdict"] != "pass":
                feedback.append(f"{name} {review['verdict']}: {compact(review['reason_ko'], 260)}")
        if row["status"] in {"reauthor", "minor_review"}:
            add_target(item_id, f"item_consensus:{row['status']}", " | ".join(feedback))
        elif row["status"] == "adjudicate":
            unresolved.append({
                "kind": "item_adjudication", "item_id": item_id,
                "issue_id": row["issue_id"], "feedback": feedback,
            })

    manual = json.loads(args.manual_actions.read_text(encoding="utf-8"))
    manual_actions = manual.get("actions", [])
    if args.manual_overrides:
        override_data = json.loads(args.manual_overrides.read_text(encoding="utf-8"))
        drop_ids = set(override_data.get("drop_issue_ids", []))
        override_rows = override_data.get("overrides", [])
        overrides = {row["issue_id"]: row for row in override_rows}
        if len(overrides) != len(override_rows):
            raise ValueError("duplicate manual override issue IDs")
        known_ids = {row["issue_id"] for row in manual_actions}
        unknown = set(overrides) - known_ids
        if unknown:
            raise ValueError(f"manual overrides reference unknown issues: {sorted(unknown)}")
        manual_actions = [
            overrides.get(action["issue_id"], action)
            for action in manual_actions
            if action["issue_id"] not in drop_ids
        ]
        additional_actions = override_data.get("additional_actions", [])
        if not isinstance(additional_actions, list) or any(
            not isinstance(action, dict) for action in additional_actions
        ):
            raise ValueError("manual_overrides.additional_actions must be an object list")
        manual_actions.extend(additional_actions)

    for action in manual_actions:
        issue_id = action["issue_id"]
        if issue_id not in by_issue:
            raise ValueError(f"manual action references unknown issue: {issue_id}")
        action_name = action["action"]
        reason = action["reason_ko"]
        source = f"manual:{action.get('source', 'unknown')}"
        if action_name == "reauthor":
            target_ids = action.get("target_item_ids")
            if target_ids is None:
                target_ids = [item["item_id"] for item in by_issue[issue_id]]
            if not isinstance(target_ids, list) or not target_ids:
                raise ValueError(f"{issue_id}: target_item_ids must be a nonempty list")
            for item_id in target_ids:
                if items.get(item_id, {}).get("issue_id") != issue_id:
                    raise ValueError(f"{issue_id}: invalid target item {item_id}")
                add_target(item_id, source, reason)
        elif action_name == "minor_fix":
            target_ids = action.get("target_item_ids")
            if not isinstance(target_ids, list) or not target_ids:
                raise ValueError(f"{issue_id}: minor_fix needs target_item_ids")
            for item_id in target_ids:
                if items.get(item_id, {}).get("issue_id") != issue_id:
                    raise ValueError(f"{issue_id}: invalid target item {item_id}")
                add_target(item_id, source, reason)
        elif action_name in {"adjudicate", "deduplicate"}:
            unresolved.append({
                "kind": f"manual_{action_name}", "issue_id": issue_id,
                "reason_ko": reason, "source": action.get("source"),
            })
        else:
            raise ValueError(f"{issue_id}: unsupported manual action {action_name}")

    if args.issue_repairs:
        for repair in read_jsonl(args.issue_repairs):
            issue_id = repair["issue_id"]
            if issue_id not in by_issue:
                raise ValueError(f"issue repair references unknown issue: {issue_id}")
            reason = repair.get("reason_ko") or "원자 쟁점 명세 교체에 따른 문항 재작성"
            for item in by_issue[issue_id]:
                add_target(item["item_id"], "issue_spec_repair", reason)

    control_sets: dict[str, dict[str, dict[str, Any]]] = {}
    for value in args.control_review:
        name, path = parse_named_path(value)
        rows = read_jsonl(path)
        by_id = {row["item_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"{name}: duplicate control review IDs")
        control_sets[name] = by_id
    if control_sets:
        expected_controls = {
            item_id for item_id, item in items.items()
            if item["item_role"] == "paired_knowledge_control"
        }
        for name, rows in control_sets.items():
            if set(rows) != expected_controls:
                raise ValueError(f"{name}: control review coverage mismatch")
        for item_id in sorted(expected_controls):
            verdicts = {name: rows[item_id]["verdict"] for name, rows in control_sets.items()}
            major_count = sum(value in MAJOR for value in verdicts.values())
            minor_count = sum(value == "minor_fix" for value in verdicts.values())
            nonpass_count = sum(value != "pass" for value in verdicts.values())
            feedback = " | ".join(
                f"{name} {rows[item_id]['verdict']}: {compact(rows[item_id]['reason_ko'], 260)}"
                for name, rows in sorted(control_sets.items())
                if rows[item_id]["verdict"] != "pass"
            )
            if major_count >= 2 or (major_count >= 1 and nonpass_count >= 2) or minor_count >= 2:
                add_target(item_id, "control_consensus", feedback)
            elif nonpass_count:
                unresolved.append({
                    "kind": "control_adjudication", "item_id": item_id,
                    "issue_id": items[item_id]["issue_id"], "verdicts": verdicts,
                    "feedback": feedback,
                })

    for row in read_jsonl(args.text_flags):
        add_target(
            row["item_id"], "deterministic_text_audit",
            f"비한국어 한자·깨진 문자 후보: {row.get('flags') or row}",
        )

    if args.adjudication_review:
        adjudication_rows = read_jsonl(args.adjudication_review)
        adjudication_by_id = {row["item_id"]: row for row in adjudication_rows}
        if len(adjudication_by_id) != len(adjudication_rows):
            raise ValueError("duplicate adjudication review item IDs")
        unresolved_item_ids = {
            row["item_id"] for row in unresolved if isinstance(row.get("item_id"), str)
        }
        unknown = set(adjudication_by_id) - unresolved_item_ids
        if unknown:
            raise ValueError(f"adjudication reviews were not unresolved: {sorted(unknown)}")
        for item_id, row in adjudication_by_id.items():
            verdict = row.get("verdict")
            if verdict not in {"pass", "minor_fix", "major_fix", "reject"}:
                raise ValueError(f"{item_id}: invalid adjudication verdict {verdict}")
            resolved_adjudications.add(item_id)
            if verdict != "pass":
                add_target(
                    item_id, f"independent_adjudication:{verdict}", row.get("reason_ko"),
                )

    if args.dedup_duplicates and args.dedup_duplicates.exists():
        for row in read_jsonl(args.dedup_duplicates):
            unresolved.append({"kind": "semantic_duplicate_issue_pair", **row})

    target_rows = [{
        "item_id": item_id,
        "issue_id": items[item_id]["issue_id"],
        "domain_id": items[item_id]["domain_id"],
        "item_role": items[item_id]["item_role"],
        "reasons": reasons[item_id],
        "evidence_sources": sorted(sources[item_id]),
    } for item_id in sorted(reasons)]

    targeted = set(reasons)
    filtered_unresolved = [
        row for row in unresolved
        if not row.get("item_id")
        or (row["item_id"] not in targeted and row["item_id"] not in resolved_adjudications)
    ]
    report = {
        "items": len(items), "target_items": len(target_rows),
        "target_issues": len({row["issue_id"] for row in target_rows}),
        "target_roles": dict(sorted(Counter(row["item_role"] for row in target_rows).items())),
        "evidence_sources": dict(sorted(Counter(
            source for row in target_rows for source in row["evidence_sources"]
        ).items())),
        "unresolved_records": len(filtered_unresolved),
        "unresolved_kinds": dict(sorted(Counter(row["kind"] for row in filtered_unresolved).items())),
        "adjudicated_items": len(resolved_adjudications),
    }
    write_jsonl(args.targets, target_rows)
    write_jsonl(args.unresolved, filtered_unresolved)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
