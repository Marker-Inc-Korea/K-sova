#!/usr/bin/env python3
"""Seal independently GLM-audited materials into one fail-closed training input."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def passing_glm_validation(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    matches = []
    for key, value in row.items():
        if not isinstance(value, dict) or value.get("effective_pass") is not True:
            continue
        model = str(value.get("validator_model") or value.get("model") or "").casefold()
        if "glm" in model:
            matches.append((key, value))
    return matches[0] if len(matches) == 1 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    quarantine = []
    reasons = Counter()
    seen_material_ids = set()
    for path in args.input:
        for row in read_jsonl(path):
            material_id = str(row.get("material_id") or "")
            validation = passing_glm_validation(row)
            reason = None
            if not material_id:
                reason = "missing_material_id"
            elif material_id in seen_material_ids:
                reason = "duplicate_material_id"
            elif row.get("evidence_status") != "complete":
                reason = "evidence_not_complete"
            elif validation is None:
                reason = "not_exactly_one_passing_glm_validation"
            elif row.get("status") != f"{validation[0]}_passed":
                reason = "status_validation_mismatch"
            elif not 1 <= len(list(dict.fromkeys(row.get("source_ids") or []))) <= 4:
                reason = "invalid_source_count"
            if reason:
                reasons[reason] += 1
                quarantine.append({
                    "material_id": material_id or None,
                    "reason": reason,
                    "input_path": str(path),
                })
                continue
            seen_material_ids.add(material_id)
            validation_key, judgment = validation
            identity = normalize(
                str(row.get("specific_issue_ko") or "") + " "
                + str(row.get("decision_object_ko") or "")
            )
            if not identity:
                reasons["empty_semantic_identity"] += 1
                quarantine.append({
                    "material_id": material_id,
                    "reason": "empty_semantic_identity",
                    "input_path": str(path),
                })
                continue
            candidates.append({
                **row,
                "status": "material_accepted_glm_pass",
                "pipeline_material_audit": {
                    "policy": "single_glm_single_round_no_correction",
                    "validator": "glm",
                    "glm": judgment,
                    "source_validation_key": validation_key,
                    "source_status": row.get("status"),
                    "source_file": str(path),
                    "semantic_rewrite": False,
                },
                "_seal_identity": identity,
            })

    # Exact normalized duplicates are not resolved by subjective preference.
    # Quarantine every member of the collision group and leave semantic-near
    # duplicates to the separately reported embedding holdout distribution.
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_identity[row["_seal_identity"]].append(row)
    sealed = []
    for values in by_identity.values():
        if len(values) > 1:
            for row in values:
                reasons["exact_semantic_identity_collision"] += 1
                quarantine.append({
                    "material_id": row["material_id"],
                    "reason": "exact_semantic_identity_collision",
                    "colliding_material_ids": sorted(value["material_id"] for value in values),
                    "input_path": row["pipeline_material_audit"]["source_file"],
                })
            continue
        row = values[0]
        row.pop("_seal_identity", None)
        sealed.append(row)

    sealed.sort(key=lambda row: row["material_id"])
    quarantine.sort(key=lambda row: (str(row.get("reason")), str(row.get("material_id"))))
    write_jsonl(args.output, sealed)
    write_jsonl(args.quarantine, quarantine)
    report = {
        "schema_version": "ksova-glm-training-material-seal-report-1.0",
        "input_files": [str(path) for path in args.input],
        "input_rows": sum(len(read_jsonl(path)) for path in args.input),
        "sealed_materials": len(sealed),
        "quarantined": len(quarantine),
        "quarantine_reasons": dict(sorted(reasons.items())),
        "policy": "single_glm_single_round_no_correction",
        "semantic_rewrite": False,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "quarantine": str(args.quarantine),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
