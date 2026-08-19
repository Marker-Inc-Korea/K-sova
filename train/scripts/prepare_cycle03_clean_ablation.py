#!/usr/bin/env python3
"""Deterministically materialize the cycle-3 clean SFT/DPO ablation data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable


EXPECTED_INPUT_SHA256 = (
    "504e4c41c372857dd5c936071770b1e9f4adda25be8f308b604b274044c5da12"
)
EXPECTED_ROWS = 113
EXPECTED_SPLITS = {"train": 95, "dev": 18}
VALIDATION_CHECKS = (
    "rejected_is_base_derived",
    "rejected_factually_bounded",
    "chosen_factually_bounded",
    "single_material_defect_in_rejected",
    "chosen_repairs_assigned_defect",
    "chosen_korean_stake_clear",
    "chosen_counterargument_addressed",
    "chosen_conditions_and_costs_present",
    "chosen_has_clear_judgment",
    "length_and_style_balanced",
    "no_government_mimicry_or_nationalist_overreach",
    "pair_useful_for_preference_learning",
)
CRITICAL_PATTERNS = (
    (
        "numeric",
        re.compile(
            r"(?<![A-Za-z가-힣])\d+(?:[.,]\d+)*(?:\s*(?:년|월|일|%|％|배|명|건|대|개|조|억|만|km|㎞|달러|원|위|차))?",
            re.IGNORECASE,
        ),
    ),
    (
        "acronym",
        re.compile(r"(?<![A-Za-z0-9])(?:[A-Z][A-Z0-9.-]{1,})(?![A-Za-z0-9])"),
    ),
    (
        "named_institution_or_instrument",
        re.compile(
            r"(?<![가-힣A-Za-z0-9])([가-힣A-Za-z·]{2,30}"
            r"(?:정부|법원|재판소|위원회|협의회|협회|연구원|연구소|기구|기금|은행|공사|대학|협정|협약|조약|법))"
            r"(?![가-힣A-Za-z0-9])"
        ),
    ),
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
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"\s+", "", text)


def critical_tokens(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for kind, pattern in CRITICAL_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1) if match.lastindex else match.group(0)
            token = normalized(raw)
            if len(token) >= 2:
                result[token] = kind
    return result


def reasons_for(row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    validation = row.get("independent_validation") or {}
    checks = validation.get("checks") or {}
    if validation.get("verdict") != "pass":
        reasons.append("independent_validation_not_pass")
    if any(checks.get(key) is not True for key in VALIDATION_CHECKS):
        reasons.append("independent_validation_check_failed")

    chosen = str(row.get("chosen_response") or "")
    rejected = str(row.get("rejected_response") or "")
    similarity = float((row.get("similarities") or {}).get("chosen_rejected") or 0)
    token_ratio = float((row.get("lengths") or {}).get("chosen_rejected_token_ratio") or 0)
    if similarity < 0.90:
        reasons.append("chosen_rejected_5gram_dice_below_0p90")
    if normalized(chosen) == normalized(rejected):
        reasons.append("chosen_rejected_identical")
    if not 0.90 <= token_ratio <= 1.10:
        reasons.append("chosen_rejected_token_ratio_outside_0p90_1p10")

    chosen_tokens = critical_tokens(chosen)
    rejected_tokens = critical_tokens(rejected)
    new_tokens = {
        token: kind for token, kind in chosen_tokens.items() if token not in rejected_tokens
    }
    issue_support = normalized(json.dumps(row.get("issue_card") or {}, ensure_ascii=False))
    unsupported_tokens = {
        token: kind for token, kind in new_tokens.items() if token not in issue_support
    }
    if unsupported_tokens:
        reasons.append("chosen_introduces_unsupported_critical_token")

    if row.get("defect_type") == "unsupported_specificity":
        if new_tokens:
            reasons.append("specificity_repair_introduces_new_critical_token")
        if len(chosen) > len(rejected) * 1.05:
            reasons.append("specificity_repair_character_growth_above_5pct")

    audit = {
        "chosen_rejected_5gram_dice": similarity,
        "chosen_rejected_token_ratio": token_ratio,
        "chosen_rejected_character_ratio": round(len(chosen) / max(1, len(rejected)), 6),
        "new_critical_tokens": new_tokens,
        "unsupported_new_critical_tokens": unsupported_tokens,
    }
    return sorted(set(reasons)), audit


def dpo_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "training_id": f"KSC3-DPO-{row['pair_id']}",
        "pair_id": row["pair_id"],
        "prompt_id": row["prompt_id"],
        "issue_id": row["issue_id"],
        "domain_id": row["domain_id"],
        "prompt_form": row["prompt_form"],
        "defect_type": row["defect_type"],
        "split": row["split"],
        "sample_weight": 1.0,
        "source_cycle": "cycle02_onpolicy",
        "filter_version": "cycle03_strict_locality_1.0",
    }
    return {
        "prompt_messages": [{"role": "user", "content": row["question_ko"]}],
        "chosen": row["chosen_response"],
        "rejected": row["rejected_response"],
        "metadata": metadata,
    }


def sft_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "training_id": f"KSC3-SFT-{row['pair_id']}",
        "pair_id": row["pair_id"],
        "prompt_id": row["prompt_id"],
        "issue_id": row["issue_id"],
        "domain_id": row["domain_id"],
        "prompt_form": row["prompt_form"],
        "defect_type": row["defect_type"],
        "split": row["split"],
        "source_cycle": "cycle02_onpolicy",
        "filter_version": "cycle03_strict_locality_1.0",
    }
    return {
        "messages": [
            {"role": "user", "content": row["question_ko"]},
            {"role": "assistant", "content": row["chosen_response"]},
        ],
        "metadata": metadata,
    }


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    if sha256(args.input) != EXPECTED_INPUT_SHA256:
        raise ValueError("unexpected cycle-2 accepted-pair input hash")
    rows = read_jsonl(args.input)
    if len(rows) != EXPECTED_ROWS or Counter(row.get("split") for row in rows) != Counter(EXPECTED_SPLITS):
        raise ValueError("unexpected cycle-2 accepted-pair row counts")
    if len({str(row.get("pair_id")) for row in rows}) != len(rows):
        raise ValueError("duplicate pair_id in input")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reasons, audit = reasons_for(row)
        if reasons:
            reason_counts.update(reasons)
            rejected.append({
                "pair_id": row["pair_id"],
                "prompt_id": row["prompt_id"],
                "issue_id": row["issue_id"],
                "split": row["split"],
                "domain_id": row["domain_id"],
                "defect_type": row["defect_type"],
                "rejection_reasons": reasons,
                "audit": audit,
            })
        else:
            accepted.append({**row, "cycle03_filter_audit": audit})

    train_rows = [row for row in accepted if row["split"] == "train"]
    dev_rows = [row for row in accepted if row["split"] == "dev"]
    train_issues = {str(row["issue_id"]) for row in train_rows}
    dev_issues = {str(row["issue_id"]) for row in dev_rows}
    if train_issues & dev_issues:
        raise ValueError("retained train/dev issues overlap")
    if not train_rows or not dev_rows:
        raise ValueError("strict filter removed an entire split")

    outputs = {
        "strict_pairs.jsonl": accepted,
        "rejected_pairs.jsonl": rejected,
        "dpo_train.jsonl": [dpo_row(row) for row in train_rows],
        "dpo_dev.jsonl": [dpo_row(row) for row in dev_rows],
        "sft_train.jsonl": [sft_row(row) for row in train_rows],
        "sft_dev.jsonl": [sft_row(row) for row in dev_rows],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in outputs.items():
        write_jsonl(args.output_dir / name, values)

    report = {
        "schema_version": "ksova-cycle03-clean-ablation-data-1.0",
        "input": {
            "path": str(args.input.resolve()),
            "rows": len(rows),
            "sha256": sha256(args.input),
        },
        "filter": {
            "retained_rows": len(accepted),
            "rejected_rows": len(rejected),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
        },
        "retained": {
            "train_rows": len(train_rows),
            "train_issues": len(train_issues),
            "dev_rows": len(dev_rows),
            "dev_issues": len(dev_issues),
            "train_dev_issue_overlap": 0,
            "train_domains": distribution(train_rows, "domain_id"),
            "dev_domains": distribution(dev_rows, "domain_id"),
            "train_defects": distribution(train_rows, "defect_type"),
            "dev_defects": distribution(dev_rows, "defect_type"),
        },
        "outputs": {
            name: {
                "path": str((args.output_dir / name).resolve()),
                "rows": len(values),
                "sha256": sha256(args.output_dir / name),
            }
            for name, values in outputs.items()
        },
    }
    write_json(args.output_dir / "preparation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
