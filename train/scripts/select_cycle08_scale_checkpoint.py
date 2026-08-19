#!/usr/bin/env python3
"""Apply the preregistered Cycle08 scale-checkpoint selection rule."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--likelihood-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-order", nargs="+", required=True)
    parser.add_argument("--minimum-train-reduction-percent", type=float, default=10.0)
    parser.add_argument("--minimum-dev-reduction-percent", type=float, default=8.0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    report = json.loads(args.likelihood_report.read_text(encoding="utf-8"))
    comparisons = report["comparisons"]
    if len(args.checkpoint_order) != len(set(args.checkpoint_order)):
        raise ValueError("checkpoint order contains duplicates")
    missing = [name for name in args.checkpoint_order if name not in comparisons]
    if missing:
        raise ValueError(f"likelihood report misses checkpoints: {missing}")

    checks: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for order, name in enumerate(args.checkpoint_order, 1):
        value = comparisons[name]
        train_reduction = float(value["train_reduction_percent"])
        dev_reduction = float(value["dev_reduction_percent"])
        checks[name] = {
            "order": order,
            "train_reduction_percent": train_reduction,
            "dev_reduction_percent": dev_reduction,
            "train_gate_pass": train_reduction
            >= args.minimum_train_reduction_percent,
            "dev_gate_pass": dev_reduction >= args.minimum_dev_reduction_percent,
        }
        checks[name]["both_gates_pass"] = (
            checks[name]["train_gate_pass"] and checks[name]["dev_gate_pass"]
        )
        if checks[name]["both_gates_pass"]:
            eligible.append(name)

    selected = eligible[0] if eligible else None
    diagnostic_lowest_dev = min(
        args.checkpoint_order, key=lambda name: float(comparisons[name]["dev_nll"])
    )
    result = {
        "schema_version": "ksova-cycle08-scale-checkpoint-selection-1.0",
        "status": "selected" if selected else "no_checkpoint_passed",
        "likelihood_report": str(args.likelihood_report.resolve()),
        "selection_rule": (
            "training NLL reduction >= 10% and issue-disjoint natural dev NLL "
            "reduction >= 8%인 가장 이른 체크포인트"
        ),
        "checkpoint_order": args.checkpoint_order,
        "minimum_train_reduction_percent": args.minimum_train_reduction_percent,
        "minimum_dev_reduction_percent": args.minimum_dev_reduction_percent,
        "checks": checks,
        "eligible_checkpoints": eligible,
        "selected_checkpoint": selected,
        "diagnostic_lowest_dev_checkpoint": diagnostic_lowest_dev,
        "generation_gate_triggered": selected is not None,
    }
    atomic_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
