#!/usr/bin/env python3
"""Independently audit the 32 generated training bundles with Codex Spark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schema(issue_ids: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "audits": {
                "type": "array",
                "minItems": len(issue_ids),
                "maxItems": len(issue_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "issue_id": {"type": "string", "enum": sorted(issue_ids)},
                        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
                        "failed_tasks": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "fact_claim_actor_separation",
                                    "korean_stake_extraction",
                                    "baseline_and_range",
                                    "response_discrimination",
                                ],
                            },
                        },
                        "reasons_ko": {
                            "type": "array",
                            "maxItems": 4,
                            "items": {"type": "string", "minLength": 5},
                        },
                    },
                    "required": ["issue_id", "verdict", "failed_tasks", "reasons_ko"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["audits"],
        "additionalProperties": False,
    }


def prompt(cases: list[dict[str, Any]]) -> str:
    return f"""K-SOVA 학습데이터의 독립 검수자다. 각 case의 validated_dossier만 근거로 generated_bundle을 검수하라. 도구·검색·외부 지식을 사용하지 않는다.

PASS 조건:
1. 네 과제의 답이 validated_examples의 사실·필수요소·사실경계와 모순되지 않고 새로운 구체 사실을 만들지 않는다.
2. fact_claim_actor_separation은 사실, 주장·평가, 행위자·관점을 실제로 구분한다.
3. korean_stake_extraction은 한국에 실제로 걸린 권리·비용·위험·가치 중 해당 항목을 구체화하되 특정 정책이나 현 정부 입장을 무조건 정답화하지 않는다. 네 항목의 표제나 전부를 기계적으로 요구하지 말고, validated_dossier상 실제 이해관계가 빠졌을 때만 실패로 본다.
4. baseline_and_range는 최소 기준, 정당한 조건부 범위, 실질적 실패 방향을 구분한다.
5. response_discrimination의 정답 후보는 기준을 지키고 오답 후보는 실제 prohibited/failure 방향이며, 판정 라벨이 맞다.
6. 모든 입력은 자립적이고 출력은 짧은 검증 가능 답안이며 숨은 사고과정·애국적 수사·외국 국민 일반화가 없다.

사소한 문체 차이는 FAIL이 아니다. 학습하면 잘못된 사실·판단·구조를 익힐 실질적 문제가 있을 때만 FAIL로 판정한다. PASS이면 failed_tasks와 reasons_ko는 빈 배열로 둔다. 각 issue_id를 정확히 한 번 반환하고 JSON 외 텍스트를 출력하지 않는다.

[CASES]
{json.dumps(cases, ensure_ascii=False, indent=2)}
"""


def valid_output(path: Path, issue_ids: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))["audits"]
        actual = [str(row["issue_id"]) for row in rows]
        if sorted(actual) != sorted(issue_ids) or len(actual) != len(set(actual)):
            return False
        for row in rows:
            if len(row["failed_tasks"]) != len(set(row["failed_tasks"])):
                return False
            if row["verdict"] == "PASS" and (row["failed_tasks"] or row["reasons_ko"]):
                return False
            if row["verdict"] == "FAIL" and not row["reasons_ko"]:
                return False
        return True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def command_build(args: argparse.Namespace) -> int:
    dossiers = {str(row["issue_id"]): row for row in read_jsonl(args.dossiers)}
    bundles = {str(row["issue_id"]): row for row in read_jsonl(args.bundles)}
    if len(dossiers) != 32 or len(bundles) != 32 or set(dossiers) != set(bundles):
        raise ValueError("audit requires the same 32 unique dossiers and bundles")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    issue_ids = sorted(dossiers)
    batches: list[dict[str, Any]] = []
    for start in range(0, len(issue_ids), args.batch_size):
        selected = issue_ids[start : start + args.batch_size]
        number = start // args.batch_size + 1
        stem = f"batch_{number:03d}"
        cases = [{
            "issue_id": issue_id,
            "validated_dossier": dossiers[issue_id],
            "generated_bundle": bundles[issue_id],
        } for issue_id in selected]
        prompt_path = args.output_dir / f"{stem}.prompt.txt"
        schema_path = args.output_dir / f"{stem}.schema.json"
        output_path = args.output_dir / f"{stem}.output.json"
        atomic_text(prompt_path, prompt(cases))
        write_json(schema_path, schema(selected))
        batches.append({
            "batch_id": stem,
            "issue_ids": selected,
            "prompt": str(prompt_path.resolve()),
            "schema": str(schema_path.resolve()),
            "output": str(output_path.resolve()),
        })
    manifest = {
        "schema_version": "ksova-codex-training-audit-batches-1.0",
        "dossiers": str(args.dossiers.resolve()),
        "bundles": str(args.bundles.resolve()),
        "dossiers_sha256": sha256(args.dossiers),
        "bundles_sha256": sha256(args.bundles),
        "issues": 32,
        "batch_size": args.batch_size,
        "batches": batches,
    }
    write_json(args.manifest, manifest)
    print(json.dumps({"issues": 32, "batches": len(batches)}))
    return 0


def command_run(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for batch in manifest["batches"]:
        output = Path(batch["output"])
        if args.resume and valid_output(output, batch["issue_ids"]):
            results.append({"batch_id": batch["batch_id"], "status": "reused"})
            continue
        command = [
            args.codex_bin, "exec", "-m", args.model, "--ephemeral",
            "--ignore-user-config", "--ignore-rules", "--sandbox", "read-only",
            "--skip-git-repo-check", "--color", "never",
            "--disable", "apps", "--disable", "plugins",
            "--disable", "tool_suggest", "--disable", "multi_agent",
            "--disable", "browser_use", "--disable", "computer_use",
            "--output-schema", batch["schema"], "-o", str(output), "-",
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            input=Path(batch["prompt"]).read_text(encoding="utf-8"),
            text=True,
            capture_output=True,
            cwd=args.cwd,
            timeout=args.timeout,
            check=False,
        )
        base = output.with_suffix("")
        atomic_text(base.with_suffix(".stdout.log"), completed.stdout)
        atomic_text(base.with_suffix(".stderr.log"), completed.stderr)
        passed = completed.returncode == 0 and valid_output(
            output, batch["issue_ids"]
        )
        results.append({
            "batch_id": batch["batch_id"],
            "status": "completed" if passed else "failed",
            "returncode": completed.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        if not passed:
            break
    success = len(results) == len(manifest["batches"]) and all(
        row["status"] in {"completed", "reused"} for row in results
    )
    report = {
        "schema_version": "ksova-codex-training-audit-run-1.0",
        "model": args.model,
        "success": success,
        "results": results,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if success else 1


def command_collect(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for batch in manifest["batches"]:
        output = Path(batch["output"])
        if not valid_output(output, batch["issue_ids"]):
            raise ValueError(f"invalid audit output: {output}")
        rows.extend(json.loads(output.read_text(encoding="utf-8"))["audits"])
    rows.sort(key=lambda row: str(row["issue_id"]))
    if len(rows) != 32 or len({row["issue_id"] for row in rows}) != 32:
        raise ValueError("audit collection does not cover 32 issues exactly")
    write_jsonl(args.output, rows)
    failures = [row for row in rows if row["verdict"] == "FAIL"]
    report = {
        "schema_version": "ksova-codex-training-audit-report-1.0",
        "audited_bundles": len(rows),
        "passed_bundles": len(rows) - len(failures),
        "failed_bundles": len(failures),
        "pass": not failures,
        "model": args.model,
        "output_sha256": sha256(args.output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--dossiers", type=Path, required=True)
    build.add_argument("--bundles", type=Path, required=True)
    build.add_argument("--batch-size", type=int, default=4)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--model", default="gpt-5.3-codex-spark")
    run.add_argument("--codex-bin", default="codex")
    run.add_argument("--cwd", type=Path, default=Path.cwd())
    run.add_argument("--timeout", type=int, default=900)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--report", type=Path, required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument("--model", default="gpt-5.3-codex-spark")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    return {
        "build": command_build,
        "run": command_run,
        "collect": command_collect,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
