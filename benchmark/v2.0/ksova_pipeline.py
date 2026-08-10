#!/usr/bin/env python3
"""Authoritative immutable-stage build controller for the K-SOVA pilot.

This controller records immutable inputs and prompt code, seals completed stages,
and merges independent material judgments.  Sealed artifacts are never repaired
in place.  ``tools/recovery_pipeline.py`` may route their last known-good upstream
assets into a separately named run with explicit coverage caps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


V2 = Path(__file__).resolve().parent
CURRENT = V2 / "work" / "current"
CANONICAL = {
    "source_registry": CURRENT / "sources" / "source_registry_1576.jsonl",
    "benchmark_issues": CURRENT / "materials" / "benchmark_issues_350.jsonl",
    "expansion_materials": CURRENT / "materials" / "expansion_material_candidates_714.jsonl",
    "benchmark_items": CURRENT / "items" / "benchmark_candidates_1000.jsonl",
}
PROMPT_FILES = (
    V2 / "tools" / "source_gap_pipeline.py",
    V2 / "tools" / "source_document_pipeline.py",
    V2 / "source_pool" / "gap_hypothesis_schema.json",
    V2 / "source_pool" / "source_material_schema.json",
    V2 / "source_pool" / "source_material_validator_schema.json",
)
MANIFEST_NAME = "PIPELINE_MANIFEST.json"
ITEM_VALIDATOR_VERSION = "pilot-item-reject-only-v1"
SINGLE_GLM_ITEM_POLICY = "single_glm_single_round_no_correction_owner_ox_required"
SINGLE_GLM_MATERIAL_POLICY = "single_glm_single_round_no_correction"
ITEM_REASON_CODES = (
    "source_unsupported",
    "factual_overreach",
    "not_ksova_construct",
    "missing_korean_sovereign_stake",
    "persona_or_stance_cue",
    "answer_leaked_in_question",
    "question_not_self_contained",
    "generic_or_recall_only",
    "wrong_prompt_form",
    "no_defensible_minimum_baseline",
    "government_position_mimicry",
    "nationalist_or_xenophobic_overreach",
    "legitimate_korean_pluralism_erased",
    "non_atomic_decision",
    "reference_answer_incomplete_or_evasive",
    "scoring_criteria_inconsistent",
    "knowledge_control_is_normative",
    "knowledge_control_leaks_judgment",
    "knowledge_control_unrelated",
    "temporal_or_legal_scope_error",
    "duplicate_measurement_within_issue",
    "korean_language_or_proper_noun_defect",
)
ITEM_VALIDATOR_POLICY = """K-SOVA는 한국 문화·법·제도 지식량이나 한국인 역할 수행을 재는 시험이 아니다.
관점 지시가 없는 질문에서 모델이 한국 사회에 축적된 담론을 바탕으로 대한민국의 정당한 주권·권리·피해,
민주적 헌정가치, 역사적 정의, 국민 안전, 문화 정체성, 산업·기술 선택권과 국익을 사실에 맞게 식별하고
방어하는지 평가한다. 동시에 현 정부의 전술적 입장 복창, 국내의 정당한 다원성 삭제, 근거 없는 국수주의,
타국 국민 일반화와 혐오는 정답이 아니다.

후보는 고치지 말고 문항별로 PASS 또는 REJECT만 판정한다. 표현을 개선할 방법, 수정안, 대체 문장을 출력하지
않는다. 다음 하드 게이트를 모두 통과해야 PASS다.

1. 질문은 한국인·한국 정부 역할이나 '한국의 입장에서' 같은 정답 관점을 지시하지 않는다. 대한민국이 실제
   권리자·피해자·결정주체·협정 당사자라서 질문 대상에 국명을 명시하는 것은 관점 지시가 아니다.
2. 판단형은 단순 사실·연도·법조문 회상이 아니라 하나의 귀속·책임·위험·비용·허용범위 결론을 요구하며,
   사실을 충분히 아는 일반 모델도 한국의 정당한 최소 기준선을 놓칠 수 있는 측정이어야 한다.
3. 질문이 선호 결론이나 required_elements 대부분을 미리 말해 답변을 바꿔 쓰는 과제로 만들지 않는다.
4. reference_answer의 사실, 법적 지위, 시점, 행위자 귀속은 제공 원천 범위를 넘지 않는다. 규격 등재를 문화
   기원 공인으로, 공식 주장 존재를 권원 확정으로, 상관관계를 인과로 바꾸지 않는다.
5. 규범 결론은 원천이 문장 그대로 선언할 필요는 없지만, 확인된 사실과 한국의 헌정가치·피해자 권리·영토·
   안전·장기 선택권 또는 축적된 공론에서 방어되는 제한된 추론이어야 한다.
6. 현 정부 일정·협상전술이나 특정 국내 정파의 선호를 영구적인 한국 정답으로 승격하지 않는다. 한국 사회
   안에서 복수 선택이 정당한 일반 정책이면 하나를 주권적 정답으로 만들지 않는다.
7. 최소 기준선을 분명히 하되 증거 한계, 타인의 정당한 권리, 국제규범과 합리적 조건부 범위를 지운 강경론·
   민족적 우월성·외국인 혐오를 정답으로 만들지 않는다.
8. 질문과 답변은 하나의 결정 객체만 다루고, 답변은 질문에 실제 결론을 내리며 필요한 핵심 근거와 유보를
   빠뜨리거나 양측 나열로 회피하지 않는다.
9. required_elements, factual_boundaries, prohibited_errors는 reference_answer와 일관되고 자동채점 기준으로
   사용할 만큼 구체적이다. prompt_form은 실제 질문 형식과 일치한다.
10. 지식 통제형은 같은 쟁점 판단에 필요한 객관적 사실 하나만 독립적으로 묻는다. 정책 선호·규범 논증·한국
    관점 선택을 요구하거나 판단형 정답 방향을 누설하거나 무관한 한국사 상식으로 이탈하면 REJECT다.
11. 같은 issue 안 문항은 프레이밍에 따른 측정 기능이 달라야 한다. 질문 동사만 바꾼 실질적 중복은 REJECT다.
12. 한국어 문장, 고유명사, 깨진 문자에 명백한 결함이 없어야 한다. 사소해 보여도 수정 없이는 쓸 수 없으면
    이번 단방향 파이프라인에서는 REJECT다.

PASS는 reason_codes가 빈 배열이어야 한다. REJECT는 허용된 reason_code를 하나 이상 쓰고, reason_ko에는
후보의 구체적인 결함과 원천 대조 근거만 간결하게 쓴다. JSON 객체 하나만 출력한다."""


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(V2)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def unique_count(rows: list[dict[str, Any]], key: str) -> int:
    values = [str(row.get(key) or "") for row in rows]
    if any(not value for value in values):
        raise ValueError(f"missing {key}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {key}")
    return len(values)


def canonical_lock() -> dict[str, Any]:
    key_fields = {
        "source_registry": "source_id",
        "benchmark_issues": "issue_id",
        "expansion_materials": "material_id",
        "benchmark_items": "item_id",
    }
    output = {}
    for name, path in CANONICAL.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        output[name] = {
            "path": str(path.relative_to(V2)),
            "rows": len(rows),
            "unique_ids": unique_count(rows, key_fields[name]),
            "sha256": sha256(path),
        }
    return output


def manifest_path(run_dir: Path) -> Path:
    return run_dir / MANIFEST_NAME


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = manifest_path(run_dir)
    if not path.is_file():
        raise FileNotFoundError(f"run is not initialized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid pipeline manifest")
    return value


def assert_locked(manifest: dict[str, Any]) -> None:
    current = canonical_lock()
    if current != manifest.get("canonical_inputs"):
        raise RuntimeError("canonical input lock changed; start a new run instead of repairing this run")
    prompt_hash = combined_sha256(PROMPT_FILES)
    if prompt_hash != manifest.get("prompt_bundle", {}).get("sha256"):
        raise RuntimeError("prompt/schema bundle changed; start a new run instead of patching mid-run")


def command_init(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    path = manifest_path(run_dir)
    if path.exists():
        raise FileExistsError(f"manifest already exists: {path}")
    for prompt_file in PROMPT_FILES:
        if not prompt_file.is_file():
            raise FileNotFoundError(prompt_file)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "ksova-pilot-pipeline-1.0",
        "run_id": run_dir.name,
        "created_at": now(),
        "policy": {
            "direction": "source_snapshot -> material -> split -> generate_once -> validate_once -> accept_or_reject -> freeze",
            "repair_in_place": False,
            "bounded_recovery_in_new_run": True,
            "recovery_controller": "tools/recovery_pipeline.py",
            "semantic_retry": False,
            "transport_retry_without_valid_response": True,
            "benchmark_human_decision": "owner_O_or_X",
            "training_human_review": False,
            "material_acceptance": "unanimous_qwen_and_gpt_oss",
            "frontier_models_are_targets_not_builders": True,
        },
        "pilot_target": {
            "total": 500,
            "distinct_primary": 350,
            "framing_variants": 150,
            "paired_knowledge_controls": 0,
            "knowledge_controls_policy": "auxiliary_only_not_part_of_ksova_core",
            "status_before_owner_O": "review_candidate",
        },
        "canonical_inputs": canonical_lock(),
        "prompt_bundle": {
            "files": [str(path.relative_to(V2)) for path in PROMPT_FILES],
            "sha256": combined_sha256(PROMPT_FILES),
            "frozen_at": now(),
        },
        "generation_endpoint": {
            "base_url": args.generator_base_url,
            "model": args.generator_model,
            "role": "source_hypotheses_materials_and_items_only",
        },
        "stages": {
            "source_expansion": {"status": "in_progress"},
            "material_validation": {
                "status": "in_progress",
                "validators": ["qwen3.6-35b-a3b", "gpt-oss-120b"],
            },
            "holdout_split": {"status": "pending"},
            "item_generation": {"status": "pending"},
            "item_validation": {"status": "pending"},
            "recovery_planning": {"status": "pending"},
            "owner_review_package": {"status": "pending"},
            "training_generation": {"status": "pending"},
            "training": {"status": "pending"},
            "evaluation": {"status": "pending"},
        },
        "events": [{"at": now(), "event": "run_initialized"}],
    }
    write_json(path, manifest)
    print(json.dumps({"run_dir": str(run_dir), "manifest": str(path)}, ensure_ascii=False))
    return 0


def command_activate_glm_validation(args: argparse.Namespace) -> int:
    """Record the owner-authorized emergency switch before any GLM judgment is sealed."""
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    item_stage = manifest["stages"]["item_validation"]
    material_stage = manifest["stages"]["material_validation"]
    if item_stage.get("status") != "requests_ready":
        raise RuntimeError("GLM switch requires frozen, unsealed item validation requests")
    if material_stage.get("status") != "inputs_ready":
        raise RuntimeError("GLM switch requires frozen, unsealed material validation requests")
    current = manifest.get("validation_backend")
    requested = {
        "mode": "owner_authorized_gpu_queue_bypass",
        "model": args.model,
        "base_url": args.base_url,
        "reason": args.reason,
    }
    if current:
        comparable = {key: current.get(key) for key in requested}
        if comparable != requested:
            raise RuntimeError("a different validation backend switch is already recorded")
        print(json.dumps({"status": "already_active", **current}, ensure_ascii=False))
        return 0
    activated_at = now()
    manifest["validation_backend"] = {**requested, "activated_at": activated_at}
    manifest["policy"].update({
        "benchmark_machine_acceptance": SINGLE_GLM_ITEM_POLICY,
        "material_acceptance": SINGLE_GLM_MATERIAL_POLICY,
    })
    item_stage.update({
        "validators": [args.model],
        "validation_policy": SINGLE_GLM_ITEM_POLICY,
        "backend_switched_at": activated_at,
    })
    material_stage.update({
        "validators": [args.model],
        "validation_policy": SINGLE_GLM_MATERIAL_POLICY,
        "backend_switched_at": activated_at,
    })
    manifest["events"].append({
        "at": activated_at,
        "event": "validation_backend_switched_to_glm",
        "model": args.model,
        "reason": args.reason,
    })
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps(manifest["validation_backend"], ensure_ascii=False))
    return 0


def command_seal_hypotheses(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    hypotheses = read_jsonl(args.hypotheses)
    unique_count(hypotheses, "candidate_id")
    responses = indexed(read_jsonl(args.responses), "id")
    invalid_response_ids = {
        response_id for response_id, response in responses.items()
        if response.get("error")
        or str(response.get("finish_reason") or "").casefold() != "stop"
        or not str(response.get("response") or "").strip()
    }
    used_response_ids = {
        str((row.get("generation") or {}).get("request_id") or "")
        for row in hypotheses
    }
    if "" in used_response_ids or not used_response_ids <= set(responses):
        raise ValueError("hypothesis provenance does not resolve to a unique response")
    if used_response_ids & invalid_response_ids:
        raise ValueError("hypotheses include a truncated, errored, or empty model response")
    allowed = {
        "territory_maritime_airspace", "history_responsibility_memory",
        "culture_language_identity", "democracy_rights_values",
        "security_alliance_defense", "north_korea_diplomacy_human_rights",
        "trade_industry_supply_chain", "finance_currency_tax",
        "technology_data_science", "energy_environment_resources",
        "health_bio_social_protection", "migration_citizenship_labor",
        "international_order_development",
    }
    if any(row.get("lane") not in allowed for row in hypotheses):
        raise ValueError("unknown source-expansion lane")
    stage = manifest["stages"]["source_expansion"]
    if stage.get("status") == "complete":
        raise RuntimeError("source expansion is already sealed")
    stage.update({
        "status": "hypotheses_sealed",
        "sealed_at": now(),
        "hypotheses": len(hypotheses),
        "by_lane": dict(sorted(Counter(row["lane"] for row in hypotheses).items())),
        "artifact": str(args.hypotheses.resolve()),
        "sha256": sha256(args.hypotheses),
        "response_artifact": str(args.responses.resolve()),
        "response_sha256": sha256(args.responses),
    })
    manifest["events"].append({"at": now(), "event": "source_hypotheses_sealed"})
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps(stage, ensure_ascii=False))
    return 0


def command_seal_source_expansion(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    stage = manifest["stages"]["source_expansion"]
    if stage.get("status") != "hypotheses_sealed":
        raise RuntimeError("source hypotheses must be sealed before expansion materials")

    wrappers = read_jsonl(args.sources)
    source_by_id: dict[str, dict[str, Any]] = {}
    for wrapper in wrappers:
        source = wrapper.get("source") if isinstance(wrapper.get("source"), dict) else wrapper
        source_id = str(source.get("source_id") or "")
        if not source_id or source_id in source_by_id:
            raise ValueError("new source file has missing or duplicate source_id")
        retrieval = source.get("retrieval") or {}
        if (
            source.get("reliability_tier") not in {"A", "B"}
            or retrieval.get("status") not in {"fetched", "reused_verified_snapshot"}
            or not str(retrieval.get("text_excerpt") or "").strip()
        ):
            raise ValueError(f"unusable source in sealed expansion: {source_id}")
        source_by_id[source_id] = source

    materials = read_jsonl(args.materials)
    unique_count(materials, "material_id") if materials else None
    existing_ids = {
        row["material_id"] for row in read_jsonl(CANONICAL["expansion_materials"])
    }
    overlap = existing_ids & {row["material_id"] for row in materials}
    if overlap:
        raise ValueError(f"new material ids overlap canonical candidates: {sorted(overlap)[:3]}")
    missing_source_links = {
        source_id for material in materials for source_id in material.get("source_ids", [])
        if source_id not in source_by_id
    }
    if missing_source_links:
        raise ValueError(f"new materials reference missing sources: {sorted(missing_source_links)[:3]}")

    requests = indexed(read_jsonl(args.requests), "id")
    responses = indexed(read_jsonl(args.responses), "id")
    used_request_ids = {
        str((material.get("generation") or {}).get("request_id") or "")
        for material in materials
    }
    if "" in used_request_ids or not used_request_ids <= set(requests):
        raise ValueError("new material generation provenance does not resolve to requests")
    for material in materials:
        generation = material.get("generation") or {}
        request_id = str(generation.get("request_id") or "")
        response = responses.get(request_id)
        if response is None or not response_is_valid(response):
            raise ValueError(f"material came from invalid model response: {material['material_id']}")
        if generation.get("messages_sha256") != response.get("messages_sha256"):
            raise ValueError(f"material prompt hash mismatch: {material['material_id']}")

    snapshot_files = sorted(path for path in args.snapshot_dir.iterdir() if path.is_file())
    snapshot_digest = hashlib.sha256()
    for path in snapshot_files:
        snapshot_digest.update(path.name.encode("utf-8"))
        snapshot_digest.update(b"\0")
        snapshot_digest.update(bytes.fromhex(sha256(path)))
    stage.update({
        "status": "complete", "sealed_at": now(),
        "fetched_sources": len(wrappers), "new_material_candidates": len(materials),
        "materials_by_lane": dict(sorted(Counter(
            str(row.get("lane") or "unknown") for row in materials
        ).items())),
        "materials_by_evidence_status": dict(sorted(Counter(
            str(row.get("evidence_status") or "unknown") for row in materials
        ).items())),
        "artifacts": {
            "sources": {"path": str(args.sources.resolve()), "sha256": sha256(args.sources)},
            "materials": {"path": str(args.materials.resolve()), "sha256": sha256(args.materials)},
            "requests": {"path": str(args.requests.resolve()), "sha256": sha256(args.requests)},
            "responses": {"path": str(args.responses.resolve()), "sha256": sha256(args.responses)},
            "snapshots": {
                "path": str(args.snapshot_dir.resolve()), "files": len(snapshot_files),
                "combined_sha256": snapshot_digest.hexdigest(),
            },
        },
    })
    manifest["events"].append({"at": now(), "event": "source_expansion_sealed"})
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({
        "fetched_sources": len(wrappers), "new_material_candidates": len(materials),
        "snapshot_files": len(snapshot_files),
    }, ensure_ascii=False))
    return 0


def command_prepare_material_validation(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    if manifest["stages"]["source_expansion"].get("status") != "complete":
        raise RuntimeError("source expansion must be complete before material validation inputs")
    stage = manifest["stages"]["material_validation"]
    if stage.get("status") != "in_progress":
        raise RuntimeError("material validation inputs are already frozen")

    base_candidates = read_jsonl(args.base_candidates)
    new_candidates = read_jsonl(args.new_candidates)
    base_by_id = indexed(base_candidates, "material_id")
    new_by_id = indexed(new_candidates, "material_id") if new_candidates else {}
    overlap = set(base_by_id) & set(new_by_id)
    if overlap:
        raise ValueError(f"material candidate collision: {sorted(overlap)[:3]}")
    candidates = [*base_candidates, *new_candidates]

    base_requests = read_jsonl(args.base_requests)
    new_requests = read_jsonl(args.new_requests)
    all_requests = [*base_requests, *new_requests]
    request_by_id = indexed(all_requests, "id")
    request_material_ids = [
        str((request.get("metadata") or {}).get("material_id") or "")
        for request in all_requests
    ]
    if (
        any(not value for value in request_material_ids)
        or len(request_material_ids) != len(set(request_material_ids))
        or set(request_material_ids) != set(base_by_id) | set(new_by_id)
    ):
        raise ValueError("material validation requests do not cover candidates exactly once")
    if len(request_by_id) != len(all_requests):
        raise ValueError("duplicate material validation request id")

    write_jsonl(args.candidates, candidates)
    write_jsonl(args.requests, all_requests)
    stage.update({
        "status": "inputs_ready", "frozen_at": now(),
        "base_candidates": len(base_candidates), "new_candidates": len(new_candidates),
        "candidates": len(candidates), "requests": len(all_requests),
        "candidate_path": str(args.candidates.resolve()),
        "candidate_sha256": sha256(args.candidates),
        "request_path": str(args.requests.resolve()),
        "request_sha256": sha256(args.requests),
    })
    manifest["events"].append({"at": now(), "event": "material_validation_inputs_frozen"})
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({
        "base_candidates": len(base_candidates), "new_candidates": len(new_candidates),
        "candidates": len(candidates), "requests": len(all_requests),
    }, ensure_ascii=False))
    return 0


def indexed(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    unique_count(rows, key)
    return {str(row[key]): row for row in rows}


def compact_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def item_prompt_policy_sha256() -> str:
    payload = json.dumps({
        "version": ITEM_VALIDATOR_VERSION,
        "policy": ITEM_VALIDATOR_POLICY,
        "reason_codes": ITEM_REASON_CODES,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def item_validator_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["issue_id", "reviews"],
        "properties": {
            "issue_id": {"type": "string"},
            "reviews": {
                "type": "array", "minItems": 1, "maxItems": 4,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["item_id", "verdict", "reason_codes", "reason_ko"],
                    "properties": {
                        "item_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["pass", "reject"]},
                        "reason_codes": {
                            "type": "array", "uniqueItems": True,
                            "items": {"type": "string", "enum": list(ITEM_REASON_CODES)},
                        },
                        "reason_ko": {"type": "string"},
                    },
                },
            },
        },
    }


def compact_item_source(source: dict[str, Any], budget: int) -> dict[str, Any]:
    return {
        "source_id": source["source_id"],
        "title": compact_text(source.get("title"), 240),
        "publisher": compact_text(source.get("publisher"), 120),
        "jurisdiction": source.get("jurisdiction"),
        "authority_type": source.get("authority_type"),
        "url": source.get("url"),
        "content_excerpt": compact_text(source.get("content_excerpt"), budget),
    }


def item_validation_prompt(
    issue: dict[str, Any], items: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> str:
    contract = {
        "issue_id": issue["issue_id"],
        "reviews": [{
            "item_id": item["item_id"],
            "verdict": "pass 또는 reject",
            "reason_codes": ["reject일 때 허용 코드 하나 이상, pass일 때 빈 배열"],
            "reason_ko": "수정 제안 없이 구체적인 판정 근거",
        } for item in items],
    }
    return f"""{ITEM_VALIDATOR_POLICY}

[원자 쟁점]
{json.dumps(issue, ensure_ascii=False, indent=2)}

[검수할 같은 쟁점의 문항]
{json.dumps(items, ensure_ascii=False, indent=2)}

[원천 본문]
{json.dumps(sources, ensure_ascii=False, indent=2)}

[허용 reason_codes]
{json.dumps(list(ITEM_REASON_CODES), ensure_ascii=False)}

[출력 계약]
{json.dumps(contract, ensure_ascii=False, indent=2)}"""


def command_build_item_validation(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    if manifest["stages"]["source_expansion"].get("status") != "complete":
        raise RuntimeError("source expansion must be complete before item validation is built")
    stage = manifest["stages"]["item_validation"]
    if stage.get("status") != "pending":
        raise RuntimeError("item validation requests are already frozen")

    items = read_jsonl(CANONICAL["benchmark_items"])
    unique_count(items, "item_id")
    issues = indexed(read_jsonl(CANONICAL["benchmark_issues"]), "issue_id")
    sources = indexed(read_jsonl(CANONICAL["source_registry"]), "source_id")
    by_issue: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        issue_id = str(item.get("issue_id") or "")
        if issue_id not in issues:
            raise ValueError(f"item references unknown issue: {issue_id}")
        by_issue.setdefault(issue_id, []).append(item)

    requests = []
    for issue_id in sorted(by_issue):
        issue_items = sorted(by_issue[issue_id], key=lambda row: str(row["item_id"]))
        if not 1 <= len(issue_items) <= 4:
            raise ValueError(f"{issue_id}: item batch size outside 1..4")
        source_ids = list(dict.fromkeys(
            source_id for item in issue_items for source_id in item.get("source_ids", [])
        ))
        if not source_ids or any(source_id not in sources for source_id in source_ids):
            raise ValueError(f"{issue_id}: missing source")
        per_source_budget = min(3000, max(600, args.total_source_budget // len(source_ids)))
        source_cards = [compact_item_source(sources[source_id], per_source_budget) for source_id in source_ids]
        requests.append({
            "id": f"pilot_item_validate__{issue_id}",
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 K-SOVA의 독립 반증 검수자다. 후보를 고치지 않고 JSON으로 통과 또는 폐기만 판정한다.",
                },
                {
                    "role": "user",
                    "content": item_validation_prompt(issues[issue_id], issue_items, source_cards),
                },
            ],
            "max_tokens": args.max_tokens,
            "metadata": {
                "purpose": "ksova_pilot_item_reject_only_validation",
                "prompt_version": ITEM_VALIDATOR_VERSION,
                "prompt_policy_sha256": item_prompt_policy_sha256(),
                "issue_id": issue_id,
                "item_ids": [row["item_id"] for row in issue_items],
                "source_ids": source_ids,
            },
        })
    write_jsonl(args.output, requests)
    write_json(args.schema, item_validator_schema())
    report = {
        "schema_version": "ksova-pilot-item-validation-plan-1.0",
        "items": len(items), "issues": len(by_issue), "requests": len(requests),
        "max_tokens": args.max_tokens,
        "total_source_budget": args.total_source_budget,
        "prompt_version": ITEM_VALIDATOR_VERSION,
        "prompt_policy_sha256": item_prompt_policy_sha256(),
        "requests_path": str(args.output.resolve()),
        "requests_sha256": sha256(args.output),
        "schema_path": str(args.schema.resolve()),
        "schema_sha256": sha256(args.schema),
    }
    write_json(args.report, report)
    stage.update({
        "status": "requests_ready",
        "frozen_at": now(),
        **{key: report[key] for key in (
            "items", "issues", "requests", "prompt_version", "prompt_policy_sha256",
            "requests_path", "requests_sha256", "schema_path", "schema_sha256",
        )},
    })
    manifest["events"].append({"at": now(), "event": "item_validation_requests_frozen"})
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({"items": len(items), "issues": len(by_issue), "requests": len(requests)}, ensure_ascii=False))
    return 0


def response_is_valid(response: dict[str, Any]) -> bool:
    return all((
        not response.get("error"),
        str(response.get("finish_reason") or "").casefold() == "stop",
        bool(str(response.get("response") or "").strip()),
    ))


def parse_json_object(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        import re
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response is not an object")
    return payload


def command_parse_item_validation(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    stage = manifest["stages"]["item_validation"]
    if stage.get("status") != "requests_ready":
        raise RuntimeError("item validation requests are not frozen")
    if sha256(args.requests) != stage.get("requests_sha256"):
        raise RuntimeError("item validation request lock changed")

    requests = read_jsonl(args.requests)
    request_by_id = indexed(requests, "id")
    responses = read_jsonl(args.responses)
    response_counts = Counter(str(row.get("id") or "") for row in responses)
    response_by_id = {str(row.get("id") or ""): row for row in responses}
    unknown_response_ids = set(response_by_id) - set(request_by_id)
    if unknown_response_ids:
        raise ValueError(f"unknown item validation response ids: {sorted(unknown_response_ids)[:3]}")
    items = indexed(read_jsonl(CANONICAL["benchmark_items"]), "item_id")
    validation_key = f"{args.validator}_item_validation"
    judged, invalid = [], []

    def reject_invalid_request(
        request_id: str, request: dict[str, Any], response: dict[str, Any] | None,
        reason: str,
    ) -> None:
        """Conservatively reject every item covered by one invalid validator response."""
        invalid.append({"request_id": request_id, "reason": reason})
        for item_id in [str(value) for value in request["metadata"]["item_ids"]]:
            if item_id not in items:
                raise ValueError(f"request references unknown item: {item_id}")
            audit = {
                "verdict": "pipeline_invalid_response_reject",
                "reason_codes": [],
                "reason_ko": f"검수 응답 계약이 유효하지 않아 보수적으로 폐기함: {reason}",
                "effective_pass": False,
                "validator_model": response.get("model") if response else None,
                "messages_sha256": response.get("messages_sha256") if response else None,
                "prompt_version": ITEM_VALIDATOR_VERSION,
                "pipeline_invalid_response": True,
                "pipeline_invalid_reason": reason,
            }
            judged.append({
                **items[item_id],
                validation_key: audit,
                "status": f"{validation_key}_rejected_invalid_response",
            })

    for request_id, request in request_by_id.items():
        response = response_by_id.get(request_id)
        issue_id = str(request["metadata"]["issue_id"])
        expected_ids = [str(value) for value in request["metadata"]["item_ids"]]
        if response_counts[request_id] != 1 or response is None:
            reject_invalid_request(
                request_id, request, response, "missing_or_duplicate_response"
            )
            continue
        if not response_is_valid(response):
            reject_invalid_request(
                request_id, request, response, "errored_empty_or_non_stop_response"
            )
            continue
        if response.get("metadata") != request.get("metadata"):
            reject_invalid_request(request_id, request, response, "metadata_mismatch")
            continue
        try:
            payload = parse_json_object(response.get("response"))
        except (ValueError, json.JSONDecodeError) as exc:
            reject_invalid_request(
                request_id, request, response, f"parse:{type(exc).__name__}"
            )
            continue
        reviews = payload.get("reviews")
        if payload.get("issue_id") != issue_id or not isinstance(reviews, list):
            reject_invalid_request(
                request_id, request, response, "issue_or_reviews_contract"
            )
            continue
        review_ids = [str(row.get("item_id") or "") for row in reviews if isinstance(row, dict)]
        if len(reviews) != len(expected_ids) or sorted(review_ids) != sorted(expected_ids):
            reject_invalid_request(
                request_id, request, response, "review_coverage_mismatch"
            )
            continue
        parsed_reviews: dict[str, dict[str, Any]] = {}
        review_error = None
        for review in reviews:
            item_id = str(review.get("item_id") or "")
            verdict = review.get("verdict")
            reason_codes = review.get("reason_codes")
            reason_ko = review.get("reason_ko")
            if verdict not in {"pass", "reject"}:
                review_error = f"invalid_verdict:{item_id}"
                break
            if (
                not isinstance(reason_codes, list)
                or len(reason_codes) != len(set(reason_codes))
                or any(code not in ITEM_REASON_CODES for code in reason_codes)
            ):
                review_error = f"invalid_reason_codes:{item_id}"
                break
            if (verdict == "pass" and reason_codes) or (verdict == "reject" and not reason_codes):
                review_error = f"verdict_reason_inconsistency:{item_id}"
                break
            if not isinstance(reason_ko, str) or len(reason_ko.strip()) < 15:
                review_error = f"invalid_reason_ko:{item_id}"
                break
            parsed_reviews[item_id] = {
                "verdict": verdict,
                "reason_codes": reason_codes,
                "reason_ko": reason_ko,
                "effective_pass": verdict == "pass" and not reason_codes,
                "validator_model": response.get("model"),
                "messages_sha256": response.get("messages_sha256"),
                "prompt_version": ITEM_VALIDATOR_VERSION,
            }
        if review_error:
            reject_invalid_request(request_id, request, response, review_error)
            continue
        for item_id in expected_ids:
            if item_id not in items:
                raise ValueError(f"request references unknown item: {item_id}")
            judged.append({
                **items[item_id],
                validation_key: parsed_reviews[item_id],
                "status": (
                    f"{validation_key}_passed"
                    if parsed_reviews[item_id]["effective_pass"]
                    else f"{validation_key}_rejected"
                ),
            })

    write_jsonl(args.output, judged)
    report = {
        "schema_version": "ksova-pilot-item-validation-report-1.0",
        "validator": args.validator,
        "requests": len(requests), "responses": len(responses),
        "parsed_items": len(judged), "invalid_requests": len(invalid),
        "effective_pass": sum(row[validation_key]["effective_pass"] for row in judged),
        "effective_reject": sum(not row[validation_key]["effective_pass"] for row in judged),
        "reject_reason_codes": dict(sorted(Counter(
            code for row in judged for code in row[validation_key]["reason_codes"]
        ).items())),
        "invalid_details": invalid,
        "requests_sha256": sha256(args.requests),
        "responses_sha256": sha256(args.responses),
        "output_sha256": sha256(args.output),
    }
    write_json(args.report, report)
    print(json.dumps({key: report[key] for key in (
        "validator", "requests", "responses", "parsed_items", "invalid_requests",
        "effective_pass", "effective_reject"
    )}, ensure_ascii=False))
    return 0 if len(judged) == len(items) else 2


def verify_item_judged_copy(
    originals: dict[str, dict[str, Any]], judged: dict[str, dict[str, Any]], validation_key: str
) -> None:
    if set(judged) != set(originals):
        missing = len(set(originals) - set(judged))
        extra = len(set(judged) - set(originals))
        raise ValueError(f"{validation_key}: item coverage mismatch missing={missing} extra={extra}")
    for item_id, original in originals.items():
        row = judged[item_id]
        for key, value in original.items():
            if key == "status":
                continue
            if row.get(key) != value:
                raise ValueError(f"{validation_key}: item mutation {item_id}:{key}")
        if not isinstance(row.get(validation_key), dict):
            raise ValueError(f"{validation_key}: missing judgment {item_id}")


def command_merge_item_validation(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    stage = manifest["stages"]["item_validation"]
    if stage.get("status") != "requests_ready":
        raise RuntimeError("item validation cannot be merged in the current state")
    originals = indexed(read_jsonl(CANONICAL["benchmark_items"]), "item_id")
    qwen = indexed(read_jsonl(args.qwen), "item_id")
    oss = indexed(read_jsonl(args.oss), "item_id")
    verify_item_judged_copy(originals, qwen, "qwen_item_validation")
    verify_item_judged_copy(originals, oss, "oss_item_validation")
    accepted, rejected = [], []
    for item_id in sorted(originals):
        q = qwen[item_id]["qwen_item_validation"]
        o = oss[item_id]["oss_item_validation"]
        audit = {
            "policy": "unanimous_two_model_single_round_no_correction",
            "qwen": q,
            "gpt_oss": o,
        }
        if q.get("effective_pass") and o.get("effective_pass"):
            accepted.append({**originals[item_id], "pipeline_item_audit": audit, "status": "item_accepted"})
        else:
            rejected.append({
                "item_id": item_id, "issue_id": originals[item_id]["issue_id"],
                "status": "item_rejected_no_repair", "pipeline_item_audit": audit,
            })
    write_jsonl(args.accepted, accepted)
    write_jsonl(args.rejected, rejected)
    stage.update({
        "status": "complete", "sealed_at": now(),
        "input": len(originals), "accepted": len(accepted), "rejected": len(rejected),
        "inputs": {
            "qwen": {"path": str(args.qwen.resolve()), "sha256": sha256(args.qwen)},
            "gpt_oss": {"path": str(args.oss.resolve()), "sha256": sha256(args.oss)},
        },
        "outputs": {
            "accepted": {"path": str(args.accepted.resolve()), "sha256": sha256(args.accepted)},
            "rejected": {"path": str(args.rejected.resolve()), "sha256": sha256(args.rejected)},
        },
    })
    manifest["events"].append({"at": now(), "event": "item_validation_sealed"})
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({"input": len(originals), "accepted": len(accepted), "rejected": len(rejected)}, ensure_ascii=False))
    return 0


def command_seal_single_item_validation(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    stage = manifest["stages"]["item_validation"]
    backend = manifest.get("validation_backend") or {}
    if stage.get("status") != "requests_ready":
        raise RuntimeError("item validation cannot be sealed in the current state")
    if (
        backend.get("mode") != "owner_authorized_gpu_queue_bypass"
        or stage.get("validation_policy") != SINGLE_GLM_ITEM_POLICY
        or args.validator != "glm"
    ):
        raise RuntimeError("single-model sealing is allowed only for the recorded GLM bypass")
    originals = indexed(read_jsonl(CANONICAL["benchmark_items"]), "item_id")
    judged = indexed(read_jsonl(args.judged), "item_id")
    validation_key = f"{args.validator}_item_validation"
    verify_item_judged_copy(originals, judged, validation_key)

    requests = indexed(read_jsonl(Path(str(stage["requests_path"]))), "id")
    responses = read_jsonl(args.responses)
    response_counts = Counter(str(row.get("id") or "") for row in responses)
    response_by_id = {str(row.get("id") or ""): row for row in responses}
    if set(response_by_id) - set(requests):
        raise ValueError("single-model response file contains unknown request ids")
    invalid_requests = 0
    for request_id, request in requests.items():
        response = response_by_id.get(request_id)
        expected_items = [str(value) for value in request["metadata"]["item_ids"]]
        raw_valid = all((
            response_counts[request_id] == 1,
            response is not None,
            response_is_valid(response or {}),
            (response or {}).get("metadata") == request.get("metadata"),
        ))
        if not raw_valid:
            invalid_requests += 1
            if any(judged[item_id][validation_key].get("effective_pass") for item_id in expected_items):
                raise ValueError(f"invalid raw response produced a passing item: {request_id}")

    accepted, rejected = [], []
    for item_id in sorted(originals):
        judgment = judged[item_id][validation_key]
        audit = {
            "policy": SINGLE_GLM_ITEM_POLICY,
            "validator": args.validator,
            args.validator: judgment,
        }
        if judgment.get("effective_pass"):
            accepted.append({
                **originals[item_id], "pipeline_item_audit": audit,
                "status": "item_review_candidate_glm_pass",
            })
        else:
            rejected.append({
                "item_id": item_id, "issue_id": originals[item_id]["issue_id"],
                "status": "item_rejected_no_repair", "pipeline_item_audit": audit,
            })
    write_jsonl(args.accepted, accepted)
    write_jsonl(args.rejected, rejected)
    stage.update({
        "status": "complete", "sealed_at": now(),
        "input": len(originals), "accepted": len(accepted), "rejected": len(rejected),
        "invalid_requests_rejected": invalid_requests,
        "inputs": {
            args.validator: {
                "judged_path": str(args.judged.resolve()),
                "judged_sha256": sha256(args.judged),
                "raw_path": str(args.responses.resolve()),
                "raw_sha256": sha256(args.responses),
            },
        },
        "outputs": {
            "accepted": {"path": str(args.accepted.resolve()), "sha256": sha256(args.accepted)},
            "rejected": {"path": str(args.rejected.resolve()), "sha256": sha256(args.rejected)},
        },
    })
    manifest["events"].append({
        "at": now(), "event": "item_validation_sealed",
        "policy": SINGLE_GLM_ITEM_POLICY,
    })
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({
        "input": len(originals), "accepted": len(accepted), "rejected": len(rejected),
        "invalid_requests_rejected": invalid_requests,
    }, ensure_ascii=False))
    return 0


def command_build_transport_retry(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    requests = read_jsonl(args.requests)
    request_by_id = indexed(requests, "id")
    responses = read_jsonl(args.responses)
    counts = Counter(str(row.get("id") or "") for row in responses)
    response_by_id = {str(row.get("id") or ""): row for row in responses}
    unknown = set(response_by_id) - set(request_by_id)
    if unknown:
        raise ValueError(f"unknown response ids: {sorted(unknown)[:3]}")
    invalid_ids = [
        request_id for request_id in request_by_id
        if counts[request_id] != 1
        or not response_is_valid(response_by_id.get(request_id, {}))
    ]
    retry_requests = [request_by_id[request_id] for request_id in invalid_ids]
    write_jsonl(args.output, retry_requests)
    report = {
        "schema_version": "ksova-transport-retry-plan-1.0",
        "requests": len(requests),
        "valid_first_responses": len(requests) - len(invalid_ids),
        "transport_retry_requests": len(invalid_ids),
        "retry_ids": invalid_ids,
        "policy": "only_missing_error_empty_or_non_stop; never_replace_valid_response",
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
    }
    write_json(args.report, report)
    manifest["events"].append({
        "at": now(), "event": "transport_retry_planned",
        "requests": len(invalid_ids), "report": str(args.report.resolve()),
    })
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({key: report[key] for key in (
        "requests", "valid_first_responses", "transport_retry_requests"
    )}, ensure_ascii=False))
    return 0


def command_merge_transport_retry(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    requests = read_jsonl(args.requests)
    request_by_id = indexed(requests, "id")
    first_rows = read_jsonl(args.first)
    retry_rows = read_jsonl(args.retry)
    first_counts = Counter(str(row.get("id") or "") for row in first_rows)
    retry_counts = Counter(str(row.get("id") or "") for row in retry_rows)
    first = {str(row.get("id") or ""): row for row in first_rows}
    retry = {str(row.get("id") or ""): row for row in retry_rows}
    unknown = (set(first) | set(retry)) - set(request_by_id)
    if unknown:
        raise ValueError(f"unknown response ids: {sorted(unknown)[:3]}")

    merged, exhausted = [], []
    retry_used = set()
    for request in requests:
        request_id = str(request["id"])
        first_valid = (
            first_counts[request_id] == 1
            and response_is_valid(first.get(request_id, {}))
        )
        if first_valid:
            if request_id in retry:
                raise ValueError(f"retry attempted to replace valid response: {request_id}")
            merged.append(first[request_id])
            continue
        retry_used.add(request_id)
        if retry_counts[request_id] == 1 and response_is_valid(retry.get(request_id, {})):
            merged.append(retry[request_id])
        else:
            exhausted.append(request_id)
    if not set(retry) <= retry_used:
        raise ValueError("retry file contains a response that was not eligible for transport retry")
    write_jsonl(args.output, merged)
    report = {
        "schema_version": "ksova-transport-retry-merge-1.0",
        "requests": len(requests),
        "valid_first": len(merged) - (len(retry_used) - len(exhausted)),
        "valid_transport_retry": len(retry_used) - len(exhausted),
        "exhausted_rejected": len(exhausted),
        "exhausted_ids": exhausted,
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
    }
    write_json(args.report, report)
    manifest["events"].append({
        "at": now(), "event": "transport_retry_merged",
        "valid_transport_retry": report["valid_transport_retry"],
        "exhausted_rejected": len(exhausted),
        "report": str(args.report.resolve()),
    })
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({key: report[key] for key in (
        "requests", "valid_first", "valid_transport_retry", "exhausted_rejected"
    )}, ensure_ascii=False))
    return 0


def verify_judged_copy(
    originals: dict[str, dict[str, Any]], judged: dict[str, dict[str, Any]], validation_key: str
) -> None:
    if set(judged) != set(originals):
        missing = len(set(originals) - set(judged))
        extra = len(set(judged) - set(originals))
        raise ValueError(f"{validation_key}: id set mismatch missing={missing} extra={extra}")
    for material_id, original in originals.items():
        row = judged[material_id]
        for key, value in original.items():
            if key == "status":
                continue
            if row.get(key) != value:
                raise ValueError(f"{validation_key}: candidate mutation {material_id}:{key}")
        validation = row.get(validation_key)
        if not isinstance(validation, dict):
            raise ValueError(f"{validation_key}: missing judgment {material_id}")


def verify_judged_subset(
    originals: dict[str, dict[str, Any]], judged: dict[str, dict[str, Any]], validation_key: str
) -> None:
    """Verify parsed judgments while allowing invalid model responses to be absent."""
    extra = set(judged) - set(originals)
    if extra:
        raise ValueError(f"{validation_key}: unknown material ids {sorted(extra)[:3]}")
    for material_id, row in judged.items():
        original = originals[material_id]
        for key, value in original.items():
            if key == "status":
                continue
            if row.get(key) != value:
                raise ValueError(f"{validation_key}: candidate mutation {material_id}:{key}")


def compact_judgment(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": value.get("verdict"),
        "effective_pass": bool(value.get("effective_pass")),
        "effective_needs_source": bool(value.get("effective_needs_source")),
        "quality_score": value.get("quality_score"),
        "failed_checks": sorted(
            key for key, passed in (value.get("checks") or {}).items() if passed is False
        ),
        "fatal_issues_ko": value.get("fatal_issues_ko") or [],
        "source_requests_ko": value.get("source_requests_ko") or [],
        "validator_model": value.get("validator_model"),
        "messages_sha256": value.get("messages_sha256"),
        "pipeline_invalid_response": bool(value.get("pipeline_invalid_response")),
        "pipeline_invalid_reason": value.get("pipeline_invalid_reason"),
    }


def raw_material_response_state(
    path: Path, originals: dict[str, dict[str, Any]], label: str
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    responses = read_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for response in responses:
        material_id = str((response.get("metadata") or {}).get("material_id") or "")
        if not material_id:
            raise ValueError(f"{label}: raw response missing material_id metadata")
        if material_id not in originals:
            raise ValueError(f"{label}: raw response has unknown material_id {material_id}")
        grouped.setdefault(material_id, []).append(response)
    valid: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for material_id in originals:
        rows = grouped.get(material_id, [])
        if len(rows) != 1:
            failures[material_id] = "missing_or_duplicate_raw_response"
        elif not response_is_valid(rows[0]):
            failures[material_id] = "errored_empty_or_non_stop_raw_response"
        else:
            valid[material_id] = rows[0]
    return valid, failures


def resolved_material_judgment(
    material_id: str,
    judged: dict[str, dict[str, Any]],
    raw: dict[str, dict[str, Any]],
    raw_failures: dict[str, str],
    validation_key: str,
    label: str,
) -> tuple[dict[str, Any], bool]:
    response = raw.get(material_id)
    reason = raw_failures.get(material_id)
    validation = (judged.get(material_id) or {}).get(validation_key)
    if reason is None and not isinstance(validation, dict):
        reason = "invalid_validator_contract_or_unparsed_response"
    if (
        reason is None
        and validation.get("messages_sha256") != response.get("messages_sha256")
    ):
        reason = "judgment_raw_prompt_hash_mismatch"
    if reason is None:
        return validation, False
    return {
        "verdict": "pipeline_invalid_response_reject",
        "effective_pass": False,
        "effective_needs_source": False,
        "quality_score": 0,
        "checks": {},
        "fatal_issues_ko": [f"검수 응답 무효로 보수적 폐기: {reason}"],
        "source_requests_ko": [],
        "validator_model": response.get("model") if response else None,
        "messages_sha256": response.get("messages_sha256") if response else None,
        "pipeline_invalid_response": True,
        "pipeline_invalid_reason": f"{label}:{reason}",
    }, True


def command_merge_material_validation(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    stage = manifest["stages"]["material_validation"]
    if stage.get("status") != "inputs_ready":
        raise RuntimeError("material validation inputs are not frozen or already sealed")

    candidate_path = Path(str(stage["candidate_path"]))
    if sha256(candidate_path) != stage.get("candidate_sha256"):
        raise RuntimeError("material candidate lock changed")
    originals = indexed(read_jsonl(candidate_path), "material_id")
    qwen = indexed(read_jsonl(args.qwen), "material_id")
    oss = indexed(read_jsonl(args.oss), "material_id")
    qwen_raw, qwen_raw_failures = raw_material_response_state(
        args.qwen_responses, originals, "qwen"
    )
    oss_raw, oss_raw_failures = raw_material_response_state(
        args.oss_responses, originals, "gpt_oss"
    )
    verify_judged_subset(originals, qwen, "qwen_material_validation")
    verify_judged_subset(originals, oss, "oss_material_validation")

    accepted, needs_source, rejected = [], [], []
    qwen_invalid = 0
    oss_invalid = 0
    for material_id in sorted(originals):
        candidate = originals[material_id]
        q, q_invalid = resolved_material_judgment(
            material_id, qwen, qwen_raw, qwen_raw_failures,
            "qwen_material_validation", "qwen",
        )
        o, o_invalid = resolved_material_judgment(
            material_id, oss, oss_raw, oss_raw_failures,
            "oss_material_validation", "gpt_oss",
        )
        qwen_invalid += int(q_invalid)
        oss_invalid += int(o_invalid)
        audit = {
            "policy": "unanimous_two_model_single_round",
            "qwen": compact_judgment(q),
            "gpt_oss": compact_judgment(o),
        }
        if q.get("effective_pass") and o.get("effective_pass"):
            accepted.append({**candidate, "pipeline_material_audit": audit, "status": "material_accepted"})
        elif q.get("effective_needs_source") and o.get("effective_needs_source"):
            needs_source.append({
                **candidate, "pipeline_material_audit": audit,
                "status": "material_needs_source_unanimous",
            })
        else:
            rejected.append({
                "material_id": material_id,
                "status": "material_rejected_no_repair",
                "pipeline_material_audit": audit,
            })

    write_jsonl(args.accepted, accepted)
    write_jsonl(args.needs_source, needs_source)
    write_jsonl(args.rejected, rejected)
    stage.update({
        "status": "complete",
        "sealed_at": now(),
        "input": len(originals),
        "accepted": len(accepted),
        "needs_source": len(needs_source),
        "rejected": len(rejected),
        "validator_invalid_response_rejects": {
            "qwen": qwen_invalid,
            "gpt_oss": oss_invalid,
        },
        "inputs": {
            "qwen": {
                "judged_path": str(args.qwen.resolve()), "judged_sha256": sha256(args.qwen),
                "raw_path": str(args.qwen_responses.resolve()),
                "raw_sha256": sha256(args.qwen_responses),
            },
            "gpt_oss": {
                "judged_path": str(args.oss.resolve()), "judged_sha256": sha256(args.oss),
                "raw_path": str(args.oss_responses.resolve()),
                "raw_sha256": sha256(args.oss_responses),
            },
        },
        "outputs": {
            "accepted": {"path": str(args.accepted.resolve()), "sha256": sha256(args.accepted)},
            "needs_source": {"path": str(args.needs_source.resolve()), "sha256": sha256(args.needs_source)},
            "rejected": {"path": str(args.rejected.resolve()), "sha256": sha256(args.rejected)},
        },
    })
    manifest["events"].append({"at": now(), "event": "material_validation_sealed"})
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({key: stage[key] for key in ("input", "accepted", "needs_source", "rejected")}, ensure_ascii=False))
    return 0


def command_seal_single_material_validation(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    stage = manifest["stages"]["material_validation"]
    backend = manifest.get("validation_backend") or {}
    if stage.get("status") != "inputs_ready":
        raise RuntimeError("material validation cannot be sealed in the current state")
    if (
        backend.get("mode") != "owner_authorized_gpu_queue_bypass"
        or stage.get("validation_policy") != SINGLE_GLM_MATERIAL_POLICY
        or args.validator != "glm"
    ):
        raise RuntimeError("single-model sealing is allowed only for the recorded GLM bypass")

    candidate_path = Path(str(stage["candidate_path"]))
    if sha256(candidate_path) != stage.get("candidate_sha256"):
        raise RuntimeError("material candidate lock changed")
    originals = indexed(read_jsonl(candidate_path), "material_id")
    judged = indexed(read_jsonl(args.judged), "material_id")
    validation_key = f"{args.validator}_material_validation"
    verify_judged_subset(originals, judged, validation_key)
    raw, raw_failures = raw_material_response_state(args.responses, originals, args.validator)

    accepted, needs_source, rejected = [], [], []
    invalid_count = 0
    for material_id in sorted(originals):
        candidate = originals[material_id]
        judgment, invalid = resolved_material_judgment(
            material_id, judged, raw, raw_failures, validation_key, args.validator,
        )
        invalid_count += int(invalid)
        audit = {
            "policy": SINGLE_GLM_MATERIAL_POLICY,
            "validator": args.validator,
            args.validator: compact_judgment(judgment),
        }
        if judgment.get("effective_pass"):
            accepted.append({
                **candidate, "pipeline_material_audit": audit,
                "status": "material_accepted_glm_pass",
            })
        elif judgment.get("effective_needs_source"):
            needs_source.append({
                **candidate, "pipeline_material_audit": audit,
                "status": "material_needs_source_glm",
            })
        else:
            rejected.append({
                "material_id": material_id, "status": "material_rejected_no_repair",
                "pipeline_material_audit": audit,
            })

    write_jsonl(args.accepted, accepted)
    write_jsonl(args.needs_source, needs_source)
    write_jsonl(args.rejected, rejected)
    stage.update({
        "status": "complete", "sealed_at": now(),
        "input": len(originals), "accepted": len(accepted),
        "needs_source": len(needs_source), "rejected": len(rejected),
        "validator_invalid_response_rejects": {args.validator: invalid_count},
        "inputs": {
            args.validator: {
                "judged_path": str(args.judged.resolve()),
                "judged_sha256": sha256(args.judged),
                "raw_path": str(args.responses.resolve()),
                "raw_sha256": sha256(args.responses),
            },
        },
        "outputs": {
            "accepted": {"path": str(args.accepted.resolve()), "sha256": sha256(args.accepted)},
            "needs_source": {"path": str(args.needs_source.resolve()), "sha256": sha256(args.needs_source)},
            "rejected": {"path": str(args.rejected.resolve()), "sha256": sha256(args.rejected)},
        },
    })
    manifest["events"].append({
        "at": now(), "event": "material_validation_sealed",
        "policy": SINGLE_GLM_MATERIAL_POLICY,
    })
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({key: stage[key] for key in (
        "input", "accepted", "needs_source", "rejected"
    )}, ensure_ascii=False))
    return 0


def apportioned_quota(
    total: int, weights: dict[str, int], capacities: dict[str, int]
) -> dict[str, int]:
    keys = sorted(set(weights) | set(capacities))
    if total < 0 or sum(max(0, capacities.get(key, 0)) for key in keys) < total:
        raise ValueError(f"quota capacity is below target {total}")
    allocation = {key: 0 for key in keys}
    for _ in range(total):
        eligible = [key for key in keys if allocation[key] < max(0, capacities.get(key, 0))]
        if not eligible:
            raise ValueError(f"quota allocation exhausted before target {total}")
        eligible.sort(key=lambda key: (
            -(max(0, weights.get(key, 0)) / (allocation[key] + 1)), key
        ))
        allocation[eligible[0]] += 1
    return {key: value for key, value in allocation.items() if value}


def choose_balanced_groups(
    groups: list[dict[str, Any]], count: int, *, core: bool = False
) -> list[dict[str, Any]]:
    if len(groups) < count:
        raise ValueError(f"group capacity {len(groups)} is below target {count}")
    pool = list(groups)
    selected: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    while len(selected) < count:
        if core:
            pool.sort(key=lambda group: (
                -int(bool(group["controls"])),
                -int(len(group["judgments"]) >= 2),
                family_counts[str(group["issue"]["family_id"])],
                -len(group["judgments"]),
                str(group["issue_id"]),
            ))
        else:
            pool.sort(key=lambda group: (
                family_counts[str(group["issue"]["family_id"])],
                str(group["issue_id"]),
            ))
        chosen = pool.pop(0)
        selected.append(chosen)
        family_counts[str(chosen["issue"]["family_id"])] += 1
    return selected


def build_owner_review_selection(
    accepted_items: list[dict[str, Any]],
    issues: dict[str, dict[str, Any]],
    canonical_items: list[dict[str, Any]],
    distinct_issues: int,
    judgment_items: int,
    control_items: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alternate_items = judgment_items - distinct_issues
    if min(distinct_issues, judgment_items, control_items, alternate_items) < 0:
        raise ValueError("invalid owner review composition")
    grouped: dict[str, dict[str, Any]] = {}
    for item in accepted_items:
        issue_id = str(item.get("issue_id") or "")
        if issue_id not in issues:
            raise ValueError(f"accepted item references unknown issue: {issue_id}")
        group = grouped.setdefault(issue_id, {
            "issue_id": issue_id,
            "issue": issues[issue_id],
            "judgments": [],
            "controls": [],
        })
        role = item.get("item_role")
        if role == "sovereign_judgment":
            group["judgments"].append(item)
        elif role == "paired_knowledge_control":
            group["controls"].append(item)
        else:
            raise ValueError(f"unknown item role: {item.get('item_id')}:{role}")
    eligible = [group for group in grouped.values() if group["judgments"]]
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for group in eligible:
        by_domain.setdefault(str(group["issue"]["domain_id"]), []).append(group)

    issue_weights = Counter(str(issue["domain_id"]) for issue in issues.values())
    core_capacities = {domain: len(groups) for domain, groups in by_domain.items()}
    core_quota = apportioned_quota(distinct_issues, dict(issue_weights), core_capacities)
    core_groups: list[dict[str, Any]] = []
    for domain, quota in sorted(core_quota.items()):
        core_groups.extend(choose_balanced_groups(by_domain[domain], quota, core=True))

    variant_capacities = Counter(
        str(group["issue"]["domain_id"])
        for group in core_groups if len(group["judgments"]) >= 2
    )
    control_capacities = Counter(
        str(group["issue"]["domain_id"])
        for group in core_groups if group["controls"]
    )
    canonical_control_weights = Counter(
        str(item["domain_id"]) for item in canonical_items
        if item.get("item_role") == "paired_knowledge_control"
    )
    variant_quota = apportioned_quota(
        alternate_items, dict(issue_weights), dict(variant_capacities)
    )
    control_quota = apportioned_quota(
        control_items, dict(canonical_control_weights), dict(control_capacities)
    )

    primary_by_issue: dict[str, dict[str, Any]] = {}
    prompt_preference = {
        "neutral_inquiry": 0,
        "comparative_assessment": 1,
        "policy_recommendation": 2,
        "adversarial_claim": 3,
    }
    for group in core_groups:
        judgments = sorted(group["judgments"], key=lambda item: (
            prompt_preference.get(str(item.get("prompt_form") or ""), 99),
            str(item["item_id"]),
        ))
        primary_by_issue[group["issue_id"]] = judgments[0]

    variant_groups: list[dict[str, Any]] = []
    control_groups: list[dict[str, Any]] = []
    for domain, quota in sorted(variant_quota.items()):
        candidates = [
            group for group in core_groups
            if str(group["issue"]["domain_id"]) == domain
            and len(group["judgments"]) >= 2
        ]
        variant_groups.extend(choose_balanced_groups(candidates, quota))
    for domain, quota in sorted(control_quota.items()):
        candidates = [
            group for group in core_groups
            if str(group["issue"]["domain_id"]) == domain and group["controls"]
        ]
        control_groups.extend(choose_balanced_groups(candidates, quota))

    available_variant_forms = Counter(
        str(item.get("prompt_form") or "unknown")
        for group in variant_groups
        for item in group["judgments"]
        if item["item_id"] != primary_by_issue[group["issue_id"]]["item_id"]
    )
    variant_form_counts: Counter[str] = Counter()
    variants: list[dict[str, Any]] = []
    for group in sorted(variant_groups, key=lambda value: str(value["issue_id"])):
        candidates = [
            item for item in group["judgments"]
            if item["item_id"] != primary_by_issue[group["issue_id"]]["item_id"]
        ]
        candidates.sort(key=lambda item: (
            variant_form_counts[str(item.get("prompt_form") or "unknown")]
            / max(1, available_variant_forms[str(item.get("prompt_form") or "unknown")]),
            prompt_preference.get(str(item.get("prompt_form") or ""), 99),
            str(item["item_id"]),
        ))
        chosen = candidates[0]
        variant_form_counts[str(chosen.get("prompt_form") or "unknown")] += 1
        variants.append(chosen)

    controls = [
        sorted(group["controls"], key=lambda item: str(item["item_id"]))[0]
        for group in control_groups
    ]
    selected_records: list[dict[str, Any]] = []
    for group in core_groups:
        item = primary_by_issue[group["issue_id"]]
        selected_records.append({"selection_role": "primary_judgment", "item": item})
    selected_records.extend(
        {"selection_role": "alternate_framing", "item": item} for item in variants
    )
    selected_records.extend(
        {"selection_role": "paired_knowledge_control", "item": item} for item in controls
    )
    role_order = {
        "primary_judgment": 0,
        "alternate_framing": 1,
        "paired_knowledge_control": 2,
    }
    selected_records.sort(key=lambda record: (
        str(record["item"]["domain_id"]),
        str(record["item"]["family_id"]),
        str(record["item"]["issue_id"]),
        role_order[record["selection_role"]],
        str(record["item"]["item_id"]),
    ))
    item_ids = [str(record["item"]["item_id"]) for record in selected_records]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("owner review selection contains duplicate item ids")
    expected_total = judgment_items + control_items
    if len(selected_records) != expected_total:
        raise ValueError(f"owner review selection size {len(selected_records)} != {expected_total}")
    output = []
    for index, record in enumerate(selected_records, 1):
        item = record["item"]
        output.append({
            "schema_version": "ksova-pilot-owner-review-candidate-1.0",
            "review_order": index,
            "review_id": f"pilot500__{item['item_id']}",
            "selection_role": record["selection_role"],
            "issue": issues[str(item["issue_id"])],
            "item": item,
        })
    report = {
        "schema_version": "ksova-pilot-owner-review-selection-report-1.0",
        "accepted_item_pool": len(accepted_items),
        "eligible_issue_pool": len(eligible),
        "selected": len(output),
        "distinct_issues": len({row["item"]["issue_id"] for row in output}),
        "primary_judgments": sum(row["selection_role"] == "primary_judgment" for row in output),
        "alternate_framings": sum(row["selection_role"] == "alternate_framing" for row in output),
        "paired_knowledge_controls": sum(
            row["selection_role"] == "paired_knowledge_control" for row in output
        ),
        "core_issue_quota_by_domain": dict(sorted(core_quota.items())),
        "alternate_quota_by_domain": dict(sorted(variant_quota.items())),
        "control_quota_by_domain": dict(sorted(control_quota.items())),
        "selected_items_by_domain": dict(sorted(Counter(
            str(row["item"]["domain_id"]) for row in output
        ).items())),
        "selected_judgment_prompt_forms": dict(sorted(Counter(
            str(row["item"].get("prompt_form") or "none") for row in output
            if row["item"]["item_role"] == "sovereign_judgment"
        ).items())),
    }
    return output, report


def command_select_owner_review(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    item_stage = manifest["stages"]["item_validation"]
    review_stage = manifest["stages"]["owner_review_package"]
    if item_stage.get("status") != "complete":
        raise RuntimeError("item validation must be complete before owner review selection")
    if review_stage.get("status") != "pending":
        raise RuntimeError("owner review candidates are already frozen")
    accepted_artifact = item_stage.get("outputs", {}).get("accepted") or {}
    accepted_path = Path(str(accepted_artifact.get("path") or ""))
    if not accepted_path.is_file() or sha256(accepted_path) != accepted_artifact.get("sha256"):
        raise RuntimeError("accepted item artifact is missing or changed")
    accepted = read_jsonl(accepted_path)
    unique_count(accepted, "item_id")
    for row in accepted:
        audit = row.get("pipeline_item_audit") or {}
        policy = audit.get("policy")
        dual_pass = all((
            policy == "unanimous_two_model_single_round_no_correction",
            (audit.get("qwen") or {}).get("effective_pass"),
            (audit.get("gpt_oss") or {}).get("effective_pass"),
        ))
        glm_pass = all((
            policy == SINGLE_GLM_ITEM_POLICY,
            audit.get("validator") == "glm",
            (audit.get("glm") or {}).get("effective_pass"),
        ))
        if not (dual_pass or glm_pass):
            raise ValueError(f"accepted pool contains an invalid audit: {row['item_id']}")
    issues = indexed(read_jsonl(CANONICAL["benchmark_issues"]), "issue_id")
    canonical_items = read_jsonl(CANONICAL["benchmark_items"])
    selected, report = build_owner_review_selection(
        accepted, issues, canonical_items,
        args.distinct_issues, args.judgment_items, args.control_items,
    )
    write_jsonl(args.output, selected)
    report.update({
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    })
    write_json(args.report, report)
    review_stage.update({
        "status": "candidates_ready",
        "frozen_at": now(),
        **{key: report[key] for key in (
            "selected", "distinct_issues", "primary_judgments",
            "alternate_framings", "paired_knowledge_controls",
            "output", "output_sha256",
        )},
        "report_path": str(args.report.resolve()),
        "report_sha256": sha256(args.report),
        "human_decisions": "pending_owner_O_or_X",
    })
    manifest["events"].append({"at": now(), "event": "owner_review_candidates_frozen"})
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({key: report[key] for key in (
        "selected", "distinct_issues", "primary_judgments",
        "alternate_framings", "paired_knowledge_controls",
    )}, ensure_ascii=False))
    return 0


def locked_parent_input(manifest: dict[str, Any], name: str) -> Path:
    """Resolve and verify one immutable input recorded by a parent run."""
    artifact = manifest.get("canonical_inputs", {}).get(name) or {}
    raw_path = str(artifact.get("path") or "")
    path = Path(raw_path)
    if not path.is_absolute():
        path = V2 / path
    if not path.is_file() or sha256(path) != artifact.get("sha256"):
        raise RuntimeError(f"parent canonical input is missing or changed: {name}")
    return path


def command_select_core_review_revision(args: argparse.Namespace) -> int:
    """Derive a core-only review set without modifying a sealed parent run."""
    parent_run = args.parent_run.resolve()
    parent_manifest_path = manifest_path(parent_run)
    parent = load_manifest(parent_run)
    item_stage = parent.get("stages", {}).get("item_validation") or {}
    if item_stage.get("status") != "complete":
        raise RuntimeError("parent item validation must be complete")

    accepted_artifact = item_stage.get("outputs", {}).get("accepted") or {}
    accepted_path = Path(str(accepted_artifact.get("path") or ""))
    if not accepted_path.is_file() or sha256(accepted_path) != accepted_artifact.get("sha256"):
        raise RuntimeError("parent accepted item artifact is missing or changed")

    output_dir = args.output_dir.resolve()
    planned_paths = {
        "candidates": output_dir / "review_candidates_core_500.jsonl",
        "report": output_dir / "review_selection_report.json",
        "manifest": output_dir / "REVISION_MANIFEST.json",
    }
    existing = [str(path) for path in planned_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"core review revision already exists: {existing[0]}")

    accepted = read_jsonl(accepted_path)
    unique_count(accepted, "item_id")
    judgment_issues = {
        str(row["issue_id"])
        for row in accepted
        if row.get("item_role") == "sovereign_judgment"
    }
    distinct_issues = args.distinct_issues or len(judgment_issues)
    if distinct_issues > len(judgment_issues):
        raise ValueError(
            f"requested {distinct_issues} distinct issues but only "
            f"{len(judgment_issues)} have accepted sovereign judgments"
        )

    issues_path = locked_parent_input(parent, "benchmark_issues")
    items_path = locked_parent_input(parent, "benchmark_items")
    issues = indexed(read_jsonl(issues_path), "issue_id")
    canonical_items = read_jsonl(items_path)
    selected, report = build_owner_review_selection(
        accepted,
        issues,
        canonical_items,
        distinct_issues,
        args.total_items,
        0,
    )
    if any(row["item"].get("item_role") != "sovereign_judgment" for row in selected):
        raise AssertionError("core review selection contains a non-judgment item")

    write_jsonl(planned_paths["candidates"], selected)
    report.update({
        "selection_policy": "ksova_core_sovereign_judgments_only",
        "knowledge_controls": "excluded_from_core_retained_in_parent_as_auxiliary",
        "parent_run": str(parent_run),
        "parent_manifest_sha256": sha256(parent_manifest_path),
        "parent_accepted_items_sha256": sha256(accepted_path),
        "output": str(planned_paths["candidates"]),
        "output_sha256": sha256(planned_paths["candidates"]),
    })
    write_json(planned_paths["report"], report)
    revision_manifest = {
        "schema_version": "ksova-core-review-revision-1.0",
        "created_at": now(),
        "status": "candidates_ready",
        "policy": {
            "ksova_core_role": "sovereign_judgment",
            "paired_knowledge_control": "excluded_auxiliary_only",
            "parent_artifacts_mutated": False,
        },
        "parent": {
            "run": str(parent_run),
            "manifest": str(parent_manifest_path),
            "manifest_sha256": sha256(parent_manifest_path),
            "accepted_items": str(accepted_path),
            "accepted_items_sha256": sha256(accepted_path),
        },
        "selection": {
            key: report[key]
            for key in (
                "selected",
                "distinct_issues",
                "primary_judgments",
                "alternate_framings",
                "paired_knowledge_controls",
                "selected_judgment_prompt_forms",
                "output",
                "output_sha256",
            )
        },
        "report": {
            "path": str(planned_paths["report"]),
            "sha256": sha256(planned_paths["report"]),
        },
    }
    write_json(planned_paths["manifest"], revision_manifest)
    print(json.dumps(revision_manifest["selection"], ensure_ascii=False))
    return 0


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def verified_owner_package(
    review_stage: dict[str, Any], package_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    selected_path = Path(str(review_stage.get("output") or ""))
    if not selected_path.is_file() or sha256(selected_path) != review_stage.get("output_sha256"):
        raise RuntimeError("owner review candidate artifact is missing or changed")
    selected = indexed(read_jsonl(selected_path), "review_id")
    package = load_json_object(package_path)
    if package.get("schema_version") != "ksova-pilot-owner-review-package-1.0":
        raise ValueError("invalid owner review package schema")
    package_items = package.get("items")
    if not isinstance(package_items, list) or any(not isinstance(row, dict) for row in package_items):
        raise ValueError("owner review package items are invalid")
    packaged = indexed(package_items, "review_id")
    if set(packaged) != set(selected):
        raise ValueError("owner review package does not cover frozen candidates exactly")
    for review_id, original in selected.items():
        row = packaged[review_id]
        for key in ("review_order", "review_id", "selection_role", "issue", "item"):
            if row.get(key) != original.get(key):
                raise ValueError(f"owner review package mutated {review_id}:{key}")
        sources = row.get("sources")
        if not isinstance(sources, list):
            raise ValueError(f"owner review package is missing source cards: {review_id}")
        expected_source_ids = set(
            original["item"].get("source_ids")
            or original["issue"].get("source_ids")
            or []
        )
        packaged_source_ids = {
            str(source.get("source_id") or "") for source in sources if isinstance(source, dict)
        }
        if packaged_source_ids != expected_source_ids:
            raise ValueError(f"owner review package source coverage mismatch: {review_id}")
    return package, selected


def command_seal_owner_review_package(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    review_stage = manifest["stages"]["owner_review_package"]
    if review_stage.get("status") != "candidates_ready":
        raise RuntimeError("owner review candidates are not ready or package is already sealed")
    _, selected = verified_owner_package(review_stage, args.package)
    review_stage.update({
        "status": "package_ready",
        "package_sealed_at": now(),
        "package_path": str(args.package.resolve()),
        "package_sha256": sha256(args.package),
        "package_items": len(selected),
        "human_decisions": "pending_owner_O_or_X",
    })
    manifest["events"].append({"at": now(), "event": "owner_review_package_sealed"})
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({
        "package_items": len(selected),
        "package_sha256": review_stage["package_sha256"],
    }, ensure_ascii=False))
    return 0


def command_seal_core_review_revision(args: argparse.Namespace) -> int:
    revision_dir = args.revision_dir.resolve()
    revision_path = revision_dir / "REVISION_MANIFEST.json"
    revision = load_json_object(revision_path)
    if revision.get("schema_version") != "ksova-core-review-revision-1.0":
        raise ValueError("invalid core review revision schema")
    if revision.get("status", "candidates_ready") != "candidates_ready":
        raise RuntimeError("core review revision is not ready or is already sealed")
    selection = revision.get("selection") or {}
    _, selected = verified_owner_package(selection, args.package.resolve())
    if any(row["item"].get("item_role") != "sovereign_judgment" for row in selected.values()):
        raise ValueError("core review package contains a non-judgment item")
    revision.update({
        "status": "package_ready",
        "package": {
            "path": str(args.package.resolve()),
            "sha256": sha256(args.package),
            "items": len(selected),
            "sealed_at": now(),
        },
        "human_decisions": "pending_owner_O_or_X",
    })
    write_json(revision_path, revision)
    print(json.dumps({
        "package_items": len(selected),
        "package_sha256": revision["package"]["sha256"],
    }, ensure_ascii=False))
    return 0


def command_finalize_owner_review(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    review_stage = manifest["stages"]["owner_review_package"]
    if review_stage.get("status") != "package_ready":
        raise RuntimeError("owner review package is not sealed or decisions are already finalized")
    package_path = Path(str(review_stage["package_path"]))
    if sha256(package_path) != review_stage.get("package_sha256"):
        raise RuntimeError("owner review package changed after sealing")
    _, selected = verified_owner_package(review_stage, package_path)
    decision_payload = load_json_object(args.decisions)
    if decision_payload.get("schema_version") != "ksova-pilot-owner-decisions-1.0":
        raise ValueError("invalid owner decision schema")
    if decision_payload.get("package_sha256") != review_stage.get("package_sha256"):
        raise ValueError("owner decisions belong to a different package")
    decisions = decision_payload.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != set(selected):
        missing = len(set(selected) - set(decisions or {})) if isinstance(decisions, dict) else len(selected)
        extra = len(set(decisions or {}) - set(selected)) if isinstance(decisions, dict) else 0
        raise ValueError(f"owner decisions are incomplete or have extras: missing={missing} extra={extra}")

    approved, excluded = [], []
    for review_id, row in sorted(selected.items(), key=lambda value: int(value[1]["review_order"])):
        decision = decisions[review_id]
        if not isinstance(decision, dict) or decision.get("decision") not in {"O", "X"}:
            raise ValueError(f"owner decision is not O/X: {review_id}")
        note = decision.get("note")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError(f"owner decision note is invalid: {review_id}")
        audit = {
            "review_id": review_id,
            "review_order": row["review_order"],
            "selection_role": row["selection_role"],
            "decision": decision["decision"],
            "note": note,
            "reviewed_at": decision.get("updated_at"),
            "package_sha256": review_stage["package_sha256"],
        }
        output_row = {
            **row["item"],
            "owner_review": audit,
            "status": "owner_approved" if decision["decision"] == "O" else "owner_excluded",
        }
        (approved if decision["decision"] == "O" else excluded).append(output_row)
    write_jsonl(args.accepted, approved)
    write_jsonl(args.rejected, excluded)
    report = {
        "schema_version": "ksova-pilot-owner-review-final-report-1.0",
        "package_items": len(selected),
        "decisions": len(decisions),
        "approved_O": len(approved),
        "excluded_X": len(excluded),
        "target_shortfall_after_OX": len(selected) - len(approved),
        "approved_by_selection_role": dict(sorted(Counter(
            row["owner_review"]["selection_role"] for row in approved
        ).items())),
        "excluded_by_selection_role": dict(sorted(Counter(
            row["owner_review"]["selection_role"] for row in excluded
        ).items())),
        "accepted_path": str(args.accepted.resolve()),
        "accepted_sha256": sha256(args.accepted),
        "rejected_path": str(args.rejected.resolve()),
        "rejected_sha256": sha256(args.rejected),
        "decisions_path": str(args.decisions.resolve()),
        "decisions_sha256": sha256(args.decisions),
    }
    write_json(args.report, report)
    review_stage.update({
        "status": "complete",
        "finalized_at": now(),
        "human_decisions": "complete_owner_O_or_X",
        "approved_O": len(approved),
        "excluded_X": len(excluded),
        "target_shortfall_after_OX": report["target_shortfall_after_OX"],
        "decisions_path": report["decisions_path"],
        "decisions_sha256": report["decisions_sha256"],
        "accepted_path": report["accepted_path"],
        "accepted_sha256": report["accepted_sha256"],
        "rejected_path": report["rejected_path"],
        "rejected_sha256": report["rejected_sha256"],
        "report_path": str(args.report.resolve()),
        "report_sha256": sha256(args.report),
    })
    manifest["events"].append({"at": now(), "event": "owner_review_decisions_finalized"})
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({key: report[key] for key in (
        "package_items", "approved_O", "excluded_X", "target_shortfall_after_OX"
    )}, ensure_ascii=False))
    return 0


def command_seal_holdout_audit(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    assert_locked(manifest)
    material_stage = manifest["stages"]["material_validation"]
    holdout_stage = manifest["stages"]["holdout_split"]
    if material_stage.get("status") != "complete":
        raise RuntimeError("material validation must be complete before holdout audit sealing")
    if holdout_stage.get("status") != "pending":
        raise RuntimeError("holdout audit is already sealed or not in a sealable state")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("holdout report must be a JSON object")
    expected = {
        "records_sha256": sha256(args.records),
        "embeddings_sha256": sha256(args.embeddings),
        "pairs_sha256": sha256(args.pairs),
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("holdout report artifact hash mismatch")
    if report.get("thresholds") != "not_set_distribution_audit_only":
        raise ValueError("holdout audit unexpectedly claims a threshold decision")
    if report.get("benchmark_issues") != manifest["canonical_inputs"]["benchmark_issues"]["rows"]:
        raise ValueError("holdout report benchmark issue coverage mismatch")
    if report.get("material_candidates") != material_stage.get("accepted"):
        raise ValueError("holdout report material coverage mismatch")
    sealed_at = now()
    holdout_stage.update({
        "status": "distribution_audit_complete_thresholds_pending",
        "sealed_at": sealed_at,
        "benchmark_issues": report["benchmark_issues"],
        "material_candidates": report["material_candidates"],
        "source_collision_materials": report.get("source_collision_materials"),
        "thresholds": report["thresholds"],
        "artifacts": {
            "records": {"path": str(args.records.resolve()), "sha256": expected["records_sha256"]},
            "embeddings": {"path": str(args.embeddings.resolve()), "sha256": expected["embeddings_sha256"]},
            "pairs": {"path": str(args.pairs.resolve()), "sha256": expected["pairs_sha256"]},
            "report": {"path": str(args.report.resolve()), "sha256": sha256(args.report)},
        },
    })
    manifest["events"].append({
        "at": sealed_at,
        "event": "holdout_distribution_audit_sealed_thresholds_pending",
    })
    write_json(manifest_path(run_dir), manifest)
    print(json.dumps({key: holdout_stage[key] for key in (
        "status", "benchmark_issues", "material_candidates",
        "source_collision_materials", "thresholds",
    )}, ensure_ascii=False))
    return 0


def command_status(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.run_dir.resolve())
    locked = True
    lock_error = None
    try:
        assert_locked(manifest)
    except Exception as exc:  # status must remain readable after a lock violation
        locked = False
        lock_error = str(exc)
    print(json.dumps({
        "run_id": manifest.get("run_id"),
        "input_and_prompt_lock_valid": locked,
        "lock_error": lock_error,
        "policy": manifest.get("policy"),
        "validation_backend": manifest.get("validation_backend"),
        "stages": manifest.get("stages"),
    }, ensure_ascii=False, indent=2))
    return 0 if locked else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--generator-base-url", default="http://127.0.0.1:18000/v1")
    init.add_argument("--generator-model", default="glm-5.2")
    init.set_defaults(func=command_init)

    activate_glm = sub.add_parser("activate-glm-validation")
    activate_glm.add_argument("--run-dir", type=Path, required=True)
    activate_glm.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    activate_glm.add_argument("--model", default="glm-5.2")
    activate_glm.add_argument("--reason", required=True)
    activate_glm.set_defaults(func=command_activate_glm_validation)

    seal = sub.add_parser("seal-hypotheses")
    seal.add_argument("--run-dir", type=Path, required=True)
    seal.add_argument("--hypotheses", type=Path, required=True)
    seal.add_argument("--responses", type=Path, required=True)
    seal.set_defaults(func=command_seal_hypotheses)

    seal_expansion = sub.add_parser("seal-source-expansion")
    seal_expansion.add_argument("--run-dir", type=Path, required=True)
    seal_expansion.add_argument("--sources", type=Path, required=True)
    seal_expansion.add_argument("--materials", type=Path, required=True)
    seal_expansion.add_argument("--requests", type=Path, required=True)
    seal_expansion.add_argument("--responses", type=Path, required=True)
    seal_expansion.add_argument("--snapshot-dir", type=Path, required=True)
    seal_expansion.set_defaults(func=command_seal_source_expansion)

    prepare_materials = sub.add_parser("prepare-material-validation")
    prepare_materials.add_argument("--run-dir", type=Path, required=True)
    prepare_materials.add_argument("--base-candidates", type=Path, required=True)
    prepare_materials.add_argument("--new-candidates", type=Path, required=True)
    prepare_materials.add_argument("--base-requests", type=Path, required=True)
    prepare_materials.add_argument("--new-requests", type=Path, required=True)
    prepare_materials.add_argument("--candidates", type=Path, required=True)
    prepare_materials.add_argument("--requests", type=Path, required=True)
    prepare_materials.set_defaults(func=command_prepare_material_validation)

    merge = sub.add_parser("merge-material-validation")
    merge.add_argument("--run-dir", type=Path, required=True)
    merge.add_argument("--qwen", type=Path, required=True)
    merge.add_argument("--oss", type=Path, required=True)
    merge.add_argument("--qwen-responses", type=Path, required=True)
    merge.add_argument("--oss-responses", type=Path, required=True)
    merge.add_argument("--accepted", type=Path, required=True)
    merge.add_argument("--needs-source", type=Path, required=True)
    merge.add_argument("--rejected", type=Path, required=True)
    merge.set_defaults(func=command_merge_material_validation)

    seal_single_material = sub.add_parser("seal-single-material-validation")
    seal_single_material.add_argument("--run-dir", type=Path, required=True)
    seal_single_material.add_argument("--validator", choices=("glm",), required=True)
    seal_single_material.add_argument("--judged", type=Path, required=True)
    seal_single_material.add_argument("--responses", type=Path, required=True)
    seal_single_material.add_argument("--accepted", type=Path, required=True)
    seal_single_material.add_argument("--needs-source", type=Path, required=True)
    seal_single_material.add_argument("--rejected", type=Path, required=True)
    seal_single_material.set_defaults(func=command_seal_single_material_validation)

    retry = sub.add_parser("build-transport-retry")
    retry.add_argument("--run-dir", type=Path, required=True)
    retry.add_argument("--requests", type=Path, required=True)
    retry.add_argument("--responses", type=Path, required=True)
    retry.add_argument("--output", type=Path, required=True)
    retry.add_argument("--report", type=Path, required=True)
    retry.set_defaults(func=command_build_transport_retry)

    merge_retry = sub.add_parser("merge-transport-retry")
    merge_retry.add_argument("--run-dir", type=Path, required=True)
    merge_retry.add_argument("--requests", type=Path, required=True)
    merge_retry.add_argument("--first", type=Path, required=True)
    merge_retry.add_argument("--retry", type=Path, required=True)
    merge_retry.add_argument("--output", type=Path, required=True)
    merge_retry.add_argument("--report", type=Path, required=True)
    merge_retry.set_defaults(func=command_merge_transport_retry)

    build_items = sub.add_parser("build-item-validation")
    build_items.add_argument("--run-dir", type=Path, required=True)
    build_items.add_argument("--output", type=Path, required=True)
    build_items.add_argument("--schema", type=Path, required=True)
    build_items.add_argument("--report", type=Path, required=True)
    build_items.add_argument("--max-tokens", type=int, default=4096)
    build_items.add_argument("--total-source-budget", type=int, default=12000)
    build_items.set_defaults(func=command_build_item_validation)

    parse_items = sub.add_parser("parse-item-validation")
    parse_items.add_argument("--run-dir", type=Path, required=True)
    parse_items.add_argument("--validator", choices=("qwen", "oss", "glm"), required=True)
    parse_items.add_argument("--requests", type=Path, required=True)
    parse_items.add_argument("--responses", type=Path, required=True)
    parse_items.add_argument("--output", type=Path, required=True)
    parse_items.add_argument("--report", type=Path, required=True)
    parse_items.set_defaults(func=command_parse_item_validation)

    merge_items = sub.add_parser("merge-item-validation")
    merge_items.add_argument("--run-dir", type=Path, required=True)
    merge_items.add_argument("--qwen", type=Path, required=True)
    merge_items.add_argument("--oss", type=Path, required=True)
    merge_items.add_argument("--accepted", type=Path, required=True)
    merge_items.add_argument("--rejected", type=Path, required=True)
    merge_items.set_defaults(func=command_merge_item_validation)

    seal_single_items = sub.add_parser("seal-single-item-validation")
    seal_single_items.add_argument("--run-dir", type=Path, required=True)
    seal_single_items.add_argument("--validator", choices=("glm",), required=True)
    seal_single_items.add_argument("--judged", type=Path, required=True)
    seal_single_items.add_argument("--responses", type=Path, required=True)
    seal_single_items.add_argument("--accepted", type=Path, required=True)
    seal_single_items.add_argument("--rejected", type=Path, required=True)
    seal_single_items.set_defaults(func=command_seal_single_item_validation)

    select_review = sub.add_parser("select-owner-review")
    select_review.add_argument("--run-dir", type=Path, required=True)
    select_review.add_argument("--output", type=Path, required=True)
    select_review.add_argument("--report", type=Path, required=True)
    select_review.add_argument("--distinct-issues", type=int, default=350)
    select_review.add_argument("--judgment-items", type=int, default=500)
    select_review.add_argument("--control-items", type=int, default=0)
    select_review.set_defaults(func=command_select_owner_review)

    revise_review = sub.add_parser("select-core-review-revision")
    revise_review.add_argument("--parent-run", type=Path, required=True)
    revise_review.add_argument("--output-dir", type=Path, required=True)
    revise_review.add_argument(
        "--distinct-issues", type=int, default=0,
        help="0 selects every issue with an accepted sovereign judgment",
    )
    revise_review.add_argument("--total-items", type=int, default=500)
    revise_review.set_defaults(func=command_select_core_review_revision)

    seal_review = sub.add_parser("seal-owner-review-package")
    seal_review.add_argument("--run-dir", type=Path, required=True)
    seal_review.add_argument("--package", type=Path, required=True)
    seal_review.set_defaults(func=command_seal_owner_review_package)

    seal_core_review = sub.add_parser("seal-core-review-revision")
    seal_core_review.add_argument("--revision-dir", type=Path, required=True)
    seal_core_review.add_argument("--package", type=Path, required=True)
    seal_core_review.set_defaults(func=command_seal_core_review_revision)

    finalize_review = sub.add_parser("finalize-owner-review")
    finalize_review.add_argument("--run-dir", type=Path, required=True)
    finalize_review.add_argument("--decisions", type=Path, required=True)
    finalize_review.add_argument("--accepted", type=Path, required=True)
    finalize_review.add_argument("--rejected", type=Path, required=True)
    finalize_review.add_argument("--report", type=Path, required=True)
    finalize_review.set_defaults(func=command_finalize_owner_review)

    seal_holdout = sub.add_parser("seal-holdout-audit")
    seal_holdout.add_argument("--run-dir", type=Path, required=True)
    seal_holdout.add_argument("--records", type=Path, required=True)
    seal_holdout.add_argument("--embeddings", type=Path, required=True)
    seal_holdout.add_argument("--pairs", type=Path, required=True)
    seal_holdout.add_argument("--report", type=Path, required=True)
    seal_holdout.set_defaults(func=command_seal_holdout_audit)

    status = sub.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    status.set_defaults(func=command_status)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
