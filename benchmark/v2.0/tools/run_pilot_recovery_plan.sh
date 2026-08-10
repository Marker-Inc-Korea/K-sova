#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$TOOLS_DIR/../../.." && pwd)
PILOT="$REPO_ROOT/benchmark/v2.0/work/runs/pilot500_20260808"
CURRENT="$REPO_ROOT/benchmark/v2.0/work/current"
OUT_DIR=${1:-"$REPO_ROOT/benchmark/v2.0/work/runs/pipeline_recovery_20260809/reuse_plan"}

python3 "$TOOLS_DIR/recovery_pipeline.py" plan \
    --sources "$CURRENT/sources/source_registry_1576.jsonl" \
    --sources "$PILOT/source_expansion/trusted_sources.jsonl" \
    --material-candidates "$PILOT/material_validation/all_material_candidates.jsonl" \
    --material-passed "$PILOT/material_validation/accepted_materials.jsonl" \
    --material-needs-source "$PILOT/material_validation/needs_source_materials.jsonl" \
    --material-rejected "$PILOT/material_validation/rejected_materials.jsonl" \
    --material-validation-requests "$PILOT/material_validation/all_material_validation_requests.jsonl" \
    --issues "$CURRENT/materials/benchmark_issues_350.jsonl" \
    --items "$CURRENT/items/benchmark_candidates_1000.jsonl" \
    --item-accepted "$PILOT/item_validation/accepted_items.jsonl" \
    --item-rejected "$PILOT/item_validation/rejected_items.jsonl" \
    --item-validation-requests "$PILOT/item_validation/item_validation_requests.jsonl" \
    --max-judgments-per-issue 2 \
    --max-controls-per-issue 0 \
    --max-passed-materials-per-source 2 \
    --output-dir "$OUT_DIR"
