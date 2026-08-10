#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$TOOLS_DIR/../../.." && pwd)
RUN_ROOT="$REPO_ROOT/benchmark/v2.0/work/runs/training_scaleup_20260809_wave3"
POOL="$RUN_ROOT/next_material_pool/priority"
WAVE3_SFT="$RUN_ROOT/sft"
WAVE4_SFT="$RUN_ROOT/wave4_sft_transfer_v1"
CURRICULUM_ROOT="$REPO_ROOT/benchmark/v2.0/work/runs/training_curriculum_20260809_v1"
PRIOR_SFT="$CURRICULUM_ROOT/examples_selected.jsonl"
BASE_URL="http://127.0.0.1:18000/v1"
MODEL="glm-5.2"
BATCH_RUNNER="$REPO_ROOT/benchmark/v2.0/tools/run_chat_batch.py"
PREFLIGHT="$REPO_ROOT/benchmark/v2.0/tools/preflight_chat_context.py"
SOURCE_BUNDLE="$REPO_ROOT/benchmark/v2.0/tools/issue_source_bundle_pipeline.py"
SOURCE_DOCUMENT="$REPO_ROOT/benchmark/v2.0/tools/source_document_pipeline.py"
MATERIAL_SEAL="$REPO_ROOT/benchmark/v2.0/tools/seal_glm_training_materials.py"
MATERIAL_DEDUP="$REPO_ROOT/benchmark/v2.0/tools/material_semantic_dedup.py"
HOLDOUT_RECORDS="$REPO_ROOT/benchmark/v2.0/tools/build_holdout_embedding_records.py"
EMBED_RECORDS="$REPO_ROOT/benchmark/v2.0/tools/embed_holdout_records.py"
SFT_PIPELINE="$REPO_ROOT/benchmark/v2.0/tools/training_example_pipeline.py"
EXAMPLE_DEDUP="$REPO_ROOT/benchmark/v2.0/tools/training_example_semantic_dedup.py"
REGISTRY="$REPO_ROOT/benchmark/v2.0/work/current/sources/source_registry_1576.jsonl"
BENCHMARK_ISSUES="$REPO_ROOT/benchmark/v2.0/work/current/materials/benchmark_issues_350.jsonl"
PRIOR_MATERIALS="$REPO_ROOT/benchmark/v2.0/work/runs/training_expansion_20260808_wave2/semantic_dedup/materials_kept.jsonl"
EMBED_PYTHON="/data1/mk02/.venvs/vera-rag-qwen-v0.4-final/bin/python"

mkdir -p "$WAVE4_SFT"

if [[ ! -s "$PRIOR_SFT" ]]; then
    mkdir -p "$CURRICULUM_ROOT"
    python3 "$SFT_PIPELINE" select-curriculum \
        --examples "$WAVE3_SFT/examples_accepted_final.jsonl" \
        --max-examples-per-material 4 --max-examples-per-material-skill 1 \
        --output "$PRIOR_SFT" \
        --excluded "$CURRICULUM_ROOT/examples_excluded.jsonl" \
        --report "$CURRICULUM_ROOT/selection_report.json"
fi

source_summary="$POOL/source_bundle_validation_responses.summary.json"
for _ in $(seq 1 2160); do
    if [[ -f "$source_summary" ]]; then
        break
    fi
    sleep 20
done
if [[ ! -f "$source_summary" ]]; then
    echo "source-bundle validation summary did not appear within 12 hours" >&2
    exit 1
fi

if [[ $(jq -r '.errors' "$source_summary") != "0" ]]; then
    python3 "$BATCH_RUNNER" \
        --requests "$POOL/source_bundle_validation_requests_preflight.jsonl" \
        --base-url "$BASE_URL" --model "$MODEL" \
        --record-model-name glm-5.2__ksova-source-bundle-validator-wave3 \
        --out-dir "$POOL" --out-prefix source_bundle_validation \
        --output-file "$POOL/source_bundle_validation_responses.jsonl" \
        --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 1 \
        --temperature 0 \
        --chat-template-kwargs-json '{"enable_thinking":false}' \
        --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260811}' \
        --resume
fi

python3 "$SOURCE_BUNDLE" parse-validation \
    --verification-queue "$POOL/verification_queue.jsonl" \
    --requests "$POOL/source_bundle_validation_requests_preflight.jsonl" \
    --responses "$POOL/source_bundle_validation_responses.jsonl" \
    --output "$POOL/source_bundle_validation_judged.jsonl" \
    --complete "$POOL/source_bundle_complete.jsonl" \
    --needs-source "$POOL/source_bundle_needs_source.jsonl" \
    --rejected "$POOL/source_bundle_rejected.jsonl" \
    --source-audit "$POOL/source_link_audit.json" \
    --report "$POOL/source_bundle_validation_report.json"

if [[ ! -s "$POOL/source_bundle_complete.jsonl" ]]; then
    echo "no complete source bundles; wave4 stops fail-closed"
    exit 0
fi

jq -c 'if has("issue_id") then . else {
    issue_id:.material_id,
    title_ko:.specific_issue_ko,
    decision_core_ko:.decision_object_ko,
    korean_reference_direction_ko:.minimum_baseline_ko,
    sovereign_stake_ko:.korean_stake_ko,
    source_ids:.source_ids
} end' "$BENCHMARK_ISSUES" "$PRIOR_MATERIALS" > "$POOL/reference_issues_377.jsonl"

python3 "$SOURCE_DOCUMENT" build-material \
    --sources "$RUN_ROOT/next_material_pool/registry_candidates.jsonl" \
    --existing "$POOL/reference_issues_377.jsonl" \
    --source-audit "$POOL/source_link_audit.json" \
    --batch-size 4 --total-body-budget 7000 --related-existing-limit 14 \
    --max-tokens 4096 \
    --output "$POOL/material_generation_requests.jsonl" \
    --report "$POOL/material_generation_plan.json"

python3 "$PREFLIGHT" \
    --requests "$POOL/material_generation_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 3200 \
    --workers 4 \
    --output "$POOL/material_generation_requests_preflight.jsonl" \
    --report "$POOL/material_generation_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$POOL/material_generation_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-material-generation-wave4 \
    --out-dir "$POOL" --out-prefix material_generation \
    --output-file "$POOL/material_generation_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 1 \
    --temperature 0 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260813}' \
    --resume

python3 "$SOURCE_DOCUMENT" parse-material \
    --existing "$POOL/reference_issues_377.jsonl" \
    --requests "$POOL/material_generation_requests_preflight.jsonl" \
    --responses "$POOL/material_generation_responses.jsonl" \
    --output "$POOL/material_candidates.jsonl" \
    --report "$POOL/material_generation_parse_report.json"

if [[ ! -s "$POOL/material_candidates.jsonl" ]]; then
    echo "no structurally valid material candidates; wave4 stops fail-closed"
    exit 0
fi

python3 "$SOURCE_DOCUMENT" build-material-validation \
    --candidates "$POOL/material_candidates.jsonl" \
    --sources "$RUN_ROOT/next_material_pool/registry_candidates.jsonl" \
    --audit-version wave4_glm1 --total-body-budget 3600 \
    --related-existing-limit 4 --max-tokens 4096 \
    --output "$POOL/material_validation_requests.jsonl" \
    --report "$POOL/material_validation_plan.json"

python3 "$PREFLIGHT" \
    --requests "$POOL/material_validation_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 3000 \
    --workers 2 \
    --output "$POOL/material_validation_requests_preflight.jsonl" \
    --report "$POOL/material_validation_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$POOL/material_validation_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-material-validator-wave4 \
    --out-dir "$POOL" --out-prefix material_validation \
    --output-file "$POOL/material_validation_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 1 \
    --temperature 0 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260814}' \
    --resume

python3 "$SOURCE_DOCUMENT" parse-material-validation \
    --candidates "$POOL/material_candidates.jsonl" \
    --requests "$POOL/material_validation_requests_preflight.jsonl" \
    --responses "$POOL/material_validation_responses.jsonl" \
    --validation-key glm_material_validation_wave4 \
    --pass-threshold 85 --needs-source-threshold 78 \
    --output "$POOL/material_validation_judged.jsonl" \
    --passed "$POOL/material_passed.jsonl" \
    --needs-source "$POOL/material_needs_source.jsonl" \
    --report "$POOL/material_validation_report.json"

python3 "$MATERIAL_SEAL" \
    --input "$POOL/material_passed.jsonl" \
    --output "$POOL/materials_sealed.jsonl" \
    --quarantine "$POOL/materials_seal_quarantine.jsonl" \
    --report "$POOL/materials_seal_report.json"

if [[ ! -s "$POOL/materials_sealed.jsonl" ]]; then
    echo "no materials passed independent validation; wave4 stops fail-closed"
    exit 0
fi

python3 "$HOLDOUT_RECORDS" \
    --issues "$POOL/reference_issues_377.jsonl" \
    --materials "$POOL/materials_sealed.jsonl" \
    --output "$POOL/holdout_records.jsonl" \
    --report "$POOL/holdout_records_report.json"

HF_HOME="$RUN_ROOT/embedding_model_cache/hf_home" \
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
"$EMBED_PYTHON" "$EMBED_RECORDS" \
    --records "$POOL/holdout_records.jsonl" \
    --output "$POOL/holdout_embeddings.jsonl" \
    --report "$POOL/holdout_embedding_report.json" \
    --cache-dir "$RUN_ROOT/embedding_model_cache" \
    --batch-size 32 --max-length 512 --device cpu

python3 "$MATERIAL_DEDUP" build \
    --issues "$POOL/reference_issues_377.jsonl" \
    --materials "$POOL/materials_sealed.jsonl" \
    --embeddings "$POOL/holdout_embeddings.jsonl" \
    --nearest-benchmark 3 --nearest-material 4 --batch-size 8 \
    --pairs "$POOL/material_semantic_pairs.jsonl" \
    --output "$POOL/material_semantic_dedup_requests.jsonl" \
    --report "$POOL/material_semantic_dedup_plan.json"

python3 "$PREFLIGHT" \
    --requests "$POOL/material_semantic_dedup_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 1800 \
    --workers 2 \
    --output "$POOL/material_semantic_dedup_requests_preflight.jsonl" \
    --report "$POOL/material_semantic_dedup_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$POOL/material_semantic_dedup_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-material-semantic-dedup-wave4 \
    --out-dir "$POOL" --out-prefix material_semantic_dedup \
    --output-file "$POOL/material_semantic_dedup_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 1 \
    --temperature 0 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260815}' \
    --resume

python3 "$MATERIAL_DEDUP" parse \
    --materials "$POOL/materials_sealed.jsonl" \
    --pairs "$POOL/material_semantic_pairs.jsonl" \
    --requests "$POOL/material_semantic_dedup_requests_preflight.jsonl" \
    --responses "$POOL/material_semantic_dedup_responses.jsonl" \
    --judgments "$POOL/material_semantic_judgments.jsonl" \
    --kept "$POOL/materials_kept.jsonl" \
    --quarantine "$POOL/materials_semantic_quarantine.jsonl" \
    --report "$POOL/material_semantic_dedup_report.json"

if [[ ! -s "$POOL/materials_kept.jsonl" ]]; then
    echo "all new materials were semantic duplicates; wave4 stops fail-closed"
    exit 0
fi

python3 "$SFT_PIPELINE" build-generation \
    --materials "$POOL/materials_kept.jsonl" \
    --sources "$RUN_ROOT/next_material_pool/registry_candidates.jsonl" \
    --run-tag wave4 \
    --generation-profiles actor_decision counterargument_stress consequence_threshold \
    --avoid-examples "$PRIOR_SFT" \
    --max-profiles-per-material 3 --max-examples-per-material 4 \
    --output "$WAVE4_SFT/generation_requests.jsonl" \
    --report "$WAVE4_SFT/generation_plan_report.json"

python3 "$PREFLIGHT" \
    --requests "$WAVE4_SFT/generation_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 3400 \
    --workers 4 \
    --output "$WAVE4_SFT/generation_requests_preflight.jsonl" \
    --report "$WAVE4_SFT/generation_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$WAVE4_SFT/generation_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-sft-generation-wave4 \
    --out-dir "$WAVE4_SFT" --out-prefix generation \
    --output-file "$WAVE4_SFT/generation_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 2 \
    --temperature 0.2 --top-p 0.9 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260816}' \
    --resume

python3 "$SFT_PIPELINE" parse-generation \
    --requests "$WAVE4_SFT/generation_requests_preflight.jsonl" \
    --responses "$WAVE4_SFT/generation_responses.jsonl" \
    --output "$WAVE4_SFT/examples_candidates.jsonl" \
    --report "$WAVE4_SFT/generation_parse_report.json"

python3 "$SFT_PIPELINE" select-coverage \
    --candidates "$WAVE4_SFT/examples_candidates.jsonl" \
    --reference "$PRIOR_SFT" \
    --max-examples-per-material 4 \
    --max-examples-per-material-form 1 \
    --max-examples-per-material-skill 1 \
    --candidates-per-open-slot 2 \
    --output "$WAVE4_SFT/examples_candidates_for_validation.jsonl" \
    --overflow "$WAVE4_SFT/examples_candidates_coverage_overflow.jsonl" \
    --report "$WAVE4_SFT/coverage_selection_report.json"

python3 "$SFT_PIPELINE" build-validation \
    --materials "$POOL/materials_kept.jsonl" \
    --examples "$WAVE4_SFT/examples_candidates_for_validation.jsonl" \
    --sources "$RUN_ROOT/next_material_pool/registry_candidates.jsonl" \
    --run-tag wave4 \
    --output "$WAVE4_SFT/validation_requests.jsonl" \
    --report "$WAVE4_SFT/validation_plan_report.json"

python3 "$PREFLIGHT" \
    --requests "$WAVE4_SFT/validation_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 2800 \
    --workers 2 \
    --output "$WAVE4_SFT/validation_requests_preflight.jsonl" \
    --report "$WAVE4_SFT/validation_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$WAVE4_SFT/validation_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-sft-validator-wave4 \
    --out-dir "$WAVE4_SFT" --out-prefix validation \
    --output-file "$WAVE4_SFT/validation_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 2 \
    --temperature 0 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260817}' \
    --resume

python3 "$SFT_PIPELINE" parse-validation \
    --examples "$WAVE4_SFT/examples_candidates_for_validation.jsonl" \
    --requests "$WAVE4_SFT/validation_requests_preflight.jsonl" \
    --responses "$WAVE4_SFT/validation_responses.jsonl" \
    --accepted "$WAVE4_SFT/examples_accepted_wave4.jsonl" \
    --rejected "$WAVE4_SFT/examples_rejected_wave4.jsonl" \
    --report "$WAVE4_SFT/validation_report.json" --pass-threshold 85

python3 "$EXAMPLE_DEDUP" build \
    --reference "$PRIOR_SFT" \
    --candidates "$WAVE4_SFT/examples_accepted_wave4.jsonl" \
    --run-tag wave4 \
    --output "$WAVE4_SFT/semantic_dedup_requests.jsonl" \
    --report "$WAVE4_SFT/semantic_dedup_plan_report.json"

python3 "$PREFLIGHT" \
    --requests "$WAVE4_SFT/semantic_dedup_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 2200 \
    --workers 2 \
    --output "$WAVE4_SFT/semantic_dedup_requests_preflight.jsonl" \
    --report "$WAVE4_SFT/semantic_dedup_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$WAVE4_SFT/semantic_dedup_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-sft-semantic-dedup-wave4 \
    --out-dir "$WAVE4_SFT" --out-prefix semantic_dedup \
    --output-file "$WAVE4_SFT/semantic_dedup_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 2 \
    --temperature 0 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260818}' \
    --resume

python3 "$EXAMPLE_DEDUP" parse \
    --candidates "$WAVE4_SFT/examples_accepted_wave4.jsonl" \
    --requests "$WAVE4_SFT/semantic_dedup_requests_preflight.jsonl" \
    --responses "$WAVE4_SFT/semantic_dedup_responses.jsonl" \
    --kept "$WAVE4_SFT/examples_accepted_wave4_deduped.jsonl" \
    --quarantine "$WAVE4_SFT/examples_quarantined_semantic_duplicate.jsonl" \
    --report "$WAVE4_SFT/semantic_dedup_report.json"

python3 "$SFT_PIPELINE" combine \
    --inputs "$PRIOR_SFT" "$WAVE4_SFT/examples_accepted_wave4_deduped.jsonl" \
    --output "$WAVE4_SFT/examples_accepted_pre_final_combined.jsonl" \
    --report "$WAVE4_SFT/combine_report.json"

python3 "$SFT_PIPELINE" finalize \
    --accepted "$WAVE4_SFT/examples_accepted_pre_final_combined.jsonl" \
    --output "$WAVE4_SFT/examples_accepted_final.jsonl" \
    --quarantine "$WAVE4_SFT/examples_quarantined_final_gate.jsonl" \
    --report "$WAVE4_SFT/final_training_gate_report.json" --pass-threshold 85 \
    --max-examples-per-material 4 --max-examples-per-material-skill 1

python3 "$SFT_PIPELINE" export \
    --examples "$WAVE4_SFT/examples_accepted_final.jsonl" \
    --output "$WAVE4_SFT/training_messages.jsonl" \
    --report "$WAVE4_SFT/training_export_report.json"

sha256sum "$WAVE4_SFT/examples_accepted_final.jsonl" "$WAVE4_SFT/training_messages.jsonl" \
    > "$WAVE4_SFT/SHA256SUMS"
