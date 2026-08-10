#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$TOOLS_DIR/../../.." && pwd)
RUN_ROOT="$REPO_ROOT/benchmark/v2.0/work/runs/benchmark_x_training_expansion_20260809_v1"
PRIOR_ROOT="$REPO_ROOT/benchmark/v2.0/work/runs/benchmark_x_training_reuse_20260809_provisional"
PRIOR_SFT="$PRIOR_ROOT/examples_final.jsonl"
MATERIALS="$RUN_ROOT/materials.jsonl"
SOURCES="$REPO_ROOT/benchmark/v2.0/work/current/sources/source_registry_1576.jsonl"
BASE_URL="http://127.0.0.1:18000/v1"
MODEL="glm-5.2"
SFT_PIPELINE="$TOOLS_DIR/training_example_pipeline.py"
PREFLIGHT="$TOOLS_DIR/preflight_chat_context.py"
BATCH_RUNNER="$REPO_ROOT/benchmark/v2.0/tools/run_chat_batch.py"
EXAMPLE_DEDUP="$TOOLS_DIR/training_example_semantic_dedup.py"
RECOMMENDATIONS="$REPO_ROOT/benchmark/v2.0/work/runs/core_issue_selection_250_20260809_v1/recommendations.jsonl"
BENCHMARK_ISSUES="$REPO_ROOT/benchmark/v2.0/work/current/materials/benchmark_issues_350.jsonl"
TRAINING_MATERIALS="$REPO_ROOT/benchmark/v2.0/work/runs/training_expansion_20260808_wave2/sealed/materials_pre_dedup.jsonl"

mkdir -p "$RUN_ROOT"

python3 "$SFT_PIPELINE" build-generation \
    --materials "$MATERIALS" --sources "$SOURCES" \
    --run-tag benchmark_x_expansion_v1 \
    --generation-profiles actor_decision counterargument_stress consequence_threshold \
    --max-profiles-per-material 3 --max-examples-per-material 4 \
    --avoid-examples "$PRIOR_SFT" \
    --output "$RUN_ROOT/generation_requests.jsonl" \
    --report "$RUN_ROOT/generation_plan_report.json"

python3 "$PREFLIGHT" \
    --requests "$RUN_ROOT/generation_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 3000 \
    --workers 4 \
    --output "$RUN_ROOT/generation_requests_preflight.jsonl" \
    --report "$RUN_ROOT/generation_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$RUN_ROOT/generation_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-owner-x-generator \
    --out-dir "$RUN_ROOT" --out-prefix generation \
    --output-file "$RUN_ROOT/generation_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 3 \
    --temperature 0.2 --top-p 0.9 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260831}' \
    --resume

python3 "$SFT_PIPELINE" parse-generation \
    --requests "$RUN_ROOT/generation_requests_preflight.jsonl" \
    --responses "$RUN_ROOT/generation_responses.jsonl" \
    --output "$RUN_ROOT/examples_candidates.jsonl" \
    --report "$RUN_ROOT/generation_parse_report.json"

python3 "$SFT_PIPELINE" select-coverage \
    --candidates "$RUN_ROOT/examples_candidates.jsonl" \
    --reference "$PRIOR_SFT" \
    --max-examples-per-material 4 --max-examples-per-material-form 1 \
    --max-examples-per-material-skill 1 --candidates-per-open-slot 2 \
    --output "$RUN_ROOT/examples_candidates_for_validation.jsonl" \
    --overflow "$RUN_ROOT/examples_candidates_coverage_overflow.jsonl" \
    --report "$RUN_ROOT/coverage_selection_report.json"

if [[ ! -s "$RUN_ROOT/examples_candidates_for_validation.jsonl" ]]; then
    echo "owner-X generation produced no structurally valid candidates"
    exit 0
fi

python3 "$SFT_PIPELINE" build-validation \
    --materials "$MATERIALS" \
    --examples "$RUN_ROOT/examples_candidates_for_validation.jsonl" \
    --sources "$SOURCES" --run-tag benchmark_x_expansion_v1 \
    --output "$RUN_ROOT/validation_requests.jsonl" \
    --report "$RUN_ROOT/validation_plan_report.json"

python3 "$PREFLIGHT" \
    --requests "$RUN_ROOT/validation_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 2600 \
    --workers 3 \
    --output "$RUN_ROOT/validation_requests_preflight.jsonl" \
    --report "$RUN_ROOT/validation_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$RUN_ROOT/validation_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-owner-x-validator \
    --out-dir "$RUN_ROOT" --out-prefix validation \
    --output-file "$RUN_ROOT/validation_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 3 \
    --temperature 0 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260832}' \
    --resume

python3 "$SFT_PIPELINE" parse-validation \
    --examples "$RUN_ROOT/examples_candidates_for_validation.jsonl" \
    --requests "$RUN_ROOT/validation_requests_preflight.jsonl" \
    --responses "$RUN_ROOT/validation_responses.jsonl" \
    --accepted "$RUN_ROOT/examples_accepted.jsonl" \
    --rejected "$RUN_ROOT/examples_rejected.jsonl" \
    --report "$RUN_ROOT/validation_report.json" --pass-threshold 85

if [[ ! -s "$RUN_ROOT/examples_accepted.jsonl" ]]; then
    echo "no generated owner-X examples passed validation"
    exit 0
fi

python3 "$EXAMPLE_DEDUP" build \
    --reference "$PRIOR_SFT" --candidates "$RUN_ROOT/examples_accepted.jsonl" \
    --run-tag benchmark_x_expansion_v1 \
    --output "$RUN_ROOT/semantic_dedup_requests.jsonl" \
    --report "$RUN_ROOT/semantic_dedup_plan_report.json"

python3 "$PREFLIGHT" \
    --requests "$RUN_ROOT/semantic_dedup_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 1800 \
    --workers 3 \
    --output "$RUN_ROOT/semantic_dedup_requests_preflight.jsonl" \
    --report "$RUN_ROOT/semantic_dedup_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$RUN_ROOT/semantic_dedup_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-owner-x-dedup \
    --out-dir "$RUN_ROOT" --out-prefix semantic_dedup \
    --output-file "$RUN_ROOT/semantic_dedup_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 3 \
    --temperature 0 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260833}' \
    --resume

python3 "$EXAMPLE_DEDUP" parse \
    --candidates "$RUN_ROOT/examples_accepted.jsonl" \
    --requests "$RUN_ROOT/semantic_dedup_requests_preflight.jsonl" \
    --responses "$RUN_ROOT/semantic_dedup_responses.jsonl" \
    --kept "$RUN_ROOT/examples_accepted_deduped.jsonl" \
    --quarantine "$RUN_ROOT/examples_semantic_quarantine.jsonl" \
    --report "$RUN_ROOT/semantic_dedup_report.json"

python3 "$SFT_PIPELINE" combine \
    --inputs "$PRIOR_SFT" "$RUN_ROOT/examples_accepted_deduped.jsonl" \
    --output "$RUN_ROOT/examples_pre_final_combined.jsonl" \
    --report "$RUN_ROOT/combine_report.json"

python3 "$SFT_PIPELINE" finalize \
    --accepted "$RUN_ROOT/examples_pre_final_combined.jsonl" \
    --output "$RUN_ROOT/examples_accepted_pre_holdout.jsonl" \
    --quarantine "$RUN_ROOT/examples_final_quarantine.jsonl" \
    --report "$RUN_ROOT/final_report.json" --pass-threshold 85 \
    --max-examples-per-material 4 --max-examples-per-material-skill 1

python3 "$SFT_PIPELINE" reconcile-review \
    --examples "$RUN_ROOT/examples_accepted_pre_holdout.jsonl" \
    --recommendations "$RECOMMENDATIONS" \
    --benchmark-issues "$BENCHMARK_ISSUES" \
    --training-materials "$TRAINING_MATERIALS" \
    --output "$RUN_ROOT/examples_accepted_final.jsonl" \
    --quarantine "$RUN_ROOT/examples_benchmark_holdout_quarantine.jsonl" \
    --report "$RUN_ROOT/benchmark_holdout_report.json"

python3 "$SFT_PIPELINE" export \
    --examples "$RUN_ROOT/examples_accepted_final.jsonl" \
    --output "$RUN_ROOT/training_messages.jsonl" \
    --report "$RUN_ROOT/export_report.json"

sha256sum "$RUN_ROOT/examples_accepted_final.jsonl" "$RUN_ROOT/training_messages.jsonl" \
    > "$RUN_ROOT/SHA256SUMS"
