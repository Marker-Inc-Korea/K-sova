#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$TOOLS_DIR/../../.." && pwd)
RUN_ROOT="$REPO_ROOT/benchmark/v2.0/work/runs/training_scaleup_20260809_wave3"
SFT_DIR="$RUN_ROOT/sft"
PIPELINE="$REPO_ROOT/benchmark/v2.0/tools/training_example_pipeline.py"
EXAMPLE_DEDUP="$REPO_ROOT/benchmark/v2.0/tools/training_example_semantic_dedup.py"
BATCH_RUNNER="$REPO_ROOT/benchmark/v2.0/tools/run_chat_batch.py"
PREFLIGHT="$REPO_ROOT/benchmark/v2.0/tools/preflight_chat_context.py"
MATERIALS="$REPO_ROOT/benchmark/v2.0/work/runs/training_expansion_20260808_wave2/semantic_dedup/materials_kept.jsonl"
SOURCES="$REPO_ROOT/benchmark/v2.0/work/current/sources/source_registry_1576.jsonl"
PRIOR_FINAL="$REPO_ROOT/benchmark/v2.0/work/runs/training_expansion_20260808_wave2/sft/examples_accepted_final.jsonl"
BASE_URL="http://127.0.0.1:18000/v1"
MODEL="glm-5.2"

generation_summary="$SFT_DIR/generation_responses.summary.json"
for _ in $(seq 1 2160); do
    if [[ -f "$generation_summary" ]]; then
        break
    fi
    sleep 20
done
if [[ ! -f "$generation_summary" ]]; then
    echo "generation summary did not appear within 12 hours" >&2
    exit 1
fi

if [[ $(jq -r '.errors' "$generation_summary") != "0" ]]; then
    python3 "$BATCH_RUNNER" \
        --requests "$SFT_DIR/generation_requests_preflight.jsonl" \
        --base-url "$BASE_URL" --model "$MODEL" \
        --record-model-name glm-5.2__ksova-sft-scaleup-generation-wave3 \
        --out-dir "$SFT_DIR" --out-prefix generation \
        --output-file "$SFT_DIR/generation_responses.jsonl" \
        --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 2 \
        --temperature 0.2 --top-p 0.9 \
        --chat-template-kwargs-json '{"enable_thinking":false}' \
        --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260809}' \
        --resume
fi

python3 "$PIPELINE" parse-generation \
    --requests "$SFT_DIR/generation_requests_preflight.jsonl" \
    --responses "$SFT_DIR/generation_responses.jsonl" \
    --output "$SFT_DIR/examples_candidates.jsonl" \
    --report "$SFT_DIR/generation_parse_report.json"

python3 "$PIPELINE" select-coverage \
    --candidates "$SFT_DIR/examples_candidates.jsonl" \
    --reference "$PRIOR_FINAL" \
    --output "$SFT_DIR/examples_candidates_coverage.jsonl" \
    --overflow "$SFT_DIR/examples_candidates_redundant.jsonl" \
    --report "$SFT_DIR/coverage_selection_report.json" \
    --max-examples-per-material 6 --max-examples-per-material-form 1 \
    --candidates-per-open-slot 2

python3 "$PIPELINE" build-validation \
    --materials "$MATERIALS" \
    --examples "$SFT_DIR/examples_candidates_coverage.jsonl" \
    --sources "$SOURCES" --run-tag wave3 \
    --output "$SFT_DIR/validation_requests.jsonl" \
    --report "$SFT_DIR/validation_plan_report.json"

python3 "$PREFLIGHT" \
    --requests "$SFT_DIR/validation_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 2400 \
    --workers 2 \
    --output "$SFT_DIR/validation_requests_preflight.jsonl" \
    --report "$SFT_DIR/validation_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$SFT_DIR/validation_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-sft-scaleup-validator-wave3 \
    --out-dir "$SFT_DIR" --out-prefix validation \
    --output-file "$SFT_DIR/validation_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 2 \
    --temperature 0 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260810}'

python3 "$PIPELINE" parse-validation \
    --examples "$SFT_DIR/examples_candidates_coverage.jsonl" \
    --requests "$SFT_DIR/validation_requests_preflight.jsonl" \
    --responses "$SFT_DIR/validation_responses.jsonl" \
    --accepted "$SFT_DIR/examples_accepted_wave3.jsonl" \
    --rejected "$SFT_DIR/examples_rejected_wave3.jsonl" \
    --report "$SFT_DIR/validation_report.json" --pass-threshold 85

python3 "$EXAMPLE_DEDUP" build \
    --reference "$PRIOR_FINAL" \
    --candidates "$SFT_DIR/examples_accepted_wave3.jsonl" \
    --run-tag wave3 \
    --output "$SFT_DIR/semantic_dedup_requests.jsonl" \
    --report "$SFT_DIR/semantic_dedup_plan_report.json"

python3 "$PREFLIGHT" \
    --requests "$SFT_DIR/semantic_dedup_requests.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --desired-max-tokens 4096 --context-margin 512 --min-output-tokens 2200 \
    --workers 2 \
    --output "$SFT_DIR/semantic_dedup_requests_preflight.jsonl" \
    --report "$SFT_DIR/semantic_dedup_preflight_report.json"

python3 "$BATCH_RUNNER" \
    --requests "$SFT_DIR/semantic_dedup_requests_preflight.jsonl" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --record-model-name glm-5.2__ksova-sft-semantic-dedup-wave3 \
    --out-dir "$SFT_DIR" --out-prefix semantic_dedup \
    --output-file "$SFT_DIR/semantic_dedup_responses.jsonl" \
    --max-tokens 4096 --timeout 1200 --retries 1 --concurrency 2 \
    --temperature 0 \
    --chat-template-kwargs-json '{"enable_thinking":false}' \
    --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260812}'

python3 "$EXAMPLE_DEDUP" parse \
    --candidates "$SFT_DIR/examples_accepted_wave3.jsonl" \
    --requests "$SFT_DIR/semantic_dedup_requests_preflight.jsonl" \
    --responses "$SFT_DIR/semantic_dedup_responses.jsonl" \
    --kept "$SFT_DIR/examples_accepted_wave3_deduped.jsonl" \
    --quarantine "$SFT_DIR/examples_quarantined_semantic_duplicate.jsonl" \
    --report "$SFT_DIR/semantic_dedup_report.json"

python3 "$PIPELINE" combine \
    --inputs "$PRIOR_FINAL" "$SFT_DIR/examples_accepted_wave3_deduped.jsonl" \
    --output "$SFT_DIR/examples_accepted_pre_final_combined.jsonl" \
    --report "$SFT_DIR/combine_report.json"

python3 "$PIPELINE" finalize \
    --accepted "$SFT_DIR/examples_accepted_pre_final_combined.jsonl" \
    --output "$SFT_DIR/examples_accepted_final.jsonl" \
    --quarantine "$SFT_DIR/examples_quarantined_final_gate.jsonl" \
    --report "$SFT_DIR/final_training_gate_report.json" --pass-threshold 85

python3 "$PIPELINE" export \
    --examples "$SFT_DIR/examples_accepted_final.jsonl" \
    --output "$SFT_DIR/training_messages.jsonl" \
    --report "$SFT_DIR/training_export_report.json"

sha256sum "$SFT_DIR/examples_accepted_final.jsonl" "$SFT_DIR/training_messages.jsonl" \
    > "$SFT_DIR/SHA256SUMS"
