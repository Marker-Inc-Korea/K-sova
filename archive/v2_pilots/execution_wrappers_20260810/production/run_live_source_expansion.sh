#!/bin/bash
set -euo pipefail

BASE_URL=${1:?base URL is required}
SERVED=${2:?served model name is required}
REPO=/data1/mk02/K-sova
V2=${REPO}/benchmark/v2.0
ROOT=${V2}/production
RUN_ID=${KSOVA_RUN_ID:-final1000}
WORK=${ROOT}/work/${RUN_ID}
ENV=/data1/mk02/ksova-bench-build/.venv
RUNTIME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${RUNTIME}/bin/python3.11
RUNNER=${REPO}/benchmark/v2.0/tools/run_chat_batch.py
WAVES=(wave3 wave4 wave5 wave6)

export PYTHONHOME=${RUNTIME} PYTHONPATH=${ENV}/lib/python3.11/site-packages
export PYTHONUNBUFFERED=1

for WAVE in "${WAVES[@]}"; do
  HYPOTHESIS_WORK=${V2}/source_pool/work/gap_hypotheses_${WAVE}
  ISSUE_WORK=${V2}/source_pool/work/gap_issue_discovery_${WAVE}
  mkdir -p "${HYPOTHESIS_WORK}" "${ISSUE_WORK}"

  GAP_EXISTING=(--existing "${V2}/source_pool/work/curated_issue_seeds.jsonl")
  while IFS= read -r path; do
    [[ "${path}" == "${HYPOTHESIS_WORK}/hypotheses.jsonl" ]] && continue
    GAP_EXISTING+=(--existing "${path}")
  done < <(find "${V2}/source_pool/work" -type f \
    \( -name 'issue_candidates.jsonl' -o -name 'hypotheses.jsonl' \) -print | sort)

  HYPOTHESIS_REQUESTS=${HYPOTHESIS_WORK}/requests.jsonl
  HYPOTHESIS_RESPONSES=${HYPOTHESIS_WORK}/responses.jsonl
  HYPOTHESES=${HYPOTHESIS_WORK}/hypotheses.jsonl
  "${PYTHON}" -B "${V2}/tools/source_gap_pipeline.py" build \
    "${GAP_EXISTING[@]}" --count 24 --max-existing-per-lane 160 --max-tokens 6144 \
    --output "${HYPOTHESIS_REQUESTS}" --report "${HYPOTHESIS_WORK}/plan.json"
  HYPOTHESIS_BODY=$(jq -c \
    --arg name "ksova_gap_hypotheses_${WAVE}" \
    '{response_format:{type:"json_schema",json_schema:{name:$name,strict:true,schema:.}},seed:20260805}' \
    "${V2}/source_pool/gap_hypothesis_schema.json")
  "${PYTHON}" -B "${RUNNER}" \
    --requests "${HYPOTHESIS_REQUESTS}" --base-url "${BASE_URL}" --model "${SERVED}" \
    --record-model-name "qwen3.6-35B-A3B-FP8__ksova-gap-hypotheses-${WAVE}" \
    --out-dir "${HYPOTHESIS_WORK}" --output-file "${HYPOTHESIS_RESPONSES}" \
    --force-max-tokens 6144 --timeout 1200 --retries 1 --concurrency 13 \
    --temperature 0.3 --top-p 0.9 --resume --resume-allow-higher-max-tokens \
    --chat-template-kwargs-json '{"enable_thinking":false}' --extra-body-json "${HYPOTHESIS_BODY}"
  "${PYTHON}" -B "${V2}/tools/source_gap_pipeline.py" parse \
    "${GAP_EXISTING[@]}" --requests "${HYPOTHESIS_REQUESTS}" \
    --responses "${HYPOTHESIS_RESPONSES}" --output "${HYPOTHESES}" \
    --report "${HYPOTHESIS_WORK}/parse_report.json"

  "${PYTHON}" -B "${V2}/tools/source_gap_pipeline.py" queries \
    --hypotheses "${HYPOTHESES}" --base-config "${V2}/source_pool/discovery_queries.json" \
    --output "${HYPOTHESIS_WORK}/discovery_queries.json" \
    --report "${HYPOTHESIS_WORK}/query_report.json"
  "${PYTHON}" -B "${V2}/tools/collect_source_discovery_leads.py" \
    --config "${HYPOTHESIS_WORK}/discovery_queries.json" \
    --output "${HYPOTHESIS_WORK}/discovery_leads.jsonl" \
    --report "${HYPOTHESIS_WORK}/discovery_report.json" \
    --delay 0.01 --retries 0 --max-results-per-query 50 --max-workers 32
  "${PYTHON}" -B "${V2}/tools/source_gap_pipeline.py" attach \
    --hypotheses "${HYPOTHESES}" --leads "${HYPOTHESIS_WORK}/discovery_leads.jsonl" \
    --min-leads 2 --min-queries 1 \
    --output "${HYPOTHESIS_WORK}/search_supported_hypotheses.jsonl" \
    --report "${HYPOTHESIS_WORK}/attach_report.json"
  "${PYTHON}" -B "${V2}/tools/source_gap_pipeline.py" filter-leads \
    --hypotheses "${HYPOTHESIS_WORK}/search_supported_hypotheses.jsonl" \
    --leads "${HYPOTHESIS_WORK}/discovery_leads.jsonl" \
    --output "${HYPOTHESIS_WORK}/filtered_discovery_leads.jsonl" \
    --report "${HYPOTHESIS_WORK}/filter_report.json"
  "${PYTHON}" -B "${V2}/tools/resolve_trusted_discovery_sources.py" \
    --leads "${HYPOTHESIS_WORK}/filtered_discovery_leads.jsonl" \
    --output "${HYPOTHESIS_WORK}/trusted_leads.jsonl" \
    --report "${HYPOTHESIS_WORK}/trusted_leads_report.json" \
    --snapshot-dir "${HYPOTHESIS_WORK}/trusted_source_snapshots" --workers 24

  ISSUE_EXISTING=()
  while IFS= read -r path; do
    [[ "${path}" == "${ISSUE_WORK}/issue_candidates.jsonl" ]] && continue
    ISSUE_EXISTING+=(--existing "${path}")
  done < <(find "${V2}/source_pool/work" -type f -name 'issue_candidates.jsonl' -print | sort)

  ISSUE_REQUESTS=${ISSUE_WORK}/requests.jsonl
  ISSUE_RESPONSES=${ISSUE_WORK}/responses.jsonl
  ISSUE_CANDIDATES=${ISSUE_WORK}/issue_candidates.jsonl
  "${PYTHON}" -B "${V2}/tools/source_issue_pipeline.py" build \
    --leads "${HYPOTHESIS_WORK}/filtered_discovery_leads.jsonl" \
    --hypotheses "${HYPOTHESIS_WORK}/search_supported_hypotheses.jsonl" \
    --seeds "${V2}/source_pool/work/curated_issue_seeds.jsonl" \
    "${ISSUE_EXISTING[@]}" --batch-size 12 \
    --output "${ISSUE_REQUESTS}" --report "${ISSUE_WORK}/plan.json"
  ISSUE_BODY=$(jq -c \
    --arg name "ksova_gap_issue_${WAVE}" \
    '{response_format:{type:"json_schema",json_schema:{name:$name,strict:true,schema:.}},seed:20260805}' \
    "${V2}/source_pool/issue_discovery_schema.json")
  "${PYTHON}" -B "${RUNNER}" \
    --requests "${ISSUE_REQUESTS}" --base-url "${BASE_URL}" --model "${SERVED}" \
    --record-model-name "qwen3.6-35B-A3B-FP8__ksova-gap-issue-${WAVE}" \
    --out-dir "${ISSUE_WORK}" --output-file "${ISSUE_RESPONSES}" \
    --force-max-tokens 4096 --timeout 1200 --retries 1 --concurrency 24 \
    --temperature 0.2 --top-p 0.9 --resume --resume-allow-higher-max-tokens \
    --chat-template-kwargs-json '{"enable_thinking":false}' --extra-body-json "${ISSUE_BODY}"
  "${PYTHON}" -B "${V2}/tools/source_issue_pipeline.py" parse \
    --seeds "${V2}/source_pool/work/curated_issue_seeds.jsonl" \
    "${ISSUE_EXISTING[@]}" --requests "${ISSUE_REQUESTS}" \
    --responses "${ISSUE_RESPONSES}" --output "${ISSUE_CANDIDATES}" \
    --report "${ISSUE_WORK}/parse_report.json"
done

MATERIAL_FILTERS=()
for WAVE in "${WAVES[@]}"; do
  MATERIAL_FILTERS+=(--include-candidate-ids \
    "${V2}/source_pool/work/gap_issue_discovery_${WAVE}/issue_candidates.jsonl")
done

"${PYTHON}" -B "${ROOT}/material_pipeline.py" build \
  --v2-root "${V2}" --existing-seeds "${WORK}/material_seeds.jsonl" \
  --request-prefix material_expansion "${MATERIAL_FILTERS[@]}" \
  --output "${WORK}/material_expansion_requests.jsonl" \
  --catalog "${WORK}/material_expansion_catalog.jsonl" \
  --report "${WORK}/material_expansion_plan.json"
MATERIAL_BODY=$(jq -c \
  '{response_format:{type:"json_schema",json_schema:{name:"ksova_v2_material_expansion",strict:true,schema:.}},seed:20260805}' \
  "${ROOT}/material_schema.json")
"${PYTHON}" -B "${RUNNER}" \
  --requests "${WORK}/material_expansion_requests.jsonl" \
  --base-url "${BASE_URL}" --model "${SERVED}" \
  --record-model-name "qwen3.6-35B-A3B-FP8__ksova-v2-material-expansion" \
  --out-dir "${WORK}" --output-file "${WORK}/material_expansion_responses.jsonl" \
  --force-max-tokens 4096 --timeout 1800 --retries 1 --concurrency 32 \
  --temperature 0.15 --top-p 0.9 --resume --resume-allow-higher-max-tokens \
  --chat-template-kwargs-json '{"enable_thinking":false}' --extra-body-json "${MATERIAL_BODY}"
"${PYTHON}" -B "${ROOT}/material_pipeline.py" parse \
  --catalog "${WORK}/material_expansion_catalog.jsonl" \
  --requests "${WORK}/material_expansion_requests.jsonl" \
  --responses "${WORK}/material_expansion_responses.jsonl" \
  --output "${WORK}/material_expansion_seeds.jsonl" \
  --sources "${WORK}/material_expansion_sources.jsonl" \
  --accepted "${WORK}/material_expansion_atoms.jsonl" \
  --report "${WORK}/material_expansion_report.json"

printf 'K-SOVA live source expansion complete: %s\n' "${WORK}/material_expansion_seeds.jsonl"
