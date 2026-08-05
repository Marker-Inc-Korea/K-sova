#!/bin/bash

set -euo pipefail

BASE_URL=${1:?base URL is required}
SERVED=${2:?served model name is required}
WAVE=${3:-wave1}
REPO=/data1/mk02/K-sova
V2=${REPO}/benchmark/v2.0
if [[ "${WAVE}" == "wave1" ]]; then
  HYPOTHESIS_WORK=${V2}/source_pool/work/gap_hypotheses
  WORK=${V2}/source_pool/work/gap_issue_discovery
else
  HYPOTHESIS_WORK=${V2}/source_pool/work/gap_hypotheses_${WAVE}
  WORK=${V2}/source_pool/work/gap_issue_discovery_${WAVE}
fi
ENV=/data1/mk02/ksova-bench-build/.venv
RUNTIME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${RUNTIME}/bin/python3.11
LEADS=${HYPOTHESIS_WORK}/filtered_discovery_leads.jsonl
REQUESTS=${WORK}/requests.jsonl
RAW=${WORK}/responses.jsonl
CANDIDATES=${WORK}/issue_candidates.jsonl
REPORT=${WORK}/parse_report.json

mkdir -p "${WORK}"
test -s "${LEADS}"
export PYTHONHOME=${RUNTIME} PYTHONPATH=${ENV}/lib/python3.11/site-packages
export PYTHONUNBUFFERED=1

EXISTING_ARGS=(
  --existing "${V2}/source_pool/work/issue_discovery/issue_candidates.jsonl"
)
if [[ "${WAVE}" != "wave1" && -s "${V2}/source_pool/work/gap_issue_discovery/issue_candidates.jsonl" ]]; then
  EXISTING_ARGS+=(--existing "${V2}/source_pool/work/gap_issue_discovery/issue_candidates.jsonl")
fi

"${PYTHON}" -B "${V2}/tools/source_issue_pipeline.py" build \
  --leads "${LEADS}" --seeds "${V2}/source_pool/work/curated_issue_seeds.jsonl" \
  "${EXISTING_ARGS[@]}" \
  --batch-size 24 --output "${REQUESTS}" --report "${WORK}/plan.json"

EXTRA_BODY=$(jq -c \
  '{response_format:{type:"json_schema",json_schema:{name:"ksova_gap_issue_discovery",strict:true,schema:.}},seed:20260805}' \
  "${V2}/source_pool/issue_discovery_schema.json")

"${PYTHON}" -B "${REPO}/benchmark/sovereign_alignment_v4/tools/run_chat_batch.py" \
  --requests "${REQUESTS}" --base-url "${BASE_URL}" \
  --model "${SERVED}" --record-model-name "gemma-4-31B-it__ksova-gap-issue-${WAVE}" \
  --out-dir "${WORK}" --output-file "${RAW}" --force-max-tokens 4096 \
  --timeout 1200 --retries 0 --concurrency 12 --temperature 0.2 --top-p 0.9 \
  --resume --resume-allow-higher-max-tokens \
  --chat-template-kwargs-json '{"enable_thinking":false}' \
  --extra-body-json "${EXTRA_BODY}"

"${PYTHON}" -B "${V2}/tools/source_issue_pipeline.py" parse \
  --seeds "${V2}/source_pool/work/curated_issue_seeds.jsonl" \
  "${EXISTING_ARGS[@]}" \
  --requests "${REQUESTS}" --responses "${RAW}" \
  --output "${CANDIDATES}" --report "${REPORT}"
