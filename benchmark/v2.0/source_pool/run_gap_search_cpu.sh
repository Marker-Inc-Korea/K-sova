#!/bin/bash

set -euo pipefail

WAVE=${1:-wave1}
REPO=/data1/mk02/K-sova
V2=${REPO}/benchmark/v2.0
if [[ "${WAVE}" == "wave1" ]]; then
  WORK=${V2}/source_pool/work/gap_hypotheses
else
  WORK=${V2}/source_pool/work/gap_hypotheses_${WAVE}
fi
HYPOTHESES=${WORK}/hypotheses.jsonl
QUERIES=${WORK}/discovery_queries.json
LEADS=${WORK}/discovery_leads.jsonl
SUPPORTED=${WORK}/search_supported_hypotheses.jsonl
FILTERED=${WORK}/filtered_discovery_leads.jsonl

test -s "${HYPOTHESES}"

python3 "${V2}/tools/source_gap_pipeline.py" queries \
  --hypotheses "${HYPOTHESES}" --base-config "${V2}/source_pool/discovery_queries.json" \
  --output "${QUERIES}" --report "${WORK}/query_report.json"

python3 "${V2}/tools/collect_source_discovery_leads.py" \
  --config "${QUERIES}" --output "${LEADS}" --report "${WORK}/discovery_report.json" \
  --delay 0.01 --retries 0 --max-results-per-query 50 --max-workers 32

python3 "${V2}/tools/source_gap_pipeline.py" attach \
  --hypotheses "${HYPOTHESES}" --leads "${LEADS}" --min-leads 2 --min-queries 1 \
  --output "${SUPPORTED}" --report "${WORK}/attach_report.json"

python3 "${V2}/tools/source_gap_pipeline.py" filter-leads \
  --hypotheses "${SUPPORTED}" --leads "${LEADS}" \
  --output "${FILTERED}" --report "${WORK}/filter_report.json"
