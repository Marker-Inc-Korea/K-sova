#!/bin/bash

set -u

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "judge work must run inside Slurm" >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
RUN_DIR=${RUN_DIR:-${ROOT}/workspace/v1_0}
JUDGE_DIR=${RUN_DIR}/evaluation/judging
REQUESTS=${REQUESTS:-${JUDGE_DIR}/target_requests.jsonl}
OUTPUT=${OUTPUT:-${JUDGE_DIR}/target_responses_s00.jsonl}
: "${BASE_URL:?set BASE_URL to the loaded vLLM /v1 endpoint}"
: "${SERVED_MODEL:?set SERVED_MODEL to the loaded vLLM model name}"

COMMON=(
  --requests "${REQUESTS}"
  --output "${OUTPUT}"
  --base-url "${BASE_URL}"
  --model "${SERVED_MODEL}"
  --timeout 1200
  --retries 0
  --concurrency "${CONCURRENCY:-12}"
  --shard-count 1
  --shard-index 0
)

primary_status=0
"${PYTHON}" -B "${ROOT}/tools/chat_batch.py" \
  "${COMMON[@]}" \
  --record-model gpt-oss-120b__core_output_judge_low_reasoning \
  --max-tokens 4096 --force-max-tokens 4096 \
  --temperature 0.0 --top-p 1.0 \
  --extra-body-json '{"seed":20260725,"reasoning_effort":"low"}' || primary_status=$?
printf 'primary_status=%s\n' "${primary_status}"

fallback_status=0
"${PYTHON}" -B "${ROOT}/tools/chat_batch.py" \
  "${COMMON[@]}" \
  --record-model gpt-oss-120b__core_output_judge_low_concise_fallback \
  --max-tokens 4096 --force-max-tokens 4096 \
  --temperature 0.2 --top-p 0.9 \
  --system-suffix \
    'Keep the internal analysis under five short steps. Decide directly from the rubric, do not reconsider a settled point, and emit the required JSON immediately.' \
  --extra-body-json \
    '{"seed":20260726,"reasoning_effort":"low","response_format":{"type":"json_object"}}' || fallback_status=$?
printf 'fallback_status=%s\n' "${fallback_status}"

if [[ "${fallback_status}" -ne 0 ]]; then
  final_status=0
  "${PYTHON}" -B "${ROOT}/tools/chat_batch.py" \
    "${COMMON[@]}" \
    --record-model gpt-oss-120b__core_output_judge_low_concise_retry \
    --max-tokens 4096 --force-max-tokens 4096 \
    --temperature 0.4 --top-p 0.85 \
    --system-suffix \
      'Keep the internal analysis under five short steps. Decide directly from the rubric, do not reconsider a settled point, and emit the required JSON immediately.' \
    --extra-body-json \
      '{"seed":20260729,"reasoning_effort":"low","response_format":{"type":"json_object"}}' || final_status=$?
  printf 'final_status=%s\n' "${final_status}"
  exit "${final_status}"
fi
