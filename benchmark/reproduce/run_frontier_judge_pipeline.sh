#!/bin/bash
# Run a schema-checked atomic-judge pipeline through an authenticated frontier
# CLI. Raw responses are preserved, and only invalid records are retried.

set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 PROVIDER REQUESTS OUT_DIR MODEL RECORD_MODEL [EVALUATION_CONFIG]" >&2
  exit 2
fi

PROVIDER=$1
REQUESTS=$(realpath "$2")
OUT=$(realpath -m "$3")
MODEL=$4
RECORD=$5
EVALUATION_CONFIG=${6:-}

case "${PROVIDER}" in
  codex)
    CONCURRENCY=${KSOVA_FRONTIER_CONCURRENCY:-16}
    ;;
  claude)
    CONCURRENCY=${KSOVA_FRONTIER_CONCURRENCY:-4}
    ;;
  *)
    echo "provider must be codex or claude" >&2
    exit 2
    ;;
esac

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT=${REPO}/benchmark/sovereign_alignment_v4
PYTHON=${KSOVA_PYTHON:-/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11}
ENV=${KSOVA_PYTHON_ENV:-/data1/mk02/ksova-bench-build/.venv}
SCHEMA=${ROOT}/schema/judge_result.schema.json
INSTRUCTIONS=${REPO}/benchmark/reproduce/frontier_judge_instructions_v2.txt
TIMEOUT=${KSOVA_FRONTIER_TIMEOUT:-900}
CLI_RETRIES=${KSOVA_FRONTIER_RETRIES:-2}

export PYTHONHOME=${PYTHON%/bin/python3.11}
export PYTHONPATH=${ENV}/lib/python3.11/site-packages
mkdir -p "${OUT}"
test -s "${REQUESTS}"
test -s "${SCHEMA}"
test -s "${INSTRUCTIONS}"
if [[ -n "${EVALUATION_CONFIG}" ]]; then
  EVALUATION_CONFIG=$(realpath "${EVALUATION_CONFIG}")
  test -s "${EVALUATION_CONFIG}"
fi

run_cli() {
  local requests=$1
  local output=$2
  if [[ ! -s "${requests}" ]]; then
    : > "${output}"
    return
  fi
  "${PYTHON}" -B "${REPO}/benchmark/reproduce/frontier_cli_eval.py" \
    --provider "${PROVIDER}" --requests "${requests}" --output "${output}" \
    --model "${MODEL}" --record-model "${RECORD}" --effort high \
    --concurrency "${CONCURRENCY}" --timeout "${TIMEOUT}" \
    --retries "${CLI_RETRIES}" --resume --instructions "${INSTRUCTIONS}" \
    --work-dir "${REPO}"
}

normalize() {
  local input=$1
  local prefix=$2
  local structural=${prefix}_structural_normalized.jsonl
  local effective=${prefix}_evidence_normalized.jsonl
  "${PYTHON}" -B "${ROOT}/tools/normalize_judge_structural_bindings.py" \
    --responses "${input}" --requests "${REQUESTS}" --schema "${SCHEMA}" \
    --output "${structural}" --report "${prefix}_structural_normalization.json" >&2
  "${PYTHON}" -B "${ROOT}/tools/normalize_judge_evidence_omissions.py" \
    --responses "${structural}" --requests "${REQUESTS}" --schema "${SCHEMA}" \
    --output "${effective}" --report "${prefix}_evidence_normalization.json" >&2
  printf '%s\n' "${effective}"
}

BASE=${OUT}/base.jsonl
run_cli "${REQUESTS}" "${BASE}"
BASE_EFFECTIVE=$(normalize "${BASE}" "${OUT}/base")
CURRENT=${BASE_EFFECTIVE}
RETRY_EFFECTIVE=()
merge_status=1

for round in 1 2; do
  RETRY_REQUESTS=${OUT}/retry_requests_round${round}.jsonl
  "${PYTHON}" -B "${ROOT}/tools/build_judge_retry_requests.py" \
    --responses "${CURRENT}" --requests "${REQUESTS}" --schema "${SCHEMA}" \
    --retry-round "${round}" --output "${RETRY_REQUESTS}" \
    --report "${OUT}/retry_request_round${round}.json"

  RETRY_RAW=${OUT}/retry_responses_round${round}.jsonl
  run_cli "${RETRY_REQUESTS}" "${RETRY_RAW}"
  RETRY_NORMALIZED=$(normalize "${RETRY_RAW}" "${OUT}/retry_responses_round${round}")
  RETRY_EFFECTIVE+=("${RETRY_NORMALIZED}")
  CURRENT=${OUT}/merged_round${round}.jsonl

  set +e
  "${PYTHON}" -B "${ROOT}/tools/merge_judge_retry_outputs.py" \
    --base-responses "${BASE_EFFECTIVE}" \
    --retry-responses "${RETRY_EFFECTIVE[@]}" \
    --requests "${REQUESTS}" --schema "${SCHEMA}" \
    --output "${CURRENT}" --report "${OUT}/merge_round${round}.json"
  merge_status=$?
  set -e
  if [[ ${merge_status} -eq 0 ]]; then
    break
  fi
done

if [[ ${merge_status} -ne 0 ]]; then
  echo "frontier judge still has invalid records after two full retries" >&2
  exit 1
fi

PARSE_CONFIG=()
if [[ -n "${EVALUATION_CONFIG}" ]]; then
  PARSE_CONFIG=(--evaluation-config "${EVALUATION_CONFIG}")
fi
"${PYTHON}" -B "${ROOT}/tools/parse_judge_outputs.py" \
  --responses "${CURRENT}" --requests "${REQUESTS}" --schema "${SCHEMA}" \
  --expected-model "${RECORD}" "${PARSE_CONFIG[@]}" \
  --output "${OUT}/parsed.jsonl" --report "${OUT}/parse_report.json"

jq -e '.status == "pass" and .requests == .valid' "${OUT}/parse_report.json" >/dev/null
printf 'Frontier judge pipeline complete: %s\n' "${OUT}"
