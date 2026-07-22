#!/bin/bash
#SBATCH --job-name=ksova-gemma-local
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/tmp/ksova-mk02/slurm/%x_%j.out
#SBATCH --error=/tmp/ksova-mk02/slurm/%x_%j.err

set -euo pipefail

LOCAL_ROOT=/tmp/ksova-mk02
INPUT=${LOCAL_ROOT}/rebuild-input
MODEL=${LOCAL_ROOT}/models/gemma-4-31B-it
ENV=/tmp/modudoc-target-6700/venv-vllm
PYTHON=${ENV}/bin/python
SERVED=gemma4-31b-ksova-rebuild
PORT=$((26000 + SLURM_JOB_ID % 12000))
WORK=${LOCAL_ROOT}/runs/${SLURM_JOB_ID}
RUN=${WORK}/run
DATA=${WORK}/data
REPORTS=${WORK}/reports
REVIEW=${WORK}/review
TOPIC_RETRY_REQUESTS=${RUN}/topic_retry_requests.jsonl
TOPIC_RETRY_RESPONSES=${RUN}/topic_retry_responses.jsonl
TOPIC_RESPONSES=${RUN}/topic_responses_all.jsonl
AUTHOR_REQUESTS=${RUN}/author_requests.jsonl
AUTHOR_RESPONSES=${RUN}/author_responses.jsonl
JSON_RESPONSE='{"response_format":{"type":"json_object"}}'
THINKING_OFF='{"enable_thinking":false}'

mkdir -p "${LOCAL_ROOT}/slurm" "${RUN}" "${DATA}" "${REPORTS}" "${REVIEW}"
test -x "${PYTHON}"
test -s "${MODEL}/config.json"
test "$(stat -c %s "${MODEL}/model-00001-of-00002.safetensors")" -eq 49784788364
test "$(stat -c %s "${MODEL}/model-00002-of-00002.safetensors")" -eq 12761549884
test -s "${INPUT}/source_run/topic_requests.jsonl"
test -s "${INPUT}/source_run/topic_responses.jsonl"
test -s "${INPUT}/tools/run_chat_batch.py"
test -s "${INPUT}/tools/prepare_authoring.py"
test -s "${INPUT}/tools/finalize_candidates.py"
test "$(find "${INPUT}/dossiers" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 24

jq -c 'select(.metadata.domain_id == "D11" or .metadata.domain_id == "D12")' \
  "${INPUT}/source_run/topic_requests.jsonl" >"${TOPIC_RETRY_REQUESTS}"
test "$(wc -l <"${TOPIC_RETRY_REQUESTS}")" -eq 2

export HF_HOME=${WORK}/hf-cache
export HOME=${WORK}/home
export XDG_CACHE_HOME=${WORK}/cache
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
mkdir -p "${HF_HOME}" "${HOME}" "${XDG_CACHE_HOME}"

VLLM_LOG=${WORK}/vllm.log
"${PYTHON}" "${ENV}/bin/vllm" serve "${MODEL}" \
  --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}" \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 49152 --max-num-seqs 2 --trust-remote-code \
  --enforce-eager --reasoning-parser gemma4 >"${VLLM_LOG}" 2>&1 &
PID=$!

cleanup() {
  if kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${PID}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${PID}" 2>/dev/null || true
  fi
  wait "${PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  curl -fs "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 "${PID}" 2>/dev/null || { tail -200 "${VLLM_LOG}"; exit 1; }
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

run_batch() {
  local requests=$1
  local output=$2
  local prefix=$3
  local max_tokens=$4
  local seed=$5
  "${PYTHON}" -B "${INPUT}/tools/run_chat_batch.py" \
    --requests "${requests}" --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${SERVED}" --record-model-name gemma4_31b_rebuild \
    --out-dir "${RUN}" --out-prefix "${prefix}" --output-file "${output}" \
    --force-max-tokens "${max_tokens}" --timeout 900 --retries 1 \
    --concurrency 2 --temperature 0.2 --top-p 0.95 \
    --chat-template-kwargs-json "${THINKING_OFF}" \
    --extra-body-json "$(jq -c --argjson seed "${seed}" '. + {seed:$seed}' <<<"${JSON_RESPONSE}")"
}

run_batch "${TOPIC_RETRY_REQUESTS}" "${TOPIC_RETRY_RESPONSES}" topic_retry 8192 2026072201
test "$(jq -s '[.[] | select(.error == null and .finish_reason == "stop" and (.response | length > 0))] | length' "${TOPIC_RETRY_RESPONSES}")" -eq 2

jq -c 'select(.error == null and .finish_reason == "stop" and (.response | length > 0))' \
  "${INPUT}/source_run/topic_responses.jsonl" >"${TOPIC_RESPONSES}"
cat "${TOPIC_RETRY_RESPONSES}" >>"${TOPIC_RESPONSES}"
test "$(wc -l <"${TOPIC_RESPONSES}")" -eq 12
test "$(jq -s '[.[].metadata.domain_id] | unique | length' "${TOPIC_RESPONSES}")" -eq 12

"${PYTHON}" -B "${INPUT}/tools/prepare_authoring.py" \
  --topic-responses "${TOPIC_RESPONSES}" \
  --dossiers "${INPUT}/dossiers" \
  --topic-map-output "${DATA}/topic_map.json" \
  --requests-output "${AUTHOR_REQUESTS}" \
  --report "${RUN}/topic_parse_report.json"

run_batch "${AUTHOR_REQUESTS}" "${AUTHOR_RESPONSES}" author 8192 2026072202
test "$(jq -s '[.[] | select(.error == null and .finish_reason == "stop" and (.response | length > 0))] | length' "${AUTHOR_RESPONSES}")" -eq 12

"${PYTHON}" -B "${INPUT}/tools/finalize_candidates.py" \
  --responses "${AUTHOR_RESPONSES}" \
  --topic-map "${DATA}/topic_map.json" \
  --dossiers "${INPUT}/dossiers" \
  --output "${DATA}/benchmark_candidates.jsonl" \
  --report "${REPORTS}/validation.json" \
  --review-md "${REVIEW}/benchmark_review_ko.md" \
  --review-html "${REVIEW}/index.html"

cp "${DATA}/topic_map.json" "${REVIEW}/topic_map.json"
cp "${DATA}/benchmark_candidates.jsonl" "${REVIEW}/benchmark_candidates.jsonl"
cp "${REPORTS}/validation.json" "${REVIEW}/validation.json"
touch "${WORK}/COMPLETE"
printf 'K-SOVA Gemma local rebuild completed: %s\n' "${WORK}"
