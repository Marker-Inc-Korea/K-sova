#!/bin/bash
#SBATCH --job-name=ksova-gemma-rebuild
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/sovereign_alignment_rebuild/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/sovereign_alignment_rebuild/logs/%x_%j.err

set -euo pipefail

ROOT=/data1/mk02/K-sova/benchmark/sovereign_alignment_rebuild
LEGACY=/data1/mk02/K-sova/benchmark/sovereign_alignment_v4
ENV=/data1/mk02/ksova-bench-build/.venv
RUNTIME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${RUNTIME}/bin/python3.11
MODEL=/data1/mk02/모델/gemma-4-31B-it
SERVED=gemma4-31b-ksova-rebuild
PORT=$((26000 + SLURM_JOB_ID % 12000))
RUN=${ROOT}/runs/${SLURM_JOB_ID}
TOPIC_REQUESTS=${RUN}/topic_requests.jsonl
TOPIC_RESPONSES=${RUN}/topic_responses.jsonl
AUTHOR_REQUESTS=${RUN}/author_requests.jsonl
AUTHOR_RESPONSES=${RUN}/author_responses.jsonl
JSON_RESPONSE='{"response_format":{"type":"json_object"}}'
THINKING_ON='{"enable_thinking":true}'
THINKING_OFF='{"enable_thinking":false}'

mkdir -p "${ROOT}/logs" "${ROOT}/data" "${ROOT}/reports" "${ROOT}/review" "${RUN}"
test -d "${MODEL}"

export PYTHONHOME=${RUNTIME} PYTHONPATH=${ENV}/lib/python3.11/site-packages
export HF_HOME=/data2/mk02/hf_cache PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export HOME=/tmp/ksova-rebuild-${SLURM_JOB_ID}
export XDG_CACHE_HOME=${HOME}/cache
mkdir -p "${HOME}" "${XDG_CACHE_HOME}"

"${PYTHON}" -B "${ROOT}/tools/build_topic_requests.py" \
  --plan "${LEGACY}/config/production_source_discovery_plan.json" \
  --output "${TOPIC_REQUESTS}"

VLLM_LOG=${ROOT}/logs/vllm_${SLURM_JOB_ID}.log
"${PYTHON}" "${ENV}/bin/vllm" serve "${MODEL}" \
  --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}" \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 49152 --max-num-seqs 2 --trust-remote-code \
  --enforce-eager --reasoning-parser gemma4 >"${VLLM_LOG}" 2>&1 &
PID=$!
trap 'kill "${PID}" 2>/dev/null || true; wait "${PID}" 2>/dev/null || true' EXIT

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
  local template_kwargs=$6
  "${PYTHON}" -B "${ROOT}/tools/run_chat_batch.py" \
    --requests "${requests}" --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${SERVED}" --record-model-name gemma4_31b_thinking_rebuild_author \
    --out-dir "${RUN}" --out-prefix "${prefix}" --output-file "${output}" \
    --force-max-tokens "${max_tokens}" --timeout 1800 --retries 1 \
    --concurrency 2 --temperature 0.2 --top-p 0.95 \
    --chat-template-kwargs-json "${template_kwargs}" \
    --extra-body-json "$(jq -c --argjson seed "${seed}" '. + {seed:$seed}' <<<"${JSON_RESPONSE}")"
}

run_batch "${TOPIC_REQUESTS}" "${TOPIC_RESPONSES}" topic_map 8192 2026072101 "${THINKING_OFF}"

"${PYTHON}" -B "${ROOT}/tools/prepare_authoring.py" \
  --topic-responses "${TOPIC_RESPONSES}" \
  --dossiers "${LEGACY}/authoring_private/dossiers/alpha_dev12" \
  --topic-map-output "${ROOT}/data/topic_map.json" \
  --requests-output "${AUTHOR_REQUESTS}" \
  --report "${RUN}/topic_parse_report.json"

run_batch "${AUTHOR_REQUESTS}" "${AUTHOR_RESPONSES}" author 12288 2026072102 "${THINKING_ON}"

"${PYTHON}" -B "${ROOT}/tools/finalize_candidates.py" \
  --responses "${AUTHOR_RESPONSES}" \
  --topic-map "${ROOT}/data/topic_map.json" \
  --dossiers "${LEGACY}/authoring_private/dossiers/alpha_dev12" \
  --output "${ROOT}/data/benchmark_candidates.jsonl" \
  --report "${ROOT}/reports/validation.json" \
  --review-md "${ROOT}/review/benchmark_review_ko.md" \
  --review-html "${ROOT}/review/index.html"

printf 'K-SOVA Gemma rebuild completed: %s\n' "${ROOT}/review/index.html"
