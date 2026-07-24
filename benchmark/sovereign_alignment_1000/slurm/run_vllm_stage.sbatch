#!/bin/bash
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/sovereign_alignment_1000/workspace/run_20260723/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/sovereign_alignment_1000/workspace/run_20260723/logs/%x_%j.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "GPU work must run inside Slurm" >&2; exit 2; }
: "${MODEL_PATH:?set MODEL_PATH}"
: "${MODEL_KIND:?set MODEL_KIND to gemma, qwen, exaone, swallow, or oss}"
: "${REQUESTS:?set REQUESTS}"
: "${OUTPUT:?set OUTPUT}"
: "${RECORD_MODEL:?set RECORD_MODEL}"

ROOT=/data1/mk02/K-sova/benchmark/sovereign_alignment_1000
ENV=/data1/mk02/ksova-bench-build/.venv
RUNTIME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${RUNTIME}/bin/python3.11
PORT=${PORT:-$((20000 + SLURM_JOB_ID % 20000))}
SERVED=${SERVED_MODEL:-ksova-${MODEL_KIND}-${SLURM_JOB_ID}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
CONCURRENCY=${CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.95}
ENABLE_THINKING=${ENABLE_THINKING:-false}
SHARD_COUNT=${SHARD_COUNT:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
EXTRA_BODY_JSON=${EXTRA_BODY_JSON:-'{"response_format":{"type":"json_object"},"seed":20260723}'}
STAGE_MODEL=${STAGE_MODEL:-0}
if [[ "${GENERATION_PROFILE:-}" == "swallow-recommended" ]]; then
  EXTRA_BODY_JSON='{"seed":20260723,"top_k":20,"min_p":0.0}'
fi

export PYTHONHOME=${RUNTIME} PYTHONPATH=${ENV}/lib/python3.11/site-packages
export HF_HOME=/data1/mk02/hf_bench PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
export HOME=/tmp/ksova-${SLURM_JOB_ID} XDG_CACHE_HOME=${HOME}/cache
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "$(dirname "${OUTPUT}")" "${ROOT}/workspace/run_20260723/logs"

if [[ "${STAGE_MODEL}" == "1" ]]; then
  command -v rsync >/dev/null
  MODEL_KEY=$(printf '%s' "${MODEL_PATH}" | sha256sum | cut -c1-12)
  LOCAL_MODEL=/tmp/ksova-models/${MODEL_KEY}
  mkdir -p /tmp/ksova-models
  exec 9>"/tmp/ksova-models/${MODEL_KEY}.lock"
  flock 9
  if [[ ! -f "${LOCAL_MODEL}/.complete" ]]; then
    rm -rf "${LOCAL_MODEL}"
    mkdir -p "${LOCAL_MODEL}"
    rsync -aL "${MODEL_PATH}/" "${LOCAL_MODEL}/"
    touch "${LOCAL_MODEL}/.complete"
  fi
  flock -u 9
  MODEL_PATH=${LOCAL_MODEL}
fi

VLLM_ARGS=(
  serve "${MODEL_PATH}" --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.86}"
  --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS}"
  --trust-remote-code --enforce-eager
)
case "${MODEL_KIND}" in
  gemma) VLLM_ARGS+=(--reasoning-parser gemma4) ;;
  qwen) export VLLM_USE_DEEP_GEMM=0; VLLM_ARGS+=(--language-model-only --reasoning-parser qwen3) ;;
  exaone|swallow) VLLM_ARGS+=(--reasoning-parser qwen3) ;;
  oss) ;;
  *) echo "unsupported MODEL_KIND=${MODEL_KIND}" >&2; exit 2 ;;
esac

SERVER_LOG=${ROOT}/workspace/run_20260723/logs/${SLURM_JOB_NAME:-stage}_${SLURM_JOB_ID}.vllm.log
"${PYTHON}" "${ENV}/bin/vllm" "${VLLM_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then ready=1; break; fi
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -200 "${SERVER_LOG}"; exit 1; }
  sleep 10
done
[[ "${ready}" == "1" ]] || { tail -200 "${SERVER_LOG}"; exit 1; }

run_batch() {
  local requests=$1 output=$2 record_model=$3 enable_thinking=$4
  local max_tokens=$5 force_max_tokens=$6
  local limit_args=() force_token_args=() template_args=()
  if [[ -n "${LIMIT:-}" ]]; then
    limit_args=(--limit "${LIMIT}")
  fi
  if [[ -n "${force_max_tokens}" ]]; then
    force_token_args=(--force-max-tokens "${force_max_tokens}")
  fi
  if [[ "${MODEL_KIND}" != "oss" ]]; then
    template_args=(--chat-template-kwargs-json "{\"enable_thinking\":${enable_thinking}}")
  fi
  "${PYTHON}" -B "${ROOT}/tools/chat_batch.py" \
    --requests "${requests}" --output "${output}" \
    --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED}" --record-model "${record_model}" \
    --max-tokens "${max_tokens}" --timeout "${TIMEOUT:-1800}" --retries "${RETRIES:-1}" \
    "${force_token_args[@]}" \
    --concurrency "${CONCURRENCY}" --temperature "${TEMPERATURE}" --top-p "${TOP_P}" \
    --shard-count "${SHARD_COUNT}" --shard-index "${SHARD_INDEX}" \
    "${template_args[@]}" \
    --extra-body-json "${EXTRA_BODY_JSON}" "${limit_args[@]}"
}

batch_status=0
run_batch "${REQUESTS}" "${OUTPUT}" "${RECORD_MODEL}" "${ENABLE_THINKING}" \
  "${MAX_TOKENS}" "${FORCE_MAX_TOKENS:-}" || batch_status=$?

if [[ -n "${SECOND_OUTPUT:-}" ]]; then
  : "${SECOND_RECORD_MODEL:?set SECOND_RECORD_MODEL with SECOND_OUTPUT}"
  run_batch "${SECOND_REQUESTS:-${REQUESTS}}" "${SECOND_OUTPUT}" "${SECOND_RECORD_MODEL}" \
    "${SECOND_ENABLE_THINKING:-true}" "${SECOND_MAX_TOKENS:-${MAX_TOKENS}}" \
    "${SECOND_FORCE_MAX_TOKENS:-}" || batch_status=$?
fi

exit "${batch_status}"
