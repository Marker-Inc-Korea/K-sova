#!/bin/bash
#SBATCH --job-name=ksova-v21-review
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/data/workspace/mk02/ksova_runtime/slurm-%x-%j.out
#SBATCH --error=/data/workspace/mk02/ksova_runtime/slurm-%x-%j.err

set -euo pipefail

REPO=/data1/mk02/K-sova
PYTHON_ROOT=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${PYTHON_ROOT}/bin/python3.11
SITE_PACKAGES=/data1/mk02/fsr/.venv-eval/lib/python3.11/site-packages

: "${KSOVA_REVIEW_MODEL:?set KSOVA_REVIEW_MODEL to qwen or oss}"
: "${KSOVA_REVIEW_REQUESTS:?set KSOVA_REVIEW_REQUESTS}"
: "${KSOVA_REVIEW_OUTPUT:?set KSOVA_REVIEW_OUTPUT}"

case "${KSOVA_REVIEW_MODEL}" in
  qwen)
    MODEL=/data1/mk02/hf_bench/local/Qwen--Qwen3.6-35B-A3B-FP8
    SERVED=qwen36-35b-a3b-ksova-v21-review
    RECORD=qwen3.6-35b-a3b__ksova-v2.1-review
    CHAT_KWARGS='{"enable_thinking":false}'
    VLLM_ARGS=(--language-model-only --reasoning-parser qwen3)
    ;;
  oss)
    MODEL=/data1/mk02/모델/gpt-oss-120b
    SERVED=gpt-oss-120b-ksova-v21-review
    RECORD=gpt-oss-120b__ksova-v2.1-review
    CHAT_KWARGS='{"reasoning_effort":"low"}'
    VLLM_ARGS=(--reasoning-parser openai_gptoss)
    ;;
  *)
    printf 'unsupported KSOVA_REVIEW_MODEL=%s\n' "${KSOVA_REVIEW_MODEL}" >&2
    exit 2
    ;;
esac

test -s "${KSOVA_REVIEW_REQUESTS}"
test -d "${MODEL}"
test -x "${PYTHON}"
test -d "${SITE_PACKAGES}/vllm"
mkdir -p "$(dirname "${KSOVA_REVIEW_OUTPUT}")" /data/workspace/mk02/ksova_runtime

export PYTHONHOME=${PYTHON_ROOT}
export PYTHONPATH=${SITE_PACKAGES}
export PYTHONNOUSERSITE=1
export HF_HOME=/data2/mk02/hf_cache PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export HOME=/tmp/ksova-v21-review-${SLURM_JOB_ID}
export XDG_CACHE_HOME=${HOME}/cache
mkdir -p "${HOME}" "${XDG_CACHE_HOME}"

PORT=$((22000 + SLURM_JOB_ID % 18000))
VLLM_LOG=/data/workspace/mk02/ksova_runtime/v21-review-${KSOVA_REVIEW_MODEL}-${SLURM_JOB_ID}.vllm.log

"${PYTHON}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}" \
  --gpu-memory-utilization 0.94 --max-model-len 16384 --max-num-seqs 32 \
  --max-num-batched-tokens 65536 --generation-config vllm --trust-remote-code \
  --enforce-eager "${VLLM_ARGS[@]}" >"${VLLM_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -200 "${VLLM_LOG}"; exit 1; }
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

"${PYTHON}" -B "${REPO}/benchmark/v2.0/tools/run_chat_batch.py" \
  --requests "${KSOVA_REVIEW_REQUESTS}" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED}" \
  --record-model-name "${RECORD}" --out-dir "$(dirname "${KSOVA_REVIEW_OUTPUT}")" \
  --output-file "${KSOVA_REVIEW_OUTPUT}" --force-max-tokens 4096 \
  --timeout 1200 --retries 1 --concurrency 32 --temperature 0.0 --top-p 0.95 \
  --chat-template-kwargs-json "${CHAT_KWARGS}" \
  --extra-body-json '{"seed":20260808}' --resume --resume-allow-higher-max-tokens
