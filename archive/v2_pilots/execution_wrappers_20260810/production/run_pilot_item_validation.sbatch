#!/bin/bash
#SBATCH --job-name=ksova-item-val
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/work/slurm_logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/work/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/data1/mk02/K-sova
V2=${REPO}/benchmark/v2.0
RUN_DIR=${KSOVA_RUN_DIR:?set KSOVA_RUN_DIR}
WORK=${RUN_DIR}/item_validation
REQUESTS=${WORK}/item_validation_requests.jsonl
SCHEMA=${WORK}/item_validator_schema.json
VALIDATOR=${KSOVA_ITEM_VALIDATOR:?set KSOVA_ITEM_VALIDATOR to qwen or oss}
ENV=${KSOVA_VLLM_ENV:-/data1/mk02/fsr/.venv-eval}
PYTHON=${ENV}/bin/python
RUNNER=${REPO}/benchmark/v2.0/tools/run_chat_batch.py
PIPELINE=${V2}/ksova_pipeline.py
PORT=$((24000 + SLURM_JOB_ID % 12000))

case "${VALIDATOR}" in
  qwen)
    MODEL=${KSOVA_QWEN_MODEL:-/data1/mk02/hf_bench/local/Qwen--Qwen3.6-35B-A3B-FP8}
    SERVED=qwen36-35b-a3b-ksova-pilot-item-validator
    RECORD=qwen3.6-35b-a3b__ksova-pilot-item-validator
    MAX_TOKENS=4096
    TEMPLATE_KWARGS='{"enable_thinking":false}'
    EXTRA_BODY=$(jq -c --arg name "ksova_pilot_item_qwen_validator" \
      '{response_format:{type:"json_schema",json_schema:{name:$name,strict:true,schema:.}},top_k:20,seed:20260808}' \
      "${SCHEMA}")
    VLLM_EXTRA=(--language-model-only --reasoning-parser qwen3)
    ;;
  oss)
    MODEL=${KSOVA_OSS_MODEL:-/data1/mk02/모델/gpt-oss-120b}
    SERVED=gpt-oss-120b-ksova-pilot-item-validator
    RECORD=gpt-oss-120b__ksova-pilot-item-validator
    MAX_TOKENS=6144
    TEMPLATE_KWARGS='{"reasoning_effort":"medium"}'
    EXTRA_BODY='{"response_format":{"type":"json_object"},"seed":20260808}'
    VLLM_EXTRA=(--generation-config vllm --reasoning-parser openai_gptoss)
    ;;
  *)
    printf 'unsupported KSOVA_ITEM_VALIDATOR=%s\n' "${VALIDATOR}" >&2
    exit 2
    ;;
esac

RAW=${WORK}/${VALIDATOR}_item_validation_responses.jsonl
JUDGED=${WORK}/${VALIDATOR}_item_judged.jsonl
REPORT=${WORK}/${VALIDATOR}_item_validation_report.json
VLLM_LOG=${WORK}/logs/${VALIDATOR}_item_vllm_${SLURM_JOB_ID}.log

mkdir -p "${WORK}/logs"
test -s "${REQUESTS}"
test -s "${SCHEMA}"
test -d "${MODEL}"
export HF_HOME=/data2/mk02/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
RUN_HOME=/tmp/ksova-pilot-item-${VALIDATOR}-${SLURM_JOB_ID}
export XDG_CACHE_HOME=${RUN_HOME}/cache
mkdir -p "${RUN_HOME}" "${XDG_CACHE_HOME}"

"${PYTHON}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}" \
  --gpu-memory-utilization 0.94 --max-model-len 32768 --max-num-seqs 32 \
  --max-num-batched-tokens 65536 --trust-remote-code --enforce-eager \
  "${VLLM_EXTRA[@]}" >"${VLLM_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -200 "${VLLM_LOG}"; exit 1; }
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

"${PYTHON}" -B "${RUNNER}" \
  --requests "${REQUESTS}" --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED}" --record-model-name "${RECORD}" \
  --out-dir "${WORK}" --output-file "${RAW}" \
  --force-max-tokens "${MAX_TOKENS}" --timeout 1800 --retries 1 \
  --concurrency 32 --temperature 0.0 --top-p 0.95 --resume \
  --chat-template-kwargs-json "${TEMPLATE_KWARGS}" --extra-body-json "${EXTRA_BODY}"

"${PYTHON}" -B "${PIPELINE}" parse-item-validation \
  --run-dir "${RUN_DIR}" --validator "${VALIDATOR}" \
  --requests "${REQUESTS}" --responses "${RAW}" \
  --output "${JUDGED}" --report "${REPORT}"

printf 'K-SOVA pilot item validation complete: %s\n' "${REPORT}"
