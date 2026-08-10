#!/bin/bash
#SBATCH --job-name=ksova-srcmat-val
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/work/slurm_logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/work/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/data1/mk02/K-sova
V2=${REPO}/benchmark/v2.0
DEFAULT_WORK=${V2}/work/current/material_expansion_20260807
WORK=${KSOVA_MATERIAL_WORK:-${DEFAULT_WORK}}
ENV=${KSOVA_VLLM_ENV:-/data1/mk02/fsr/.venv-eval}
PYTHON=${ENV}/bin/python
VALIDATOR=${KSOVA_MATERIAL_VALIDATOR:?set KSOVA_MATERIAL_VALIDATOR to qwen or oss}
TAG=${KSOVA_MATERIAL_VALIDATION_TAG:-source_material}
CANDIDATES=${KSOVA_MATERIAL_CANDIDATES:-${DEFAULT_WORK}/source_material_candidates_expanded.jsonl}
SOURCES=${KSOVA_MATERIAL_SOURCES:-${DEFAULT_WORK}/trusted_source_packets_expanded.jsonl}
SCHEMA=${V2}/source_pool/source_material_validator_schema.json
PIPELINE=${V2}/tools/source_document_pipeline.py
RUNNER=${REPO}/benchmark/v2.0/tools/run_chat_batch.py
PORT=$((28000 + SLURM_JOB_ID % 10000))

case "${VALIDATOR}" in
  qwen)
    MODEL=${KSOVA_QWEN_MODEL:-/data1/mk02/hf_bench/local/Qwen--Qwen3.6-35B-A3B-FP8}
    SERVED=qwen36-35b-a3b-ksova-source-material-validator
    RECORD=qwen3.6-35b-a3b__ksova-source-material-validator
    MAX_TOKENS=4096
    CONCURRENCY=32
    HF_CACHE=/data2/mk02/hf_cache
    TEMPLATE_KWARGS='{"enable_thinking":false}'
    AUDIT_VERSION=qwen-strict-v2
    EXTRA_BODY=$(jq -c --arg name "ksova_source_material_qwen_validator" \
      '{response_format:{type:"json_schema",json_schema:{name:$name,strict:true,schema:.}},top_k:20,seed:20260807}' \
      "${SCHEMA}")
    VLLM_EXTRA=(--language-model-only --reasoning-parser qwen3)
    ;;
  oss)
    MODEL=${KSOVA_OSS_MODEL:-/data1/mk02/모델/gpt-oss-120b}
    SERVED=gpt-oss-120b-ksova-source-material-validator
    RECORD=gpt-oss-120b__ksova-source-material-validator
    MAX_TOKENS=8192
    CONCURRENCY=32
    HF_CACHE=/data2/mk02/hf_cache
    OSS_REASONING_EFFORT=${KSOVA_OSS_REASONING_EFFORT:-medium}
    TEMPLATE_KWARGS=$(printf '{"reasoning_effort":"%s"}' "${OSS_REASONING_EFFORT}")
    AUDIT_VERSION=oss-strict-v2-${OSS_REASONING_EFFORT}
    EXTRA_BODY='{"response_format":{"type":"json_object"},"seed":20260807}'
    VLLM_EXTRA=(--generation-config vllm --reasoning-parser openai_gptoss)
    ;;
  *)
    printf 'unsupported KSOVA_MATERIAL_VALIDATOR=%s\n' "${VALIDATOR}" >&2
    exit 2
    ;;
esac

PREBUILT_REQUESTS=${KSOVA_MATERIAL_REQUESTS:-}
REQUESTS=${PREBUILT_REQUESTS:-${WORK}/${VALIDATOR}_${TAG}_validation_requests.jsonl}
RAW=${WORK}/${VALIDATOR}_${TAG}_validation_responses.jsonl
JUDGED=${WORK}/${VALIDATOR}_${TAG}_judged.jsonl
PASSED=${WORK}/${VALIDATOR}_${TAG}_passed.jsonl
NEEDS_SOURCE=${WORK}/${VALIDATOR}_${TAG}_needs_source.jsonl
REPORT=${WORK}/${VALIDATOR}_${TAG}_validation_report.json
PLAN=${WORK}/${VALIDATOR}_${TAG}_validation_plan.json
VLLM_LOG=${WORK}/logs/${VALIDATOR}_${TAG}_vllm_${SLURM_JOB_ID}.log

mkdir -p "${WORK}/logs"
test -s "${CANDIDATES}"
test -d "${MODEL}"
export HF_HOME=${HF_CACHE} PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
RUN_HOME=/tmp/ksova-source-material-${VALIDATOR}-${SLURM_JOB_ID}
export XDG_CACHE_HOME=${RUN_HOME}/cache
mkdir -p "${RUN_HOME}" "${XDG_CACHE_HOME}"

if [[ -n "${PREBUILT_REQUESTS}" ]]; then
  test -s "${PREBUILT_REQUESTS}"
else
  test -s "${SOURCES}"
  "${PYTHON}" -B "${PIPELINE}" build-material-validation \
    --candidates "${CANDIDATES}" --sources "${SOURCES}" \
    --audit-version "${AUDIT_VERSION}" --total-body-budget 8000 \
    --max-tokens "${MAX_TOKENS}" --output "${REQUESTS}" --report "${PLAN}"
fi

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
  --concurrency "${CONCURRENCY}" --temperature 0.0 --top-p 0.95 \
  --resume --resume-allow-higher-max-tokens \
  --chat-template-kwargs-json "${TEMPLATE_KWARGS}" --extra-body-json "${EXTRA_BODY}"

"${PYTHON}" -B "${PIPELINE}" parse-material-validation \
  --candidates "${CANDIDATES}" --requests "${REQUESTS}" --responses "${RAW}" \
  --validation-key "${VALIDATOR}_material_validation" \
  --output "${JUDGED}" --passed "${PASSED}" --needs-source "${NEEDS_SOURCE}" \
  --report "${REPORT}"

printf 'K-SOVA source material validation complete: %s\n' "${REPORT}"
