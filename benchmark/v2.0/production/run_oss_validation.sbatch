#!/bin/bash
#SBATCH --job-name=ksova-v2-prod-oss
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/production/work/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/production/work/logs/%x_%j.err

set -euo pipefail

REPO=/data1/mk02/K-sova
V2=${REPO}/benchmark/v2.0
ROOT=${V2}/production
RUN_ID=${KSOVA_RUN_ID:-wave1}
WORK=${ROOT}/work/${RUN_ID}
LOG_DIR=${ROOT}/work/logs
ENV=/data1/mk02/ksova-bench-build/.venv
RUNTIME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${RUNTIME}/bin/python3.11
MODEL=/data1/mk02/모델/gpt-oss-120b
SERVED=gpt-oss-120b-ksova-v2-production-validator
ITEMS=${WORK}/qwen_passed.jsonl
REQUESTS=${WORK}/oss_validation_requests.jsonl
PLAN=${WORK}/oss_validation_plan.json
RAW=${WORK}/oss_validation_responses.jsonl
JUDGED=${WORK}/oss_judged.jsonl
PASSED=${WORK}/double_validated.jsonl
REPORT=${WORK}/oss_validation_report.json
FINAL=${WORK}/wave_selected.jsonl
FINAL_REPORT=${WORK}/wave_selection_report.json
MAX_PER_SEED=${KSOVA_MAX_PER_SEED:-1}
PORT=$((26000 + SLURM_JOB_ID % 14000))
VLLM_LOG=${LOG_DIR}/${RUN_ID}_oss_validator_vllm_${SLURM_JOB_ID}.log

mkdir -p "${WORK}" "${LOG_DIR}"
test -s "${ITEMS}"
test -d "${MODEL}"

export PYTHONHOME=${RUNTIME} PYTHONPATH=${ENV}/lib/python3.11/site-packages
export HF_HOME=/data2/mk02/hf_cache PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export HOME=/tmp/ksova-prod-oss-${SLURM_JOB_ID}
export XDG_CACHE_HOME=${HOME}/cache
mkdir -p "${HOME}" "${XDG_CACHE_HOME}"

"${PYTHON}" -B "${ROOT}/pipeline.py" build-validator \
  --stage oss --items "${ITEMS}" --output "${REQUESTS}" --report "${PLAN}"

"${PYTHON}" "${ENV}/bin/vllm" serve "${MODEL}" \
  --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}" \
  --gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 32 \
  --max-num-batched-tokens 65536 --generation-config vllm \
  --trust-remote-code --enforce-eager \
  --reasoning-parser openai_gptoss >"${VLLM_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -200 "${VLLM_LOG}"; exit 1; }
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

"${PYTHON}" -B "${REPO}/benchmark/sovereign_alignment_v4/tools/run_chat_batch.py" \
  --requests "${REQUESTS}" --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED}" --record-model-name "gpt-oss-120b__ksova-v2-production-validator" \
  --out-dir "${WORK}" --output-file "${RAW}" --force-max-tokens 6144 \
  --timeout 1800 --retries 1 --concurrency 32 --temperature 0.0 --top-p 1.0 \
  --resume --resume-allow-higher-max-tokens \
  --chat-template-kwargs-json '{"reasoning_effort":"low"}' \
  --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260806}'

"${PYTHON}" -B "${ROOT}/pipeline.py" parse-validator \
  --stage oss --threshold 85 --items "${ITEMS}" --requests "${REQUESTS}" \
  --responses "${RAW}" --output "${JUDGED}" --passed "${PASSED}" --report "${REPORT}"

"${PYTHON}" -B "${ROOT}/pipeline.py" finalize \
  --items "${PASSED}" --target 1000 --max-per-seed "${MAX_PER_SEED}" \
  --output "${FINAL}" --report "${FINAL_REPORT}"

printf 'K-SOVA OSS validation and wave selection complete: %s\n' "${FINAL_REPORT}"
