#!/bin/bash
#SBATCH --job-name=ksova-solar-proven-smoke1
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/work/runs/second_pass_20260810_v1/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/work/runs/second_pass_20260810_v1/logs/%x_%j.err

set -euo pipefail

export KSOVA_RELEASE_DIR=/data1/mk02/K-sova/benchmark/v2.0/releases/first_pass_20260810_v1
export KSOVA_RUNTIME_DIR=/data1/mk02/K-sova/benchmark/v2.0/work/runs/second_pass_20260810_v1
export KSOVA_MODEL_PATH=/data1/mk02/ksova_models/Solar-Open-100B-NotaMoEQuant-Int4-runtime-54028f45
export KSOVA_MODEL_ID=solar_open_100b_proven_smoke1_v3_historical_runtime
export KSOVA_REQUEST_LIMIT=1
export KSOVA_VLLM_ENV=${KSOVA_VLLM_ENV:-/tmp/ksova-solar-vllm012-historical}
export KSOVA_CLIENT_ENV=/data1/mk02/fsr/.venv-eval
export KSOVA_TENSOR_PARALLEL_SIZE=1
export KSOVA_REASONING_PARSER=solar_open
export KSOVA_TOOL_CALL_PARSER=solar_open
export KSOVA_ENABLE_AUTO_TOOL_CHOICE=true
# Historical successful runs effectively loaded only this processor. Repeated
# --logits-processors flags caused argparse to retain the final value.
export KSOVA_LOGITS_PROCESSORS=vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor
export KSOVA_CHAT_TEMPLATE_KWARGS_JSON='{"enable_thinking":false}'
# Match the previously successful request body: no seed/top_p override and a
# 1.05 repetition penalty.
export KSOVA_TOP_P=
export KSOVA_EXTRA_BODY_JSON='{"repetition_penalty":1.05}'
export KSOVA_MAX_MODEL_LEN=16384
export KSOVA_MAX_OUTPUT_TOKENS=4096
export KSOVA_MAX_NUM_SEQS=1
export KSOVA_MAX_NUM_BATCHED_TOKENS=16384
export KSOVA_CONCURRENCY=1
export KSOVA_GPU_MEMORY_UTILIZATION=0.75
export KSOVA_DTYPE=bfloat16
export KSOVA_GENERATION_CONFIG=
export KSOVA_PRESERVE_VLLM_DEFAULTS=true
export KSOVA_HF_HOME=/tmp/ksova-solar-hf

exec /data1/mk02/K-sova/benchmark/v2.0/release_tools/run_local_model_480.sh
