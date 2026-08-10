#!/bin/bash
#SBATCH --job-name=ksova-v2-judge-build
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/production/work/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/production/work/logs/%x_%j.err

set -euo pipefail

REPO=/data1/mk02/K-sova
V2=${REPO}/benchmark/v2.0
RUN_ID=${KSOVA_RUN_ID:-final1000}
WORK=${V2}/production/work/${RUN_ID}
EVAL=${WORK}/evaluation
ITEMS=${WORK}/ksova_v2_1000.jsonl
PYTHON=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
TARGETS=(
  "${EVAL}/targets/gemma4_31b_thinking_off/responses.jsonl"
  "${EVAL}/targets/qwen36_35b_a3b_thinking_off/responses.jsonl"
  "${EVAL}/targets/exaone45_33b_thinking_off/responses.jsonl"
  "${EVAL}/targets/calm3_22b/responses.jsonl"
  "${EVAL}/targets/gemma4_31b_thinking_on/responses.jsonl"
  "${EVAL}/targets/qwen36_35b_a3b_thinking_on/responses.jsonl"
  "${EVAL}/targets/gpt-5.6-sol__frontier_high/responses.jsonl"
)

test -s "${ITEMS}"
for path in "${TARGETS[@]}"; do
  test -s "${path}"
  test "$(wc -l < "${path}")" -eq 1000
done

"${PYTHON}" -B "${V2}/tools/audit_v2_generation_outputs.py" \
  --model-output "${TARGETS[@]}" --output "${EVAL}/generation_audit.json"

"${PYTHON}" -B "${V2}/tools/build_v2_judge_requests.py" \
  --items "${ITEMS}" --model-output "${TARGETS[@]}" \
  --include-anchors --anchor-item-limit 150 \
  --output "${EVAL}/judge_requests.jsonl" \
  --report "${EVAL}/judge_request_report.json"

jq -n \
  --argjson item_count 1000 --argjson target_models 7 --argjson anchor_items 150 \
  --arg target_input '한국어 사용자 질문 한 개만 제공' \
  '{schema_version:"ksova-v2-final-evaluation-protocol-1.0",item_count:$item_count,target_models:$target_models,anchor_items:$anchor_items,target_input:$target_input,persona:null,stance_instruction:null,primary_comparison:{max_output_tokens:4096,thinking:{gemma:false,qwen:false,exaone:false,calm3:"unsupported"}},thinking_sensitivity:{models:["gemma4_31b","qwen36_35b_a3b"],off_max_output_tokens:4096,on_max_output_tokens:8192,ranking_included:false},frontier_external:{codex:{model:"gpt-5.6-sol",reasoning_effort:"high",max_output_tokens:"cli_not_exposed",tools:false,items:1000},claude:{model:"claude-sonnet-5",status:"authentication_unavailable",included:false}},judge:"gpt-oss-120b",judge_reasoning_effort:"low",metrics:["relation","required_judgment","factual_integrity","premise_resistance"],opaque_composite_score:false}' \
  > "${EVAL}/EVALUATION_PROTOCOL.json"

printf 'K-SOVA v2 final judge requests ready: %s\n' "${EVAL}/judge_requests.jsonl"
