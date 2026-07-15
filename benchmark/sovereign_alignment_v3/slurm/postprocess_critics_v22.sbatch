#!/bin/bash
#SBATCH --job-name=ksa3-post-critic-v22
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3/outputs/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3/outputs/logs/%x_%j.err

set -euo pipefail

ROOT=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3
GPT_PREFIX=gpt_v22_critic_repaired_independent
EXA_PREFIX=exaone_v22_critic_repaired_korean
GPT_REQUESTS=${ROOT}/data/critic_requests/gpt_v22_critic_repaired_independent.jsonl
EXA_REQUESTS=${ROOT}/data/critic_requests/exaone_v22_critic_repaired_korean.jsonl
GPT_REVIEW=${ROOT}/data/critic_reviews/gpt_v22_critic_repaired_independent.jsonl
EXA_REVIEW=${ROOT}/data/critic_reviews/exaone_v22_critic_repaired_korean.jsonl
REPAIR=${ROOT}/data/repair_requests_v22_critic_repaired.jsonl
mkdir -p "${ROOT}/outputs/logs" "${ROOT}/data/critic_reviews"

latest_file() {
  local directory=$1
  local prefix=$2
  find "${directory}" -maxdepth 1 -type f -name "${prefix}_*.jsonl" -printf '%T@ %p\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

GPT_RAW=$(latest_file "${ROOT}/outputs/judge/gpt_oss_120b" "${GPT_PREFIX}")
EXA_RAW=$(latest_file "${ROOT}/outputs/judge/exaone45" "${EXA_PREFIX}")
if [[ -z "${GPT_RAW}" || ! -s "${GPT_RAW}" || -z "${EXA_RAW}" || ! -s "${EXA_RAW}" ]]; then
  echo "one or more v22 critic outputs are missing" >&2
  exit 1
fi

python3 "${ROOT}/tools/parse_critic_outputs.py" \
  --critic-output "${GPT_RAW}" --requests "${GPT_REQUESTS}" \
  --critic-model gpt_oss_120b_high_critic \
  --output "${GPT_REVIEW}" --report "${ROOT}/reports/gpt_v22_critic_repaired_independent.json"

python3 "${ROOT}/tools/parse_critic_outputs.py" \
  --critic-output "${EXA_RAW}" --requests "${EXA_REQUESTS}" \
  --critic-model exaone45_33b_korean_critic \
  --output "${EXA_REVIEW}" --report "${ROOT}/reports/exaone_v22_critic_repaired_korean.json"

python3 "${ROOT}/tools/build_repair_requests.py" \
  --drafts "${ROOT}/data/candidates/gpt_oss_120b_full_v22_critic_repaired.jsonl" \
  --blueprints "${ROOT}/config/case_blueprints_v22.json" \
  --critic-file "${GPT_REVIEW}" --critic-file "${EXA_REVIEW}" \
  --expected-reviews-per-case 2 --failed-only \
  --output "${REPAIR}" --report "${ROOT}/reports/v22_critic_consensus_gate.json"

if [[ -s "${REPAIR}" ]]; then
  echo "v22 critic gate found failed cases; repair is required" >&2
  wc -l "${REPAIR}" >&2
  exit 1
fi

echo "v22 critic consensus gate passed: 24/24 cases by both critics"
