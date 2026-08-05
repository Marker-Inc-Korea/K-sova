#!/bin/bash
#SBATCH --job-name=ksova-v2-final-assemble
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/production/work/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/production/work/logs/%x_%j.err

set -euo pipefail

REPO=/data1/mk02/K-sova
V2=${REPO}/benchmark/v2.0
ROOT=${V2}/production
RUN_ID=${KSOVA_RUN_ID:-final1000}
WORK=${ROOT}/work/${RUN_ID}
PYTHON=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11

"${PYTHON}" -B "${ROOT}/assemble_final.py" \
  --qwen "${WORK}/qwen_judged.jsonl" --oss "${WORK}/oss_judged.jsonl" \
  --approved-items "${V2}/calibration_150/audit/human_approved_unique_items.jsonl" \
  --human-review-report "${V2}/calibration_150/validation/human_review_summary.json" \
  --target 1000 --adversarial-ratio 0.20 --max-adversarial-ratio 0.35 \
  --duplicate-threshold 0.70 --max-per-family 20 --max-c-only-ratio 0.20 \
  --max-single-source-ratio 0.15 \
  --double-validated "${WORK}/double_validated.jsonl" \
  --output "${WORK}/ksova_v2_1000.jsonl" \
  --reserve "${WORK}/validated_reserve.jsonl" \
  --duplicate-audit "${WORK}/semantic_duplicate_audit.jsonl" \
  --report "${WORK}/final_assembly_report.json"

printf 'K-SOVA v2 final assembly complete: %s\n' "${WORK}/final_assembly_report.json"
