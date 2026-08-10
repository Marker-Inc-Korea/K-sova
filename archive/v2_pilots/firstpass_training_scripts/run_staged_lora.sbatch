#!/bin/bash
#SBATCH --job-name=ksova-qwen35-lora
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/work/runs/first_pass_20260810_v1/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/work/runs/first_pass_20260810_v1/logs/%x_%j.err

set -euo pipefail

REPO=/data1/mk02/K-sova
PYTHON=/data1/mk02/fsr/.venv-eval/bin/python
MODEL=/data/home/mk02/ksova_models/Qwen--Qwen3.5-9B
DATA=${REPO}/benchmark/v2.0/releases/first_pass_20260810_v1/training
OUTPUT=${REPO}/benchmark/v2.0/work/runs/first_pass_20260810_v1/training/qwen35_9b_lora

export HF_HOME=/data/home/mk02/.cache/huggingface
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"${PYTHON}" -B "${REPO}/benchmark/v2.0/release_tools/train_staged_lora.py" \
  --model-path "${MODEL}" --data-dir "${DATA}" --output-dir "${OUTPUT}" \
  --epochs 1 --batch-size 1 --gradient-accumulation 4 --max-length 2048 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
