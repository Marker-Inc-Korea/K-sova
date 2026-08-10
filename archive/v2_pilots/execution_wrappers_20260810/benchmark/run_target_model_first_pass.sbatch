#!/bin/bash
#SBATCH --job-name=ksova-480-1g
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/work/runs/first_pass_20260810_v1/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/work/runs/first_pass_20260810_v1/logs/%x_%j.err

set -euo pipefail
export KSOVA_TENSOR_PARALLEL_SIZE=1
exec /data1/mk02/K-sova/benchmark/v2.0/release_tools/run_local_model_480.sh
