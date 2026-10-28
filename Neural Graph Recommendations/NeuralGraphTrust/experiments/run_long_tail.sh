#!/usr/bin/env bash
# Experiment 3 — Long-Tail / Sparse User Evaluation.
# Trains on the clean graph (re-uses results/logs from run_clean_graph.sh
# if present), then re-evaluates each saved checkpoint sliced by user-degree
# bucket (sparse / medium / dense + strict tail le-5).
#
# Implementation note: rather than maintain a separate trainer, we simply
# call experiments/long_tail_eval.py which loads each model from its run
# config and re-evaluates with grouped user subsets.
# Usage:  bash experiments/run_long_tail.sh [config1.yaml ...]

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(configs/ml100k.yaml configs/ml1m.yaml)
fi

MODELS=(lightgcn attentive_lightgcn simgcl attentive_simgcl)
SEEDS=(2024 2025 2026)

for CFG in "${CONFIGS[@]}"; do
    for M in "${MODELS[@]}"; do
        for S in "${SEEDS[@]}"; do
            echo "===== long-tail ${CFG} / ${M} / seed=${S} ====="
            python -m experiments.long_tail_eval \
                --config "${CFG}" --model "${M}" --seed "${S}" --tag long_tail
        done
    done
done

python -m experiments.aggregate long_tail
