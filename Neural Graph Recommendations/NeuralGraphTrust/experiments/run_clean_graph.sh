#!/usr/bin/env bash
# Experiment 1 — Clean Graph Benchmark.
# Runs all six models, three seeds each, on the requested dataset(s).
# Results are aggregated to results/tables/<dataset>__<model>__clean.json
# Usage:  bash experiments/run_clean_graph.sh [config1.yaml config2.yaml ...]

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(configs/ml100k.yaml configs/ml1m.yaml)
fi

MODELS=(ngcf lightgcn attentive_lightgcn simgcl attentive_simgcl)
SEEDS=(2024 2025 2026)

for CFG in "${CONFIGS[@]}"; do
    for M in "${MODELS[@]}"; do
        echo "===== ${CFG} / ${M} ====="
        for S in "${SEEDS[@]}"; do
            python -m src.train --config "${CFG}" --model "${M}" --seed "${S}" \
                --tag clean
        done
    done
done

python -m experiments.aggregate clean
