#!/usr/bin/env bash
# Experiment 2 — Noisy Edge Robustness (the headline experiment).
# Compares LightGCN / SimGCL / Attentive-LightGCN / Attentive-SimGCL at
# noise ratios 0.05, 0.10, 0.20.
# Usage:  bash experiments/run_noisy_graph.sh [config1.yaml ...]

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(configs/ml100k.yaml configs/ml1m.yaml)
fi

MODELS=(lightgcn attentive_lightgcn simgcl attentive_simgcl)
SEEDS=(2024 2025 2026)
NOISE=(0.05 0.10 0.20)

for CFG in "${CONFIGS[@]}"; do
    for M in "${MODELS[@]}"; do
        for N in "${NOISE[@]}"; do
            for S in "${SEEDS[@]}"; do
                echo "===== ${CFG} / ${M} / noise=${N} / seed=${S} ====="
                python -m src.train --config "${CFG}" --model "${M}" --seed "${S}" \
                    --noise_ratio "${N}" --tag "noise${N}"
            done
        done
    done
done

python -m experiments.aggregate noise
