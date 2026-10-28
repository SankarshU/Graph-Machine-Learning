#!/usr/bin/env bash
# End-to-end smoke test on a tiny synthetic dataset (200 users, 300 items).
# Should finish in ~30s on CPU. If this passes, the pipeline is wired correctly.
#
# Run from repo root:
#   bash experiments/smoke_test.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Generate toy data
python -m data.preprocessing.make_toy --out data/toy --seed 0

# 2. Run each of the five models for a handful of epochs
for M in lightgcn ngcf attentive_lightgcn simgcl attentive_simgcl; do
    echo "----- smoke: $M -----"
    python -m src.train --config configs/toy.yaml --model "$M" --seed 2024 \
        --tag smoke --device cpu --epochs 10
done

# 3. Aggregator should pick up runs
python -m experiments.aggregate smoke

echo
echo "[smoke_test] Done. See results/logs/ and results/tables/smoke__summary.csv"
