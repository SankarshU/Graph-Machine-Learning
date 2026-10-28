#!/usr/bin/env bash
# MovieLens-1M: LightGCN vs Attentive-LightGCN, clean / 10% / 20% uniform noise, one seed per call (SEED env).
# Attentive runs go through graphtrust_eval (accuracy + reliability + repair; isolated repair on 2000 sampled users).
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=configs/ml1m_arxiv.yaml; DEV=${DEV:-cuda}; S=${SEED:?set SEED}
mkdir -p results/logs/ml1m_stdout
i=0
for N in 0.0 0.1 0.2; do
  i=$((i+1)); ( timeout 4h python3 -m experiments.graphtrust_eval --config $CFG --model attentive_lightgcn --noise $N --noise_type uniform --seed $S --K_attn 256 --iso_users 2000 --device $DEV > results/logs/ml1m_stdout/att_$N.log 2>&1; echo "done att $N rc=$?" ) &
  i=$((i+1)); ( timeout 4h python3 -m src.train --config $CFG --model lightgcn --seed $S --noise_ratio $N --tag arxiv_noise$N --K_attn 256 --device $DEV > results/logs/ml1m_stdout/lgcn_$N.log 2>&1; echo "done lgcn $N rc=$?" ) &
done
wait
echo ALL_DONE
