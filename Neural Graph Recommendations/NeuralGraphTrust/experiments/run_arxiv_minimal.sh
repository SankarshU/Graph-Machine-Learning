#!/usr/bin/env bash
# Minimal arXiv experiment set: ML-100K, K_attn=256, 3 seeds, N jobs in parallel on one GPU.
# Attentive models at noise>0 are trained inside the repair script (it also logs accuracy).
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=configs/ml100k_arxiv.yaml; DEV=${DEV:-cuda}; PAR=${PAR:-9}; SEEDS=${SEEDS:-"2024 2025 2026"}; START=${START:-1}; MOD=${MOD:-1}; REM=${REM:-0}
jobs_list=()
for S in $SEEDS; do
  for N in 0.1 0.2; do
    for M in attentive_lightgcn attentive_simgcl; do
      jobs_list+=("python3 -m experiments.edge_reliability_and_repair --config $CFG --model $M --noise $N --seed $S --K_attn 256 --device $DEV")
    done
  done
  for M in attentive_lightgcn attentive_simgcl; do
    jobs_list+=("python3 -m src.train --config $CFG --model $M --seed $S --noise_ratio 0.0 --tag arxiv_noise0.0 --K_attn 256 --device $DEV")
  done
  for N in 0.0 0.1 0.2; do
    for M in lightgcn simgcl; do
      jobs_list+=("python3 -m src.train --config $CFG --model $M --seed $S --noise_ratio $N --tag arxiv_noise$N --K_attn 256 --device $DEV")
    done
  done
done
mkdir -p results/logs/arxiv_stdout
i=0
for J in "${jobs_list[@]}"; do
  i=$((i+1))
  [ "$i" -lt "$START" ] && continue
  [ $((i % MOD)) -ne "$REM" ] && continue
  ( eval "$J" > results/logs/arxiv_stdout/job$i.log 2>&1; echo "done job$i rc=$?" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
done
wait
echo ALL_DONE
