#!/usr/bin/env bash
# Full arXiv grid v2 (ML-100K, K_attn=256, 3 seeds). Attentive models run through graphtrust_eval
# (which also logs their accuracy); LightGCN/SimGCL through src.train. Base jobs already finished
# or still running on this machine are skipped. MOD/REM split a grid across machines.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=configs/ml100k_arxiv.yaml; DEV=${DEV:-cuda}; PAR=${PAR:-6}; MOD=${MOD:-1}; REM=${REM:-0}; ONLY=${ONLY:-}
jobs_list=()
for S in 2024 2025 2026; do
  for M in attentive_lightgcn attentive_simgcl; do
    for N in 0.0 0.1 0.2; do
      jobs_list+=("python3 -m experiments.graphtrust_eval --config $CFG --model $M --noise $N --noise_type uniform --seed $S --K_attn 256 --device $DEV")
    done
    jobs_list+=("python3 -m experiments.graphtrust_eval --config $CFG --model $M --noise 0.1 --noise_type pop --seed $S --K_attn 256 --device $DEV")
  done
  for N in 0.0 0.1 0.2; do
    for M in lightgcn simgcl; do
      jobs_list+=("BASE:$M:$N:$S:python3 -m src.train --config $CFG --model $M --seed $S --noise_ratio $N --tag arxiv_noise$N --K_attn 256 --device $DEV")
    done
  done
  for M in lightgcn simgcl; do
    jobs_list+=("python3 -m src.train --config $CFG --model $M --seed $S --noise_ratio 0.1 --noise_type pop --tag arxiv_pop0.1 --K_attn 256 --device $DEV")
  done
done
mkdir -p results/logs/gt_stdout
i=0
for J in "${jobs_list[@]}"; do
  i=$((i+1))
  [ $((i % MOD)) -ne "$REM" ] && continue
  if [ -n "$ONLY" ] && ! [[ " $ONLY " == *" $i "* ]]; then continue; fi
  if [[ "$J" == BASE:* ]]; then
    IFS=: read -r _ M N S CMD <<< "$J"; J=$CMD
    [ -f "results/logs/ml100k__${M}__arxiv_noise${N}__seed${S}.json" ] && { echo "skip done $M $N $S"; continue; }
    pgrep -f "[m]odel $M --seed $S --noise_ratio $N " >/dev/null && { echo "skip running $M $N $S"; continue; }
  fi
  ( eval "timeout 3h $J" > results/logs/gt_stdout/job$i.log 2>&1; echo "done job$i rc=$?" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
done
wait
echo ALL_DONE
