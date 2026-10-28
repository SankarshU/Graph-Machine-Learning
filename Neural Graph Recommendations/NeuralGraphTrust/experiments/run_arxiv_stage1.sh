#!/usr/bin/env bash
# Stage-1 replication under the paper's protocol: NGCF vs Attention-NGCF (masked), 3 seeds,
# clean / 10% / 20% uniform noise / 10% popularity noise. Accuracy only (logged to ml100k__runs.jsonl).
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=configs/ml100k_arxiv.yaml; DEV=${DEV:-cuda}; PAR=${PAR:-8}
jobs_list=()
for S in 2024 2025 2026; do
  for M in ngcf attentive_ngcf; do
    for N in 0.0 0.1 0.2; do
      jobs_list+=("python3 -m src.train --config $CFG --model $M --seed $S --noise_ratio $N --tag arxiv_noise$N --K_attn 256 --device $DEV")
    done
    jobs_list+=("python3 -m src.train --config $CFG --model $M --seed $S --noise_ratio 0.1 --noise_type pop --tag arxiv_pop0.1 --K_attn 256 --device $DEV")
  done
done
mkdir -p results/logs/s1_stdout
i=0
for J in "${jobs_list[@]}"; do
  i=$((i+1))
  ( eval "timeout 3h $J" > results/logs/s1_stdout/job$i.log 2>&1; echo "done job$i rc=$?" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
done
wait
echo ALL_DONE
