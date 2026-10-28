#!/usr/bin/env bash
# Experiment 4 — Ablation Study.
#
# We run the standard "Attentive-LightGCN component ablation":
#   gamma_init=0  (collapses to LightGCN baseline)
#   gamma_init=0.1 / 0.2 / 0.3 / 0.5  with gamma_learnable=False
#   gamma_init=0.2 with gamma_learnable=True (default)
#   gamma_init=0.2 with gamma_per_layer=True
# and the same sweep for Attentive-SimGCL.
#
# Hyperparams are passed via small ad-hoc YAML overlays (configs/ablation/*).
# To keep this file simple we inline them with sed.
# Usage:  bash experiments/run_ablation.sh [base_config.yaml]

set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${1:-configs/ml100k.yaml}
TMP=$(mktemp -d)
trap "rm -rf ${TMP}" EXIT

SEEDS=(2024 2025 2026)
MODELS=(attentive_lightgcn attentive_simgcl)

# Variants
declare -a TAGS_GAMMA=(g0.0 g0.1 g0.2 g0.3 g0.5)
declare -a VAL_GAMMA=(0.0 0.1 0.2 0.3 0.5)

# 1) Fixed gammas
for M in "${MODELS[@]}"; do
    for k in "${!VAL_GAMMA[@]}"; do
        VAL=${VAL_GAMMA[$k]}
        TAG=${TAGS_GAMMA[$k]}_fixed
        OVL="${TMP}/${M}_${TAG}.yaml"
        cp "${BASE}" "${OVL}"
        python - "${OVL}" "${VAL}" <<'PY'
import sys, yaml
p, v = sys.argv[1], float(sys.argv[2])
cfg = yaml.safe_load(open(p))
cfg['model']['gamma_init'] = v
cfg['model']['gamma_learnable'] = False
cfg['model']['gamma_per_layer'] = False
yaml.safe_dump(cfg, open(p, 'w'))
PY
        for S in "${SEEDS[@]}"; do
            python -m src.train --config "${OVL}" --model "${M}" --seed "${S}" --tag "ablation_${TAG}"
        done
    done

    # 2) Learnable scalar gamma
    OVL="${TMP}/${M}_g_learn.yaml"; cp "${BASE}" "${OVL}"
    python - "${OVL}" <<'PY'
import sys, yaml
p = sys.argv[1]
cfg = yaml.safe_load(open(p))
cfg['model']['gamma_init'] = 0.2
cfg['model']['gamma_learnable'] = True
cfg['model']['gamma_per_layer'] = False
yaml.safe_dump(cfg, open(p, 'w'))
PY
    for S in "${SEEDS[@]}"; do
        python -m src.train --config "${OVL}" --model "${M}" --seed "${S}" --tag ablation_g_learn
    done

    # 3) Per-layer learnable gamma
    OVL="${TMP}/${M}_g_per_layer.yaml"; cp "${BASE}" "${OVL}"
    python - "${OVL}" <<'PY'
import sys, yaml
p = sys.argv[1]
cfg = yaml.safe_load(open(p))
cfg['model']['gamma_init'] = 0.2
cfg['model']['gamma_learnable'] = True
cfg['model']['gamma_per_layer'] = True
yaml.safe_dump(cfg, open(p, 'w'))
PY
    for S in "${SEEDS[@]}"; do
        python -m src.train --config "${OVL}" --model "${M}" --seed "${S}" --tag ablation_g_per_layer
    done
done

python -m experiments.aggregate ablation
