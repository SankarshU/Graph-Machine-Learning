# NeuralGraphTrust

Code for *NeuralGraphTrust: Learned Edge Reliability and Inference-Time Graph Repair for Neural Graph Recommendation under Noisy Feedback* (Pratyush Kiran Uppuluri, 2026).

The work develops in three stages:

1. **Stage 1, Attention-Augmented NGCF** (`attentive_ngcf`): additive neighbor attention inside NGCF message construction. Background for this project: [`../NGCF`](../NGCF).
2. **Stage 2, gated masked attention** (`attentive_lightgcn`, `attentive_simgcl`): masked neighbor attention mixed into LightGCN propagation through a learned gate, `E^(l+1) = ((1-γ) Â + γ A_α) E^(l)`. In the SimGCL variant, attention is recomputed on each perturbed view. The degree-normalized attention `r = ñ·α` is used as an edge-reliability score.
3. **Stage 3, the NeuralGraphTrust repair loop** (`experiments/graphtrust_eval.py`): for each user at inference time, it masks the least reliable edges within a budget, re-propagates without retraining, and checks the edit by counterfactual re-ranking.

## Layout

```
src/models/            ngcf, attentive_ngcf (Stage 1); lightgcn, simgcl; attentive_lightgcn,
                       attentive_simgcl (Stage 2); _attention.py (masked neighbour attention),
                       _base.py (shared embedding / BPR / contrastive-view code)
src/graphtrust/        NeuralGraphTrust (Stage 3)
  reliability.py       diagnose: per-edge scores r, rel, dot, cos and their AUROC
  policies.py          act: pi0, pi_rel and matched controls (dot, cos, random, oracle)
  repair.py            verify: frozen-parameter re-propagation on the edited graph, gate, targeting
  evaluation.py        per-user metrics (joint and isolated repair), degree groups, fidelity
src/train.py           training CLI (any model / dataset / seed / noise level)
src/noise_injection.py labeled random-edge injection (uniform or popularity-biased)
experiments/graphtrust_eval.py   runs the NeuralGraphTrust study for one trained model
experiments/make_arxiv_tables.py tables of the paper, generated from results/logs
experiments/run_arxiv_*.sh       the runs reported in the paper
experiments/smoke_test.sh        end-to-end check of all six models on a synthetic graph
tests/                 unit tests (pytest): attention, propagation identities, noise, repair
configs/               hyperparameters (ml100k_arxiv.yaml, ml1m_arxiv.yaml, ...)
data/preprocessing/    make_movielens.py: GroupLens MovieLens -> per-user 80/20 split
results/logs/          run logs behind the reported MovieLens-100K numbers
```

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests          # a few seconds on CPU

# data: download MovieLens-100K from GroupLens, then
python -m data.preprocessing.make_movielens ml100k /path/to/ml-100k/u.data

# accuracy under injected noise (one run)
python -m src.train --config configs/ml100k_arxiv.yaml --model attentive_lightgcn \
    --seed 2024 --noise_ratio 0.1 --tag arxiv_noise0.1 --K_attn 256

# reliability and NeuralGraphTrust repair on the same setting
python -m experiments.graphtrust_eval --config configs/ml100k_arxiv.yaml --model attentive_lightgcn \
    --noise 0.1 --noise_type uniform --seed 2024 --K_attn 256

# all runs in the paper, then the tables
bash experiments/run_arxiv_graphtrust.sh
python experiments/make_arxiv_tables.py
```

The MovieLens data are not redistributed here; see the GroupLens terms of use.
