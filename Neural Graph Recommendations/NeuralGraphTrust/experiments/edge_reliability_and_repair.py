"""Edge-reliability diagnostics + GraphTrust deterministic repair (inference time).

For one (dataset, model, noise ratio, seed):
  1. Train on the noise-injected graph (src.train.run_single).
  2. Edge-reliability AUROC (true = positive class, injected = negative class):
       * attn  : degree-normalised attention n_v * alpha_{u,i} at the last layer (user rows),
                 only for edges that are in the attention neighbour slots;
       * dot   : final-embedding score <e_u, e_i> of the same model, on the same edges.
  3. GraphTrust deterministic repair policy (no LLM): for every user, mask the edges whose
     reliability score is below tau (n_v * alpha < tau), capped at `cap` of the user's edges;
     rebuild hat_A and the neighbour pad WITHOUT retraining, re-propagate, re-evaluate.
     Controls with the same per-user mask counts: random edges, lowest dot score,
     and an oracle that masks injected edges.

Writes one JSON line per run to results/logs/<dataset>__repair.jsonl.

Usage:
  python -m experiments.edge_reliability_and_repair --config configs/ml100k.yaml \
      --model attentive_lightgcn --noise 0.1 --seed 2024 --K_attn 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from src.train import load_config, run_single, REPO_ROOT
from src.utils import build_norm_adj, csr_to_torch_sparse, build_neighbor_pad


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """P(score_true > score_injected), ties counted half (Mann-Whitney)."""
    s = np.concatenate([pos_scores, neg_scores])
    ranks = s.argsort().argsort().astype(np.float64) + 1
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n_p, n_n = len(pos_scores), len(neg_scores)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def edge_scores(model, data):
    """Per-(user, item) slot: label, attention reliability, dot score."""
    model._cached_user = None
    model._cached_item = None
    model.cache_eval_embeddings(save_alpha_layer=model.n_layers - 1)
    U = model.eval_user_embeddings().float().cpu().numpy()
    I = model.eval_item_embeddings().float().cpu().numpy()
    alpha = model.attn.last_alpha.cpu().numpy()[: data.n_users]
    nb = model.attn.last_neighbors.cpu().numpy()[: data.n_users]
    mask = model.attn.last_mask.cpu().numpy()[: data.n_users]
    label = {(int(u), int(i)): int(l) for (u, i), l in zip(data.train_pairs, data.edge_labels)}
    rows = []
    for u in range(data.n_users):
        valid = mask[u] > 0.5
        n_v = int(valid.sum())
        for a, g in zip(alpha[u][valid], nb[u][valid]):
            i = int(g) - data.n_users
            rows.append((u, i, label.get((u, i), 1), n_v * float(a), float(U[u] @ I[i])))
    return np.array(rows, dtype=np.float64)


def evaluate_with_mask(model, data, evaluator, device, drop: set, K_attn: int, seed: int):
    """Counterfactual: remove `drop` edges from the graph seen at inference, re-propagate."""
    keep = np.array([(int(u), int(i)) not in drop for u, i in data.train_pairs])
    pairs = data.train_pairs[keep]
    csr = sp.csr_matrix((np.ones(len(pairs), np.float32), (pairs[:, 0], pairs[:, 1])),
                        shape=(data.n_users, data.n_items))
    old = (model.norm_adj, model.neighbors, model.valid_mask)
    model.norm_adj = csr_to_torch_sparse(build_norm_adj(data.n_users, data.n_items, csr), device)
    pad = build_neighbor_pad(data.n_users, data.n_items, csr, K=K_attn,
                             rng=np.random.default_rng(seed), device=device)
    model.neighbors, model.valid_mask = pad.neighbors, pad.valid_mask
    model._cached_user = None
    model._cached_item = None
    res = evaluator.evaluate(get_user_emb=model.eval_user_embeddings,
                             get_item_emb=model.eval_item_embeddings, device=device)
    model.norm_adj, model.neighbors, model.valid_mask = old
    model._cached_user = None
    model._cached_item = None
    return {k: float(v) for k, v in res.metrics.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, choices=["attentive_lightgcn", "attentive_simgcl"])
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--K_attn", type=int, default=256)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tau", type=float, default=0.5, help="mask if n_v*alpha < tau")
    p.add_argument("--cap", type=float, default=0.2, help="max fraction of a user's edges masked")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    t0 = time.time()
    out, model, data, evaluator, device = run_single(
        cfg, a.model, seed=a.seed, noise_ratio=a.noise, epochs=a.epochs,
        tag=f"repair_noise{a.noise}", device_spec=a.device, K_attn=a.K_attn, return_model=True)
    model.eval()

    R = edge_scores(model, data)
    lab, att, dot = R[:, 2], R[:, 3], R[:, 4]
    rec = {"dataset": cfg["dataset"]["name"], "model": a.model, "noise": a.noise, "seed": a.seed,
           "K_attn": a.K_attn, "tau": a.tau, "cap": a.cap,
           "gamma": [float(g) for g in model.gamma.all_values().detach().cpu()],
           "train_best_metrics_noisy_graph": {k: v for k, v in out.items() if "@" in k},
           "edge_coverage": float(len(R) / data.n_train),
           "n_true_slots": int((lab == 1).sum()), "n_inj_slots": int((lab == 0).sum())}
    if (lab == 0).any():
        rec["auroc_attn"] = auroc(att[lab == 1], att[lab == 0])
        rec["auroc_dot"] = auroc(dot[lab == 1], dot[lab == 0])
        rec["mean_norm_alpha_true"] = float(att[lab == 1].mean())
        rec["mean_norm_alpha_inj"] = float(att[lab == 0].mean())

    # ---- GraphTrust deterministic policy: per-user low-reliability edges, capped
    rng = np.random.default_rng(a.seed + 99)
    users = R[:, 0].astype(int)
    drop_attn, drop_dot, drop_rand, drop_oracle = set(), set(), set(), set()
    for u in np.unique(users):
        idx = np.where(users == u)[0]
        n_cap = int(np.floor(a.cap * len(idx)))
        cand = idx[att[idx] < a.tau]
        n = min(len(cand), n_cap)
        if n == 0:
            continue
        pick = cand[np.argsort(att[cand])[:n]]
        drop_attn |= {(u, int(R[j, 1])) for j in pick}
        drop_dot |= {(u, int(R[j, 1])) for j in idx[np.argsort(dot[idx])[:n]]}
        drop_rand |= {(u, int(R[j, 1])) for j in rng.choice(idx, size=n, replace=False)}
        inj = idx[lab[idx] == 0]
        drop_oracle |= {(u, int(R[j, 1])) for j in inj[:n]}
    lab_of = {(int(r[0]), int(r[1])): int(r[2]) for r in R}
    prec = lambda d: float(np.mean([lab_of[e] == 0 for e in d])) if d else float("nan")
    rec["n_masked"] = len(drop_attn)
    rec["mask_precision_attn"] = prec(drop_attn)
    rec["mask_precision_dot"] = prec(drop_dot)
    rec["mask_precision_random"] = prec(drop_rand)
    rec["injected_fraction"] = float((lab == 0).mean())
    rec["eval_no_repair"] = evaluate_with_mask(model, data, evaluator, device, set(), a.K_attn, a.seed)
    for name, d in (("attn", drop_attn), ("dot", drop_dot), ("random", drop_rand), ("oracle", drop_oracle)):
        rec[f"eval_repair_{name}"] = evaluate_with_mask(model, data, evaluator, device, d, a.K_attn, a.seed)
    rec["wall_s"] = round(time.time() - t0, 1)

    path = REPO_ROOT / cfg["experiment"]["log_dir"] / f"{cfg['dataset']['name']}__repair.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if not k.startswith("eval_")}, indent=1))
    for k, v in rec.items():
        if k.startswith("eval_"):
            print(k, {m: round(x, 4) for m, x in v.items() if m in ("recall@20", "ndcg@20")})


if __name__ == "__main__":
    main()
