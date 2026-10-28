"""GraphTrust evaluation on one trained attentive model. Everything after training is
inference-time: the graph used for propagation is edited, the parameters are frozen.

For one (dataset, model, noise ratio, noise type, seed):
  1. Train on the noise-injected graph (src.train.run_single); save a checkpoint.
  2. Edge table over the user-side attended edges: label (1 original / 0 injected),
       r      = n~_u * alpha_ui                     (degree-normalised reliability, Eq. rel)
       rel    = alpha_ui * sqrt(n_u n_i)            (<1 iff down-weighted vs LightGCN, Prop. edge)
       dot    = <e_u, e_i> on final embeddings;  cos = cosine of the same.
     AUROC of each score for original-vs-injected (only when injected edges exist).
  3. Policies (per user, cap c on the number of masked edges):
       pi0(tau)        mask attended edges with r < tau, lowest r first         (tau sweep)
       pi_rel          mask attended edges with rel < 1, lowest rel first
       matched controls at the counts of pi0(tau_main): lowest dot, lowest cos, random, oracle
       soft            pi0(tau_main) edges kept but scaled by rho in the adjacency
  4. For each policy: masked count, precision (fraction injected), recall of injected edges,
     Recall@20 / NDCG@20 after re-propagation, per-user so that we can also report
       gate      : revert a user's repair when the top-k Jaccard distance exceeds budget b
       targeted  : repair only the riskiest half of candidate users (risk = share of edges with r<tau)
       groups    : paired effect by original user degree tercile (sparse / medium / dense)
       fidelity  : slate change caused by the cited edges vs a random same-size set
  5. One JSON line -> results/logs/<dataset>__graphtrust.jsonl

Usage:
  python -m experiments.graphtrust_eval --config configs/ml100k_arxiv.yaml --model attentive_lightgcn \
      --noise 0.1 --noise_type uniform --seed 2024 --K_attn 256 --device cuda
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
from src.utils import build_norm_adj, csr_to_torch_sparse


# ----------------------------------------------------------------------------- helpers
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_original > score_injected), ties count one half (Mann-Whitney)."""
    s = np.concatenate([pos, neg])
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    ranks = s.argsort().argsort().astype(np.float64) + 1
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def per_user_eval(U: torch.Tensor, I: torch.Tensor, data, k: int = 20):
    """Full-ranking top-k per user with training items excluded.
    Returns top (n_users,k) item ids, recall@k and ndcg@k per user (nan without test items)."""
    S = U @ I.T
    rows, cols = [], []
    for u, items in data.train_user_items.items():
        rows.extend([u] * len(items)); cols.extend(items)
    S[torch.tensor(rows, device=S.device), torch.tensor(cols, device=S.device)] = -float("inf")
    top = S.topk(k, dim=1).indices.cpu().numpy()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rec = np.full(data.n_users, np.nan); ndcg = np.full(data.n_users, np.nan)
    for u in range(data.n_users):
        test = data.test_user_items.get(u, [])
        if not test:
            continue
        hit = np.isin(top[u], list(test))
        rec[u] = hit.sum() / len(test)
        ndcg[u] = (hit * disc).sum() / disc[: min(len(test), k)].sum()
    return top, rec, ndcg


def fast_norm_adj(n_users: int, n_items: int, R: sp.csr_matrix) -> sp.csr_matrix:
    """D^{-1/2} A D^{-1/2} on the bipartite graph; identical to src.utils.build_norm_adj, vectorised."""
    A = sp.bmat([[None, R], [R.T, None]], format="csr", dtype=np.float32)
    deg = np.asarray(A.sum(axis=1)).ravel()
    dis = np.zeros_like(deg, dtype=np.float64)
    dis[deg > 0] = deg[deg > 0] ** -0.5
    D = sp.diags(dis)
    return (D @ A @ D).tocsr()


def propagate_on_graph(model, data, device, K: int, seed: int, edge_weight: dict):
    """Re-propagate with frozen parameters on an edited graph.
    edge_weight maps (u, i) -> weight in [0, 1]; missing edges keep weight 1; weight 0 removes the edge.
    hat_A is rebuilt from the weighted graph. The attention neighbour lists are the ones used in
    training; a removed edge is masked out of both endpoints' lists (no re-sampling, so the repair
    is the only change), and an edge with weight in (0, 1) stays in the lists."""
    pairs = data.train_pairs
    idx_of = getattr(data, "_pair_index", None)
    if idx_of is None:                       # (u, i) -> row in train_pairs, built once per dataset
        idx_of = {(int(u), int(i)): k for k, (u, i) in enumerate(pairs)}
        data._pair_index = idx_of
    w = np.ones(len(pairs), dtype=np.float32)
    for e, x in edge_weight.items():
        w[idx_of[e]] = x
    keep = w > 0
    csr = sp.csr_matrix((w[keep], (pairs[keep, 0], pairs[keep, 1])), shape=(data.n_users, data.n_items))
    old = (model.norm_adj, model.neighbors, model.valid_mask)
    model.norm_adj = csr_to_torch_sparse(fast_norm_adj(data.n_users, data.n_items, csr), device)
    removed = [(int(u), int(i)) for (u, i), x in edge_weight.items() if x <= 0]
    vm = model.valid_mask
    if removed:
        nbrs = model.neighbors.cpu().numpy(); vm = model.valid_mask.clone()
        nu = data.n_users
        for u, i in removed:
            hit_u = np.where(nbrs[u] == i + nu)[0]        # item i in user u's list (global id)
            hit_i = np.where(nbrs[nu + i] == u)[0]        # user u in item i's list
            if len(hit_u):
                vm[u, torch.as_tensor(hit_u, device=vm.device)] = 0.0
            if len(hit_i):
                vm[nu + i, torch.as_tensor(hit_i, device=vm.device)] = 0.0
    model.valid_mask = vm
    model._cached_user = None; model._cached_item = None
    with torch.no_grad():
        U, I = model.propagate()
    model.norm_adj, model.neighbors, model.valid_mask = old
    model._cached_user = None; model._cached_item = None
    return U.float(), I.float()


def isolated_user_eval(model, data, device, K, seed, S, users, items, k, top0):
    """Per-user counterfactual: for each user u with a non-empty set, remove ONLY u's edges in S[u],
    re-propagate with frozen parameters and rank for u alone. Returns per-user top-k, recall, ndcg
    (users without a repair keep their original slate and metrics)."""
    top = top0.copy(); rec = np.full(data.n_users, np.nan); nd = np.full(data.n_users, np.nan)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for u, pick in S.items():
        drop = {(int(users[j]), int(items[j])): 0.0 for j in pick}
        U, I = propagate_on_graph(model, data, device, K, seed, drop)
        sc = U[u] @ I.T
        seen = data.train_user_items.get(u, [])
        if seen:
            sc[torch.as_tensor(seen, device=sc.device)] = -float("inf")
        t = sc.topk(k).indices.cpu().numpy(); top[u] = t
        test = data.test_user_items.get(u, [])
        if test:
            hit = np.isin(t, list(test))
            rec[u] = hit.sum() / len(test)
            nd[u] = (hit * disc).sum() / disc[: min(len(test), k)].sum()
    return top, rec, nd


def jaccard_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a))
    for j in range(len(a)):
        sa, sb = set(a[j].tolist()), set(b[j].tolist())
        out[j] = 1.0 - len(sa & sb) / len(sa | sb)
    return out


def nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, choices=["attentive_lightgcn", "attentive_simgcl"])
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--noise_type", default="uniform", choices=["uniform", "pop"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--K_attn", type=int, default=256)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--taus", default="0.25,0.5,0.75")
    p.add_argument("--tau_main", type=float, default=0.5)
    p.add_argument("--cap", type=float, default=0.2)
    p.add_argument("--caps", default="0.05,0.1,0.2", help="cap sweep at tau_main (dose response)")
    p.add_argument("--rho", type=float, default=0.5, help="soft-repair weight for masked edges")
    p.add_argument("--budget", type=float, default=0.5, help="gate: max top-k Jaccard distance")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--iso_users", type=int, default=0, help="per-user (isolated) evaluation on a seeded random sample of this many repaired users; 0 = all")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    tag = f"arxiv_noise{a.noise}" if a.noise_type == "uniform" else f"arxiv_pop{a.noise}"
    t0 = time.time()
    out, model, data, evaluator, device = run_single(
        cfg, a.model, seed=a.seed, noise_ratio=a.noise, epochs=a.epochs, tag=tag,
        device_spec=a.device, K_attn=a.K_attn, return_model=True, noise_type=a.noise_type)
    model.eval()
    name = f"{cfg['dataset']['name']}__{a.model}__{a.noise_type}{a.noise}__seed{a.seed}"

    # ---- checkpoint (parameters + the exact neighbour lists used in training)
    ck = REPO_ROOT / "results" / "ckpt"; ck.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "neighbors": model.neighbors.cpu(),
                "valid_mask": model.valid_mask.cpu(), "args": vars(a), "best_metrics": out},
               ck / f"{name}.pt")

    # ---- edge table (user side, last layer)
    model._cached_user = None; model._cached_item = None
    model.cache_eval_embeddings(save_alpha_layer=model.n_layers - 1)
    U0 = model.eval_user_embeddings().float(); I0 = model.eval_item_embeddings().float()
    Un = torch.nn.functional.normalize(U0, dim=-1); In = torch.nn.functional.normalize(I0, dim=-1)
    alpha = model.attn.last_alpha.cpu().numpy()[: data.n_users]
    nb = model.attn.last_neighbors.cpu().numpy()[: data.n_users]
    mask = model.attn.last_mask.cpu().numpy()[: data.n_users]
    label = {(int(u), int(i)): int(l) for (u, i), l in zip(data.train_pairs, data.edge_labels)}
    n_u_full = np.asarray(data.user_item_csr.sum(axis=1)).ravel()   # degree incl. injected (what the model saw)
    n_i_full = np.asarray(data.user_item_csr.sum(axis=0)).ravel()
    orig_deg = np.zeros(data.n_users)
    for (u, i), l in zip(data.train_pairs, data.edge_labels):
        orig_deg[int(u)] += int(l)
    rows = []
    for u in range(data.n_users):
        valid = mask[u] > 0.5; nt = int(valid.sum())
        if nt == 0:
            continue
        items = nb[u][valid] - data.n_users
        al = alpha[u][valid]
        du = torch.tensor(items, device=device)
        dots = (U0[u] @ I0[du].T).cpu().numpy(); coss = (Un[u] @ In[du].T).cpu().numpy()
        for j, i in enumerate(items):
            i = int(i)
            rows.append((u, i, label.get((u, i), 1), nt * al[j], al[j] * np.sqrt(n_u_full[u] * n_i_full[i]), dots[j], coss[j]))
    R = np.array(rows, dtype=np.float64)
    users, items, lab, r, rel, dot, cos = (R[:, c] for c in range(7))
    has_noise = bool((lab == 0).any())

    rec = {"dataset": cfg["dataset"]["name"], "model": a.model, "noise": a.noise, "noise_type": a.noise_type,
           "seed": a.seed, "K_attn": a.K_attn, "tau_main": a.tau_main, "cap": a.cap, "rho": a.rho,
           "budget": a.budget, "gamma": [float(g) for g in model.gamma.all_values().detach().cpu()],
           "train_best": {k: v for k, v in out.items() if "@" in k},
           "edge_coverage": float(len(R) / data.n_train), "n_edges_scored": int(len(R)),
           "injected_fraction": float((lab == 0).mean())}
    if has_noise:
        for nm, sc in (("r", r), ("rel", rel), ("dot", dot), ("cos", cos)):
            rec[f"auroc_{nm}"] = auroc(sc[lab == 1], sc[lab == 0])
        rec["mean_r_original"] = float(r[lab == 1].mean()); rec["mean_r_injected"] = float(r[lab == 0].mean())
        rec["frac_rel_below1_original"] = float((rel[lab == 1] < 1).mean())
        rec["frac_rel_below1_injected"] = float((rel[lab == 0] < 1).mean())

    # ---- baseline slates / metrics (graph as trained, parameters frozen)
    Ub, Ib = propagate_on_graph(model, data, device, a.K_attn, a.seed, {})
    top0, rec0, ndcg0 = per_user_eval(Ub, Ib, data, a.k)
    rec["eval_no_repair"] = {"recall": nanmean(rec0), "ndcg": nanmean(ndcg0)}

    # ---- policies -> per-user masked sets
    rng = np.random.default_rng(a.seed + 99)
    by_user = {u: np.where(users == u)[0] for u in np.unique(users).astype(int)}
    n_inj_total = int((lab == 0).sum())

    def policy_sets(score, thresh, cap=None):
        cap = a.cap if cap is None else cap
        S = {}
        for u, idx in by_user.items():
            cap_n = int(np.floor(cap * len(idx)))
            cand = idx[score[idx] < thresh]
            n = min(len(cand), cap_n)
            if n:
                S[u] = cand[np.argsort(score[cand])[:n]]
        return S

    def matched(S_ref, score=None, kind="lowest"):
        S = {}
        for u, pick in S_ref.items():
            idx = by_user[u]; n = len(pick)
            if kind == "lowest":
                S[u] = idx[np.argsort(score[idx])[:n]]
            elif kind == "random":
                S[u] = rng.choice(idx, size=n, replace=False)
            elif kind == "oracle":
                inj = idx[lab[idx] == 0]
                S[u] = inj[:n] if len(inj) else np.array([], dtype=int)
        return S

    policies = {}
    for t in [float(x) for x in a.taus.split(",")]:
        policies[f"pi0_tau{t}"] = policy_sets(r, t)
    for c in [float(x) for x in a.caps.split(",")]:
        policies[f"pi0_cap{c}"] = policy_sets(r, a.tau_main, cap=c)
    S_main = policies[f"pi0_tau{a.tau_main}"]
    policies["pi_rel"] = policy_sets(rel, 1.0)
    policies["ctrl_dot"] = matched(S_main, dot, "lowest")
    policies["ctrl_cos"] = matched(S_main, cos, "lowest")
    policies["ctrl_random"] = matched(S_main, kind="random")
    policies["ctrl_oracle"] = matched(S_main, kind="oracle")

    results = {}
    slates = {}
    for pname, S in policies.items():
        drop = {(int(users[j]), int(items[j])): 0.0 for pick in S.values() for j in pick}
        Uq, Iq = propagate_on_graph(model, data, device, a.K_attn, a.seed, drop)
        top1, rec1, ndcg1 = per_user_eval(Uq, Iq, data, a.k)
        slates[pname] = (top1, rec1, ndcg1)
        masked = [j for pick in S.values() for j in pick]
        res = {"n_masked": len(masked), "n_users_repaired": len(S),
               "recall": nanmean(rec1), "ndcg": nanmean(ndcg1),
               "delta_ndcg": nanmean(ndcg1) - nanmean(ndcg0)}
        if has_noise and masked:
            res["precision"] = float((lab[masked] == 0).mean())
            res["recall_injected"] = float((lab[masked] == 0).sum() / max(n_inj_total, 1))
        results[pname] = res
    # soft repair of the main policy
    soft = {(int(users[j]), int(items[j])): a.rho for pick in S_main.values() for j in pick}
    Uq, Iq = propagate_on_graph(model, data, device, a.K_attn, a.seed, soft)
    top1, rec1, ndcg1 = per_user_eval(Uq, Iq, data, a.k)
    results["soft_main"] = {"n_masked": len(soft), "recall": nanmean(rec1), "ndcg": nanmean(ndcg1),
                            "delta_ndcg": nanmean(ndcg1) - nanmean(ndcg0)}
    rec["policies"] = results

    # ---- GraphTrust components, evaluated PER USER IN ISOLATION (only u's own edges change)
    main = f"pi0_tau{a.tau_main}"
    t_iso = time.time()
    S_iso, S_rand_iso = S_main, policies["ctrl_random"]
    if a.iso_users and len(S_main) > a.iso_users:
        pick = set(np.random.default_rng(a.seed + 7).choice(sorted(S_main.keys()), size=a.iso_users, replace=False).tolist())
        S_iso = {u: v for u, v in S_main.items() if u in pick}
        S_rand_iso = {u: v for u, v in policies["ctrl_random"].items() if u in pick}
    topI, recI, ndI = isolated_user_eval(model, data, device, a.K_attn, a.seed, S_iso, users, items, a.k, top0)
    topR, recR, ndR = isolated_user_eval(model, data, device, a.K_attn, a.seed, S_rand_iso, users, items, a.k, top0)
    repaired = np.array(sorted(S_iso.keys()), dtype=int)
    rec["iso_users_evaluated"] = int(len(repaired))
    def filled(nd1):   # users without a repair keep their original metric
        x = ndcg0.copy(); m_ = np.isfinite(nd1); x[m_] = nd1[m_]; return x
    ndI_f, ndR_f = filled(ndI), filled(ndR)
    ev = np.isfinite(ndI)
    rec["isolated_per_repaired_user"] = {"pi0_delta": nanmean(ndI[ev] - ndcg0[ev]) if ev.any() else float("nan"),
                                         "random_delta": nanmean(ndR[np.isfinite(ndR)] - ndcg0[np.isfinite(ndR)]) if np.isfinite(ndR).any() else float("nan")}
    rec["isolated"] = {
        "pi0": {"ndcg": nanmean(ndI_f), "delta_ndcg": nanmean(ndI_f) - nanmean(ndcg0)},
        "random": {"ndcg": nanmean(ndR_f), "delta_ndcg": nanmean(ndR_f) - nanmean(ndcg0)},
        "seconds": round(time.time() - t_iso, 1)}
    jd = jaccard_dist(top0[repaired], topI[repaired]) if len(repaired) else np.array([])
    jr = jaccard_dist(top0[repaired], topR[repaired]) if len(repaired) else np.array([])
    # gate: revert users whose slate moved more than the budget
    nd_gate = ndcg0.copy()
    accepted = repaired[jd <= a.budget] if len(repaired) else repaired
    nd_gate[accepted] = ndI_f[accepted]
    rec["gate"] = {"n_candidates": int(len(repaired)), "acceptance_rate": float(len(accepted) / max(len(repaired), 1)),
                   "mean_jaccard_change": nanmean(jd), "ndcg": nanmean(nd_gate), "delta_ndcg": nanmean(nd_gate) - nanmean(ndcg0)}
    # targeted: riskiest half of candidate users
    risk = {u: float((r[by_user[u]] < a.tau_main).mean()) for u in repaired}
    if len(repaired):
        thr = np.median(list(risk.values()))
        chosen = np.array([u for u in repaired if risk[u] >= thr], dtype=int)
    else:
        chosen = repaired
    nd_t = ndcg0.copy(); nd_t[chosen] = ndI_f[chosen]
    masked_t = [j for u in chosen for j in S_main[u]]
    rec["targeted"] = {"n_users": int(len(chosen)), "n_masked": len(masked_t), "ndcg": nanmean(nd_t),
                       "delta_ndcg": nanmean(nd_t) - nanmean(ndcg0),
                       "precision": float((lab[masked_t] == 0).mean()) if (has_noise and masked_t) else float("nan")}
    # degree groups (original degree terciles), paired per user, isolated repair
    q = np.quantile(orig_deg, [1 / 3, 2 / 3])
    groups = {"sparse": orig_deg <= q[0], "medium": (orig_deg > q[0]) & (orig_deg <= q[1]), "dense": orig_deg > q[1]}
    rec["groups"] = {g: {"n": int(m_.sum()), "ndcg_no_repair": nanmean(ndcg0[m_]), "ndcg_repair": nanmean(ndI_f[m_]),
                         "delta_ndcg": nanmean(ndI_f[m_] - ndcg0[m_])} for g, m_ in groups.items()}
    # fidelity: slate change from the cited edges vs a random same-size set of the same user's edges
    rec["fidelity"] = {"jaccard_cited": nanmean(jd), "jaccard_random": nanmean(jr),
                       "frac_slate_changed_cited": float((jd > 0).mean()) if len(repaired) else float("nan"),
                       "frac_slate_changed_random": float((jr > 0).mean()) if len(repaired) else float("nan")}
    rec["wall_s"] = round(time.time() - t0, 1)

    path = REPO_ROOT / cfg["experiment"]["log_dir"] / f"{cfg['dataset']['name']}__graphtrust.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    brief = {k: v for k, v in rec.items() if k in ("auroc_r", "auroc_rel", "auroc_dot", "auroc_cos", "eval_no_repair", "isolated", "gate", "targeted", "fidelity")}
    brief["policies"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in results.items()}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
