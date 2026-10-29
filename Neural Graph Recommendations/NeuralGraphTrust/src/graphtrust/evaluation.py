"""Per-user ranking metrics for repair: joint (all users repaired at once) and isolated
(each user's own edges only), degree groups and explanation fidelity."""
from __future__ import annotations

import numpy as np
import torch

from .repair import propagate_on_graph


def nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _hit_metrics(top: np.ndarray, test, disc: np.ndarray, k: int):
    """recall@k and ndcg@k of one top-k slate against the user's test items."""
    hit = np.isin(top, list(test))
    return hit.sum() / len(test), (hit * disc).sum() / disc[: min(len(test), k)].sum()


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
        rec[u], ndcg[u] = _hit_metrics(top[u], test, disc, k)
    return top, rec, ndcg


def isolated_user_eval(model, data, device, S: dict, t, k: int, top0: np.ndarray):
    """Per-user counterfactual: for each user u with a non-empty set, remove ONLY u's edges in S[u],
    re-propagate with frozen parameters and rank for u alone. Returns per-user top-k, recall, ndcg
    (users without a repair keep their original slate; their metrics are nan)."""
    top = top0.copy(); rec = np.full(data.n_users, np.nan); nd = np.full(data.n_users, np.nan)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for u, pick in S.items():
        drop = {(int(t.users[j]), int(t.items[j])): 0.0 for j in pick}
        U, I = propagate_on_graph(model, data, device, drop)
        sc = U[u] @ I.T
        seen = data.train_user_items.get(u, [])
        if seen:
            sc[torch.as_tensor(seen, device=sc.device)] = -float("inf")
        tk = sc.topk(k).indices.cpu().numpy(); top[u] = tk
        test = data.test_user_items.get(u, [])
        if test:
            rec[u], nd[u] = _hit_metrics(tk, test, disc, k)
    return top, rec, nd


def fill_unrepaired(nd_repair: np.ndarray, nd0: np.ndarray) -> np.ndarray:
    """Per-user metric after repair, with users that were not repaired keeping their original value."""
    x = nd0.copy(); m_ = np.isfinite(nd_repair); x[m_] = nd_repair[m_]; return x


def original_degree(data) -> np.ndarray:
    """User degree without injected edges (label 1 only)."""
    orig_deg = np.zeros(data.n_users)
    for (u, i), l in zip(data.train_pairs, data.edge_labels):
        orig_deg[int(u)] += int(l)
    return orig_deg


def degree_groups(orig_deg: np.ndarray, nd0: np.ndarray, nd1: np.ndarray) -> dict:
    """Paired effect by original-degree tercile (sparse / medium / dense)."""
    q = np.quantile(orig_deg, [1 / 3, 2 / 3])
    groups = {"sparse": orig_deg <= q[0], "medium": (orig_deg > q[0]) & (orig_deg <= q[1]), "dense": orig_deg > q[1]}
    return {g: {"n": int(m_.sum()), "ndcg_no_repair": nanmean(nd0[m_]), "ndcg_repair": nanmean(nd1[m_]),
                "delta_ndcg": nanmean(nd1[m_] - nd0[m_])} for g, m_ in groups.items()}


def fidelity(jd: np.ndarray, jr: np.ndarray, n_repaired: int) -> dict:
    """Slate change caused by the cited edges vs a random same-size set of the same user's edges."""
    return {"jaccard_cited": nanmean(jd), "jaccard_random": nanmean(jr),
            "frac_slate_changed_cited": float((jd > 0).mean()) if n_repaired else float("nan"),
            "frac_slate_changed_random": float((jr > 0).mean()) if n_repaired else float("nan")}
