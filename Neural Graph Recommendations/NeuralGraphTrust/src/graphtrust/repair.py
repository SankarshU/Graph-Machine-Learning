"""Verify: re-propagate with frozen parameters on an edited graph, and the checks applied to an edit
(slate change, the Jaccard gate, risk targeting)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from src.utils import csr_to_torch_sparse


def fast_norm_adj(n_users: int, n_items: int, R: sp.csr_matrix) -> sp.csr_matrix:
    """D^{-1/2} A D^{-1/2} on the bipartite graph; identical to src.utils.build_norm_adj, vectorised."""
    A = sp.bmat([[None, R], [R.T, None]], format="csr", dtype=np.float32)
    deg = np.asarray(A.sum(axis=1)).ravel()
    dis = np.zeros_like(deg, dtype=np.float64)
    dis[deg > 0] = deg[deg > 0] ** -0.5
    D = sp.diags(dis)
    return (D @ A @ D).tocsr()


def propagate_on_graph(model, data, device, edge_weight: dict):
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


def jaccard_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard distance between two (n, k) arrays of item ids (top-k slates)."""
    out = np.zeros(len(a))
    for j in range(len(a)):
        sa, sb = set(a[j].tolist()), set(b[j].tolist())
        out[j] = 1.0 - len(sa & sb) / len(sa | sb)
    return out


def gate(repaired: np.ndarray, jd: np.ndarray, budget: float) -> np.ndarray:
    """Users whose repair is accepted: slate Jaccard distance within the budget (the rest are reverted)."""
    return repaired[jd <= budget] if len(repaired) else repaired


def target_users(repaired: np.ndarray, r: np.ndarray, by_user: dict, tau: float) -> np.ndarray:
    """Riskiest half of the candidate users; risk = share of a user's scored edges with r < tau."""
    risk = {u: float((r[by_user[u]] < tau).mean()) for u in repaired}
    if not len(repaired):
        return repaired
    thr = np.median(list(risk.values()))
    return np.array([u for u in repaired if risk[u] >= thr], dtype=int)
