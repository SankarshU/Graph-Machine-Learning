"""Diagnose: per-edge reliability scores on a trained attentive model.

For the user-side attended edges at one layer (the last by default):
    r    = n~_u * alpha_ui                 degree-normalised reliability (r = 1 is uniform attention)
    rel  = alpha_ui * sqrt(n_u n_i)        < 1 iff the edge is down-weighted vs LightGCN (Prop. edge)
    dot  = <e_u, e_i>, cos = cosine        on the final embeddings, as non-attention controls
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EdgeTable:
    """One row per scored (user, item) edge; label 1 = original, 0 = injected."""
    users: np.ndarray
    items: np.ndarray
    label: np.ndarray
    r: np.ndarray
    rel: np.ndarray
    dot: np.ndarray
    cos: np.ndarray

    def __len__(self) -> int:
        return len(self.users)

    @property
    def has_noise(self) -> bool:
        return bool((self.label == 0).any())

    def by_user(self) -> dict:
        """user id -> row indices of that user's scored edges."""
        return {u: np.where(self.users == u)[0] for u in np.unique(self.users).astype(int)}


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_original > score_injected), ties count one half (Mann-Whitney)."""
    s = np.concatenate([pos, neg])
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    ranks = s.argsort().argsort().astype(np.float64) + 1
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def edge_table(model, data, device, layer: int | None = None) -> EdgeTable:
    """Score every user-side attended edge of `model` (attention of `layer`, default the last).
    Degrees n_u, n_i are those of the graph the model was trained on, injected edges included."""
    layer = model.n_layers - 1 if layer is None else layer
    model._cached_user = None; model._cached_item = None
    model.cache_eval_embeddings(save_alpha_layer=layer)
    U0 = model.eval_user_embeddings().float(); I0 = model.eval_item_embeddings().float()
    Un = torch.nn.functional.normalize(U0, dim=-1); In = torch.nn.functional.normalize(I0, dim=-1)
    alpha = model.attn.last_alpha.cpu().numpy()[: data.n_users]
    nb = model.attn.last_neighbors.cpu().numpy()[: data.n_users]
    mask = model.attn.last_mask.cpu().numpy()[: data.n_users]
    label = {(int(u), int(i)): int(l) for (u, i), l in zip(data.train_pairs, data.edge_labels)}
    n_u_full = np.asarray(data.user_item_csr.sum(axis=1)).ravel()   # degree incl. injected (what the model saw)
    n_i_full = np.asarray(data.user_item_csr.sum(axis=0)).ravel()
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
    return EdgeTable(*(R[:, c] for c in range(7)))


def reliability_summary(t: EdgeTable) -> dict:
    """AUROC of each score for original-vs-injected and the mean r per class (only with injected edges)."""
    lab = t.label
    out = {}
    for nm, sc in (("r", t.r), ("rel", t.rel), ("dot", t.dot), ("cos", t.cos)):
        out[f"auroc_{nm}"] = auroc(sc[lab == 1], sc[lab == 0])
    out["mean_r_original"] = float(t.r[lab == 1].mean()); out["mean_r_injected"] = float(t.r[lab == 0].mean())
    out["frac_rel_below1_original"] = float((t.rel[lab == 1] < 1).mean())
    out["frac_rel_below1_injected"] = float((t.rel[lab == 0] < 1).mean())
    return out
