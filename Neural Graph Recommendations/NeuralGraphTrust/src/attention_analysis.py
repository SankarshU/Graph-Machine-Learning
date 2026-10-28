"""Attention diagnostics:
  * mean attention mass on TRUE edges vs INJECTED edges,
  * attention entropy by user-degree bucket,
  * histogram of attention weights for each set,
  * scalar summary written to results/tables/<name>__attn.json,
  * figures written to results/figures/.

Usage from a training/eval script (after model.cache_eval_embeddings(save_alpha_layer=L-1)):

    from src.attention_analysis import analyze_attention
    analyze_attention(model, data, out_dir="results/figures", tag="ml100k_attlgcn_n10")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .data import InteractionData


# ---------------------------------------------------------------------------

def attention_mass_on_edges(
    model,
    data: InteractionData,
) -> Dict[str, float]:
    """For each saved attention slot of each USER node u (not item), compute
    the alpha mass placed on item neighbors that are TRUE vs INJECTED edges.

    Returns aggregate summary.
    """
    attn = model.attn
    if attn.last_alpha is None:
        raise RuntimeError(
            "No saved attention. Call model.cache_eval_embeddings(save_alpha_layer=L-1) first."
        )
    alpha = attn.last_alpha.cpu().numpy()         # (N_total, K)
    neighbors = attn.last_neighbors.cpu().numpy() # (N_total, K) global ids
    mask = attn.last_mask.cpu().numpy()           # (N_total, K)

    n_users = data.n_users
    # We only inspect user rows -> their neighbors are items (global id - n_users)
    user_rows = np.arange(n_users)

    # Build O(1) lookup: for each (u, i) is the edge true (1), injected (0), or absent?
    true_edge: Dict[Tuple[int, int], int] = {}
    for k, (u, i) in enumerate(data.train_pairs):
        true_edge[(int(u), int(i))] = int(data.edge_labels[k])

    mass_true, mass_injected = 0.0, 0.0
    n_true_slots, n_inj_slots = 0, 0

    for u in user_rows:
        a = alpha[u]
        nb = neighbors[u]
        m = mask[u]
        for j in range(a.shape[0]):
            if m[j] < 0.5:
                continue
            global_id = int(nb[j])
            if global_id < n_users:
                continue  # not an item neighbor
            i = global_id - n_users
            label = true_edge.get((int(u), i), 1)
            if label == 1:
                mass_true += float(a[j])
                n_true_slots += 1
            else:
                mass_injected += float(a[j])
                n_inj_slots += 1

    summary = {
        "mean_alpha_true": mass_true / max(n_true_slots, 1),
        "mean_alpha_injected": mass_injected / max(n_inj_slots, 1),
        "n_true_slots": n_true_slots,
        "n_injected_slots": n_inj_slots,
        "ratio_true_to_injected_alpha": (
            (mass_true / max(n_true_slots, 1))
            / max(mass_injected / max(n_inj_slots, 1), 1e-12)
        ),
    }
    return summary


def attention_entropy_by_degree(
    model,
    data: InteractionData,
    bins: Tuple[int, ...] = (1, 5, 20, 100, 1_000_000),
) -> Dict[str, float]:
    """Mean attention entropy of users grouped by training-set degree bucket."""
    attn = model.attn
    H = attn.attention_entropy()
    if H is None:
        raise RuntimeError("Run a save_alpha pass before calling this.")
    H = H.cpu().numpy()
    deg = data.user_degrees()
    user_H = H[: data.n_users]
    out: Dict[str, float] = {}
    edges = list(bins)
    for lo, hi in zip([0] + edges[:-1], edges):
        sel = (deg >= lo) & (deg < hi)
        if sel.sum() > 0:
            out[f"H_users_deg_[{lo},{hi})"] = float(user_H[sel].mean())
    out["H_users_overall"] = float(user_H.mean())
    return out


def analyze_attention(
    model,
    data: InteractionData,
    out_dir: str | Path,
    tag: str,
    save_figures: bool = True,
) -> Dict[str, float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, float] = {}
    summary.update(attention_mass_on_edges(model, data))
    summary.update(attention_entropy_by_degree(model, data))

    # JSON
    (out_dir / f"{tag}__attn_summary.json").write_text(json.dumps(summary, indent=2))

    if save_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return summary

        attn = model.attn
        alpha = attn.last_alpha.cpu().numpy()
        mask = attn.last_mask.cpu().numpy()

        # Histogram of alpha values, valid slots only
        flat = alpha[mask > 0.5]
        plt.figure(figsize=(5, 3))
        plt.hist(flat, bins=50)
        plt.xlabel("attention weight alpha_ij")
        plt.ylabel("count")
        plt.title(f"alpha distribution ({tag})")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}__alpha_hist.png", dpi=150)
        plt.close()

    return summary
