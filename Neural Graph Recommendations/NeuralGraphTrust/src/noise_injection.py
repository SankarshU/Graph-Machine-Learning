"""Random-edge noise injection with per-edge true / injected labels.

This is the operational backbone of Experiment 2 (the headline experiment):
inject random user-item edges into the training graph, train the model on the
corrupted graph, and check whether attentive aggregation degrades less than
fixed-Laplacian propagation. The labels (true / injected) are kept so that
attention_analysis.py can compute attention mass on each set separately.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from .data import InteractionData


def add_random_edges(
    data: InteractionData,
    ratio: float,
    seed: int,
    noise_type: str = "uniform",
) -> InteractionData:
    """Return a NEW InteractionData with `ratio * n_train` random edges injected.

    Each injected edge (u, i) is sampled from pairs that are in neither the training
    nor the test set of u, so an injected edge is never a held-out true preference
    and never removes a test item from the ranked candidates.

    Edge labels (`data.edge_labels`) are extended: 1 for original edges, 0 for
    injected.
    """
    if ratio <= 0:
        return data
    n_inject = int(round(ratio * data.n_train))
    if n_inject <= 0:
        return data

    rng = np.random.default_rng(seed * 7919 + 31)
    train_sets = {u: set(items) for u, items in data.train_user_items.items()}
    # forbidden pairs for injection: observed train AND test interactions
    for u, items in data.test_user_items.items():
        train_sets.setdefault(int(u), set()).update(int(i) for i in items)
    # "pop": injected items are drawn with probability proportional to item training degree
    # (popularity-biased noise); "uniform": every item equally likely.
    item_p = None
    if noise_type == "pop":
        deg = np.asarray(data.user_item_csr.sum(axis=0)).ravel().astype(np.float64)
        item_p = deg / deg.sum()
    elif noise_type != "uniform":
        raise ValueError(f"unknown noise_type {noise_type}")

    inj_users = np.empty(n_inject, dtype=np.int64)
    inj_items = np.empty(n_inject, dtype=np.int64)
    placed = 0
    # Sample with rejection
    while placed < n_inject:
        bulk = max(1, n_inject - placed)
        cand_u = rng.integers(0, data.n_users, size=bulk)
        cand_i = (rng.integers(0, data.n_items, size=bulk) if item_p is None
                  else rng.choice(data.n_items, size=bulk, p=item_p))
        for u, i in zip(cand_u, cand_i):
            seen = train_sets.setdefault(int(u), set())
            if i in seen:
                continue
            inj_users[placed] = u
            inj_items[placed] = i
            seen.add(int(i))
            placed += 1
            if placed >= n_inject:
                break

    # Build new train structure
    new_train = {u: list(items) for u, items in data.train_user_items.items()}
    for u, i in zip(inj_users, inj_items):
        new_train.setdefault(int(u), []).append(int(i))

    new_pairs = np.concatenate(
        [data.train_pairs, np.stack([inj_users, inj_items], axis=1)], axis=0
    )
    new_labels = np.concatenate(
        [np.ones(data.n_train, dtype=np.int8), np.zeros(n_inject, dtype=np.int8)],
        axis=0,
    )

    rows = new_pairs[:, 0]
    cols = new_pairs[:, 1]
    vals = np.ones(new_pairs.shape[0], dtype=np.float32)
    new_csr = sp.csr_matrix(
        (vals, (rows, cols)), shape=(data.n_users, data.n_items)
    )
    new_csr.sum_duplicates()
    new_csr.data[:] = 1.0

    return InteractionData(
        n_users=data.n_users,
        n_items=data.n_items,
        train_user_items=new_train,
        test_user_items=deepcopy(data.test_user_items),
        train_pairs=new_pairs,
        user_item_csr=new_csr,
        edge_labels=new_labels,
    )
