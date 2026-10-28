"""Interaction-graph data loader.

Format on disk (matches the convention used by the prior NGCF_with_Attention code
and by LightGCN's released datasets):

  data/<name>/train.txt
  data/<name>/test.txt

Each line is space-separated:
  <user_id> <item_id_1> <item_id_2> ...
where user/item ids are 0-indexed contiguous integers.

If the dataset directory does not exist we look for a fallback under
`Neural Graph Recommendations/` (the historical location used by the user's
notebooks). If neither is present we raise with a clear message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOTS = [
    REPO_ROOT.parent / "Neural Graph Recommendations" / "data",
    REPO_ROOT.parent / "Neural Graph Recommendations" / "Code" / "data",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_split_file(path: Path) -> Dict[int, List[int]]:
    user_to_items: Dict[int, List[int]] = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                uid = int(parts[0])
                items = [int(x) for x in parts[1:]]
            except ValueError:
                continue
            user_to_items.setdefault(uid, []).extend(items)
    return user_to_items


def _resolve_data_dir(data_dir: str | Path) -> Path:
    p = Path(data_dir)
    if p.exists():
        return p
    # Try repo-relative
    if not p.is_absolute():
        candidate = REPO_ROOT / p
        if candidate.exists():
            return candidate
    # Legacy fallback
    name = p.name
    for legacy in LEGACY_ROOTS:
        cand = legacy / name
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not locate dataset directory: {data_dir}.\n"
        "Expected layout:  <data_dir>/train.txt and <data_dir>/test.txt"
    )


# ---------------------------------------------------------------------------
# InteractionData
# ---------------------------------------------------------------------------

@dataclass
class InteractionData:
    """Bipartite interaction data + sparse adjacency.

    Attributes
    ----------
    n_users, n_items : int
    train_user_items : Dict[int, List[int]]    user -> list of train item ids
    test_user_items  : Dict[int, List[int]]    user -> list of test item ids
    train_pairs      : np.ndarray (n_train, 2)  rows of (user, item)
    user_item_csr    : sp.csr_matrix (n_users, n_items)  binary 0/1
    edge_labels      : np.ndarray (n_train,)   1 = real edge, 0 = injected noise
                       (filled by noise_injection.add_random_edges)
    """

    n_users: int
    n_items: int
    train_user_items: Dict[int, List[int]]
    test_user_items: Dict[int, List[int]]
    train_pairs: np.ndarray
    user_item_csr: sp.csr_matrix
    edge_labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))

    @property
    def n_train(self) -> int:
        return self.train_pairs.shape[0]

    @property
    def n_test(self) -> int:
        return sum(len(v) for v in self.test_user_items.values())

    # -----------------------------------------------------------------
    # User-degree groups (long-tail evaluation hooks)
    # -----------------------------------------------------------------
    def user_degrees(self) -> np.ndarray:
        deg = np.zeros(self.n_users, dtype=np.int64)
        for u, items in self.train_user_items.items():
            deg[u] = len(items)
        return deg


def load_interactions(data_dir: str | Path) -> InteractionData:
    root = _resolve_data_dir(data_dir)
    train = _parse_split_file(root / "train.txt")
    test = _parse_split_file(root / "test.txt")

    # Determine n_users / n_items
    max_user = max(
        max(train.keys(), default=-1),
        max(test.keys(), default=-1),
    )
    max_item = -1
    for items in train.values():
        if items:
            max_item = max(max_item, max(items))
    for items in test.values():
        if items:
            max_item = max(max_item, max(items))
    n_users = max_user + 1
    n_items = max_item + 1
    if n_users <= 0 or n_items <= 0:
        raise ValueError(f"Empty dataset at {root}")

    rows, cols = [], []
    for u, items in train.items():
        rows.extend([u] * len(items))
        cols.extend(items)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    data = np.ones(rows.shape[0], dtype=np.float32)
    csr = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    csr.sum_duplicates()
    csr.data[:] = 1.0  # binary

    train_pairs = np.stack([rows, cols], axis=1)
    edge_labels = np.ones(train_pairs.shape[0], dtype=np.int8)

    return InteractionData(
        n_users=n_users,
        n_items=n_items,
        train_user_items={u: list(items) for u, items in train.items()},
        test_user_items={u: list(items) for u, items in test.items()},
        train_pairs=train_pairs,
        user_item_csr=csr,
        edge_labels=edge_labels,
    )


# ---------------------------------------------------------------------------
# BPR sampler
# ---------------------------------------------------------------------------

class BPRSampler:
    """Vectorized BPR negative sampler.

    For each (u, i+) it draws i- uniformly from items the user has NOT seen in
    the training set (rejection sampling against `train_user_items[u]`).
    """

    def __init__(self, data: InteractionData, rng: np.random.Generator):
        self.data = data
        self.rng = rng
        self._train_sets = {u: set(items) for u, items in data.train_user_items.items()}

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self.data.n_train, size=batch_size)
        users = self.data.train_pairs[idx, 0]
        pos_items = self.data.train_pairs[idx, 1]
        neg_items = np.empty(batch_size, dtype=np.int64)
        n_items = self.data.n_items
        for k, u in enumerate(users):
            seen = self._train_sets.get(int(u), set())
            while True:
                cand = int(self.rng.integers(0, n_items))
                if cand not in seen:
                    neg_items[k] = cand
                    break
        return users.astype(np.int64), pos_items.astype(np.int64), neg_items
