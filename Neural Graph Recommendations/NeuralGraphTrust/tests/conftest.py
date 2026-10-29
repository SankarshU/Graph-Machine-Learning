"""Shared fixtures: a tiny synthetic interaction graph written in the on-disk format and
loaded through src.data, so tests exercise the real loader."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_interactions  # noqa: E402
from src.utils import build_neighbor_pad, build_norm_adj, csr_to_torch_sparse  # noqa: E402


def write_split(root: Path, n_users=30, n_items=40, seed=0):
    """Each user gets 3-12 train items and 1-3 disjoint test items; item 0..n_items-1 all used."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    tr, te = [], []
    for u in range(n_users):
        items = rng.permutation(n_items)
        n_tr, n_te = int(rng.integers(3, 13)), int(rng.integers(1, 4))
        tr.append([u] + items[:n_tr].tolist())
        te.append([u] + items[n_tr:n_tr + n_te].tolist())
    tr[0] = [0] + list(range(n_items))[: n_items - 2]       # make sure the largest item ids appear
    te[0] = [0, n_items - 2, n_items - 1]
    for name, rows in (("train.txt", tr), ("test.txt", te)):
        (root / name).write_text("\n".join(" ".join(map(str, r)) for r in rows) + "\n")
    return root


@pytest.fixture
def data(tmp_path):
    return load_interactions(write_split(tmp_path / "toy"))


def graph_inputs(data, K=8, seed=0):
    """norm_adj and padded neighbour lists, as built by src.train.run_single."""
    dev = torch.device("cpu")
    norm_adj = csr_to_torch_sparse(build_norm_adj(data.n_users, data.n_items, data.user_item_csr), dev)
    pad = build_neighbor_pad(data.n_users, data.n_items, data.user_item_csr, K=K,
                             rng=np.random.default_rng(seed), device=dev)
    return norm_adj, pad
