"""Utilities: seeding, device selection, sparse-tensor construction, structured logging.

Salvaged spirit from Neural Graph Recommendations / NGCF_with_Attention but rewritten
cleanly. In particular this fixes:
  * id-clamp bug (`clamp(max=n_item-1)` applied to user ids in the old code);
  * neighbor padding by repetition (old code repeated the last neighbor; we use a
    real `valid_mask` and ignore padded slots in attention softmax);
  * frozen-with-attributes anomaly (we never half-train: either all parameters
    train, or you explicitly freeze a tensor — no implicit trainable W with frozen E).
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set seeds for python, numpy, torch (cpu + cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is intentionally NOT enforced globally because cuBLAS
    # nondeterminism for sparse matmul can dominate. We rely on seed +
    # mean ± std across seeds for statistical reliability.


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec into a torch.device.

    spec: 'auto' | 'cpu' | 'cuda' | 'mps'
    """
    spec = (spec or "auto").lower()
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if spec == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Sparse adjacency  (LightGCN-style symmetric normalization on the bipartite
# graph extended to a (U+I) x (U+I) square matrix)
# ---------------------------------------------------------------------------

def build_norm_adj(
    n_users: int,
    n_items: int,
    user_item_csr: sp.csr_matrix,
) -> sp.csr_matrix:
    """Construct  hat_A = D^{-1/2} A D^{-1/2}  on the bipartite graph.

    A is a (U+I) x (U+I) symmetric matrix:
        [ 0      R   ]
        [ R^T    0   ]
    """
    n = n_users + n_items
    A = sp.dok_matrix((n, n), dtype=np.float32)
    R = user_item_csr.tolil()
    A[:n_users, n_users:] = R
    A[n_users:, :n_users] = R.T
    A = A.tocsr()

    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = np.power(deg, -0.5, out=np.zeros_like(deg, dtype=np.float64), where=deg > 0)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
    D = sp.diags(deg_inv_sqrt)
    return (D @ A @ D).tocsr()


def csr_to_torch_sparse(mat: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """Convert a scipy CSR matrix to a torch sparse COO tensor on `device`."""
    coo = mat.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


# ---------------------------------------------------------------------------
# Padded neighbor lists for masked attention
# ---------------------------------------------------------------------------

@dataclass
class NeighborPad:
    """Padded neighbor lists with a real validity mask.

    `neighbors[v, j]` is the global node id (in [0, n_users + n_items)) of the
    j-th sampled / kept neighbor of v.  `valid_mask[v, j]` is 1 if that slot is
    a real neighbor and 0 if it is a pad slot.

    For graphs where some nodes have fewer than `K` neighbors, pad slots are set
    to a sentinel id (we use 0) AND the mask is 0; downstream softmax MUST add
    the mask as -inf to invalid logits before softmax.
    """

    neighbors: torch.Tensor   # (n_nodes, K) long
    valid_mask: torch.Tensor  # (n_nodes, K) float (1 / 0)
    n_nodes: int
    K: int


def build_neighbor_pad(
    n_users: int,
    n_items: int,
    user_item_csr: sp.csr_matrix,
    K: int,
    rng: np.random.Generator,
    device: torch.device,
) -> NeighborPad:
    """Build padded neighbor lists for the bipartite graph.

    For each user u, neighbors are item ids shifted by +n_users (so that ids
    live in the same global [0, n_users+n_items) space as embeddings).
    For each item i, neighbors are user ids in [0, n_users).

    If a node has more than K neighbors we sample uniformly without replacement
    using `rng` (deterministic given seed). If fewer, we pad with id=0 and
    mask=0.
    """
    n_nodes = n_users + n_items
    neighbors = np.zeros((n_nodes, K), dtype=np.int64)
    mask = np.zeros((n_nodes, K), dtype=np.float32)

    # users
    R_csr = user_item_csr
    indptr, indices = R_csr.indptr, R_csr.indices
    for u in range(n_users):
        nb = indices[indptr[u]:indptr[u + 1]]
        if nb.size == 0:
            continue
        if nb.size > K:
            nb = rng.choice(nb, size=K, replace=False)
        k = nb.size
        neighbors[u, :k] = nb + n_users  # global id
        mask[u, :k] = 1.0

    # items
    R_T = user_item_csr.T.tocsr()
    indptr_t, indices_t = R_T.indptr, R_T.indices
    for i in range(n_items):
        nb = indices_t[indptr_t[i]:indptr_t[i + 1]]
        if nb.size == 0:
            continue
        if nb.size > K:
            nb = rng.choice(nb, size=K, replace=False)
        k = nb.size
        neighbors[n_users + i, :k] = nb
        mask[n_users + i, :k] = 1.0

    return NeighborPad(
        neighbors=torch.from_numpy(neighbors).to(device),
        valid_mask=torch.from_numpy(mask).to(device),
        n_nodes=n_nodes,
        K=K,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "attsimgcl", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(h)
    logger.propagate = False
    return logger


def append_jsonl(path: Path | str, record: Dict[str, Any]) -> None:
    """Append a single JSON record to a .jsonl file (creates parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@contextmanager
def timer():
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0
