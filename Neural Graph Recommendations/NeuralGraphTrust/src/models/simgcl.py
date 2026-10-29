"""SimGCL (Yu et al., SIGIR 2022).

LightGCN-style propagation, but at *every* propagation step the embeddings are
perturbed with random L2-normalized noise of magnitude epsilon:

    tilde_E = E + epsilon * Delta / ||Delta||_2

Two perturbed views are constructed independently and a contrastive (InfoNCE)
loss is added to BPR. The original SimGCL paper applies the noise per-layer;
we implement that. For evaluation we run a clean (no-noise) forward pass.
"""

from __future__ import annotations

import torch

from ._base import ContrastiveViewsMixin
from .lightgcn import LightGCN


class SimGCL(ContrastiveViewsMixin, LightGCN):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        norm_adj: torch.Tensor = None,
        cl_eps: float = 0.1,
        cl_temp: float = 0.2,
        **_,
    ):
        super().__init__(n_users, n_items, embed_dim=embed_dim, n_layers=n_layers, norm_adj=norm_adj)
        self.cl_eps = cl_eps
        self.cl_temp = cl_temp
