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
import torch.nn as nn
import torch.nn.functional as F


class SimGCL(nn.Module):
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
        super().__init__()
        if norm_adj is None:
            raise ValueError("SimGCL requires `norm_adj`.")
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.cl_eps = cl_eps
        self.cl_temp = cl_temp
        self.register_buffer("norm_adj", norm_adj, persistent=False)

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self._cached_user: torch.Tensor | None = None
        self._cached_item: torch.Tensor | None = None

    # ----------------------------------------------------------------
    def _propagate(self, perturb: bool) -> tuple[torch.Tensor, torch.Tensor]:
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_E = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.norm_adj, E)
            if perturb:
                noise = torch.rand_like(E)  # uniform U(0,1), as in SimGCL (Yu et al. 2022)
                # normalize per-row, then sign-align by sign(E) as in the SimGCL paper
                noise = F.normalize(noise, dim=-1) * self.cl_eps * torch.sign(E)
                E = E + noise
            all_E.append(E)
        E_final = torch.stack(all_E, dim=1).mean(dim=1)
        u, i = torch.split(E_final, [self.n_users, self.n_items], dim=0)
        return u, i

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate(perturb=False)

    def propagate_perturbed(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate(perturb=True)

    # ----------------------------------------------------------------
    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ):
        u_all, i_all = self.propagate()
        u = u_all[users]
        ip = i_all[pos_items]
        in_ = i_all[neg_items]
        pos_scores = (u * ip).sum(dim=-1)
        neg_scores = (u * in_).sum(dim=-1)
        u0 = self.user_embedding(users)
        ip0 = self.item_embedding(pos_items)
        in0 = self.item_embedding(neg_items)

        # Two perturbed views (used by simgcl_cl_loss in losses.py)
        u_v1, i_v1 = self.propagate_perturbed()
        u_v2, i_v2 = self.propagate_perturbed()

        return pos_scores, neg_scores, (u0, ip0, in0), (u_v1, i_v1, u_v2, i_v2)

    @torch.no_grad()
    def cache_eval_embeddings(self) -> None:
        u, i = self.propagate()
        self._cached_user = u.detach()
        self._cached_item = i.detach()

    def eval_user_embeddings(self) -> torch.Tensor:
        if self._cached_user is None:
            self.cache_eval_embeddings()
        return self._cached_user

    def eval_item_embeddings(self) -> torch.Tensor:
        if self._cached_item is None:
            self.cache_eval_embeddings()
        return self._cached_item
