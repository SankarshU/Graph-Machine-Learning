"""LightGCN (He et al., SIGIR 2020).

E^{(l+1)} = hat_A E^{(l)},   hat_A = D^{-1/2} A D^{-1/2}
E_final = (1 / (L+1)) * sum_{l=0}^{L} E^{(l)}                 # mean over layers
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        norm_adj: torch.Tensor = None,        # torch.sparse_coo_tensor (U+I, U+I)
        **_,                                  # swallow extra kwargs from registry
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        if norm_adj is None:
            raise ValueError("LightGCN requires `norm_adj` (a torch sparse adjacency).")
        self.register_buffer("norm_adj", norm_adj, persistent=False)

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Cached propagated embeddings for evaluation.
        self._cached_user: torch.Tensor | None = None
        self._cached_item: torch.Tensor | None = None

    # ----------------------------------------------------------------
    # Propagation
    # ----------------------------------------------------------------
    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (user_embed_final, item_embed_final) after L layers."""
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_E = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.norm_adj, E)
            all_E.append(E)
        E_final = torch.stack(all_E, dim=1).mean(dim=1)
        u, i = torch.split(E_final, [self.n_users, self.n_items], dim=0)
        return u, i

    # ----------------------------------------------------------------
    # BPR forward (for the loss)
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
        # L2 reg on raw embedding-table entries (not propagated).
        u0 = self.user_embedding(users)
        ip0 = self.item_embedding(pos_items)
        in0 = self.item_embedding(neg_items)
        return pos_scores, neg_scores, (u0, ip0, in0)

    # ----------------------------------------------------------------
    # Eval-time embeddings
    # ----------------------------------------------------------------
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
