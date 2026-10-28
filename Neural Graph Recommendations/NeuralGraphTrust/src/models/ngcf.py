"""NGCF (Wang et al., SIGIR 2019) — clean reimplementation as historical baseline.

E^{(l)} = LeakyReLU(  (L + I) E^{(l-1)} W_1^{(l)}
                     + (L E^{(l-1)} ⊙ E^{(l-1)}) W_2^{(l)} )

We follow the standard NGCF formulation rather than the LBR submission's
modified version, because that submission had undefined symbols (rho_ui, e*)
which are exactly the notation problems the reviewer flagged. This NGCF is
identical in formulation to the original paper and serves only as a
historical baseline so we can claim a like-for-like comparison against the
prior work.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        norm_adj: torch.Tensor = None,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        if norm_adj is None:
            raise ValueError("NGCF requires `norm_adj`.")
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.register_buffer("norm_adj", norm_adj, persistent=False)

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Layer transforms — same dim per layer.
        self.W1 = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)])
        self.W2 = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)])
        for lin in list(self.W1) + list(self.W2):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self._cached_user: torch.Tensor | None = None
        self._cached_item: torch.Tensor | None = None

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_E = [E]
        for layer in range(self.n_layers):
            LE = torch.sparse.mm(self.norm_adj, E)
            term1 = self.W1[layer](LE + E)
            term2 = self.W2[layer](LE * E)
            E = F.leaky_relu(term1 + term2, negative_slope=0.2)
            E = F.dropout(E, p=self.dropout_rate, training=self.training)
            E = F.normalize(E, p=2, dim=1)
            all_E.append(E)
        E_final = torch.cat(all_E, dim=1)  # NGCF concatenates layer outputs
        u, i = torch.split(E_final, [self.n_users, self.n_items], dim=0)
        return u, i

    def forward(self, users, pos_items, neg_items):
        u_all, i_all = self.propagate()
        u = u_all[users]
        ip = i_all[pos_items]
        in_ = i_all[neg_items]
        pos_scores = (u * ip).sum(dim=-1)
        neg_scores = (u * in_).sum(dim=-1)
        u0 = self.user_embedding(users)
        ip0 = self.item_embedding(pos_items)
        in0 = self.item_embedding(neg_items)
        return pos_scores, neg_scores, (u0, ip0, in0)

    @torch.no_grad()
    def cache_eval_embeddings(self) -> None:
        was_training = self.training
        self.eval()
        u, i = self.propagate()
        self._cached_user = u.detach()
        self._cached_item = i.detach()
        if was_training:
            self.train()

    def eval_user_embeddings(self) -> torch.Tensor:
        if self._cached_user is None:
            self.cache_eval_embeddings()
        return self._cached_user

    def eval_item_embeddings(self) -> torch.Tensor:
        if self._cached_item is None:
            self.cache_eval_embeddings()
        return self._cached_item
