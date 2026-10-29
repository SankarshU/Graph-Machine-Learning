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

from ._base import GraphRecBase


class NGCF(GraphRecBase):
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
            raise ValueError(f"{type(self).__name__} requires `norm_adj`.")
        self.dropout_rate = dropout
        self.register_buffer("norm_adj", norm_adj, persistent=False)
        self._init_embeddings(n_users, n_items, embed_dim, n_layers)

        # Layer transforms — same dim per layer.
        self.W1 = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)])
        self.W2 = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)])
        for lin in list(self.W1) + list(self.W2):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _layer(self, layer: int, E_in: torch.Tensor) -> torch.Tensor:
        """One NGCF layer on the input table E_in (E^{(l-1)}, or the attention table in Stage 1)."""
        LE = torch.sparse.mm(self.norm_adj, E_in)
        term1 = self.W1[layer](LE + E_in)
        term2 = self.W2[layer](LE * E_in)
        E = F.leaky_relu(term1 + term2, negative_slope=0.2)
        E = F.dropout(E, p=self.dropout_rate, training=self.training)
        return F.normalize(E, p=2, dim=1)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_E = [E]
        for layer in range(self.n_layers):
            E = self._layer(layer, E)
            all_E.append(E)
        E_final = torch.cat(all_E, dim=1)  # NGCF concatenates layer outputs
        u, i = torch.split(E_final, [self.n_users, self.n_items], dim=0)
        return u, i
