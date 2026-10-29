"""LightGCN (He et al., SIGIR 2020).

E^{(l+1)} = hat_A E^{(l)},   hat_A = D^{-1/2} A D^{-1/2}
E_final = (1 / (L+1)) * sum_{l=0}^{L} E^{(l)}                 # mean over layers
"""

from __future__ import annotations

import torch

from ._base import GraphRecBase, perturb_embeddings


class LightGCN(GraphRecBase):
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
        if norm_adj is None:
            raise ValueError(f"{type(self).__name__} requires `norm_adj` (a torch sparse adjacency).")
        self.register_buffer("norm_adj", norm_adj, persistent=False)
        self._init_embeddings(n_users, n_items, embed_dim, n_layers)

    def _propagate(self, perturb: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """L layers and mean-pool; `perturb` adds the SimGCL noise after every layer."""
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_E = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.norm_adj, E)
            if perturb:
                E = perturb_embeddings(E, self.cl_eps)
            all_E.append(E)
        E_final = torch.stack(all_E, dim=1).mean(dim=1)
        u, i = torch.split(E_final, [self.n_users, self.n_items], dim=0)
        return u, i

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (user_embed_final, item_embed_final) after L layers."""
        return self._propagate(perturb=False)
