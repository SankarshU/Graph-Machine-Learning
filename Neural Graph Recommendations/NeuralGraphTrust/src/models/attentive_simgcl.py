"""Attentive-SimGCL.

The LightGCN-side propagation is replaced with the residual-gated attentive
propagation from Attentive-LightGCN. Two perturbed views are constructed by
adding L2-normalized noise to the *initial* embedding tables (per SimGCL),
and attention is RE-COMPUTED for each view rather than reusing the clean
attention scores. This directly addresses the README §7.4 requirement
"Recompute attention for each perturbed view. Do not reuse stale attention
weights."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._attention import MaskedNeighborAttention
from .attentive_lightgcn import _Gamma


class AttentiveSimGCL(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        norm_adj: torch.Tensor = None,
        neighbor_pad=None,
        attn_hidden: int = 64,
        gamma_init: float = 0.2,
        gamma_learnable: bool = True,
        gamma_per_layer: bool = False,
        cl_eps: float = 0.1,
        cl_temp: float = 0.2,
        **_,
    ):
        super().__init__()
        if norm_adj is None or neighbor_pad is None:
            raise ValueError("Attentive-SimGCL requires `norm_adj` and `neighbor_pad`.")
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.cl_eps = cl_eps
        self.cl_temp = cl_temp
        self.register_buffer("norm_adj", norm_adj, persistent=False)
        self.register_buffer("neighbors", neighbor_pad.neighbors, persistent=False)
        self.register_buffer("valid_mask", neighbor_pad.valid_mask, persistent=False)

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.attn = MaskedNeighborAttention(embed_dim, attn_hidden)
        self.gamma = _Gamma(n_layers, gamma_init, gamma_learnable, gamma_per_layer)

        self._cached_user: torch.Tensor | None = None
        self._cached_item: torch.Tensor | None = None

    # ----------------------------------------------------------------
    def _propagate(
        self,
        perturb: bool,
        save_alpha_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_E = [E]
        for layer in range(self.n_layers):
            E_lap = torch.sparse.mm(self.norm_adj, E)
            save = (save_alpha_layer is not None and layer == save_alpha_layer)
            E_attn = self.attn(E, self.neighbors, self.valid_mask, save_alpha=save)
            g = self.gamma.value(layer)
            E = (1.0 - g) * E_lap + g * E_attn
            if perturb:
                noise = torch.rand_like(E)  # uniform U(0,1), as in SimGCL (Yu et al. 2022)
                noise = F.normalize(noise, dim=-1) * self.cl_eps * torch.sign(E)
                E = E + noise
            all_E.append(E)
        E_final = torch.stack(all_E, dim=1).mean(dim=1)
        u, i = torch.split(E_final, [self.n_users, self.n_items], dim=0)
        return u, i

    def propagate(self, save_alpha_layer: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate(perturb=False, save_alpha_layer=save_alpha_layer)

    def propagate_perturbed(self) -> tuple[torch.Tensor, torch.Tensor]:
        # IMPORTANT: this calls self.attn fresh every layer with the perturbed E,
        # so attention weights are computed *for the noisy view*, not stale.
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

        u_v1, i_v1 = self.propagate_perturbed()
        u_v2, i_v2 = self.propagate_perturbed()

        return pos_scores, neg_scores, (u0, ip0, in0), (u_v1, i_v1, u_v2, i_v2)

    @torch.no_grad()
    def cache_eval_embeddings(self, save_alpha_layer: int | None = None) -> None:
        u, i = self.propagate(save_alpha_layer=save_alpha_layer)
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
