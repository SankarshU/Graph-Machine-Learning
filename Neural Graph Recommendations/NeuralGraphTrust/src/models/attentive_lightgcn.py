"""Attentive-LightGCN.

Layer update (README §7.3):
    E^{(l+1)} = (1 - gamma) * hat_A E^{(l)}  +  gamma * E^{attn,(l)}

where E^{attn,(l)} is masked 1-hop attention over the *current* embeddings.

gamma can be:
  * fixed scalar,
  * single learnable scalar (sigmoid-bounded to (0, 1)),
  * per-layer learnable scalar.

Final embedding is the mean over layers (LightGCN convention) so the model
collapses to LightGCN exactly when gamma=0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._attention import MaskedNeighborAttention


class _Gamma(nn.Module):
    """Residual-gate parameter with three modes."""

    def __init__(self, n_layers: int, init: float, learnable: bool, per_layer: bool):
        super().__init__()
        self.n_layers = n_layers
        self.learnable = learnable
        self.per_layer = per_layer
        # store inverse-sigmoid so sigmoid(raw_init) ~= init
        init = max(min(float(init), 0.999), 0.001)
        raw = torch.log(torch.tensor(init / (1 - init)))
        if per_layer:
            param = nn.Parameter(raw.repeat(n_layers))
        else:
            param = nn.Parameter(raw.unsqueeze(0))
        if not learnable:
            param.requires_grad_(False)
        self.raw = param

    def value(self, layer: int) -> torch.Tensor:
        if self.per_layer:
            return torch.sigmoid(self.raw[layer])
        return torch.sigmoid(self.raw[0])

    def all_values(self) -> torch.Tensor:
        return torch.sigmoid(self.raw)


class AttentiveLightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        norm_adj: torch.Tensor = None,
        neighbor_pad=None,                  # utils.NeighborPad (or None to disable attention)
        attn_hidden: int = 64,
        gamma_init: float = 0.2,
        gamma_learnable: bool = True,
        gamma_per_layer: bool = False,
        **_,
    ):
        super().__init__()
        if norm_adj is None or neighbor_pad is None:
            raise ValueError("Attentive-LightGCN requires both `norm_adj` and `neighbor_pad`.")
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
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

    def propagate(
        self,
        save_alpha_layer: int | None = None,
        perturb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate L layers and mean-pool.

        save_alpha_layer : if not None, save attention alphas at that layer index
                           (only the last call's alphas are kept).
        perturb          : optional (delta_users, delta_items) tensors to add to
                           the *initial* embeddings before propagation. Used by
                           Attentive-SimGCL to build perturbed views.
        """
        E0_u = self.user_embedding.weight
        E0_i = self.item_embedding.weight
        if perturb is not None:
            d_u, d_i = perturb
            E0_u = E0_u + d_u
            E0_i = E0_i + d_i
        E = torch.cat([E0_u, E0_i], dim=0)
        all_E = [E]

        for layer in range(self.n_layers):
            E_lap = torch.sparse.mm(self.norm_adj, E)
            save = (save_alpha_layer is not None and layer == save_alpha_layer)
            E_attn = self.attn(E, self.neighbors, self.valid_mask, save_alpha=save)
            g = self.gamma.value(layer)
            E = (1.0 - g) * E_lap + g * E_attn
            all_E.append(E)

        E_final = torch.stack(all_E, dim=1).mean(dim=1)
        u, i = torch.split(E_final, [self.n_users, self.n_items], dim=0)
        return u, i

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
        return pos_scores, neg_scores, (u0, ip0, in0)

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
