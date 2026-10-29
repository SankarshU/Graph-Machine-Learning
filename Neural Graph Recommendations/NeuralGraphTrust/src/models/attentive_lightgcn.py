"""Attentive-LightGCN.

Layer update:
    E^{(l+1)} = (1 - gamma) * hat_A E^{(l)}  +  gamma * E^{attn,(l)}

where E^{attn,(l)} is masked 1-hop attention over the *current* embeddings.

gamma can be:
  * fixed scalar,
  * single learnable scalar (sigmoid-bounded to (0, 1)),
  * per-layer learnable scalar.

Final embedding is the mean over layers (LightGCN convention), so the model
tends to LightGCN as gamma -> 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._attention import MaskedNeighborAttention
from ._base import GraphRecBase, perturb_embeddings


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


class AttentiveLightGCN(GraphRecBase):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        norm_adj: torch.Tensor = None,
        neighbor_pad=None,                  # utils.NeighborPad
        attn_hidden: int = 64,
        gamma_init: float = 0.2,
        gamma_learnable: bool = True,
        gamma_per_layer: bool = False,
        **_,
    ):
        super().__init__()
        if norm_adj is None or neighbor_pad is None:
            raise ValueError(f"{type(self).__name__} requires both `norm_adj` and `neighbor_pad`.")
        self.register_buffer("norm_adj", norm_adj, persistent=False)
        self.register_buffer("neighbors", neighbor_pad.neighbors, persistent=False)
        self.register_buffer("valid_mask", neighbor_pad.valid_mask, persistent=False)
        self._init_embeddings(n_users, n_items, embed_dim, n_layers)

        self.attn = MaskedNeighborAttention(embed_dim, attn_hidden)
        self.gamma = _Gamma(n_layers, gamma_init, gamma_learnable, gamma_per_layer)

    def _propagate(
        self,
        perturb: bool = False,
        save_alpha_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate L layers and mean-pool.

        perturb          : add the SimGCL noise after every layer (Attentive-SimGCL views);
                           attention is recomputed on the perturbed embeddings of each layer.
        save_alpha_layer : if not None, save attention alphas at that layer index
                           (only the last call's alphas are kept).
        """
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_E = [E]
        for layer in range(self.n_layers):
            E_lap = torch.sparse.mm(self.norm_adj, E)
            save = (save_alpha_layer is not None and layer == save_alpha_layer)
            E_attn = self.attn(E, self.neighbors, self.valid_mask, save_alpha=save)
            g = self.gamma.value(layer)
            E = (1.0 - g) * E_lap + g * E_attn
            if perturb:
                E = perturb_embeddings(E, self.cl_eps)
            all_E.append(E)
        E_final = torch.stack(all_E, dim=1).mean(dim=1)
        u, i = torch.split(E_final, [self.n_users, self.n_items], dim=0)
        return u, i

    def propagate(self, save_alpha_layer: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate(perturb=False, save_alpha_layer=save_alpha_layer)
