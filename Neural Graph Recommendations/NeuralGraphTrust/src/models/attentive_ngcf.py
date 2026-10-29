"""Attention-Augmented NGCF (Stage 1 of the paper), with masked attention.

Layer l (paper, Sec. Stage 1):
    e*_v = sum_k alpha_vk e_{nu(v,k)}          masked additive attention over the neighbor list
    E^(l) = LeakyReLU( (hat_A + I) E* W1 + (hat_A E* (.) E*) W2 )
i.e. the attention table replaces E^(l-1) inside NGCF's propagation, as in the late-breaking
submission, but padded slots are masked before the softmax (no repeated-neighbor padding).
"""
from __future__ import annotations

import torch

from ._attention import MaskedNeighborAttention
from .ngcf import NGCF


class AttentiveNGCF(NGCF):
    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3, norm_adj=None,
                 neighbor_pad=None, attn_hidden=64, dropout=0.1, **_):
        super().__init__(n_users, n_items, embed_dim=embed_dim, n_layers=n_layers,
                         norm_adj=norm_adj, dropout=dropout)
        if neighbor_pad is None:
            raise ValueError("AttentiveNGCF requires `neighbor_pad`.")
        self.register_buffer("neighbors", neighbor_pad.neighbors, persistent=False)
        self.register_buffer("valid_mask", neighbor_pad.valid_mask, persistent=False)
        self.attn = MaskedNeighborAttention(embed_dim, attn_hidden)

    def propagate(self, save_alpha_layer: int | None = None):
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_E = [E]
        for layer in range(self.n_layers):
            save = save_alpha_layer is not None and layer == save_alpha_layer
            E_star = self.attn(E, self.neighbors, self.valid_mask, save_alpha=save)
            E = self._layer(layer, E_star)
            all_E.append(E)
        E_final = torch.cat(all_E, dim=1)
        u, i = torch.split(E_final, [self.n_users, self.n_items], dim=0)
        return u, i
