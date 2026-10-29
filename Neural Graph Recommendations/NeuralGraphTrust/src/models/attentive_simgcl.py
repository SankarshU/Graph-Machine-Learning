"""Attentive-SimGCL.

The LightGCN-side propagation is replaced with the residual-gated attentive
propagation of Attentive-LightGCN. Two perturbed views are built with the SimGCL
perturbation after every layer, and attention is RE-COMPUTED inside each view
(on the perturbed embeddings) rather than reusing the clean attention scores.
"""

from __future__ import annotations

import torch

from ._base import ContrastiveViewsMixin
from .attentive_lightgcn import AttentiveLightGCN


class AttentiveSimGCL(ContrastiveViewsMixin, AttentiveLightGCN):
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
        super().__init__(n_users, n_items, embed_dim=embed_dim, n_layers=n_layers, norm_adj=norm_adj,
                         neighbor_pad=neighbor_pad, attn_hidden=attn_hidden, gamma_init=gamma_init,
                         gamma_learnable=gamma_learnable, gamma_per_layer=gamma_per_layer)
        self.cl_eps = cl_eps
        self.cl_temp = cl_temp
