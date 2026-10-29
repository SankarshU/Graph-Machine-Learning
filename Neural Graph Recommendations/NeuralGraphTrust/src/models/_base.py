"""Shared pieces of the graph-CF models: embedding tables, the BPR forward pass,
cached evaluation embeddings, and the SimGCL perturbation / contrastive views.

Subclasses implement `propagate()` (clean pass) and, if contrastive, `_propagate(perturb=True)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def perturb_embeddings(E: torch.Tensor, eps: float) -> torch.Tensor:
    """SimGCL perturbation (Yu et al. 2022): uniform U(0,1) noise, L2-normalised per row,
    scaled by eps and sign-aligned with E."""
    noise = torch.rand_like(E)
    noise = F.normalize(noise, dim=-1) * eps * torch.sign(E)
    return E + noise


class GraphRecBase(nn.Module):
    """Embedding tables + BPR forward + evaluation cache."""

    def _init_embeddings(self, n_users: int, n_items: int, embed_dim: int, n_layers: int) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        # Cached propagated embeddings for evaluation.
        self._cached_user: torch.Tensor | None = None
        self._cached_item: torch.Tensor | None = None

    def propagate(self, **kw) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
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

    @torch.no_grad()
    def cache_eval_embeddings(self, save_alpha_layer: int | None = None) -> None:
        """Clean propagation in eval mode (dropout off); attentive models can also save
        the attention weights of one layer."""
        was_training = self.training
        self.eval()
        u, i = self.propagate() if save_alpha_layer is None else self.propagate(save_alpha_layer=save_alpha_layer)
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


class ContrastiveViewsMixin:
    """SimGCL-style training pass: BPR outputs plus two independently perturbed views
    (consumed by losses.simgcl_cl_loss). Requires `_propagate(perturb=True)` and `cl_eps`."""

    def propagate_perturbed(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate(perturb=True)

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        pos_scores, neg_scores, regs = super().forward(users, pos_items, neg_items)
        u_v1, i_v1 = self.propagate_perturbed()
        u_v2, i_v2 = self.propagate_perturbed()
        return pos_scores, neg_scores, regs, (u_v1, i_v1, u_v2, i_v2)
