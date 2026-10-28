"""Losses: BPR, InfoNCE / SimGCL contrastive, and L2 weight regularization.

Definitions
-----------
BPR (Rendle et al., 2009):
    L_BPR = - mean_{(u, i+, i-)}  log sigmoid(s(u, i+) - s(u, i-))

SimGCL contrastive loss (Yu et al., 2022):
    Per perturbed view v1, v2 of the embedding table, compute InfoNCE over
    users and over items:
        L_CL_user = - 1/|B|  sum_{u in B}  log
            exp(<v1_u, v2_u> / tau)  /  sum_{u' in B} exp(<v1_u, v2_{u'}> / tau)
    Symmetric for items. We return  L_CL = (L_CL_user + L_CL_item) / 2.

L2 regularization:
    Applied directly to the *initial* user and item embeddings (the table
    parameters) for the (u, i+, i-) triplet, not the propagated embeddings.
    This matches LightGCN / SimGCL convention.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """L_BPR = - mean log sigmoid(pos - neg).

    Uses softplus for numerical stability:
        - log sigmoid(x) = softplus(-x)
    """
    return F.softplus(neg_scores - pos_scores).mean()


def l2_reg(*tensors: torch.Tensor) -> torch.Tensor:
    """Sum-of-squares L2 norm over multiple tensors (mean over batch)."""
    if not tensors:
        return torch.tensor(0.0)
    total = 0.0
    n = tensors[0].shape[0] if tensors[0].dim() else 1
    for t in tensors:
        total = total + (t ** 2).sum() / max(n, 1)
    return 0.5 * total


def infonce(
    z1: torch.Tensor,         # (B, d)  view 1
    z2: torch.Tensor,         # (B, d)  view 2
    temperature: float = 0.2,
) -> torch.Tensor:
    """Symmetric InfoNCE between two views of the SAME batch of nodes.

    Positives are diagonal pairs (z1_i, z2_i). Negatives are the other rows.
    This matches the SimGCL formulation.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = (z1 @ z2.t()) / temperature             # (B, B)
    labels = torch.arange(z1.size(0), device=z1.device)
    # Symmetric: contrast both directions
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def simgcl_cl_loss(
    user_view1: torch.Tensor, user_view2: torch.Tensor,
    item_view1: torch.Tensor, item_view2: torch.Tensor,
    user_idx: torch.Tensor, item_idx: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Combined user-side + item-side InfoNCE on the BPR mini-batch users/items.

    Indices are de-duplicated first (as in the SimGCL reference code): a BPR batch
    samples users/items with replacement, and duplicate rows would otherwise be
    treated as negatives of themselves.
    """
    user_idx = torch.unique(user_idx)
    item_idx = torch.unique(item_idx)
    z_u_1 = user_view1[user_idx]
    z_u_2 = user_view2[user_idx]
    z_i_1 = item_view1[item_idx]
    z_i_2 = item_view2[item_idx]
    return 0.5 * (infonce(z_u_1, z_u_2, temperature) + infonce(z_i_1, z_i_2, temperature))
