"""Masked 1-hop attention layer shared by Attentive-LightGCN and Attentive-SimGCL.

Specification (README §7.2):
    s_ij    = v_a^T tanh( W_a [e_i || e_j] )
    alpha_ij = softmax_{j in N(i)}  s_ij
    E^{attn}_i = sum_{j in N(i)}  alpha_ij  e_j

Critical implementation details:
  * attention is computed ONLY over true 1-hop neighbors;
  * padded slots are ignored by adding -inf to their logits BEFORE softmax;
  * attention scores are exposed via `last_alpha` for analysis (true vs injected
    edge mass, entropy by degree).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedNeighborAttention(nn.Module):
    """Masked attention over fixed-K padded neighbor lists.

    Inputs at forward:
      E_all : (N_total, d)   the embedding table for all U+I nodes
      neighbors  : (N_total, K) long  global neighbor ids (pad slots = 0)
      valid_mask : (N_total, K) float (1 valid, 0 pad)

    Output:
      E_attn : (N_total, d)   neighbor-attended representation
    """

    def __init__(self, embed_dim: int, attn_hidden: int = 64):
        super().__init__()
        self.W_a = nn.Linear(2 * embed_dim, attn_hidden, bias=True)
        self.v_a = nn.Linear(attn_hidden, 1, bias=False)
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.zeros_(self.W_a.bias)
        nn.init.xavier_uniform_(self.v_a.weight)

        # For attention diagnostics. Only filled when `save_alpha=True`.
        self.last_alpha: torch.Tensor | None = None
        self.last_neighbors: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None

    def forward(
        self,
        E_all: torch.Tensor,
        neighbors: torch.Tensor,
        valid_mask: torch.Tensor,
        save_alpha: bool = False,
    ) -> torch.Tensor:
        N, d = E_all.shape
        K = neighbors.shape[1]

        # W_a [e_i || e_j] + b  ==  (W_a[:, :d] e_i + b) + W_a[:, d:] e_j ; both halves are
        # computed once per node (N, h) and the neighbor half is gathered, which avoids
        # materialising the (N, K, 2d) concatenation. Same parameters, same function.
        W, b = self.W_a.weight, self.W_a.bias
        h_i = F.linear(E_all, W[:, :d], b)                     # (N, h)
        h_j = F.embedding(neighbors, F.linear(E_all, W[:, d:]))  # (N, K, h)
        s = self.v_a(torch.tanh(h_i.unsqueeze(1) + h_j)).squeeze(-1)   # (N, K)
        e_j = F.embedding(neighbors, E_all)                    # (N, K, d) for the weighted sum

        # Mask padded slots BEFORE softmax.
        s = s.masked_fill(valid_mask < 0.5, float("-inf"))

        # If a node has zero neighbors, every slot is -inf -> softmax NaN.
        # Detect such rows and short-circuit to a zero row.
        no_neighbors = (valid_mask.sum(dim=1, keepdim=True) < 0.5)   # (N, 1)
        # Replace -inf rows with zeros for softmax, then zero them out below.
        s = torch.where(no_neighbors.expand_as(s), torch.zeros_like(s), s)

        alpha = F.softmax(s, dim=1)                             # (N, K)
        alpha = alpha * (1.0 - no_neighbors.float())            # zero rows for isolated nodes

        # Weighted sum: (N, K, 1) * (N, K, d) -> (N, d)
        E_attn = (alpha.unsqueeze(-1) * e_j).sum(dim=1)

        if save_alpha:
            self.last_alpha = alpha.detach()
            self.last_neighbors = neighbors.detach()
            self.last_mask = valid_mask.detach()

        return E_attn

    def attention_entropy(self) -> torch.Tensor | None:
        """Per-node attention entropy from the last forward pass.

        H_i = - sum_j alpha_ij  log(alpha_ij)         (over valid slots only)
        Returns a 1D tensor of length N or None if no save_alpha pass yet.
        """
        if self.last_alpha is None:
            return None
        a = self.last_alpha.clamp_min(1e-12)
        return -(a * a.log()).sum(dim=1)
