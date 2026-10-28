"""Full-rank evaluation: Recall@K, NDCG@K, HR@K  with train-item exclusion.

This is *full-rank*: each test user is scored against every item, train items
are masked to -inf, and we read off the top-K. The earlier NGCF_with_Attention
code had only NDCG@10; here we report Recall/NDCG/HR at multiple Ks and across
user-degree slices for the long-tail experiment.

NDCG follows the standard definition where relevance is binary and the IDCG
sums 1 / log2(rank+1) for the top-min(|GT|, K) positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .data import InteractionData


# ---------------------------------------------------------------------------
# Top-K metrics
# ---------------------------------------------------------------------------

def _ndcg_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    """hits: 0/1 length-k array marking whether each top-k slot is a hit."""
    if n_relevant <= 0:
        return 0.0
    # DCG of the actual ranking
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(hits * discounts))
    # IDCG = best possible DCG given n_relevant relevant items in the top k
    ideal_len = min(n_relevant, k)
    idcg = float(np.sum(discounts[:ideal_len]))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(hits: np.ndarray, n_relevant: int) -> float:
    if n_relevant <= 0:
        return 0.0
    return float(hits.sum()) / float(n_relevant)


def _hit_at_k(hits: np.ndarray) -> float:
    return 1.0 if hits.sum() > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    metrics: Dict[str, float]            # overall mean across users
    per_user_hits: Dict[int, np.ndarray] # user -> hits-vector at max(K)
    K_max: int


class Evaluator:
    """Full-rank evaluator with optional user-degree slicing."""

    def __init__(self, data: InteractionData, ks: Iterable[int] = (10, 20)):
        self.data = data
        self.ks = sorted(set(int(k) for k in ks))
        self.k_max = max(self.ks)

    @torch.no_grad()
    def evaluate(
        self,
        get_user_emb,                    # callable -> (n_users, d) tensor
        get_item_emb,                    # callable -> (n_items, d) tensor
        device: torch.device,
        eval_batch: int = 512,
        user_subset: Optional[Iterable[int]] = None,
    ) -> EvalResult:
        """Evaluate the model.

        get_user_emb / get_item_emb return the *eval-time* embeddings (after
        propagation, attention, etc.). They take no arguments — the model
        prepares and caches them.
        """
        U = get_user_emb().to(device)
        I = get_item_emb().to(device)

        # Mask out users with no test items
        test_users = sorted(self.data.test_user_items.keys()) \
            if user_subset is None else sorted(user_subset)
        test_users = [u for u in test_users if self.data.test_user_items.get(u)]

        # Per-K accumulators
        recall = {k: 0.0 for k in self.ks}
        ndcg = {k: 0.0 for k in self.ks}
        hit = {k: 0.0 for k in self.ks}
        n_eval = 0
        per_user_hits: Dict[int, np.ndarray] = {}

        for start in range(0, len(test_users), eval_batch):
            batch_users = test_users[start:start + eval_batch]
            u_idx = torch.as_tensor(batch_users, device=device, dtype=torch.long)
            scores = U[u_idx] @ I.t()                 # (B, n_items)

            # Mask training interactions to -inf
            for row, u in enumerate(batch_users):
                seen = self.data.train_user_items.get(u, [])
                if seen:
                    scores[row, seen] = float("-inf")

            topk = torch.topk(scores, k=self.k_max, dim=1).indices.cpu().numpy()
            for row, u in enumerate(batch_users):
                gt = set(self.data.test_user_items[u])
                if not gt:
                    continue
                hits_full = np.array([1 if topk[row, j] in gt else 0
                                      for j in range(self.k_max)], dtype=np.int8)
                per_user_hits[u] = hits_full
                n_rel = len(gt)
                for k in self.ks:
                    h = hits_full[:k]
                    recall[k] += _recall_at_k(h, n_rel)
                    ndcg[k] += _ndcg_at_k(h, n_rel, k)
                    hit[k] += _hit_at_k(h)
                n_eval += 1

        metrics: Dict[str, float] = {}
        if n_eval > 0:
            for k in self.ks:
                metrics[f"recall@{k}"] = recall[k] / n_eval
                metrics[f"ndcg@{k}"] = ndcg[k] / n_eval
                metrics[f"hr@{k}"] = hit[k] / n_eval
        else:
            for k in self.ks:
                metrics[f"recall@{k}"] = 0.0
                metrics[f"ndcg@{k}"] = 0.0
                metrics[f"hr@{k}"] = 0.0
        metrics["n_eval_users"] = float(n_eval)

        return EvalResult(metrics=metrics, per_user_hits=per_user_hits, K_max=self.k_max)


# ---------------------------------------------------------------------------
# Mean +/- std across seeds
# ---------------------------------------------------------------------------

def aggregate_seeds(per_seed_metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Given a list of metric dicts, return {metric: (mean, std)}."""
    if not per_seed_metrics:
        return {}
    keys = sorted({k for m in per_seed_metrics for k in m})
    out: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = np.asarray([m.get(k, np.nan) for m in per_seed_metrics], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        out[k] = (float(vals.mean()), float(vals.std(ddof=0)))
    return out
