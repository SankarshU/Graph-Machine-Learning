"""Long-tail user split + sliced evaluation.

Two splits are supported (README §9.3):

  * Quantile split: bottom 33% / middle 33% / top 33% by training-set degree.
  * Strict tail: tail = users with <=5 train interactions; medium = 6..20; head = >20.

Honesty: per the README, we use the phrase "sparse-user" / "long-tail",
not "cold-start". Cold-start would require explicitly held-out users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .data import InteractionData
from .evaluate import Evaluator


@dataclass
class UserGroups:
    name: str
    groups: Dict[str, np.ndarray]   # group_name -> user-id array


def quantile_split(data: InteractionData) -> UserGroups:
    deg = data.user_degrees()
    users = np.arange(data.n_users)
    # Use only users that appear in test (otherwise their degree doesn't matter)
    test_users = np.array(sorted(data.test_user_items.keys()), dtype=np.int64)
    deg_t = deg[test_users]
    if deg_t.size == 0:
        return UserGroups("quantile", {"sparse": users[:0], "medium": users[:0], "dense": users[:0]})
    q33, q66 = np.quantile(deg_t, [0.33, 0.66])
    groups = {
        "sparse": test_users[deg_t <= q33],
        "medium": test_users[(deg_t > q33) & (deg_t <= q66)],
        "dense": test_users[deg_t > q66],
    }
    return UserGroups("quantile", groups)


def strict_tail_split(data: InteractionData) -> UserGroups:
    deg = data.user_degrees()
    test_users = np.array(sorted(data.test_user_items.keys()), dtype=np.int64)
    deg_t = deg[test_users]
    groups = {
        "tail_le5": test_users[deg_t <= 5],
        "medium_6_20": test_users[(deg_t >= 6) & (deg_t <= 20)],
        "head_gt20": test_users[deg_t > 20],
    }
    return UserGroups("strict", groups)


def evaluate_by_group(
    evaluator: Evaluator,
    user_groups: UserGroups,
    get_user_emb,
    get_item_emb,
    device,
) -> Dict[str, Dict[str, float]]:
    """Run evaluator.evaluate restricted to each group."""
    out = {}
    for gname, uids in user_groups.groups.items():
        if uids.size == 0:
            out[gname] = {}
            continue
        res = evaluator.evaluate(get_user_emb, get_item_emb, device=device, user_subset=uids)
        out[gname] = res.metrics
    return out
