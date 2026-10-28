"""Generate a tiny synthetic interaction dataset for smoke testing.

Use:  python -m data.preprocessing.make_toy --out data/toy --seed 0

Creates train.txt and test.txt in the format expected by `src.data.load_interactions`.
We plant `n_clusters` latent user-item affinity clusters so a model that
actually learns can beat random; this lets the smoke test verify that
gradients flow and metrics increase, not just that no crash occurs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def make_toy(out_dir: Path, n_users: int, n_items: int, n_clusters: int,
             interactions_per_user: int, test_holdout: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_cluster = rng.integers(0, n_clusters, size=n_users)
    items_per_cluster = np.array_split(rng.permutation(n_items), n_clusters)

    train_lines, test_lines = [], []
    for u in range(n_users):
        cluster_items = items_per_cluster[user_cluster[u]]
        # 80% of interactions from own cluster, 20% from random items
        own = rng.choice(cluster_items, size=int(interactions_per_user * 0.8), replace=False)
        out = rng.choice(n_items, size=interactions_per_user - own.size, replace=False)
        items = np.unique(np.concatenate([own, out]))
        if items.size <= test_holdout:
            test_holdout_u = max(1, items.size // 5)
        else:
            test_holdout_u = test_holdout
        rng.shuffle(items)
        test_items = items[:test_holdout_u]
        train_items = items[test_holdout_u:]
        train_lines.append(f"{u} " + " ".join(map(str, train_items)))
        test_lines.append(f"{u} " + " ".join(map(str, test_items)))

    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n")
    (out_dir / "test.txt").write_text("\n".join(test_lines) + "\n")
    print(f"[make_toy] wrote {out_dir}/train.txt and test.txt: "
          f"{n_users} users, {n_items} items, {n_clusters} clusters")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--n_users", type=int, default=200)
    ap.add_argument("--n_items", type=int, default=300)
    ap.add_argument("--n_clusters", type=int, default=8)
    ap.add_argument("--interactions_per_user", type=int, default=20)
    ap.add_argument("--test_holdout", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make_toy(
        Path(args.out), args.n_users, args.n_items, args.n_clusters,
        args.interactions_per_user, args.test_holdout, args.seed,
    )


if __name__ == "__main__":
    main()
