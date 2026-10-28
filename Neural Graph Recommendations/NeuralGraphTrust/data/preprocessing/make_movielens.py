"""Convert raw MovieLens (GroupLens) to the implicit train/test format used here.

Every rating is one implicit interaction; per user, a seeded random 20% of items
(rounded, at least 1) goes to test and the rest to train. Ids are remapped to
contiguous 0-based integers. Each output line: <user> <item_1> <item_2> ...

Usage:
  python -m data.preprocessing.make_movielens ml100k /path/to/ml-100k/u.data
  python -m data.preprocessing.make_movielens ml1m   /path/to/ml-1m/ratings.dat
"""
import collections
import pathlib
import sys

import numpy as np


def write(name, triples, seed=42):
    users = sorted({u for u, _ in triples}); items = sorted({i for _, i in triples})
    um = {u: k for k, u in enumerate(users)}; im = {i: k for k, i in enumerate(items)}
    d = collections.defaultdict(list)
    for u, i in triples:
        d[um[u]].append(im[i])
    rng = np.random.default_rng(seed); tr, te = {}, {}
    for u, its in d.items():
        its = np.array(sorted(set(its))); rng.shuffle(its); n = max(1, int(round(0.2 * len(its))))
        te[u] = its[:n].tolist(); tr[u] = its[n:].tolist()
    out = pathlib.Path(__file__).resolve().parents[1] / name; out.mkdir(parents=True, exist_ok=True)
    for fn, dd in (("train.txt", tr), ("test.txt", te)):
        with open(out / fn, "w") as f:
            for u in sorted(dd):
                f.write(" ".join(map(str, [u] + dd[u])) + "\n")
    print(name, "users", len(users), "items", len(items), "train", sum(map(len, tr.values())), "test", sum(map(len, te.values())))


if __name__ == "__main__":
    name, path = sys.argv[1], sys.argv[2]
    sep = "::" if path.endswith("ratings.dat") else "\t"
    triples = [tuple(map(int, l.split(sep)[:2])) for l in open(path, encoding="latin-1")]
    write(name, triples)
