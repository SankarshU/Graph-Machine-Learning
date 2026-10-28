"""Aggregate per-run JSON logs into a CSV table per dataset / experiment.

Usage:
    python -m experiments.aggregate clean
    python -m experiments.aggregate noise
    python -m experiments.aggregate long_tail
    python -m experiments.aggregate ablation

Reads every results/logs/<dataset>__runs.jsonl and groups by
(dataset, model, tag, noise_ratio), reporting mean and std of all numeric fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def aggregate(experiment_filter: str = "all") -> None:
    log_dir = REPO_ROOT / "results" / "logs"
    table_dir = REPO_ROOT / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    rows_by_key = defaultdict(list)
    for jsonl in log_dir.glob("*__runs.jsonl"):
        for rec in _read_jsonl(jsonl):
            tag = str(rec.get("tag", "default"))
            if experiment_filter != "all" and not (
                tag == experiment_filter or tag.startswith(experiment_filter)
            ):
                continue
            key = (rec.get("dataset"), rec.get("model"), tag, rec.get("noise_ratio", 0.0))
            rows_by_key[key].append(rec)

    if not rows_by_key:
        print(f"[aggregate] no runs match filter={experiment_filter!r}", file=sys.stderr)
        return

    # Determine the union of numeric columns
    all_cols = set()
    for runs in rows_by_key.values():
        for r in runs:
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    all_cols.add(k)
    metric_cols = sorted(all_cols)

    out_csv = table_dir / f"{experiment_filter}__summary.csv"
    with out_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        header = ["dataset", "model", "tag", "noise_ratio", "n_seeds"]
        for c in metric_cols:
            header += [f"{c}_mean", f"{c}_std"]
        wr.writerow(header)
        for (ds, mdl, tag, nr), runs in sorted(rows_by_key.items()):
            row = [ds, mdl, tag, nr, len(runs)]
            for c in metric_cols:
                vals = np.asarray([r.get(c, np.nan) for r in runs], dtype=np.float64)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    row += ["", ""]
                else:
                    row += [f"{vals.mean():.6f}", f"{vals.std(ddof=0):.6f}"]
            wr.writerow(row)
    print(f"[aggregate] wrote {out_csv} ({len(rows_by_key)} groups)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="all",
                    help="Tag prefix filter (e.g. 'clean', 'noise', 'long_tail', 'ablation').")
    args = ap.parse_args()
    aggregate(args.filter)


if __name__ == "__main__":
    main()
