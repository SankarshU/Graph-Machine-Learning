"""Thin wrapper: train + sliced (long-tail) evaluation in one process.

We do NOT re-instantiate the model after training. Instead we pass
`long_tail=True` to `src.train.run_single`, which performs the sliced eval
on the very same in-memory weights.
"""

from __future__ import annotations

import argparse

from src.train import load_config, run_single


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tag", default="long_tail")
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--noise_ratio", type=float, default=0.0)
    p.add_argument("--attn_diagnostics", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_single(
        cfg, args.model, seed=args.seed, tag=args.tag,
        device_spec=args.device, epochs=args.epochs,
        noise_ratio=args.noise_ratio,
        long_tail=True,
        attn_diagnostics=args.attn_diagnostics,
    )


if __name__ == "__main__":
    main()
