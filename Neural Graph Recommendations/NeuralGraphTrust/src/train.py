"""Single-command trainer for any model + dataset + seed.

Examples
--------
# Smoke test (CPU-friendly):
python -m src.train --config configs/ml100k.yaml --model lightgcn --seed 2024 --epochs 5

# Full SimGCL run:
python -m src.train --config configs/ml1m.yaml --model attentive_simgcl --seed 2024

# 10% noise injection:
python -m src.train --config configs/ml100k.yaml --model attentive_lightgcn \
    --seed 2024 --noise_ratio 0.10

# Tag a run:
python -m src.train --config configs/ml100k.yaml --model attentive_simgcl \
    --seed 2024 --tag clean
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .data import BPRSampler, InteractionData, load_interactions
from .evaluate import EvalResult, Evaluator, aggregate_seeds
from .losses import bpr_loss, l2_reg, simgcl_cl_loss
from .models import build_model
from .utils import (
    append_jsonl,
    build_neighbor_pad,
    build_norm_adj,
    count_parameters,
    csr_to_torch_sparse,
    get_logger,
    resolve_device,
    set_seed,
    timer,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
log = get_logger("train")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_single(
    cfg: Dict[str, Any],
    model_name: str,
    seed: int,
    noise_ratio: float = 0.0,
    epochs: int | None = None,
    tag: str = "default",
    device_spec: str | None = None,
    K_attn: int = 32,
    long_tail: bool = False,
    attn_diagnostics: bool = False,
    return_model: bool = False,
    noise_type: str = "uniform",
) -> Dict[str, Any]:
    """Train one model, return final eval metrics dict (with extras).

    If `long_tail=True`, after training also runs sliced evaluation
    (sparse / medium / dense + tail<=5 / 6-20 / >20) on the SAME model and
    appends per-group rows to results/logs/<dataset>__runs.jsonl.

    If `attn_diagnostics=True` and the model has an `attn` module, also runs
    attention_analysis on a save_alpha pass and writes a JSON summary +
    PNG histogram.
    """
    set_seed(seed)

    # ---- Data
    data = load_interactions(cfg["dataset"]["data_dir"])
    log.info(
        f"Loaded {cfg['dataset']['name']}: "
        f"users={data.n_users} items={data.n_items} "
        f"train_pairs={data.n_train} test_pairs={data.n_test}"
    )

    if noise_ratio and noise_ratio > 0:
        from .noise_injection import add_random_edges
        data = add_random_edges(data, ratio=float(noise_ratio), seed=seed, noise_type=noise_type)
        log.info(f"Injected noise_ratio={noise_ratio}: now {data.n_train} train pairs "
                 f"({(data.edge_labels == 0).sum()} injected).")

    device = resolve_device(device_spec or cfg["train"].get("device", "auto"))
    log.info(f"Using device={device}")

    # ---- Adjacency + neighbor pad
    norm_adj_csr = build_norm_adj(data.n_users, data.n_items, data.user_item_csr)
    norm_adj = csr_to_torch_sparse(norm_adj_csr, device)

    rng = np.random.default_rng(seed)
    pad = build_neighbor_pad(
        data.n_users, data.n_items, data.user_item_csr, K=K_attn,
        rng=rng, device=device,
    )

    # ---- Model
    mcfg = cfg["model"]
    model = build_model(
        model_name,
        n_users=data.n_users,
        n_items=data.n_items,
        embed_dim=mcfg["embed_dim"],
        n_layers=mcfg["n_layers"],
        norm_adj=norm_adj,
        neighbor_pad=pad,
        attn_hidden=mcfg.get("attn_hidden", 64),
        gamma_init=mcfg.get("gamma_init", 0.2),
        gamma_learnable=mcfg.get("gamma_learnable", True),
        gamma_per_layer=mcfg.get("gamma_per_layer", False),
        cl_eps=mcfg.get("cl_eps", 0.1),
        cl_temp=mcfg.get("cl_temp", 0.2),
    ).to(device)
    n_params = count_parameters(model)
    log.info(f"Model={model_name} parameters={n_params:,}")

    # ---- Sampler / optimizer / evaluator
    sampler = BPRSampler(data, rng=np.random.default_rng(seed + 1))
    tcfg = cfg["train"]
    optim = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

    evaluator = Evaluator(data, ks=tcfg.get("ks", [10, 20]))

    # ---- Train loop
    n_epochs = int(epochs if epochs is not None else tcfg["epochs"])
    batch_size = int(tcfg["batch_size"])
    n_batches = max(1, data.n_train // batch_size)
    cl_lambda = float(tcfg.get("cl_lambda", 0.0))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    cl_temp = float(mcfg.get("cl_temp", 0.2))
    early_stop_patience = int(tcfg.get("early_stop_patience", 1_000_000))
    eval_every = int(tcfg.get("eval_every", 5))

    is_cl_model = model_name in ("simgcl", "attentive_simgcl")
    best_metric = -1.0
    best_metrics: Dict[str, float] = {}
    epochs_no_improve = 0

    epoch_times = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        total_bpr = 0.0
        total_cl = 0.0
        for _ in range(n_batches):
            users_np, pos_np, neg_np = sampler.sample_batch(batch_size)
            users = torch.from_numpy(users_np).to(device)
            pos_items = torch.from_numpy(pos_np).to(device)
            neg_items = torch.from_numpy(neg_np).to(device)

            optim.zero_grad()

            if is_cl_model:
                pos_s, neg_s, regs, views = model(users, pos_items, neg_items)
                u_v1, i_v1, u_v2, i_v2 = views
                bpr = bpr_loss(pos_s, neg_s)
                cl = simgcl_cl_loss(
                    u_v1, u_v2, i_v1, i_v2,
                    user_idx=users, item_idx=pos_items,
                    temperature=cl_temp,
                )
                reg = l2_reg(*regs) * weight_decay
                loss = bpr + cl_lambda * cl + reg
                total_cl += float(cl.item())
            else:
                pos_s, neg_s, regs = model(users, pos_items, neg_items)
                bpr = bpr_loss(pos_s, neg_s)
                reg = l2_reg(*regs) * weight_decay
                loss = bpr + reg

            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            total_bpr += float(bpr.item())

        epoch_t = time.perf_counter() - t0
        epoch_times.append(epoch_t)

        if epoch % eval_every == 0 or epoch == n_epochs:
            model.eval()
            # Force a fresh propagate cache for evaluation.
            model._cached_user = None
            model._cached_item = None
            res = evaluator.evaluate(
                get_user_emb=model.eval_user_embeddings,
                get_item_emb=model.eval_item_embeddings,
                device=device,
            )
            target = res.metrics.get("recall@20", res.metrics.get("recall@10", 0.0))
            log.info(
                f"epoch {epoch:>4d}/{n_epochs} | "
                f"loss={total_loss / n_batches:.4f} bpr={total_bpr / n_batches:.4f} "
                + (f"cl={total_cl / n_batches:.4f} " if is_cl_model else "")
                + f"| recall@10={res.metrics.get('recall@10', 0):.4f} "
                + f"recall@20={res.metrics.get('recall@20', 0):.4f} "
                + f"ndcg@10={res.metrics.get('ndcg@10', 0):.4f} "
                + f"ndcg@20={res.metrics.get('ndcg@20', 0):.4f} "
                + f"| t={epoch_t:.2f}s"
            )
            if target > best_metric:
                best_metric = target
                best_metrics = dict(res.metrics)
                epochs_no_improve = 0
            else:
                epochs_no_improve += eval_every
            if epochs_no_improve >= early_stop_patience:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {epochs_no_improve} epochs).")
                break

    # ---- Final summary
    out: Dict[str, Any] = {
        "model": model_name,
        "dataset": cfg["dataset"]["name"],
        "seed": seed,
        "noise_ratio": noise_ratio,
        "noise_type": noise_type,
        "tag": tag,
        "n_params": n_params,
        "epoch_time_mean": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "epoch_time_std": float(np.std(epoch_times)) if epoch_times else 0.0,
        **{k: v for k, v in best_metrics.items() if isinstance(v, (int, float))},
    }
    log_path = REPO_ROOT / cfg["experiment"]["log_dir"] / f"{cfg['dataset']['name']}__{model_name}__{tag}__seed{seed}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Wrote {log_path}")

    # Append a row to the dataset-level CSV-of-runs (jsonl)
    run_jsonl = REPO_ROOT / cfg["experiment"]["log_dir"] / f"{cfg['dataset']['name']}__runs.jsonl"
    append_jsonl(run_jsonl, out)

    # ---- Optional: long-tail sliced evaluation (same in-memory model)
    if long_tail:
        from .long_tail_split import (
            evaluate_by_group, quantile_split, strict_tail_split,
        )
        model.eval()
        model._cached_user = None
        model._cached_item = None
        qs = quantile_split(data)
        sg = strict_tail_split(data)
        res_q = evaluate_by_group(evaluator, qs,
                                  model.eval_user_embeddings,
                                  model.eval_item_embeddings, device=device)
        res_s = evaluate_by_group(evaluator, sg,
                                  model.eval_user_embeddings,
                                  model.eval_item_embeddings, device=device)
        for split_name, group_dict in (("quantile", res_q), ("strict", res_s)):
            for gname, metrics in group_dict.items():
                row = {
                    "dataset": cfg["dataset"]["name"],
                    "model": model_name,
                    "seed": seed,
                    "tag": f"{tag}__{split_name}__{gname}",
                    "noise_ratio": noise_ratio,
                    "n_params": n_params,
                }
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        row[k] = v
                append_jsonl(run_jsonl, row)
        log.info(f"long_tail evaluation done: quantile={list(res_q.keys())} strict={list(res_s.keys())}")

    # ---- Optional: attention diagnostics (only meaningful for attentive_*)
    if attn_diagnostics and hasattr(model, "attn"):
        from .attention_analysis import analyze_attention
        # Force a save_alpha pass at the last layer.
        model._cached_user = None
        model._cached_item = None
        if hasattr(model, "cache_eval_embeddings"):
            try:
                model.cache_eval_embeddings(save_alpha_layer=model.n_layers - 1)
            except TypeError:
                model.cache_eval_embeddings()
        try:
            attn_summary = analyze_attention(
                model, data,
                out_dir=REPO_ROOT / cfg["experiment"]["figure_dir"],
                tag=f"{cfg['dataset']['name']}__{model_name}__seed{seed}__{tag}",
            )
            row = {
                "dataset": cfg["dataset"]["name"],
                "model": model_name, "seed": seed,
                "tag": f"{tag}__attn_diag",
                "noise_ratio": noise_ratio,
            }
            for k, v in attn_summary.items():
                if isinstance(v, (int, float)):
                    row[k] = v
            append_jsonl(run_jsonl, row)
            log.info(f"attention diagnostics: {attn_summary}")
        except Exception as e:
            log.warning(f"attention diagnostics failed: {e}")

    if return_model:
        return out, model, data, evaluator, device
    return out


# ---------------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------------

def run_multiseed(
    cfg: Dict[str, Any],
    model_name: str,
    seeds: list[int],
    **kw,
) -> Dict[str, tuple[float, float]]:
    metrics_per_seed = []
    for seed in seeds:
        out = run_single(cfg, model_name, seed=seed, **kw)
        metrics_per_seed.append(out)
    agg = aggregate_seeds(metrics_per_seed)

    # Persist mean/std table
    table_path = REPO_ROOT / cfg["experiment"]["table_dir"] / f"{cfg['dataset']['name']}__{model_name}__{kw.get('tag','default')}.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w") as f:
        json.dump({k: {"mean": m, "std": s} for k, (m, s) in agg.items()}, f, indent=2)
    log.info(f"Wrote aggregated table {table_path}")
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to dataset config YAML.")
    p.add_argument("--model", required=True,
                   choices=["ngcf", "attentive_ngcf", "lightgcn", "attentive_lightgcn",
                            "simgcl", "attentive_simgcl"])
    p.add_argument("--seed", type=int, default=None,
                   help="Single seed. If omitted, run all seeds in config.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--noise_ratio", type=float, default=0.0)
    p.add_argument("--noise_type", type=str, default="uniform", choices=["uniform", "pop"])
    p.add_argument("--tag", type=str, default="default")
    p.add_argument("--device", type=str, default=None,
                   choices=[None, "auto", "cpu", "cuda", "mps"])
    p.add_argument("--K_attn", type=int, default=32,
                   help="Number of padded neighbors per node.")
    p.add_argument("--long_tail", action="store_true",
                   help="Also run sliced (sparse/medium/dense + tail<=5) eval.")
    p.add_argument("--attn_diagnostics", action="store_true",
                   help="If model is attentive, save alpha histogram + true/injected mass.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        run_single(
            cfg, args.model, seed=args.seed,
            noise_ratio=args.noise_ratio, epochs=args.epochs,
            tag=args.tag, device_spec=args.device, K_attn=args.K_attn,
            long_tail=args.long_tail, attn_diagnostics=args.attn_diagnostics,
            noise_type=args.noise_type,
        )
    else:
        run_multiseed(
            cfg, args.model, seeds=cfg["experiment"]["seeds"],
            noise_ratio=args.noise_ratio, epochs=args.epochs,
            tag=args.tag, device_spec=args.device, K_attn=args.K_attn,
            long_tail=args.long_tail, attn_diagnostics=args.attn_diagnostics,
        )


if __name__ == "__main__":
    main()
