"""NeuralGraphTrust evaluation on one trained attentive model. Everything after training is
inference-time: the graph used for propagation is edited, the parameters are frozen.
The library code is in src/graphtrust/; this script runs the study and writes one log line.

For one (dataset, model, noise ratio, noise type, seed):
  1. Train on the noise-injected graph (src.train.run_single); save a checkpoint.
  2. Edge table over the user-side attended edges (graphtrust.reliability): label, r, rel, dot, cos,
     and the AUROC of each score for original-vs-injected (only when injected edges exist).
  3. Policies (graphtrust.policies), per user, cap c on the number of masked edges:
       pi0(tau)        mask attended edges with r < tau, lowest r first         (tau and cap sweeps)
       pi_rel          mask attended edges with rel < 1, lowest rel first
       matched controls at the counts of pi0(tau_main): lowest dot, lowest cos, random, oracle
       soft            pi0(tau_main) edges kept but scaled by rho in the adjacency
  4. For each policy: masked count, precision (fraction injected), recall of injected edges,
     Recall@20 / NDCG@20 after re-propagation (all users at once), and, repairing each user in
     isolation (only u's own edges change):
       gate      : revert a user's repair when the top-k Jaccard distance exceeds budget b
       targeted  : repair only the riskiest half of candidate users (risk = share of edges with r<tau)
       groups    : paired effect by original user degree tercile (sparse / medium / dense)
       fidelity  : slate change caused by the cited edges vs a random same-size set
  5. One JSON line -> results/logs/<dataset>__graphtrust.jsonl

Usage:
  python -m experiments.graphtrust_eval --config configs/ml100k_arxiv.yaml --model attentive_lightgcn \
      --noise 0.1 --noise_type uniform --seed 2024 --K_attn 256 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from src.graphtrust import (build_policies, degree_groups, edge_table, edge_weights, fidelity,
                            fill_unrepaired, gate, isolated_user_eval, jaccard_dist, nanmean,
                            original_degree, per_user_eval, propagate_on_graph, reliability_summary,
                            target_users)
from src.train import load_config, run_single, REPO_ROOT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True, choices=["attentive_lightgcn", "attentive_simgcl"])
    p.add_argument("--noise", type=float, required=True)
    p.add_argument("--noise_type", default="uniform", choices=["uniform", "pop"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--K_attn", type=int, default=256)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--taus", default="0.25,0.5,0.75")
    p.add_argument("--tau_main", type=float, default=0.5)
    p.add_argument("--cap", type=float, default=0.2)
    p.add_argument("--caps", default="0.05,0.1,0.2", help="cap sweep at tau_main (dose response)")
    p.add_argument("--rho", type=float, default=0.5, help="soft-repair weight for masked edges")
    p.add_argument("--budget", type=float, default=0.5, help="gate: max top-k Jaccard distance")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--iso_users", type=int, default=0, help="per-user (isolated) evaluation on a seeded random sample of this many repaired users; 0 = all")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    cfg = load_config(a.config)
    tag = f"arxiv_noise{a.noise}" if a.noise_type == "uniform" else f"arxiv_pop{a.noise}"
    t0 = time.time()
    out, model, data, evaluator, device = run_single(
        cfg, a.model, seed=a.seed, noise_ratio=a.noise, epochs=a.epochs, tag=tag,
        device_spec=a.device, K_attn=a.K_attn, return_model=True, noise_type=a.noise_type)
    model.eval()
    name = f"{cfg['dataset']['name']}__{a.model}__{a.noise_type}{a.noise}__seed{a.seed}"

    # ---- checkpoint (parameters + the exact neighbour lists used in training)
    ck = REPO_ROOT / "results" / "ckpt"; ck.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "neighbors": model.neighbors.cpu(),
                "valid_mask": model.valid_mask.cpu(), "args": vars(a), "best_metrics": out},
               ck / f"{name}.pt")

    # ---- diagnose: edge table (user side, last layer)
    t = edge_table(model, data, device)
    lab = t.label
    rec = {"dataset": cfg["dataset"]["name"], "model": a.model, "noise": a.noise, "noise_type": a.noise_type,
           "seed": a.seed, "K_attn": a.K_attn, "tau_main": a.tau_main, "cap": a.cap, "rho": a.rho,
           "budget": a.budget, "gamma": [float(g) for g in model.gamma.all_values().detach().cpu()],
           "train_best": {k: v for k, v in out.items() if "@" in k},
           "edge_coverage": float(len(t) / data.n_train), "n_edges_scored": int(len(t)),
           "injected_fraction": float((lab == 0).mean())}
    if t.has_noise:
        rec.update(reliability_summary(t))

    # ---- baseline slates / metrics (graph as trained, parameters frozen)
    Ub, Ib = propagate_on_graph(model, data, device, {})
    top0, rec0, ndcg0 = per_user_eval(Ub, Ib, data, a.k)
    rec["eval_no_repair"] = {"recall": nanmean(rec0), "ndcg": nanmean(ndcg0)}

    # ---- act: policies -> per-user masked sets
    by_user = t.by_user()
    n_inj_total = int((lab == 0).sum())
    policies = build_policies(t, by_user, taus=[float(x) for x in a.taus.split(",")],
                              caps=[float(x) for x in a.caps.split(",")], tau_main=a.tau_main,
                              cap=a.cap, rng=np.random.default_rng(a.seed + 99))
    main_name = f"pi0_tau{a.tau_main}"
    S_main = policies[main_name]

    # ---- verify: joint repair (every user's edits at once)
    results = {}
    for pname, S in policies.items():
        Uq, Iq = propagate_on_graph(model, data, device, edge_weights(t, S))
        top1, rec1, ndcg1 = per_user_eval(Uq, Iq, data, a.k)
        masked = [j for pick in S.values() for j in pick]
        res = {"n_masked": len(masked), "n_users_repaired": len(S),
               "recall": nanmean(rec1), "ndcg": nanmean(ndcg1),
               "delta_ndcg": nanmean(ndcg1) - nanmean(ndcg0)}
        if t.has_noise and masked:
            res["precision"] = float((lab[masked] == 0).mean())
            res["recall_injected"] = float((lab[masked] == 0).sum() / max(n_inj_total, 1))
        results[pname] = res
    # soft repair of the main policy
    soft = edge_weights(t, S_main, a.rho)
    Uq, Iq = propagate_on_graph(model, data, device, soft)
    top1, rec1, ndcg1 = per_user_eval(Uq, Iq, data, a.k)
    results["soft_main"] = {"n_masked": len(soft), "recall": nanmean(rec1), "ndcg": nanmean(ndcg1),
                            "delta_ndcg": nanmean(ndcg1) - nanmean(ndcg0)}
    rec["policies"] = results

    # ---- verify: each user repaired in isolation (only u's own edges change)
    t_iso = time.time()
    S_iso, S_rand_iso = S_main, policies["ctrl_random"]
    if a.iso_users and len(S_main) > a.iso_users:
        pick = set(np.random.default_rng(a.seed + 7).choice(sorted(S_main.keys()), size=a.iso_users, replace=False).tolist())
        S_iso = {u: v for u, v in S_main.items() if u in pick}
        S_rand_iso = {u: v for u, v in policies["ctrl_random"].items() if u in pick}
    topI, recI, ndI = isolated_user_eval(model, data, device, S_iso, t, a.k, top0)
    topR, recR, ndR = isolated_user_eval(model, data, device, S_rand_iso, t, a.k, top0)
    repaired = np.array(sorted(S_iso.keys()), dtype=int)
    rec["iso_users_evaluated"] = int(len(repaired))
    ndI_f, ndR_f = fill_unrepaired(ndI, ndcg0), fill_unrepaired(ndR, ndcg0)
    ev, evR = np.isfinite(ndI), np.isfinite(ndR)
    rec["isolated_per_repaired_user"] = {"pi0_delta": nanmean(ndI[ev] - ndcg0[ev]) if ev.any() else float("nan"),
                                         "random_delta": nanmean(ndR[evR] - ndcg0[evR]) if evR.any() else float("nan")}
    rec["isolated"] = {
        "pi0": {"ndcg": nanmean(ndI_f), "delta_ndcg": nanmean(ndI_f) - nanmean(ndcg0)},
        "random": {"ndcg": nanmean(ndR_f), "delta_ndcg": nanmean(ndR_f) - nanmean(ndcg0)},
        "seconds": round(time.time() - t_iso, 1)}
    jd = jaccard_dist(top0[repaired], topI[repaired]) if len(repaired) else np.array([])
    jr = jaccard_dist(top0[repaired], topR[repaired]) if len(repaired) else np.array([])
    # gate: revert users whose slate moved more than the budget
    accepted = gate(repaired, jd, a.budget)
    nd_gate = ndcg0.copy(); nd_gate[accepted] = ndI_f[accepted]
    rec["gate"] = {"n_candidates": int(len(repaired)), "acceptance_rate": float(len(accepted) / max(len(repaired), 1)),
                   "mean_jaccard_change": nanmean(jd), "ndcg": nanmean(nd_gate), "delta_ndcg": nanmean(nd_gate) - nanmean(ndcg0)}
    # targeted: riskiest half of candidate users
    chosen = target_users(repaired, t.r, by_user, a.tau_main)
    nd_t = ndcg0.copy(); nd_t[chosen] = ndI_f[chosen]
    masked_t = [j for u in chosen for j in S_main[u]]
    rec["targeted"] = {"n_users": int(len(chosen)), "n_masked": len(masked_t), "ndcg": nanmean(nd_t),
                       "delta_ndcg": nanmean(nd_t) - nanmean(ndcg0),
                       "precision": float((lab[masked_t] == 0).mean()) if (t.has_noise and masked_t) else float("nan")}
    # degree groups (original degree terciles), paired per user, isolated repair
    rec["groups"] = degree_groups(original_degree(data), ndcg0, ndI_f)
    # explain: slate change from the cited edges vs a random same-size set of the same user's edges
    rec["fidelity"] = fidelity(jd, jr, len(repaired))
    rec["wall_s"] = round(time.time() - t0, 1)

    path = REPO_ROOT / cfg["experiment"]["log_dir"] / f"{cfg['dataset']['name']}__graphtrust.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    brief = {k: v for k, v in rec.items() if k in ("auroc_r", "auroc_rel", "auroc_dot", "auroc_cos", "eval_no_repair", "isolated", "gate", "targeted", "fidelity")}
    brief["policies"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in results.items()}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
