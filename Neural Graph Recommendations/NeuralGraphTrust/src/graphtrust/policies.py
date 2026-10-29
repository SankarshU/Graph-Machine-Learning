"""Act: controllers that choose, per user, which scored edges to mask.

A policy is a dict  user -> array of EdgeTable row indices  (users with nothing to mask are absent).
"""
from __future__ import annotations

import numpy as np


def threshold_policy(score: np.ndarray, by_user: dict, thresh: float, cap: float) -> dict:
    """Mask a user's edges with score < thresh, lowest first, at most floor(cap * #scored edges).
    pi0 is threshold_policy(r, ., tau, c); pi_rel is threshold_policy(rel, ., 1.0, c)."""
    S = {}
    for u, idx in by_user.items():
        cap_n = int(np.floor(cap * len(idx)))
        cand = idx[score[idx] < thresh]
        n = min(len(cand), cap_n)
        if n:
            S[u] = cand[np.argsort(score[cand])[:n]]
    return S


def matched_control(S_ref: dict, by_user: dict, kind: str, score: np.ndarray | None = None,
                    label: np.ndarray | None = None, rng: np.random.Generator | None = None) -> dict:
    """Mask as many edges per user as S_ref does, chosen by another rule:
    'lowest' (lowest `score`), 'random' (uniform, `rng`), or 'oracle' (injected edges first, by `label`)."""
    S = {}
    for u, pick in S_ref.items():
        idx = by_user[u]; n = len(pick)
        if kind == "lowest":
            S[u] = idx[np.argsort(score[idx])[:n]]
        elif kind == "random":
            S[u] = rng.choice(idx, size=n, replace=False)
        elif kind == "oracle":
            inj = idx[label[idx] == 0]
            S[u] = inj[:n] if len(inj) else np.array([], dtype=int)
    return S


def build_policies(t, by_user: dict, taus, caps, tau_main: float, cap: float, rng: np.random.Generator) -> dict:
    """All policies of the study, in reporting order: pi0 tau sweep, pi0 cap sweep, pi_rel and the
    controls matched to pi0(tau_main): lowest dot, lowest cos, random, oracle."""
    policies = {}
    for tau in taus:
        policies[f"pi0_tau{tau}"] = threshold_policy(t.r, by_user, tau, cap)
    for c in caps:
        policies[f"pi0_cap{c}"] = threshold_policy(t.r, by_user, tau_main, c)
    S_main = policies[f"pi0_tau{tau_main}"]
    policies["pi_rel"] = threshold_policy(t.rel, by_user, 1.0, cap)
    policies["ctrl_dot"] = matched_control(S_main, by_user, "lowest", score=t.dot)
    policies["ctrl_cos"] = matched_control(S_main, by_user, "lowest", score=t.cos)
    policies["ctrl_random"] = matched_control(S_main, by_user, "random", rng=rng)
    policies["ctrl_oracle"] = matched_control(S_main, by_user, "oracle", label=t.label)
    return policies


def edge_weights(t, S: dict, weight: float = 0.0) -> dict:
    """(user, item) -> weight for every masked edge of policy S (0 removes the edge; (0,1) is soft repair)."""
    return {(int(t.users[j]), int(t.items[j])): weight for pick in S.values() for j in pick}
