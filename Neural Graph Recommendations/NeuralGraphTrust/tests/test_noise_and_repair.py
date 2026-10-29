"""Noise injection, repair re-propagation and the per-user evaluation used by NeuralGraphTrust."""
import numpy as np
import pytest
import torch

from conftest import graph_inputs
from src.evaluate import Evaluator
from src.graphtrust import (edge_table, isolated_user_eval, matched_control, per_user_eval,
                            propagate_on_graph, threshold_policy)
from src.models import AttentiveLightGCN
from src.noise_injection import add_random_edges


@pytest.mark.parametrize("noise_type", ["uniform", "pop"])
def test_noise_injection_is_seeded_labelled_and_never_hits_observed_pairs(data, noise_type):
    a = add_random_edges(data, ratio=0.2, seed=7, noise_type=noise_type)
    b = add_random_edges(data, ratio=0.2, seed=7, noise_type=noise_type)
    c = add_random_edges(data, ratio=0.2, seed=8, noise_type=noise_type)
    assert np.array_equal(a.train_pairs, b.train_pairs) and np.array_equal(a.edge_labels, b.edge_labels)
    assert not np.array_equal(a.train_pairs, c.train_pairs)
    n_inj = int(round(0.2 * data.n_train))
    assert a.n_train == data.n_train + n_inj
    assert (a.edge_labels[: data.n_train] == 1).all() and (a.edge_labels[data.n_train:] == 0).all()
    assert np.array_equal(a.train_pairs[: data.n_train], data.train_pairs)
    observed = {(int(u), int(i)) for u, i in data.train_pairs}
    observed |= {(u, i) for u, its in data.test_user_items.items() for i in its}
    injected = [(int(u), int(i)) for u, i in a.train_pairs[data.n_train:]]
    assert not observed & set(injected)
    assert len(set(injected)) == n_inj                        # no duplicates
    assert a.user_item_csr.nnz == a.n_train


def _model(data, K=8, seed=0):
    norm_adj, pad = graph_inputs(data, K=K)
    torch.manual_seed(seed)
    m = AttentiveLightGCN(data.n_users, data.n_items, embed_dim=16, n_layers=3, norm_adj=norm_adj, neighbor_pad=pad)
    return m.eval()


def test_empty_repair_leaves_scores_unchanged(data):
    m = _model(data)
    with torch.no_grad():
        U0, I0 = m.propagate()
    dev = torch.device("cpu")
    for edits in ({}, {(int(u), int(i)): 1.0 for u, i in data.train_pairs[:5]}):
        U, I = propagate_on_graph(m, data, dev, edits)
        assert torch.allclose(U, U0, atol=1e-6) and torch.allclose(I, I0, atol=1e-6)
    # the model's own graph buffers are restored after a non-empty edit
    vm, adj = m.valid_mask.clone(), m.norm_adj
    propagate_on_graph(m, data, dev, {(int(data.train_pairs[0, 0]), int(data.train_pairs[0, 1])): 0.0})
    assert torch.equal(m.valid_mask, vm) and m.norm_adj is adj


def test_removed_edge_changes_only_the_graph(data):
    m = _model(data)
    dev = torch.device("cpu")
    u, i = (int(x) for x in data.train_pairs[0])
    U0, I0 = propagate_on_graph(m, data, dev, {})
    U1, I1 = propagate_on_graph(m, data, dev, {(u, i): 0.0})
    assert not torch.allclose(U0[u], U1[u])


def test_per_user_ndcg_matches_evaluator(data):
    m = _model(data)
    dev = torch.device("cpu")
    U, I = propagate_on_graph(m, data, dev, {})
    _, rec, nd = per_user_eval(U, I, data, k=20)
    ev = Evaluator(data, ks=[20]).evaluate(lambda: U, lambda: I, device=dev)
    assert abs(np.nanmean(nd) - ev.metrics["ndcg@20"]) < 1e-12
    assert abs(np.nanmean(rec) - ev.metrics["recall@20"]) < 1e-12


def test_policies_respect_threshold_cap_and_matched_counts(data):
    noisy = add_random_edges(data, ratio=0.3, seed=1)
    m = _model(noisy)
    t = edge_table(m, noisy, torch.device("cpu"))
    by_user = t.by_user()
    S = threshold_policy(t.r, by_user, thresh=1.0, cap=0.5)
    for u, pick in S.items():
        assert len(pick) <= int(np.floor(0.5 * len(by_user[u])))
        assert (t.r[pick] < 1.0).all() and set(pick) <= set(by_user[u])
        rest = np.setdiff1d(by_user[u][t.r[by_user[u]] < 1.0], pick)
        assert len(rest) == 0 or t.r[pick].max() <= t.r[rest].min()     # lowest first
    rng = np.random.default_rng(0)
    for kind, kw in (("lowest", {"score": t.dot}), ("random", {"rng": rng}), ("oracle", {"label": t.label})):
        C = matched_control(S, by_user, kind, **kw)
        assert set(C) == set(S)
        for u in S:
            assert len(C[u]) <= len(S[u]) and set(C[u]) <= set(by_user[u])
            if kind != "oracle":
                assert len(C[u]) == len(S[u])


def test_isolated_eval_only_changes_repaired_users(data):
    m = _model(data)
    dev = torch.device("cpu")
    t = edge_table(m, data, dev)
    U, I = propagate_on_graph(m, data, dev, {})
    top0, _, nd0 = per_user_eval(U, I, data, k=20)
    by_user = t.by_user()
    S = {u: by_user[u][:1] for u in list(by_user)[:3]}
    top, rec, nd = isolated_user_eval(m, data, dev, S, t, 20, top0)
    others = np.setdiff1d(np.arange(data.n_users), list(S))
    assert np.array_equal(top[others], top0[others]) and np.isnan(nd[others]).all()
    assert np.isfinite(nd[list(S)]).all()
