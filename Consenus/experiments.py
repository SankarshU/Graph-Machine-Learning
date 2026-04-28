"""
ICML 2026 workshop submission - planned ablation experiments (E1, E2, E3).

E1: Baseline comparison. Bayesian posterior vs raw scaled variance vs
    ||Delta C||_F as detectors over a perturbation-strength sweep wt.

E2: Benign vs malicious structural change. Within-SCC weight bump
    (no SCC topology change) vs cross-SCC edge injection (Assumption 2
    violation). Compare posterior at matched ||Delta C||_F.

E3: SCC-aware vs global variance. At a fixed cross-SCC injection,
    compute global scaled variance and per-SCC variance, and show that
    the per-SCC view localizes the affected block.

Self-contained: re-implements just enough of the notebook's pipeline
(build_c_i, simulate, scaled-variance signal, Bayesian update). Outputs
go to ./figs/ and ./results.csv next to this script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup: 7 users, 7 directories, three SCCs ({0,1,2}, {3,4}, singletons {5},{6}).
# We simulate per-topic scalar opinion dynamics on each directory's logic
# matrix C_d (m x m, row-stochastic). The cross-directory influence W (n x n)
# is the same as in the notebook. For our detection ablations, the relevant
# transient is on the directories that get a structurally-anomalous C_d.
# ---------------------------------------------------------------------------

M = 7  # users
N = 7  # directories

# Block-structured directory similarity (taken from notebook cell 18)
W_raw = np.array([
    [0.28, 1/3, 1/3,   0,   0, 0.05,   0],
    [1/3,  1/3, 1/3,   0,   0,   0,    0],
    [1/3,  1/3, 1/3,   0,   0, 0.05,   0],
    [0.05, 0,   0,   0.45, 0.5, 0,     0],
    [0.05, 0,   0,   0.45, 0.5, 0,     0],
    [0.10, 0.05, 0,  0.10, 0.05, 0.65, 0.05],
    [0,    0,   0,    0,   0,   0.05, 0.95],
])
W = W_raw / W_raw.sum(axis=1, keepdims=True)


def row_normalize(M):
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return M / s


def build_baseline_C():
    """Baseline user-user logic matrix shared by all directories.
    SCCs: {0,1,2}, {3,4}, {5}, {6}. Row-stochastic after normalize.
    """
    C = np.zeros((M, M))
    # SCC1 fully connected
    for a in range(3):
        for b in range(3):
            C[a, b] = 1.0
    # SCC2 fully connected
    for a in (3, 4):
        for b in (3, 4):
            C[a, b] = 1.0
    # Singletons self-loops
    C[5, 5] = 1.0
    C[6, 6] = 1.0
    return row_normalize(C)


def build_C_with_cross_scc_edge(from_node, to_nodes, wt):
    """Add a cross-SCC edge from user `from_node` into each user in
    `to_nodes` with raw weight wt, then row-normalize. Asymmetric: not all
    members of the target SCC are compromised, which models a realistic
    single-agent hijack and creates within-SCC opinion divergence during
    the transient.
    """
    C = np.zeros((M, M))
    for a in range(3):
        for b in range(3):
            C[a, b] = 1.0
    for a in (3, 4):
        for b in (3, 4):
            C[a, b] = 1.0
    C[5, 5] = 1.0
    C[6, 6] = 1.0
    for u in to_nodes:
        C[u, from_node] = wt  # cross-SCC injection (raw, before renorm)
    return row_normalize(C)


def build_C_within_scc_bump(scc_user_a, scc_user_b, wt):
    """Benign change: bump weight on an EXISTING within-SCC edge before
    renormalization. SCC topology unchanged.
    """
    C = np.zeros((M, M))
    for a in range(3):
        for b in range(3):
            C[a, b] = 1.0
    for a in (3, 4):
        for b in (3, 4):
            C[a, b] = 1.0
    C[5, 5] = 1.0
    C[6, 6] = 1.0
    # bump existing edge weight (no SCC change)
    C[scc_user_a, scc_user_b] = wt
    return row_normalize(C)


def simulate_per_topic(C_list, x0, T):
    """
    C_list: list of N matrices (m x m), one per directory.
    x0: shape (m, n) initial opinions (rows=users, cols=directories).
    For each directory d, evolve x[:, d] under x_d(t+1) = C_list[d] x_d(t).
    The W matrix mixes across directories; here we approximate by per-topic
    dynamics (which is the Ye-et-al. closed-SCC, fixed-W limit and is the
    regime our detector targets).

    Returns history of shape (T+1, m, n).
    """
    x = x0.copy()
    hist = [x.copy()]
    for _ in range(T):
        x_new = np.zeros_like(x)
        # Mix across directories first via W (right-multiply along directory axis):
        # for each user u, mixed_u[d] = sum_e W[d, e] x[u, e]
        x_mixed = x @ W.T
        for d in range(N):
            x_new[:, d] = C_list[d] @ x_mixed[:, d]
        x = x_new
        hist.append(x.copy())
    return np.array(hist)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def scaled_variance(x, scale=500.0):
    """Mean over directories of cross-user variance of (scale*x)."""
    v = np.var(scale * x, axis=0)  # shape (N,)
    return float(np.mean(v))


def per_scc_variance(x, scale=500.0, sccs=((0, 1, 2), (3, 4), (5,), (6,))):
    """Return dict {scc_idx -> mean per-topic variance restricted to scc rows}."""
    out = {}
    for j, scc in enumerate(sccs):
        rows = list(scc)
        if len(rows) < 2:
            out[j] = 0.0
            continue
        v = np.var(scale * x[rows, :], axis=0)
        out[j] = float(np.mean(v))
    return out


def bayes_update(prior, likelihood, eps=0.005):
    return (likelihood * prior) / (
        likelihood * prior + (1 - likelihood) * (1 - prior) + eps
    )


def detector_signals(x_prev, x_now, scale=500.0, alpha=0.01):
    v_now = scaled_variance(x_now, scale=scale)
    v_prev = scaled_variance(x_prev, scale=scale)
    delta = max(v_now - v_prev, 0.0)
    likelihood = 1.0 - np.exp(-alpha * delta)
    return dict(v_now=v_now, v_prev=v_prev, delta_v=delta, likelihood=likelihood)


# ---------------------------------------------------------------------------
# E1: Baseline comparison (Bayesian vs raw variance vs Frobenius norm)
# ---------------------------------------------------------------------------

def _max_transient_dv(C_baseline_list, C_perturbed_list, x_steady,
                      T_track=40, scale=500.0, mode="scc"):
    """Run perturbed dynamics from steady state, returning the maximum
    single-step positive Delta v over the first T_track steps (transient).
    mode='scc' uses max over SCCs of per-SCC scaled variance (Algorithm 2
    with the SCC-aware refinement of Section 6.3); mode='global' uses the
    paper's global scaled variance.
    """
    hist = simulate_per_topic(C_perturbed_list, x_steady, T=T_track)
    if mode == "global":
        vs = [scaled_variance(hist[t], scale=scale) for t in range(T_track + 1)]
    else:
        vs = []
        for t in range(T_track + 1):
            pv = per_scc_variance(hist[t], scale=scale)
            vs.append(max(pv.values()))
    deltas = [max(vs[t] - vs[t - 1], 0.0) for t in range(1, T_track + 1)]
    return float(max(deltas)) if deltas else 0.0, vs


def run_E1(seed=0, T_warm=2000, T_track=20):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1, 1, size=(M, N))

    C_hat = build_baseline_C()
    base_C_list = [C_hat] * N
    hist_base = simulate_per_topic(base_C_list, x0, T=T_warm)
    x_steady = hist_base[-1]

    wts = [1, 2, 5, 10, 50, 100, 250, 500, 1000]
    rows = []
    alpha = 0.05  # likelihood sensitivity (tuned so wt=5 reaches mid-range)

    def transient_dv_sequence(C_list):
        hist = simulate_per_topic(C_list, x_steady, T=T_track)
        vs = []
        for t in range(T_track + 1):
            pv = per_scc_variance(hist[t])
            vs.append(max(pv.values()))
        return [max(vs[t] - vs[t - 1], 0.0) for t in range(1, T_track + 1)]

    for wt in wts:
        C_pert = build_C_with_cross_scc_edge(from_node=1, to_nodes=[3], wt=wt)
        C_list = [C_hat] * 3 + [C_pert] * 2 + [C_hat] * 2
        dvs = transient_dv_sequence(C_list)
        max_dv = float(max(dvs)) if dvs else 0.0

        # Static: single-shot Bayes on max evidence with fixed prior 0.1
        L_max = 1.0 - np.exp(-alpha * max_dv)
        post_static = float(bayes_update(0.1, L_max))

        # Online: sequential Bayes over T_track timesteps, prior 0.1
        pi_t = 0.1
        for dv in dvs:
            if dv < 1e-6:
                continue  # no-evidence step (does not shift posterior)
            L_t = 1.0 - np.exp(-alpha * dv)
            pi_t = float(bayes_update(pi_t, L_t))
        post_online = pi_t

        frob = float(np.linalg.norm(C_pert - C_hat, ord="fro"))

        rows.append(dict(
            wt=wt, max_dv=max_dv,
            posterior_static=post_static, posterior_online=post_online,
            frobenius=frob,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results_E1.csv"), index=False)

    # Plot all three on a single figure
    fig, ax = plt.subplots(figsize=(6, 4))
    # Normalize each to [0,1] for visual comparison
    def norm01(s):
        s = np.asarray(s, dtype=float)
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min())
        return s
    ax.semilogx(df["wt"], df["posterior_online"], "o-", label="Bayesian (online)")
    ax.semilogx(df["wt"], df["posterior_static"], "s--", label="Bayesian (static)")
    ax.semilogx(df["wt"], norm01(df["max_dv"]), "^:", label="Variance only (norm.)")
    ax.semilogx(df["wt"], norm01(df["frobenius"]), "d-.",
                label=r"$\|\Delta C\|_F$ (norm.)")
    ax.set_xlabel("Cross-SCC edge weight $w_t$ (log scale)")
    ax.set_ylabel("Detector score")
    ax.set_title("E1: Bayesian vs. baseline detectors")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "E1_baseline_comparison.png"), dpi=160)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# E2: Benign within-SCC bump vs malicious cross-SCC injection
# ---------------------------------------------------------------------------

def run_E2(seed=0, T_warm=2000, T_track=20):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1, 1, size=(M, N))
    C_hat = build_baseline_C()
    base_C_list = [C_hat] * N
    hist_base = simulate_per_topic(base_C_list, x0, T=T_warm)
    x_steady = hist_base[-1]

    wts = [1, 2, 5, 10, 50, 100]
    rows = []
    alpha = 0.05  # likelihood sensitivity (tuned so wt=5 reaches mid-range)

    def online_post(C_list):
        hist = simulate_per_topic(C_list, x_steady, T=T_track)
        vs = [max(per_scc_variance(hist[t]).values()) for t in range(T_track + 1)]
        dvs = [max(vs[t] - vs[t - 1], 0.0) for t in range(1, T_track + 1)]
        max_dv = float(max(dvs)) if dvs else 0.0
        pi_t = 0.1
        for dv in dvs:
            if dv < 1e-6:
                continue  # no-evidence step (does not shift posterior)
            L_t = 1.0 - np.exp(-alpha * dv)
            pi_t = float(bayes_update(pi_t, L_t))
        return max_dv, pi_t

    for wt in wts:
        # benign: bump existing within-SCC edge weight (Assumption 2 holds)
        C_benign = build_C_within_scc_bump(scc_user_a=3, scc_user_b=4, wt=wt)
        Cb = [C_hat] * 3 + [C_benign] * 2 + [C_hat] * 2
        dv_b, post_b = online_post(Cb)
        frob_b = float(np.linalg.norm(C_benign - C_hat, ord="fro"))

        # malicious: cross-SCC u_1 -> {u_3} (Assumption 2 violated)
        C_mal = build_C_with_cross_scc_edge(from_node=1, to_nodes=[3], wt=wt)
        Cm = [C_hat] * 3 + [C_mal] * 2 + [C_hat] * 2
        dv_m, post_m = online_post(Cm)
        frob_m = float(np.linalg.norm(C_mal - C_hat, ord="fro"))

        rows.append(dict(
            wt=wt,
            benign_dv=dv_b, benign_post=post_b, benign_frob=frob_b,
            malic_dv=dv_m, malic_post=post_m, malic_frob=frob_m,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results_E2.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(df["wt"], df["benign_post"], "o--",
                label="Benign within-SCC (Assumption 2 holds)")
    ax.semilogx(df["wt"], df["malic_post"], "s-",
                label="Malicious cross-SCC (Assumption 2 violated)")
    ax.set_xlabel("Perturbation strength $w_t$ (log scale)")
    ax.set_ylabel(r"Posterior $\pi_t$ (static prior)")
    ax.set_title("E2: Benign vs malicious structural change")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "E2_benign_vs_malicious.png"), dpi=160)
    plt.close(fig)

    # Posterior at matched ||DeltaC||_F
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["benign_frob"], df["benign_post"], "o--", label="Benign within-SCC")
    ax.plot(df["malic_frob"], df["malic_post"], "s-", label="Malicious cross-SCC")
    ax.set_xlabel(r"$\|\Delta C\|_F$")
    ax.set_ylabel(r"Posterior $\pi_t$")
    ax.set_title("E2: Posterior at matched $\\|\\Delta C\\|_F$")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "E2_matched_frobenius.png"), dpi=160)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# E3: SCC-aware vs global variance
# ---------------------------------------------------------------------------

def run_E3(seed=0, T_warm=5000, wt=10):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1, 1, size=(M, N))

    C_hat = build_baseline_C()
    hist_base = simulate_per_topic([C_hat] * N, x0, T=T_warm)
    x_baseline_final = hist_base[-1]

    C_pert = build_C_with_cross_scc_edge(from_node=1, to_nodes=[3], wt=wt)
    C_list = [C_hat] * 3 + [C_pert] * 2 + [C_hat] * 2

    # Run perturbation for shorter horizon to track transient
    T_track = 200
    hist = simulate_per_topic(C_list, x_baseline_final, T=T_track)

    # Global and per-SCC variance over time
    rows = []
    for t in range(T_track + 1):
        gv = scaled_variance(hist[t])
        pv = per_scc_variance(hist[t])
        rows.append(dict(t=t, global_var=gv,
                         scc1_var=pv[0], scc2_var=pv[1],
                         scc3_var=pv[2], scc4_var=pv[3]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results_E3.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["t"], df["global_var"], "k-", lw=2, label="Global variance")
    ax.plot(df["t"], df["scc1_var"], "b--", label="SCC$_1$ = {u$_0$,u$_1$,u$_2$}")
    ax.plot(df["t"], df["scc2_var"], "r-", label="SCC$_2$ = {u$_3$,u$_4$} (attacked)")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Time step (post-injection)")
    ax.set_ylabel("Cross-user variance (scaled, symlog)")
    ax.set_title(f"E3: SCC-aware variance localizes the attacked block ($w_t={wt}$)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "E3_scc_aware.png"), dpi=160)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running E1: baseline comparison...")
    df_e1 = run_E1()
    print(df_e1.to_string(index=False))
    print()

    print("Running E2: benign vs malicious...")
    df_e2 = run_E2()
    print(df_e2.to_string(index=False))
    print()

    print("Running E3: SCC-aware vs global variance...")
    df_e3 = run_E3()
    print(df_e3.head(10).to_string(index=False))
    print(f"... ({len(df_e3)} rows total)")
    print()
    print(f"Figures written to {FIGS}/")
    print(f"CSV results: results_E1.csv, results_E2.csv, results_E3.csv")
