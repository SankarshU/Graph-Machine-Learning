#!/usr/bin/env python
# coding: utf-8
"""Consensus-Bayesian anomaly experiment script with saved outputs.

This is the cleaned anomaly path. It saves theorem tables, sweep tables, and
Plotly figures so the experiment can be verified after execution.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from opinion_dynamics_common_clean_saved import (
    compute_convergence_indicators_verbose,
    compute_minimal_external_topics_cascade,
    ensure_dir,
    return_theorems,
    row_normalize,
    run_dependency_aware_simulation,
)


def build_anomaly_baseline() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    W_raw = np.array([
        [0.28, 1 / 3, 1 / 3, 0.0, 0.0, 0.05, 0.0],
        [1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0],
        [1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.05, 0.0],
        [0.05, 0.0, 0.0, 0.45, 0.5, 0.0, 0.0],
        [0.05, 0.0, 0.0, 0.45, 0.5, 0.0, 0.0],
        [0.1, 0.05, 0.0, 0.1, 0.05, 0.65, 0.05],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.95],
    ])
    W_anom = W_raw / W_raw.sum(axis=1, keepdims=True)

    num_users = 7
    C_anom = np.zeros((num_users, num_users))
    C_anom[0, 1] = C_anom[1, 0] = 1
    C_anom[0, 2] = C_anom[2, 0] = 1
    C_anom[1, 2] = C_anom[2, 1] = 1
    C_anom[0, 0] = C_anom[1, 1] = C_anom[2, 2] = 1 / 3
    C_anom[3, 4] = C_anom[4, 3] = 1 / 2
    C_anom[3, 3] = 1
    C_anom[4, 4] = 1
    C_anom[5, 5] = 1
    C_anom[6, 6] = 1

    C_hat_anom = row_normalize(C_anom)
    C_list_hat_anom = [C_hat_anom] * 7
    return W_anom, C_hat_anom, C_list_hat_anom


def build_c_i(num_users: int = 7, from_node: int | None = None, to_node: list[int] | None = None, wt: float = 5):
    C = np.zeros((num_users, num_users))
    C[0, 1] = C[1, 0] = 1
    C[0, 2] = C[2, 0] = 1
    C[1, 2] = C[2, 1] = 1
    C[0, 0] = C[1, 1] = C[2, 2] = 1 / 3
    C[3, 4] = C[4, 3] = 1
    C[3, 3] = C[4, 4] = 1 / 2
    C[5, 5] = 1
    C[6, 6] = 1
    if from_node is not None and to_node is not None:
        for node in to_node:
            C[node, from_node] = 1
    C_bar_1 = row_normalize(C)

    C = np.zeros((num_users, num_users))
    C[0, 1] = C[1, 0] = 1
    C[0, 2] = C[2, 0] = 1
    C[1, 2] = C[2, 1] = 1
    C[0, 0] = C[1, 1] = C[2, 2] = 1 / 3
    C[3, 4] = C[4, 3] = 1
    C[3, 3] = C[4, 4] = 1 / 2
    C[5, 5] = 1
    C[6, 6] = 1
    if from_node is not None and to_node is not None:
        for node in to_node:
            C[node, from_node] = wt
    C_bar_2 = row_normalize(C)
    return C_bar_1, C_bar_2


def compute_anomaly_score(x_prev: np.ndarray, x_now: np.ndarray, prior: float = 0.1, scale_factor: float = 500, alpha: float = 0.01, verbose: bool = True) -> dict:
    x_prev = np.array(x_prev)
    x_now = np.array(x_now)
    if x_prev.ndim == 1:
        x_prev = x_prev.reshape(-1, 1)
    if x_now.ndim == 1:
        x_now = x_now.reshape(-1, 1)

    topic_variances = np.var(x_now * scale_factor, axis=0)
    scaled_variance = np.mean(topic_variances)
    topic_variances_prev = np.var(x_prev * scale_factor, axis=0)
    scaled_variance_prev = np.mean(topic_variances_prev)
    delta_variance = max(scaled_variance - scaled_variance_prev, 0)
    likelihood = 1 - np.exp(-alpha * delta_variance)
    posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior) + 0.005)
    info = {
        "Cur_topic_variances": topic_variances.tolist(),
        "Cur_scaled_variance": float(scaled_variance),
        "Prev_topic_variances": topic_variances_prev.tolist(),
        "Prev_scaled_variance": float(scaled_variance_prev),
        "Drift": None,
        "Delta_variance_times_drift": None,
        "Likelihood_exp": float(likelihood),
        "posterior": float(posterior),
    }
    if verbose:
        print(info)
    return info


def run_anomaly_block_simulation(cdf: pd.DataFrame, C_list: list[np.ndarray], W: np.ndarray, T: int = 5000):
    return run_dependency_aware_simulation(
        cdf, C_list, W, T=T, theorem2_mode="full", init_mode="anomaly", verbose=True, return_simple=True
    )


def save_plotly_figure(fig: go.Figure, html_path: Path):
    ensure_dir(html_path.parent)
    fig.write_html(str(html_path))
    try:
        png_path = html_path.with_suffix(".png")
        fig.write_image(str(png_path))
    except Exception:
        pass


def make_static_plot(df_static: pd.DataFrame, save_dir: Path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_static["wt"], y=df_static["posterior"], mode="lines+markers", name="Posterior (Static Prior)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df_static["wt"], y=df_static["Likelihood_exp"], mode="lines+markers", name="Likelihood", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df_static["wt"], y=df_static["Cur_scaled_variance"], mode="lines+markers", name="Current Scaled Variance", yaxis="y2", line=dict(dash="dot")))
    fig.update_layout(
        title="Anomaly Probability vs Abnormal Access Weight (Static Prior)",
        xaxis=dict(title="Relative access rate of abnormal user within the connected component", type="log", tickvals=[1, 2, 5, 10, 50, 100, 250, 500, 1000]),
        yaxis=dict(title="Posterior / Likelihood", side="left"),
        yaxis2=dict(title="Scaled variance", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )
    save_plotly_figure(fig, save_dir / "static_prior_plot.html")
    fig.show()


def make_online_plot(df_online: pd.DataFrame, save_dir: Path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_online["wt"], y=df_online["posterior"], mode="lines+markers", name="Posterior (Online Prior)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df_online["wt"], y=df_online["Likelihood_exp"], mode="lines+markers", name="Likelihood", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df_online["wt"], y=df_online["Cur_scaled_variance"], mode="lines+markers", name="Current Scaled Variance", yaxis="y2", line=dict(dash="dot")))
    fig.update_layout(
        title="Anomaly Probability vs Abnormal Access Weight (Online Prior)",
        xaxis=dict(title="Relative access rate of abnormal user within the connected component", type="log", tickvals=[2, 5, 10, 50, 100, 250, 500, 1000, 2000]),
        yaxis=dict(title="Posterior / Likelihood", side="left"),
        yaxis2=dict(title="Scaled variance", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )
    save_plotly_figure(fig, save_dir / "online_prior_plot.html")
    fig.show()


def make_comparison_plot(df_static: pd.DataFrame, df_online: pd.DataFrame, save_dir: Path):
    df_online_renamed = df_online.rename(columns={"Cur_scaled_variance": "Cur_scaled_variance_online", "posterior": "posterior_online"})[["wt", "Cur_scaled_variance_online", "posterior_online"]]
    df_static_renamed = df_static.rename(columns={"Cur_scaled_variance": "Cur_scaled_variance_static", "posterior": "posterior_static"})[["wt", "Cur_scaled_variance_static", "posterior_static"]]
    df_merged = pd.merge(df_static_renamed, df_online_renamed, on="wt")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_merged["wt"], y=df_merged["posterior_static"], name="Posterior (Static)", mode="lines+markers", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df_merged["wt"], y=df_merged["posterior_online"], name="Posterior (Online)", mode="lines+markers", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df_merged["wt"], y=df_merged["Cur_scaled_variance_online"], name="Scaled Variance (Online)", mode="lines+markers", yaxis="y2", line=dict(dash="dot")))
    fig.update_layout(
        title="Static vs Online Prior: Posterior Comparison",
        xaxis=dict(title="Relative access rate of abnormal user within the connected component", type="log"),
        yaxis=dict(title="Posterior", side="left"),
        yaxis2=dict(title="Scaled variance (online)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )
    save_plotly_figure(fig, save_dir / "static_vs_online_comparison.html")
    fig.show()


def main(output_dir: str = "anomaly_outputs"):
    outdir = ensure_dir(output_dir)
    W_anom, C_hat_anom, C_list_hat_anom = build_anomaly_baseline()

    cdf1_anom = compute_minimal_external_topics_cascade(return_theorems(C_hat_anom, verbose=True))
    cdf1_anom.to_csv(outdir / "baseline_theorem_table.csv", index=False)
    print(cdf1_anom)

    res1, res1s = run_anomaly_block_simulation(cdf1_anom, C_list_hat_anom, W_anom, T=5000)
    diag_base = compute_convergence_indicators_verbose(res1)
    pd.DataFrame([
        {"J": J, "J_converged": info["J_converged"], **{f"topic_{k}": v for k, v in info["Topic_convergence"].items()}}
        for J, info in diag_base.items()
    ]).to_csv(outdir / "baseline_convergence.csv", index=False)

    dft_static = {}
    perturbed_tables = []
    for wt in [1, 2, 5, 10, 50, 100, 250, 500, 1000]:
        C_bar_1_anom, C_bar_2_anom = build_c_i(num_users=7, from_node=1, to_node=[3, 4], wt=wt)
        cdf2_anom = compute_minimal_external_topics_cascade(return_theorems(C_bar_1_anom, verbose=True))
        cdf2_tmp = cdf2_anom.copy()
        cdf2_tmp["wt"] = wt
        perturbed_tables.append(cdf2_tmp)
        print(cdf2_anom)
        C_list_bar_anom = [C_bar_1_anom] * 3 + [C_bar_2_anom] * 2 + [C_bar_1_anom] * 2
        res2, res2s = run_anomaly_block_simulation(cdf2_anom, C_list_bar_anom, W_anom, T=5000)
        x_prev = res1s["J2"]["x_final"]
        x_now = res2s["J2"]["x_final"]
        print("-------wt", wt)
        dft_static[wt] = compute_anomaly_score(x_prev, x_now, prior=0.1, verbose=True)

    pd.concat(perturbed_tables, ignore_index=True).to_csv(outdir / "perturbed_theorem_tables_static.csv", index=False)
    df_static = pd.DataFrame.from_dict(dft_static, orient="index")
    df_static["wt"] = df_static.index
    df_static = df_static.drop(columns=["Drift", "Delta_variance_times_drift"])
    df_static.to_csv(outdir / "static_prior_table.csv", index=False)
    print(df_static)
    make_static_plot(df_static, outdir)

    dft_online = {}
    priorx = 0.10
    perturbed_tables_online = []
    for wt in [2, 5, 10, 50, 100, 250, 500, 1000, 2000]:
        C_bar_1_anom, C_bar_2_anom = build_c_i(num_users=7, from_node=1, to_node=[3, 4], wt=wt)
        cdf2_anom = compute_minimal_external_topics_cascade(return_theorems(C_bar_1_anom, verbose=True))
        cdf2_tmp = cdf2_anom.copy()
        cdf2_tmp["wt"] = wt
        perturbed_tables_online.append(cdf2_tmp)
        print(cdf2_anom)
        C_list_bar_anom = [C_bar_1_anom] * 3 + [C_bar_2_anom] * 2 + [C_bar_1_anom] * 2
        res2, res2s = run_anomaly_block_simulation(cdf2_anom, C_list_bar_anom, W_anom, T=5000)
        x_prev = res1s["J2"]["x_final"]
        x_now = res2s["J2"]["x_final"]
        print("-------wt", wt)
        debug_info = compute_anomaly_score(x_prev, x_now, prior=priorx, verbose=True)
        priorx = debug_info["posterior"]
        print("updated prior =", priorx)
        dft_online[wt] = debug_info

    pd.concat(perturbed_tables_online, ignore_index=True).to_csv(outdir / "perturbed_theorem_tables_online.csv", index=False)
    df_online = pd.DataFrame.from_dict(dft_online, orient="index")
    df_online["wt"] = df_online.index
    df_online = df_online.drop(columns=["Drift", "Delta_variance_times_drift"])
    df_online.to_csv(outdir / "online_prior_table.csv", index=False)
    print(df_online)
    make_online_plot(df_online, outdir)
    make_comparison_plot(df_static, df_online, outdir)

    print(f"Saved anomaly outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
