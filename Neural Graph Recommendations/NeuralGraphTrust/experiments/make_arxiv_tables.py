"""Build paper/arxiv/results_tables.tex from the run logs. Every number traces to a log line.

Inputs (results/logs/):
  ml100k__runs.jsonl        accuracy rows written by src.train (tags arxiv_noise{N}, arxiv_pop{N})
  ml100k__graphtrust.jsonl  rows written by experiments.graphtrust_eval
Duplicates (same model / setting / seed, e.g. a re-run) keep the last row.
"""
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
OUT = ROOT / "paper" / "arxiv" / "results_tables.tex"
NAMES = {"ngcf": "NGCF", "attentive_ngcf": "Attention-NGCF (Stage 1)", "lightgcn": "LightGCN", "simgcl": "SimGCL",
         "attentive_lightgcn": "Attentive-LightGCN", "attentive_simgcl": "Attentive-SimGCL"}
SEEDS = {2024, 2025, 2026}


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def dedupe(rows, key):
    d = OrderedDict()
    for r in rows:
        d[key(r)] = r
    return list(d.values())


def fin(xs):
    return np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)


def ms(xs, f="{:.3f}"):
    """mean with a small +/- s.d.; compact so tables fit the text width."""
    xs = fin(xs)
    if len(xs) == 0:
        return "--"
    sd = xs.std(ddof=1) if len(xs) > 1 else 0.0
    mean = f.format(xs.mean())
    return mean + r"{\scriptsize$\,\pm$" + f.replace("+", "").format(sd).lstrip("0") + "}"


def bold(cell):
    return r"\textbf{" + cell + "}"


def m(xs, f="{:.3f}"):
    xs = fin(xs)
    return f.format(xs.mean()) if len(xs) else "--"


def wins(deltas):
    d = fin(deltas)
    return f"{int((d > 0).sum())}/{len(d)}"


runs = dedupe([r for r in load(LOG / "ml100k__runs.jsonl")
               if str(r.get("tag", "")).startswith(("arxiv_noise", "arxiv_pop")) and r.get("seed") in SEEDS
               and "recall@20" in r],
              key=lambda r: (r["model"], r["tag"], r["seed"]))
gt = dedupe([r for r in load(LOG / "ml100k__graphtrust.jsonl") if r.get("seed") in SEEDS],
            key=lambda r: (r["model"], r["noise_type"], r["noise"], r["seed"]))


def acc(model, tag):
    rs = [r for r in runs if r["model"] == model and r["tag"] == tag]
    return [r["recall@20"] for r in rs], [r["ndcg@20"] for r in rs]


def G(model, ntype, noise):
    return [r for r in gt if r["model"] == model and r["noise_type"] == ntype and abs(r["noise"] - noise) < 1e-9]


L = [r"\paragraph{Results.} All numbers below are generated from the run logs by \texttt{experiments/make\_arxiv\_tables.py}; cells are mean $\pm$ s.d. over the seeds, with $n$ (number of seeds) stated per row. $\Delta$ columns are paired within seed (the same trained model before and after repair) and ``wins'' counts seeds with $\Delta>0$."]

# ---------------------------------------------------------------- Table A/A2: accuracy (NDCG, Recall)
SETTINGS = [("arxiv_noise0.0", "Clean"), ("arxiv_noise0.1", "10\\%"), ("arxiv_noise0.2", "20\\%"), ("arxiv_pop0.1", "10\\% pop.")]
MODELS_ACC = ["ngcf", "attentive_ngcf", "lightgcn", "attentive_lightgcn", "simgcl", "attentive_simgcl"]
for metric, idx, lab, cap in [("NDCG@20", 1, "tab:acc", "Accuracy under injected noise on MovieLens-100K: NDCG@20, best test evaluation during training, mean $\\pm$ s.d. over seeds ($n$ seeds). Uniform noise at 10\\% and 20\\% of training edges; ``pop.'' injects 10\\% with probability proportional to item degree. The first two rows replicate Stage~1 under this protocol, with masked attention. Right: relative drop from the clean graph, $100(\\mathrm{clean}-\\mathrm{noisy})/\\mathrm{clean}$, paired within seed (lower is more robust; negative means no loss). Best value per column in bold."),
                              ("Recall@20", 0, "tab:acc_recall", "As Table~\\ref{tab:acc}, Recall@20.")]:
    drops = idx == 1  # main NDCG table also reports the paired relative drop from the clean graph
    L.append(r"\begin{table}[t]\centering\small" + "\n\\caption{" + cap + "}\n\\label{" + lab + "}\n"
             r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lccccc" + ("ccc" if drops else "") + r"}\toprule" + "\n"
             + (r" & & \multicolumn{4}{c}{NDCG@20 $\uparrow$} & \multicolumn{3}{c}{drop from clean (\%) $\downarrow$}\\\cmidrule(lr){3-6}\cmidrule(lr){7-9}" + "\n" if drops else "")
             + "Model & $n$ & " + " & ".join(x[1] for x in SETTINGS) + ((" & " + " & ".join(x[1] for x in SETTINGS[1:])) if drops else "") + r"\\\midrule")
    def drop(mo, tag):
        """paired within seed: 100*(clean - noisy)/clean, mean over seeds"""
        c = {r["seed"]: r["ndcg@20"] for r in runs if r["model"] == mo and r["tag"] == SETTINGS[0][0]}
        n = {r["seed"]: r["ndcg@20"] for r in runs if r["model"] == mo and r["tag"] == tag}
        return [100 * (c[k] - n[k]) / c[k] for k in sorted(set(c) & set(n))]
    best_drop = {tag: min(np.mean(drop(mm, tag)) for mm in MODELS_ACC if drop(mm, tag)) for tag, _ in SETTINGS[1:]} if drops else {}
    best = {tag: max((np.mean(acc(mm, tag)[idx]) for mm in MODELS_ACC if acc(mm, tag)[idx]), default=None) for tag, _ in SETTINGS}
    for mo in MODELS_ACC:
        cells, ns = [], []
        for tag, _ in SETTINGS:
            v = acc(mo, tag)[idx]
            ns.append(len(v))
            c = ms(v) if v else "--"
            if v and best[tag] is not None and abs(np.mean(v) - best[tag]) < 1e-12:
                c = bold(c)
            cells.append(c)
        nn = [x for x in ns if x]
        if drops:
            for tag, _ in SETTINGS[1:]:
                dv = drop(mo, tag)
                c = ms(dv, "{:.1f}") if dv else "--"
                if dv and abs(np.mean(dv) - best_drop[tag]) < 1e-12:
                    c = bold(c)
                cells.append(c)
        L.append(f"{NAMES[mo]} & {min(nn) if nn else 0}{'' if len(set(nn)) <= 1 else '--' + str(max(nn))} & " + " & ".join(cells) + r"\\")
        if mo in ("attentive_ngcf", "attentive_lightgcn"):
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")

# ---------------------------------------------------------------- Table B: reliability
L.append(r"""\begin{table}[t]\centering\small
\caption{Edge reliability on the trained attentive models (end-of-training state): AUROC for separating original from injected training edges (0.5 = chance) using the degree-normalized attention $r$ (Eq.~\eqref{eq:rel}), the LightGCN-relative score $\alpha_{ui}\sqrt{n_u n_i}$ (Prop.~\ref{prop:edge}), and the same model's inner product and cosine; $\bar r$ on original / injected edges; learned gate $\gamma$. Mean $\pm$ s.d. over $n$ seeds.}
\label{tab:auroc}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llccccccc}\toprule
Model & Noise & $n$ & $r$ & rel. & dot & cos & $\bar r$ orig./inj. & $\gamma$\\\midrule""")
for mo in ["attentive_lightgcn", "attentive_simgcl"]:
    for nt, nz, lab in [("uniform", 0.1, "10\\%"), ("uniform", 0.2, "20\\%"), ("pop", 0.1, "10\\% pop.")]:
        rs = G(mo, nt, nz)
        if not rs:
            continue
        L.append(f"{NAMES[mo]} & {lab} & {len(rs)} & {ms([r.get('auroc_r') for r in rs])} & {ms([r.get('auroc_rel') for r in rs])} & "
                 f"{ms([r.get('auroc_dot') for r in rs])} & {ms([r.get('auroc_cos') for r in rs])} & "
                 f"{m([r.get('mean_r_original') for r in rs],'{:.2f}')}\\,/\\,{m([r.get('mean_r_injected') for r in rs],'{:.2f}')} & {m([r['gamma'][0] for r in rs],'{:.2f}')}\\\\")
L.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")

# ---------------------------------------------------------------- Table C1: mask precision
L.append(r"""\begin{table}[t]\centering\small
\caption{Which edges NeuralGraphTrust masks ($\pi_0$, $\tau{=}0.5$, cap $c{=}0.2$; end-of-training state). Precision = fraction of masked edges that were injected; ``chance'' is the injected share of all scored edges. Controls mask the same number of edges per user as $\pi_0$ (the oracle up to that number when fewer injected edges are attended). Mean over $n$ seeds.}
\label{tab:repair}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llccccccc}\toprule
Model & Noise & $n$ & masked & chance & $\pi_0$ & $\pi_{\mathrm{rel}}$ & dot & random\\\midrule""")
REPAIR_SETTINGS = [("uniform", 0.0, "clean"), ("uniform", 0.1, "10\\%"), ("uniform", 0.2, "20\\%"), ("pop", 0.1, "10\\% pop.")]
for mo in ["attentive_lightgcn", "attentive_simgcl"]:
    for nt, nz, lab in REPAIR_SETTINGS:
        rs = G(mo, nt, nz)
        if not rs:
            continue
        P = lambda k: m([r["policies"][k].get("precision", np.nan) for r in rs], "{:.2f}")
        L.append(f"{NAMES[mo]} & {lab} & {len(rs)} & {m([r['policies']['pi0_tau0.5']['n_masked'] for r in rs],'{:.0f}')} & {m([r['injected_fraction'] for r in rs],'{:.2f}')} & "
                 f"{P('pi0_tau0.5')} & {P('pi_rel')} & {P('ctrl_dot')} & {P('ctrl_random')}\\\\")
L.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")

# ---------------------------------------------------------------- Table C2: ranking effect
L.append(r"""\begin{table}[t]\centering\small
\caption{Effect of NeuralGraphTrust repair on NDCG@20 (no retraining; end-of-training state). ``None'' is the unrepaired model. ``Joint'' repairs all users in one pass; ``per-user'' repairs each user in isolation, so only that user's edges change. $\Delta$ is paired within seed; [wins] counts seeds with $\Delta>0$. Clean rows measure harm when there is nothing to repair.}
\label{tab:repair_ndcg}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llccccc}\toprule
Model & Noise & none & oracle (joint) & $\Delta$ $\pi_0$ joint & $\Delta$ $\pi_0$ per-user & $\Delta$ random per-user\\\midrule""")
for mo in ["attentive_lightgcn", "attentive_simgcl"]:
    for nt, nz, lab in REPAIR_SETTINGS:
        rs = G(mo, nt, nz)
        if not rs:
            continue
        none = [r["eval_no_repair"]["ndcg"] for r in rs]
        dj = [r["policies"]["pi0_tau0.5"]["ndcg"] - r["eval_no_repair"]["ndcg"] for r in rs]
        di = [r["isolated"]["pi0"]["delta_ndcg"] for r in rs]
        dr = [r["isolated"]["random"]["delta_ndcg"] for r in rs]
        L.append(f"{NAMES[mo]} & {lab} & {ms(none)} & {ms([r['policies']['ctrl_oracle']['ndcg'] for r in rs])} & "
                 f"{ms(dj,'{:+.4f}')} [{wins(dj)}] & {ms(di,'{:+.4f}')} [{wins(di)}] & {ms(dr,'{:+.4f}')} [{wins(dr)}]\\\\")
L.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")

# ---------------------------------------------------------------- Table D: components
L.append(r"""\begin{table}[t]\centering\small
\caption{NeuralGraphTrust components under uniform noise (end-of-training state, mean over seeds). Cap sweep at $\tau{=}0.5$ and soft repair ($\omega{=}0.5$) are joint passes; gate ($b{=}0.5$ on top-20 Jaccard), targeting (riskier half of candidate users) and fidelity are per-user. Fidelity: mean top-20 Jaccard change from the cited edges vs.\ a random set of the same size.}
\label{tab:components}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llcccc}\toprule
Model & Noise & cap $c$: masked / prec. / $\Delta$ & soft $\Delta$ & gate acc. / $\Delta$ & targeted prec. / $\Delta$\\\midrule""")
for mo in ["attentive_lightgcn", "attentive_simgcl"]:
    for nz, lab in [(0.1, "10\\%"), (0.2, "20\\%")]:
        rs = G(mo, "uniform", nz)
        if not rs:
            continue
        caps = []
        for c in ["0.05", "0.1", "0.2"]:
            k = f"pi0_cap{float(c)}"
            dc = [r["policies"][k]["ndcg"] - r["eval_no_repair"]["ndcg"] for r in rs]
            caps.append(f"{c}: {m([r['policies'][k]['n_masked'] for r in rs],'{:.0f}')} / {m([r['policies'][k].get('precision', np.nan) for r in rs],'{:.2f}')} / {m(dc,'{:+.4f}')}")
        ds = [r["policies"]["soft_main"]["ndcg"] - r["eval_no_repair"]["ndcg"] for r in rs]
        L.append(f"{NAMES[mo]} & {lab} & \\begin{{tabular}}[t]{{@{{}}l@{{}}}}{caps[0]}\\\\{caps[1]}\\\\{caps[2]}\\end{{tabular}} & {m(ds,'{:+.4f}')} & "
                 f"{m([r['gate']['acceptance_rate'] for r in rs],'{:.2f}')} / {m([r['gate']['delta_ndcg'] for r in rs],'{:+.4f}')} & "
                 f"{m([r['targeted']['precision'] for r in rs],'{:.2f}')} / {m([r['targeted']['delta_ndcg'] for r in rs],'{:+.4f}')}\\\\[2pt]")
L.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")

# ---------------------------------------------------------------- Table E: fidelity + degree groups
L.append(r"""\begin{table}[t]\centering\small
\caption{Per-user repair by original user degree tercile (paired $\Delta$NDCG@20 of $\pi_0$; group sizes in parentheses) and explanation fidelity (top-20 Jaccard change from the cited edges vs.\ a random same-size set). Uniform noise; end-of-training state; mean over seeds.}
\label{tab:groups}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llcccc}\toprule
Model & Noise & sparse & medium & dense & fidelity cited / random\\\midrule""")
for mo in ["attentive_lightgcn", "attentive_simgcl"]:
    for nz, lab in [(0.1, "10\\%"), (0.2, "20\\%")]:
        rs = G(mo, "uniform", nz)
        if not rs:
            continue
        cells = [f"{m([r['groups'][g]['delta_ndcg'] for r in rs],'{:+.4f}')} ({rs[0]['groups'][g]['n']})" for g in ("sparse", "medium", "dense")]
        L.append(f"{NAMES[mo]} & {lab} & " + " & ".join(cells) + f" & {m([r['fidelity']['jaccard_cited'] for r in rs])} / {m([r['fidelity']['jaccard_random'] for r in rs])}\\\\")
L.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")


# ---------------------------------------------------------------- split: main text vs appendix
OUT_MAIN = ROOT / "paper" / "arxiv" / "results_main.tex"
OUT_APP = ROOT / "paper" / "arxiv" / "results_appendix.tex"
import re as _re
_full = "\n".join(L[1:])
_blocks = [b for b in _re.split(r"(?=\\begin\{table\})", _full) if b.strip()]
intro_par, ndcg_table, rest = L[0], _blocks[0], _blocks[1:]
M = [intro_par, ndcg_table]
mo = "attentive_lightgcn"
UNI = [("uniform", 0.1, "10\\%"), ("uniform", 0.2, "20\\%")]
M.append(r"""\begin{table}[t]\centering\small
\caption{Attention as an edge-reliability signal in Attentive-LightGCN under uniform noise (end-of-training state; mean $\pm$ s.d. over three seeds). $\bar r$: mean degree-normalized attention (Eq.~\eqref{eq:rel}) on original and injected training edges ($\bar r=1$ is uniform attention). AUROC for separating original from injected edges with $r$; learned gate $\gamma$. Each noise level is compared with fixed degree weights (LightGCN), for which $r\equiv1$ on every edge (Proposition~\ref{prop:uniform}) and the AUROC is 0.5; better value in bold. Ratio: mean $\bar r$ on original over mean $\bar r$ on injected edges. Full results, including other scores, popularity-biased noise and Attentive-SimGCL, are in Table~\ref{tab:auroc}.}
\label{tab:rel_main}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llccccc}\toprule
Noise & Edge weights & $\bar r$ original & $\bar r$ injected $\downarrow$ & ratio $\uparrow$ & AUROC($r$) $\uparrow$ & $\gamma$\\\midrule""")
for i, (nt, nz, lab) in enumerate(UNI):
    rs = G(mo, nt, nz)
    ratio = [np.mean([r['mean_r_original'] for r in rs]) / np.mean([r['mean_r_injected'] for r in rs])]
    if i:
        M.append(r"\midrule")
    M.append(f"\\multirow{{2}}{{*}}{{{lab}}} & fixed (LightGCN) & 1.00 & 1.00 & 1.0$\\times$ & 0.500 & --\\\\")
    M.append(f" & learned (Att.-LightGCN) & {ms([r['mean_r_original'] for r in rs],'{:.2f}')} & {bold(ms([r['mean_r_injected'] for r in rs],'{:.2f}'))} & "
             f"{bold(m(ratio,'{:.1f}') + '$\\times$')} & {bold(ms([r['auroc_r'] for r in rs]))} & {ms([r['gamma'][0] for r in rs],'{:.2f}')}\\\\")
M.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")
M.append(r"""\begin{table}[t]\centering\small
\caption{NeuralGraphTrust repair with $\pi_0$ ($\tau{=}0.5$, $c{=}0.2$) on Attentive-LightGCN under uniform noise, each user repaired in isolation, no retraining (three seeds). Precision: fraction of masked edges that were injected, against the chance level and against masking the same number of random edges. $\Delta$NDCG@20 is paired within seed; [wins] counts seeds with $\Delta>0$. Further variants, clean graphs and Attentive-SimGCL are in Tables~\ref{tab:repair}--\ref{tab:groups}.}
\label{tab:repair_main}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lccccc}\toprule
Noise & precision $\pi_0$ & chance & precision random & $\Delta$ $\pi_0$ [wins] & $\Delta$ random [wins]\\\midrule""")
for nt, nz, lab in UNI:
    rs = G(mo, nt, nz)
    di = [r["isolated"]["pi0"]["delta_ndcg"] for r in rs]; dr = [r["isolated"]["random"]["delta_ndcg"] for r in rs]
    pp = [r['policies']['pi0_tau0.5']['precision'] for r in rs]; pr = [r['policies']['ctrl_random']['precision'] for r in rs]
    c_pp, c_pr = ms(pp, '{:.2f}'), ms(pr, '{:.2f}')
    c_di, c_dr = ms(di, '{:+.4f}') + f' [{wins(di)}]', ms(dr, '{:+.4f}') + f' [{wins(dr)}]'
    if np.mean(pp) > np.mean(pr): c_pp = bold(c_pp)
    else: c_pr = bold(c_pr)
    if np.mean(di) > np.mean(dr): c_di = bold(c_di)
    else: c_dr = bold(c_dr)
    M.append(f"{lab} & {c_pp} & {m([r['injected_fraction'] for r in rs], '{:.2f}')} & {c_pr} & {c_di} & {c_dr}" + r"\\")
M.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")
# Stage-1 table: left = as reported in the RecSys 2025 LBR submission (its Table 3; NDCG@10, its own split);
# right = re-run under this paper's protocol (NDCG@20, mean over seeds, from the logs).
LBR = [("Side attributes", "frozen", 0.147, 0.143), ("Side attributes", "trained", 0.216, 0.216),
       ("Random", "frozen", 0.238, 0.223), ("Random", "trained", 0.231, 0.236)]
RERUN = [("arxiv_noise0.0", "clean"), ("arxiv_noise0.1", r"10\% noise"), ("arxiv_noise0.2", r"20\% noise"), ("arxiv_pop0.1", r"10\% pop.")]
S1 = [r"""\begin{table}[t]\centering\small
\caption{Stage~1 results. Left: as reported in the late-breaking submission and our repository~\cite{uppuluri2024repo}, NDCG@10 on MovieLens-100K with that submission's split (97{,}782 interactions; 80{,}808 train / 16{,}974 test) and 52-dimensional one-hot side attributes; the number of runs per cell was not recorded. Right: re-run under the protocol of Section~\ref{sec:exp}, NDCG@20, mean $\pm$ s.d. over three seeds, with masked attention and randomly initialized, trained embeddings. The two halves use different splits and metrics and are not directly comparable. Better value per row in bold.}
\label{tab:stage1}
\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llcc@{\hspace{14pt}}lcc}\toprule
\multicolumn{4}{c}{As reported (NDCG@10)} & \multicolumn{3}{c}{Re-run (NDCG@20)}\\\cmidrule(r{10pt}){1-4}\cmidrule(l){5-7}
Initialization & Embeddings & NGCF & + attention & Graph & NGCF & + attention\\\midrule"""]
for k in range(4):
    a_, e_, x_, y_ = LBR[k]
    lx, ly = f"{x_:.3f}", f"{y_:.3f}"
    if x_ > y_: lx = bold(lx)
    elif y_ > x_: ly = bold(ly)
    tag, glab = RERUN[k]
    vn, va = acc("ngcf", tag)[1], acc("attentive_ngcf", tag)[1]
    cn, ca = ms(vn), ms(va)
    if vn and va:
        if np.mean(vn) > np.mean(va): cn = bold(cn)
        else: ca = bold(ca)
    S1.append(f"{a_} & {e_} & {lx} & {ly} & {glab} & {cn} & {ca}" + r"\\")
S1.append(r"\bottomrule\end{tabular}\end{adjustbox}\end{table}")
(ROOT / "paper" / "arxiv" / "results_stage1.tex").write_text("\n".join(S1) + "\n")
OUT_MAIN.write_text("\n".join(M) + "\n")
OUT_APP.write_text("\n".join(rest) + "\n")
OUT.write_text("\n".join(L) + "\n")
summary = {"n_runs_rows": len(runs), "n_graphtrust_rows": len(gt),
           "acc": {f"{r['model']}@{r['tag']}@{r['seed']}": {"recall20": r["recall@20"], "ndcg20": r["ndcg@20"]} for r in runs},
           "gt": {f"{r['model']}@{r['noise_type']}{r['noise']}@{r['seed']}":
                  {k: r.get(k) for k in ("auroc_r", "auroc_rel", "auroc_dot", "auroc_cos", "eval_no_repair", "gate", "targeted", "fidelity", "groups", "gamma")}
                  | {"isolated": r.get("isolated"), "pi0": r["policies"]["pi0_tau0.5"], "oracle": r["policies"]["ctrl_oracle"], "random": r["policies"]["ctrl_random"],
                     "rel": r["policies"]["pi_rel"], "soft": r["policies"]["soft_main"]} for r in gt}}
(ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "tables" / "arxiv_summary.json").write_text(json.dumps(summary, indent=1))
print(f"runs rows: {len(runs)}  graphtrust rows: {len(gt)}  -> {OUT}")
