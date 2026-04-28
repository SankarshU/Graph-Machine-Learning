"""Common utilities for opinion-dynamics replication and anomaly experiments.

This module consolidates the helper code that was previously duplicated across
multiple notebooks/scripts. It is intentionally written in a notebook-friendly
style so it can be imported from scripts or pasted into Jupyter cells.

Conventions used throughout
--------------------------
- Topics are represented internally with 0-based indices.
- In the logical-dependency matrices C_i, an entry C_i[p, q] != 0 means
  "topic p depends on topic q".
- SCC labels J1, J2, ... are assigned by sorting SCCs by their minimum topic
  index, so the labeling stays stable and reproducible.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def row_normalize(C: np.ndarray) -> np.ndarray:
    """Row-normalize a matrix.

    Zero-sum rows are left unchanged by treating their row sum as 1.
    This keeps the function safe for partially sparse constructions.
    """
    C = np.array(C, dtype=float, copy=True)
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return C / row_sums


def step1_identify_sccs(C_i: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    G = nx.from_numpy_array(C_i, create_using=nx.DiGraph)
    sccs = list(nx.strongly_connected_components(G))
    sccs_sorted = sorted(sccs, key=lambda scc: min(scc))

    index_to_nodes = {f"J{j + 1}": sorted(list(scc)) for j, scc in enumerate(sccs_sorted)}
    rows = [{"J_index": J, "Topics": [u + 1 for u in users]} for J, users in index_to_nodes.items()]
    return pd.DataFrame(rows), index_to_nodes


def step2_assign_scc_status(C_i: np.ndarray, index_to_nodes: Dict[str, List[int]]) -> Dict[str, str]:
    status_map: Dict[str, str] = {}
    for J_index, nodes in index_to_nodes.items():
        nodes_set = set(nodes)
        outside_nodes = set(range(C_i.shape[1])) - nodes_set
        has_external_dep = any(C_i[p, q] != 0 for p in nodes_set for q in outside_nodes)
        status_map[J_index] = "Open" if has_external_dep else "Closed"
    return status_map


def step3_local_dependencies_corrected(C_i: np.ndarray, index_to_nodes: Dict[str, List[int]]) -> Dict[str, Dict[int, List[int]]]:
    local_dep_map: Dict[str, Dict[int, List[int]]] = {}
    for J_index, nodes in index_to_nodes.items():
        local_J_hat: Dict[int, List[int]] = {}
        for p in nodes:
            J_hat_p = [q + 1 for q in range(C_i.shape[1]) if q != p and C_i[p, q] != 0]
            local_J_hat[p + 1] = sorted(J_hat_p)
        local_dep_map[J_index] = local_J_hat
    return local_dep_map


def step4_external_dependencies(local_dep_map: Dict[str, Dict[int, List[int]]], index_to_nodes: Dict[str, List[int]]) -> Dict[str, List[int]]:
    ext_dep_map: Dict[str, List[int]] = {}
    for J_index, nodes in index_to_nodes.items():
        node_set = set(u + 1 for u in nodes)
        all_j_hat = local_dep_map[J_index]
        union_hat = set()
        for deps in all_j_hat.values():
            union_hat.update(deps)
        external = sorted(list(union_hat - node_set))
        ext_dep_map[J_index] = external
    return ext_dep_map


def step5_dependency_conditions_dfs(C_i: np.ndarray, index_to_nodes: Dict[str, List[int]]) -> Dict[str, List[str]]:
    """Build block dependency conditions.

    If C[p, q] != 0 then topic p depends on q, so in the condensation graph an edge
    J_a -> J_b means J_a depends on J_b. Therefore dependency conditions are built
    from descendants, not ancestors.
    """
    G = nx.from_numpy_array(C_i, create_using=nx.DiGraph)
    sccs = list(nx.strongly_connected_components(G))
    sccs_sorted = sorted(sccs, key=lambda scc: min(scc))
    condensation = nx.condensation(G, sccs_sorted)

    scc_idx_to_J = {idx: f"J{idx + 1}" for idx in range(len(sccs_sorted))}
    dependency_condition_map: Dict[str, List[str]] = {}
    for scc_idx in range(len(sccs_sorted)):
        Jj = scc_idx_to_J[scc_idx]
        deps = nx.descendants(condensation, scc_idx)
        if not deps:
            dependency_condition_map[Jj] = ["Evaluate First"]
        else:
            dependent_Js = [scc_idx_to_J[d] for d in sorted(deps)]
            dependency_condition_map[Jj] = sorted(dependent_Js, key=lambda x: int(x[1:]))
    return dependency_condition_map


def debug_annotate_theorem_table_from_steps(
    index_to_nodes: Dict[str, List[int]],
    local_dep_map: Dict[str, Dict[int, List[int]]],
    dependency_condition_map: Dict[str, List[str]],
    status_map: Dict[str, str],
    C_list: Optional[Sequence[np.ndarray]] = None,
    external_consensus_values: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    rows = []
    for J_name, topics_0b in index_to_nodes.items():
        topics_1b = [t + 1 for t in topics_0b]
        nodes_set = set(topics_0b)
        status = status_map[J_name]

        local_deps = local_dep_map.get(J_name, {})
        external_deps = sorted(set(q for p, deps in local_deps.items() for q in deps if (q - 1) not in nodes_set))
        dep_condition = dependency_condition_map.get(J_name, ["Evaluate First"])
        theorem = "Unknown"

        if status == "Closed":
            theorem = "Theorem 3" if len(topics_0b) == 1 else "Theorem 2"
        elif status == "Open":
            if len(topics_0b) == 1:
                p = topics_0b[0]
                deps = local_dep_map.get(J_name, {}).get(p + 1, [])
                if len(deps) == 1 and C_list is not None:
                    q = deps[0] - 1
                    lhs_vals = [1 - Ci[p, p] for Ci in C_list]
                    rhs_vals = [abs(Ci[p, q]) for Ci in C_list]
                    signs_ok = all(Ci[p, q] < 0 for Ci in C_list)
                    if all(np.isclose(lhs, rhs, atol=1e-3) for lhs, rhs in zip(lhs_vals, rhs_vals)) and signs_ok:
                        theorem = "Theorem 3 (via Corollary 2.1)"
                    elif external_consensus_values is not None and all(dep - 1 in external_consensus_values for dep in deps):
                        try:
                            Gamma_pp = np.diag([Ci[p, p] for Ci in C_list])
                            rhs_sum = np.zeros(len(C_list))
                            for ext_q in deps:
                                q_idx = ext_q - 1
                                Gamma_pq = np.diag([Ci[p, q_idx] for Ci in C_list])
                                rhs_sum += Gamma_pq @ np.ones(len(C_list)) * external_consensus_values[q_idx]
                            lhs_matrix = np.eye(len(C_list)) - Gamma_pp
                            kappa_vector = np.linalg.solve(lhs_matrix, rhs_sum)
                            theorem = "Theorem 3" if np.allclose(kappa_vector, kappa_vector[0], atol=1e-5) else "Theorem 4"
                        except Exception:
                            theorem = "Theorem 4"
                    else:
                        theorem = "Unknown (external consensus required)"
                else:
                    theorem = "Unknown (external consensus required)"
            else:
                theorem = "Theorem 4 / Eqn (12)"
                if C_list is not None:
                    all_rows = [[Ci[p] for p in topics_0b] for Ci in C_list]
                    reference = all_rows[0]
                    if all(all(np.allclose(ref, row, atol=1e-4) for ref, row in zip(reference, user_rows)) for user_rows in all_rows[1:]):
                        theorem += " (via Corollary 3.2)"

        rows.append({
            "J_index": J_name,
            "Topic_Indices_0b": topics_0b,
            "Topics": topics_1b,
            "Status": status,
            "Local_dependencies": local_deps,
            "External_dependencies": external_deps,
            "Dependency_Condition": dep_condition,
            "Apply_Theorem": theorem,
        })
    return pd.DataFrame(rows)


def return_theorems(C_i: np.ndarray, verbose: bool = False) -> pd.DataFrame:
    global users
    users = list(range(C_i.shape[0]))
    _, index_to_nodes_hat = step1_identify_sccs(C_i)
    status_map_hat = step2_assign_scc_status(C_i, index_to_nodes_hat)
    local_dep_map_hat = step3_local_dependencies_corrected(C_i, index_to_nodes_hat)
    ext_dep_map_hat = step4_external_dependencies(local_dep_map_hat, index_to_nodes_hat)
    dependency_condition_map_hat = step5_dependency_conditions_dfs(C_i, index_to_nodes_hat)
    if verbose:
        print("Status map for C_i:")
        print(status_map_hat)
        print("local_dep_map_hat:", local_dep_map_hat)
        print("ext_dep_map_hat:", ext_dep_map_hat)
        print("dependency_condition_map_hat:", dependency_condition_map_hat)
    return debug_annotate_theorem_table_from_steps(
        index_to_nodes=index_to_nodes_hat,
        local_dep_map=local_dep_map_hat,
        dependency_condition_map=dependency_condition_map_hat,
        status_map=status_map_hat,
        C_list=[C_i],
        external_consensus_values=None,
    )


def compute_minimal_external_topics_cascade(cdf: pd.DataFrame) -> pd.DataFrame:
    cdf = cdf.copy()
    J_block_to_topics = {row["J_index"]: row["Topic_Indices_0b"] for _, row in cdf.iterrows()}
    J_block_to_ext_topics = {row["J_index"]: sorted(set((t - 1) for t in row.get("External_dependencies", []))) for _, row in cdf.iterrows()}
    topic_to_J: Dict[int, str] = {}
    for J_block, topics in J_block_to_topics.items():
        for t in topics:
            topic_to_J[t] = J_block

    def gather_needed_topics(J_block: str, visited: Optional[set] = None) -> List[int]:
        if visited is None:
            visited = set()
        if J_block in visited:
            return []
        visited.add(J_block)
        needed: List[int] = []
        for t_ext in J_block_to_ext_topics.get(J_block, []):
            if t_ext not in topic_to_J:
                continue
            owner_J = topic_to_J[t_ext]
            if owner_J == J_block:
                continue
            needed.extend(J_block_to_topics.get(owner_J, []))
            needed.extend(gather_needed_topics(owner_J, visited))
        return needed

    cdf["Minimal_External_Topics"] = [sorted(set(gather_needed_topics(row["J_index"]))) for _, row in cdf.iterrows()]
    return cdf


def simulate_opinion_dynamics_singleton(topic_index: int, W: np.ndarray, C_list: Sequence[np.ndarray], x0: Optional[np.ndarray] = None, T: int = 20, return_history: bool = True, init_mode: str = "paper") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n = len(C_list)
    assert W.shape == (n, n), "W must be square with size equal to number of agents"
    if x0 is None:
        x = np.random.uniform(0.11, 0.112, size=n) if init_mode == "anomaly" else np.random.uniform(-1, 1, size=n)
    else:
        x = np.copy(x0)
    x_hist = [x.copy()] if return_history else None
    for _ in range(T):
        x_next = np.zeros_like(x)
        for i in range(n):
            c_pp = C_list[i][topic_index, topic_index]
            influence = sum(W[i, j] * x[j] for j in range(n))
            x_next[i] = c_pp * influence
        x = x_next
        if return_history:
            x_hist.append(x.copy())
    return (x, np.array(x_hist)) if return_history else (x, None)


def simulate_opinion_dynamics_corollary2(topic_index: int, dependency_indices: Sequence[int], W: np.ndarray, C_list: Sequence[np.ndarray], external_consensus_values: Dict[int, float], x0: Optional[np.ndarray] = None, T: int = 50, return_history: bool = True, init_mode: str = "paper") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n = len(C_list)
    if x0 is None:
        x = np.random.uniform(0.11, 0.112, size=n) if init_mode == "anomaly" else np.random.uniform(-1, 1, size=n)
    else:
        x = np.copy(x0)
    x_hist = [x.copy()] if return_history else None
    for _ in range(T):
        x_next = np.zeros_like(x)
        for i in range(n):
            c_pp = C_list[i][topic_index, topic_index]
            influence_sum = sum(W[i, j] * x[j] for j in range(n))
            logic_sum = sum(C_list[i][topic_index, q] * external_consensus_values[q] for q in dependency_indices)
            x_next[i] = c_pp * influence_sum + logic_sum
        x = x_next
        if return_history:
            x_hist.append(x.copy())
    return (x, np.array(x_hist)) if return_history else (x, None)


def simulate_theorem4_multitopic(topic_indices: Sequence[int], C_list: Sequence[np.ndarray], W: np.ndarray, external_consensus_values: Dict[int, float], x0: Optional[np.ndarray] = None, T: int = 50, return_history: bool = True, init_mode: str = "paper") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n = len(C_list)
    t = len(topic_indices)
    if x0 is None:
        x = np.random.uniform(0.11, 0.112, size=(n, t)) if init_mode == "anomaly" else np.random.uniform(-1, 1, size=(n, t))
    else:
        x = np.copy(x0)
    x_hist = [x.copy()] if return_history else None
    for _ in range(T):
        x_next = np.zeros_like(x)
        for i in range(n):
            for local_idx, p in enumerate(topic_indices):
                c_pp = C_list[i][p, p]
                influence = sum(W[i, j] * x[j, local_idx] for j in range(n))
                logic_term = 0.0
                for q in external_consensus_values:
                    if q not in topic_indices:
                        ext_val = external_consensus_values[q]
                        ext_vector = ext_val if isinstance(ext_val, np.ndarray) else np.ones(n) * ext_val
                        logic_term += C_list[i][p, q] * ext_vector[i]
                x_next[i, local_idx] = c_pp * influence + logic_term
        x = x_next
        if return_history:
            x_hist.append(x.copy())
    return (x, np.array(x_hist)) if return_history else (x, None)


def simulate_theorem2_multitopic(topic_indices: Sequence[int], C_list: Sequence[np.ndarray], W: np.ndarray, x0: Optional[np.ndarray] = None, T: int = 50, return_history: bool = True, init_mode: str = "paper", verify_closedness: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n = len(C_list)
    t = len(topic_indices)
    if verify_closedness:
        for C in C_list:
            for p in topic_indices:
                for q in range(C.shape[1]):
                    if C[p, q] != 0 and q not in topic_indices:
                        raise ValueError(f"Topic {p} depends on external topic {q}, violating Theorem 2.")
    if x0 is None:
        x = np.random.uniform(0.11, 0.112, size=(n, t)) if init_mode == "anomaly" else np.random.uniform(-1, 1, size=(n, t))
    else:
        x = np.copy(x0)
    x_hist = [x.copy()] if return_history else None
    for _ in range(T):
        x_next = np.zeros_like(x)
        for i in range(n):
            for local_idx, p in enumerate(topic_indices):
                c_pp = C_list[i][p, p]
                influence = sum(W[i, j] * x[j, local_idx] for j in range(n))
                logic_term = sum(C_list[i][p, q] * x[i, topic_indices.index(q)] for q in topic_indices if q != p)
                x_next[i, local_idx] = c_pp * influence + logic_term
        x = x_next
        if return_history:
            x_hist.append(x.copy())
    return (x, np.array(x_hist)) if return_history else (x, None)


def run_dependency_aware_simulation(cdf: pd.DataFrame, C_list: Sequence[np.ndarray], W: np.ndarray, T: int = 50, theorem2_mode: str = "sliced", init_mode: str = "paper", verbose: bool = True, return_simple: bool = False):
    cdfdct = cdf.to_dict("records")
    pending_J_blocks = {row["J_index"]: row for row in cdfdct}
    completed_J_blocks = set()
    external_consensus_values: Dict[int, np.ndarray] = {}
    results: Dict[str, dict] = {}
    results_simple: Dict[str, dict] = {}
    iteration = 0
    max_iters = 50
    while pending_J_blocks and iteration < max_iters:
        iteration += 1
        ready_J = []
        for J, row in pending_J_blocks.items():
            deps = row.get("Dependency_Condition", [])
            ext_deps = row.get("Minimal_External_Topics", [])
            deps_ready = all(dep in completed_J_blocks or dep == "Evaluate First" for dep in deps)
            ext_ready = all(k in external_consensus_values for k in ext_deps)
            if deps_ready and (len(ext_deps) == 0 or ext_ready):
                ready_J.append(J)
        if not ready_J:
            debug_state = {J: {"Dependency_Condition": row.get("Dependency_Condition", []), "Minimal_External_Topics": row.get("Minimal_External_Topics", []), "available_external_topics": sorted(external_consensus_values.keys())} for J, row in pending_J_blocks.items()}
            raise RuntimeError(f"Deadlock in dependency resolution: {debug_state}")
        for J in ready_J:
            row = pending_J_blocks[J]
            topics = row["Topic_Indices_0b"]
            theorem = row["Apply_Theorem"]
            ext_deps = row.get("Minimal_External_Topics", [])
            C_list_sliced = [Ci[np.ix_(topics, topics)] for Ci in deepcopy(C_list)]
            W_sliced = W[np.ix_(topics, topics)]
            if verbose:
                print(f"\n=== Processing {J} ===")
                print(f"Topics involved (0-based): {topics}")
                print(f"Theorem to apply: {theorem}")
                print(f"External dependencies: {ext_deps}")
            if "Theorem 2" in theorem:
                if theorem2_mode == "full":
                    x_final, x_hist = simulate_theorem2_multitopic(topic_indices=topics, C_list=C_list, W=W, T=T, return_history=True, init_mode=init_mode, verify_closedness=True)
                else:
                    x_final, x_hist = simulate_theorem2_multitopic(topic_indices=list(range(len(topics))), C_list=C_list_sliced, W=W_sliced, T=T, return_history=True, init_mode=init_mode, verify_closedness=True)
            elif theorem == "Theorem 3":
                x_final, x_hist = simulate_opinion_dynamics_singleton(topic_index=topics[0], C_list=C_list, W=W, T=T, return_history=True, init_mode=init_mode)
            elif theorem == "Theorem 3 (via Corollary 2.1)":
                ext_vals = {k: external_consensus_values[k] for k in ext_deps}
                x_final, x_hist = simulate_opinion_dynamics_corollary2(topic_index=topics[0], dependency_indices=ext_deps, W=W, C_list=C_list, external_consensus_values=ext_vals, T=50, return_history=True, init_mode=init_mode)
            elif "Theorem 4" in theorem or "external consensus" in theorem:
                ext_vals = {k: external_consensus_values[k] for k in ext_deps}
                x_final, x_hist = simulate_theorem4_multitopic(topic_indices=topics, C_list=C_list, W=W, external_consensus_values=ext_vals, T=T, return_history=True, init_mode=init_mode)
            else:
                x_final, x_hist = None, None

            if x_final is not None:
                x_arr = np.asarray(x_final)
                if x_arr.ndim == 1:
                    for t in topics:
                        external_consensus_values[t] = x_arr[0] if np.allclose(x_arr, x_arr[0], atol=1e-4) else x_arr
                else:
                    for i, t in enumerate(topics):
                        vals = x_arr[:, i]
                        external_consensus_values[t] = vals[0] if np.allclose(vals, vals[0], atol=1e-4) else vals

            results[J] = {
                "topics": topics,
                "Apply_Theorem": theorem,
                "x_final": np.asarray(x_final).tolist() if x_final is not None else None,
                "x_hist": np.asarray(x_hist).tolist() if x_hist is not None else None,
            }
            results_simple[J] = {"topics": topics, "x_final": np.asarray(x_final).tolist() if x_final is not None else None}
            completed_J_blocks.add(J)
            del pending_J_blocks[J]
            if verbose:
                print(f"✅ Completed {J}")
                print(f"Updated external consensus values: {external_consensus_values}")
    if iteration >= max_iters:
        print("⚠️ Stopped early: Max iterations reached")
    return (results, results_simple) if return_simple else (results, None)


def compute_convergence_indicators_verbose(results: Dict[str, dict], tol: float = 1e-4) -> Dict[str, dict]:
    convergence_dict = {}
    for J, data in results.items():
        topics = data["topics"]
        x_final = data["x_final"]
        if x_final is None:
            convergence_dict[J] = {"J_converged": "Not Converged", "Topic_convergence": {t: "Not Converged" for t in topics}}
            continue
        x_final = np.asarray(x_final)
        topic_convergence = {}
        if x_final.ndim == 1:
            topic_convergence[topics[0]] = "Converged" if np.allclose(x_final, x_final[0], atol=tol) else "Not Converged"
        else:
            for i, t in enumerate(topics):
                vals = x_final[:, i]
                topic_convergence[t] = "Converged" if np.allclose(vals, vals[0], atol=tol) else "Not Converged"
        convergence_dict[J] = {"J_converged": "Converged" if all(v == "Converged" for v in topic_convergence.values()) else "Not Converged", "Topic_convergence": topic_convergence}
    return convergence_dict


def plot_all_topics_opinion_dynamics(
    histories: Sequence[np.ndarray],
    topic_labels: Sequence[str],
    logic_group_indices: Optional[Sequence[int]] = None,
    alt_group_indices: Optional[Sequence[int]] = None,
    colors: Optional[Sequence[str]] = None,
    title: str = "Multi-Topic Opinion Evolution",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """Plot the multi-topic trajectory overlay used in the replication figures.

    If save_path is provided, the figure is saved as a PNG.
    """
    n_topics = len(histories)
    n_users = histories[0].shape[1]
    T = histories[0].shape[0] - 1

    fig = plt.figure(figsize=(10, 6))
    for topic_idx, x_hist in enumerate(histories):
        for i in range(n_users):
            linestyle = ":"
            if logic_group_indices and i in logic_group_indices:
                linestyle = "-"
            elif alt_group_indices and i in alt_group_indices:
                linestyle = "--"
            color = colors[topic_idx] if colors else None
            plt.plot(range(T + 1), x_hist[:, i], linestyle, color=color)

    legend_patches = [plt.Line2D([0], [0], color=colors[i], label=topic_labels[i]) for i in range(n_topics)] if colors else None
    if legend_patches is not None:
        plt.legend(handles=legend_patches)
    plt.title(title)
    plt.xlabel("Time, t")
    plt.ylabel("Opinion value, x_i(t)")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
