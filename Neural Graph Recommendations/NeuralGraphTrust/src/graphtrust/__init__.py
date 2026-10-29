"""NeuralGraphTrust (Stage 3): inference-time graph repair on a trained attentive recommender.

  reliability  diagnose: per-edge scores r, rel, dot, cos and their AUROC against injected edges
  policies     act: pi0, pi_rel and matched controls (lowest dot / cos, random, oracle)
  repair       verify: frozen-parameter re-propagation on the edited graph, Jaccard gate, targeting
  evaluation   per-user metrics (joint and isolated repair), degree groups, explanation fidelity
"""

from .evaluation import (degree_groups, fidelity, fill_unrepaired, isolated_user_eval, nanmean,
                         original_degree, per_user_eval)
from .policies import build_policies, edge_weights, matched_control, threshold_policy
from .reliability import EdgeTable, auroc, edge_table, reliability_summary
from .repair import fast_norm_adj, gate, jaccard_dist, propagate_on_graph, target_users

__all__ = [
    "EdgeTable", "auroc", "edge_table", "reliability_summary",
    "build_policies", "edge_weights", "matched_control", "threshold_policy",
    "fast_norm_adj", "gate", "jaccard_dist", "propagate_on_graph", "target_users",
    "degree_groups", "fidelity", "fill_unrepaired", "isolated_user_eval", "nanmean",
    "original_degree", "per_user_eval",
]
