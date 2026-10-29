"""Model registry: NGCF, Attention-NGCF (Stage 1), LightGCN, SimGCL and the gated attentive models (Stage 2)."""

from .ngcf import NGCF
from .lightgcn import LightGCN
from .attentive_lightgcn import AttentiveLightGCN
from .simgcl import SimGCL
from .attentive_simgcl import AttentiveSimGCL
from .attentive_ngcf import AttentiveNGCF


MODEL_REGISTRY = {
    "ngcf": NGCF,
    "lightgcn": LightGCN,
    "attentive_lightgcn": AttentiveLightGCN,
    "simgcl": SimGCL,
    "attentive_simgcl": AttentiveSimGCL,
    "attentive_ngcf": AttentiveNGCF,
}


def build_model(name: str, **kwargs):
    """Instantiate a model by name."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "NGCF",
    "LightGCN",
    "AttentiveLightGCN",
    "SimGCL",
    "AttentiveSimGCL",
    "AttentiveNGCF",
    "MODEL_REGISTRY",
    "build_model",
]
