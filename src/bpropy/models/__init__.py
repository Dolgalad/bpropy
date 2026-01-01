from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple

# ---- Model registry (start small; move to bpropy.models.registry later) ----
from bpropy.models.logit_models import GCNVarLogits, HeteroGCNVarLogits, HeteroGCNVarLogits_v2, ResidualSAGEVarLogits
from bpropy.models.mrf_models import GCN_MRF_v4, Hetero_GCN_MRF_v4

MODEL_REGISTRY: Dict[str, Any] = {
    "GCNVarLogits": GCNVarLogits,
    "HeteroGCNVarLogits": HeteroGCNVarLogits,
    "HeteroGCNVarLogits_v2": HeteroGCNVarLogits_v2,
    "GCN_MRF_v4": GCN_MRF_v4,
    "Hetero_GCN_MRF_v4": Hetero_GCN_MRF_v4,

    "ResidualSAGEVarLogits": ResidualSAGEVarLogits


}

