"""PyTorch-native, differentiable inference on discrete factor graphs.

Public entry points:

- ``FactorGraph``: canonical tensor-only topology and PyG import/export.
- ``CompiledFactorGraph``: immutable, reusable GPU scatter/gather plan.
- ``lbp`` / ``mean_field``: unrolled differentiable inference functions.
- ``LBP`` / ``NaiveMeanField``: PyTorch ``nn.Module`` wrappers.

The existing ``bpropy.mrf`` research implementation remains untouched.
"""
from .graph import FactorGraph
from .plan import CompiledFactorGraph
from .algorithms import InferenceResult, LBP, NaiveMeanField, lbp, mean_field

__all__ = [
    "FactorGraph", "CompiledFactorGraph", "InferenceResult",
    "LBP", "NaiveMeanField", "lbp", "mean_field",
]
