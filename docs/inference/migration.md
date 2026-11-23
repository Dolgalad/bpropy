# Migration from the original `bpropy.mrf` package

**The old inference routines and datasets are retained unchanged.** The new
`bpropy.inference` API is deliberately separate; it is not a drop-in replacement.

## Structure and parameter conversion

Original `FactorGraph(pb)` derives binary factors from MIS/SCP constraints,
with the first scope variable as the least-significant bit in the overcomplete
sufficient-statistics encoding. The new API accepts arbitrary discrete factor
scopes and uses PyTorch row-major table order (last scope variable fastest).

```python
import torch
from bpropy.inference import lbp
from bpropy.inference.legacy import (
    from_legacy_factor_graph,
    legacy_to_row_major,
)

# old_graph = bpropy.mrf.factor_graph.FactorGraph(pb)
# old_theta = your original theta, ordered by binary bit index
new_graph = from_legacy_factor_graph(old_graph)
plan = new_graph.compile()
new_theta = legacy_to_row_major(new_graph, old_theta)
result = lbp(plan, new_theta, iterations=30)
```

The reorder operation uses differentiable `index_select`. The inverse is
`row_major_to_legacy(new_graph, values)`.

The old `LBPEdges` and `NMFEdges` are **not accepted** by the new routines.
Use `plan = new_graph.compile()` to build a single reusable index plan shared
by BP and NMF. On CUDA move `plan.to("cuda")`, then pass `theta.cuda()`.

## Differences that affect numerical reproduction

- Legacy LBP damped normalized messages **in log space**. New LBP damps
  **probabilities** and then logs the mixture. Set `damping=0` to compare
  numerical outputs of the undamped algorithms.
- Legacy NMF used its own damping convention, and previously fixed some
  intermediate tensors to float32. New NMF uses the input theta's dtype.
- Factor IDs are distinct from variable IDs. Duplicate scopes are permitted.
- Both new algorithms return an `InferenceResult`, not `BPSolution` / `NMFSolution`.
- The number of iterations is fixed by default (`tol=None`). Use `tol=...`
  only for optional convergence-based early stopping.
- The old examples may attach MIS/SCP-specific semantics to unary factors;
  the new graph adds **no** factors automatically.

## Backward-compatibility workflow

1. Keep the legacy algorithms and their unit tests as reference code.
2. Generate small random binary instances and fixed `theta` fixtures.
3. Compare **undamped** LBP and NMF outputs after converting factor order;
   also compare tree cases against brute-force inference.
4. Compare gradients with `torch.autograd.gradcheck` and input-parameter
   gradients in the original experiment models.
5. Only then migrate MIS/SCP wrappers to the general representation.

**Known limits:** the adapter assumes binary legacy variables. The current
`bpropy.mrf` API has extra experiment-specific graph metadata; models that
rely on that metadata should continue to use the legacy class while being
ported. Results may differ when early-stopping or damping semantics differ.
