# `bpropy` — differentiable graphical-model inference and research code

`bpropy` provides research implementations of GPU-oriented, differentiable
inference in discrete factor graphs using PyTorch. It also contains the
combinatorial optimization experiments used in the associated thesis work.

## New, reusable inference API (experimental)

The new **`bpropy.inference`** package offers a general tensor-based factor
graph representation (including different discrete cardinalities), compiled
GPU-friendly scatter indices, differentiable sum-product belief propagation,
and naive mean-field inference. It does not depend on NetworkX, PyG, or
`torch_scatter` at runtime; PyG interchange is optional.

See the [inference documentation](docs/inference/README.md),
[migration guide](docs/inference/migration.md),
[GPU benchmarking guide](docs/inference/benchmarking.md), and
[publishing checklist](docs/inference/release.md).

```python
import torch
from bpropy.inference import FactorGraph, lbp, mean_field

fg = FactorGraph.from_scopes([2, 2], [(0,), (0, 1)])
plan = fg.compile().to("cuda" if torch.cuda.is_available() else "cpu")
theta = torch.nn.Parameter(torch.randn(plan.num_configurations, device=plan.device))
result = lbp(plan, theta, iterations=20)
loss = -result.marginal(0)[1].log()
loss.backward()
```

Use `python examples/inference_quickstart.py` to run a full example.

## Legacy code and experiments

The original `bpropy.mrf`, `bpropy.co`, research datasets, models, CLI entry
points and experiment scripts remain available and **are not deleted or
silently migrated** by the new inference package. The old API has different
binary factor-table ordering and message damping conventions; use the
migration guide before attempting numerical comparisons.

## Publication status

The new inference API is **not yet a published PyPI release**. Packaging the
original research dependencies, bundled C utilities, licensing, and GPU
performance validation are tracked in the release checklist.
