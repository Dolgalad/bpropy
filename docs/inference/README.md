# GPU-compatible graphical-model inference in `bpropy`

This is a new, **experimental API** for discrete graphical-model inference. It
lives alongside the original `bpropy.mrf` research code, which remains unchanged.
Its algorithms are differentiable **pure PyTorch** operations, not custom CUDA
kernels, and can run on CPUs and PyTorch-supported GPUs. `torch_scatter` and
`torch_geometric` are **not** required to run the inference core.

## Installation

From the project root, after choosing a suitable PyTorch build for your system:

```sh
python -m pip install -e '.[inference]'
python -m unittest discover -s tests/inference -v
python examples/inference_quickstart.py
```

For optional PyTorch Geometric interoperability, install `bpropy[pyg]`. **Note:**
the current top-level `bpropy` distribution still includes its legacy research
dependencies and builds the SCG3 binary. Packaging simplification, an explicit
project license, and a PyPI release are separate follow-up tasks.

## Ten-line example

```python
import torch
from bpropy.inference import FactorGraph, lbp, mean_field

graph = FactorGraph.from_scopes(
    cardinalities=[2, 2, 3],
    scopes=[(0,), (0, 1), (1, 2)],
)
plan = graph.compile().to("cuda" if torch.cuda.is_available() else "cpu")
theta = torch.nn.Parameter(torch.randn(plan.num_configurations, device=plan.device))
bp_result = lbp(plan, theta, iterations=20, damping=0.2)
nmf_result = mean_field(plan, theta, iterations=20, damping=0.2)
loss = -bp_result.marginal(0)[1].clamp_min(1e-12).log()
loss.backward()
```

`theta` can be produced by any PyTorch network; the graph's compilation can
remain cached for the entire training run if connectivity/cardinalities do not
change. Only move and compile the topology **outside** the training loop.

## Mathematical representation

A factor graph represents a joint distribution of discrete variables:

\[
p(x_0,\ldots,x_{V-1}) \propto
\exp\left[\sum_{f=0}^{F-1} \theta_f(x_{S_f})\right],
\]

where factor scopes `S_f` are **ordered**, and `theta_f` is a dense table of
log potentials. No unary potentials are added automatically. Duplicate scopes
are allowed and represent separate factors. A variable with no incident factor
has uniform marginal. Constant/empty factors are also supported.

`FactorGraph` consists solely of index tensors:

| Field | Shape | Meaning |
|---|---:|---|
| `cardinalities` | `[V]` | Number of states for each variable |
| `factor_ptr` | `[F+1]` | CSR pointers into the incidence array |
| `factor_vars` | `[E]` | Variable IDs, concatenated in factor-scope order |
| `variable_graph_ptr` | `[B+1]` | Variable offsets of independent batched graphs |
| `factor_graph_ptr` | `[B+1]` | Factor offsets of independent batched graphs |
| `potential_ptr` | `[F+1]` | *Derived* table offsets; not model parameters |

The full `log_potentials` parameter is a **one-dimensional** PyTorch floating
point tensor of length `potential_ptr[-1]`. All factor tables use standard
**PyTorch row-major order**: in `(v0, v1, v2)` the *last* variable `v2`
changes fastest. For a binary pair `(v0, v1)`, the table ordering is
`(0,0), (0,1), (1,0), (1,1)`.

To access the table of factor `f`, slice `theta[potential_ptr[f]:potential_ptr[f+1]]`.
Reshape to the factor scope's cardinalities to obtain the standard dense table.
`-inf` entries represent forbidden configurations (supported by LBP).

### Compiled plan

`CompiledFactorGraph = graph.compile()` precomputes reusable *indices only*:

| Field | Shape | Meaning |
|---|---:|---|
| `state_ptr` | `[V+1]` | Offsets of variable-state beliefs |
| `state_var` | `[S]` | Variable ID per variable-state entry |
| `config_ptr` | `[F+1]` | Offsets into log-potential tables |
| `config_factor` | `[T]` | Factor ID per table entry |
| `message_ptr` | `[E+1]` | Offsets for one categorical message per factor incidence |
| `message_incidence` | `[M]` | Incidence ID per message-state slot |
| `message_var_state` | `[M]` | Variable-state ID per message-state slot |
| `entry_config` | `[A]` | Factor configuration ID for each participating variable |
| `entry_message` | `[A]` | Message-state ID for that variable in that configuration |

Here `S=sum_i cardinalities[i]`, `T=sum_f prod(cardinalities[scope(f)])`,
`M=sum_{(f,i)} cardinalities[i]`, and
`A=sum_f (len(scope(f))*prod(cardinalities[scope(f)]))`.

The old LBP edge construction materialized approximately
`sum_f (len(scope(f))**2 * 2**len(scope(f)))` connections on binary factors.
The new plan's configuration entries scale with `A`, reducing index storage
for high-arity factors. **This is an asymptotic storage improvement, not a
measured runtime speedup.** Runtime and peak memory must be benchmarked on
actual CPU/GPU workloads. Dense potential tables still scale exponentially
with factor arity; `compile()` rejects a factor with more than 1,000,000
configurations by default, as well as more than 8,000,000 total entries.

## Inference algorithms

### Loopy Belief Propagation

```python
result = lbp(plan, theta, iterations=30, damping=0.2,
             tol=None, return_factor_beliefs=True)
```

- Synchronous **sum-product** updates in log-space.
- Variable-to-factor and factor-to-variable messages are contiguous flat
  buffers, with a segment per factor incidence.
- Factor-to-variable updates marginalize using a parallel grouped
  `logsumexp` over each factor assignment.
- Uses PyTorch `scatter_add` / `scatter_reduce` on the selected device.
- `damping=0`: no damping; positive damping is a **probability-space convex
  combination** of old and new *normalized* messages.
- By default, performs a **fixed** number of iterations for differentiable
  unrolled training. Optional `tol` triggers an early stop, including a GPU
  scalar synchronization each iteration.
- Returns factor *and* variable approximate marginals. For tree factor graphs
  the results match exact inference once messages have propagated.
- Supports `-inf` configurations for hard constraints as long as the model
  admits valid assignments. Models with contradictory zero-probability
  constraints can lead to undefined marginals.

### Naive mean field

```python
result = mean_field(plan, theta, iterations=30, damping=0.2)
```

- Fully factorized `q(x)=prod_i q_i(x_i)` with synchronous fixed-point updates.
- Shares exactly the same graph, table order, and `InferenceResult` API.
- Uses parallel scatter sums and cavity-expectation products.
- Returns **variable** marginals; `factor_beliefs=None` since variational
  products are not true factor marginals.
- The current implementation **requires finite log potentials**. For
  hard constraints use LBP or soften `-inf` energies to finite penalties.
- The default numerical validation of finite mean-field energies involves
  one GPU synchronization *before* the iteration loop. Set
  `validate_finite=False` only when inputs are guaranteed finite to remove
  that synchronization and permit `torch.compile` fullgraph tracing.

`InferenceResult` fields: `variable_beliefs` (flat tensor), `factor_beliefs`
(flat tensor or None), `iterations`, `residual` (scalar tensor), `converged`
(bool or None), and `algorithm`. Use `result.marginal(i)` for one variable and
`result.factor_marginal(f)` for one factor.

### Batching

Use a disjoint union rather than padding factors to a common arity:

```python
batched_graph = FactorGraph.batch([graph_a, graph_b])
plan = batched_graph.compile().to(device)
theta = torch.cat([theta_a, theta_b]).to(device)
result = lbp(plan, theta, iterations=20)
```

The result contains one flat set of marginals, with variable IDs offset
appropriately. The graph records `variable_graph_ptr` and `factor_graph_ptr`
for splitting results by graph. Mixing graphs with different categorical
cardinalities and factor degrees is supported.

### PyTorch Geometric integration

```python
from torch_geometric.loader import DataLoader

pyg_a = graph_a.to_pyg(theta_a)
pyg_b = graph_b.to_pyg(theta_b)
batch = next(iter(DataLoader([pyg_a, pyg_b], batch_size=2)))
core_graph = FactorGraph.from_pyg(batch)
plan = core_graph.compile().to(device)
theta = FactorGraph.potentials_from_pyg(batch).to(device)
result = lbp(plan, theta)
```

The `HeteroData` has `variable` and `factor` node types, reciprocal incidence
edge types `('variable', 'in', 'factor')` and `('factor', 'has', 'variable')`,
and an **incidence slot** number preserving the factor variable order. Factor
potentials are stored as one *flat* tensor on the factor node store, rather
than as pointer values that would be incorrectly offset by default PyG
batching. `torch_geometric` is imported lazily, so it is optional.

Converting an already-batched `FactorGraph` directly to PyG retains the
variable/factor `ptr` and `batch` attributes. Do not batch such an object a
second time through PyG; feed individual graphs into the loader instead.

PyG's ordinary `MessagePassing` base class is intentionally **not** used for
the BP kernel: factor-to-variable messages require a reduction over all
configurations of the remaining variables, which is not ordinary neighbor
aggregation.

## Compatibility with original bpropy

See [Migration](migration.md). In particular, the legacy bit order for binary
factor tables differs from the standard row-major order. Use the explicit
converter instead of reusing the old `theta` buffers unchanged.

## Reproducible CPU/GPU benchmarking

See [CUDA validation and benchmarking](benchmarking.md) for CPU/CUDA parity,
backward timing, GPU-event latency, memory accounting, a reproducible GPU
sweep and optional comparisons with the legacy `torch_scatter` kernels.
GPU-specific claims must be supported by measurements from actual CUDA hardware.

## Quality gates before publishing

See [Release checklist](release.md). Tests cover exact tree marginals, hard
constraints, variable cardinalities, batched models, PyTorch autograd,
`gradcheck`, and optional PyG roundtrips / CUDA parity. Passing CPU tests does
**not** imply that CUDA performance or torch.compile has been validated.
