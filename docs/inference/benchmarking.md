# CUDA validation and performance benchmarking

The inference core implements parallel message-passing operations with ordinary
PyTorch gather/scatter kernels. It **does not require** `torch_scatter` or a
custom CUDA extension. Whether that is faster than the legacy implementation
must be established empirically on a particular GPU. Passing CPU tests alone
is not evidence of GPU correctness or speed.

## Correctness tests

Run from the repository root with an installed PyTorch build compatible with
the selected NVIDIA GPU (and CUDA driver):

```bash
PYTHONPATH=src python -m unittest discover -s tests/inference -v
```

When CUDA is available, this exercises CPU/CUDA belief and gradient parity for
both LBP and mean field (float32 and float64), heterogeneous state counts,
ternary factors, batching, isolated variables, hard-constraint LBP, and
repeatability within realistic floating-point tolerances. Tests requiring
CUDA are **skipped** on the standard CPU-only GitHub Actions runners.

The tests deliberately compare **fixed iteration counts** and use identical
factor ordering, potentials, and damping. On CUDA, scatter accumulation order
can vary; strict bit-for-bit equality is not required. Hard-constraint
(`-inf`) checks cover LBP forward inference but not gradients through exact
zero-probability configurations.

## Single-case benchmark

```bash
PYTHONPATH=src python benchmarks/inference_core.py \
  --device cuda --verify-cpu \
  --variables 500 --factors 1200 --arity 3 \
  --iterations 20 --repeats 7 --warmup 3 \
  --json-out results/inference/gpu.json \
  --csv-out results/inference/gpu.csv
```

To benchmark a batch, add `--graphs 16`; `--variables` and `--factors` then
refer to counts **per graph**. `--cardinality 3 --mixed-cardinalities` generates
independent categorical factors with varying state counts. `--dtype float64`
tests double precision. `--compile inductor` requests native compilation;
leave it at the default `none` for a clear eager-PyTorch baseline. Use a
positive `--warmup` value to exclude initial compilation and allocator effects.

`--verify-cpu` compares both variable beliefs and a differentiable weighted
loss's parameter gradients to **CPU results on the same parameters**. A
failed allclose check exits with a nonzero status instead of recording
misleading speed results. The default reference tolerance is
`atol=1e-4, rtol=1e-3`, adjustable via `--atol` and `--rtol`.

The output includes:

- **Topology compilation** on CPU and separately timed transfer to the device.
- **Forward, backward-only, and forward+backward** latency for both LBP and
  mean field. `--no-backward` disables gradient timings.
- Synchronized wall-clock latency (Python dispatch **plus** GPU execution)
  and, on CUDA, separate CUDA-event device latency. Timed operations have a
  separate warmup and the device is synchronized for every sample.
- Peak **additional PyTorch CUDA tensor allocation** above the allocation at
  the start of a timed phase. For backward-only this excludes the forward graph
  prepared before timing; it does **not** reflect total driver memory / VRAM.
- Model size and compiled index storage, PyTorch/CUDA versions, GPU name,
  compute capability, random seed, damping and iteration count.

The benchmark does not include model compilation inside inference timings,
and does not download data. `--seed` fixes graph and parameter sampling
independently of CPU versus CUDA execution.

## Automated GPU suite (including CPU timings)

On an already allocated GPU (including a Slurm GPU allocation), run:

```bash
bash benchmarks/run_gpu_suite.sh results/inference-gpu
```

This first runs correctness tests, then three matching CPU/CUDA workloads:
small pairwise, medium ternary, and a batched pairwise workload. The output
folder contains JSON and CSV files. You can customize with environment
variables `INFERENCE_BENCH_REPEATS`, `INFERENCE_BENCH_WARMUP`,
`INFERENCE_BENCH_COMPILE` and `INFERENCE_BENCH_CPU=0` (to omit separate CPU
latency measurements). No Slurm partition or account is hardcoded; request the
GPU via your cluster's own job configuration.

For a compute node with a GPU not supported by your installed PyTorch build,
install an appropriate official CUDA PyTorch build *before* running the suite;
CUDA availability alone does not guarantee kernel support for that device.

## Legacy `torch_scatter` comparison

If the original research dependencies and `torch_scatter` are installed:

```bash
PYTHONPATH=src python benchmarks/inference_core.py \
  --device cuda --variables 100 --factors 250 --arity 2 \
  --iterations 20 --damping 0 --repeats 7 --warmup 3 \
  --compare-legacy --no-backward \
  --json-out results/inference/legacy-cuda.json
```

This deliberately restricts models to **one binary unary/pairwise factor graph
using float32 and zero damping**. The legacy graph builder sorts the scopes;
the new representation is reordered to match it, and potential tables are
converted between the old little-endian and new row-major layouts. Both
implementations use the same fixed number of message-passing iterations.

The result includes legacy forward timing, CPU/GPU-compatible native forward
timing, a ratio of legacy/new wall time, and numerical comparisons of LBP
factor marginals and NMF variable marginals. **Read the comparison flags**:
legacy code uses a different message-update convention, and a timing ratio is
not evidence of equivalent outputs if the numerical checks disagree. The old
kernels also synchronize inside their loops; the new fixed-iteration kernels
do not. We do **not** present a backward-speed comparison with the old research
implementation because its differentiability has not been established.

The benchmark is intentionally not a universal comparison with all graphical
model inference libraries; broader baselines (such as PGMax and torch-bp)
require their own carefully aligned inference protocols.

## Benchmarking caveats

* Run both implementations on the same hardware, dtype, factor topology,
  log-potentials, iteration count and convergence policy.
* Keep warmups and repeats consistent; avoid shared/overloaded GPUs and
  thermal throttling; record the GPU type and PyTorch build.
* Compare **forward+backward** as well as forward-only performance for
  differentiable training. A fast inference kernel is not automatically a
  fast training kernel.
* On very small graphs, Python/dispatch overhead can dominate kernel time;
  compare CUDA events with synchronized wall time to understand the difference.
* A custom CUDA kernel, implicit differentiation, GPU graphs and sparse/high
  arity representations are future experiments, not features already claimed
  by the current implementation.
