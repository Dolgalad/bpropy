"""Reproducible CPU/CUDA benchmarks for differentiable factor-graph inference.

The *same* compiled graph and potential values are used for every method and
backend. Compilation/transfer are measured separately from steady-state
inference. CUDA reports synchronized wall time and CUDA-event device time;
never interpret asynchronous Python enqueue time as GPU execution time.

Examples (from the repository root)::

    PYTHONPATH=src python benchmarks/inference_core.py --device cpu --variables 100 --factors 200
    PYTHONPATH=src python benchmarks/inference_core.py --device cuda --verify-cpu --json-out results/gpu.json
    PYTHONPATH=src python benchmarks/inference_core.py --device cuda --graphs 16 --arity 3 --compile inductor

Legacy comparisons require the old research dependencies, torch-scatter and
binary factors of arity one or two. They run on precisely matched topology and
log potentials (including the legacy little-endian factor-table conversion).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from bpropy.inference import FactorGraph, LBP, NaiveMeanField, lbp, mean_field


MIB = 1024 ** 2


def build_graph(*, variables: int, factors: int, cardinality: int,
                arity: int, graphs: int, seed: int, mixed_cardinalities: bool,
                legacy_compatible: bool = False) -> FactorGraph:
    """Make a deterministic disjoint batch of random, bounded-arity factors."""
    rng = random.Random(seed)
    parts = []
    for _ in range(graphs):
        cards = ([rng.randint(2, cardinality) for _ in range(variables)]
                 if mixed_cardinalities else [cardinality] * variables)
        scopes = [tuple(rng.sample(range(variables), arity)) for _ in range(factors)]
        if legacy_compatible:
            # The old binary builder sorts variables *inside* each factor and
            # then sorts factors lexicographically. Preserve its ordering.
            scopes = sorted((tuple(sorted(scope)) for scope in scopes),
                            key=lambda scope: (len(scope), scope))
        parts.append(FactorGraph.from_scopes(cards, scopes))
    return parts[0] if len(parts) == 1 else FactorGraph.batch(parts)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _single_trial(fn: Callable[[], Any], device: torch.device) -> dict[str, float]:
    """Time one operation and measure *additional* CUDA tensor memory.

    Peak memory is relative to the allocated bytes just before the operation;
    for backward-only this excludes the already-built forward autograd graph.
    It is not a measure of total process/driver VRAM.
    """
    _sync(device)
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
    wall_start = time.perf_counter()
    output = fn()
    if device.type == "cuda":
        stop.record()
        stop.synchronize()
    wall_ms = 1000 * (time.perf_counter() - wall_start)
    sample = {"wall_ms": wall_ms}
    if device.type == "cuda":
        sample["cuda_event_ms"] = start.elapsed_time(stop)
        sample["peak_additional_cuda_mib"] = max(
            0, torch.cuda.max_memory_allocated(device) - baseline) / MIB
    del output
    return sample


def measure(prepare: Callable[[], Callable[[], Any]], *, device: torch.device,
            repeats: int, warmup: int) -> dict[str, float]:
    """Measure an operation after setup, with per-sample warmup and sync."""
    for _ in range(warmup):
        _single_trial(prepare(), device)
    samples = [_single_trial(prepare(), device) for _ in range(repeats)]
    out: dict[str, float] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        out[f"{key}_median"] = statistics.median(values)
        out[f"{key}_min"] = min(values)
        out[f"{key}_max"] = max(values)
    return out


def _modules(plan, *, iterations: int, damping: float, compilation: str):
    modules = {
        "lbp": LBP(iterations=iterations, damping=damping, return_factor_beliefs=False),
        "mean_field": NaiveMeanField(iterations=iterations, damping=damping,
                                     validate_finite=False),
    }
    if compilation != "none":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch version")
        modules = {name: torch.compile(module, backend=compilation, fullgraph=True)
                   for name, module in modules.items()}
    return {name: lambda theta, m=module: m(plan, theta).variable_beliefs
            for name, module in modules.items()}


def _gradient(fn: Callable[[torch.Tensor], torch.Tensor], theta: torch.Tensor) -> torch.Tensor:
    # Use variable beliefs (not theta) for a non-constant differentiable loss.
    beliefs = fn(theta)
    # Per-state weights are intentionally non-uniform: sum(beliefs) is constant.
    state_weights = torch.linspace(0.1, 1.1, beliefs.numel(),
                                   device=theta.device, dtype=theta.dtype)
    loss = (beliefs * state_weights).sum() / max(1, beliefs.numel())
    return torch.autograd.grad(loss, theta)[0]


def _loss_and_theta(fn: Callable[[torch.Tensor], torch.Tensor],
                    theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = theta.detach().clone().requires_grad_(True)
    beliefs = fn(t)
    weights = torch.linspace(0.1, 1.1, beliefs.numel(),
                             dtype=theta.dtype, device=theta.device)
    return (beliefs * weights).sum() / max(1, beliefs.numel()), t


def benchmark_algorithm(fn: Callable[[torch.Tensor], torch.Tensor], theta: torch.Tensor,
                        *, repeats: int, warmup: int, do_backward: bool) -> dict:
    """Report forward (no-grad), backward-only, and complete training step."""
    device = theta.device

    def forward_prepare():
        def run():
            with torch.no_grad():
                return fn(theta)
        return run

    result = {"forward": measure(forward_prepare, device=device,
                                 repeats=repeats, warmup=warmup)}
    if do_backward:
        def backward_prepare():
            loss, t = _loss_and_theta(fn, theta)
            return lambda: torch.autograd.grad(loss, t)[0]

        def step_prepare():
            t = theta.detach().clone().requires_grad_(True)
            return lambda: _gradient(fn, t)

        result["backward_only"] = measure(backward_prepare, device=device,
                                          repeats=repeats, warmup=warmup)
        result["forward_backward"] = measure(step_prepare, device=device,
                                              repeats=repeats, warmup=warmup)
    return result


def compare_cpu(*, graph: FactorGraph, theta: torch.Tensor, gpu_plan,
                iterations: int, damping: float, atol: float, rtol: float) -> dict:
    """Compare all variable beliefs AND parameter gradients on identical inputs."""
    if theta.device.type != "cuda":
        raise ValueError("CPU-vs-GPU reference needs a CUDA input")
    cpu_plan = graph.compile()
    cpu_theta = theta.detach().cpu()
    comparisons = {}
    for name, fn, kw in (
        ("lbp", lbp, {"return_factor_beliefs": False}),
        ("mean_field", mean_field, {"validate_finite": False}),
    ):
        t_cpu = cpu_theta.clone().requires_grad_(True)
        t_gpu = theta.detach().clone().requires_grad_(True)
        left = fn(cpu_plan, t_cpu, iterations=iterations, damping=damping, **kw)
        right = fn(gpu_plan, t_gpu, iterations=iterations, damping=damping, **kw)
        b_cpu, b_gpu = left.variable_beliefs.detach(), right.variable_beliefs.detach().cpu()
        grad_cpu = _gradient(lambda x: fn(cpu_plan, x, iterations=iterations,
                                           damping=damping, **kw).variable_beliefs, t_cpu)
        grad_gpu = _gradient(lambda x: fn(gpu_plan, x, iterations=iterations,
                                           damping=damping, **kw).variable_beliefs, t_gpu).detach().cpu()
        beliefs_ok = bool(torch.allclose(b_cpu, b_gpu, atol=atol, rtol=rtol))
        gradients_ok = bool(torch.allclose(grad_cpu, grad_gpu, atol=atol, rtol=rtol))
        finite_ok = bool(torch.isfinite(b_gpu).all() and torch.isfinite(grad_gpu).all())
        comparisons[name] = {
            "beliefs_allclose": beliefs_ok, "gradients_allclose": gradients_ok,
            "finite": finite_ok,
            "max_belief_abs_error": float((b_cpu - b_gpu).abs().max().item()),
            "max_gradient_abs_error": float((grad_cpu - grad_gpu).abs().max().item()),
            "atol": atol, "rtol": rtol,
        }
        if not (beliefs_ok and gradients_ok and finite_ok):
            raise RuntimeError(f"{name} CPU/CUDA forward or gradient parity FAILED: {comparisons[name]}")
    return comparisons


def benchmark_legacy(*, variables: int, scopes: list[tuple[int, ...]],
                     theta: torch.Tensor, iterations: int, repeats: int,
                     warmup: int, damping: float, graph: FactorGraph,
                     plan, atol: float, rtol: float) -> dict:
    """Optional original torch_scatter benchmark, with table-order conversion.

    The experimental original algorithm mutates message buffers and is therefore
    timed **forward only**; no claim about differentiable backward parity is made.
    """
    try:
        from bpropy.mrf.factor_graph import FactorGraph as OldGraph
        from bpropy.mrf.lbp_edges import LBPEdges
        from bpropy.mrf.nmf_edges import NMFEdges
        from bpropy.mrf.lbp import loopy_belief_propagation
        from bpropy.mrf.naive_mean_field import naive_mean_field
        from bpropy.inference.legacy import from_legacy_factor_graph, row_major_to_legacy, legacy_to_row_major
    except ImportError as exc:
        raise RuntimeError(
            "Legacy comparison needs bpropy research dependencies and torch_scatter; "
            "install them following docs/installation.md"
        ) from exc
    old_graph = OldGraph()
    old_graph.append_instance_from_scopes(variables, [list(s) for s in scopes],
                                           add_unaries=False)
    old_topology = from_legacy_factor_graph(old_graph)
    if old_topology.factor_vars.tolist() != graph.factor_vars.tolist():
        raise ValueError("Legacy builder reordered the factors; cannot compare unmatched models")
    original_theta = row_major_to_legacy(graph, theta)
    bp_edges = LBPEdges(old_graph).to(theta.device)
    mf_edges = NMFEdges(old_graph).to(theta.device)
    num_v2f = int(bp_edges.n_f2v_msg.sum().item())
    if num_v2f != original_theta.numel():
        raise ValueError("Legacy LBP benchmark expects unary/pairwise binary factors")

    def old_bp():
        messages = torch.zeros(num_v2f, dtype=theta.dtype, device=theta.device)
        return loopy_belief_propagation(original_theta, messages, bp_edges,
                                         max_iter=iterations, tol=0.0, damping=damping)

    def old_mf():
        return naive_mean_field(original_theta, mf_edges,
                                max_iter=iterations, tol=0.0, damping=damping)

    with torch.no_grad():
        a = old_bp()
        b = old_mf()
        new_bp = lbp(plan, theta, iterations=iterations, damping=damping)
        new_mf = mean_field(plan, theta, iterations=iterations, damping=damping,
                            validate_finite=False)
    old_bp_factors = legacy_to_row_major(graph, a.mu)
    comparisons = {
        "lbp_factor_beliefs": {
            "allclose": bool(torch.allclose(old_bp_factors, new_bp.factor_beliefs,
                                             atol=atol, rtol=rtol)),
            "max_abs_error": float((old_bp_factors - new_bp.factor_beliefs).abs().max().item()),
        },
        "mean_field_variable_beliefs": {
            "allclose": bool(torch.allclose(b.mu_v, new_mf.variable_beliefs,
                                             atol=atol, rtol=rtol)),
            "max_abs_error": float((b.mu_v - new_mf.variable_beliefs).abs().max().item()),
        },
    }
    results = {
        "comparison": comparisons,
        "lbp_forward": measure(lambda: old_bp, device=theta.device,
                               repeats=repeats, warmup=warmup),
        "mean_field_forward": measure(lambda: old_mf, device=theta.device,
                                      repeats=repeats, warmup=warmup),
        "notes": "Legacy forward only; original kernels do not have a verified autograd interface."
    }
    return results


def _parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", choices=("cpu", "cuda"),
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--variables", type=int, default=300, help="Variables per graph")
    p.add_argument("--factors", type=int, default=600, help="Factors per graph")
    p.add_argument("--graphs", type=int, default=1, help="Independent graphs per batch")
    p.add_argument("--arity", type=int, default=2, help="Variables per factor")
    p.add_argument("--cardinality", type=int, default=2)
    p.add_argument("--mixed-cardinalities", action="store_true")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--damping", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--no-backward", action="store_true")
    p.add_argument("--compile", choices=("none", "eager", "inductor"), default="none",
                   help="Optional torch.compile backend; compile time excluded from steady-state timing")
    p.add_argument("--verify-cpu", action="store_true",
                   help="Check both marginal and gradient parity against CPU using identical model")
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--compare-legacy", action="store_true",
                   help="Optional original torch_scatter forward benchmark (binary, unary/pairwise only)")
    p.add_argument("--json-out", type=Path, help="Write full, machine-readable run report")
    p.add_argument("--csv-out", type=Path, help="Write one row per algorithm/phase")
    args = p.parse_args(argv)
    if args.variables < 1 or args.factors < 0 or args.graphs < 1:
        p.error("Require variables >= 1, factors >= 0, graphs >= 1")
    if args.factors == 0 and not args.no_backward:
        p.error("Factor-free graphs have no trainable potentials; use --no-backward")
    if args.arity < 1 or args.arity > args.variables:
        p.error("Require 1 <= arity <= variables")
    if args.cardinality < 2 or args.repeats < 1 or args.warmup < 0 or args.iterations < 1:
        p.error("Require cardinality >= 2, repeats/iterations >= 1, warmup >= 0")
    if not (0 <= args.damping < 1):
        p.error("Damping must be in [0, 1)")
    if args.atol < 0 or args.rtol < 0:
        p.error("Tolerance must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("Requested CUDA, but it is not available")
    if args.verify_cpu and args.device != "cuda":
        p.error("--verify-cpu requires --device cuda")
    if args.compare_legacy and (args.graphs != 1 or args.mixed_cardinalities or
                                args.cardinality != 2 or args.arity > 2 or args.factors < 1
                                or args.dtype != "float32" or args.damping != 0):
        p.error("Legacy comparison requires one binary unary/pairwise graph, "
                "float32 and damping=0 for comparable update semantics")
    return args


def _metadata(device: torch.device) -> dict:
    value = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        value["gpu_name"] = torch.cuda.get_device_name(device)
        value["compute_capability"] = list(torch.cuda.get_device_capability(device))
    return value


def _write_csv(path: Path, report: dict) -> None:
    rows = []
    for algorithm, timings in report["benchmarks"].items():
        for phase, measurements in timings.items():
            rows.append({
                "device": report["environment"]["device"],
                "torch_version": report["environment"]["torch"],
                "algorithm": algorithm, "phase": phase,
                "variables": report["graph"]["variables"],
                "factors": report["graph"]["factors"],
                "graphs": report["parameters"]["graphs"],
                "iterations": report["parameters"]["iterations"],
                "dtype": report["parameters"]["dtype"],
                "compiler": report["parameters"]["compile"],
                **measurements,
            })
    if "legacy" in report:
        for name in ("lbp", "mean_field"):
            rows.append({
                "device": report["environment"]["device"],
                "torch_version": report["environment"]["torch"],
                "algorithm": f"legacy_{name}", "phase": "forward",
                "variables": report["graph"]["variables"],
                "factors": report["graph"]["factors"],
                "graphs": report["parameters"]["graphs"],
                "iterations": report["parameters"]["iterations"],
                "dtype": report["parameters"]["dtype"],
                "compiler": "none",
                **report["legacy"][f"{name}_forward"],
            })
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    # Use a dedicated CPU generator so potential values never depend on the
    # selected accelerator or the installed CUDA version.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    graph = build_graph(variables=args.variables, factors=args.factors,
                        cardinality=args.cardinality, arity=args.arity,
                        graphs=args.graphs, seed=args.seed,
                        mixed_cardinalities=args.mixed_cardinalities,
                        legacy_compatible=args.compare_legacy)
    t0 = time.perf_counter()
    cpu_plan = graph.compile()
    build_ms = 1000 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    plan = cpu_plan.to(device)
    _sync(device)
    transfer_ms = 1000 * (time.perf_counter() - t0)
    dtype = getattr(torch, args.dtype)
    theta_cpu = torch.randn(plan.num_configurations, generator=generator, dtype=dtype) * 0.3
    theta = theta_cpu.to(device)
    modules = _modules(plan, iterations=args.iterations, damping=args.damping,
                       compilation=args.compile)
    results = {
        "schema_version": 2,
        "environment": _metadata(device),
        "parameters": {key: getattr(args, key) for key in (
            "variables", "factors", "graphs", "arity", "cardinality",
            "mixed_cardinalities", "dtype", "seed", "iterations", "damping",
            "warmup", "repeats", "compile", "verify_cpu", "compare_legacy")},
        "graph": {
            "variables": graph.num_variables, "factors": graph.num_factors,
            "configurations": plan.num_configurations,
            "states": plan.num_states, "message_states": plan.num_message_states,
            "configuration_entries": int(plan.entry_config.numel()),
            "compiled_index_mib": sum(x.numel() * x.element_size() for x in (
                plan.state_ptr, plan.state_var, plan.config_ptr,
                plan.config_factor, plan.message_ptr, plan.message_incidence,
                plan.message_var_state, plan.entry_config, plan.entry_message,
                plan.graph.cardinalities, plan.graph.factor_ptr, plan.graph.factor_vars,
                plan.graph.variable_graph_ptr, plan.graph.factor_graph_ptr)) / MIB,
        },
        "compilation": {"cpu_build_ms": build_ms, "transfer_ms": transfer_ms},
    }
    if args.verify_cpu:
        results["cpu_cuda_parity"] = compare_cpu(
            graph=graph, theta=theta, gpu_plan=plan, iterations=args.iterations,
            damping=args.damping, atol=args.atol, rtol=args.rtol)
    results["benchmarks"] = {
        name: benchmark_algorithm(fn, theta, repeats=args.repeats,
                                  warmup=args.warmup, do_backward=not args.no_backward)
        for name, fn in modules.items()
    }
    if args.compare_legacy:
        scopes = [graph.scope(f) for f in range(graph.num_factors)]
        results["legacy"] = benchmark_legacy(
            variables=args.variables, scopes=scopes, theta=theta, plan=plan,
            graph=graph, iterations=args.iterations, damping=args.damping,
            repeats=args.repeats, warmup=args.warmup, atol=args.atol, rtol=args.rtol)
        results["legacy"]["forward_speedup_vs_legacy_wall"] = {
            method: results["legacy"][f"{method}_forward"]["wall_ms_median"]
                    / results["benchmarks"][method]["forward"]["wall_ms_median"]
            for method in ("lbp", "mean_field")
        }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.csv_out:
        _write_csv(args.csv_out, results)
    return results


def main(argv: list[str] | None = None):
    print(json.dumps(run(argv), indent=2))


if __name__ == "__main__":
    main()
