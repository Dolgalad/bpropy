"""Reproducible baseline benchmark for the new compiled inference kernels.

This is deliberately *not* a legacy comparison. Benchmark legacy bpropy only
with a matched model, identical factor table ordering and equivalent damping.

Example:
  PYTHONPATH=src python benchmarks/inference_core.py --variables 500 --factors 1500
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from bpropy.inference import FactorGraph, lbp, mean_field


def benchmark(fn, theta, *, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    if theta.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        if theta.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fn()
        if theta.is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    result = {"median_ms": sorted(times)[len(times) // 2],
              "min_ms": min(times), "max_ms": max(times)}
    if theta.is_cuda:
        result["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variables", type=int, default=300)
    parser.add_argument("--factors", type=int, default=600)
    parser.add_argument("--cardinality", type=int, default=2)
    parser.add_argument("--arity", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.variables < 1 or args.factors < 0 or not (1 <= args.arity <= args.variables):
        parser.error("Require variables >= 1, factors >= 0, and 1 <= arity <= variables")
    if args.cardinality < 1 or args.repeats < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("Invalid cardinality or benchmarking configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Requested CUDA, but it is not available")

    random.seed(2026)
    torch.manual_seed(2026)
    scopes = [tuple(random.sample(range(args.variables), args.arity))
              for _ in range(args.factors)]
    graph = FactorGraph.from_scopes([args.cardinality] * args.variables, scopes)
    t0 = time.perf_counter()
    plan = graph.compile()
    build_ms = (time.perf_counter() - t0) * 1000
    plan = plan.to(args.device)
    theta = torch.randn(plan.num_configurations, dtype=torch.float32,
                        device=plan.device) * 0.3
    results = {
        "device": args.device, "torch_version": torch.__version__,
        "n_variables": args.variables, "n_factors": args.factors,
        "n_states": plan.num_states, "n_configs": plan.num_configurations,
        "n_message_states": plan.num_message_states,
        "n_configuration_entries": plan.entry_config.numel(),
        "iterations": args.iterations, "build_ms_cpu": build_ms,
    }
    results["lbp"] = benchmark(lambda: lbp(plan, theta, iterations=args.iterations,
                                             return_factor_beliefs=False), theta,
                               repeats=args.repeats, warmup=args.warmup)
    results["mean_field"] = benchmark(lambda: mean_field(plan, theta,
                                                   iterations=args.iterations,
                                                   validate_finite=False), theta,
                                      repeats=args.repeats, warmup=args.warmup)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
