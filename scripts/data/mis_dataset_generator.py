#!/usr/bin/env python3
"""
Generate reproducible MIS datasets with optional train/val/test splitting
and save assignment vectors alongside each problem (JSON/NPZ via SerializableLPMixin).

Parameters that can be fixed or ranged (inclusive):
  - nv: number of vertices ("N")
  - max_deg: hard degree cap
  - avg_deg or p: either target average degree or ER-like edge probability

Examples:
  # 100 instances with nv in [200,300], max_deg in [6,10], avg_deg in [3.0,5.0]
  python mis_dataset_generator.py out 200:300 6:10 100 --avg-deg 3.0:5.0 --seed 123

  # Fixed nv=400, max_deg=8, p in [0.01, 0.03], save CBC & LP assignments, split 80/10/10
  python mis_dataset_generator.py out 400 8 300 --p 0.01:0.03 --solve both --relax --splits 80 10 10
"""
import argparse, os, sys, time, json
sys.path.append("../..")
from typing import Optional, Tuple, List
import numpy as np

# Make local package importable if running from repo checkout
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Adjust these imports to your project structure if different:
from bpropy.co.mis import MIS                  # your MIS problem class
from bpropy.graphs.random_graphs import random_mis_bounded_degree  # the bounded-degree generator


# ---------------- utilities ----------------
def time_f(func, *args, **kwargs):
    t0 = time.time(); r = func(*args, **kwargs); return time.time() - t0, r

def parse_range_or_scalar_int(s: str) -> Tuple[int, int]:
    """Return (low, high) inclusive for 'int' or 'low:high'."""
    if ":" in s:
        a, b = s.split(":")
        return int(a), int(b)
    v = int(s)
    return v, v

def parse_range_or_scalar_float(s: Optional[str]) -> Optional[Tuple[float, float]]:
    """Return (low, high) inclusive for 'float' or 'low:high', or None if s is None."""
    if s is None:
        return None
    if ":" in s:
        a, b = s.split(":")
        return float(a), float(b)
    v = float(s)
    return v, v

def sample_int(rng: np.random.Generator, bounds: Tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))

def sample_float(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    if lo == hi:
        return float(lo)
    # inclusive-ish: np.random.Generator.uniform is [low, high)
    x = rng.uniform(lo, np.nextafter(hi, np.float64(np.inf)))
    return float(x)

def normalize_splits(splits: Optional[List[int]], N: int) -> Optional[Tuple[int, int, int]]:
    if splits is None:
        return None
    if len(splits) != 3 or sum(splits) <= 0:
        raise ValueError("--splits expects three non-negative integers (e.g., 80 10 10)")
    p = np.array(splits, dtype=float); p /= p.sum()
    counts = np.floor(p * N + 1e-9).astype(int)
    for i in range(N - counts.sum()):
        counts[i % 3] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])

def split_selector(idx_global: int, N_total: int, counts: Tuple[int, int, int]) -> str:
    n_tr, n_va, n_te = counts
    if idx_global < n_tr: return "train"
    if idx_global < n_tr + n_va: return "val"
    return "test"

def ensure_dirs(base: str, split_counts: Optional[Tuple[int, int, int]]):
    os.makedirs(base, exist_ok=True)
    if split_counts is not None:
        for sub in ("train", "val", "test"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="MIS dataset generation (reproducible; with splits & assignments)")
    ap.add_argument("output_dir")

    # Allow ranges for nv & max_deg
    ap.add_argument("nv", help="int or 'low:high' (number of vertices)")
    ap.add_argument("max_deg", help="int or 'low:high' (degree cap)")

    ap.add_argument("N", type=int, help="number of instances")

    # Reproducibility / sharding
    ap.add_argument("--seed", type=int, default=0, help="base seed; per-instance seed = seed + global_index")
    ap.add_argument("--index-offset", type=int, default=0, help="global index offset (for SLURM arrays)")

    # Splits
    ap.add_argument("--splits", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                    help="optional split percentages; e.g. --splits 80 10 10")

    # Graph density control: choose ONE of --avg-deg or --p (each can be scalar or range)
    ap.add_argument("--avg-deg", type=str, default=None, help="target average degree (float or 'low:high')")
    ap.add_argument("--p", type=str, default=None, help="ER-like edge prob (float or 'low:high')")
    ap.add_argument("--ensure-min-deg", type=int, default=0,
                    help="best-effort minimum degree for low-degree nodes (≤ max_deg)")

    # Which assignments to compute
    ap.add_argument("--solve", choices=["none", "cbc", "scipy", "all", "kamis"], default="none",
                    help="additionally compute CBC and/or SciPy assignments")
    ap.add_argument("--relax", action="store_true",
                    help="with --solve=scipy, solve the LP relaxation")

    # Saving knobs
    ap.add_argument("--threshold-bytes", type=int, default=8000,
                    help="arrays ≥ threshold go to NPZ sidecar")
    ap.add_argument("--export-legacy", action="store_true",
                    help="also write <stem>_sol.npy (cbc > scipy > greedy)")
    ap.add_argument("--prefix", default="", help="filename prefix within split folder")

    args = ap.parse_args()

    # Validate exclusive choice avg-deg vs p
    if (args.avg_deg is None) == (args.p is None):
        # either both None or both provided → allow both None (we'll default), forbid both provided
        if args.avg_deg is not None and args.p is not None:
            raise ValueError("Provide at most one of --avg-deg or --p")

    nv_range = parse_range_or_scalar_int(args.nv)
    md_range = parse_range_or_scalar_int(args.max_deg)
    avg_deg_range = parse_range_or_scalar_float(args.avg_deg)
    p_range = parse_range_or_scalar_float(args.p)

    split_counts = normalize_splits(args.splits, args.N)
    ensure_dirs(args.output_dir, split_counts)
    N_total = args.N

    for i_local in range(args.N):
        idx_global = args.index_offset + i_local
        out_dir = (args.output_dir if split_counts is None
                   else os.path.join(args.output_dir,
                                     split_selector(idx_global % N_total, N_total, split_counts)))
        stem = os.path.join(out_dir, f"{args.prefix}{idx_global}")
        inst_seed = (args.seed or 0) + idx_global
        rng = np.random.default_rng(inst_seed)

        # Sample per-instance parameters
        nv_i = sample_int(rng, nv_range)
        max_deg_i = sample_int(rng, md_range)
        avg_deg_i = sample_float(rng, avg_deg_range) if avg_deg_range is not None else None
        p_i = sample_float(rng, p_range) if p_range is not None else None

        # Generate MIS instance with a hard degree cap
        # NOTE: random_mis_bounded_degree signature:
        #   (nv, max_deg, target_avg_deg=None, p=None, ensure_min_deg=0, seed=None, max_trials_factor=20)
        t_gen = time.time()
        nv_out, edges, w = random_mis_bounded_degree(
            nv=nv_i,
            max_deg=max_deg_i,
            target_avg_deg=avg_deg_i,
            p=p_i,
            ensure_min_deg=min(args.ensure_min_deg, max_deg_i),
            seed=inst_seed,
        )
        # Build MIS problem
        pb = MIS.from_edges(nv_out, edges, weights=w)
        t_gen = time.time() - t_gen

        # Assignments
        t_greedy, x_greedy = time_f(pb.greedy)
        x_cbc = x_scipy = None
        if args.solve in ("cbc", "all"):
            _, res_ip = time_f(pb.solve, solver="cbc", relax_integrality=False)  # maximization
            x_cbc = res_ip.x
        if args.solve in ("scipy", "all"):
            _, res_lp = time_f(pb.solve, solver="scipy", relax_integrality=args.relax)
            x_scipy = res_lp.x

        if args.solve in ("kamis", "all"):
            _, res_kamis = time_f(pb.solve, solver="kamis")
            x_kamis = res_kamis.x

        # Save problem + assignments (mixin handles JSON/NPZ)
        pb.save(stem, split_large=True, threshold_bytes=args.threshold_bytes)
        pb.save_assignments(
            stem,
            assignments={
                "greedy": x_greedy,
                **({"cbc": x_cbc} if x_cbc is not None else {}),
                **({"kamis": x_kamis} if x_kamis is not None else {}),
                **({ "scipy_lp" if args.relax else "scipy": x_scipy } if x_scipy is not None else {}),
            },
            threshold_bytes=args.threshold_bytes,
        )

        # Optional legacy
        if args.export_legacy:
            # prefer CBC > SciPy > greedy
            np.save(stem + "_sol.npy",
                    (x_cbc if x_cbc is not None else (x_scipy if x_scipy is not None else x_greedy)).astype(float))

        # Progress
        msg = (f"[idx={idx_global}] seed={inst_seed} nv={nv_i} max_deg={max_deg_i} "
               f"{'avg_deg='+str(round(avg_deg_i,2)) if avg_deg_i is not None else ''}"
               f"{' p='+str(round(p_i,4)) if p_i is not None else ''} "
               f"gen={t_gen:.3f}s greedy={t_greedy:.3f}s")
        print(msg, flush=True)


if __name__ == "__main__":
    main()

