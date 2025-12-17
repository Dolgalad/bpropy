#!/usr/bin/env python3
import argparse, os, sys, time, json
sys.path.append("../..")
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

# repo root import
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bpropy.co.scp import SCP, random_scp_items_k, random_scp_items_k_balanced

def time_f(func, *args, **kwargs):
    t0 = time.time(); r = func(*args, **kwargs); return time.time() - t0, r

def make_instance(n_items: int, n_sets: int, k: int, seed: int, balanced: bool) -> SCP:
    return (random_scp_items_k_balanced if balanced else random_scp_items_k)(
        n=n_items, m=n_sets, k=k, seed=seed
    )

def normalize_splits(splits: Optional[List[int]], N: int) -> Optional[Tuple[int,int,int]]:
    if splits is None: return None
    p = np.array(splits, dtype=float); p = p / p.sum()
    counts = np.floor(p * N + 1e-9).astype(int)
    for i in range(N - counts.sum()): counts[i % 3] += 1
    return tuple(map(int, counts))

def split_selector(idx_global: int, N_total: int, counts: Tuple[int,int,int]) -> str:
    n_tr, n_va, n_te = counts
    if idx_global < n_tr: return "train"
    if idx_global < n_tr + n_va: return "val"
    return "test"

def ensure_dirs(base: str, split_counts: Optional[Tuple[int,int,int]]):
    os.makedirs(base, exist_ok=True)
    if split_counts:
        for sub in ("train","val","test"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)

def parse_range_or_int(s: str) -> Tuple[int,int]:
    """
    Parse a string as either 'int' or 'low:high'.
    Returns a pair (low, high), inclusive.
    """
    if ":" in s:
        a, b = s.split(":")
        return int(a), int(b)
    else:
        v = int(s)
        return v, v

def sample_from_range(rng: np.random.Generator, bounds: Tuple[int,int]) -> int:
    low, high = bounds
    return int(rng.integers(low, high+1))

def main():
    ap = argparse.ArgumentParser(description="SCP dataset generation with assignments (reproducible)")
    ap.add_argument("output_dir")
    ap.add_argument("n_items", help="int or 'low:high'")
    ap.add_argument("n_sets", help="int or 'low:high'")
    ap.add_argument("k", help="int or 'low:high'")
    ap.add_argument("N", type=int)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index-offset", type=int, default=0)
    ap.add_argument("--splits", type=int, nargs=3, metavar=("TRAIN","VAL","TEST"))
    ap.add_argument("--balanced", action="store_true")

    ap.add_argument("--solve", choices=["none","cbc","scipy","both"], default="none")
    ap.add_argument("--relax", action="store_true")

    ap.add_argument("--threshold-bytes", type=int, default=8000)
    ap.add_argument("--export-legacy", action="store_true")
    ap.add_argument("--prefix", default="")

    args = ap.parse_args()

    n_range = parse_range_or_int(args.n_items)
    m_range = parse_range_or_int(args.n_sets)
    k_range = parse_range_or_int(args.k)

    split_counts = normalize_splits(args.splits, args.N)
    ensure_dirs(args.output_dir, split_counts)
    N_total = args.N

    for i_local in range(args.N):
        idx_global = args.index_offset + i_local
        out_dir = (args.output_dir if not split_counts
                   else os.path.join(args.output_dir, split_selector(idx_global % N_total, N_total, split_counts)))
        stem = os.path.join(out_dir, f"{args.prefix}{idx_global}")
        inst_seed = (args.seed or 0) + idx_global

        rng = np.random.default_rng(inst_seed)

        # draw actual values for this instance
        n_i = sample_from_range(rng, n_range)
        m_i = sample_from_range(rng, m_range)
        k_i = sample_from_range(rng, k_range)

        # 1) instance
        t_gen = time.time()
        pb = make_instance(n_i, m_i, k_i, inst_seed, args.balanced)
        t_gen = time.time() - t_gen

        # 2) assignments
        t_greedy, x_greedy = time_f(pb.greedy)
        x_cbc = x_scipy = None
        if args.solve in ("cbc","both"):
            t_cbc, res_ip = time_f(pb.solve, solver="cbc", relax_integrality=False)
            x_cbc = res_ip.x
        if args.solve in ("scipy","both"):
            t_scipy, res_lp = time_f(pb.solve, solver="scipy", relax_integrality=args.relax)
            x_scipy = res_lp.x

        # 3) save problem + assignments
        pb.save(stem, split_large=True, threshold_bytes=args.threshold_bytes)
        # NEW: single call handles all assignment vectors
        pb.save_assignments(
            stem,
            assignments={
                "greedy": x_greedy,
                **({"cbc": x_cbc} if x_cbc is not None else {}),
                **({ "scipy_lp" if args.relax else "scipy": x_scipy } if x_scipy is not None else {}),
            },
            threshold_bytes=args.threshold_bytes,
        )

        # 4) optional legacy files
        if args.export_legacy:
            np.save(stem + ".npy", pb.adj_matrix())
            # prefer CBC > SciPy > greedy
            np.save(stem + "_sol.npy",
                    (x_cbc if x_cbc is not None else (x_scipy if x_scipy is not None else x_greedy)).astype(float))

        msg = f"[idx={idx_global}] seed={inst_seed} gen={t_gen:.3f}s"
        msg += f" greedy={t_greedy:.3f}s"
        if x_cbc is not None:   msg += f" cbc={t_cbc:.3f}s"
        if x_scipy is not None: msg += f" scipy={'LP' if args.relax else 'MILP'} {t_scipy:.3f}s"
        print(msg, flush=True)

if __name__ == "__main__":
    main()

