# src/bpropy/cli/generate_scp_dataset.py
from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, Dict

import numpy as np

from bpropy.cli.datasets.common import (
    ensure_split_dirs,
    exists_any_artifact,
    iter_indices,
    normalize_splits,
    parse_int_range,
    split_from_index,
    default_instance_stem,
    sample_int,
    time_f,
    write_metadata_once,
)

from bpropy.co.generators.scp import random_scp_items_k, random_scp_items_k_balanced


def _add_common_sharding_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("Sharding / indexing")
    g.add_argument("--seed", type=int, default=0, help="Base seed; instance seed = seed + idx_global.")
    # preferred
    g.add_argument("--shard-id", type=int, default=None, help="Shard id in [0, num_shards). Preferred on Slurm arrays.")
    g.add_argument("--num-shards", type=int, default=None, help="Total number of shards. Preferred on Slurm arrays.")
    # backward compat
    g.add_argument("--index-offset", type=int, default=0, help="Backward-compatible global index offset.")
    g.add_argument("--N", type=int, default=None, help="Backward-compatible: number of instances to generate locally.")


def add_scp_subparser(subparsers) -> None:
    ap = subparsers.add_parser("scp", help="Generate Set Covering Problem datasets")
    ap.add_argument("output_dir")

    ap.add_argument("--n-items", required=True, help="int or 'low:high'")
    ap.add_argument("--n-sets", required=True, help="int or 'low:high'")
    ap.add_argument("--k", required=True, help="int or 'low:high'")
    ap.add_argument("--balanced", action="store_true", help="Use balanced generator if available.")

    ap.add_argument("--total", type=int, required=True, help="Total dataset size (global).")
    ap.add_argument("--splits", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                    help="Optional split weights/percentages (e.g. 80 10 10).")

    # Solving options (your original script had 'both') :contentReference[oaicite:4]{index=4}
    ap.add_argument("--solve", choices=["none", "cbc", "scipy", "both"], default="none")
    ap.add_argument("--relax", action="store_true")

    ap.add_argument("--threshold-bytes", type=int, default=8000)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--export-legacy", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--metadata", action="store_true")

    _add_common_sharding_args(ap)
    ap.set_defaults(_handler=run_scp_generation)


def run_scp_generation(args: argparse.Namespace) -> int:
    if args.total <= 0:
        raise ValueError("--total must be > 0.")

    n_range = parse_int_range(args.n_items)
    m_range = parse_int_range(args.n_sets)
    k_range = parse_int_range(args.k)

    split_counts = normalize_splits(args.splits, args.total) if args.splits else None
    ensure_split_dirs(args.output_dir, split_counts)

    indices, _total, shardplan = iter_indices(
        args.total,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        index_offset=args.index_offset,
        n_local=args.N,
    )

    if args.metadata:
        meta: Dict[str, Any] = {
            "problem": "SCP",
            "total": int(args.total),
            "splits": list(args.splits) if args.splits else None,
            "split_counts": list(split_counts) if split_counts else None,
            "shard": asdict(shardplan) if shardplan else None,
            "params": {
                "n_items": args.n_items,
                "n_sets": args.n_sets,
                "k": args.k,
                "balanced": bool(args.balanced),
            },
            "solve": {"mode": args.solve, "relax": bool(args.relax)},
            "threshold_bytes": int(args.threshold_bytes),
            "seed_base": int(args.seed),
        }
        write_metadata_once(args.output_dir, meta)

    gen_fn = random_scp_items_k_balanced if args.balanced else random_scp_items_k

    for idx_global in indices:
        inst_seed = int(args.seed) + int(idx_global)
        rng = np.random.default_rng(inst_seed)

        n_i = sample_int(rng, n_range)
        m_i = sample_int(rng, m_range)
        k_i = sample_int(rng, k_range)

        split = split_from_index(idx_global, split_counts) if split_counts else None
        stem = default_instance_stem(args.output_dir, idx_global, split, prefix=args.prefix)

        if args.resume and exists_any_artifact(stem):
            print(f"[skip] idx={idx_global} exists: {stem}", flush=True)
            continue

        # 1) instance
        t_gen, pb = time_f(gen_fn, n=n_i, m=m_i, k=k_i, seed=inst_seed)

        # 2) assignments
        t_greedy, x_greedy = time_f(pb.greedy)
        x_cbc = x_scipy = None
        t_cbc = t_scipy = None

        if args.solve in ("cbc", "both"):
            t_cbc, res_ip = time_f(pb.solve, solver="cbc", relax_integrality=False)
            x_cbc = res_ip.x
        if args.solve in ("scipy", "both"):
            t_scipy, res_lp = time_f(pb.solve, solver="scipy", relax_integrality=bool(args.relax))
            x_scipy = res_lp.x

        # 3) save
        pb.save(str(stem), split_large=True, threshold_bytes=int(args.threshold_bytes))
        assignments: Dict[str, Any] = {"greedy": x_greedy}
        if x_cbc is not None:
            assignments["cbc"] = x_cbc
        if x_scipy is not None:
            assignments["scipy_lp" if args.relax else "scipy"] = x_scipy
        pb.save_assignments(str(stem), assignments=assignments, threshold_bytes=int(args.threshold_bytes))

        # 4) optional legacy (mirrors your existing behavior) :contentReference[oaicite:5]{index=5}
        if args.export_legacy:
            np.save(str(stem) + ".npy", pb.adj_matrix())
            best = x_cbc if x_cbc is not None else (x_scipy if x_scipy is not None else x_greedy)
            np.save(str(stem) + "_sol.npy", best.astype(float))

        msg = f"[idx={idx_global}] seed={inst_seed} split={split or '-'} gen={t_gen:.3f}s greedy={t_greedy:.3f}s"
        if t_cbc is not None:
            msg += f" cbc={t_cbc:.3f}s"
        if t_scipy is not None:
            msg += f" scipy={'LP' if args.relax else 'MILP'}={t_scipy:.3f}s"
        print(msg, flush=True)

    return 0

