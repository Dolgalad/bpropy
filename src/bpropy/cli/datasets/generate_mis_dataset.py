# src/bpropy/cli/generate_mis_dataset.py
from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence

import numpy as np

from bpropy.cli.datasets.common import (
    ensure_split_dirs,
    exists_any_artifact,
    iter_indices,
    normalize_splits,
    parse_float_range,
    parse_int_range,
    split_from_index,
    default_instance_stem,
    sample_float,
    sample_int,
    time_f,
    write_metadata_once,
)

from bpropy.co.mis import MIS
from bpropy.co.generators.mis import random_mis_bounded_degree


def _add_common_sharding_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("Sharding / indexing")
    g.add_argument("--seed", type=int, default=0, help="Base seed; instance seed = seed + idx_global.")
    # preferred
    g.add_argument("--shard-id", type=int, default=None, help="Shard id in [0, num_shards). Preferred on Slurm arrays.")
    g.add_argument("--num-shards", type=int, default=None, help="Total number of shards. Preferred on Slurm arrays.")
    # backward compat
    g.add_argument("--index-offset", type=int, default=0, help="Backward-compatible global index offset.")
    g.add_argument("--N", type=int, default=None, help="Backward-compatible: number of instances to generate locally.")


def add_mis_subparser(subparsers) -> None:
    ap = subparsers.add_parser("mis", help="Generate Maximum Independent Set datasets")
    ap.add_argument("output_dir", help="Output directory (will create train/val/test subfolders if --splits is used).")

    # Generator parameter ranges
    ap.add_argument("--nv", required=True, help="Number of vertices: int or 'low:high'.")
    ap.add_argument("--max-deg", required=True, help="Max degree cap: int or 'low:high'.")
    ap.add_argument("--avg-deg", default=None, help="Target average degree: float or 'low:high'. (exclusive with --p)")
    ap.add_argument("--p", default=None, help="ER-like edge prob: float or 'low:high'. (exclusive with --avg-deg)")
    ap.add_argument("--ensure-min-deg", type=int, default=0, help="Best-effort min degree for low-degree nodes (<= max_deg).")

    # Dataset size
    ap.add_argument("--total", type=int, required=True, help="Total dataset size (global).")
    ap.add_argument("--splits", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                    help="Optional split weights or percentages (e.g. 80 10 10).")

    # Solving options (matches your current behavior) :contentReference[oaicite:2]{index=2}
    ap.add_argument("--solve", choices=["none", "cbc", "scipy", "kamis", "all"], default="none",
                    help="Compute assignments in addition to greedy.")
    ap.add_argument("--relax", action="store_true", help="With --solve=scipy/all, solve the LP relaxation.")

    # Saving / housekeeping
    ap.add_argument("--threshold-bytes", type=int, default=8000, help="Arrays >= threshold saved to NPZ sidecar.")
    ap.add_argument("--prefix", default="", help="Filename prefix within each split folder.")
    ap.add_argument("--export-legacy", action="store_true",
                    help="Also write <stem>_sol.npy (cbc > kamis > scipy > greedy).")
    ap.add_argument("--resume", action="store_true", help="Skip instances whose output already exists.")
    ap.add_argument("--metadata", action="store_true", help="Write output_dir/metadata.json once.")

    _add_common_sharding_args(ap)

    ap.set_defaults(_handler=run_mis_generation)


def run_mis_generation(args: argparse.Namespace) -> int:
    # validate density args
    if args.avg_deg is not None and args.p is not None:
        raise ValueError("Provide at most one of --avg-deg or --p.")
    if args.total <= 0:
        raise ValueError("--total must be > 0.")

    nv_range = parse_int_range(args.nv)
    md_range = parse_int_range(args.max_deg)
    avg_deg_range = parse_float_range(args.avg_deg)
    p_range = parse_float_range(args.p)

    split_counts = normalize_splits(args.splits, args.total) if args.splits else None
    ensure_split_dirs(args.output_dir, split_counts)

    # indices to generate
    indices, total_global, shardplan = iter_indices(
        args.total,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        index_offset=args.index_offset,
        n_local=args.N,
    )

    if args.metadata:
        meta: Dict[str, Any] = {
            "problem": "MIS",
            "total": int(args.total),
            "splits": list(args.splits) if args.splits else None,
            "split_counts": list(split_counts) if split_counts else None,
            "shard": asdict(shardplan) if shardplan else None,
            "params": {
                "nv": args.nv,
                "max_deg": args.max_deg,
                "avg_deg": args.avg_deg,
                "p": args.p,
                "ensure_min_deg": int(args.ensure_min_deg),
            },
            "solve": {"mode": args.solve, "relax": bool(args.relax)},
            "threshold_bytes": int(args.threshold_bytes),
            "seed_base": int(args.seed),
        }
        write_metadata_once(args.output_dir, meta)

    for idx_global in indices:
        inst_seed = int(args.seed) + int(idx_global)
        rng = np.random.default_rng(inst_seed)

        nv_i = sample_int(rng, nv_range)
        max_deg_i = sample_int(rng, md_range)
        avg_deg_i = sample_float(rng, avg_deg_range) if avg_deg_range is not None else None
        p_i = sample_float(rng, p_range) if p_range is not None else None

        split = split_from_index(idx_global, split_counts) if split_counts else None
        stem = default_instance_stem(args.output_dir, idx_global, split, prefix=args.prefix)

        if args.resume and exists_any_artifact(stem):
            print(f"[skip] idx={idx_global} exists: {stem}", flush=True)
            continue

        # 1) generate
        t_gen0 = time_f(
            random_mis_bounded_degree,
            nv=nv_i,
            max_deg=max_deg_i,
            target_avg_deg=avg_deg_i,
            p=p_i,
            ensure_min_deg=min(int(args.ensure_min_deg), int(max_deg_i)),
            seed=inst_seed,
        )
        t_gen, pb = t_gen0

        # 2) assignments
        t_greedy, x_greedy = time_f(pb.greedy)
        x_cbc = x_scipy = x_kamis = None
        t_cbc = t_scipy = t_kamis = None

        if args.solve in ("cbc", "all"):
            t_cbc, res_ip = time_f(pb.solve, solver="cbc", relax_integrality=False)
            x_cbc = res_ip.x

        if args.solve in ("scipy", "all"):
            t_scipy, res_lp = time_f(pb.solve, solver="scipy", relax_integrality=bool(args.relax))
            x_scipy = res_lp.x

        if args.solve in ("kamis", "all"):
            t_kamis, res_k = time_f(pb.solve, solver="kamis")
            x_kamis = res_k.x

        # 3) save
        pb.save(str(stem), split_large=True, threshold_bytes=int(args.threshold_bytes))
        assignments: Dict[str, Any] = {"greedy": x_greedy.x}
        if x_cbc is not None:
            assignments["cbc"] = x_cbc
        if x_kamis is not None:
            assignments["kamis"] = x_kamis
        if x_scipy is not None:
            assignments["scipy_lp" if args.relax else "scipy"] = x_scipy

        pb.save_assignments(str(stem), assignments=assignments, threshold_bytes=int(args.threshold_bytes))

        if args.export_legacy:
            # prefer CBC > KAMIS > SciPy > greedy (reasonable for MIS)
            best = x_cbc if x_cbc is not None else (x_kamis if x_kamis is not None else (x_scipy if x_scipy is not None else x_greedy))
            np.save(str(stem) + "_sol.npy", best.astype(float))

        # log
        msg = (
            f"[idx={idx_global}] seed={inst_seed} split={split or '-'} "
            f"nv={nv_i} max_deg={max_deg_i} "
            f"{'avg_deg=' + str(round(avg_deg_i, 3)) if avg_deg_i is not None else ''}"
            f"{' p=' + str(round(p_i, 6)) if p_i is not None else ''} "
            f"gen={t_gen:.3f}s greedy={t_greedy:.3f}s"
        )
        if t_cbc is not None:
            msg += f" cbc={t_cbc:.3f}s"
        if t_kamis is not None:
            msg += f" kamis={t_kamis:.3f}s"
        if t_scipy is not None:
            msg += f" scipy={'LP' if args.relax else 'MILP'}={t_scipy:.3f}s"
        print(msg, flush=True)

    return 0

