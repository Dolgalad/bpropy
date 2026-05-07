from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from progress.bar import Bar

from bpropy.cli.train.checkpoint import load_checkpoint
from bpropy.cli.train.config import load_run_config, merge_cli_overrides, validate_config
from bpropy.cli.train.data import make_loaders
from bpropy.cli.train.logging import make_run_paths
from bpropy.cli.train.loop import build_model
from bpropy.cli.train.method_registry import METHOD_REGISTRY
from bpropy.cli.train.metrics import _sync
from bpropy.datasets.prefetcher import CUDAPrefetcher
from bpropy.gd.greedy_decimation import greedy_decimation


# Shared full-ILP cache location for all runs.
# You can change this constant once if you want to move the cache.
OPT_CACHE_DIR = Path(
    os.environ.get(
        "BPROPY_NAIR_OPT_CACHE",
        os.path.expanduser("/lustre/fswork/projects/rech/bnq/udj74tc/bpropy_shared_cache/nair_full_ilp_opt"),
    )
)


def build_base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bpropy-test-nair", add_help=False)
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    return p


def build_full_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bpropy-test-nair")

    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")

    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    # run/logging
    p.add_argument("--run-name", dest="run_name", default=argparse.SUPPRESS)
    p.add_argument("--log-dir", dest="log_dir", default=argparse.SUPPRESS)

    # data
    p.add_argument("--data-root", dest="data_root", default=argparse.SUPPRESS)
    p.add_argument("--split", dest="split", default=argparse.SUPPRESS)
    p.add_argument("--val-split", dest="val_split", default=argparse.SUPPRESS)
    p.add_argument("--test-split", dest="test_split", default=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=argparse.SUPPRESS)
    p.add_argument("--num-workers", type=int, dest="num_workers", default=argparse.SUPPRESS)
    p.add_argument("--feature-mode", dest="feature_mode", default=argparse.SUPPRESS)
    p.add_argument("--hetero", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--no-hetero", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--with-lbp", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--with-nmf", action="store_true", default=argparse.SUPPRESS)

    # model/loss
    p.add_argument("--model", dest="model_name", default=argparse.SUPPRESS)
    p.add_argument("--loss", dest="loss_name", default=argparse.SUPPRESS)

    # test-only
    p.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default=argparse.SUPPRESS,
        help="Path to checkpoint (.pt). If omitted, uses <run>/checkpoints/best.pt then last.pt.",
    )
    p.add_argument(
        "--test-data-root",
        dest="test_data_root",
        default=argparse.SUPPRESS,
        help="Override cfg.data.root for test evaluation only.",
    )

    return p


def _apply_test_only_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if hasattr(args, "test_data_root"):
        cfg.setdefault("data", {})["root"] = getattr(args, "test_data_root")
    if hasattr(args, "test_split"):
        cfg.setdefault("data", {})["test_split"] = getattr(args, "test_split")
    return cfg


def _pick_checkpoint_path(cfg: Dict[str, Any], paths, args: argparse.Namespace) -> str:
    if hasattr(args, "checkpoint"):
        return getattr(args, "checkpoint")

    best = os.path.join(paths.ckpt_dir, "best.pt")
    last = os.path.join(paths.ckpt_dir, "last.pt")
    if os.path.exists(best):
        return best
    if os.path.exists(last):
        return last

    raise FileNotFoundError(
        f"No checkpoint found. Looked for {best!r} and {last!r}. "
        "Pass --checkpoint to specify one."
    )


def _ms(x: List[float]) -> tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return 0.0, 0.0
    if a.size == 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _canonicalize_f(F) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(map(int, f))) for f in F))


def _problem_instance_id(pb) -> str:
    """
    Deterministic SCP instance identifier using only SCP structural attributes.
    Uses ne(), F, w.
    """
    try:
        ne = int(pb.ne())
        F = _canonicalize_f(pb.F)
        w = np.asarray(pb.w, dtype=np.float64).reshape(-1)
    except AttributeError as exc:
        raise ValueError("Expected an SCP-like problem with ne(), F and w.") from exc

    h = hashlib.sha1()
    h.update(np.asarray([ne], dtype=np.int64).tobytes())

    for f in F:
        arr = np.asarray(f, dtype=np.int32)
        h.update(arr.tobytes())
        h.update(b"|")

    h.update(w.tobytes())
    return h.hexdigest()


def _opt_cache_file(pb, solver_name: str, timelimit: int) -> Path:
    instance_id = _problem_instance_id(pb)
    return OPT_CACHE_DIR / f"{instance_id}_solver-{solver_name}_tl-{timelimit}.json"


def _solve_full_opt_with_cache(pb, solver_name: str, timelimit: int, dry_run=False) -> tuple[float, float, bool]:
    """
    Cache only the full optimization result:
      returns (objective_value, solve_time_s, from_cache)
    """
    if dry_run:
        sol_opt = pb.solve(solver=solver_name, verbose=False, timelimit=timelimit)
        return

    cache_path = _opt_cache_file(pb, solver_name, timelimit)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")

    print("cache path : ", cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)

        if cache_path.exists():
            with open(cache_path, "r") as f:
                payload = json.load(f)
            return float(payload["objective"]), float(payload["solve_time_s"]), True

        t0 = time.perf_counter()
        sol_opt = pb.solve(solver=solver_name, verbose=False, timelimit=timelimit)
        t1 = time.perf_counter()

        opt_val = float(np.asarray(sol_opt.x).sum())
        solve_time_s = float(t1 - t0)

        payload = {
            "objective": opt_val,
            "solve_time_s": solve_time_s,
            "solver": solver_name,
            "timelimit": timelimit,
            "instance_id": _problem_instance_id(pb),
        }
        _atomic_write_json(cache_path, payload)
        return opt_val, solve_time_s, False


def final_test_eval_nair(
    cfg: Dict[str, Any],
    adapter,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    checkpoint_path: str,
    log_dir: str,
) -> str:
    model.eval()

    run_cfg = cfg.setdefault("run", {})
    base = run_cfg.get("run_name") or run_cfg.get("name") or cfg.get("run_name") or "run"
    seed = int(cfg.get("seed", 0))

    # Distinct suffix so Nair tests cannot overwrite ordinary test runs.
    unique_name = f"{base}_nair_test_seed{seed}"
    run_cfg["run_name"] = unique_name
    run_cfg["name"] = unique_name
    cfg["run_name"] = unique_name

    paths = make_run_paths(cfg)
    log_dir = paths.log_dir

    print("IN TEST logdir =", log_dir)
    print("Shared full-ILP cache =", str(OPT_CACHE_DIR))

    timelimit = 300
    solver_name = "scip"

    props = [0.5, 0.6, 0.7, 0.8, 0.9]
    grd_vals = {p: [] for p in props}
    gd_vals = {p: [] for p in props}
    opt_vals = []
    opt_cache_hits = 0
    opt_cache_misses = 0

    c=0
    limit = 100000

    solver_dry_run = True 

    with torch.no_grad():
        bar = Bar("[Test]          ", max=len(loader))
        for pb, data, y in CUDAPrefetcher(loader, device):
            problem = pb[0]

            _sync(device)

            # Full optimization: cached.
            if solver_dry_run:
                _solve_full_opt_with_cache(
                    problem,
                    solver_name=solver_name,
                    timelimit=10,
                    dry_run = True
                )
                solver_dry_run = False
            opt_val, t_solve_opt, from_cache = _solve_full_opt_with_cache(
                problem,
                solver_name=solver_name,
                timelimit=timelimit,
            )
            if from_cache:
                opt_cache_hits += 1
            else:
                opt_cache_misses += 1

            opt_vals.append([float(opt_val), float(t_solve_opt)])
            print("\tOPT      :", opt_vals[-1])

            # Greedy scores pass
            tg0 = time.perf_counter()
            pred_grd = adapter.forward_fn(model, data)
            scores = adapter.greedy_scores_fn(problem, data, pred_grd)
            _sync(device)
            pred_t = time.perf_counter() - tg0

            for prop in props:
                # Greedy + partial ILP: not cached
                tg2 = time.perf_counter()
                grd_sol, grd_free = problem.greedy(scores=scores, fix_prop=prop).x
                tg3 = time.perf_counter()
                solve_grd_t = tg3 - tg2

                sol_partial = problem.solve(
                    solver=solver_name,
                    verbose=False,
                    partial_solution=(grd_sol, grd_free),
                    timelimit=timelimit,
                )
                sol_partial_val = float(np.asarray(sol_partial.x).sum())
                _sync(device)
                tg4 = time.perf_counter()

                solve_partial_t = tg4 - tg3
                solve_nair_t = solve_grd_t + solve_partial_t

                partial_grd_gap = (sol_partial_val - opt_val) / opt_val
                grd_vals[prop].append([partial_grd_gap, solve_nair_t])
                print(f"\tGRD p={prop}:", grd_vals[prop][-1])

                # Greedy decimation + partial ILP: not cached
                td0 = time.perf_counter()
                pred_dec = adapter.forward_fn(model, data)
                backend = adapter.decimation_backend_fn(problem, data, pred_dec, cfg, device=device)
                padapter = adapter.decimation_adapter_fn(problem, cfg, fix_prop=prop)
                gd_sol, gd_free = greedy_decimation(problem, backend, padapter)
                _sync(device)
                td1 = time.perf_counter()

                solve_gd_t = td1 - td0

                tg3 = time.perf_counter()
                sol_partial_gd = problem.solve(
                    solver=solver_name,
                    verbose=False,
                    partial_solution=(gd_sol, gd_free),
                    timelimit=timelimit,
                )
                sol_partial_gd_val = float(np.asarray(sol_partial_gd.x).sum())
                _sync(device)
                tg4 = time.perf_counter()

                solve_partial_gd_t = tg4 - tg3
                solve_nair_gd_t = solve_gd_t + solve_partial_gd_t

                partial_gd_gap = (sol_partial_gd_val - opt_val) / opt_val
                gd_vals[prop].append([partial_gd_gap, solve_nair_gd_t])
                print(f"\tGD p={prop} :", gd_vals[prop][-1])

            print("-----------------")
            bar.next()
            c += 1
            if c > limit:
                break

        bar.finish()

    summary = {
        "count": len(opt_vals),
        "solver": solver_name,
        "timelimit": timelimit,
        "opt_cache_dir": str(OPT_CACHE_DIR),
        "opt_cache_hits": opt_cache_hits,
        "opt_cache_misses": opt_cache_misses,
        "opt": {
            "mean_objective": float(np.mean([x[0] for x in opt_vals])) if opt_vals else None,
            "std_objective": float(np.std([x[0] for x in opt_vals], ddof=1)) if len(opt_vals) > 1 else 0.0,
            "mean_time_s": float(np.mean([x[1] for x in opt_vals])) if opt_vals else None,
            "std_time_s": float(np.std([x[1] for x in opt_vals], ddof=1)) if len(opt_vals) > 1 else 0.0,
        },
        "grd": {},
        "gd": {},
    }

    for prop in props:
        gaps = [x[0] for x in grd_vals[prop]]
        times = [x[1] for x in grd_vals[prop]]
        summary["grd"][str(prop)] = {
            "gap_mean": _ms(gaps)[0],
            "gap_std": _ms(gaps)[1],
            "time_mean_s": _ms(times)[0],
            "time_std_s": _ms(times)[1],
        }

    for prop in props:
        gaps = [x[0] for x in gd_vals[prop]]
        times = [x[1] for x in gd_vals[prop]]
        summary["gd"][str(prop)] = {
            "gap_mean": _ms(gaps)[0],
            "gap_std": _ms(gaps)[1],
            "time_mean_s": _ms(times)[0],
            "time_std_s": _ms(times)[1],
        }

    run_dir = Path(checkpoint_path).resolve().parent.parent
    out_path = run_dir / "test_eval_nair.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return str(out_path)


def run_test(cfg: Dict[str, Any], checkpoint_path: str) -> Dict[str, Any]:
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = make_run_paths(cfg)
    loaders = make_loaders(cfg, splits=["test"])

    method_name = cfg.get("train", {}).get("method", "var_logits")
    eval_method = cfg.get("eval", {}).get("method") or method_name
    if eval_method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown eval method '{eval_method}'. Known: {list(METHOD_REGISTRY.keys())}")
    adapter_eval = METHOD_REGISTRY[eval_method]
    adapter_eval.loss_name = cfg.get("eval", {}).get("loss-name") or cfg["loss"]["name"]

    model = build_model(cfg, device)
    load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)

    out_path = final_test_eval_nair(
        cfg,
        adapter_eval,
        model,
        loaders.test_loader,
        device,
        checkpoint_path,
        paths.log_dir,
    )

    return {"_test_eval_path": out_path}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    base = build_base_parser()
    base_args, _remaining = base.parse_known_args(argv)

    cfg = load_run_config(base_args.config)
    validate_config(cfg)

    parser = build_full_parser(cfg)
    args = parser.parse_args(argv)

    cfg = merge_cli_overrides(cfg, args)
    cfg = _apply_test_only_overrides(cfg, args)

    if getattr(args, "print_config", False):
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return 0

    paths = make_run_paths(cfg)
    ckpt_path = _pick_checkpoint_path(cfg, paths, args)
    print("checkpoint path =", ckpt_path)

    stats = run_test(cfg, ckpt_path)
    print(stats)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
