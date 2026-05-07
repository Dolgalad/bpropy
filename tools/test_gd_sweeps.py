from __future__ import annotations

import argparse
import json
import os
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


VALID_DECODING = {"GD", "EGG"}


def build_base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bpropy-test-nair", add_help=False)
    p.add_argument("--config", required=True)
    p.add_argument("--print-config", action="store_true")
    return p


def build_full_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bpropy-test-nair")

    p.add_argument("--config", required=True)
    p.add_argument("--print-config", action="store_true")
    p.add_argument("--checkpoint", default=argparse.SUPPRESS)
    p.add_argument("--dataset", required=False, default=None)

    p.add_argument("--max-test-instances", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fix-prop",
        type=float,
        required=True,
        help="Proportion of variables to fix during decimation.",
    )
    p.add_argument(
        "--decoding",
        type=str,
        required=True,
        choices=sorted(VALID_DECODING),
        help="Decoding strategy: GD or EGG.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to the JSON file where per-instance run data will be stored.",
    )
    p.add_argument(
        "--solver",
        type=str,
        default="scip",
        help="Solver used for the reduced problem.",
    )
    p.add_argument(
        "--timelimit",
        type=int,
        default=300,
        help="Solver time limit in seconds for the reduced problem.",
    )

    return p


def _pick_checkpoint_path(cfg: Dict[str, Any], paths, args: argparse.Namespace) -> str:
    if hasattr(args, "checkpoint"):
        return getattr(args, "checkpoint")

    best = os.path.join(paths.ckpt_dir, "best.pt")
    last = os.path.join(paths.ckpt_dir, "last.pt")

    if os.path.exists(best):
        return best
    if os.path.exists(last):
        return last

    raise FileNotFoundError("No checkpoint found.")


def _problem_stem(problem, fallback_idx: int) -> str:
    candidates = [
        getattr(problem, "stem", None),
        getattr(problem, "name", None),
        getattr(problem, "instance_name", None),
        getattr(problem, "id", None),
    ]

    path_like = getattr(problem, "path", None) or getattr(problem, "filepath", None)
    if path_like is not None:
        try:
            candidates.append(Path(path_like).stem)
        except Exception:
            pass

    for cand in candidates:
        if cand is None:
            continue
        text = str(cand).strip()
        if text:
            return Path(text).stem

    return f"instance_{fallback_idx:06d}"


def _decoding_run_every(decoding: str) -> int:
    if decoding == "EGG":
        return 1000000
    if decoding == "GD":
        return 1
    raise ValueError(f"Unknown decoding strategy: {decoding}")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)

def _ensure_greedy_decimation_cfg(cfg: Dict[str, Any]) -> None:
    cfg.setdefault("eval", {})
    cfg["eval"].setdefault("greedy_decimation", {})
    cfg["eval"]["greedy_decimation"].setdefault("args", {})
    cfg["eval"]["greedy_decimation"].setdefault("adapter",{"pick_mode":"min_entropy","value_mode":"argmax"})

def final_test_eval_nair(
    cfg: Dict[str, Any],
    adapter,
    model,
    loader,
    device,
    checkpoint_path: str,
    max_instances: Optional[int],
    fix_prop: float,
    decoding: str,
    output_json: str,
    solver_name: str,
    timelimit: int,
    seed: int,
) -> str:
    model.eval()

    output_path = Path(output_json)
    run_every = _decoding_run_every(decoding)

    results: List[Dict[str, Any]] = []
    c = 0
    limit = max_instances

    payload = {
        "summary": {
            "count": 0,
            "seed": int(seed),
            "fix_prop": float(fix_prop),
            "decoding": decoding,
            "solver": solver_name,
            "timelimit_s": int(timelimit),
            "checkpoint_path": checkpoint_path,
            "config": cfg.get("config", None),
            "completed": False
        },
        "instances": [],
    }

    _write_json(output_path, payload)


    with torch.no_grad():
        bar = Bar("[Test]", max=len(loader))

        for idx, (pb, data, y) in enumerate(CUDAPrefetcher(loader, device)):
            problem = pb[0]
            stem = Path(loader.dataset.stems[idx]).name


            solution_values = []
            run_every_values = [50,20,10,5,1]

            for run_every in [50, 20, 10, 5, 1]:
                t_pre0 = time.perf_counter()
                pred = adapter.forward_fn(model, data)
                _sync(device)
                t_fwd1 = time.perf_counter()
    
                _ensure_greedy_decimation_cfg(cfg)

                cfg["eval"]["greedy_decimation"]["args"]["run_every"] = run_every
                backend = adapter.decimation_backend_fn(problem, data, pred, cfg, device=device)
                padapter = adapter.decimation_adapter_fn(problem, cfg, fix_prop=fix_prop)
                t_pre1 = time.perf_counter()

                #sol, free = greedy_decimation(problem, backend, padapter)
                sol = greedy_decimation(problem, backend, padapter)

                _sync(device)
                t_dec1 = time.perf_counter()

                #partial = problem.solve(
                #    solver=solver_name,
                #    verbose=False,
                #    partial_solution=(sol, free),
                #    timelimit=timelimit,
                #)
                #t_sol1 = time.perf_counter()

                #try:
                #    solution_value = float(np.asarray(partial.x).sum())
                #except Exception:
                #    solution_value = float("nan")

                solution_values.append(sol.sum())
                print("run_every = ", run_every , " val = ", sol.sum(), " t = ", t_dec1 - t_pre1)
            import matplotlib.pyplot as plt

            os.makedirs("test_gd_sweeps", exist_ok=True)

            fig = plt.figure()
            plt.plot(run_every_values, solution_values)
            plt.savefig(os.path.join("test_gd_sweeps/{idx}.png"))
            plt.close(fig)

            bar.next()
            c += 1
            if limit is not None and c >= limit:
                break

        bar.finish()

    return str(output_path)



def run_test(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    *,
    fix_prop: float,
    decoding: str,
    output_json: str,
    solver_name: str,
    timelimit: int,
    seed: int,
    dataset_root: Optional[str]  = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if  dataset_root:
        cfg["data"]["root"] = dataset_root
    loaders = make_loaders(cfg, splits=["test"])

    adapter = METHOD_REGISTRY[cfg["train"]["method"]]
    model = build_model(cfg, device)

    load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)

    return final_test_eval_nair(
        cfg,
        adapter,
        model,
        loaders.test_loader,
        device,
        checkpoint_path,
        max_instances=cfg.get("max_test_instances"),
        fix_prop=fix_prop,
        decoding=decoding,
        output_json=output_json,
        solver_name=solver_name,
        timelimit=timelimit,
        seed=seed,
    )



def main():
    base = build_base_parser()
    args, _ = base.parse_known_args()

    cfg = load_run_config(args.config)
    validate_config(cfg)

    parser = build_full_parser(cfg)
    args = parser.parse_args()

    cfg = merge_cli_overrides(cfg, args)

    if hasattr(args, "max_test_instances"):
        cfg["max_test_instances"] = args.max_test_instances

    paths = make_run_paths(cfg)
    ckpt = _pick_checkpoint_path(cfg, paths, args)

    stats = run_test(
        cfg,
        ckpt,
        fix_prop=args.fix_prop,
        decoding=args.decoding,
        output_json=args.output_json,
        solver_name=args.solver,
        timelimit=args.timelimit,
        seed=args.seed,
        dataset_root=args.dataset,
    )
    print(stats)


if __name__ == "__main__":
    main()
