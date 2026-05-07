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

import matplotlib.pyplot as plt


OPT_CACHE_DIR = Path(
    os.environ.get(
        "BPROPY_NAIR_OPT_CACHE",
        os.path.expanduser(
            "/lustre/fswork/projects/rech/bnq/udj74tc/bpropy_shared_cache/nair_full_ilp_opt"
        ),
    )
)


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

    p.add_argument("--max-test-instances", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to the JSON file where per-instance run data will be stored.",
    )
    p.add_argument(
        "--decodings",
        type=str,
        nargs="+",
        default=["EGG", "GD"],
        choices=sorted(VALID_DECODING),
        help="Which decoding modes to run. Example: --decodings EGG GD or --decodings GD",
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
    output_json: str,
    seed: int,
    decodings: List[str],
) -> str:
    model.eval()

    output_path = Path(output_json)

    results: List[Dict[str, Any]] = []
    c = 0
    limit = max_instances

    payload = {
        "summary": {
            "count": 0,
            "seed": int(seed),
            "checkpoint_path": checkpoint_path,
            "config": cfg.get("config", None),
            "completed": False
        },
        "instances": [],
    }

    #_write_json(output_path, payload)

    egg_step_scores = []
    gd_step_scores = []

    with torch.no_grad():
        bar = Bar("[Test]", max=len(loader))

        for idx, (pb, data, y) in enumerate(CUDAPrefetcher(loader, device)):
            problem = pb[0]
            stem = Path(loader.dataset.stems[idx]).name

            t_pre0 = time.perf_counter()
            pred = adapter.forward_fn(model, data)
            _sync(device)
            t_fwd1 = time.perf_counter()

            _ensure_greedy_decimation_cfg(cfg)

            entry = {}

            # EGG
            if "EGG" in decodings:
                cfg["eval"]["greedy_decimation"]["args"]["run_every"] = 1000000
                backend = adapter.decimation_backend_fn(problem, data, pred, cfg, device=device)
                padapter = adapter.decimation_adapter_fn(problem, cfg)
                t_pre1 = time.perf_counter()

                _, egg_log = greedy_decimation(problem, backend, padapter, return_log=True)
                _sync(device)
                t_dec1 = time.perf_counter()

                egg_step_scores.append([entry["score"] for entry in egg_log["info"]])
                _egg_step_scores = [entry["score"] for entry in egg_log["info"]]
                is_monotone = all(_egg_step_scores[i] <= _egg_step_scores[i+1] for i in range(len(_egg_step_scores)-1))
                print("EGG monotone = ", is_monotone)
                if not is_monotone:
                    tol = 1e-10
                    violations = [
                        (i, _egg_step_scores[i], _egg_step_scores[i+1])
                            for i in range(len(_egg_step_scores)-1)
                            if _egg_step_scores[i+1] + tol < _egg_step_scores[i]
                        ]
                    print(violations)


                entry["egg"] = egg_log

            # GD
            if "GD" in decodings:
                cfg["eval"]["greedy_decimation"]["args"]["run_every"] = 1
                backend = adapter.decimation_backend_fn(problem, data, pred, cfg, device=device)
                padapter = adapter.decimation_adapter_fn(problem, cfg)
                t_pre1 = time.perf_counter()

                _, gd_log = greedy_decimation(problem, backend, padapter, return_log=True)
                _sync(device)
                t_dec1 = time.perf_counter()
                gd_step_scores.append([entry["score"] for entry in gd_log["info"]])
                is_monotone = all(gd_step_scores[i] <= gd_step_scores[i+1] for i in range(len(gd_step_scores)-1))
                print("GD monotone = ", is_monotone)

                entry["gd"] = gd_log

            results.append(entry)

            payload["instances"] = results
            payload["summary"]["count"] = len(results)
            _write_json(output_path, payload)

            bar.next()
            c += 1
            if limit is not None and c >= limit:
                break

        bar.finish()

    ## plot for debugging purposes
    #egg_max_len = max(len(a) for a in egg_step_scores)
    #gd_max_len = max(len(a) for a in gd_step_scores)

    #egg_padded_scores = np.array([
    #        np.pad(a, (0, egg_max_len - len(a)), constant_values=np.nan)
    #            for a in egg_step_scores
    #            ])
    #gd_padded_scores = np.array([
    #        np.pad(a, (0, gd_max_len - len(a)), constant_values=np.nan)
    #            for a in gd_step_scores
    #            ])

    #fig = plt.figure()
    #egg_mean = np.nanmean(egg_padded_scores, axis=0)
    #egg_std = np.nanstd(egg_padded_scores, axis=0)
    #x = np.arange(len(egg_mean))

    #plt.plot(egg_mean, label="EGG")
    #plt.fill_between(
    #    x,
    #    egg_mean - egg_std,
    #    egg_mean + egg_std,
    #    alpha=0.2
    #    )
    #gd_mean = np.nanmean(gd_padded_scores, axis=0)
    #gd_std = np.nanstd(gd_padded_scores, axis=0)
    #x = np.arange(len(gd_mean))
    #plt.plot(gd_mean, label="GD")
    #plt.fill_between(
    #    x,
    #    gd_mean - gd_std,
    #    gd_mean + gd_std,
    #    alpha=0.2
    #    )


    #plt.legend()
    #plt.savefig("dummy.png")
    
    

    payload["summary"]["completed"] = True
    _write_json(output_path, payload)
    return str(output_path)



def run_test(
    cfg: Dict[str, Any],
    checkpoint_path: str,
    *,
    output_json: str,
    decodings: List[str],
    seed: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        output_json=output_json,
        seed=seed,
        decodings=decodings,
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
        output_json=args.output_json,
        seed=args.seed,
        decodings=args.decodings,
    )
    print(stats)


if __name__ == "__main__":
    main()
