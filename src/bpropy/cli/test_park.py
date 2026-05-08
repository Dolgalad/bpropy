from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from bpropy.cli.train.config import load_run_config, merge_cli_overrides, validate_config
from bpropy.cli.park.loop import final_test_eval
from bpropy.cli.train.data import make_loaders
from bpropy.cli.train.logging import make_run_paths
from bpropy.cli.train.method_registry import METHOD_REGISTRY


def build_base_parser() -> argparse.ArgumentParser:
    """
    Stage-1 parser: ONLY --config (and maybe --print-config).
    We parse this first to load JSON config before building full parser.
    Mirrors bpropy-train. :contentReference[oaicite:1]{index=1}
    """
    p = argparse.ArgumentParser(prog="bpropy-test", add_help=False)
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    return p


def build_full_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    """
    Stage-2 parser: same override flags as bpropy-train, plus test-only flags.
    Uses argparse.SUPPRESS so only explicitly-provided args become overrides. :contentReference[oaicite:2]{index=2}
    """
    p = argparse.ArgumentParser(prog="bpropy-test")

    p.add_argument(
        "--output-json",
        dest="output_json",
        default=argparse.SUPPRESS,
        help="Path to the JSON file where test evaluation results will be written.",
    )

    # still include these so `-h` shows them and they can be re-parsed
    p.add_argument("--config", required=True, help="Path to JSON config.")
    p.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")

    # --- common overrides (same as train) ---
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    # run/logging
    p.add_argument("--run-name", dest="run_name", default=argparse.SUPPRESS)
    p.add_argument("--log-dir", dest="log_dir", default=argparse.SUPPRESS)

    # data
    p.add_argument("--data-root", dest="data_root", default=argparse.SUPPRESS)
    p.add_argument("--split", dest="split", default=argparse.SUPPRESS)
    p.add_argument("--val-split", dest="val_split", default=argparse.SUPPRESS)
    p.add_argument("--test-split", dest="test_split", default=argparse.SUPPRESS)
    p.add_argument("--cache-dir", dest="cache_dir", default=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, dest="batch_size", default=argparse.SUPPRESS)
    p.add_argument("--num-workers", type=int, dest="num_workers", default=argparse.SUPPRESS)
    p.add_argument("--feature-mode", dest="feature_mode", default=argparse.SUPPRESS)
    p.add_argument("--hetero", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--no-hetero", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--with-lbp", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--with-nmf", action="store_true", default=argparse.SUPPRESS)


    # -------- test-only overrides --------
    p.add_argument(
        "--test-data-root",
        dest="test_data_root",
        default=argparse.SUPPRESS,
        help="Override cfg.data.root for test evaluation only.",
    )

     # -------- Park / canonical MRF overrides --------
    p.add_argument("--M", type=float, default=argparse.SUPPRESS,
                   help="Canonical MRF penalty magnitude. Infeasible assignments get theta=-M.")
    p.add_argument("--max-iter", type=int, default=argparse.SUPPRESS,
                   help="Default BP/NMF max_iter for both EGG and GD unless overridden.")
    p.add_argument("--damping", type=float, default=argparse.SUPPRESS,
                   help="Default BP/NMF damping for both EGG and GD unless overridden.")
    p.add_argument("--tol", type=float, default=argparse.SUPPRESS,
                   help="Canonical parameter tolerance and inference tolerance.")

    p.add_argument("--inference", choices=["loopy_belief_propagation", "naive_mean_field"],
                   default=argparse.SUPPRESS)

    p.add_argument("--egg-max-iter", type=int, default=argparse.SUPPRESS)
    p.add_argument("--egg-damping", type=float, default=argparse.SUPPRESS)
    p.add_argument("--gd-max-iter", type=int, default=argparse.SUPPRESS)
    p.add_argument("--gd-damping", type=float, default=argparse.SUPPRESS)
    p.add_argument("--gd-run-every", type=int, default=argparse.SUPPRESS)

    p.add_argument("--unary-bias", type=float, default=argparse.SUPPRESS)
    p.add_argument("--no-unaries", action="store_true", default=argparse.SUPPRESS)
    p.add_argument("--neg-inf", type=float, default=argparse.SUPPRESS)

    return p


def _apply_test_only_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # Override test dataset root if requested
    if hasattr(args, "test_data_root"):
        cfg.setdefault("data", {})["root"] = getattr(args, "test_data_root")
    if hasattr(args, "test_split"):
        cfg.setdefault("data", {})["test_split"] = getattr(args, "test_split")

    return cfg


def _apply_park_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    park = cfg.setdefault("park", {})

    if hasattr(args, "M"):
        park["M"] = float(args.M)

    if hasattr(args, "max_iter"):
        park["max_iter"] = int(args.max_iter)
        park.setdefault("egg", {})["max_iter"] = int(args.max_iter)
        park.setdefault("gd", {})["max_iter"] = int(args.max_iter)

    if hasattr(args, "damping"):
        park["damping"] = float(args.damping)
        park.setdefault("egg", {})["damping"] = float(args.damping)
        park.setdefault("gd", {})["damping"] = float(args.damping)

    if hasattr(args, "tol"):
        park["tol"] = float(args.tol)
        park["bp_tol"] = float(args.tol)

    if hasattr(args, "inference"):
        park["inference"] = args.inference

    if hasattr(args, "egg_max_iter"):
        park.setdefault("egg", {})["max_iter"] = int(args.egg_max_iter)

    if hasattr(args, "egg_damping"):
        park.setdefault("egg", {})["damping"] = float(args.egg_damping)

    if hasattr(args, "gd_max_iter"):
        park.setdefault("gd", {})["max_iter"] = int(args.gd_max_iter)

    if hasattr(args, "gd_damping"):
        park.setdefault("gd", {})["damping"] = float(args.gd_damping)

    if hasattr(args, "gd_run_every"):
        park.setdefault("gd", {})["run_every"] = int(args.gd_run_every)

    if hasattr(args, "unary_bias"):
        park["unary_bias"] = float(args.unary_bias)

    if hasattr(args, "no_unaries"):
        park["add_unaries"] = not bool(args.no_unaries)

    if hasattr(args, "neg_inf"):
        park["neg_inf"] = float(args.neg_inf)

    return cfg

def run_test(cfg: Dict[str, Any], output_json: Optional[str] = None) -> Dict[str, Any]:
    # seeds (match train script behavior)
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build run paths (also determines where test_eval.json is written) :contentReference[oaicite:3]{index=3}
    paths = make_run_paths(cfg)

    # data loaders
    loaders = make_loaders(cfg, splits=["test"])

    # method adapter for eval (match training logic) 
    method_name = cfg.get("train", {}).get("method", "var_logits")
    eval_method = cfg.get("eval", {}).get("method") or method_name
    if eval_method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown eval method '{eval_method}'. Known: {list(METHOD_REGISTRY.keys())}")
    adapter_eval = METHOD_REGISTRY[eval_method]
    adapter_eval.loss_name = cfg.get("eval", {}).get("loss-name") or cfg["loss"]["name"]


    # run test eval (writes json) :contentReference[oaicite:5]{index=5}
    out_path = final_test_eval(
        cfg=cfg,
        adapter=adapter_eval,
        loader=loaders.test_loader,
        device=device,
        log_dir=paths.log_dir,
        output_json=output_json,
    )

    # print results to stdout
    with open(out_path, "r") as f:
        stats = json.load(f)

    # also include where it was written (useful in CLI)
    stats["_test_eval_path"] = out_path
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # stage 1 parse to get config path
    base = build_base_parser()
    base_args, _remaining = base.parse_known_args(argv)

    cfg = load_run_config(base_args.config)
    validate_config(cfg)  # ensure defaults exist / fill missing

    # stage 2 parse with full parser
    parser = build_full_parser(cfg)
    args = parser.parse_args(argv)

    # apply overrides only for flags that were explicitly provided
    cfg = merge_cli_overrides(cfg, args)
    cfg = _apply_test_only_overrides(cfg, args)
    cfg = _apply_park_overrides(cfg, args)

    if getattr(args, "print_config", False):
        # reuse your config dumper if you want; otherwise just json-dump
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return 0

    paths = make_run_paths(cfg)

    stats = run_test(cfg, output_json=getattr(args, "output_json", None))
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

